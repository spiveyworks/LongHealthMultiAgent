from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .timeline import TIMELINE_COLUMNS, TimelineStore

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

DATE_LIKE = {"date", "measurement_date", "timestamp"}
VALUE_LIKE = {"value", "result", "reading"}
VARIABLE_LIKE = {"variable", "analyte", "metric", "measure", "biomarker"}
UNIT_LIKE = {"unit", "units"}


def ingest_and_normalize(
    *,
    subject_id: str,
    source_paths: Iterable[str | Path],
    timeline_store: TimelineStore,
    artifact_dir: Path,
) -> Dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    unified_records: List[dict] = []
    summaries: List[str] = []
    source_paths = list(source_paths)

    for path in source_paths:
        path = Path(path)
        if not path.exists():
            continue
        frame = _load_dataframe(path)
        normalized, note = normalize_frame(frame, source=path.name, subject_id=subject_id)
        unified_records.extend(normalized.to_dict(orient="records"))
        summaries.append(note)

    if unified_records:
        timeline_store.append(unified_records)

    timeline_path = timeline_store.save(artifact_dir / "timeline.parquet")
    qc_report = run_qc_report(timeline_store.data)
    qc_path = artifact_dir / "qc_report.json"
    qc_path.write_text(json.dumps(qc_report, indent=2))

    figures = generate_qc_figures(timeline_store.data, artifact_dir)

    artifacts = {
        "timeline": {"path": str(timeline_path), "description": "Unified timeline", "mime_type": "application/x-parquet"},
        "qc_report": {"path": str(qc_path), "description": "Data quality report", "mime_type": "application/json"},
    }
    artifacts.update(figures)

    return {
        "summary": "; ".join(summaries) if summaries else "No new data ingested; using existing timeline",
        "preview": summaries[0] if summaries else "",
        "qc_report": str(qc_path),
        "artifacts": artifacts,
    }


def _load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_frame(frame: pd.DataFrame, *, source: str, subject_id: str) -> Tuple[pd.DataFrame, str]:
    df = frame.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if is_wide(df):
        df_long = melt_wide(df, source=source, subject_id=subject_id)
        note = f"Melted wide sheet {source} into {len(df_long)} rows"
        return df_long, note
    df_long = normalize_long(df, source=source, subject_id=subject_id)
    note = f"Normalized long sheet {source} into {len(df_long)} rows"
    return df_long, note


def is_wide(df: pd.DataFrame) -> bool:
    date_like_columns = sum(_looks_like_date(col) for col in df.columns)
    non_date_columns = len(df.columns) - date_like_columns
    threshold = max(2, len(df.columns) - max(3, non_date_columns))
    return date_like_columns >= threshold


def melt_wide(df: pd.DataFrame, *, source: str, subject_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    first_col = df.columns[0]
    variable_series = df[first_col]
    units = None
    if len(df.columns) > 1 and str(df.columns[1]).lower() in UNIT_LIKE:
        units = df.iloc[:, 1]
        value_columns = df.columns[2:]
    else:
        value_columns = df.columns[1:]
    melted = df.melt(id_vars=[first_col] + ([df.columns[1]] if units is not None else []), var_name="measurement_date", value_name="value")
    melted.rename(columns={first_col: "variable_name"}, inplace=True)
    if units is not None:
        melted.rename(columns={df.columns[1]: "unit"}, inplace=True)
    else:
        melted["unit"] = ""
    melted["subject_id"] = subject_id
    melted["source"] = source
    melted["metadata"] = "{}"
    melted["measurement_date"] = pd.to_datetime(melted["measurement_date"], errors="coerce", utc=True)
    melted = melted.dropna(subset=["measurement_date"])
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value"])
    return melted[TIMELINE_COLUMNS]


def normalize_long(df: pd.DataFrame, *, source: str, subject_id: str) -> pd.DataFrame:
    lower_map = {str(col).lower(): col for col in df.columns}

    date_col = _find_first(lower_map, DATE_LIKE)
    value_col = _find_first(lower_map, VALUE_LIKE)
    variable_col = _find_first(lower_map, VARIABLE_LIKE)
    unit_col = _find_first(lower_map, UNIT_LIKE)

    if date_col is None or value_col is None or variable_col is None:
        raise ValueError("Unable to map columns to required schema")

    normalized = pd.DataFrame({
        "subject_id": subject_id,
        "measurement_date": pd.to_datetime(df[lower_map[date_col]], errors="coerce", utc=True),
        "variable_name": df[lower_map[variable_col]].astype(str),
        "value": pd.to_numeric(df[lower_map[value_col]], errors="coerce"),
        "unit": df[lower_map[unit_col]].astype(str) if unit_col else "",
        "source": source,
        "metadata": df.apply(lambda row: json.dumps({"raw": row.dropna().to_dict()}), axis=1),
    })
    normalized = normalized.dropna(subset=["measurement_date"])
    return normalized[TIMELINE_COLUMNS]


def _find_first(lower_map: Dict[str, str], candidates: Iterable[str]) -> str | None:
    for key, original in lower_map.items():
        base = key.strip().lower()
        if base in candidates:
            return base
    for candidate in candidates:
        for key in lower_map.keys():
            if candidate in key:
                return key
    return None


def _looks_like_date(value: str) -> bool:
    try:
        pd.to_datetime(value)
        return True
    except Exception:
        return False


def run_qc_report(timeline_df: pd.DataFrame) -> Dict[str, object]:
    if timeline_df.empty:
        return {"status": "empty", "issues": ["No data ingested"], "metrics": {}}

    metrics = {}
    issues: List[str] = []

    grouped = timeline_df.groupby("variable_name")
    missingness = (grouped["value"].apply(lambda series: float(series.isna().mean())).to_dict())
    metrics["missingness"] = missingness

    duplicates = int(timeline_df.duplicated(subset=["subject_id", "measurement_date", "variable_name"]).sum())
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate entries")

    def robust_z(series: pd.Series) -> pd.Series:
        median = series.median()
        mad = np.median(np.abs(series - median)) or 1.0
        return 0.6745 * (series - median) / mad

    outlier_summary: Dict[str, int] = {}
    for name, group in grouped:
        numeric = group["value"].dropna()
        if numeric.empty:
            continue
        z_scores = robust_z(numeric)
        outlier_summary[name] = int((np.abs(z_scores) > 3.5).sum())
    metrics["outliers"] = outlier_summary

    unit_variability = timeline_df.groupby("variable_name")["unit"].nunique().to_dict()
    inconsistent_units = [name for name, count in unit_variability.items() if count > 1]
    if inconsistent_units:
        issues.append(f"Variables with inconsistent units: {', '.join(inconsistent_units)}")

    return {"status": "ok", "issues": issues, "metrics": metrics}


def generate_qc_figures(timeline_df: pd.DataFrame, artifact_dir: Path) -> Dict[str, Dict[str, object]]:
    figures: Dict[str, Dict[str, object]] = {}
    if plt is None or timeline_df.empty:
        return figures

    artifact_dir.mkdir(parents=True, exist_ok=True)

    pivot = timeline_df.pivot_table(
        index="measurement_date",
        columns="variable_name",
        values="value",
    )

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    missing = pivot.isna().mean()
    missing.sort_values(ascending=False).plot(kind="bar", ax=ax1)
    ax1.set_title("Missingness by variable")
    ax1.set_ylabel("Fraction missing")
    fig1.tight_layout()
    missing_path = artifact_dir / "qc_missingness.png"
    fig1.savefig(missing_path)
    plt.close(fig1)
    figures["qc_missingness"] = {
        "path": str(missing_path),
        "description": "Missingness by variable",
        "mime_type": "image/png",
    }

    if pivot.shape[1] >= 1:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        pivot.interpolate().plot(ax=ax2)
        ax2.set_title("Interpolated timelines (sample)")
        ax2.set_ylabel("Value")
        fig2.tight_layout()
        timeline_path = artifact_dir / "qc_timelines.png"
        fig2.savefig(timeline_path)
        plt.close(fig2)
        figures["qc_timelines"] = {
            "path": str(timeline_path),
            "description": "Interpolated timelines",
            "mime_type": "image/png",
        }

    corr = pivot.corr().fillna(0)
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    cax = ax3.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax3.set_xticks(range(len(corr.columns)))
    ax3.set_xticklabels(corr.columns, rotation=90)
    ax3.set_yticks(range(len(corr.columns)))
    ax3.set_yticklabels(corr.columns)
    ax3.set_title("Correlation heatmap (QC)")
    fig3.colorbar(cax, ax=ax3, fraction=0.046, pad=0.04)
    fig3.tight_layout()
    corr_path = artifact_dir / "qc_corr_heatmap.png"
    fig3.savefig(corr_path)
    plt.close(fig3)
    figures["qc_corr_heatmap"] = {
        "path": str(corr_path),
        "description": "QC correlation heatmap",
        "mime_type": "image/png",
    }
    return figures
