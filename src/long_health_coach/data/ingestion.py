from __future__ import annotations

import gzip
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET

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
UNIT_LIKE = {"unit", "units", "unit of measure", "uom"}

APPLE_HEALTH_EXPORT = "apple_health_export/export.xml"

VARIABLE_NAME_MAP = {
    "a1c": ("Hemoglobin A1C", "percent"),
    "hba1c": ("Hemoglobin A1C", "percent"),
    "hemoglobin a1c": ("Hemoglobin A1C", "percent"),
    "hdl chol": ("HDL Cholesterol", "mg/dL"),
    "hdl c": ("HDL Cholesterol", "mg/dL"),
    "hdl cholesterol": ("HDL Cholesterol", "mg/dL"),
    "ldl c": ("LDL Cholesterol", "mg/dL"),
    "ldl chol calc": ("LDL Cholesterol", "mg/dL"),
    "ldl cholesterol": ("LDL Cholesterol", "mg/dL"),
    "chol": ("Total Cholesterol", "mg/dL"),
    "cholesterol total": ("Total Cholesterol", "mg/dL"),
    "total cholesterol": ("Total Cholesterol", "mg/dL"),
    "triglycerides": ("Triglycerides", "mg/dL"),
    "crp": ("High-Sensitivity CRP", "mg/L"),
    "hs crp": ("High-Sensitivity CRP", "mg/L"),
    "high sensitivity crp": ("High-Sensitivity CRP", "mg/L"),
    "vitamin d": ("25-(OH) Vitamin D", "ng/mL"),
    "triglycerides hdl ratio": ("Triglycerides:HDL Ratio", ""),
    "total cholesterol hdl ratio": ("Total Cholesterol:HDL Ratio", ""),
    "pwv": ("Pulse Wave Velocity", "m/s"),
    "respiratory rate": ("Respiratory Rate", "breaths/min"),
    "vo2 max": ("VO2 Max", "mL/kg/min"),
    "body mass": ("Body Mass", "lb"),
    "body mass index": ("Body Mass Index", "kg/m^2"),
    "body fat percentage": ("Body Fat Percentage", "fraction"),
}

CRONOMETER_COLUMN_MAP = {
    "Energy (kcal)": ("Daily Average Dietary Energy Consumed", "kcal", None),
    "Carbs (g)": ("Daily Average Dietary Carbohydrates", "g", None),
    "Protein (g)": ("Daily Average Dietary Protein", "g", None),
    "Fat (g)": ("Daily Average Dietary Fat Total", "g", None),
    "Saturated (g)": ("Daily Average Dietary Fat Saturated", "g", None),
    "Monounsaturated (g)": (
        "Daily Average Dietary Fat Monounsaturated",
        "g",
        None,
    ),
    "Polyunsaturated (g)": (
        "Daily Average Dietary Fat Polyunsaturated",
        "g",
        None,
    ),
    "Fiber (g)": ("Daily Average Dietary Fiber", "g", None),
    "Sugars (g)": ("Daily Average Dietary Sugar", "g", None),
    "Water (g)": ("Daily Average Dietary Water", "mL", None),
    "Caffeine (mg)": ("Daily Average Dietary Caffeine", "mg", None),
    "Cholesterol (mg)": ("Daily Average Dietary Cholesterol", "mg", None),
    "Vitamin C (mg)": ("Daily Average Dietary Vitamin C", "mg", None),
    "Vitamin D (IU)": (
        "Daily Average Dietary Vitamin D",
        "mcg",
        lambda s: s / 40.0,
    ),
    "Vitamin E (mg)": ("Daily Average Dietary Vitamin E", "mg", None),
    "Vitamin K (ug)": ("Daily Average Dietary Vitamin K", "mcg", None),
    "Folate (ug)": ("Daily Average Dietary Folate", "mcg", None),
    "B1 (Thiamine) (mg)": ("Daily Average Dietary Thiamin", "mg", None),
    "B2 (Riboflavin) (mg)": ("Daily Average Dietary Riboflavin", "mg", None),
    "B3 (Niacin) (mg)": ("Daily Average Dietary Niacin", "mg", None),
    "B5 (Pantothenic Acid) (mg)": (
        "Daily Average Dietary Pantothenic Acid",
        "mg",
        None,
    ),
    "B6 (Pyridoxine) (mg)": ("Daily Average Dietary Vitamin B6", "mg", None),
    "B12 (Cobalamin) (ug)": ("Daily Average Dietary Vitamin B12", "mcg", None),
    "Calcium (mg)": ("Daily Average Dietary Calcium", "mg", None),
    "Copper (mg)": ("Daily Average Dietary Copper", "mg", None),
    "Iron (mg)": ("Daily Average Dietary Iron", "mg", None),
    "Magnesium (mg)": ("Daily Average Dietary Magnesium", "mg", None),
    "Manganese (mg)": ("Daily Average Dietary Manganese", "mg", None),
    "Phosphorus (mg)": ("Daily Average Dietary Phosphorus", "mg", None),
    "Potassium (mg)": ("Daily Average Dietary Potassium", "mg", None),
    "Selenium (ug)": ("Daily Average Dietary Selenium", "mcg", None),
    "Sodium (mg)": ("Daily Average Dietary Sodium", "mg", None),
    "Zinc (mg)": ("Daily Average Dietary Zinc", "mg", None),
}

CRONOMETER_SUPPLEMENTS = {
    "Daily Average BroccoMax": {
        "unit": "capsules/day",
        "food_names": ["BroccoMax"],
    },
    "Daily Average Cvs Health, 100% Pure Omega-3 Krill Oil, 500 mg": {
        "unit": "servings/day",
        "food_names": ["Cvs Health, 100% Pure Omega-3 Krill Oil, 500 mg"],
    },
    "Daily Average Nature's Bounty, Krill Oil": {
        "unit": "servings/day",
        "food_names": ["Nature's Bounty, Krill Oil"],
    },
    "Daily Average Nature Made, Fish Oil 1200 mg 720 mg Omega 3": {
        "unit": "servings/day",
        "food_names": [
            "Nature Made, Fish Oil 1200 mg 720 mg Omega 3",
            "Nature Made, Fish Oil, 720 mg Omega-3",
            "Nature Made, Fish Oil 1200 mg",
        ],
    },
    "Daily Average Force Factor, Total Beets Original Drink Powder, Pomegranate Berry": {
        "unit": "servings/day",
        "food_names": [
            "Force Factor, Total Beets Original Drink Powder, Pomegranate Berry",
        ],
    },
}

APPLE_HEALTH_TYPE_MAP = {
    "HKQuantityTypeIdentifierActiveEnergyBurned": (
        "Active Energy Burned",
        "kcal",
        "sum",
    ),
    "HKQuantityTypeIdentifierBasalEnergyBurned": (
        "Basal Energy Burned",
        "kcal",
        "sum",
    ),
    "HKQuantityTypeIdentifierAppleExerciseTime": (
        "Apple Exercise Time",
        "minutes",
        "sum",
    ),
    "HKQuantityTypeIdentifierRestingHeartRate": (
        "Resting Heart Rate",
        "bpm",
        "median",
    ),
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": (
        "Heart Rate Variability SDNN (intraday median, across day mean)",
        "ms",
        "median",
    ),
    "HKQuantityTypeIdentifierRespiratoryRate": (
        "Respiratory Rate",
        "breaths/min",
        "median",
    ),
    "HKQuantityTypeIdentifierBodyMass": ("Body Mass", "lb", "last"),
    "HKQuantityTypeIdentifierBodyFatPercentage": (
        "Body Fat Percentage",
        "fraction",
        "last",
    ),
    "HKQuantityTypeIdentifierBodyMassIndex": (
        "Body Mass Index",
        "kg/m^2",
        "last",
    ),
    "HKQuantityTypeIdentifierVO2Max": ("VO2 Max", "mL/kg/min", "last"),
    "HKQuantityTypeIdentifierBodyTemperature": (
        "Body Temperature",
        "degF",
        "median",
    ),
    "HKQuantityTypeIdentifierBloodPressureSystolic": (
        "Blood Pressure Systolic - Any Time",
        "mmHg",
        "median",
    ),
    "HKQuantityTypeIdentifierBloodPressureDiastolic": (
        "Blood Pressure Diastolic - Any Time",
        "mmHg",
        "median",
    ),
    "HKQuantityTypeIdentifierDietaryWater": (
        "Daily Average Dietary Water",
        "mL",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryEnergyConsumed": (
        "Daily Average Dietary Energy Consumed",
        "kcal",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryProtein": (
        "Daily Average Dietary Protein",
        "g",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryCarbohydrates": (
        "Daily Average Dietary Carbohydrates",
        "g",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryFatTotal": (
        "Daily Average Dietary Fat Total",
        "g",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryFatSaturated": (
        "Daily Average Dietary Fat Saturated",
        "g",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryFatMonounsaturated": (
        "Daily Average Dietary Fat Monounsaturated",
        "g",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryFatPolyunsaturated": (
        "Daily Average Dietary Fat Polyunsaturated",
        "g",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryFiber": (
        "Daily Average Dietary Fiber",
        "g",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietarySugar": (
        "Daily Average Dietary Sugar",
        "g",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryCaffeine": (
        "Daily Average Dietary Caffeine",
        "mg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietarySodium": (
        "Daily Average Dietary Sodium",
        "mg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryPotassium": (
        "Daily Average Dietary Potassium",
        "mg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryVitaminC": (
        "Daily Average Dietary Vitamin C",
        "mg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryVitaminD": (
        "Daily Average Dietary Vitamin D",
        "mcg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryVitaminA": (
        "Daily Average Dietary Vitamin A",
        "mcg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryVitaminE": (
        "Daily Average Dietary Vitamin E",
        "mg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryVitaminK": (
        "Daily Average Dietary Vitamin K",
        "mcg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryCalcium": (
        "Daily Average Dietary Calcium",
        "mg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryIron": (
        "Daily Average Dietary Iron",
        "mg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryMagnesium": (
        "Daily Average Dietary Magnesium",
        "mg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryZinc": (
        "Daily Average Dietary Zinc",
        "mg",
        "sum",
    ),
    "HKQuantityTypeIdentifierDietaryCholesterol": (
        "Daily Average Dietary Cholesterol",
        "mg",
        "sum",
    ),
}

AMOUNT_RE = re.compile(r"[-+]?\d*\.?\d+")


def ingest_and_normalize(
    *,
    subject_id: str,
    source_paths: Iterable[str | Path],
    timeline_store: TimelineStore,
    artifact_dir: Path,
) -> Dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[str] = []
    source_paths = list(source_paths)

    for path in source_paths:
        path = Path(path)
        if not path.exists():
            summaries.append(f"Missing source {path}")
            continue
        for frame, note in _load_and_normalize_source(path, subject_id=subject_id):
            if frame.empty:
                summaries.append(note)
                continue
            timeline_store.append(frame.to_dict(orient="records"))
            summaries.append(note)

    timeline_path = timeline_store.save(artifact_dir / "timeline.parquet")
    qc_report = run_qc_report(timeline_store.data)
    qc_path = artifact_dir / "qc_report.json"
    qc_path.write_text(json.dumps(qc_report, indent=2))

    figures = generate_qc_figures(timeline_store.data, artifact_dir)

    artifacts = {
        "timeline": {
            "path": str(timeline_path),
            "description": "Unified timeline",
            "mime_type": "application/x-parquet",
        },
        "qc_report": {
            "path": str(qc_path),
            "description": "Data quality report",
            "mime_type": "application/json",
        },
    }
    artifacts.update(figures)

    return {
        "summary": "; ".join(summaries) if summaries else "No new data ingested; using existing timeline",
        "preview": summaries[0] if summaries else "",
        "qc_report": str(qc_path),
        "artifacts": artifacts,
    }


def _load_and_normalize_source(path: Path, *, subject_id: str) -> List[Tuple[pd.DataFrame, str]]:
    if _is_apple_health_export(path):
        frame = parse_apple_health_export(path, subject_id=subject_id)
        return [(frame, f"Parsed Apple Health export {path.name} into {len(frame)} rows")]

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return _load_excel_frames(path, subject_id=subject_id)

    if path.name.lower().endswith(".vcf.gz") or path.suffix.lower() == ".vcf":
        try:
            frame = parse_vcf_summary(path, subject_id=subject_id)
            note = f"Parsed VCF summary {path.name} into {len(frame)} rows"
            return [(frame, note)]
        except Exception as exc:  # pragma: no cover - defensive for malformed VCFs
            return [(pd.DataFrame(columns=TIMELINE_COLUMNS), f"Skipped VCF file {path.name} ({exc})")]

    frame = _load_dataframe(path)
    normalized, note = normalize_frame(frame, source=path.name, subject_id=subject_id)
    return [(normalized, note)]


def _load_excel_frames(path: Path, *, subject_id: str) -> List[Tuple[pd.DataFrame, str]]:
    xl = pd.ExcelFile(path)
    frames: List[Tuple[pd.DataFrame, str]] = []

    if "cronometer" in path.name.lower():
        df = xl.parse(xl.sheet_names[0])
        if _is_cronometer_servings_frame(df):
            normalized, note = normalize_cronometer_servings(
                df,
                source=path.name,
                subject_id=subject_id,
            )
            return [(normalized, note)]

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        if df.empty:
            continue
        if _is_cronometer_servings_frame(df):
            normalized, note = normalize_cronometer_servings(
                df,
                source=f"{path.name}:{sheet_name}",
                subject_id=subject_id,
            )
            frames.append((normalized, note))
            continue
        try:
            normalized, note = normalize_frame(
                df,
                source=f"{path.name}:{sheet_name}",
                subject_id=subject_id,
            )
        except ValueError as exc:
            frames.append(
                (
                    pd.DataFrame(columns=TIMELINE_COLUMNS),
                    f"Skipped sheet {path.name}:{sheet_name} ({exc})",
                )
            )
            continue
        frames.append((normalized, note))

    if not frames:
        frames.append((pd.DataFrame(columns=TIMELINE_COLUMNS), f"No usable sheets found in {path.name}"))
    return frames


def _load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_frame(frame: pd.DataFrame, *, source: str, subject_id: str) -> Tuple[pd.DataFrame, str]:
    df = frame.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if df.empty:
        return pd.DataFrame(columns=TIMELINE_COLUMNS), f"Empty sheet {source}"

    if is_wide(df):
        df_long = melt_wide(df, source=source, subject_id=subject_id)
        note = f"Melted wide sheet {source} into {len(df_long)} rows"
        return df_long, note

    if _can_normalize_long(df):
        df_long = normalize_long(df, source=source, subject_id=subject_id)
        note = f"Normalized long sheet {source} into {len(df_long)} rows"
        return df_long, note

    date_value_col = _find_date_value_column(df)
    if date_value_col:
        df_long = melt_columnar(df, date_col=date_value_col, source=source, subject_id=subject_id)
        note = f"Melted columnar sheet {source} into {len(df_long)} rows"
        return df_long, note

    raise ValueError("Unable to map columns to required schema")


def _can_normalize_long(df: pd.DataFrame) -> bool:
    lower_map = {str(col).lower(): col for col in df.columns}
    return (
        _find_first(lower_map, DATE_LIKE) is not None
        and _find_first(lower_map, VALUE_LIKE) is not None
        and _find_first(lower_map, VARIABLE_LIKE) is not None
    )


def is_wide(df: pd.DataFrame) -> bool:
    date_like_columns = sum(_looks_like_date(col) for col in df.columns)
    non_date_columns = len(df.columns) - date_like_columns
    threshold = max(2, len(df.columns) - max(3, non_date_columns))
    return date_like_columns >= threshold


def melt_wide(df: pd.DataFrame, *, source: str, subject_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    date_columns = [col for col in df.columns if _looks_like_date(col)]
    non_date_columns = [col for col in df.columns if col not in date_columns]
    if not date_columns or not non_date_columns:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)

    variable_col = non_date_columns[0]
    unit_col = _find_unit_column(non_date_columns[1:])
    metadata_cols = [col for col in non_date_columns if col not in {variable_col, unit_col}]
    id_vars = [variable_col] + ([unit_col] if unit_col else []) + metadata_cols

    melted = df.melt(
        id_vars=id_vars,
        value_vars=date_columns,
        var_name="measurement_date",
        value_name="value",
    )
    melted.rename(columns={variable_col: "variable_name"}, inplace=True)
    if unit_col:
        melted.rename(columns={unit_col: "unit"}, inplace=True)
    else:
        melted["unit"] = ""

    melted["raw_variable_name"] = melted["variable_name"].astype(str)
    melted["raw_unit"] = melted["unit"].astype(str)
    melted["measurement_date"] = pd.to_datetime(
        melted["measurement_date"],
        errors="coerce",
        utc=True,
    ).dt.normalize()
    melted = melted.dropna(subset=["measurement_date"])
    melted["value"] = melted["value"].apply(_parse_numeric)
    melted = melted.dropna(subset=["value"])

    mapped_names = []
    mapped_units = []
    for name, unit in zip(melted["raw_variable_name"], melted["raw_unit"]):
        mapped_name, mapped_unit = _normalize_variable_and_unit(name, unit)
        mapped_names.append(mapped_name)
        mapped_units.append(mapped_unit)
    melted["variable_name"] = mapped_names
    melted["unit"] = mapped_units

    melted["variable_name"] = melted["variable_name"].astype(str).str.strip()
    melted = melted[melted["variable_name"].notna()]
    melted = melted[melted["variable_name"].ne("")]
    melted = melted[melted["variable_name"].ne("nan")]

    def build_metadata(row) -> str:
        meta: Dict[str, object] = {}
        for col in metadata_cols:
            value = row.get(col)
            if pd.notna(value):
                meta[_metadata_key(col)] = value
        if row["raw_variable_name"] != row["variable_name"]:
            meta["raw_variable"] = row["raw_variable_name"]
        if row["raw_unit"] and row["raw_unit"] != row["unit"]:
            meta["raw_unit"] = row["raw_unit"]
        return _json_metadata(meta)

    melted["metadata"] = melted.apply(build_metadata, axis=1) if metadata_cols else "{}"
    melted["subject_id"] = subject_id
    melted["source"] = source
    return melted[TIMELINE_COLUMNS]


def melt_columnar(df: pd.DataFrame, *, date_col: str, source: str, subject_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    value_columns = [col for col in df.columns if col != date_col]
    if not value_columns:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    melted = df.melt(
        id_vars=[date_col],
        value_vars=value_columns,
        var_name="variable_name_raw",
        value_name="value",
    )
    melted["measurement_date"] = pd.to_datetime(
        melted[date_col],
        errors="coerce",
        utc=True,
    ).dt.normalize()
    melted["value"] = melted["value"].apply(_parse_numeric)
    melted = melted.dropna(subset=["measurement_date", "value"])

    var_map = {col: _normalize_variable_and_unit(col, "") for col in value_columns}
    melted["variable_name"] = melted["variable_name_raw"].map(
        lambda x: var_map.get(x, (str(x), ""))[0]
    )
    melted["unit"] = melted["variable_name_raw"].map(
        lambda x: var_map.get(x, ("", ""))[1]
    )

    metadata = []
    for raw_name, mapped_name in zip(melted["variable_name_raw"], melted["variable_name"]):
        if raw_name == mapped_name:
            metadata.append("{}")
        else:
            metadata.append(_json_metadata({"raw_variable": raw_name}))
    melted["metadata"] = metadata
    melted["subject_id"] = subject_id
    melted["source"] = source
    return melted[TIMELINE_COLUMNS]


def normalize_long(df: pd.DataFrame, *, source: str, subject_id: str) -> pd.DataFrame:
    lower_map = {str(col).lower(): col for col in df.columns}

    date_col = _find_first(lower_map, DATE_LIKE)
    value_col = _find_first(lower_map, VALUE_LIKE)
    variable_col = _find_first(lower_map, VARIABLE_LIKE)
    unit_col = _find_first(lower_map, UNIT_LIKE)

    if date_col is None or value_col is None or variable_col is None:
        raise ValueError("Unable to map columns to required schema")

    raw_variables = df[lower_map[variable_col]].astype(str)
    raw_units = df[lower_map[unit_col]].astype(str) if unit_col else ""

    mapped_names = []
    mapped_units = []
    units_iter = raw_units if isinstance(raw_units, pd.Series) else [raw_units] * len(raw_variables)
    for name, unit in zip(raw_variables, units_iter):
        mapped_name, mapped_unit = _normalize_variable_and_unit(name, unit)
        mapped_names.append(mapped_name)
        mapped_units.append(mapped_unit)

    normalized = pd.DataFrame(
        {
            "subject_id": subject_id,
            "measurement_date": pd.to_datetime(
                df[lower_map[date_col]],
                errors="coerce",
                utc=True,
            ).dt.normalize(),
            "variable_name": mapped_names,
            "value": df[lower_map[value_col]].apply(_parse_numeric),
            "unit": mapped_units,
            "source": source,
            "metadata": df.apply(lambda row: json.dumps({"raw": row.dropna().to_dict()}), axis=1),
        }
    )
    normalized = normalized.dropna(subset=["measurement_date"])
    normalized = normalized.dropna(subset=["value"])
    return normalized[TIMELINE_COLUMNS]


def normalize_cronometer_servings(frame: pd.DataFrame, *, source: str, subject_id: str) -> Tuple[pd.DataFrame, str]:
    df = frame.copy()
    df.columns = [_normalize_column_header(col).strip() for col in df.columns]
    if "Day" not in df.columns or "Food Name" not in df.columns:
        raise ValueError("Missing required Cronometer columns")

    df["measurement_date"] = pd.to_datetime(df["Day"], errors="coerce", utc=True).dt.normalize()
    df = df.dropna(subset=["measurement_date"])

    frames: List[pd.DataFrame] = []
    for column, (variable_name, unit, transform) in CRONOMETER_COLUMN_MAP.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if transform:
            values = transform(values)
        daily = values.groupby(df["measurement_date"]).sum(min_count=1)
        metadata = _json_metadata({"source_column": column, "aggregation": "daily_sum"})
        frames.append(
            _series_to_records(
                daily,
                variable_name,
                unit,
                source,
                subject_id,
                metadata,
            )
        )

    for variable_name, spec in CRONOMETER_SUPPLEMENTS.items():
        subset = df[df["Food Name"].astype(str).str.strip().isin(spec["food_names"])]
        if subset.empty:
            continue
        amounts = subset["Amount"].apply(_parse_amount)
        daily = amounts.groupby(subset["measurement_date"]).sum(min_count=1)
        metadata = _json_metadata({
            "source_food": ",".join(spec["food_names"]),
            "aggregation": "daily_sum",
        })
        frames.append(
            _series_to_records(
                daily,
                variable_name,
                spec["unit"],
                source,
                subject_id,
                metadata,
            )
        )

    egg_subset = df[
        df.get("Category", "").astype(str).str.contains(
            "Dairy and Egg Products",
            case=False,
            na=False,
        )
        & df["Food Name"].astype(str).str.contains(r"\begg", case=False, na=False)
        & ~df["Food Name"].astype(str).str.contains("beater", case=False, na=False)
        & ~df["Food Name"].astype(str).str.contains("cadbury", case=False, na=False)
    ]
    if not egg_subset.empty:
        amounts = egg_subset["Amount"].apply(_parse_amount)
        daily = amounts.groupby(egg_subset["measurement_date"]).sum(min_count=1)
        metadata = _json_metadata({"source_food": "egg_items", "aggregation": "daily_sum"})
        frames.append(
            _series_to_records(
                daily,
                "Daily Average Whole Eggs",
                "count/day",
                source,
                subject_id,
                metadata,
            )
        )

    if not frames:
        return pd.DataFrame(columns=TIMELINE_COLUMNS), f"No usable Cronometer rows found in {source}"
    combined = pd.concat(frames, ignore_index=True)
    return combined[TIMELINE_COLUMNS], f"Aggregated Cronometer servings {source} into {len(combined)} rows"


def parse_apple_health_export(path: Path, *, subject_id: str) -> pd.DataFrame:
    sums: Dict[Tuple[pd.Timestamp, str, str], float] = defaultdict(float)
    medians: Dict[Tuple[pd.Timestamp, str, str], List[float]] = defaultdict(list)
    lasts: Dict[Tuple[pd.Timestamp, str, str], Tuple[datetime, float]] = {}
    metadata_map: Dict[Tuple[pd.Timestamp, str, str], str] = {}

    record_types = set(APPLE_HEALTH_TYPE_MAP.keys())

    def handle_record(attrs: Dict[str, str]) -> None:
        record_type = attrs.get("type")
        if record_type not in record_types:
            return
        raw_value = _safe_float(attrs.get("value"))
        if raw_value is None:
            return
        unit = attrs.get("unit", "")
        start_date = attrs.get("startDate") or attrs.get("endDate")
        if not start_date:
            return
        parsed_date = _parse_apple_health_datetime(start_date)
        if parsed_date is None:
            return
        measurement_date = _normalize_dt_to_utc_date(parsed_date)
        variable_name, target_unit, agg = APPLE_HEALTH_TYPE_MAP[record_type]
        converted = _convert_apple_health_value(raw_value, unit, target_unit, record_type)
        if converted is None:
            return
        key = (measurement_date, variable_name, target_unit)
        if unit:
            metadata_map[key] = _json_metadata({"apple_type": record_type, "raw_unit": unit})
        else:
            metadata_map[key] = _json_metadata({"apple_type": record_type})
        if agg == "sum":
            sums[key] += converted
        elif agg == "median":
            medians[key].append(converted)
        elif agg == "last":
            prev = lasts.get(key)
            if prev is None or parsed_date > prev[0]:
                lasts[key] = (parsed_date, converted)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            with zf.open(APPLE_HEALTH_EXPORT) as handle:
                for attrs in _iter_apple_health_records(handle):
                    handle_record(attrs)
    else:
        with path.open("rb") as handle:
            for attrs in _iter_apple_health_records(handle):
                handle_record(attrs)

    frames = []
    for key, value in sums.items():
        frames.append(_apple_health_record(subject_id, path.name, key, value, metadata_map.get(key, "{}")))
    for key, values in medians.items():
        if values:
            frames.append(_apple_health_record(subject_id, path.name, key, float(np.median(values)), metadata_map.get(key, "{}")))
    for key, value in lasts.items():
        frames.append(_apple_health_record(subject_id, path.name, key, float(value[1]), metadata_map.get(key, "{}")))

    if not frames:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined[TIMELINE_COLUMNS]


def parse_vcf_summary(path: Path, *, subject_id: str) -> pd.DataFrame:
    total_records = 0
    alt_alleles = 0
    snv_count = 0
    indel_count = 0
    other_count = 0
    transitions = 0
    transversions = 0
    het_count = 0
    hom_alt_count = 0
    hom_ref_count = 0
    missing_genotype_count = 0
    sample_name = ""
    file_format = ""

    with _open_vcf(path) as handle:
        for line in handle:
            if not line:
                continue
            if line.startswith("##"):
                if line.startswith("##fileformat="):
                    file_format = line.strip().split("=", 1)[1]
                continue
            if line.startswith("#CHROM"):
                parts = line.strip().split("\t")
                if len(parts) > 9:
                    sample_name = parts[9]
                continue

            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            total_records += 1
            ref = parts[3]
            alt_field = parts[4]
            if alt_field in {"", "."}:
                continue
            alts = alt_field.split(",")
            alt_alleles += len(alts)
            for alt in alts:
                if alt.startswith("<") or alt == "*":
                    other_count += 1
                    continue
                if len(ref) == 1 and len(alt) == 1:
                    snv_count += 1
                    if _is_transition(ref, alt):
                        transitions += 1
                    else:
                        transversions += 1
                else:
                    indel_count += 1

            if len(parts) >= 10:
                genotype = parts[9].split(":", 1)[0] if parts[9] else ""
                gt_class = _classify_genotype(genotype)
                if gt_class == "missing":
                    missing_genotype_count += 1
                elif gt_class == "hom_ref":
                    hom_ref_count += 1
                elif gt_class == "hom_alt":
                    hom_alt_count += 1
                elif gt_class == "het":
                    het_count += 1

    measurement_date = _measurement_date_from_path(path)
    metadata = _json_metadata(
        {
            "sample": sample_name,
            "file_format": file_format,
        }
    )
    records: List[dict] = []

    def add_metric(name: str, value: float, unit: str) -> None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        records.append(
            {
                "subject_id": subject_id,
                "measurement_date": measurement_date,
                "variable_name": name,
                "value": float(value),
                "unit": unit,
                "source": path.name,
                "metadata": metadata,
            }
        )

    add_metric("Genomics Total Variants", total_records, "count")
    add_metric("Genomics Alt Allele Count", alt_alleles, "count")
    add_metric("Genomics SNV Count", snv_count, "count")
    add_metric("Genomics Indel Count", indel_count, "count")
    add_metric("Genomics Other Variant Count", other_count, "count")
    add_metric("Genomics Transition Count", transitions, "count")
    add_metric("Genomics Transversion Count", transversions, "count")
    ttv = transitions / transversions if transversions else np.nan
    add_metric("Genomics TiTv Ratio", ttv, "ratio")

    if sample_name:
        add_metric("Genomics Heterozygous Count", het_count, "count")
        add_metric("Genomics Homozygous Alt Count", hom_alt_count, "count")
        add_metric("Genomics Homozygous Ref Count", hom_ref_count, "count")
        add_metric("Genomics Missing Genotype Count", missing_genotype_count, "count")
        if total_records:
            call_rate = 1 - (missing_genotype_count / total_records)
            add_metric("Genomics Call Rate", call_rate, "ratio")

    if not records:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    return pd.DataFrame(records)[TIMELINE_COLUMNS]


def _apple_health_record(
    subject_id: str,
    source: str,
    key: Tuple[pd.Timestamp, str, str],
    value: float,
    metadata: str,
) -> pd.DataFrame:
    measurement_date, variable_name, unit = key
    return pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "measurement_date": measurement_date,
                "variable_name": variable_name,
                "value": value,
                "unit": unit,
                "source": source,
                "metadata": metadata,
            }
        ]
    )


def _iter_apple_health_records(handle) -> Iterable[Dict[str, str]]:
    context = ET.iterparse(handle, events=("end",))
    for _, elem in context:
        if elem.tag == "Record":
            attrs = dict(elem.attrib)
            elem.clear()
            yield attrs
        else:
            elem.clear()


def _parse_apple_health_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        try:
            return pd.to_datetime(value, utc=True).to_pydatetime()
        except Exception:
            return None


def _normalize_dt_to_utc_date(dt: datetime) -> pd.Timestamp:
    dt_utc = dt.astimezone(timezone.utc)
    return pd.Timestamp(datetime(dt_utc.year, dt_utc.month, dt_utc.day, tzinfo=timezone.utc))


def _convert_apple_health_value(
    value: float,
    unit: str,
    target_unit: str,
    record_type: str,
) -> float | None:
    unit_lower = unit.strip().lower()
    if unit_lower == "":
        return value

    if record_type == "HKQuantityTypeIdentifierBodyMass":
        if unit_lower == "kg":
            return value * 2.20462
        if unit_lower in {"lb", "lbs"}:
            return value
    if record_type == "HKQuantityTypeIdentifierBodyFatPercentage":
        if unit_lower in {"%", "percent"}:
            return value / 100.0 if value > 1 else value
        if unit_lower in {"count", "1"}:
            return value
    if record_type == "HKQuantityTypeIdentifierBodyTemperature":
        if unit_lower == "degc":
            return value * 9 / 5 + 32
        if unit_lower == "degf":
            return value
    if record_type in {
        "HKQuantityTypeIdentifierBloodPressureSystolic",
        "HKQuantityTypeIdentifierBloodPressureDiastolic",
    }:
        if unit_lower == "kpa":
            return value * 7.50062
        if unit_lower == "mmhg":
            return value
    if record_type == "HKQuantityTypeIdentifierAppleExerciseTime":
        if unit_lower in {"min", "minutes"}:
            return value
        if unit_lower == "s":
            return value / 60.0
    if record_type in {
        "HKQuantityTypeIdentifierRestingHeartRate",
        "HKQuantityTypeIdentifierRespiratoryRate",
    }:
        if unit_lower in {"count/min", "bpm"}:
            return value
    if record_type == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
        if unit_lower == "ms":
            return value
    if record_type in {
        "HKQuantityTypeIdentifierActiveEnergyBurned",
        "HKQuantityTypeIdentifierBasalEnergyBurned",
        "HKQuantityTypeIdentifierDietaryEnergyConsumed",
    }:
        if unit_lower in {"kcal", "cal"}:
            return value
        if unit_lower == "kj":
            return value / 4.184
    if record_type == "HKQuantityTypeIdentifierDietaryWater":
        if unit_lower in {"ml", "milliliter", "milliliters"}:
            return value
        if unit_lower in {"l", "liter", "liters"}:
            return value * 1000
        if unit_lower in {"fl_oz_us", "fl oz"}:
            return value * 29.5735
    if record_type == "HKQuantityTypeIdentifierDietaryVitaminD":
        if unit_lower == "iu":
            return value / 40.0
        if unit_lower == "mcg":
            return value
    if record_type == "HKQuantityTypeIdentifierDietaryVitaminA":
        if unit_lower == "iu":
            return value / 3.33
        if unit_lower == "mcg":
            return value

    if unit_lower == target_unit.lower() or target_unit == "":
        return value
    return None


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


def _find_date_value_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        series = df[col]
        if series.dropna().empty:
            continue
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed.notna().mean() >= 0.6:
            return col
    return None


def _find_unit_column(columns: Iterable[str]) -> str | None:
    for col in columns:
        if str(col).strip().lower() in UNIT_LIKE:
            return col
    return None


def _normalize_column_header(value: str) -> str:
    text = str(value)
    return text.replace("\u00b5", "u").replace("\u03bc", "u")


def _normalize_key(value: str) -> str:
    base = str(value).strip().lower()
    base = re.sub(r"\(.*?\)", "", base)
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"[^a-z0-9% ]", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _normalize_variable_and_unit(name: str, unit: str | None) -> Tuple[str, str]:
    base = str(name).strip()
    unit_value = str(unit).strip() if unit is not None else ""
    key = _normalize_key(base)
    mapped = VARIABLE_NAME_MAP.get(key)
    if mapped:
        mapped_name, mapped_unit = mapped
        if mapped_unit and (not unit_value or unit_value.lower() == base.lower()):
            unit_value = mapped_unit
        return mapped_name, unit_value
    return base, unit_value


def _metadata_key(value: str) -> str:
    return _normalize_key(value).replace(" ", "_")


def _json_metadata(meta: Dict[str, object]) -> str:
    return json.dumps(meta, separators=(",", ":")) if meta else "{}"


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_numeric(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return np.nan
    text = text.replace(",", "")
    if text.startswith("<"):
        numeric = _safe_float(text[1:])
        return numeric / 2 if numeric is not None else np.nan
    if text.startswith(">"):
        numeric = _safe_float(text[1:])
        return numeric if numeric is not None else np.nan
    numeric = _safe_float(text)
    return numeric if numeric is not None else np.nan


def _parse_amount(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text == "":
        return np.nan
    numbers = [float(num) for num in AMOUNT_RE.findall(text)]
    if not numbers:
        return np.nan
    if "x" in text and len(numbers) >= 2:
        return numbers[0] * numbers[1]
    return numbers[0]


def _series_to_records(
    series: pd.Series,
    variable_name: str,
    unit: str,
    source: str,
    subject_id: str,
    metadata: str,
) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    frame = series.reset_index()
    frame.columns = ["measurement_date", "value"]
    frame["subject_id"] = subject_id
    frame["variable_name"] = variable_name
    frame["unit"] = unit
    frame["source"] = source
    frame["metadata"] = metadata
    frame = frame.dropna(subset=["value"])
    return frame[TIMELINE_COLUMNS]


def _is_cronometer_servings_frame(df: pd.DataFrame) -> bool:
    cols = {str(col).strip() for col in df.columns}
    return {"Day", "Food Name", "Amount"}.issubset(cols) and any("Energy" in col for col in cols)


def _is_apple_health_export(path: Path) -> bool:
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                return APPLE_HEALTH_EXPORT in zf.namelist()
        except zipfile.BadZipFile:
            return False
    if path.suffix.lower() == ".xml":
        return path.name.lower().startswith("export")
    return False


def _open_vcf(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _measurement_date_from_path(path: Path) -> pd.Timestamp:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return pd.Timestamp(datetime(ts.year, ts.month, ts.day, tzinfo=timezone.utc))


def _is_transition(ref: str, alt: str) -> bool:
    pair = f"{ref.upper()}{alt.upper()}"
    return pair in {"AG", "GA", "CT", "TC"}


def _classify_genotype(genotype: str) -> str:
    if not genotype:
        return "missing"
    alleles = re.split(r"[|/]", genotype)
    if any(allele == "." or allele == "" for allele in alleles):
        return "missing"
    if all(allele == "0" for allele in alleles):
        return "hom_ref"
    if len(set(alleles)) == 1:
        return "hom_alt"
    return "het"


def run_qc_report(timeline_df: pd.DataFrame) -> Dict[str, object]:
    if timeline_df.empty:
        return {"status": "empty", "issues": ["No data ingested"], "metrics": {}}

    metrics = {}
    issues: List[str] = []

    grouped = timeline_df.groupby("variable_name")
    missingness = (grouped["value"].apply(lambda series: float(series.isna().mean())).to_dict())
    metrics["missingness"] = missingness

    duplicates = int(
        timeline_df.duplicated(subset=["subject_id", "measurement_date", "variable_name"]).sum()
    )
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


def generate_qc_figures(
    timeline_df: pd.DataFrame,
    artifact_dir: Path,
) -> Dict[str, Dict[str, object]]:
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
