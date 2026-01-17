from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..data.timeline import TimelineStore

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def run_eda(*, timeline: TimelineStore, artifact_dir: Path, preferences: Dict[str, object]) -> Dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = timeline.data.copy()
    if df.empty:
        empty_path = artifact_dir / "eda_summary.json"
        empty_path.write_text("[]")
        return {
            "summary": "Timeline is empty; awaiting data for EDA",
            "summary_table": str(empty_path),
            "preview": "No data",
            "artifacts": {"eda_summary": {"path": str(empty_path), "description": "EDA summary", "mime_type": "application/json"}},
            "highlights": [],
        }

    df = df.dropna(subset=["value"])
    df["measurement_date"] = pd.to_datetime(df["measurement_date"], utc=True)

    trend_summary = _compute_trends(df)
    summary_path = artifact_dir / "eda_summary.json"
    trend_summary.to_json(summary_path, orient="records")

    highlights = _extract_highlights(trend_summary)

    figures = {}
    if plt is not None:
        figures.update(_plot_time_series(df, artifact_dir))
        figures.update(_plot_rolling_stats(df, artifact_dir))

    artifacts = {
        "eda_summary": {
            "path": str(summary_path),
            "description": "EDA trend summary",
            "mime_type": "application/json",
        },
    }
    artifacts.update(figures)

    summary = highlights[0] if highlights else "EDA completed"
    return {
        "summary": summary,
        "summary_table": str(summary_path),
        "preview": summary,
        "artifacts": artifacts,
        "highlights": highlights,
    }


def _compute_trends(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, group in df.groupby("variable_name"):
        group = group.sort_values("measurement_date")
        n = len(group)
        if n < 3:
            continue
        x = (group["measurement_date"] - group["measurement_date"].min()).dt.days.values.reshape(-1, 1)
        y = group["value"].values
        finite_mask = np.isfinite(y)
        if finite_mask.sum() < 2:
            continue
        x = x.flatten()[finite_mask]
        y = y[finite_mask]
        try:
            slope = np.polyfit(x, y, 1)[0] if len(y) >= 2 else 0.0
        except np.linalg.LinAlgError:
            continue
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        records.append(
            {
                "variable_name": name,
                "trend_direction": trend_direction,
                "slope": float(slope),
                "n_points": int(n),
                "latest_value": float(group.iloc[-1]["value"]),
            }
        )
    if not records:
        return pd.DataFrame(
            [{"variable_name": "n/a", "trend_direction": "flat", "slope": 0.0, "n_points": 0, "latest_value": 0.0}]
        )
    summary_df = pd.DataFrame(records)
    summary_df.sort_values(by="n_points", ascending=False, inplace=True)
    return summary_df


def _extract_highlights(trend_summary: pd.DataFrame) -> list[str]:
    highlights = []
    for _, row in trend_summary.head(3).iterrows():
        highlights.append(
            f"{row['variable_name']} is {row['trend_direction']} (slope={row['slope']:.3f}, latest={row['latest_value']:.2f})."
        )
    return highlights


def _plot_time_series(df: pd.DataFrame, artifact_dir: Path) -> Dict[str, Dict[str, object]]:
    top_vars = (
        df.groupby("variable_name")["measurement_date"]
        .count()
        .sort_values(ascending=False)
        .head(4)
        .index.tolist()
    )
    subset = df[df["variable_name"].isin(top_vars)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in subset.groupby("variable_name"):
        ax.plot(group["measurement_date"], group["value"], marker="o", label=name)
    ax.set_title("Top variable timelines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig_path = artifact_dir / "eda_time_series.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return {
        "eda_time_series": {
            "path": str(fig_path),
            "description": "Top variable timelines",
            "mime_type": "image/png",
        }
    }


def _plot_rolling_stats(df: pd.DataFrame, artifact_dir: Path) -> Dict[str, Dict[str, object]]:
    df = df.sort_values("measurement_date")
    # Example metric: rolling median for first variable
    first_var = df["variable_name"].unique()[0]
    subset = df[df["variable_name"] == first_var].copy()
    subset.set_index("measurement_date", inplace=True)
    subset["rolling_median"] = subset["value"].rolling(window=3, min_periods=1).median()
    subset["rolling_iqr"] = subset["value"].rolling(window=5, min_periods=1).apply(lambda x: np.subtract(*np.percentile(x, [75, 25])), raw=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(subset.index, subset["value"], color="lightgray", label="Value")
    ax.plot(subset.index, subset["rolling_median"], color="navy", label="Rolling median")
    ax.fill_between(
        subset.index,
        subset["rolling_median"] - subset["rolling_iqr"] / 2,
        subset["rolling_median"] + subset["rolling_iqr"] / 2,
        color="navy",
        alpha=0.1,
        label="Rolling IQR",
    )
    ax.set_title(f"Rolling normality bands for {first_var}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig_path = artifact_dir / "eda_rolling_stats.png"
    fig.savefig(fig_path)
    plt.close(fig)
    return {
        "eda_rolling_stats": {
            "path": str(fig_path),
            "description": "Rolling median and IQR",
            "mime_type": "image/png",
        }
    }
