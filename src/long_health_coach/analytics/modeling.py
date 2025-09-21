from __future__ import annotations

import json
from math import erf, sqrt
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from ..data.timeline import TimelineStore

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def run_exposure_outcome_models(
    *,
    timeline: TimelineStore,
    artifact_dir: Path,
    preferences: Dict[str, object],
) -> Dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = timeline.data.copy()
    if df.empty:
        path = artifact_dir / "model_results.json"
        path.write_text("[]")
        return {
            "summary": "No data available for modeling",
            "results_table": str(path),
            "preview": "No data",
            "artifacts": {"model_results": {"path": str(path), "description": "Model results", "mime_type": "application/json"}},
            "notes": "",
            "headlines": [],
        }

    pivot = _prepare_wide_table(df)
    if pivot.empty:
        path = artifact_dir / "model_results.json"
        path.write_text("[]")
        return {
            "summary": "Insufficient overlap between variables for modeling",
            "results_table": str(path),
            "preview": "No overlap",
            "artifacts": {"model_results": {"path": str(path), "description": "Model results", "mime_type": "application/json"}},
            "notes": "",
            "headlines": [],
        }

    exposures, outcomes = _select_variables(pivot.columns, preferences)
    results = _run_pairwise_models(pivot, exposures, outcomes)
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="p_value", inplace=True)

    results_path = artifact_dir / "model_results.json"
    results_df.to_json(results_path, orient="records")

    figures = {}
    if plt is not None and not results_df.empty:
        figures.update(_plot_effect_sizes(results_df, artifact_dir))
        figures.update(_plot_lag_scan(pivot, results_df.iloc[0], artifact_dir))

    summary = (
        f"Modeled {len(results_df)} exposure→outcome pairs; "
        f"top association {results_df.iloc[0]['exposure']}→{results_df.iloc[0]['outcome']} "
        f"beta={results_df.iloc[0]['beta']:.3f}" if not results_df.empty else "Modeling completed"
    )

    artifacts = {
        "model_results": {
            "path": str(results_path),
            "description": "Exposure-outcome model results",
            "mime_type": "application/json",
        }
    }
    artifacts.update(figures)

    headlines = [
        f"{row['exposure']} → {row['outcome']} beta={row['beta']:.3f} (p={row['p_value']:.3g})"
        for _, row in results_df.head(3).iterrows()
    ]

    notes = "Results use simple OLS with normal approximation; verify with domain experts when acting." if not results_df.empty else ""

    return {
        "summary": summary,
        "results_table": str(results_path),
        "preview": summary,
        "artifacts": artifacts,
        "notes": notes,
        "headlines": headlines,
    }


def _prepare_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="measurement_date",
        columns="variable_name",
        values="value",
        aggfunc="mean",
    )
    pivot.sort_index(inplace=True)
    pivot = pivot.dropna(how="all", axis=1)
    pivot = pivot.interpolate(limit_direction="both")
    pivot = pivot.dropna()
    return pivot


def _select_variables(columns: Iterable[str], preferences: Dict[str, object]) -> tuple[List[str], List[str]]:
    columns = list(columns)
    exposures_pref = preferences.get("exposures")
    outcomes_pref = preferences.get("outcomes")
    if exposures_pref:
        exposures = [col for col in exposures_pref if col in columns]
    else:
        exposures = [col for col in columns if any(keyword in col.lower() for keyword in ["exercise", "activity", "supplement", "steps"])]
        if not exposures:
            exposures = columns[: max(1, len(columns) // 3)]
    if outcomes_pref:
        outcomes = [col for col in outcomes_pref if col in columns]
    else:
        outcomes = [col for col in columns if any(keyword in col.lower() for keyword in ["pace", "glucose", "chol", "sleep"])]
        if not outcomes:
            outcomes = columns
    exposures = [col for col in exposures if col not in outcomes]
    return exposures, outcomes


def _run_pairwise_models(pivot: pd.DataFrame, exposures: List[str], outcomes: List[str]) -> List[dict]:
    results: List[dict] = []
    for outcome in outcomes:
        if outcome not in pivot.columns or pivot[outcome].std(ddof=0) == 0:
            continue
        y = pivot[outcome].values
        for exposure in exposures:
            if exposure not in pivot.columns or exposure == outcome:
                continue
            x = pivot[exposure].values
            if np.allclose(x.std(ddof=0), 0):
                continue
            beta, se = _simple_ols(x, y)
            if se == 0:
                continue
            t_stat = beta / se
            p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
            ci_low = beta - 1.96 * se
            ci_high = beta + 1.96 * se
            results.append(
                {
                    "outcome": outcome,
                    "exposure": exposure,
                    "beta": float(beta),
                    "se": float(se),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "p_value": float(p_value),
                }
            )
    return results


def _simple_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones_like(x), x])
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta_hat
    residuals = y - y_hat
    n = len(x)
    dof = max(1, n - X.shape[1])
    sigma2 = float((residuals @ residuals) / dof)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    beta = float(beta_hat[1])
    se = float(np.sqrt(max(cov[1, 1], 1e-12)))
    return beta, se


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + erf(z / sqrt(2)))


def _plot_effect_sizes(results_df: pd.DataFrame, artifact_dir: Path) -> Dict[str, Dict[str, object]]:
    top = results_df.head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top)), top["beta"], xerr=1.96 * top["se"], color="teal")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{row.exposure}→{row.outcome}" for row in top.itertuples()])
    ax.set_xlabel("Beta (95% CI)")
    ax.set_title("Exposure→Outcome Effect Sizes")
    fig.tight_layout()
    path = artifact_dir / "model_effect_sizes.png"
    fig.savefig(path)
    plt.close(fig)
    return {
        "model_effect_sizes": {
            "path": str(path),
            "description": "Effect sizes with 95% CI",
            "mime_type": "image/png",
        }
    }


def _plot_lag_scan(pivot: pd.DataFrame, best_row: pd.Series, artifact_dir: Path) -> Dict[str, Dict[str, object]]:
    exposure = best_row["exposure"]
    outcome = best_row["outcome"]
    exposure_series = pivot[exposure]
    outcome_series = pivot[outcome]
    max_lag = 7
    lags = range(0, max_lag + 1)
    correlations = []
    for lag in lags:
        shifted = exposure_series.shift(lag)
        aligned = pd.concat([shifted, outcome_series], axis=1).dropna()
        if aligned.empty:
            correlations.append(np.nan)
        else:
            correlations.append(aligned.corr().iloc[0, 1])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(lags), correlations, marker="o")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Lag scan: {exposure} vs {outcome}")
    fig.tight_layout()
    path = artifact_dir / "model_lag_scan.png"
    fig.savefig(path)
    plt.close(fig)
    return {
        "model_lag_scan": {
            "path": str(path),
            "description": "Lagged correlation scan",
            "mime_type": "image/png",
        }
    }
