"""Model comparison utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelComparisonReport:
    """Data structure for a model's evaluation metrics on a cohort."""
    model_name: str
    metrics_dict: dict[str, float]
    cohort: str


def compare_models(reports: list[ModelComparisonReport]) -> pd.DataFrame:
    """Compare multiple models across metrics.

    Args:
        reports: List of ModelComparisonReport instances.

    Returns:
        DataFrame summarizing the comparisons.
    """
    records = []
    for report in reports:
        row = {"model_name": report.model_name, "cohort": report.cohort}
        row.update(report.metrics_dict)
        records.append(row)
    return pd.DataFrame(records)


def _norm_cdf(z: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_sf(z: float) -> float:
    """Standard normal survival function."""
    return 1.0 - _norm_cdf(z)


def delong_test(y_true: np.ndarray | pd.Series, y_score_a: np.ndarray | pd.Series, y_score_b: np.ndarray | pd.Series) -> float:
    """Compute p-value for comparing two AUROCs using DeLong's test.

    Args:
        y_true: True binary labels.
        y_score_a: Predicted scores from model A.
        y_score_b: Predicted scores from model B.

    Returns:
        p-value from DeLong's test.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_score_a, pd.Series):
        y_score_a = y_score_a.to_numpy()
    if isinstance(y_score_b, pd.Series):
        y_score_b = y_score_b.to_numpy()

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    m = np.sum(pos_mask)
    n = np.sum(neg_mask)

    if m == 0 or n == 0:
        raise ValueError("Requires both positive and negative samples.")

    pos_preds_a = y_score_a[pos_mask]
    neg_preds_a = y_score_a[neg_mask]
    pos_preds_b = y_score_b[pos_mask]
    neg_preds_b = y_score_b[neg_mask]

    # Compute empirical structural components
    V10_a = np.array([np.sum(neg_preds_a < pa) + 0.5 * np.sum(neg_preds_a == pa) for pa in pos_preds_a]) / n
    V01_a = np.array([np.sum(pos_preds_a > na) + 0.5 * np.sum(pos_preds_a == na) for na in neg_preds_a]) / m
    V10_b = np.array([np.sum(neg_preds_b < pb) + 0.5 * np.sum(neg_preds_b == pb) for pb in pos_preds_b]) / n
    V01_b = np.array([np.sum(pos_preds_b > nb) + 0.5 * np.sum(pos_preds_b == nb) for nb in neg_preds_b]) / m

    auc_a = np.mean(V10_a)
    auc_b = np.mean(V10_b)

    S10 = np.cov(V10_a, V10_b, rowvar=False) if m > 1 else np.zeros((2, 2))
    S01 = np.cov(V01_a, V01_b, rowvar=False) if n > 1 else np.zeros((2, 2))

    S = S10 / m + S01 / n

    L = np.array([1, -1])
    diff = auc_a - auc_b
    var = L @ S @ L.T

    if var == 0:
        return 1.0

    z = diff / np.sqrt(var)
    return float(2 * _norm_sf(abs(z)))


def bootstrap_comparison(
    y_true: np.ndarray | pd.Series,
    y_score_a: np.ndarray | pd.Series,
    y_score_b: np.ndarray | pd.Series,
    metric_fn: Callable[..., float],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> dict[str, float]:
    """Compare two models using bootstrap resampling.

    Args:
        y_true: True labels.
        y_score_a: Scores from model A.
        y_score_b: Scores from model B.
        metric_fn: Function to compute metric.
        n_bootstrap: Number of iterations.
        seed: Random seed.

    Returns:
        Dictionary with delta, p_value, ci_lower, ci_upper.
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_score_a, pd.Series):
        y_score_a = y_score_a.to_numpy()
    if isinstance(y_score_b, pd.Series):
        y_score_b = y_score_b.to_numpy()

    rng = np.random.default_rng(seed)
    n = len(y_true)

    base_a = metric_fn(y_true, y_score_a)
    base_b = metric_fn(y_true, y_score_b)
    base_diff = base_a - base_b

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_a = metric_fn(y_true[idx], y_score_a[idx])
        boot_b = metric_fn(y_true[idx], y_score_b[idx])
        diffs.append(boot_a - boot_b)

    diffs_arr = np.array(diffs)
    ci_lower = float(np.percentile(diffs_arr, 2.5))
    ci_upper = float(np.percentile(diffs_arr, 97.5))

    p_value = float(np.mean(diffs_arr <= 0)) if base_diff > 0 else float(np.mean(diffs_arr >= 0))
    p_value = min(p_value * 2, 1.0)

    return {
        "delta": base_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def select_best_model(reports: list[ModelComparisonReport], metric: str = "auroc", validation_set: str = "external") -> str | None:
    """Select the best model based on a specified metric and validation set cohort.

    Args:
        reports: List of comparison reports.
        metric: Metric to optimize.
        validation_set: Cohort name to use for selection.

    Returns:
        Name of the best model, or None if no matching reports found.
    """
    valid_reports = [r for r in reports if r.cohort == validation_set and metric in r.metrics_dict]
    if not valid_reports:
        return None

    best_report = max(valid_reports, key=lambda r: r.metrics_dict[metric])
    return best_report.model_name
