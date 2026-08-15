"""Late-fusion utilities for modality-level scores."""

from __future__ import annotations

import pandas as pd


def weighted_score_fusion(scores: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Fuse scores while renormalising over modalities present in each row.

    A missing modality is represented by ``NaN``. Rows with no available
    modality return ``NaN`` rather than a falsely precise zero.
    """

    missing = [column for column in weights if column not in scores.columns]
    if missing:
        raise ValueError(f"Missing score columns: {', '.join(missing)}")
    if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Fusion weights must be non-negative and sum to a positive value")

    weight_series = pd.Series(weights, dtype=float)
    aligned = scores[list(weights)].apply(pd.to_numeric, errors="coerce")
    available = aligned.notna()
    numerator = aligned.fillna(0.0).mul(weight_series, axis="columns").sum(axis=1)
    denominator = available.mul(weight_series, axis="columns").sum(axis=1)
    fused = numerator.div(denominator.where(denominator > 0))
    return pd.Series(fused, index=scores.index, name="fused_score")
