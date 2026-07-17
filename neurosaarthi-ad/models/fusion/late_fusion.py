"""Late-fusion utilities for modality-level scores."""

from __future__ import annotations

import pandas as pd


def weighted_score_fusion(scores: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    missing = [column for column in weights if column not in scores.columns]
    if missing:
        raise ValueError(f"Missing score columns: {', '.join(missing)}")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    fused = sum(scores[column] * weight for column, weight in weights.items()) / total_weight
    return pd.Series(fused, index=scores.index, name="fused_score")

