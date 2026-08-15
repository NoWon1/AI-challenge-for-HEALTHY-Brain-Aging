"""Leakage checks for longitudinal prediction tables."""

from __future__ import annotations

import pandas as pd


def assert_no_future_features(frame: pd.DataFrame, anchor_col: str = "anchor_days", feature_time_col: str = "feature_days") -> None:
    leaking = frame[frame[feature_time_col] > frame[anchor_col]]
    if not leaking.empty:
        raise ValueError(f"Found {len(leaking)} feature rows after the prediction anchor")


def assert_disjoint_participants(train_ids: set[str], *other_id_sets: set[str]) -> None:
    for index, ids in enumerate(other_id_sets, start=1):
        overlap = train_ids.intersection(ids)
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Train participants overlap split {index}: {preview}")

