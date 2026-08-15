"""Subject-level split helpers."""

from __future__ import annotations

import random

import pandas as pd


def subject_split(
    frame: pd.DataFrame,
    participant_col: str = "participant_id",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    participants = sorted(frame[participant_col].dropna().unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(participants)
    test_count = max(1, int(round(len(participants) * test_size)))
    test_ids = set(participants[:test_count])
    train = frame[~frame[participant_col].isin(test_ids)].copy()
    test = frame[frame[participant_col].isin(test_ids)].copy()
    return train, test

