"""Shared feature utilities."""

from __future__ import annotations

import pandas as pd


def pivot_modality_features(features: pd.DataFrame, modality: str) -> pd.DataFrame:
    subset = features[features["modality"] == modality]
    if subset.empty:
        return pd.DataFrame()
    return subset.pivot_table(
        index=["participant_id", "visit_id"],
        columns="feature_name",
        values="value",
        aggfunc="first",
    ).reset_index()


def add_missingness_indicators(frame: pd.DataFrame, protected_columns: set[str] | None = None) -> pd.DataFrame:
    protected = protected_columns or set()
    output = frame.copy()
    for column in frame.columns:
        if column not in protected:
            output[f"{column}__missing"] = frame[column].isna().astype(int)
    return output

