"""Calibration summaries for risk scores."""

from __future__ import annotations

import pandas as pd


def calibration_bins(
    y_true: pd.Series, y_score: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()

    if frame.empty:
        return pd.DataFrame(columns=["mean_predicted", "observed_rate", "n"]).astype(
            {"n": int}
        )

    if frame["y_score"].nunique() == 1:
        frame["bin"] = 1
    else:
        frame["bin"] = pd.qcut(
            frame["y_score"], q=min(n_bins, len(frame)), duplicates="drop"
        )

    return (
        frame.groupby("bin", observed=True)
        .agg(
            mean_predicted=("y_score", "mean"),
            observed_rate=("y_true", "mean"),
            n=("y_true", "size"),
        )
        .reset_index(drop=True)
    )
