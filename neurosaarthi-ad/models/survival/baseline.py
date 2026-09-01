"""Dependency-light survival baselines."""

from __future__ import annotations

import pandas as pd


def kaplan_meier_table(frame: pd.DataFrame, time_col: str = "event_time_days", event_col: str = "event") -> pd.DataFrame:
    """Compute a simple Kaplan-Meier table for event times."""
    # ⚡ Bolt: Vectorized Kaplan-Meier computation avoids O(N^2) row-wise filtering and looping

    table = frame[[time_col, event_col]].dropna()
    if table.empty:
        return pd.DataFrame(columns=["time", "at_risk", "events", "censored", "survival"])

    agg = table.groupby(time_col).agg(
        events=(event_col, "sum"),
        total=(event_col, "size")
    )
    agg["censored"] = agg["total"] - agg["events"]

    # Calculate at_risk by reverse cumulative sum of total
    agg["at_risk"] = agg["total"].iloc[::-1].cumsum().iloc[::-1]

    # Calculate survival
    agg["survival"] = (1.0 - (agg["events"] / agg["at_risk"]).fillna(0)).cumprod()

    agg = agg.reset_index().rename(columns={time_col: "time"})
    # Cast back to match original integer types
    agg["events"] = agg["events"].astype(int)
    agg["censored"] = agg["censored"].astype(int)
    agg["at_risk"] = agg["at_risk"].astype(int)

    return agg[["time", "at_risk", "events", "censored", "survival"]]

