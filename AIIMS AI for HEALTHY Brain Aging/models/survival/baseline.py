"""Dependency-light survival baselines."""

from __future__ import annotations

import pandas as pd


def kaplan_meier_table(frame: pd.DataFrame, time_col: str = "event_time_days", event_col: str = "event") -> pd.DataFrame:
    """Compute a simple Kaplan-Meier table for event times."""

    table = frame[[time_col, event_col]].dropna().sort_values(time_col)
    rows = []
    survival = 1.0
    for time, group in table.groupby(time_col):
        at_risk = int((table[time_col] >= time).sum())
        events = int(group[event_col].sum())
        censored = int((group[event_col] == 0).sum())
        if at_risk > 0:
            survival *= 1.0 - (events / at_risk)
        rows.append({"time": time, "at_risk": at_risk, "events": events, "censored": censored, "survival": survival})
    return pd.DataFrame(rows)

