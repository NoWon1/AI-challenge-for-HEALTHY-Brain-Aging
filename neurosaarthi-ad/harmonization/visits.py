"""Visit alignment helpers."""

from __future__ import annotations

import pandas as pd


def add_baseline_offsets(visits: pd.DataFrame, participant_col: str = "participant_id", date_col: str = "visit_date") -> pd.DataFrame:
    """Add days from first observed visit for each participant."""

    output = visits.copy()
    output[date_col] = pd.to_datetime(output[date_col])
    baseline = output.groupby(participant_col)[date_col].transform("min")
    output["baseline_days"] = (output[date_col] - baseline).dt.days
    return output


def require_monotonic_visits(visits: pd.DataFrame, participant_col: str = "participant_id", order_col: str = "visit_index") -> None:
    # ⚡ Bolt: Fast-path vectorised monotonicity check avoids slow .groupby() loops for the happy path
    if visits.empty:
        return

    sorted_visits = visits.sort_values(participant_col, kind="mergesort")
    p_arr = sorted_visits[participant_col].to_numpy()
    v_arr = sorted_visits[order_col].to_numpy()
    same_pid = p_arr[1:] == p_arr[:-1]
    diffs = v_arr[1:] - v_arr[:-1]

    if bool((diffs[same_pid] >= 0).all()):
        return

    for participant_id, group in visits.groupby(participant_col, sort=False):
        values = group[order_col].tolist()
        if values != sorted(values):
            raise ValueError("Visits are not monotonic for one or more participants")

