"""Visit alignment helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_baseline_offsets(visits: pd.DataFrame, participant_col: str = "participant_id", date_col: str = "visit_date") -> pd.DataFrame:
    """Add days from first observed visit for each participant."""

    output = visits.copy()
    output[date_col] = pd.to_datetime(output[date_col])
    baseline = output.groupby(participant_col)[date_col].transform("min")
    output["baseline_days"] = (output[date_col] - baseline).dt.days
    return output


def require_monotonic_visits(visits: pd.DataFrame, participant_col: str = "participant_id", order_col: str = "visit_index") -> None:
    # ⚡ Bolt: Fast-path vectorised monotonicity check avoids slow loops for the happy path
    if visits.empty:
        return

    # Use fully vectorized numpy slice logic to avoid slow O(N) Pandas groupby loop overhead
    # We must sort by participant_id first to ensure contiguous groups for slice comparisons
    # NOTE: Do NOT sort by order_col, otherwise we are just reordering the visits and defeating the check!
    # A stable sort ensures the original row order within each participant group is preserved.
    sorted_visits = visits.sort_values([participant_col], kind="stable")
    pids = sorted_visits[participant_col].to_numpy()
    vals = sorted_visits[order_col].to_numpy()
    same_pid = pids[1:] == pids[:-1]
    diffs = vals[1:] - vals[:-1]

    if bool(np.all(diffs[same_pid] >= 0)):
        return

    for participant_id, group in visits.groupby(participant_col, sort=False):
        values = group[order_col].tolist()
        if values != sorted(values):
            raise ValueError("Visits are not monotonic for one or more participants")

