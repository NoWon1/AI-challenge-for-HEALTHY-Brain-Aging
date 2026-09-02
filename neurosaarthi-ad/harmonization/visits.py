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
    # ⚡ Bolt: Fast-path vectorised monotonicity check avoids slow loops for the happy path
    if visits.empty:
        return

    if visits.groupby(participant_col, sort=False)[order_col].diff().fillna(0).ge(0).all():
        return

    for participant_id, group in visits.groupby(participant_col, sort=False):
        values = group[order_col].tolist()
        if values != sorted(values):
            raise ValueError(f"Visits are not monotonic for participant {participant_id}")

