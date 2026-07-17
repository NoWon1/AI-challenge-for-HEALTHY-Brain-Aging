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
    for participant_id, group in visits.sort_values([participant_col, order_col]).groupby(participant_col):
        values = group[order_col].tolist()
        if values != sorted(values):
            raise ValueError(f"Visits are not monotonic for participant {participant_id}")

