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
    if len(visits) <= 1:
        return

    groups = visits[participant_col].to_numpy()
    values = visits[order_col].to_numpy()

    # Identify consecutive records belonging to the same participant
    same_group = groups[1:] == groups[:-1]

    # Evaluate monotonicity only across intra-group transitions
    is_monotonic = values[1:][same_group] >= values[:-1][same_group]

    if not np.all(is_monotonic):
        raise ValueError("Visits are not monotonic for one or more participants")

