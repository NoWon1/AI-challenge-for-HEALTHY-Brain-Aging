"""Base interfaces for cohort ETL adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class CohortTables:
    participants: pd.DataFrame
    visits: pd.DataFrame
    modality_features: pd.DataFrame
    outcomes: pd.DataFrame


class CohortAdapter(ABC):
    cohort_name: str

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    @abstractmethod
    def extract(self) -> CohortTables:
        """Return data mapped into the common data model."""


    def _build_outcomes(self, raw: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        """Build standard progression outcomes from visits."""
        rows = []
        if visits.empty:
            return pd.DataFrame()

        for pid, group in visits.groupby('participant_id'):
            group = group.sort_values('baseline_days')
            baseline_dx = group.iloc[0]['diagnosis'] if len(group) > 0 else 'unknown'
            baseline_visit = f"{pid}-V0"

            conversion_days = None
            for _, visit in group.iterrows():
                if baseline_dx == 'cognitively_unimpaired' and visit['diagnosis'] in ('mci', 'dementia'):
                    conversion_days = visit['baseline_days']
                    break
                elif baseline_dx == 'mci' and visit['diagnosis'] == 'dementia':
                    conversion_days = visit['baseline_days']
                    break

            follow_up_days = group.iloc[-1]['baseline_days']

            if conversion_days is not None:
                event_time_days = conversion_days
            else:
                event_time_days = max(follow_up_days, 1)

            for horizon in (1, 3, 5):
                horizon_days = int(round(horizon * 365.25))
                event = 1 if (conversion_days is not None and event_time_days <= horizon_days) else 0
                rows.append({
                    'outcome_id': f"{pid}-risk-{horizon}y",
                    'participant_id': pid,
                    'anchor_visit_id': baseline_visit,
                    'endpoint': f'incident_progression_{horizon}y',
                    'horizon_days': horizon_days,
                    'event': event,
                    'event_time_days': event_time_days,
                    'future_score': np.nan,
                    'censoring_reason': 'study_end' if event == 0 else '',
                })

        return pd.DataFrame(rows).sort_values(['participant_id', 'horizon_days']).reset_index(drop=True)
