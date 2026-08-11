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
        if visits.empty:
            return pd.DataFrame()

        # ⚡ Bolt Optimization: Replace iterrows with vectorized grouping
        v = visits.sort_values(['participant_id', 'baseline_days'])

        first_v = v.groupby('participant_id', as_index=False).first()
        last_v = v.groupby('participant_id', as_index=False).last()

        v = v.merge(first_v[['participant_id', 'diagnosis']].rename(columns={'diagnosis': 'base_dx'}), on='participant_id')

        conv_mask = ((v['base_dx'] == 'cognitively_unimpaired') & v['diagnosis'].isin(['mci', 'dementia'])) | \
                    ((v['base_dx'] == 'mci') & (v['diagnosis'] == 'dementia'))

        first_conv = v[conv_mask].groupby('participant_id')['baseline_days'].first()

        df = first_v[['participant_id']].copy()
        df['conversion_days'] = df['participant_id'].map(first_conv)
        df['follow_up_days'] = df['participant_id'].map(last_v.set_index('participant_id')['baseline_days'])

        df['event_time_days'] = df['conversion_days'].combine_first(df['follow_up_days'].clip(lower=1)).astype(int)
        df['anchor_visit_id'] = df['participant_id'] + '-V0'

        dfs = []
        for horizon in (1, 3, 5):
            horizon_days = int(round(horizon * 365.25))
            h_df = df.copy()
            h_df['outcome_id'] = h_df['participant_id'] + f'-risk-{horizon}y'
            h_df['endpoint'] = f'incident_progression_{horizon}y'
            h_df['horizon_days'] = horizon_days
            h_df['event'] = (h_df['conversion_days'].notna() & (h_df['event_time_days'] <= horizon_days)).astype(int)
            h_df['future_score'] = np.nan
            h_df['censoring_reason'] = np.where(h_df['event'] == 0, 'study_end', '')
            dfs.append(h_df)

        res = pd.concat(dfs)[['outcome_id', 'participant_id', 'anchor_visit_id', 'endpoint', 'horizon_days', 'event', 'event_time_days', 'future_score', 'censoring_reason']]
        return res.sort_values(['participant_id', 'horizon_days']).reset_index(drop=True)
