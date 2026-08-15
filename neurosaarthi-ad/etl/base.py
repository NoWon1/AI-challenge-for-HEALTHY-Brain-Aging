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

        # ⚡ Bolt: Vectorized progression outcome calculation avoids slow .iterrows() loop
        v = visits.sort_values(['participant_id', 'baseline_days'])
        first_visits = v.groupby('participant_id', as_index=False).first()
        last_visits = v.groupby('participant_id', as_index=False).last()

        v = v.merge(first_visits[['participant_id', 'diagnosis']], on='participant_id', suffixes=('', '_base'))

        is_conv = (((v['diagnosis_base'] == 'cognitively_unimpaired') & v['diagnosis'].isin(['mci', 'dementia'])) |
                   ((v['diagnosis_base'] == 'mci') & (v['diagnosis'] == 'dementia')))

        conv_days = v[is_conv].groupby('participant_id')['baseline_days'].first()

        summary = pd.DataFrame({'participant_id': first_visits['participant_id']}).set_index('participant_id')
        summary['follow_up'] = last_visits.set_index('participant_id')['baseline_days']
        summary['conv'] = conv_days
        summary['event_time_days'] = summary['conv'].combine_first(summary['follow_up'].clip(lower=1))

        horizons = pd.DataFrame({'horizon_years': [1, 3, 5], 'horizon_days': [365, 1096, 1826]}) # int(round(horizon * 365.25))
        outcomes = summary.reset_index().merge(horizons, how='cross')

        outcomes['outcome_id'] = outcomes['participant_id'] + "-risk-" + outcomes['horizon_years'].astype(str) + "y"
        outcomes['anchor_visit_id'] = outcomes['participant_id'] + "-V0"
        outcomes['endpoint'] = "incident_progression_" + outcomes['horizon_years'].astype(str) + "y"
        outcomes['event'] = (outcomes['conv'].notna() & (outcomes['event_time_days'] <= outcomes['horizon_days'])).astype(int)
        outcomes['future_score'] = np.nan
        outcomes['censoring_reason'] = np.where(outcomes['event'] == 0, 'study_end', '')

        cols = ['outcome_id', 'participant_id', 'anchor_visit_id', 'endpoint', 'horizon_days', 'event', 'event_time_days', 'future_score', 'censoring_reason']
        return outcomes[cols].sort_values(['participant_id', 'horizon_days']).reset_index(drop=True)
