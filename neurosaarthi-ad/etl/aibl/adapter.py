"""AIBL cohort ETL adapter.

Parses AIBL clinical CSV tables.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from etl.base import CohortAdapter, CohortTables

_AIBL_DX_MAP = {
    1: 'cognitively_unimpaired',
    2: 'mci',
    3: 'dementia',
}

@dataclass
class AiblAdapter(CohortAdapter):
    data_dir: str | Path = 'data/raw/aibl'
    cohort_name: str = 'AIBL'
    
    def extract(self) -> CohortTables:
        root = Path(self.data_dir)
        clinical_path = root / 'aibl_clinical.csv'
        
        if not clinical_path.exists():
            raise FileNotFoundError(
                f'AIBL clinical data not found at {clinical_path}. '
                'Please place the standard AIBL CSV file here.'
            )
            
        raw = pd.read_csv(clinical_path, low_memory=False)
        
        participants = self._build_participants(raw)
        visits = self._build_visits(raw)
        features = self._build_features(raw, visits)
        outcomes = self._build_outcomes(raw, visits)
        
        return CohortTables(
            participants=participants,
            visits=visits,
            modality_features=features,
            outcomes=outcomes,
        )

    def _build_participants(self, raw: pd.DataFrame) -> pd.DataFrame:
        baseline = raw.drop_duplicates('RID', keep='first').copy()
        
        return pd.DataFrame({
            'participant_id': 'AIBL-' + baseline['RID'].astype(str),
            'cohort': self.cohort_name,
            'sex': baseline.get('PTGENDER', pd.Series(dtype=str)).map({1: 'Male', 2: 'Female', 'Male': 'Male', 'Female': 'Female'}).fillna('Unknown'),
            'birth_year': pd.Series(dtype='Int64'), # Typically derived if age and examdate are present
            'education_years': pd.to_numeric(baseline.get('PTEDUCAT', np.nan), errors='coerce').fillna(0),
            'language': 'cohort_recorded',
            'urban_rural': 'reference',
        })

    def _build_visits(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw['participant_id'] = 'AIBL-' + raw['RID'].astype(str)
        
        viscode_order = {'bl': 0, 'm18': 1, 'm36': 2, 'm54': 3, 'm72': 4, 'm90': 5, 'm108': 6, 'm126': 7}
        raw['visit_index'] = raw.get('VISCODE', pd.Series(dtype=str)).str.lower().map(viscode_order).fillna(0).astype(int)
        
        # Approximate baseline days from visit_index if no date exists
        raw['baseline_days'] = raw['visit_index'] * (18 * 30) # Roughly 18 months per index typically
        
        raw['diagnosis'] = pd.to_numeric(raw.get('DXCURREN', np.nan), errors='coerce').map(_AIBL_DX_MAP).fillna('unknown')
        raw['cdr_global'] = pd.to_numeric(raw.get('CDRSB', np.nan), errors='coerce').fillna(0.0) / 18.0
        
        visits = pd.DataFrame({
            'visit_id': raw['participant_id'] + '-V' + raw['visit_index'].astype(str),
            'participant_id': raw['participant_id'],
            'cohort': self.cohort_name,
            'visit_index': raw['visit_index'],
            'age_at_visit': pd.to_numeric(raw.get('AGE', 70.0), errors='coerce'),
            'visit_date': pd.NaT,
            'baseline_days': raw['baseline_days'],
            'diagnosis': raw['diagnosis'],
            'cdr_global': raw['cdr_global'],
            'cognitive_status': raw['diagnosis'],
        })
        return visits.sort_values(['participant_id', 'visit_index']).reset_index(drop=True)

    def _build_features(self, raw: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw['participant_id'] = 'AIBL-' + raw['RID'].astype(str)
        viscode_order = {'bl': 0, 'm18': 1, 'm36': 2, 'm54': 3, 'm72': 4, 'm90': 5, 'm108': 6, 'm126': 7}
        raw['visit_index'] = raw.get('VISCODE', pd.Series(dtype=str)).str.lower().map(viscode_order).fillna(0).astype(int)
        raw['visit_id'] = raw['participant_id'] + '-V' + raw['visit_index'].astype(str)
        
        feature_map = {
            'MMSCORE': ('cognition', 'cognitive_score', 'points'),
        }
        
        rows = []
        for _, record in raw.iterrows():
            for source_col, (modality, feature_name, unit) in feature_map.items():
                if source_col in record and pd.notna(record[source_col]):
                    rows.append({
                        'feature_row_id': f"{record['visit_id']}-{feature_name}",
                        'participant_id': record['participant_id'],
                        'visit_id': record['visit_id'],
                        'cohort': self.cohort_name,
                        'modality': modality,
                        'feature_name': feature_name,
                        'value': float(record[source_col]),
                        'unit': unit,
                        'source_variable': source_col,
                        'qc_flag': 'pass',
                        'derived': False,
                    })
        return pd.DataFrame(rows).sort_values(['participant_id', 'visit_id']).reset_index(drop=True) if rows else pd.DataFrame()

    def _build_outcomes(self, raw: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw['participant_id'] = 'AIBL-' + raw['RID'].astype(str)
        viscode_order = {'bl': 0, 'm18': 1, 'm36': 2, 'm54': 3, 'm72': 4, 'm90': 5, 'm108': 6, 'm126': 7}
        raw['visit_index'] = raw.get('VISCODE', pd.Series(dtype=str)).str.lower().map(viscode_order).fillna(0).astype(int)
        raw['baseline_days'] = raw['visit_index'] * (18 * 30)
        raw['diagnosis'] = pd.to_numeric(raw.get('DXCURREN', np.nan), errors='coerce').map(_AIBL_DX_MAP).fillna('unknown')
        
        rows = []
        for pid, group in raw.groupby('participant_id'):
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
