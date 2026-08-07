"""OASIS cohort ETL adapter.

Parses OASIS-3 tabular data (demographics and clinical sessions).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from etl.base import CohortAdapter, CohortTables

@dataclass
class OasisAdapter(CohortAdapter):
    data_dir: str | Path = 'data/raw/oasis'
    cohort_name: str = 'OASIS'
    
    def extract(self) -> CohortTables:
        root = Path(self.data_dir)
        demographics_path = root / 'OASIS3_demographics.csv'
        clinical_path = root / 'OASIS3_clinical.csv'
        
        if not demographics_path.exists() or not clinical_path.exists():
            raise FileNotFoundError(
                f'OASIS data not found. Please ensure OASIS3_demographics.csv '
                f'and OASIS3_clinical.csv exist in {root}'
            )
            
        demo_raw = pd.read_csv(demographics_path, low_memory=False)
        clin_raw = pd.read_csv(clinical_path, low_memory=False)
        
        # Merge clinical and demo
        raw = clin_raw.merge(demo_raw, on='Subject', how='left')
        
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
        baseline = raw.drop_duplicates('Subject', keep='first').copy()
        
        sex_map = {'M': 'Male', 'F': 'Female'}
        
        return pd.DataFrame({
            'participant_id': 'OASIS-' + baseline['Subject'].astype(str),
            'cohort': self.cohort_name,
            'sex': baseline.get('M/F', pd.Series(dtype=str)).map(sex_map).fillna('Unknown'),
            'birth_year': pd.Series(dtype='Int64'), # Not explicitly available without exam date deduction usually
            'education_years': pd.to_numeric(baseline.get('EDUC', np.nan), errors='coerce').fillna(0),
            'language': 'cohort_recorded',
            'urban_rural': 'reference',
        })

    def _build_visits(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw['participant_id'] = 'OASIS-' + raw['Subject'].astype(str)
        # Infer visit index from dates or session labels if available, here we just enumerate
        raw = raw.sort_values(['Subject', 'Age'])
        raw['visit_index'] = raw.groupby('Subject').cumcount()
        raw['exam_date'] = pd.NaT # Without exact dates, rely on age diffs for baseline_days
        
        baseline_age = raw.groupby('participant_id')['Age'].transform('min')
        raw['baseline_days'] = ((raw['Age'] - baseline_age) * 365.25).fillna(0).astype(int)
        
        # DX mapping. OASIS usually has dx1 (Cognitively normal, AD Dementia, etc)
        def map_dx(val):
            if pd.isna(val): return 'unknown'
            v = str(val).lower()
            if 'normal' in v or 'cn' in v: return 'cognitively_unimpaired'
            if 'mci' in v: return 'mci'
            if 'ad' in v or 'dementia' in v: return 'dementia'
            return 'unknown'
            
        raw['diagnosis'] = raw.get('dx1', pd.Series(dtype=str)).apply(map_dx)
        raw['cdr_global'] = pd.to_numeric(raw.get('CDR', np.nan), errors='coerce').fillna(0.0) / 3.0
        
        visits = pd.DataFrame({
            'visit_id': raw['participant_id'] + '-V' + raw['visit_index'].astype(str),
            'participant_id': raw['participant_id'],
            'cohort': self.cohort_name,
            'visit_index': raw['visit_index'],
            'age_at_visit': pd.to_numeric(raw['Age'], errors='coerce'),
            'visit_date': raw['exam_date'],
            'baseline_days': raw['baseline_days'],
            'diagnosis': raw['diagnosis'],
            'cdr_global': raw['cdr_global'],
            'cognitive_status': raw['diagnosis'],
        })
        return visits.sort_values(['participant_id', 'visit_index']).reset_index(drop=True)

    def _build_features(self, raw: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw['participant_id'] = 'OASIS-' + raw['Subject'].astype(str)
        raw = raw.sort_values(['Subject', 'Age'])
        raw['visit_index'] = raw.groupby('Subject').cumcount()
        raw['visit_id'] = raw['participant_id'] + '-V' + raw['visit_index'].astype(str)
        
        feature_map = {
            'MMSE': ('cognition', 'cognitive_score', 'points'),
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
        # Simplified outcome builder relying on baseline_days
        raw = raw.copy()
        raw['participant_id'] = 'OASIS-' + raw['Subject'].astype(str)
        raw = raw.sort_values(['Subject', 'Age'])
        raw['visit_index'] = raw.groupby('Subject').cumcount()
        
        baseline_age = raw.groupby('participant_id')['Age'].transform('min')
        raw['baseline_days'] = ((raw['Age'] - baseline_age) * 365.25).fillna(0).astype(int)
        
        def map_dx(val):
            if pd.isna(val): return 'unknown'
            v = str(val).lower()
            if 'normal' in v or 'cn' in v: return 'cognitively_unimpaired'
            if 'mci' in v: return 'mci'
            if 'ad' in v or 'dementia' in v: return 'dementia'
            return 'unknown'
            
        raw['diagnosis'] = raw.get('dx1', pd.Series(dtype=str)).apply(map_dx)
        
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
