"""NACC cohort ETL adapter.

Parses NACC UDS (Uniform Data Set) files into the NeuroSaarthi-AD
common data model. Expected files: NACC investigator CSV.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from etl.base import CohortAdapter, CohortTables

_NACC_DX_MAP = {
    1: 'cognitively_unimpaired',  # Normal
    2: 'cognitively_unimpaired',  # Impaired not MCI
    3: 'mci',                     # MCI
    4: 'dementia',                # Dementia
}

@dataclass
class NaccAdapter(CohortAdapter):
    data_dir: str | Path = 'data/raw/nacc'
    cohort_name: str = 'NACC'
    
    def extract(self) -> CohortTables:
        root = Path(self.data_dir)
        uds_path = root / 'investigator_nacc.csv'
        
        if not uds_path.exists():
            raise FileNotFoundError(
                f'NACC UDS data not found at {uds_path}. '
                'Please place the standard NACC investigator file here.'
            )
            
        raw = pd.read_csv(uds_path, low_memory=False)
        
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
        baseline = raw.drop_duplicates('NACCID', keep='first').copy()
        sex_map = {1: 'Male', 2: 'Female'}
        
        return pd.DataFrame({
            'participant_id': 'NACC-' + baseline['NACCID'].astype(str),
            'cohort': self.cohort_name,
            'sex': baseline.get('SEX', pd.Series(dtype=float)).map(sex_map).fillna('Unknown'),
            'birth_year': baseline.get('BIRTHYR', pd.Series(dtype='Int64')).astype('Int64'),
            'education_years': pd.to_numeric(baseline.get('EDUC', np.nan), errors='coerce').fillna(0),
            'language': 'cohort_recorded',
            'urban_rural': 'reference',
        })

    def _build_visits(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw['participant_id'] = 'NACC-' + raw['NACCID'].astype(str)
        raw['visit_index'] = pd.to_numeric(raw.get('VISITNUM', 0), errors='coerce').fillna(0).astype(int)
        
        # Dates
        if all(col in raw.columns for col in ['VISITYR', 'VISITMO', 'VISITDAY']):
            raw['exam_date'] = pd.to_datetime(
                raw[['VISITYR', 'VISITMO', 'VISITDAY']].rename(
                    columns={'VISITYR': 'year', 'VISITMO': 'month', 'VISITDAY': 'day'}
                ), errors='coerce'
            )
        else:
            raw['exam_date'] = pd.NaT
            
        baseline_dates = raw.groupby('participant_id')['exam_date'].transform('min')
        raw['baseline_days'] = (raw['exam_date'] - baseline_dates).dt.days.fillna(0).astype(int)
        
        raw['diagnosis'] = raw.get('NACCUDSD', pd.Series(dtype=float)).map(_NACC_DX_MAP).fillna('unknown')
        raw['cdr_global'] = pd.to_numeric(raw.get('CDRGLOB', np.nan), errors='coerce').fillna(0.0) / 3.0
        
        visits = pd.DataFrame({
            'visit_id': raw['participant_id'] + '-V' + raw['visit_index'].astype(str),
            'participant_id': raw['participant_id'],
            'cohort': self.cohort_name,
            'visit_index': raw['visit_index'],
            'age_at_visit': pd.to_numeric(raw.get('NACCAGE', 70.0), errors='coerce'),
            'visit_date': raw['exam_date'],
            'baseline_days': raw['baseline_days'],
            'diagnosis': raw['diagnosis'],
            'cdr_global': raw['cdr_global'],
            'cognitive_status': raw['diagnosis'],
        })
        return visits.sort_values(['participant_id', 'visit_index']).reset_index(drop=True)

    def _build_features(self, raw: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw['participant_id'] = 'NACC-' + raw['NACCID'].astype(str)
        raw['visit_index'] = pd.to_numeric(raw.get('VISITNUM', 0), errors='coerce').fillna(0).astype(int)
        raw['visit_id'] = raw['participant_id'] + '-V' + raw['visit_index'].astype(str)
        
        # NACCAPOE mapping to E4 count
        apoe_map = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 0}
        if 'NACCAPOE' in raw.columns:
            raw['apoe_e4_count'] = pd.to_numeric(raw['NACCAPOE'], errors='coerce').map(apoe_map)
        
        feature_map = {
            'NACCMMSE': ('cognition', 'cognitive_score', 'points'),
            'MOCATOTS': ('cognition', 'moca_score', 'points'),
            'apoe_e4_count': ('genomics', 'apoe_e4_count', 'alleles'),
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
        raw['participant_id'] = 'NACC-' + raw['NACCID'].astype(str)
        if all(col in raw.columns for col in ['VISITYR', 'VISITMO', 'VISITDAY']):
            raw['exam_date'] = pd.to_datetime(
                raw[['VISITYR', 'VISITMO', 'VISITDAY']].rename(
                    columns={'VISITYR': 'year', 'VISITMO': 'month', 'VISITDAY': 'day'}
                ), errors='coerce'
            )
        else:
            raw['exam_date'] = pd.NaT
        raw['diagnosis'] = raw.get('NACCUDSD', pd.Series(dtype=float)).map(_NACC_DX_MAP).fillna('unknown')
        
        rows = []
        for pid, group in raw.groupby('participant_id'):
            group = group.sort_values('exam_date')
            baseline_dx = group.iloc[0]['diagnosis'] if len(group) > 0 else 'unknown'
            baseline_date = group.iloc[0]['exam_date']
            baseline_visit = f"{pid}-V0"
            
            conversion_date = None
            for _, visit in group.iterrows():
                if baseline_dx == 'cognitively_unimpaired' and visit['diagnosis'] in ('mci', 'dementia'):
                    conversion_date = visit['exam_date']
                    break
                elif baseline_dx == 'mci' and visit['diagnosis'] == 'dementia':
                    conversion_date = visit['exam_date']
                    break
            
            last_date = group.iloc[-1]['exam_date']
            follow_up_days = (last_date - baseline_date).days if pd.notna(last_date) and pd.notna(baseline_date) else 0
            
            if conversion_date is not None and pd.notna(baseline_date):
                event_time_days = (conversion_date - baseline_date).days
            else:
                event_time_days = max(follow_up_days, 1)
            
            for horizon in (1, 3, 5):
                horizon_days = int(round(horizon * 365.25))
                event = 1 if (conversion_date is not None and event_time_days <= horizon_days) else 0
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
