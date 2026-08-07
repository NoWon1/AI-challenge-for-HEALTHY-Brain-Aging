"""ADNI cohort ETL adapter.

Parses ADNIMERGE.csv and related tables from the Alzheimer's Disease
Neuroimaging Initiative into the NeuroSaarthi-AD common data model.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from etl.base import CohortAdapter, CohortTables

_DX_MAP = {
    'CN': 'cognitively_unimpaired',
    'SMC': 'cognitively_unimpaired',
    'EMCI': 'mci',
    'LMCI': 'mci', 
    'MCI': 'mci',
    'AD': 'dementia',
    'Dementia': 'dementia',
}

@dataclass
class AdniAdapter(CohortAdapter):
    data_dir: str | Path = 'data/raw/adni'
    cohort_name: str = 'ADNI'
    
    def extract(self) -> CohortTables:
        root = Path(self.data_dir)
        merge_path = root / 'ADNIMERGE.csv'
        
        if not merge_path.exists():
            raise FileNotFoundError(
                f'ADNIMERGE.csv not found at {merge_path}. '
                'Download from https://adni.loni.usc.edu/ and place in data/raw/adni/'
            )
        
        raw = pd.read_csv(merge_path, low_memory=False)
        
        # Build participants table
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
        baseline = raw.drop_duplicates('RID', keep='first')
        return pd.DataFrame({
            'participant_id': 'ADNI-' + baseline['RID'].astype(str).str.zfill(4),
            'cohort': self.cohort_name,
            'sex': baseline['PTGENDER'].map({'Male': 'Male', 'Female': 'Female'}).fillna('Unknown'),
            'birth_year': (baseline['EXAMDATE'].apply(lambda x: pd.to_datetime(x, errors='coerce')).dt.year - baseline['AGE'].fillna(70)).astype('Int64'),
            'education_years': baseline['PTEDUCAT'].fillna(0).astype(float),
            'language': 'cohort_recorded',
            'urban_rural': 'reference',
        })
    
    def _build_visits(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw['participant_id'] = 'ADNI-' + raw['RID'].astype(str).str.zfill(4)
        raw['exam_date'] = pd.to_datetime(raw['EXAMDATE'], errors='coerce')
        
        # Calculate visit index from VISCODE
        viscode_order = {'bl': 0, 'sc': 0}
        for m in range(6, 180, 6):
            viscode_order[f'm{m:02d}'] = m // 6
            viscode_order[f'm{m}'] = m // 6
        
        raw['visit_index'] = raw['VISCODE'].str.lower().map(viscode_order).fillna(0).astype(int)
        
        # Baseline days
        baseline_dates = raw.groupby('participant_id')['exam_date'].transform('min')
        raw['baseline_days'] = (raw['exam_date'] - baseline_dates).dt.days.fillna(0).astype(int)
        
        raw['diagnosis'] = raw['DX'].map(_DX_MAP).fillna('unknown')
        raw['cdr_global'] = raw.get('CDRSB', pd.Series(dtype=float)).fillna(0.0) / 18.0  # Normalize CDR-SB
        
        visits = pd.DataFrame({
            'visit_id': raw['participant_id'] + '-V' + raw['visit_index'].astype(str),
            'participant_id': raw['participant_id'],
            'cohort': self.cohort_name,
            'visit_index': raw['visit_index'],
            'age_at_visit': raw['AGE'].fillna(70.0),
            'visit_date': raw['exam_date'],
            'baseline_days': raw['baseline_days'],
            'diagnosis': raw['diagnosis'],
            'cdr_global': raw['cdr_global'],
            'cognitive_status': raw['diagnosis'],
        })
        return visits.sort_values(['participant_id', 'visit_index']).reset_index(drop=True)
    
    def _build_features(self, raw: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw['participant_id'] = 'ADNI-' + raw['RID'].astype(str).str.zfill(4)
        raw['visit_index'] = raw['VISCODE'].str.lower().map(
            {**{'bl': 0, 'sc': 0}, **{f'm{m:02d}': m//6 for m in range(6,180,6)}, **{f'm{m}': m//6 for m in range(6,180,6)}}
        ).fillna(0).astype(int)
        raw['visit_id'] = raw['participant_id'] + '-V' + raw['visit_index'].astype(str)
        
        feature_map = {
            # Cognition
            'MMSE': ('cognition', 'cognitive_score', 'points'),
            'RAVLT_immediate': ('cognition', 'memory_score', 'z-score'),
            # MRI
            'Hippocampus': ('mri', 'hippocampal_volume_mm3', 'mm3'),
            'Ventricles': ('mri', 'ventricular_volume_mm3', 'mm3'),
            'WholeBrain': ('mri', 'whole_brain_volume_mm3', 'mm3'),
            'Entorhinal': ('mri', 'entorhinal_thickness_mm', 'mm'),
            'MidTemp': ('mri', 'midtemp_thickness_mm', 'mm'),
            # Genomics
            'APOE4': ('genomics', 'apoe_e4_count', 'alleles'),
        }
        
        rows = []
        for _, record in raw.iterrows():
            for source_col, (modality, feature_name, unit) in feature_map.items():
                value = record.get(source_col)
                if pd.notna(value):
                    rows.append({
                        'feature_row_id': f"{record['visit_id']}-{feature_name}",
                        'participant_id': record['participant_id'],
                        'visit_id': record['visit_id'],
                        'cohort': self.cohort_name,
                        'modality': modality,
                        'feature_name': feature_name,
                        'value': float(value),
                        'unit': unit,
                        'source_variable': source_col,
                        'qc_flag': 'pass',
                        'derived': False,
                    })
        return pd.DataFrame(rows).sort_values(['participant_id', 'visit_id']).reset_index(drop=True)
    
