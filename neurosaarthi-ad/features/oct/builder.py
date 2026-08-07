"""OCT/OCTA feature builders with retinal biomarker indices."""
from __future__ import annotations
import numpy as np
import pandas as pd
from features.base import pivot_modality_features

def build_oct_features(features: pd.DataFrame) -> pd.DataFrame:
    wide = pivot_modality_features(features, "oct")
    wide = _add_derived_oct(wide)
    return wide

def _add_derived_oct(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if 'rnfl_um' in result.columns:
        # RNFL thinning severity
        result['rnfl_thin'] = (result['rnfl_um'] < 85).astype(float)
        result['rnfl_severe_thin'] = (result['rnfl_um'] < 75).astype(float)
    if 'vessel_density_percent' in result.columns:
        result['low_vessel_density'] = (result['vessel_density_percent'] < 44).astype(float)
    if 'gfaz_area_mm2' in result.columns:
        result['enlarged_faz'] = (result['gfaz_area_mm2'] > 0.35).astype(float)
    # Retinal composite
    retinal_flags = [c for c in result.columns if c in ('rnfl_thin', 'low_vessel_density', 'enlarged_faz')]
    if retinal_flags:
        result['retinal_risk_count'] = result[retinal_flags].sum(axis=1)
    return result
