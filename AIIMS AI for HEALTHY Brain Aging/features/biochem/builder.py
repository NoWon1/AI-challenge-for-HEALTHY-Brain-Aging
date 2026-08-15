"""Blood biochemistry feature builders with metabolic indices."""
from __future__ import annotations
import numpy as np
import pandas as pd
from features.base import pivot_modality_features

def build_biochem_features(features: pd.DataFrame) -> pd.DataFrame:
    wide = pivot_modality_features(features, "biochem")
    wide = _add_derived_biochem(wide)
    return wide

def _add_derived_biochem(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # Metabolic risk index (composite of available markers)
    components = []
    if 'hba1c_percent' in result.columns:
        components.append((result['hba1c_percent'] > 5.7).astype(float))
        result['prediabetic_flag'] = (result['hba1c_percent'] >= 5.7).astype(float)
    if 'fasting_glucose_mg_dl' in result.columns:
        components.append((result['fasting_glucose_mg_dl'] > 100).astype(float))
    if 'total_cholesterol_mg_dl' in result.columns:
        components.append((result['total_cholesterol_mg_dl'] > 240).astype(float))
        result['high_cholesterol_flag'] = (result['total_cholesterol_mg_dl'] > 240).astype(float)
    if components:
        result['metabolic_risk_count'] = sum(components)
    # Inflammation index
    if 'hs_crp_mg_l' in result.columns:
        result['log_crp'] = np.log1p(result['hs_crp_mg_l'])
        result['high_inflammation'] = (result['hs_crp_mg_l'] > 3.0).astype(float)
    return result
