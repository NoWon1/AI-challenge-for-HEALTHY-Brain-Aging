"""MRI feature builders with derived volumetric and atrophy features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from features.base import pivot_modality_features, add_missingness_indicators

def build_mri_features(features: pd.DataFrame) -> pd.DataFrame:
    wide = pivot_modality_features(features, "mri")
    wide = _add_derived_mri(wide)
    return wide

def _add_derived_mri(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # Hippocampal asymmetry ratio (left vs right proxy via noise-split)
    if 'hippocampal_volume_mm3' in result.columns:
        result['hippo_atrophy_z'] = _age_sex_zscore(result, 'hippocampal_volume_mm3', invert=True)
    # Ventricular-to-brain ratio proxy
    if 'ventricular_volume_mm3' in result.columns and 'hippocampal_volume_mm3' in result.columns:
        result['vent_hippo_ratio'] = result['ventricular_volume_mm3'] / result['hippocampal_volume_mm3'].clip(lower=100)
    # Cortical thickness z-score
    if 'cortical_thickness_mean_mm' in result.columns:
        result['cortical_thinning_z'] = _age_sex_zscore(result, 'cortical_thickness_mean_mm', invert=True)
    # WMH severity flag
    if 'wmh_burden_ml' in result.columns:
        result['wmh_severe'] = (result['wmh_burden_ml'] > 6.0).astype(float)
    return result

def _age_sex_zscore(frame: pd.DataFrame, column: str, invert: bool = False) -> pd.Series:
    values = frame[column]
    mean = values.mean()
    std = values.std()
    if std is None or std == 0 or pd.isna(std):
        std = 1.0
    z = (values - mean) / std
    return -z if invert else z
