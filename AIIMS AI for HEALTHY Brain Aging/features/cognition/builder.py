"""Cognitive feature builders with composite scores and reserve proxies."""
from __future__ import annotations
import numpy as np
import pandas as pd
from features.base import pivot_modality_features

def build_cognition_features(features: pd.DataFrame) -> pd.DataFrame:
    wide = pivot_modality_features(features, "cognition")
    wide = _add_derived_cognition(wide)
    return wide

def _add_derived_cognition(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # Domain composite: average of available z-scores
    z_cols = [c for c in result.columns if c.endswith('_score') and c != 'cognitive_score']
    if z_cols:
        result['domain_composite_z'] = result[z_cols].mean(axis=1)
    # Cognitive reserve proxy (if education available from a merge)
    # This is a placeholder - will be populated when merged with participant data
    if 'cognitive_score' in result.columns:
        result['cognitive_impairment_flag'] = (result['cognitive_score'] < 24.0).astype(float)
        result['mild_impairment_flag'] = ((result['cognitive_score'] >= 24.0) & (result['cognitive_score'] < 27.0)).astype(float)
    return result
