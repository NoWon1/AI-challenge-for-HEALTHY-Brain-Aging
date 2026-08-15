"""Genomics feature builders with APOE stratification and ancestry-aware processing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from features.base import pivot_modality_features

def build_genomics_features(features: pd.DataFrame) -> pd.DataFrame:
    wide = pivot_modality_features(features, "genomics")
    wide = _add_derived_genomics(wide)
    return wide

def _add_derived_genomics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if 'apoe_e4_count' in result.columns:
        # APOE haplotype encoding
        result['apoe_e4_carrier'] = (result['apoe_e4_count'] > 0).astype(float)
        result['apoe_e4_homozygous'] = (result['apoe_e4_count'] == 2).astype(float)
        # Simplified PRS proxy (weighted by APOE as dominant locus)
        result['ad_prs_proxy'] = result['apoe_e4_count'] * 0.47  # Log-OR weight from literature
    if 'ancestry_pc1' in result.columns:
        # Indian ancestry flag (PC1 > 0.5 suggests South Asian clustering in this synthetic setup)
        result['indian_ancestry_flag'] = (result['ancestry_pc1'] > 0.5).astype(float)
        # Ancestry-aware QC: flag if ancestry outside typical European range
        result['non_european_ancestry'] = (result['ancestry_pc1'].abs() > 0.3).astype(float)
    return result
