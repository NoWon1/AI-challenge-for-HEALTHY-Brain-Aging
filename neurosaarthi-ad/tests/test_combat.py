import numpy as np
import pandas as pd
import pytest

from harmonization.combat import TrainOnlyComBat

def _make_synthetic():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        'cohort': ['A']*50 + ['B']*50,
        'f1': np.concatenate([rng.normal(0, 1, 50), rng.normal(2, 1, 50)]),
        'f2': np.concatenate([rng.normal(0, 1, 50), rng.normal(-2, 1, 50)]),
        'age': rng.normal(65, 5, 100),
        'sex_binary': rng.integers(0, 2, 100),
    })

def test_combat_reduces_batch_effect():
    df = _make_synthetic()
    combat = TrainOnlyComBat(batch_col='cohort', feature_columns=['f1', 'f2'], preserve_columns=['age', 'sex_binary'])
    
    harmonized = combat.fit_transform(df)
    
    mean_a = harmonized.loc[harmonized['cohort'] == 'A', 'f1'].mean()
    mean_b = harmonized.loc[harmonized['cohort'] == 'B', 'f1'].mean()
    
    # The batch effect was 2.0, should be close to 0 now
    assert abs(mean_a - mean_b) < 0.5

def test_combat_preserves_biology():
    df = _make_synthetic()
    # Create age correlation
    df['f1'] += df['age'] * 0.1
    
    combat = TrainOnlyComBat(batch_col='cohort', feature_columns=['f1', 'f2'], preserve_columns=['age', 'sex_binary'])
    harmonized = combat.fit_transform(df)
    
    # Age correlation should still exist
    corr_orig = df['f1'].corr(df['age'])
    corr_harm = harmonized['f1'].corr(harmonized['age'])
    
    assert corr_harm > 0.5 * corr_orig

def test_combat_unseen_batch():
    df = _make_synthetic()
    combat = TrainOnlyComBat(batch_col='cohort', feature_columns=['f1', 'f2'], preserve_columns=['age', 'sex_binary'])
    combat.fit(df)
    
    df_new = pd.DataFrame({
        'cohort': ['C']*10,
        'f1': np.random.normal(5, 1, 10),
        'f2': np.random.normal(5, 1, 10),
        'age': np.random.normal(65, 5, 10),
        'sex_binary': np.random.randint(0, 2, 10),
    })
    
    harmonized = combat.transform(df_new)
    assert not harmonized.isna().any().any()
