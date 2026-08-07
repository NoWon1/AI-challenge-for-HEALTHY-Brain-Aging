import numpy as np
import pandas as pd
import pytest

from models.classification.lightgbm_risk import GBMRiskClassifier

def _make_synthetic(n=50, seed=99):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'f1': rng.normal(0, 1, n),
        'f2': rng.normal(0, 1, n),
        'f3': rng.normal(0, 1, n),
        'event': rng.integers(0, 2, n),
        'event_time_days': rng.integers(100, 2000, n),
    })

def test_gbm_fit_and_predict_bounded():
    df = _make_synthetic()
    model = GBMRiskClassifier(feature_columns=['f1', 'f2', 'f3'])
    model.fit(df)
    preds = model.predict_risk(df)
    assert (preds >= 0).all()
    assert (preds <= 1).all()

def test_gbm_feature_importance_keys_match():
    df = _make_synthetic()
    model = GBMRiskClassifier(feature_columns=['f1', 'f2', 'f3'])
    model.fit(df)
    imp = model.feature_importance()
    assert set(imp.keys()) == {'f1', 'f2', 'f3'}

def test_gbm_handles_nan_gracefully():
    df = _make_synthetic()
    df.loc[0, 'f1'] = np.nan
    model = GBMRiskClassifier(feature_columns=['f1', 'f2', 'f3'])
    model.fit(df)
    preds = model.predict_risk(df)
    assert not preds.isna().any()
