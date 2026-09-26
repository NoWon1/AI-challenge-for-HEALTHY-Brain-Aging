import numpy as np
import pandas as pd
import pytest
import unittest.mock as mock
import warnings

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

def test_gbm_shap_values():
    df = _make_synthetic()
    model = GBMRiskClassifier(feature_columns=['f1', 'f2', 'f3'])
    model.fit(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_vals = model.shap_values(df)

    assert shap_vals is not None
    assert shap_vals.shape == (len(df), 3)

def test_gbm_shap_values_unavailable():
    df = _make_synthetic()
    model = GBMRiskClassifier(feature_columns=['f1', 'f2', 'f3'])
    model.fit(df)

    with mock.patch('models.classification.lightgbm_risk.SHAP_AVAILABLE', False):
        with pytest.warns(UserWarning, match="shap library is not installed"):
            shap_vals = model.shap_values(df)
            assert shap_vals is None

def test_gbm_shap_values_not_fitted():
    df = _make_synthetic()
    model = GBMRiskClassifier(feature_columns=['f1', 'f2', 'f3'])

    with pytest.raises(RuntimeError, match="Classifier must be fitted before getting SHAP values"):
        model.shap_values(df)

def test_gbm_shap_values_calibrated():
    df = _make_synthetic()
    model = GBMRiskClassifier(feature_columns=['f1', 'f2', 'f3'], calibrate=True)
    model.fit(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_vals = model.shap_values(df)

    assert shap_vals is not None
    assert shap_vals.shape == (len(df), 3)
