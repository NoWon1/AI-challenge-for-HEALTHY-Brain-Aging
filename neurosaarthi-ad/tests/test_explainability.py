import numpy as np
import pandas as pd
import pytest

from evaluation.explainability import SHAPExplainer

class MockModel:
    def __init__(self):
        self.feature_importances_ = np.array([0.5, 0.3, 0.2])
    def predict(self, X):
        return np.sum(X, axis=1)

def _make_synthetic():
    return pd.DataFrame({
        'f1': np.random.normal(0, 1, 10),
        'f2': np.random.normal(0, 1, 10),
        'f3': np.random.normal(0, 1, 10),
    })

def test_shap_explain_returns_dataframe():
    df = _make_synthetic()
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=42)
    model.fit(df, np.random.rand(10))
    explainer = SHAPExplainer(model=model, feature_names=['f1', 'f2', 'f3'], model_type='tree')
    
    # Normally SHAP needs real explainer setup, but if shap is not present, it uses fallback
    # If it is present, tree explainer might fail on a mock model without trees, but we can just mock it or assume it falls back
    # or handle the dataframe return
    res = explainer.explain(df)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == df.shape

def test_shap_top_drivers_shape():
    df = _make_synthetic()
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=42)
    model.fit(df, np.random.rand(10))
    explainer = SHAPExplainer(model=model, feature_names=['f1', 'f2', 'f3'], model_type='tree')
    
    top = explainer.top_drivers(df, k=2)
    assert len(top) == len(df)
    assert 'feature_1' in top.columns
    assert 'feature_2' in top.columns

def test_shap_modality_groups():
    df = _make_synthetic()
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=42)
    model.fit(df, np.random.rand(10))
    explainer = SHAPExplainer(model=model, feature_names=['f1', 'f2', 'f3'], model_type='tree')
    
    modality_map = {
        'mod_a': ['f1', 'f2'],
        'mod_b': ['f3']
    }
    res = explainer.explain_modality_groups(df, modality_map)
    assert 'mod_a' in res.columns
    assert 'mod_b' in res.columns
    assert len(res) == len(df)
