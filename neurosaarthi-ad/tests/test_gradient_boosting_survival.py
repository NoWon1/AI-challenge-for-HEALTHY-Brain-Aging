import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from models.survival.gradient_boosting import GradientBoostingSurvivalModel

def generate_mock_data():
    np.random.seed(42)
    # 100 samples
    n = 100
    age = np.random.uniform(50, 90, n)
    biomarker1 = np.random.normal(10, 2, n)
    # Insert NaNs in features
    biomarker1[np.random.choice(n, 10, replace=False)] = np.nan
    
    # Create somewhat meaningful survival target
    risk = age * 0.1 + np.nan_to_num(biomarker1, nan=10) * 0.5
    time = np.random.exponential(scale=1000 / risk)
    # Mixed censored (False) and uncensored (True)
    event = np.random.binomial(1, 0.7, n).astype(bool)
    
    frame = pd.DataFrame({
        "age": age,
        "biomarker1": biomarker1,
        "event_time_days": time,
        "event": event
    })
    return frame

def test_missing_sksurv(monkeypatch):
    import models.survival.gradient_boosting as gbm_module
    monkeypatch.setattr(gbm_module, "SKSURV_AVAILABLE", False)
    with pytest.raises(ImportError, match="sksurv is required"):
        GradientBoostingSurvivalModel(feature_columns=["age"])

def test_unfitted_model():
    model = GradientBoostingSurvivalModel(feature_columns=["age"])
    frame = pd.DataFrame({"age": [60]})
    with pytest.raises(NotFittedError):
        model.predict_risk_scores(frame)
    with pytest.raises(NotFittedError):
        model.predict_survival_function(frame)

def test_missing_feature_columns():
    model = GradientBoostingSurvivalModel(feature_columns=["age", "missing_col"])
    frame = pd.DataFrame({"age": [60], "event_time_days": [10], "event": [True]})
    with pytest.raises(ValueError, match="Missing feature columns"):
        model.fit(frame)

def test_full_pipeline_and_correctness():
    frame = generate_mock_data()
    features = ["age", "biomarker1"]
    
    model = GradientBoostingSurvivalModel(feature_columns=features, seed=42)
    
    # 1. Test fit with NaNs, censored/uncensored
    model.fit(frame)
    assert hasattr(model, "model_")
    assert hasattr(model, "imputer_")
    
    # 2. Test predict_risk_scores on data containing NaNs
    risk_scores = model.predict_risk_scores(frame)
    assert len(risk_scores) == len(frame)
    assert not risk_scores.isna().any()
    
    # 3. Test C-index correctness
    c_index = model.concordance_index(frame)
    assert 0.0 <= c_index <= 1.0
    
    # 4. Test permutation importance
    perm_imp = model.permutation_importance(frame, n_repeats=2)
    assert len(perm_imp) == 2
    assert "importance_mean" in perm_imp.columns
    
    # 5. Test arbitrary time_points (including extreme ones that test clipping)
    time_points = np.array([1, 10, 50, 100, 10000]) # 10000 is likely out of bounds
    surv_df = model.predict_survival_function(frame, time_points=time_points)
    
    # Shape should match N_samples x N_timepoints
    assert surv_df.shape == (len(frame), len(time_points))
    
    # Values should be within [0, 1]
    assert (surv_df >= 0.0).all().all()
    assert (surv_df <= 1.0).all().all()
    
    # Survival probabilities should be monotonic (non-increasing over time)
    diffs = surv_df.diff(axis=1).iloc[:, 1:]
    # Using 1e-6 to account for minor floating point inaccuracies
    assert (diffs <= 1e-6).all().all()
