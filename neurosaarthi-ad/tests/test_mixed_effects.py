import numpy as np
import pandas as pd
import pytest

from models.progression.mixed_effects import MixedEffectsTrajectory

def _make_longitudinal(n=30, seed=99):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        baseline = rng.normal(27, 2)
        for t in [0, 1, 2, 3]:
            rows.append({
                'participant_id': f'P{i}',
                'f1': rng.normal(0, 1),
                'horizon_years': float(t),
                'future_score': baseline - 0.5 * t + rng.normal(0, 0.5),
            })
    return pd.DataFrame(rows)

def test_mixed_effects_fit_predict():
    df = _make_longitudinal()
    model = MixedEffectsTrajectory(feature_columns=['f1'])
    model.fit(df)
    preds = model.predict(df)
    assert len(preds) == len(df)
    assert not preds.isna().any()

def test_mixed_effects_intervals():
    df = _make_longitudinal()
    model = MixedEffectsTrajectory(feature_columns=['f1'])
    model.fit(df)
    
    # Mock residuals
    res = np.random.normal(0, 1, len(df))
    model.calibrate_conformal(res)
    
    preds, lower, upper = model.predict_with_intervals(df)
    assert (lower <= preds).all()
    assert (preds <= upper).all()

def test_conformal_calibration():
    df = _make_longitudinal()
    model = MixedEffectsTrajectory(feature_columns=['f1'])
    res = np.array([-1, 0.5, 2, -3, 1.5])
    model.calibrate_conformal(res, alpha=0.1)
    assert model.conformal_q_ > 0
