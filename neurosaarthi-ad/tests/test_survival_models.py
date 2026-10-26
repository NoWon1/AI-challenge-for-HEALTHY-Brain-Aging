import numpy as np
import pandas as pd
import pytest

from models.survival.rsf import RandomSurvivalForestModel

try:
    from models.survival.cox_boost import CoxBoostModel
except ImportError:
    CoxBoostModel = None


def _make_synthetic(n=50, seed=99):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "event": rng.integers(0, 2, n),
            "event_time_days": rng.integers(100, 2000, n),
        }
    )


def test_rsf_fit_predict_risk_scores():
    df = _make_synthetic()
    model = RandomSurvivalForestModel(feature_columns=["f1", "f2"])
    model.fit(df)
    scores = model.predict_risk_scores(df)
    assert len(scores) == len(df)


def test_rsf_survival_function_shape():
    df = _make_synthetic()
    model = RandomSurvivalForestModel(feature_columns=["f1", "f2"])
    model.fit(df)
    times = np.array([500, 1000])
    sf = model.predict_survival_function(df, time_points=times)
    assert sf.shape == (len(df), 2)


@pytest.mark.skipif(CoxBoostModel is None, reason="CoxBoostModel not available")
def test_cox_boost_fit_and_predict():
    df = _make_synthetic()
    model = CoxBoostModel(feature_columns=["f1", "f2"])
    model.fit(df)
    scores = model.predict_risk_scores(df)
    assert len(scores) == len(df)


@pytest.mark.skipif(CoxBoostModel is None, reason="CoxBoostModel not available")
def test_cox_boost_concordance():
    df = _make_synthetic()
    model = CoxBoostModel(feature_columns=["f1", "f2"])
    model.fit(df)
    c_index = model.concordance_index(df)
    assert 0.0 <= c_index <= 1.0
