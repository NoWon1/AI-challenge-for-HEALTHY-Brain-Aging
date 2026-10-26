import numpy as np
import pandas as pd
import pytest

from models.progression.baseline import CognitiveTrajectoryRegressor


def _make_synthetic(n=50, seed=99):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "f3": rng.normal(0, 1, n),
            "future_score": rng.normal(27, 2, n),
        }
    )


def test_cognitive_trajectory_regressor_fit_predict():
    df = _make_synthetic()
    # Add a NaN to ensure imputation works
    df.loc[0, "f1"] = np.nan

    model = CognitiveTrajectoryRegressor(feature_columns=["f1", "f2", "f3"])
    model.fit(df)

    preds = model.predict(df)

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(df)
    assert preds.name == "predicted_future_score"
    assert not preds.isna().any()


def test_cognitive_trajectory_regressor_fallback(monkeypatch):
    from models.progression import baseline

    monkeypatch.setattr(baseline, "Pipeline", None)

    df = _make_synthetic()
    df.loc[0, "f1"] = np.nan

    model = CognitiveTrajectoryRegressor(feature_columns=["f1", "f2", "f3"])

    # Verify we are using the fallback by checking pipeline is None
    assert model.pipeline is None

    model.fit(df)

    # Verify fallback properties were set
    assert hasattr(model, "coef_")

    preds = model.predict(df)

    assert isinstance(preds, pd.Series)
    assert len(preds) == len(df)
    assert preds.name == "predicted_future_score"
    assert not preds.isna().any()


def test_cognitive_trajectory_regressor_predict_before_fit_fallback(monkeypatch):
    from models.progression import baseline

    monkeypatch.setattr(baseline, "Pipeline", None)

    df = _make_synthetic()
    model = CognitiveTrajectoryRegressor(feature_columns=["f1", "f2"])

    with pytest.raises(RuntimeError, match="must be fitted before predict"):
        model.predict(df)
