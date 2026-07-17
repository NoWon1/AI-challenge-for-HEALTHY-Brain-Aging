import pandas as pd

from harmonization.leakage import assert_no_future_features
from harmonization.train_only import TrainOnlyStandardizer


def test_standardizer_uses_training_statistics():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [100.0]})
    scaler = TrainOnlyStandardizer(columns=["x"]).fit(train)
    transformed = scaler.transform(test)
    assert round(float(transformed.loc[0, "x"]), 3) == 120.025


def test_future_feature_guard_raises():
    frame = pd.DataFrame({"anchor_days": [100], "feature_days": [101]})
    try:
        assert_no_future_features(frame)
    except ValueError as exc:
        assert "after the prediction anchor" in str(exc)
    else:
        raise AssertionError("Expected leakage guard to raise")

