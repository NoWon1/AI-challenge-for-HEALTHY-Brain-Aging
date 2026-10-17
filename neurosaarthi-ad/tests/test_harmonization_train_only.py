import pytest
import pandas as pd
import numpy as np

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


def test_train_only_standardizer_fit_state():
    """Verify fit method correctly populates means and stds and handles zero std."""
    # Create test data with normal variation and zero variation
    df = pd.DataFrame({
        "var_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        "zero_var_col": [42.0, 42.0, 42.0, 42.0, 42.0]
    })

    scaler = TrainOnlyStandardizer(columns=["var_col", "zero_var_col"])

    # Assert initial state
    assert scaler.fitted_ is False
    assert scaler.means_ == {}
    assert scaler.stds_ == {}

    # Fit the scaler
    scaler.fit(df)

    # Assert state after fit
    assert scaler.fitted_ is True

    # Check means
    assert "var_col" in scaler.means_
    assert "zero_var_col" in scaler.means_
    assert scaler.means_["var_col"] == 3.0
    assert scaler.means_["zero_var_col"] == 42.0

    # Check standard deviations (should use ddof=0 which is population std)
    # var_col std = sqrt(((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5) = sqrt((4+1+0+1+4)/5) = sqrt(2) approx 1.414
    assert "var_col" in scaler.stds_
    assert "zero_var_col" in scaler.stds_
    np.testing.assert_approx_equal(scaler.stds_["var_col"], np.sqrt(2.0))

    # Zero standard deviation should be replaced with 1.0
    assert scaler.stds_["zero_var_col"] == 1.0


def test_train_only_standardizer_unfitted_transform_raises():
    """Verify transform raises an error if not fitted."""
    scaler = TrainOnlyStandardizer(columns=["x"])
    df = pd.DataFrame({"x": [1.0, 2.0]})

    with pytest.raises(RuntimeError, match="must be fitted before transform"):
        scaler.transform(df)


def test_train_only_standardizer_fit_transform():
    """Verify fit_transform applies both operations sequentially."""
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})

    scaler = TrainOnlyStandardizer(columns=["x"])
    transformed = scaler.fit_transform(df)

    # Mean should be 20.0, Std (ddof=0) should be sqrt(100+0+100/3) = sqrt(200/3) approx 8.1649
    assert scaler.fitted_ is True
    assert scaler.means_["x"] == 20.0

    expected_values = [(x - 20.0) / np.sqrt(200/3) for x in [10.0, 20.0, 30.0]]
    np.testing.assert_array_almost_equal(transformed["x"].values, expected_values)
