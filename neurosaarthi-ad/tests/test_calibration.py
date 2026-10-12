import numpy as np
import pandas as pd

from evaluation.calibration import calibration_bins


def test_calibration_bins_basic():
    y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 1, 1])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.55, 0.85])

    result = calibration_bins(y_true, y_score, n_bins=5)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["mean_predicted", "observed_rate", "n"]
    assert len(result) == 5
    assert result["n"].sum() == 10


def test_calibration_bins_drops_na():
    y_true = pd.Series([0, 1, np.nan, 1, 0])
    y_score = pd.Series([0.1, np.nan, 0.5, 0.9, 0.2])

    result = calibration_bins(y_true, y_score, n_bins=2)

    assert len(result) <= 2
    assert result["n"].sum() == 3  # 2 samples were dropped due to NaNs


def test_calibration_bins_fewer_samples_than_bins():
    y_true = pd.Series([0, 1])
    y_score = pd.Series([0.1, 0.9])

    result = calibration_bins(y_true, y_score, n_bins=10)

    assert len(result) <= 2
    assert result["n"].sum() == 2


def test_calibration_bins_identical_scores():
    y_true = pd.Series([0, 1, 0, 1, 0])
    y_score = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])

    result = calibration_bins(y_true, y_score, n_bins=5)

    # Should get 1 bin since all scores are identical
    assert len(result) == 1
    assert result.iloc[0]["n"] == 5
    assert result.iloc[0]["observed_rate"] == 0.4  # 2 ones out of 5


def test_calibration_bins_empty():
    y_true = pd.Series([], dtype=float)
    y_score = pd.Series([], dtype=float)

    result = calibration_bins(y_true, y_score, n_bins=5)
    assert len(result) == 0
    assert list(result.columns) == ["mean_predicted", "observed_rate", "n"]
