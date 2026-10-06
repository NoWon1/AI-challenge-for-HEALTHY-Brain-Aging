import numpy as np
import pandas as pd
import pytest

from models.utils.validation import validate_feature_matrix


def test_validate_feature_matrix_happy_path():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    res = validate_feature_matrix(df, allow_nan=False)
    pd.testing.assert_frame_equal(df, res)

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    res_arr = validate_feature_matrix(arr, allow_nan=False)
    np.testing.assert_array_equal(arr, res_arr)


def test_validate_feature_matrix_inf_raises_value_error():
    df = pd.DataFrame({"a": [1.0, np.inf], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="Input feature matrix contains non-finite values \\(NaN or Inf\\)."):
        validate_feature_matrix(df, allow_nan=True)

    arr = np.array([[1.0, -np.inf], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Input feature matrix contains non-finite values \\(NaN or Inf\\)."):
        validate_feature_matrix(arr, allow_nan=True)


def test_validate_feature_matrix_nan_behavior():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0]})
    # Should raise if allow_nan is False
    with pytest.raises(ValueError, match="allow_nan=False"):
        validate_feature_matrix(df, allow_nan=False)
    # Should not raise if allow_nan is True
    res = validate_feature_matrix(df, allow_nan=True)
    pd.testing.assert_frame_equal(df, res)


def test_validate_feature_matrix_invalid_type_raises_type_error():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["not", "a number"]})
    with pytest.raises(TypeError, match="Feature matrix must contain numeric types."):
        validate_feature_matrix(df, allow_nan=True)

    arr = np.array([["str", 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError, match="Feature matrix must contain numeric types."):
        validate_feature_matrix(arr, allow_nan=True)


def test_validate_feature_matrix_none_raises_value_error():
    with pytest.raises(ValueError, match="Input feature matrix cannot be None."):
        validate_feature_matrix(None)

def test_integration_gradient_boosting():
    from models.survival.gradient_boosting import GradientBoostingSurvivalModel
    model = GradientBoostingSurvivalModel(["age", "b"])
    model.model_ = "mock"
    model.feature_columns_ = model.feature_columns
    df = pd.DataFrame({"age": [10.0, np.inf], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.predict_risk_scores(df)

def test_integration_rsf():
    from models.survival.rsf import RandomSurvivalForestModel
    model = RandomSurvivalForestModel(["f1"])
    model.model = "mock"
    df = pd.DataFrame({"f1": [np.nan, 1.0]})
    with pytest.raises(ValueError, match="allow_nan=False"):
        model.predict_risk_scores(df)

def test_integration_cox_boost():
    from models.survival.cox_boost import CoxBoostModel
    model = CoxBoostModel(["f1"])
    model._model = "mock"
    df = pd.DataFrame({"f1": [np.inf, 1.0]})
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.predict_risk_scores(df)

def test_integration_mixed_effects():
    from models.progression.mixed_effects import MixedEffectsTrajectory
    model = MixedEffectsTrajectory(["f1"])
    model.global_model_ = "mock"
    model._is_statsmodels = False
    df = pd.DataFrame({"f1": [np.inf, 1.0]})
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.predict(df)

def test_integration_lgbm():
    from models.classification.lightgbm_risk import GBMRiskClassifier
    model = GBMRiskClassifier(["f1"])
    model.model = "mock"
    df = pd.DataFrame({"f1": [np.inf, 1.0]})
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.predict_risk(df)
