import numpy as np
import pandas as pd
import pytest

from evaluation.comparison import bootstrap_comparison
from sklearn.metrics import roc_auc_score

def test_bootstrap_identical_models():
    """Test that two identical models have delta 0 and p-value 1.0."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 10)
    y_score = np.random.uniform(0, 1, len(y_true))

    result = bootstrap_comparison(
        y_true, y_score, y_score, metric_fn=roc_auc_score, n_bootstrap=100, seed=42
    )

    assert abs(result["delta"]) < 1e-6
    assert abs(result["p_value"] - 1.0) < 1e-6
    assert abs(result["ci_lower"]) < 1e-6
    assert abs(result["ci_upper"]) < 1e-6

def test_bootstrap_model_a_better():
    """Test when Model A is strictly better than Model B."""
    y_true = np.array([0, 0, 0, 1, 1, 1] * 20)
    # Model A is perfect
    y_score_a = y_true.copy()
    # Model B is random
    np.random.seed(42)
    y_score_b = np.random.uniform(0, 1, len(y_true))

    result = bootstrap_comparison(
        y_true, y_score_a, y_score_b, metric_fn=roc_auc_score, n_bootstrap=100, seed=42
    )

    assert result["delta"] > 0
    assert result["p_value"] < 0.05
    assert result["ci_lower"] > 0
    assert result["ci_upper"] > 0

def test_bootstrap_model_b_better():
    """Test when Model B is strictly better than Model A."""
    y_true = np.array([0, 0, 0, 1, 1, 1] * 20)
    # Model A is random
    np.random.seed(42)
    y_score_a = np.random.uniform(0, 1, len(y_true))
    # Model B is perfect
    y_score_b = y_true.copy()

    result = bootstrap_comparison(
        y_true, y_score_a, y_score_b, metric_fn=roc_auc_score, n_bootstrap=100, seed=42
    )

    assert result["delta"] < 0
    assert result["p_value"] < 0.05
    assert result["ci_upper"] < 0
    assert result["ci_lower"] < 0

def test_bootstrap_pandas_series():
    """Test that passing pandas Series works identically to numpy arrays."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 10)
    y_score_a = np.random.uniform(0, 1, len(y_true))
    y_score_b = np.random.uniform(0, 1, len(y_true))

    res_np = bootstrap_comparison(
        y_true, y_score_a, y_score_b, metric_fn=roc_auc_score, n_bootstrap=100, seed=42
    )

    res_pd = bootstrap_comparison(
        pd.Series(y_true), pd.Series(y_score_a), pd.Series(y_score_b), metric_fn=roc_auc_score, n_bootstrap=100, seed=42
    )

    assert res_np["delta"] == pytest.approx(res_pd["delta"])
    assert res_np["p_value"] == pytest.approx(res_pd["p_value"])
    assert res_np["ci_lower"] == pytest.approx(res_pd["ci_lower"])
    assert res_np["ci_upper"] == pytest.approx(res_pd["ci_upper"])


def test_select_best_model():
    from evaluation.comparison import select_best_model, ModelComparisonReport

    reports = [
        ModelComparisonReport(model_name="ModelA", metrics_dict={"auroc": 0.8}, cohort="external"),
        ModelComparisonReport(model_name="ModelB", metrics_dict={"auroc": 0.85}, cohort="external"),
        ModelComparisonReport(model_name="ModelC", metrics_dict={"auroc": 0.9}, cohort="internal")
    ]

    assert select_best_model(reports, metric="auroc", validation_set="external") == "ModelB"
    assert select_best_model(reports, metric="auroc", validation_set="internal") == "ModelC"
    assert select_best_model(reports, metric="f1_score", validation_set="external") is None
    assert select_best_model(reports, metric="auroc", validation_set="unknown_cohort") is None

def test_compare_models():
    from evaluation.comparison import compare_models, ModelComparisonReport

    reports = [
        ModelComparisonReport(model_name="ModelA", metrics_dict={"auroc": 0.8, "f1": 0.7}, cohort="external"),
        ModelComparisonReport(model_name="ModelB", metrics_dict={"auroc": 0.85, "f1": 0.75}, cohort="external"),
    ]

    df = compare_models(reports)
    assert len(df) == 2
    assert list(df.columns) == ["model_name", "cohort", "auroc", "f1"]
    assert df.iloc[0]["model_name"] == "ModelA"
    assert df.iloc[1]["auroc"] == 0.85
