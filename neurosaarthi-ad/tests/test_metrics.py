import warnings

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    _average_precision_fallback,
    _roc_auc_fallback,
    binary_metrics,
)


def test_binary_metrics_sklearn():
    y_true = pd.Series([0, 1, 1, 0, 1])
    y_score = pd.Series([0.1, 0.8, 0.6, 0.3, 0.9])

    result = binary_metrics(y_true, y_score)
    assert "auroc" in result
    assert "auprc" in result
    assert "brier" in result

    assert isinstance(result["auroc"], float)
    assert isinstance(result["auprc"], float)
    assert isinstance(result["brier"], float)

    assert result["auroc"] > 0.5
    assert result["auprc"] > 0.5
    assert result["brier"] < 0.25


def test_binary_metrics_fallback(monkeypatch):
    import evaluation.metrics as metrics_module

    monkeypatch.setattr(metrics_module, "roc_auc_score", None)

    y_true = pd.Series([0, 1, 1, 0, 1])
    y_score = pd.Series([0.1, 0.8, 0.6, 0.3, 0.9])

    result = binary_metrics(y_true, y_score)
    assert "auroc" in result
    assert "auprc" in result
    assert "brier" in result

    assert isinstance(result["auroc"], float)
    assert isinstance(result["auprc"], float)
    assert isinstance(result["brier"], float)

    # In this specific array, score perfectly separates 1s and 0s
    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0


def test_roc_auc_fallback_error():
    y_true = pd.Series([1, 1, 1])
    y_score = pd.Series([0.1, 0.8, 0.6])
    with pytest.raises(
        ValueError, match="AUROC requires at least one positive and one negative"
    ):
        _roc_auc_fallback(y_true, y_score)


def test_average_precision_fallback_error():
    y_true = pd.Series([0, 0, 0])
    y_score = pd.Series([0.1, 0.8, 0.6])
    with pytest.raises(ValueError, match="AUPRC requires at least one positive"):
        _average_precision_fallback(y_true, y_score)


def test_binary_metrics_perfect_scores():
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.0, 0.1, 0.9, 1.0])

    res = binary_metrics(y_true, y_score)
    assert np.isclose(res["auroc"], 1.0)
    assert np.isclose(res["auprc"], 1.0)
    assert res["brier"] < 0.01


def test_binary_metrics_sklearn_error():
    y_true = pd.Series([1, 1, 1])
    y_score = pd.Series([0.1, 0.8, 0.6])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # In scikit-learn, providing only 1 class to ROC AUC can either raise ValueError or warn and return NaN
        # depending on version / configuration.
        try:
            res = binary_metrics(y_true, y_score)
            assert np.isnan(res["auroc"])
        except ValueError:
            pass


def test_roc_auc_fallback_error_all_negatives():
    y_true = pd.Series([0, 0, 0])
    y_score = pd.Series([0.1, 0.8, 0.6])
    with pytest.raises(
        ValueError, match="AUROC requires at least one positive and one negative"
    ):
        _roc_auc_fallback(y_true, y_score)
