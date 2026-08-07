import numpy as np
import pandas as pd
import pytest

from evaluation.uncertainty import BootstrapCI, ConformalPredictor, ReliabilityDiagram, ood_score
from evaluation.comparison import delong_test

def test_bootstrap_ci_contains_point():
    ci = BootstrapCI(n_bootstrap=50, seed=42)
    data = np.random.normal(0, 1, 100)
    def mean_fn(x): return float(np.mean(x))
    
    point, lower, upper = ci.compute(mean_fn, data)
    assert lower <= point <= upper

def test_conformal_coverage():
    cp = ConformalPredictor()
    residuals = np.random.normal(0, 1, 1000)
    width = cp.calibrate(residuals, alpha=0.1)
    
    preds = np.zeros(1000)
    y_true = residuals
    
    lower, upper = cp.predict_interval(preds, width)
    coverage = cp.coverage(y_true, lower, upper)
    
    assert abs(coverage - 0.9) < 0.05

def test_reliability_diagram_bins():
    rd = ReliabilityDiagram()
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.uniform(0, 1, 100)
    
    df = rd.compute(y_true, y_pred, n_bins=5)
    assert len(df) == 5
    assert 'ece' in df.columns

def test_ood_score_self_is_zero():
    train = np.random.normal(0, 1, (100, 5))
    scores = ood_score(train[:5], train)
    assert (scores < 3).all()  # Mahalanobis distance should be small for in-distribution

def test_delong_identical_scores():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])
    
    pval = delong_test(y_true, y_score, y_score)
    assert abs(pval - 1.0) < 1e-5
