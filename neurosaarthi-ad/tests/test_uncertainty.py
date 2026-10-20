from unittest.mock import patch

import numpy as np

from evaluation.comparison import delong_test
from evaluation.uncertainty import (
    BootstrapCI,
    ConformalPredictor,
    ReliabilityDiagram,
    ood_score,
)


def test_bootstrap_ci_contains_point():
    ci = BootstrapCI(n_bootstrap=50, seed=42)
    data = np.random.normal(0, 1, 100)

    def mean_fn(x):
        return float(np.mean(x))

    point, lower, upper = ci.compute(mean_fn, data)
    assert lower <= point <= upper


def test_conformal_coverage():
    cp = ConformalPredictor()
    rng = np.random.default_rng(42)
    # The conformal predictor calibration assumes absolute residuals
    residuals = np.abs(rng.normal(0, 1, 5000))
    width = cp.calibrate(residuals, alpha=0.1)

    preds = np.zeros(5000)
    y_true = rng.normal(0, 1, 5000)

    lower, upper = cp.predict_interval(preds, width)
    coverage = cp.coverage(y_true, lower, upper)

    assert abs(coverage - 0.9) < 0.05


def test_reliability_diagram_bins():
    rd = ReliabilityDiagram()
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.uniform(0, 1, 100)

    df = rd.compute(y_true, y_pred, n_bins=5)
    assert len(df) == 5
    assert "ece" in df.columns


def test_ood_score_self_is_zero():
    rng = np.random.default_rng(42)
    train = rng.normal(0, 1, (100, 5))
    scores = ood_score(train[:5], train)
    assert (
        scores < 3
    ).all()  # Mahalanobis distance should be small for in-distribution


def test_delong_identical_scores():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])

    pval = delong_test(y_true, y_score, y_score)
    assert abs(pval - 1.0) < 1e-5


def test_ood_score_singular_matrix_fallback():
    # Create perfectly correlated features to ensure the covariance matrix is singular
    # This should trigger np.linalg.LinAlgError and fallback to np.linalg.pinv
    rng = np.random.default_rng(42)
    feature1 = rng.normal(0, 1, 100)
    feature2 = feature1 * 2.0  # Perfectly correlated

    # Train features: shape (100, 2)
    train = np.column_stack((feature1, feature2))

    # Query features
    query = np.array(
        [
            [np.mean(feature1), np.mean(feature2)],  # Exact mean
            [10.0, 20.0],  # Far point on the manifold
        ]
    )

    # Track if pinv is called
    with patch("numpy.linalg.pinv", wraps=np.linalg.pinv) as mock_pinv:
        # This should not raise LinAlgError
        scores = ood_score(query, train)

        # Verify fallback was used
        assert mock_pinv.called

    # Check that scores are computed and have correct shape
    assert len(scores) == 2

    # Mean should have distance very close to 0
    assert abs(scores[0]) < 1e-5

    # Far point should have a high score
    assert scores[1] > 1.0
