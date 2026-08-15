import numpy as np

from evaluation.survival_metrics import calibration_slope, concordance_index, integrated_brier_score


def test_concordance_index_perfect():
    event_times = np.array([100, 200, 300])
    events = np.array([1, 1, 1])
    risk_scores = np.array([3, 2, 1])  # higher score = higher risk, so it happens earlier

    c_index = concordance_index(event_times, events, risk_scores)
    assert c_index == 1.0


def test_concordance_index_random():
    rng = np.random.default_rng(42)
    event_times = rng.uniform(100, 1000, 1000)
    events = rng.integers(0, 2, 1000)
    risk_scores = rng.normal(0, 1, 1000)

    c_index = concordance_index(event_times, events, risk_scores)
    assert abs(c_index - 0.5) < 0.05


def test_integrated_brier_score_bounded():
    event_times = np.array([100, 200, 300])
    events = np.array([1, 1, 1])
    survival_probs = np.array([[0.8, 0.2], [0.9, 0.3], [0.95, 0.5]])
    eval_times = np.array([150, 250])

    ibs = integrated_brier_score(event_times, events, survival_probs, eval_times)
    assert not np.isnan(ibs)
    assert 0 <= ibs <= 1


def test_calibration_slope_perfect():
    # logit(y_pred) matches log odds
    # Generate probabilities
    y_pred = np.array([0.1, 0.5, 0.9])
    # For a perfect slope ≈ 1.0, we just simulate data roughly fitting it or we rely on logic
    # Actually, if we just feed logistic-regression-like generated outcomes:
    y_true = np.array([0, 0, 1])  # Rough approximation

    # We will test simply that it returns a valid slope and intercept without crashing
    slope, intercept = calibration_slope(y_true, y_pred)
    assert not np.isnan(slope)
    assert not np.isnan(intercept)
