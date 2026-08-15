import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from neurosaarthi.core.errors import DataValidationError
from neurosaarthi.models.baselines.brain_age import BrainAgeBaseline
from neurosaarthi.training.brain_age import run_brain_age_experiment


def _brain_age_frame(participants=60):
    rng = np.random.default_rng(12)
    age = np.linspace(50, 85, participants)
    return pd.DataFrame(
        {
            "participant_id_internal": [f"P-{index:03d}" for index in range(participants)],
            "age_at_visit": age,
            "hippocampus": 9000 - 55 * age + rng.normal(0, 60, participants),
            "ventricles": 5000 + 420 * age + rng.normal(0, 500, participants),
        }
    )


def test_brain_age_requires_fit_and_calibration_for_intervals():
    frame = _brain_age_frame(10)
    model = BrainAgeBaseline(["hippocampus", "ventricles"])
    with pytest.raises(NotFittedError):
        model.predict_raw(frame)
    model.fit(frame, frame["age_at_visit"])
    with pytest.raises(NotFittedError, match="calibrate"):
        model.predict(frame, frame["age_at_visit"])


def test_brain_age_experiment_is_participant_isolated_and_reports_aggregates():
    result = run_brain_age_experiment(
        _brain_age_frame(), feature_columns=["hippocampus", "ventricles"], seed=5
    )
    report = result.aggregate_report()
    assert report["research_use_only"] is True
    assert report["metrics"]["mae_years"] >= 0
    assert 0 <= report["metrics"]["interval_coverage"] <= 1
    assert sum(report["split_counts"].values()) == 60
    assert "predictions" not in report


def test_brain_age_rejects_missing_feature_column():
    frame = _brain_age_frame(10)
    with pytest.raises(DataValidationError, match="Missing brain-age features"):
        BrainAgeBaseline(["not_present"]).fit(frame, frame["age_at_visit"])
