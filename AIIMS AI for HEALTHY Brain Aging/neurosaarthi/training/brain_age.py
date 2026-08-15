"""Participant-isolated brain-age train/calibration/test experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from neurosaarthi.core.errors import DataValidationError
from neurosaarthi.data.splitters.longitudinal import assert_participant_isolation, group_shuffle_split
from neurosaarthi.models.baselines.brain_age import BrainAgeBaseline


@dataclass(frozen=True)
class BrainAgeMetrics:
    participants: int
    rows: int
    mae_years: float
    rmse_years: float
    r2: float
    mean_brain_age_gap_years: float
    interval_coverage: float
    interval_mean_width_years: float


@dataclass(frozen=True)
class BrainAgeExperimentResult:
    model: BrainAgeBaseline
    metrics: BrainAgeMetrics
    train_participants: int
    calibration_participants: int
    test_participants: int

    def aggregate_report(self) -> dict[str, object]:
        """Return aggregate metrics only; participant predictions are excluded."""

        return {
            "task": "brain_age_regression",
            "research_use_only": True,
            "metrics": asdict(self.metrics),
            "split_counts": {
                "train_participants": self.train_participants,
                "calibration_participants": self.calibration_participants,
                "test_participants": self.test_participants,
            },
            "calibration": self.model.calibration_metadata,
        }


def run_brain_age_experiment(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    age_col: str = "age_at_visit",
    participant_col: str = "participant_id_internal",
    test_size: float = 0.20,
    calibration_size: float = 0.20,
    seed: int = 42,
) -> BrainAgeExperimentResult:
    """Fit, calibrate, and test without participant overlap."""

    if not 0 < calibration_size < 1 - test_size:
        raise DataValidationError("calibration_size must be positive and leave data for training")
    required = {age_col, participant_col, *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"Brain-age experiment is missing columns: {', '.join(missing)}")
    development, test = group_shuffle_split(
        frame, participant_col=participant_col, test_size=test_size, seed=seed
    )
    relative_calibration_size = calibration_size / (1.0 - test_size)
    train, calibration = group_shuffle_split(
        development,
        participant_col=participant_col,
        test_size=relative_calibration_size,
        seed=seed + 1,
    )
    assert_participant_isolation(train, calibration, test, participant_col=participant_col)
    model = BrainAgeBaseline(feature_columns, random_state=seed)
    model.fit(train, train[age_col])
    model.calibrate(calibration, calibration[age_col])
    prediction = model.predict(test, test[age_col], include_interval=True)
    observed = pd.to_numeric(test[age_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise DataValidationError("Test ages must be finite")
    lower = np.asarray(prediction.lower_age)
    upper = np.asarray(prediction.upper_age)
    metrics = BrainAgeMetrics(
        participants=int(test[participant_col].nunique()),
        rows=len(test),
        mae_years=float(mean_absolute_error(observed, prediction.predicted_age)),
        rmse_years=float(np.sqrt(mean_squared_error(observed, prediction.predicted_age))),
        r2=float(r2_score(observed, prediction.predicted_age)),
        mean_brain_age_gap_years=float(np.mean(prediction.brain_age_gap)),
        interval_coverage=float(np.mean((observed >= lower) & (observed <= upper))),
        interval_mean_width_years=float(np.mean(upper - lower)),
    )
    return BrainAgeExperimentResult(
        model=model,
        metrics=metrics,
        train_participants=int(train[participant_col].nunique()),
        calibration_participants=int(calibration[participant_col].nunique()),
        test_participants=int(test[participant_col].nunique()),
    )
