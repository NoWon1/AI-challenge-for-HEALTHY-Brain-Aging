"""Leakage-resistant participant-level and temporal splitting."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurosaarthi.core.errors import DataValidationError


@dataclass(frozen=True)
class GroupFold:
    fold: int
    train_index: np.ndarray
    validation_index: np.ndarray


def _participants(frame: pd.DataFrame, participant_col: str) -> np.ndarray:
    if participant_col not in frame:
        raise DataValidationError(f"Missing participant column: {participant_col}")
    if frame[participant_col].isna().any():
        raise DataValidationError("Participant identifiers cannot be missing")
    values = frame[participant_col].drop_duplicates().astype(str).to_numpy()
    if len(values) < 2:
        raise DataValidationError("At least two participants are required for a split")
    return values


def group_shuffle_split(
    frame: pd.DataFrame,
    *,
    participant_col: str = "participant_id_internal",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete participants into train and test sets."""

    if not 0 < test_size < 1:
        raise DataValidationError("test_size must be between 0 and 1")
    participants = _participants(frame, participant_col)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(participants)
    test_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * test_size)))
    test_ids = set(shuffled[:test_count])
    test = frame[frame[participant_col].astype(str).isin(test_ids)].copy()
    train = frame[~frame[participant_col].astype(str).isin(test_ids)].copy()
    assert_participant_isolation(train, test, participant_col=participant_col)
    return train, test


def group_kfold(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    participant_col: str = "participant_id_internal",
    seed: int = 42,
) -> Iterator[GroupFold]:
    """Yield deterministic shuffled participant-level folds."""

    participants = _participants(frame, participant_col)
    if not 2 <= n_splits <= len(participants):
        raise DataValidationError("n_splits must be between 2 and the participant count")
    shuffled = np.random.default_rng(seed).permutation(participants)
    for fold, validation_ids in enumerate(np.array_split(shuffled, n_splits)):
        validation_mask = frame[participant_col].astype(str).isin(set(validation_ids))
        yield GroupFold(
            fold=fold,
            train_index=np.flatnonzero(~validation_mask.to_numpy()),
            validation_index=np.flatnonzero(validation_mask.to_numpy()),
        )


def site_held_out_split(
    frame: pd.DataFrame,
    held_out_site: str,
    *,
    site_col: str = "site",
    participant_col: str = "participant_id_internal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _domain_held_out_split(frame, site_col, held_out_site, participant_col)


def cohort_held_out_split(
    frame: pd.DataFrame,
    held_out_cohort: str,
    *,
    cohort_col: str = "cohort",
    participant_col: str = "participant_id_internal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _domain_held_out_split(frame, cohort_col, held_out_cohort, participant_col)


def _domain_held_out_split(
    frame: pd.DataFrame,
    domain_col: str,
    held_out_value: str,
    participant_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if domain_col not in frame:
        raise DataValidationError(f"Missing domain column: {domain_col}")
    test = frame[frame[domain_col].astype(str) == held_out_value].copy()
    train = frame[frame[domain_col].astype(str) != held_out_value].copy()
    if train.empty or test.empty:
        raise DataValidationError("Held-out split must produce non-empty train and test sets")
    leaking = set(train[participant_col].astype(str)) & set(test[participant_col].astype(str))
    if leaking:
        raise DataValidationError(
            "Participants occur in both held-out domain and development domains; resolve cohort/site provenance first"
        )
    return train, test


def assert_participant_isolation(
    *frames: pd.DataFrame, participant_col: str = "participant_id_internal"
) -> None:
    """Assert pairwise-disjoint participant IDs across any number of frames."""

    seen: set[str] = set()
    for index, frame in enumerate(frames):
        if participant_col not in frame:
            raise DataValidationError(f"Frame {index} is missing {participant_col}")
        current = set(frame[participant_col].dropna().astype(str))
        overlap = seen & current
        if overlap:
            raise DataValidationError(
                f"Participant leakage detected across splits ({len(overlap)} identifiers)"
            )
        seen.update(current)


def assert_no_future_features(
    frame: pd.DataFrame,
    *,
    feature_time_col: str = "feature_time_days",
    prediction_origin_col: str = "prediction_origin_days",
) -> None:
    """Reject features acquired after the row's prediction origin."""

    missing = {feature_time_col, prediction_origin_col} - set(frame.columns)
    if missing:
        raise DataValidationError(f"Temporal leakage check is missing columns: {', '.join(sorted(missing))}")
    times = frame[[feature_time_col, prediction_origin_col]].apply(pd.to_numeric, errors="coerce")
    if times.isna().any().any():
        raise DataValidationError("Temporal leakage fields must be numeric and complete")
    violations = times[feature_time_col] > times[prediction_origin_col]
    if violations.any():
        raise DataValidationError(f"Future-feature leakage detected in {int(violations.sum())} rows")
