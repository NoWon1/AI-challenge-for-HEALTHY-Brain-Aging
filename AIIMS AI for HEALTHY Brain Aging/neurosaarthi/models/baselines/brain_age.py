"""Elastic-net brain-age baseline with held-out bias and interval calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neurosaarthi.core.errors import DataValidationError


@dataclass(frozen=True)
class BrainAgePrediction:
    predicted_age: np.ndarray
    brain_age_gap: np.ndarray
    lower_age: np.ndarray | None = None
    upper_age: np.ndarray | None = None


class BrainAgeBaseline:
    """Feature-based elastic-net baseline.

    ``fit`` learns the regression only from the development set. ``calibrate``
    then learns age-bias correction and a split-conformal residual quantile on
    a participant-disjoint calibration set.
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        *,
        alpha: float = 0.05,
        l1_ratio: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if not feature_columns:
            raise DataValidationError("At least one brain-age feature is required")
        self.feature_columns = list(dict.fromkeys(feature_columns))
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.random_state = int(random_state)
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    ElasticNet(
                        alpha=self.alpha,
                        l1_ratio=self.l1_ratio,
                        max_iter=20_000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self._fitted = False
        self._bias_slope: float | None = None
        self._bias_intercept: float | None = None
        self._interval_quantile: float | None = None
        self._interval_alpha: float | None = None

    def fit(self, frame: pd.DataFrame, chronological_age: Sequence[float]) -> BrainAgeBaseline:
        X = self._features(frame)
        y = _numeric_target(chronological_age, len(frame), "chronological_age")
        if len(frame) < 3:
            raise DataValidationError("Brain-age fitting requires at least three rows")
        self.pipeline.fit(X, y)
        self._fitted = True
        self._bias_slope = None
        self._bias_intercept = None
        self._interval_quantile = None
        self._interval_alpha = None
        return self

    def calibrate(
        self,
        frame: pd.DataFrame,
        chronological_age: Sequence[float],
        *,
        interval_alpha: float = 0.10,
    ) -> BrainAgeBaseline:
        self._require_fitted()
        if not 0 < interval_alpha < 1:
            raise DataValidationError("interval_alpha must be between 0 and 1")
        age = _numeric_target(chronological_age, len(frame), "chronological_age")
        if len(age) < 3:
            raise DataValidationError("Brain-age calibration requires at least three rows")
        raw = self.predict_raw(frame)
        gap = raw - age
        slope, intercept = np.polyfit(age, gap, deg=1)
        corrected = raw - (slope * age + intercept)
        residuals = np.abs(age - corrected)
        rank = int(np.ceil((len(residuals) + 1) * (1 - interval_alpha)))
        rank = min(max(rank, 1), len(residuals))
        self._bias_slope = float(slope)
        self._bias_intercept = float(intercept)
        self._interval_quantile = float(np.partition(residuals, rank - 1)[rank - 1])
        self._interval_alpha = float(interval_alpha)
        return self

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self.pipeline.predict(self._features(frame)), dtype=float)

    def predict(
        self,
        frame: pd.DataFrame,
        chronological_age: Sequence[float],
        *,
        include_interval: bool = True,
    ) -> BrainAgePrediction:
        self._require_fitted()
        age = _numeric_target(chronological_age, len(frame), "chronological_age")
        predicted = self.predict_raw(frame)
        if self._bias_slope is not None and self._bias_intercept is not None:
            predicted = predicted - (self._bias_slope * age + self._bias_intercept)
        lower = upper = None
        if include_interval:
            if self._interval_quantile is None:
                raise NotFittedError("Call calibrate before requesting brain-age intervals")
            lower = predicted - self._interval_quantile
            upper = predicted + self._interval_quantile
        return BrainAgePrediction(
            predicted_age=predicted,
            brain_age_gap=predicted - age,
            lower_age=lower,
            upper_age=upper,
        )

    @property
    def calibration_metadata(self) -> dict[str, float] | None:
        if self._bias_slope is None or self._bias_intercept is None or self._interval_quantile is None:
            return None
        return {
            "bias_slope": self._bias_slope,
            "bias_intercept": self._bias_intercept,
            "interval_alpha": float(self._interval_alpha),
            "interval_half_width_years": self._interval_quantile,
        }

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in frame]
        if missing:
            raise DataValidationError(f"Missing brain-age features: {', '.join(missing)}")
        features = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        entirely_missing = features.columns[features.isna().all()].tolist()
        if entirely_missing and self._fitted:
            # The fitted imputer can handle a column that was empty at fit time,
            # but a new entirely-missing required block is usually a contract error.
            fitted_statistics = np.asarray(self.pipeline.named_steps["imputer"].statistics_)
            trained_empty = {
                name
                for name, statistic in zip(self.feature_columns, fitted_statistics, strict=False)
                if np.isnan(statistic)
            }
            unexpected = sorted(set(entirely_missing) - trained_empty)
            if unexpected:
                raise DataValidationError(
                    f"Required feature blocks are entirely missing: {', '.join(unexpected)}"
                )
        return features

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError("BrainAgeBaseline is not fitted")


def _numeric_target(values: Sequence[float], expected_length: int, name: str) -> np.ndarray:
    result = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if len(result) != expected_length:
        raise DataValidationError(f"{name} length does not match the feature frame")
    if not np.isfinite(result).all():
        raise DataValidationError(f"{name} must be finite and complete")
    return result
