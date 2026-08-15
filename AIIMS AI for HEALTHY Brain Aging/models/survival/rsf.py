"""Random Survival Forest with a documented laptop-safe fallback."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.linear_model import LogisticRegression

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    SKSURV_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional environment
    RandomSurvivalForest = None
    Surv = None
    concordance_index_censored = None
    SKSURV_AVAILABLE = False


@dataclass
class RandomSurvivalForestModel:
    """Random Survival Forest wrapper with train-only preprocessing.

    When scikit-survival is unavailable, a logistic event model plus a
    constant-baseline exponential survival curve keeps laptop/demo inference
    functional. That fallback is an engineering baseline, not an equivalent
    scientific replacement for an RSF.
    """

    feature_columns: list[str]
    n_estimators: int = 200
    max_depth: int = 8
    min_samples_leaf: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        self.model = None
        self.imputer_: SimpleImputer | None = None
        self.min_time_: float | None = None
        self.max_time_: float | None = None
        self.baseline_hazard_per_day_: float | None = None
        self.reference_probability_: float | None = None
        self._using_fallback = False

    def fit(
        self,
        frame: pd.DataFrame,
        time_col: str = "event_time_days",
        event_col: str = "event",
    ) -> RandomSurvivalForestModel:
        """Fit on complete survival targets and train-only imputed features."""

        self._validate_columns(frame, include_targets=(time_col, event_col))
        clean = frame.dropna(subset=[time_col, event_col]).copy()
        times = pd.to_numeric(clean[time_col], errors="coerce").to_numpy(dtype=float)
        if len(clean) < 3 or not np.isfinite(times).all() or (times <= 0).any():
            raise ValueError("Survival fitting requires at least three finite positive event/censoring times")
        events = clean[event_col].astype(bool).to_numpy()
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        features = self.imputer_.fit_transform(clean[self.feature_columns])
        self.min_time_ = float(np.min(times))
        self.max_time_ = float(np.max(times))

        if SKSURV_AVAILABLE:
            self.model = RandomSurvivalForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed,
                n_jobs=1,
            )
            target = Surv.from_arrays(event=events, time=times)
            self.model.fit(features, target)
            self._using_fallback = False
        else:
            warnings.warn(
                "scikit-survival is unavailable; using the documented logistic/exponential survival fallback.",
                stacklevel=2,
            )
            if np.unique(events).size == 1:
                self.model = DummyClassifier(strategy="constant", constant=int(events[0]))
            else:
                self.model = LogisticRegression(
                    max_iter=5000, class_weight="balanced", random_state=self.seed
                )
            self.model.fit(features, events.astype(int))
            event_count = float(events.sum())
            self.baseline_hazard_per_day_ = (event_count + 0.5) / (float(times.sum()) + self.max_time_)
            fitted_probabilities = self._fallback_event_probability(features)
            self.reference_probability_ = float(np.clip(np.mean(fitted_probabilities), 1e-5, 1 - 1e-5))
            self._using_fallback = True
        return self

    def predict_risk_scores(self, frame: pd.DataFrame) -> pd.Series:
        """Predict risk scores; larger values indicate shorter expected survival."""

        features = self._transform(frame)
        if self._using_fallback:
            scores = self._fallback_event_probability(features)
        else:
            scores = np.asarray(self.model.predict(features), dtype=float)
        return pd.Series(scores, index=frame.index, name="risk_score")

    def predict_survival_function(
        self,
        frame: pd.DataFrame,
        time_points: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Return monotone survival probabilities.

        Requested times below the fitted support return survival 1.0; times
        above it are explicitly clipped to the largest fitted time.
        """

        features = self._transform(frame)
        requested = self._time_grid(time_points)
        assert self.min_time_ is not None and self.max_time_ is not None
        if self._using_fallback:
            probabilities = self._fallback_event_probability(features)
            hazards = self._fallback_hazards(probabilities)
            evaluation_times = np.clip(requested, 0.0, self.max_time_)
            values = np.exp(-hazards[:, None] * evaluation_times[None, :])
        else:
            functions = self.model.predict_survival_function(features)
            values = np.ones((len(frame), len(requested)), dtype=float)
            in_support = requested >= self.min_time_
            safe_times = np.clip(requested[in_support], self.min_time_, self.max_time_)
            for row, function in enumerate(functions):
                values[row, in_support] = function(safe_times)
        values = np.minimum.accumulate(np.clip(values, 0.0, 1.0), axis=1)
        return pd.DataFrame(values, index=frame.index, columns=requested)

    def predict_survival(
        self,
        frame: pd.DataFrame,
        time_points: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Stable interface used by the local dashboard."""

        if time_points is None:
            time_points = np.asarray([365.25, 3 * 365.25, 5 * 365.25], dtype=float)
        return self.predict_survival_function(frame, time_points=time_points)

    def concordance_index(
        self,
        frame: pd.DataFrame,
        time_col: str = "event_time_days",
        event_col: str = "event",
    ) -> float:
        """Compute Harrell's C-index using event/censoring information."""

        self._validate_columns(frame, include_targets=(time_col, event_col))
        clean = frame.dropna(subset=[time_col, event_col])
        times = pd.to_numeric(clean[time_col], errors="coerce").to_numpy(dtype=float)
        events = clean[event_col].astype(bool).to_numpy()
        if not np.isfinite(times).all() or (times <= 0).any():
            raise ValueError("Survival times must be finite and positive")
        risks = self.predict_risk_scores(clean).to_numpy()
        if concordance_index_censored is not None:
            return float(concordance_index_censored(events, times, risks)[0])
        return _harrell_c_index(events, times, risks)

    def permutation_importance(
        self,
        frame: pd.DataFrame,
        time_col: str = "event_time_days",
        event_col: str = "event",
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """Compute feature importance using an explicit survival-aware C-index scorer."""

        self._validate_columns(frame, include_targets=(time_col, event_col))
        clean = frame.dropna(subset=[time_col, event_col])
        features = self._transform(clean)
        times = pd.to_numeric(clean[time_col], errors="coerce").to_numpy(dtype=float)
        events = clean[event_col].astype(bool).to_numpy()
        target = np.array(list(zip(events, times, strict=True)), dtype=[("event", "?"), ("time", "<f8")])

        def survival_scorer(estimator, values, structured_target) -> float:
            if self._using_fallback:
                if hasattr(estimator, "predict_proba") and estimator.predict_proba(values).shape[1] > 1:
                    risk = estimator.predict_proba(values)[:, 1]
                else:
                    risk = np.asarray(estimator.predict(values), dtype=float)
            else:
                risk = np.asarray(estimator.predict(values), dtype=float)
            observed = structured_target["event"]
            durations = structured_target["time"]
            if concordance_index_censored is not None:
                return float(concordance_index_censored(observed, durations, risk)[0])
            return _harrell_c_index(observed, durations, risk)

        result = sklearn_permutation_importance(
            self.model,
            features,
            target,
            scoring=survival_scorer,
            n_repeats=n_repeats,
            random_state=self.seed,
            n_jobs=1,
        )
        return pd.DataFrame(
            {"importance_mean": result.importances_mean, "importance_std": result.importances_std},
            index=self.feature_columns,
        ).sort_values("importance_mean", ascending=False)

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        self._validate_columns(frame)
        assert self.imputer_ is not None
        return np.asarray(self.imputer_.transform(frame[self.feature_columns]), dtype=float)

    def _time_grid(self, time_points: np.ndarray | None) -> np.ndarray:
        self._require_fitted()
        if time_points is None:
            if not self._using_fallback and hasattr(self.model, "unique_times_"):
                requested = np.asarray(self.model.unique_times_, dtype=float)
            else:
                assert self.max_time_ is not None
                requested = np.linspace(0.0, self.max_time_, 50)
        else:
            requested = np.asarray(time_points, dtype=float)
        if requested.ndim != 1 or requested.size == 0 or not np.isfinite(requested).all():
            raise ValueError("time_points must be a non-empty finite one-dimensional array")
        if (requested < 0).any() or np.any(np.diff(requested) < 0):
            raise ValueError("time_points must be non-negative and sorted")
        return requested

    def _fallback_event_probability(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.model.predict_proba(features)
        classes = np.asarray(self.model.classes_)
        if 1 in classes:
            return np.asarray(probabilities[:, int(np.flatnonzero(classes == 1)[0])], dtype=float)
        return np.zeros(len(features), dtype=float)

    def _fallback_hazards(self, probability: np.ndarray) -> np.ndarray:
        assert self.baseline_hazard_per_day_ is not None and self.reference_probability_ is not None
        clipped = np.clip(probability, 1e-5, 1 - 1e-5)
        reference = self.reference_probability_
        log_odds = np.log(clipped / (1 - clipped)) - np.log(reference / (1 - reference))
        return self.baseline_hazard_per_day_ * np.exp(np.clip(log_odds, -5.0, 5.0))

    def _validate_columns(self, frame: pd.DataFrame, include_targets: tuple[str, ...] = ()) -> None:
        missing = [column for column in [*self.feature_columns, *include_targets] if column not in frame]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _require_fitted(self) -> None:
        if self.model is None or self.imputer_ is None:
            raise NotFittedError("RandomSurvivalForestModel must be fitted before prediction")


def _harrell_c_index(events: np.ndarray, times: np.ndarray, risks: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    for left in range(len(times)):
        for right in range(left + 1, len(times)):
            if times[left] == times[right]:
                continue
            earlier, later = (left, right) if times[left] < times[right] else (right, left)
            if not events[earlier]:
                continue
            comparable += 1.0
            if risks[earlier] > risks[later]:
                concordant += 1.0
            elif risks[earlier] == risks[later]:
                concordant += 0.5
    return float(concordant / comparable) if comparable else 0.5
