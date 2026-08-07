"""Gradient-boosted Cox proportional hazards model with graceful fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    _HAS_SKSURV = True
except ImportError:
    _HAS_SKSURV = False

try:
    from lifelines import CoxPHFitter
    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def _to_structured(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Convert parallel arrays to scikit-survival structured array."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )


@dataclass
class CoxBoostModel:
    """Gradient-boosted Cox model for time-to-event prediction.

    Uses ``scikit-survival``'s ``GradientBoostingSurvivalAnalysis`` when
    available; falls back to ``lifelines.CoxPHFitter``; and if neither is
    installed uses a simple logistic-regression hazard proxy so that the
    module remains importable and testable in lightweight environments.

    Parameters
    ----------
    feature_columns : list[str]
        Feature names used as predictors.
    n_estimators : int
        Number of boosting stages (scikit-survival only).
    learning_rate : float
        Shrinkage applied to each tree (scikit-survival only).
    max_depth : int
        Maximum tree depth (scikit-survival only).
    subsample : float
        Row subsampling fraction per stage.
    seed : int
        Random state for reproducibility.
    """

    feature_columns: list[str]
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    seed: int = 42

    # Private fitted state ------------------------------------------------
    _model: Any = field(default=None, repr=False, init=False)
    _imputer: SimpleImputer = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _scaler: StandardScaler = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    _backend: str = field(default="", repr=False, init=False)
    _baseline_times: np.ndarray | None = field(default=None, repr=False, init=False)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        frame: pd.DataFrame,
        time_col: str = "event_time_days",
        event_col: str = "event",
    ) -> "CoxBoostModel":
        """Fit the model on *frame* using available backend."""
        X = frame[self.feature_columns].copy()
        times = frame[time_col].to_numpy(dtype=float)
        events = frame[event_col].to_numpy(dtype=int)

        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        X_clean = pd.DataFrame(
            self._scaler.fit_transform(self._imputer.fit_transform(X)),
            columns=self.feature_columns,
        )

        if _HAS_SKSURV:
            self._backend = "sksurv"
            y = _to_structured(times, events)
            self._model = GradientBoostingSurvivalAnalysis(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                random_state=self.seed,
            )
            self._model.fit(X_clean.to_numpy(), y)
        elif _HAS_LIFELINES:
            self._backend = "lifelines"
            df = X_clean.copy()
            df["T"] = times
            df["E"] = events
            self._model = CoxPHFitter(penalizer=0.1)
            self._model.fit(df, duration_col="T", event_col="E")
        else:
            self._backend = "logistic_fallback"
            from sklearn.linear_model import LogisticRegression

            # Discrete-time proxy: predict event within median follow-up
            median_time = float(np.median(times))
            binary_label = ((events == 1) & (times <= median_time)).astype(int)
            self._model = LogisticRegression(
                max_iter=1200, class_weight="balanced", random_state=self.seed
            )
            self._model.fit(X_clean.to_numpy(), binary_label)

        self._baseline_times = np.sort(np.unique(times[events == 1]))
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _prepare(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self.feature_columns].copy()
        return self._scaler.transform(self._imputer.transform(X))

    def predict_risk_scores(self, frame: pd.DataFrame) -> np.ndarray:
        """Return risk scores (higher = worse prognosis)."""
        X = self._prepare(frame)
        if self._backend == "sksurv":
            return self._model.predict(X)
        elif self._backend == "lifelines":
            df = pd.DataFrame(X, columns=self.feature_columns)
            return self._model.predict_partial_hazard(df).to_numpy().ravel()
        else:
            return self._model.predict_proba(X)[:, 1]

    def predict_survival_function(
        self,
        frame: pd.DataFrame,
        time_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return survival probabilities at *time_points* for each row.

        Returns array of shape ``(n_samples, n_times)``.
        """
        if time_points is None:
            time_points = np.array([365.25, 365.25 * 3, 365.25 * 5])

        X = self._prepare(frame)

        if self._backend == "sksurv":
            surv_fns = self._model.predict_survival_function(X)
            n = len(surv_fns)
            result = np.full((n, len(time_points)), np.nan)
            for i, fn in enumerate(surv_fns):
                for j, t in enumerate(time_points):
                    result[i, j] = float(fn(t))
            return np.clip(result, 0.0, 1.0)
        elif self._backend == "lifelines":
            df = pd.DataFrame(X, columns=self.feature_columns)
            partial_hazard = self._model.predict_partial_hazard(df).to_numpy().ravel()
            # Approximate survival via S(t) = S0(t)^exp(risk)
            baseline_surv = np.exp(-np.searchsorted(self._baseline_times, time_points) * 0.01)
            result = np.column_stack(
                [baseline_surv[np.newaxis, :] ** ph for ph in partial_hazard]
            ).T  # shape (n_samples, n_times) -- transposed from column stack
            return np.clip(result, 0.0, 1.0)
        else:
            probs = self._model.predict_proba(X)[:, 1]
            # Approximate: S(t) ≈ (1-p)^(t/365.25)
            result = np.column_stack(
                [(1.0 - probs) ** (t / 365.25) for t in time_points]
            )
            return np.clip(result, 0.0, 1.0)

    def concordance_index(
        self,
        frame: pd.DataFrame,
        time_col: str = "event_time_days",
        event_col: str = "event",
    ) -> float:
        """Compute Harrell's C-index on *frame*."""
        risk = self.predict_risk_scores(frame)
        times = frame[time_col].to_numpy(dtype=float)
        events = frame[event_col].to_numpy(dtype=int)

        if _HAS_SKSURV:
            from sksurv.metrics import concordance_index_censored

            c, *_ = concordance_index_censored(events.astype(bool), times, risk)
            return float(c)
        # Pure-numpy fallback
        concordant = 0
        permissible = 0
        for i in range(len(times)):
            if events[i] == 0:
                continue
            for j in range(len(times)):
                if i == j:
                    continue
                if times[j] > times[i]:
                    permissible += 1
                    if risk[j] < risk[i]:
                        concordant += 1
                    elif risk[j] == risk[i]:
                        concordant += 0.5
        return float(concordant / max(permissible, 1))

    @property
    def backend(self) -> str:
        """Return which backend was used for fitting."""
        return self._backend
