"""Gradient Boosting Survival Analysis implementation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError

try:
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except ImportError:
    GradientBoostingSurvivalAnalysis = None
    Surv = None
    SKSURV_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    SKLEARN_INSPECTION_AVAILABLE = True
except ImportError:
    SKLEARN_INSPECTION_AVAILABLE = False


@dataclass
class GradientBoostingSurvivalModel:
    """Gradient Boosting Survival Analysis model wrapper."""

    feature_columns: list[str]
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        if not SKSURV_AVAILABLE:
            raise ImportError(
                "sksurv is required for GradientBoostingSurvivalModel but is not installed."
            )

    def _validate_data(self, frame: pd.DataFrame, is_fit: bool = False) -> None:
        """Validate columns and state."""
        if not is_fit and not hasattr(self, "model_"):
            raise NotFittedError("Model must be fitted before prediction or evaluation.")
            
        missing_cols = [col for col in self.feature_columns if col not in frame.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in data: {missing_cols}")

    def fit(self, frame: pd.DataFrame, time_col: str = "event_time_days", event_col: str = "event") -> "GradientBoostingSurvivalModel":
        """Fit the model to survival data."""
        self._validate_data(frame, is_fit=True)
        
        self.model_ = GradientBoostingSurvivalAnalysis(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.seed,
        )
        self.imputer_ = SimpleImputer(strategy="median")
        self.feature_columns_ = self.feature_columns.copy()
        
        # Check targets
        if time_col not in frame.columns or event_col not in frame.columns:
            raise ValueError(f"Missing target columns: {time_col} and/or {event_col}")
            
        clean_frame = frame.dropna(subset=[time_col, event_col])
        if not np.isfinite(clean_frame[time_col]).all():
            raise ValueError(f"Target time column '{time_col}' contains non-finite values.")
            
        X = clean_frame[self.feature_columns_].copy()
        X_imputed = self.imputer_.fit_transform(X)
        
        y = Surv.from_arrays(
            event=clean_frame[event_col].astype(bool), 
            time=clean_frame[time_col]
        )
        
        self.model_.fit(X_imputed, y)
        return self

    def predict_risk_scores(self, frame: pd.DataFrame) -> pd.Series:
        """Predict risk scores (higher = worse prognosis)."""
        self._validate_data(frame, is_fit=False)
            
        X = frame[self.feature_columns_].copy()
        X_imputed = self.imputer_.transform(X)
        
        risk_scores = self.model_.predict(X_imputed)
            
        return pd.Series(risk_scores, index=frame.index, name="risk_score")

    def predict_survival_function(self, frame: pd.DataFrame, time_points: np.ndarray | None = None) -> pd.DataFrame:
        """Return survival probabilities at given time points."""
        self._validate_data(frame, is_fit=False)
            
        X = frame[self.feature_columns_].copy()
        X_imputed = self.imputer_.transform(X)
        
        surv_funcs = self.model_.predict_survival_function(X_imputed)
        
        results = {}
        for i, sf in enumerate(surv_funcs):
            if time_points is not None:
                # Handle requested times outside the fitted survival-time range
                min_time, max_time = sf.x.min(), sf.x.max()
                safe_time_points = np.clip(time_points, min_time, max_time)
                results[frame.index[i]] = sf(safe_time_points)
            else:
                results[frame.index[i]] = sf(sf.x)
                if i == 0 and time_points is None:
                    time_points = sf.x
                    
        df = pd.DataFrame(results).T
        if time_points is not None:
            df.columns = time_points
        return df

    def concordance_index(self, frame: pd.DataFrame, time_col: str = "event_time_days", event_col: str = "event") -> float:
        """Calculate the concordance index (C-index)."""
        self._validate_data(frame, is_fit=False)
            
        clean_frame = frame.dropna(subset=[time_col, event_col])
        if not np.isfinite(clean_frame[time_col]).all():
            raise ValueError(f"Target time column '{time_col}' contains non-finite values.")
            
        risk_scores = self.predict_risk_scores(clean_frame)
        
        try:
            from sksurv.metrics import concordance_index_censored
            y_event = clean_frame[event_col].astype(bool).values
            y_time = clean_frame[time_col].values
            c_index, _, _, _, _ = concordance_index_censored(y_event, y_time, risk_scores)
            return c_index
        except ImportError:
            warnings.warn("sksurv is required for concordance_index_censored. Using lifelines if available.")
            try:
                from lifelines.utils import concordance_index
                return concordance_index(clean_frame[time_col], -risk_scores, clean_frame[event_col])
            except ImportError:
                warnings.warn("Neither sksurv nor lifelines is available. Returning 0.0.")
                return 0.0

    def permutation_importance(self, frame: pd.DataFrame, time_col: str = "event_time_days", event_col: str = "event", n_repeats: int = 5) -> pd.DataFrame:
        """Calculate permutation importance."""
        if not SKLEARN_INSPECTION_AVAILABLE:
            warnings.warn("sklearn permutation_importance is not available.")
            return pd.DataFrame()
            
        self._validate_data(frame, is_fit=False)
            
        clean_frame = frame.dropna(subset=[time_col, event_col])
        if not np.isfinite(clean_frame[time_col]).all():
            raise ValueError(f"Target time column '{time_col}' contains non-finite values.")
        
        X = clean_frame[self.feature_columns_].copy()
        X_imputed = self.imputer_.transform(X)
        
        y = Surv.from_arrays(
            event=clean_frame[event_col].astype(bool), 
            time=clean_frame[time_col]
        )
        
        # sksurv models' .score() computes C-index, so we don't need a custom scorer explicitly
        # sklearn's permutation_importance uses estimator.score() if scoring is None
        result = permutation_importance(
            self.model_, X_imputed, y, n_repeats=n_repeats, random_state=self.seed, n_jobs=-1
        )
        
        return pd.DataFrame({
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }, index=self.feature_columns_).sort_values("importance_mean", ascending=False)
