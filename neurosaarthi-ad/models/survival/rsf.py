"""Random Survival Forest implementation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from sksurv.ensemble import RandomSurvivalForest
    SKSURV_AVAILABLE = True
except ImportError:
    RandomSurvivalForest = None
    SKSURV_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    SKLEARN_INSPECTION_AVAILABLE = True
except ImportError:
    SKLEARN_INSPECTION_AVAILABLE = False


@dataclass
class RandomSurvivalForestModel:
    """Random Survival Forest model wrapper."""

    feature_columns: list[str]
    n_estimators: int = 200
    max_depth: int = 8
    min_samples_leaf: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        self.model = None

    def fit(self, frame: pd.DataFrame, time_col: str = "event_time_days", event_col: str = "event") -> "RandomSurvivalForestModel":
        """Fit the model to survival data."""
        if SKSURV_AVAILABLE:
            self.model = RandomSurvivalForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed,
            )
            
            # sksurv requires a structured array for the target
            # Drop NaN times/events
            clean_frame = frame.dropna(subset=[time_col, event_col, *self.feature_columns])
            
            # Create structured array
            y = np.array(
                list(zip(clean_frame[event_col].astype(bool), clean_frame[time_col])),
                dtype=[('event', '?'), ('time', '<f8')]
            )
            
            self.model.fit(clean_frame[self.feature_columns], y)
        else:
            # Fallback to Logistic Regression + Discrete intervals as mentioned in prompt
            warnings.warn("sksurv is not installed. Falling back to simple LogisticRegression.")
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.impute import SimpleImputer
                from sklearn.pipeline import Pipeline
            except ImportError:
                raise ImportError("sklearn is required for the fallback model.")

            self.model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("lr", LogisticRegression(max_iter=1000, random_state=self.seed))
            ])
            clean_frame = frame.dropna(subset=[event_col])
            self.model.fit(clean_frame[self.feature_columns], clean_frame[event_col])
            
        return self

    def predict_risk_scores(self, frame: pd.DataFrame) -> pd.Series:
        """Predict risk scores (higher = worse prognosis)."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
            
        x = frame[self.feature_columns]
        
        if SKSURV_AVAILABLE and isinstance(self.model, RandomSurvivalForest):
            # Impute NaNs manually if necessary for sksurv, which doesn't support NaNs natively
            # But the prompt doesn't strictly say RSF must handle NaNs. We will fill with medians for robustness.
            x = x.fillna(x.median())
            risk_scores = self.model.predict(x)
        else:
            # Fallback Logistic Regression risk
            x = x.fillna(x.median())
            risk_scores = self.model.predict_proba(x)[:, 1]
            
        return pd.Series(risk_scores, index=frame.index, name="risk_score")

    def predict_survival_function(self, frame: pd.DataFrame, time_points: np.ndarray | None = None) -> pd.DataFrame:
        """Return survival probabilities at given time points."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
            
        if not SKSURV_AVAILABLE or not isinstance(self.model, RandomSurvivalForest):
            warnings.warn("Survival function prediction is not fully supported by the fallback model.")
            # Dummy output
            return pd.DataFrame(index=frame.index)
            
        x = frame[self.feature_columns].fillna(frame[self.feature_columns].median())
        surv_funcs = self.model.predict_survival_function(x)
        
        # surv_funcs is an array of StepFunction objects
        results = {}
        for i, sf in enumerate(surv_funcs):
            if time_points is not None:
                results[frame.index[i]] = sf(time_points)
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
        if self.model is None:
            raise RuntimeError("Model must be fitted before evaluation.")
            
        clean_frame = frame.dropna(subset=[time_col, event_col])
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
                # lifelines expects higher score = better prognosis (longer survival), 
                # so we might need to negate risk_scores or be careful.
                # concordance_index(event_times, predicted_scores, event_observed)
                return concordance_index(clean_frame[time_col], -risk_scores, clean_frame[event_col])
            except ImportError:
                warnings.warn("Neither sksurv nor lifelines is available. Returning 0.0.")
                return 0.0

    def permutation_importance(self, frame: pd.DataFrame, time_col: str = "event_time_days", event_col: str = "event", n_repeats: int = 5) -> pd.DataFrame:
        """Calculate permutation importance."""
        if not SKLEARN_INSPECTION_AVAILABLE:
            warnings.warn("sklearn permutation_importance is not available.")
            return pd.DataFrame()
            
        if self.model is None:
            raise RuntimeError("Model must be fitted before permutation importance.")
            
        clean_frame = frame.dropna(subset=[time_col, event_col, *self.feature_columns])
        
        if SKSURV_AVAILABLE and isinstance(self.model, RandomSurvivalForest):
            y = np.array(
                list(zip(clean_frame[event_col].astype(bool), clean_frame[time_col])),
                dtype=[('event', '?'), ('time', '<f8')]
            )
        else:
            y = clean_frame[event_col]
            
        x = clean_frame[self.feature_columns]
        
        result = permutation_importance(
            self.model, x, y, n_repeats=n_repeats, random_state=self.seed, n_jobs=-1
        )
        
        return pd.DataFrame({
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }, index=self.feature_columns).sort_values("importance_mean", ascending=False)
