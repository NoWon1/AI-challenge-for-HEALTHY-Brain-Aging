"""LightGBM-based gradient-boosted risk classifier."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LGBM_AVAILABLE = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    HistGradientBoostingClassifier = None

try:
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    CalibratedClassifierCV = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


@dataclass
class GBMRiskClassifier:
    """LightGBM-based risk classifier with graceful fallback."""

    feature_columns: list[str]
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 5
    seed: int = 42
    calibrate: bool = False

    def __post_init__(self) -> None:
        self.model = None
        self._is_calibrated = False

    def _get_base_model(self) -> Any:
        """Get the underlying gradient boosting model."""
        if LGBM_AVAILABLE:
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.seed,
                importance_type='gain',
            )
        elif HistGradientBoostingClassifier is not None:
            warnings.warn("LightGBM not found, falling back to HistGradientBoostingClassifier.")
            return HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.seed,
            )
        else:
            raise ImportError("Neither lightgbm nor sklearn.ensemble.HistGradientBoostingClassifier is available.")

    def fit(self, frame: pd.DataFrame, target_col: str = "event") -> "GBMRiskClassifier":
        """Fit the gradient boosting model with optional isotonic calibration."""
        x = frame[self.feature_columns]
        y = frame[target_col]

        base_model = self._get_base_model()

        if self.calibrate and CalibratedClassifierCV is not None:
            self.model = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=5)
            self.model.fit(x, y)
            self._is_calibrated = True
        else:
            if self.calibrate and CalibratedClassifierCV is None:
                warnings.warn("Calibration requested but scikit-learn is missing. Fitting uncalibrated model.")
            self.model = base_model
            self.model.fit(x, y)
            self._is_calibrated = False

        return self

    def predict_risk(self, frame: pd.DataFrame) -> pd.Series:
        """Return probability estimates for the risk."""
        if self.model is None:
            raise RuntimeError("Classifier must be fitted before predict_risk is called.")
        
        x = frame[self.feature_columns]
        probabilities = self.model.predict_proba(x)[:, 1]
        return pd.Series(probabilities, index=frame.index, name="risk")

    def feature_importance(self) -> dict[str, float]:
        """Return a dictionary mapping feature names to their importance (gain)."""
        if self.model is None:
            raise RuntimeError("Classifier must be fitted before getting feature importance.")
        
        # If calibrated, extracting feature importance from the ensemble of calibrated models is complex.
        # We will extract it from a single model if possible, or warn.
        model_to_inspect = self.model
        if self._is_calibrated:
            warnings.warn("Extracting feature importance from first calibrated estimator.")
            model_to_inspect = self.model.calibrated_classifiers_[0].estimator

        if hasattr(model_to_inspect, "feature_importances_"):
            importances = model_to_inspect.feature_importances_
            if hasattr(model_to_inspect, "feature_name_"):
                # LightGBM
                names = model_to_inspect.feature_name_
            else:
                names = self.feature_columns
            return dict(zip(names, importances))
        else:
            warnings.warn("Underlying model does not support feature importances.")
            return {}

    def shap_values(self, frame: pd.DataFrame) -> np.ndarray | None:
        """Extract SHAP values if the library is available, else return None."""
        if not SHAP_AVAILABLE:
            warnings.warn("shap library is not installed. Returning None.")
            return None
        if self.model is None:
            raise RuntimeError("Classifier must be fitted before getting SHAP values.")

        model_to_inspect = self.model
        if self._is_calibrated:
            model_to_inspect = self.model.calibrated_classifiers_[0].estimator

        x = frame[self.feature_columns]
        try:
            explainer = shap.TreeExplainer(model_to_inspect)
            shap_vals = explainer.shap_values(x)
            # shap_values might return a list for multiclass/binary, we want the positive class
            if isinstance(shap_vals, list):
                return shap_vals[1]
            return shap_vals
        except Exception as e:
            warnings.warn(f"Failed to extract SHAP values: {e}")
            return None

    def fit_multi_horizon(self, frame: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5)) -> dict[int, "GBMRiskClassifier"]:
        """Train separate models per horizon.
        
        Assumes target columns exist as `event_{horizon}` in the frame.
        """
        models = {}
        for horizon in horizons:
            target = f"event_{horizon}"
            if target not in frame.columns:
                warnings.warn(f"Target column '{target}' not found. Skipping horizon {horizon}.")
                continue
            
            # create a new instance with the same parameters
            model = GBMRiskClassifier(
                feature_columns=self.feature_columns,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                seed=self.seed + horizon,  # optional: jitter seed
                calibrate=self.calibrate,
            )
            # drop nans in target
            sub_frame = frame.dropna(subset=[target])
            model.fit(sub_frame, target_col=target)
            models[horizon] = model
            
        return models
