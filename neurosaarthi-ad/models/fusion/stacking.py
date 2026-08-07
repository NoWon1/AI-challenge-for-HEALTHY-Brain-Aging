"""Stacked ensemble for modality fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    LogisticRegression = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


@dataclass
class StackedFusionModel:
    """A 2-level stacked ensemble for multimodal fusion."""
    
    base_model_names: list[str]
    meta_learner_type: str = "logistic"
    n_folds: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        if self.meta_learner_type == "logistic" and LogisticRegression is None:
            raise ImportError("scikit-learn is required for StackedFusionModel with logistic regression")
        if self.meta_learner_type == "lightgbm" and lgb is None:
            raise ImportError("lightgbm is required for StackedFusionModel with lightgbm")

    def fit(self, base_predictions_dict: dict[str, np.ndarray | pd.Series], y_true: np.ndarray | pd.Series, participant_ids: np.ndarray | pd.Series) -> "StackedFusionModel":
        """Fit the meta-learner on out-of-fold base model predictions.
        
        It is expected that `base_predictions_dict` contains out-of-fold predictions
        gathered from training the base models on k-1 folds and predicting on the held-out fold.
        """
        X = pd.DataFrame(base_predictions_dict)[self.base_model_names]
        y = np.asarray(y_true)
        
        if self.meta_learner_type == "logistic":
            self._medians = X.median()
            X_imputed = X.fillna(self._medians)
            self.meta_learner_ = LogisticRegression(random_state=self.seed)
            self.meta_learner_.fit(X_imputed, y)
        elif self.meta_learner_type == "lightgbm":
            self.meta_learner_ = lgb.LGBMClassifier(random_state=self.seed)
            self.meta_learner_.fit(X, y)
        else:
            raise ValueError(f"Unknown meta_learner_type: {self.meta_learner_type}")
        
        return self

    def predict(self, base_predictions_dict: dict[str, np.ndarray | pd.Series]) -> pd.Series:
        """Predict the final fused score using the meta-learner."""
        X = pd.DataFrame(base_predictions_dict)[self.base_model_names]
        if self.meta_learner_type == "logistic":
            X = X.fillna(self._medians)
            
        preds = self.meta_learner_.predict_proba(X)[:, 1]
        return pd.Series(preds, index=X.index, name="fused_score")
        
    def learned_weights(self) -> dict[str, float]:
        """Return the meta-learner's learned modality weights."""
        if self.meta_learner_type == "logistic":
            return dict(zip(self.base_model_names, self.meta_learner_.coef_[0]))
        elif self.meta_learner_type == "lightgbm":
            return dict(zip(self.base_model_names, self.meta_learner_.feature_importances_))
        return {}
