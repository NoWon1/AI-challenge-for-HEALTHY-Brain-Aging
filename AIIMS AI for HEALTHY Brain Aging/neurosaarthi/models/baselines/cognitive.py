"""Elastic-net future cognitive-score baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neurosaarthi.core.errors import DataValidationError


class CognitivePredictionBaseline:
    """Predict a future continuous cognitive score at an explicit horizon."""

    def __init__(self, feature_columns: Sequence[str], *, random_state: int = 42) -> None:
        if "prediction_horizon_days" not in feature_columns:
            raise DataValidationError("prediction_horizon_days must be an explicit model feature")
        self.feature_columns = list(dict.fromkeys(feature_columns))
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=20_000, random_state=random_state),
                ),
            ]
        )
        self._fitted = False

    def fit(self, frame: pd.DataFrame, future_score: Sequence[float]) -> CognitivePredictionBaseline:
        X = self._features(frame)
        y = pd.to_numeric(pd.Series(future_score), errors="coerce").to_numpy(dtype=float)
        if len(y) != len(X) or not np.isfinite(y).all():
            raise DataValidationError("future_score must be finite, complete, and aligned with features")
        if len(y) < 3:
            raise DataValidationError("Cognitive regression requires at least three rows")
        self.pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("CognitivePredictionBaseline is not fitted")
        return np.asarray(self.pipeline.predict(self._features(frame)), dtype=float)

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_columns if column not in frame]
        if missing:
            raise DataValidationError(f"Missing cognitive-regression features: {', '.join(missing)}")
        horizon = pd.to_numeric(frame["prediction_horizon_days"], errors="coerce")
        if horizon.isna().any() or (horizon <= 0).any():
            raise DataValidationError("prediction_horizon_days must be finite and positive")
        return frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
