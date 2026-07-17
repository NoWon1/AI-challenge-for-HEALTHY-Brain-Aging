"""Baseline cognitive trajectory model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    SimpleImputer = Ridge = Pipeline = StandardScaler = None


@dataclass
class CognitiveTrajectoryRegressor:
    feature_columns: list[str]

    def __post_init__(self) -> None:
        self.pipeline = None
        if Pipeline is not None:
            self.pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            )

    def fit(self, frame: pd.DataFrame, target_col: str = "future_score") -> "CognitiveTrajectoryRegressor":
        if self.pipeline is not None:
            self.pipeline.fit(frame[self.feature_columns], frame[target_col])
            return self

        x = frame[self.feature_columns].astype(float)
        y = frame[target_col].astype(float).to_numpy()
        self.medians_ = x.median()
        x = x.fillna(self.medians_)
        self.means_ = x.mean()
        self.stds_ = x.std(ddof=0).replace(0, 1.0)
        z = (x - self.means_) / self.stds_
        design = np.column_stack([np.ones(len(z)), z.to_numpy()])
        penalty = np.eye(design.shape[1]) * 1.0
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.pipeline is not None:
            predictions = self.pipeline.predict(frame[self.feature_columns])
            return pd.Series(predictions, index=frame.index, name="predicted_future_score")

        if not hasattr(self, "coef_"):
            raise RuntimeError("CognitiveTrajectoryRegressor must be fitted before predict")
        x = frame[self.feature_columns].astype(float).fillna(self.medians_)
        z = (x - self.means_) / self.stds_
        design = np.column_stack([np.ones(len(z)), z.to_numpy()])
        predictions = design @ self.coef_
        return pd.Series(predictions, index=frame.index, name="predicted_future_score")
