"""Baseline fixed-horizon risk classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    SimpleImputer = LogisticRegression = Pipeline = StandardScaler = None


@dataclass
class RiskClassifier:
    feature_columns: list[str]

    def __post_init__(self) -> None:
        self.pipeline = None
        if Pipeline is not None:
            self.pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )

    def fit(self, frame: pd.DataFrame, target_col: str = "event") -> "RiskClassifier":
        if self.pipeline is not None:
            self.pipeline.fit(frame[self.feature_columns], frame[target_col])
            return self

        x = frame[self.feature_columns].astype(float)
        y = frame[target_col].astype(int)
        self.medians_ = x.median()
        x = x.fillna(self.medians_)
        self.means_ = x.mean()
        self.stds_ = x.std(ddof=0).replace(0, 1.0)
        z = (x - self.means_) / self.stds_
        pos = z[y == 1].mean()
        neg = z[y == 0].mean()
        self.direction_ = (pos - neg).fillna(0.0)
        self.intercept_ = -float((pos + neg).fillna(0.0).dot(self.direction_) / 2.0)
        return self

    def predict_risk(self, frame: pd.DataFrame) -> pd.Series:
        if self.pipeline is not None:
            probabilities = self.pipeline.predict_proba(frame[self.feature_columns])[:, 1]
            return pd.Series(probabilities, index=frame.index, name="risk")

        if not hasattr(self, "direction_"):
            raise RuntimeError("RiskClassifier must be fitted before predict_risk")
        x = frame[self.feature_columns].astype(float).fillna(self.medians_)
        z = (x - self.means_) / self.stds_
        logits = z.dot(self.direction_) + self.intercept_
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return pd.Series(probabilities, index=frame.index, name="risk")
