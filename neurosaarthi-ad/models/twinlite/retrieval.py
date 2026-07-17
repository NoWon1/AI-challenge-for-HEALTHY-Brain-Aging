"""Nearest-neighbor trajectory retrieval for digital twin lite demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TwinLiteRetriever:
    feature_columns: list[str]
    participant_col: str = "participant_id"

    def fit(self, frame: pd.DataFrame) -> "TwinLiteRetriever":
        self.reference_ = frame.reset_index(drop=True).copy()
        features = self.reference_[self.feature_columns].astype(float)
        self.medians_ = features.median().fillna(0.0)
        imputed = features.fillna(self.medians_)
        self.means_ = imputed.mean()
        self.stds_ = imputed.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.matrix_ = ((imputed - self.means_) / self.stds_).to_numpy()
        return self

    def query(self, row: pd.Series, k: int = 5, exclude_participant_id: str | None = None) -> pd.DataFrame:
        if not hasattr(self, "matrix_"):
            raise RuntimeError("TwinLiteRetriever must be fitted before query")
        vector = row[self.feature_columns].astype(float).fillna(self.medians_)
        vector = ((vector - self.means_) / self.stds_).to_numpy()
        distances = np.linalg.norm(self.matrix_ - vector, axis=1)
        result = self.reference_.copy()
        result["distance"] = distances
        if exclude_participant_id is not None:
            result = result[result[self.participant_col] != exclude_participant_id]
        return result.sort_values("distance").head(k).reset_index(drop=True)
