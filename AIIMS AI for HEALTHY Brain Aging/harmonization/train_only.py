"""Train-only preprocessing primitives to prevent validation leakage."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class TrainOnlyStandardizer:
    columns: list[str]
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyStandardizer":
        self.means_ = {column: float(frame[column].mean()) for column in self.columns}
        self.stds_ = {}
        for column in self.columns:
            std = float(frame[column].std(ddof=0))
            self.stds_[column] = std if std > 0 else 1.0
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TrainOnlyStandardizer must be fitted before transform")
        output = frame.copy()
        for column in self.columns:
            output[column] = (output[column] - self.means_[column]) / self.stds_[column]
        return output

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

