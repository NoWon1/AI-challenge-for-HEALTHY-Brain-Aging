"""Base interfaces for cohort ETL adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CohortTables:
    participants: pd.DataFrame
    visits: pd.DataFrame
    modality_features: pd.DataFrame
    outcomes: pd.DataFrame


class CohortAdapter(ABC):
    cohort_name: str

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    @abstractmethod
    def extract(self) -> CohortTables:
        """Return data mapped into the common data model."""

