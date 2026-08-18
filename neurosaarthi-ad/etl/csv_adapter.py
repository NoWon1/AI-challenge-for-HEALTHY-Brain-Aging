"""Generic CSV adapter for already-tabulated cohort exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_contracts.schema import validate_columns
from etl.base import CohortAdapter, CohortTables


class CsvCohortAdapter(CohortAdapter):
    """Load common-data-model CSV files from a folder."""

    cohort_name = "csv"

    def extract(self) -> CohortTables:
        tables: dict[str, pd.DataFrame] = {}
        base_dir = Path(self.raw_dir).resolve()
        for table_name in ["participants", "visits", "modality_features", "outcomes"]:
            path = (Path(self.raw_dir) / f"{table_name}.csv").resolve()
            if not path.is_relative_to(base_dir):
                raise PermissionError(f"Path traversal detected: {path}")
            frame = pd.read_csv(path)
            validate_columns(table_name, list(frame.columns))
            tables[table_name] = frame
        return CohortTables(**tables)

