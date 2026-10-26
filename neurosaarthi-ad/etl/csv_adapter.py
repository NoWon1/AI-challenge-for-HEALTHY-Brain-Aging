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
        for table_name in ["participants", "visits", "modality_features", "outcomes"]:
            # Defense-in-depth: Prevent path traversal by extracting only the filename
            safe_name = Path(table_name).name
            path = (Path(self.raw_dir) / f"{safe_name}.csv").resolve()
            frame = pd.read_csv(path)
            validate_columns(table_name, list(frame.columns))
            tables[table_name] = frame
        return CohortTables(**tables)
