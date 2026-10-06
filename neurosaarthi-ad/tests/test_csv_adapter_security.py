import pytest
import pandas as pd
from pathlib import Path
from etl.csv_adapter import CsvCohortAdapter
from etl.base import CohortTables
from data_contracts.schema import load_contract

def test_csv_adapter_enforces_strict_schema_allowlist(tmp_path):
    # Create raw dir
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Create valid participants table but with an extra column
    pd.DataFrame({
        "participant_id": ["P1", "P2"],
        "cohort": ["A", "B"],
        "extra_sensitive_column": ["Secret1", "Secret2"]
    }).to_csv(raw_dir / "participants.csv", index=False)

    # Create other tables with valid required columns
    pd.DataFrame({
        "visit_id": ["V1"], "participant_id": ["P1"], "cohort": ["A"],
        "visit_index": [0], "age_at_visit": [65]
    }).to_csv(raw_dir / "visits.csv", index=False)

    pd.DataFrame({
        "feature_row_id": ["F1"], "participant_id": ["P1"], "visit_id": ["V1"],
        "cohort": ["A"], "modality": ["clinical"], "feature_name": ["score"], "value": [1.0]
    }).to_csv(raw_dir / "modality_features.csv", index=False)

    pd.DataFrame({
        "outcome_id": ["O1"], "participant_id": ["P1"], "anchor_visit_id": ["V1"],
        "endpoint": ["E1"], "horizon_days": [365], "event": [1]
    }).to_csv(raw_dir / "outcomes.csv", index=False)

    adapter = CsvCohortAdapter(raw_dir)
    tables = adapter.extract()

    assert "extra_sensitive_column" not in tables.participants.columns
    assert "participant_id" in tables.participants.columns
