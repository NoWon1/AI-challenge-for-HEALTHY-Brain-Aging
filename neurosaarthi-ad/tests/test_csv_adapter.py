import pytest
from pathlib import Path
import pandas as pd
from etl.csv_adapter import CsvCohortAdapter

@pytest.fixture
def valid_csv_dir(tmp_path):
    # Create the 4 required CSV files with the required columns
    (tmp_path / "participants.csv").write_text("participant_id,cohort\nP01,csv_cohort\n")
    (tmp_path / "visits.csv").write_text("visit_id,participant_id,cohort,visit_index,age_at_visit\nV01,P01,csv_cohort,0,70.5\n")
    (tmp_path / "modality_features.csv").write_text("feature_row_id,participant_id,visit_id,cohort,modality,feature_name,value\nF01,P01,V01,csv_cohort,MRI,vol,100\n")
    (tmp_path / "outcomes.csv").write_text("outcome_id,participant_id,anchor_visit_id,endpoint,horizon_days,event\nO01,P01,V01,progression,365,0\n")
    return tmp_path

def test_csv_adapter_extract_success(valid_csv_dir):
    adapter = CsvCohortAdapter(valid_csv_dir)
    tables = adapter.extract()

    assert tables is not None
    assert isinstance(tables.participants, pd.DataFrame)
    assert len(tables.participants) == 1
    assert list(tables.participants.columns) == ["participant_id", "cohort"]

    assert isinstance(tables.visits, pd.DataFrame)
    assert len(tables.visits) == 1

    assert isinstance(tables.modality_features, pd.DataFrame)
    assert len(tables.modality_features) == 1

    assert isinstance(tables.outcomes, pd.DataFrame)
    assert len(tables.outcomes) == 1


def test_csv_adapter_missing_file(tmp_path):
    # Only create one file to trigger FileNotFoundError on the next one
    (tmp_path / "participants.csv").write_text("participant_id,cohort\nP01,csv_cohort\n")

    adapter = CsvCohortAdapter(tmp_path)
    with pytest.raises(FileNotFoundError):
        adapter.extract()


def test_csv_adapter_missing_column(tmp_path):
    # missing 'cohort' from participants
    (tmp_path / "participants.csv").write_text("participant_id\nP01\n")

    adapter = CsvCohortAdapter(tmp_path)
    with pytest.raises(ValueError, match="participants is missing required columns: cohort"):
        adapter.extract()
