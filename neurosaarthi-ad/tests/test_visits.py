import pandas as pd
import pytest

from harmonization.visits import add_baseline_offsets


def test_add_baseline_offsets():
    visits = pd.DataFrame({
        "participant_id": ["A", "A", "B", "B", "C"],
        "visit_date": ["2020-01-01", "2020-01-11", "2021-05-01", "2021-05-03", "2022-01-01"]
    })

    result = add_baseline_offsets(visits)

    # Check if column is added
    assert "baseline_days" in result.columns

    # Check calculated values
    expected = [0, 10, 0, 2, 0]
    assert result["baseline_days"].tolist() == expected

def test_add_baseline_offsets_custom_columns():
    visits = pd.DataFrame({
        "pid": ["A", "A"],
        "vdate": ["2020-01-01", "2020-01-11"]
    })

    result = add_baseline_offsets(visits, participant_col="pid", date_col="vdate")
    assert result["baseline_days"].tolist() == [0, 10]

def test_add_baseline_offsets_empty():
    visits = pd.DataFrame({
        "participant_id": pd.Series([], dtype=str),
        "visit_date": pd.Series([], dtype=str)
    })

    result = add_baseline_offsets(visits)
    assert "baseline_days" in result.columns
    assert len(result) == 0

def test_add_baseline_offsets_unsorted():
    visits = pd.DataFrame({
        "participant_id": ["A", "A", "A"],
        "visit_date": ["2020-01-11", "2020-01-01", "2020-01-05"]
    })

    result = add_baseline_offsets(visits)
    expected = [10, 0, 4]
    assert result["baseline_days"].tolist() == expected
