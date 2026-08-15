import pandas as pd
import pytest

from harmonization.visits import add_baseline_offsets, require_monotonic_visits


def test_require_monotonic_visits_happy_path():
    df = pd.DataFrame({"participant_id": [1, 1, 2, 2], "visit_index": [1, 2, 1, 2]})
    # Should not raise
    require_monotonic_visits(df)


def test_require_monotonic_visits_raises():
    df = pd.DataFrame({"participant_id": [1, 1, 1], "visit_index": [1, 3, 2]})
    with pytest.raises(ValueError, match="Visits are not monotonic for participant 1"):
        require_monotonic_visits(df)


def test_require_monotonic_visits_multiple_participants_raises():
    df = pd.DataFrame({"participant_id": [1, 1, 2, 2, 2], "visit_index": [1, 2, 1, 3, 2]})
    with pytest.raises(ValueError, match="Visits are not monotonic for participant 2"):
        require_monotonic_visits(df)


def test_require_monotonic_visits_empty():
    df = pd.DataFrame(columns=["participant_id", "visit_index"])
    # Should not raise
    require_monotonic_visits(df)


def test_add_baseline_offsets():
    visits = pd.DataFrame(
        {
            "participant_id": ["A", "A", "B", "B", "C"],
            "visit_date": ["2020-01-01", "2020-01-11", "2021-05-01", "2021-05-03", "2022-01-01"],
        }
    )

    result = add_baseline_offsets(visits)

    # Check if column is added
    assert "baseline_days" in result.columns

    # Check calculated values
    expected = [0, 10, 0, 2, 0]
    assert result["baseline_days"].tolist() == expected


def test_add_baseline_offsets_custom_columns():
    visits = pd.DataFrame({"pid": ["A", "A"], "vdate": ["2020-01-01", "2020-01-11"]})

    result = add_baseline_offsets(visits, participant_col="pid", date_col="vdate")
    assert result["baseline_days"].tolist() == [0, 10]


def test_add_baseline_offsets_empty():
    visits = pd.DataFrame(
        {"participant_id": pd.Series([], dtype=str), "visit_date": pd.Series([], dtype=str)}
    )

    result = add_baseline_offsets(visits)
    assert "baseline_days" in result.columns
    assert len(result) == 0


def test_add_baseline_offsets_unsorted():
    visits = pd.DataFrame(
        {"participant_id": ["A", "A", "A"], "visit_date": ["2020-01-11", "2020-01-01", "2020-01-05"]}
    )

    result = add_baseline_offsets(visits)
    expected = [10, 0, 4]
    assert result["baseline_days"].tolist() == expected
