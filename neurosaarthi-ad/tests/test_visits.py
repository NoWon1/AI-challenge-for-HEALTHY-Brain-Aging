import pandas as pd
import pytest
from harmonization.visits import require_monotonic_visits

def test_require_monotonic_visits_happy_path():
    df = pd.DataFrame({
        'participant_id': [1, 1, 2, 2],
        'visit_index': [1, 2, 1, 2]
    })
    # Should not raise
    require_monotonic_visits(df)

def test_require_monotonic_visits_raises():
    df = pd.DataFrame({
        'participant_id': [1, 1, 1],
        'visit_index': [1, 3, 2]
    })
    with pytest.raises(ValueError, match="Visits are not monotonic for participant 1"):
        require_monotonic_visits(df)

def test_require_monotonic_visits_multiple_participants_raises():
    df = pd.DataFrame({
        'participant_id': [1, 1, 2, 2, 2],
        'visit_index': [1, 2, 1, 3, 2]
    })
    with pytest.raises(ValueError, match="Visits are not monotonic for participant 2"):
        require_monotonic_visits(df)

def test_require_monotonic_visits_empty():
    df = pd.DataFrame(columns=['participant_id', 'visit_index'])
    # Should not raise
    require_monotonic_visits(df)
