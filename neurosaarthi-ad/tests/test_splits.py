import pandas as pd

from evaluation.splits import subject_split
from harmonization.leakage import assert_disjoint_participants


def test_subject_split_has_no_participant_overlap():
    frame = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "value": range(8),
        }
    )
    train, test = subject_split(frame, test_size=0.25, seed=7)
    assert_disjoint_participants(set(train["participant_id"]), set(test["participant_id"]))

