import pandas as pd
import pytest

from neurosaarthi.core.errors import DataValidationError
from neurosaarthi.data.splitters.longitudinal import (
    assert_no_future_features,
    assert_participant_isolation,
    group_shuffle_split,
)


def test_subjects_never_cross_train_and_test():
    frame = pd.DataFrame(
        {
            "participant_id_internal": [f"P-{index // 2}" for index in range(20)],
            "visit": list(range(20)),
        }
    )
    train, test = group_shuffle_split(frame, test_size=0.3, seed=9)
    assert_participant_isolation(train, test)


def test_feature_acquired_after_prediction_origin_is_rejected():
    frame = pd.DataFrame(
        {
            "feature_time_days": [0, 400],
            "prediction_origin_days": [0, 365],
        }
    )
    with pytest.raises(DataValidationError, match="Future-feature leakage"):
        assert_no_future_features(frame)


def test_features_at_origin_are_allowed():
    frame = pd.DataFrame({"feature_time_days": [0, 100, 365], "prediction_origin_days": [0, 200, 365]})
    assert_no_future_features(frame)
