import math

import pandas as pd

from models.fusion.late_fusion import weighted_score_fusion


def test_weighted_fusion_renormalises_over_present_modalities():
    scores = pd.DataFrame({"clinical": [0.2, 0.4], "mri": [0.8, float("nan")]})
    fused = weighted_score_fusion(scores, {"clinical": 0.75, "mri": 0.25})
    assert math.isclose(float(fused.iloc[0]), 0.35)
    assert math.isclose(float(fused.iloc[1]), 0.4)


def test_weighted_fusion_returns_nan_when_every_modality_is_missing():
    scores = pd.DataFrame({"clinical": [float("nan")], "mri": [float("nan")]})
    fused = weighted_score_fusion(scores, {"clinical": 0.75, "mri": 0.25})
    assert math.isnan(float(fused.iloc[0]))
