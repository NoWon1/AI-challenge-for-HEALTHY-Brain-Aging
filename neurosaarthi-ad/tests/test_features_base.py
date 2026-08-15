import pandas as pd
from features.base import pivot_modality_features, add_missingness_indicators

def test_pivot_modality_features():
    data = {
        "modality": ["MRI", "MRI", "MRI", "PET", "PET"],
        "participant_id": ["P1", "P1", "P2", "P1", "P2"],
        "visit_id": ["V1", "V1", "V1", "V1", "V2"],
        "feature_name": ["vol", "thk", "vol", "suvr", "suvr"],
        "value": [1.5, 2.0, 1.6, 1.2, 1.1]
    }
    df = pd.DataFrame(data)

    # Test valid modality
    res_mri = pivot_modality_features(df, "MRI")
    assert not res_mri.empty
    assert list(res_mri.columns) == ["participant_id", "visit_id", "thk", "vol"]
    assert len(res_mri) == 2

    # P1 V1 should have both
    p1_v1 = res_mri[(res_mri["participant_id"] == "P1") & (res_mri["visit_id"] == "V1")].iloc[0]
    assert p1_v1["vol"] == 1.5
    assert p1_v1["thk"] == 2.0

    # P2 V1 should have vol, missing thk
    p2_v1 = res_mri[(res_mri["participant_id"] == "P2") & (res_mri["visit_id"] == "V1")].iloc[0]
    assert p2_v1["vol"] == 1.6
    assert pd.isna(p2_v1["thk"])

    # Test invalid modality
    res_blood = pivot_modality_features(df, "BLOOD")
    assert res_blood.empty
