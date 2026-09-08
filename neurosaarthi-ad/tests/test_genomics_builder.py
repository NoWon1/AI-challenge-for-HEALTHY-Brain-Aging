import pandas as pd
import numpy as np
from features.genomics.builder import build_genomics_features, _add_derived_genomics

def test_add_derived_genomics_apoe():
    # Test with apoe_e4_count
    df = pd.DataFrame({
        "participant_id": ["P1", "P2", "P3", "P4"],
        "apoe_e4_count": [0, 1, 2, np.nan]
    })

    res = _add_derived_genomics(df)

    assert "apoe_e4_carrier" in res.columns
    assert "apoe_e4_homozygous" in res.columns
    assert "ad_prs_proxy" in res.columns

    # Check carrier
    assert res.loc[0, "apoe_e4_carrier"] == 0.0
    assert res.loc[1, "apoe_e4_carrier"] == 1.0
    assert res.loc[2, "apoe_e4_carrier"] == 1.0
    assert res.loc[3, "apoe_e4_carrier"] == 0.0  # NaN > 0 is False

    # Check homozygous
    assert res.loc[0, "apoe_e4_homozygous"] == 0.0
    assert res.loc[1, "apoe_e4_homozygous"] == 0.0
    assert res.loc[2, "apoe_e4_homozygous"] == 1.0
    assert res.loc[3, "apoe_e4_homozygous"] == 0.0

    # Check PRS proxy
    assert np.isclose(res.loc[0, "ad_prs_proxy"], 0.0)
    assert np.isclose(res.loc[1, "ad_prs_proxy"], 0.47)
    assert np.isclose(res.loc[2, "ad_prs_proxy"], 0.94)
    assert pd.isna(res.loc[3, "ad_prs_proxy"])

def test_add_derived_genomics_ancestry():
    # Test with ancestry_pc1
    df = pd.DataFrame({
        "participant_id": ["P1", "P2", "P3", "P4", "P5"],
        "ancestry_pc1": [0.1, -0.4, 0.6, 0.0, np.nan]
    })

    res = _add_derived_genomics(df)

    assert "indian_ancestry_flag" in res.columns
    assert "non_european_ancestry" in res.columns

    # Check indian ancestry flag (PC1 > 0.5)
    assert res.loc[0, "indian_ancestry_flag"] == 0.0
    assert res.loc[1, "indian_ancestry_flag"] == 0.0
    assert res.loc[2, "indian_ancestry_flag"] == 1.0
    assert res.loc[3, "indian_ancestry_flag"] == 0.0
    assert res.loc[4, "indian_ancestry_flag"] == 0.0 # NaN > 0.5 is False

    # Check non_european_ancestry (|PC1| > 0.3)
    assert res.loc[0, "non_european_ancestry"] == 0.0
    assert res.loc[1, "non_european_ancestry"] == 1.0
    assert res.loc[2, "non_european_ancestry"] == 1.0
    assert res.loc[3, "non_european_ancestry"] == 0.0
    assert res.loc[4, "non_european_ancestry"] == 0.0 # NaN > 0.3 is False

def test_add_derived_genomics_empty():
    # Test without expected columns
    df = pd.DataFrame({
        "participant_id": ["P1", "P2"],
        "other_feature": [1.0, 2.0]
    })

    res = _add_derived_genomics(df)

    assert "apoe_e4_carrier" not in res.columns
    assert "indian_ancestry_flag" not in res.columns
    assert list(res.columns) == ["participant_id", "other_feature"]

def test_build_genomics_features():
    # Test full pipeline with long format data
    data = {
        "modality": ["genomics", "genomics", "genomics", "MRI", "genomics"],
        "participant_id": ["P1", "P1", "P2", "P1", "P3"],
        "visit_id": ["V1", "V1", "V1", "V1", "V1"],
        "feature_name": ["apoe_e4_count", "ancestry_pc1", "apoe_e4_count", "vol", "ancestry_pc1"],
        "value": [1.0, 0.6, 2.0, 1.5, 0.1]
    }
    df = pd.DataFrame(data)

    res = build_genomics_features(df)

    # Should only process genomics
    assert "vol" not in res.columns
    assert "apoe_e4_count" in res.columns
    assert "ancestry_pc1" in res.columns

    # Derived columns should be added
    assert "apoe_e4_carrier" in res.columns
    assert "indian_ancestry_flag" in res.columns

    # Check P1 values
    p1 = res[res["participant_id"] == "P1"].iloc[0]
    assert p1["apoe_e4_count"] == 1.0
    assert p1["ancestry_pc1"] == 0.6
    assert p1["apoe_e4_carrier"] == 1.0
    assert p1["indian_ancestry_flag"] == 1.0
    assert np.isclose(p1["ad_prs_proxy"], 0.47)

    # Check P2 values
    p2 = res[res["participant_id"] == "P2"].iloc[0]
    assert p2["apoe_e4_count"] == 2.0
    assert pd.isna(p2["ancestry_pc1"])
    assert p2["apoe_e4_homozygous"] == 1.0
    assert p2["indian_ancestry_flag"] == 0.0 # NaN > 0.5 is False

    # Check P3 values
    p3 = res[res["participant_id"] == "P3"].iloc[0]
    assert pd.isna(p3["apoe_e4_count"])
    assert p3["ancestry_pc1"] == 0.1
    assert p3["apoe_e4_carrier"] == 0.0 # NaN > 0 is False
    assert p3["indian_ancestry_flag"] == 0.0
