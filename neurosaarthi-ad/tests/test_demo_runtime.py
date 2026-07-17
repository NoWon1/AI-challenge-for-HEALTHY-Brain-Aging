import numpy as np
import pytest

from demo.runtime import PRESET_PROFILES, build_demo_runtime, profile_with
from demo.synthetic import generate_demo_cohort


@pytest.fixture(scope="module")
def runtime():
    return build_demo_runtime(generate_demo_cohort(seed=9, n_per_cohort=36), n_bootstrap=3, seed=9)


def test_global_tlsa_and_sanscog_roles_are_isolated(runtime):
    assert set(runtime.train["role"]) == {"global_train", "tlsa_adaptation"}
    assert "SANSCOG" not in set(runtime.train["cohort"])
    assert set(runtime.baseline.loc[runtime.baseline["role"] == "external_validation", "cohort"]) == {"SANSCOG"}
    assert set(runtime.train["participant_id"]).isdisjoint(set(runtime.validation["participant_id"]))


def test_prediction_is_monotonic_bounded_and_returns_twins(runtime):
    profile = PRESET_PROFILES["Case C · Multimodal high-risk profile"]
    forecast = runtime.predict(profile)
    risks = forecast.risks

    assert list(risks["horizon"]) == [1, 3, 5]
    assert np.all(np.diff(risks["risk"]) >= -1e-12)
    assert ((risks[["risk", "lower", "upper"]]) >= 0).all().all()
    assert ((risks[["risk", "lower", "upper"]]) <= 1).all().all()
    assert (risks["lower"] <= risks["risk"]).all()
    assert (risks["risk"] <= risks["upper"]).all()
    assert len(forecast.twins) == 5
    assert profile.participant_id not in set(forecast.twins["participant_id"])
    assert list(forecast.trajectory["year"]) == [0.0, 1.0, 2.0, 3.0, 5.0]


def test_missing_modalities_are_renormalised_instead_of_failing(runtime):
    base = PRESET_PROFILES["Case A · Resilient urban profile"]
    sparse = profile_with(
        base,
        hippocampal_volume_mm3=None,
        wmh_burden_ml=None,
        hba1c_percent=None,
        hs_crp_mg_l=None,
        rnfl_um=None,
        vessel_density_percent=None,
        apoe_e4_count=None,
        ancestry_pc1=None,
    )
    forecast = runtime.predict(sparse)
    assert forecast.risks["risk"].notna().all()
    assert forecast.available_modalities == ("Cognition + clinical",)
    assert any("renormalised" in warning for warning in forecast.warnings)
    assert any("Limited multimodal coverage" in warning for warning in forecast.warnings)


def test_validation_outputs_are_prediction_derived(runtime):
    assert set(runtime.validation_summary["validation_set"]) == {
        "Held-out public cohorts",
        "TLSA adaptation check",
        "SANSCOG external validation",
    }
    assert runtime.validation_summary[["auroc", "auprc", "brier"]].notna().all().all()
    assert len(runtime.calibration) >= 2
    assert (runtime.quality_checks["status"] == "Passed").all()
    assert "Full multimodal" in set(runtime.ablation["scenario"])
