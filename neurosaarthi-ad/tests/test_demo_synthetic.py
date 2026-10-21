import pandas as pd

from data_contracts.schema import validate_columns
from demo.synthetic import COHORTS, generate_demo_cohort


def test_default_demo_cohort_has_840_longitudinal_participants():
    bundle = generate_demo_cohort()
    participants = bundle.tables.participants
    visits = bundle.tables.visits

    assert len(participants) == 840
    assert set(participants["cohort"]) == set(COHORTS)
    assert participants["synthetic"].all()
    visit_counts = visits.groupby("participant_id").size()
    assert visit_counts.between(3, 6).all()
    # ⚡ Bolt: Vectorized monotonicity check avoids slow .groupby() loop
    sorted_visits = visits.sort_values("participant_id", kind="mergesort")
    p_arr = sorted_visits["participant_id"].to_numpy()
    v_arr = sorted_visits["baseline_days"].to_numpy()
    same_pid = p_arr[1:] == p_arr[:-1]
    diffs = v_arr[1:] - v_arr[:-1]
    assert bool((diffs[same_pid] >= 0).all()) if len(sorted_visits) > 0 else True

    for name, frame in {
        "participants": bundle.tables.participants,
        "visits": bundle.tables.visits,
        "modality_features": bundle.tables.modality_features,
        "outcomes": bundle.tables.outcomes,
    }.items():
        validate_columns(name, list(frame.columns))


def test_demo_generation_is_deterministic_and_records_real_conversions():
    first = generate_demo_cohort(seed=17, n_per_cohort=6)
    second = generate_demo_cohort(seed=17, n_per_cohort=6)
    pd.testing.assert_frame_equal(first.baseline, second.baseline)
    pd.testing.assert_frame_equal(first.tables.visits, second.tables.visits)

    conversions = first.harmonization_manifest["conversion"]
    assert (conversions != "identity").any()
    assert {"source_variable", "source_unit", "canonical_unit", "provenance"}.issubset(
        first.harmonization_manifest.columns
    )
