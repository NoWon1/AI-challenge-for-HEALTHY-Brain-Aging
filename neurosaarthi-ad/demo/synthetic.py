"""Deterministic longitudinal data for the local NeuroSaarthi-AD demo.

The generator intentionally creates cohort-specific aliases, units, modality
availability, and visit cadence before mapping every record into the common
data model. Nothing in this module represents a real participant or a reported
cohort result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from etl.base import CohortTables


COHORTS = ("ADNI", "NACC", "AIBL", "OASIS", "UK Biobank", "TLSA", "SANSCOG")
PUBLIC_COHORTS = frozenset({"ADNI", "NACC", "AIBL", "OASIS", "UK Biobank"})
INDIAN_COHORTS = frozenset({"TLSA", "SANSCOG"})

FEATURE_SPECS: dict[str, tuple[str, str]] = {
    "cognitive_score": ("cognition", "points"),
    "memory_score": ("cognition", "z-score"),
    "executive_score": ("cognition", "z-score"),
    "hippocampal_volume_mm3": ("mri", "mm3"),
    "wmh_burden_ml": ("mri", "mL"),
    "hba1c_percent": ("biochem", "%"),
    "hs_crp_mg_l": ("biochem", "mg/L"),
    "rnfl_um": ("oct", "um"),
    "vessel_density_percent": ("oct", "%"),
    "apoe_e4_count": ("genomics", "alleles"),
    "ancestry_pc1": ("genomics", "z-score"),
}

SHORT_NAMES = {
    "cognitive_score": "cog_total",
    "memory_score": "memory_z",
    "executive_score": "exec_z",
    "hippocampal_volume_mm3": "hippo_vol",
    "wmh_burden_ml": "wmh",
    "hba1c_percent": "hba1c",
    "hs_crp_mg_l": "hscrp",
    "rnfl_um": "rnfl",
    "vessel_density_percent": "vessel_density",
    "apoe_e4_count": "apoe4",
    "ancestry_pc1": "pc1",
}

MISSINGNESS: dict[str, dict[str, float]] = {
    "ADNI": {"mri": 0.08, "biochem": 0.12, "oct": 0.86, "genomics": 0.24},
    "NACC": {"mri": 0.34, "biochem": 0.36, "oct": 0.91, "genomics": 0.44},
    "AIBL": {"mri": 0.14, "biochem": 0.14, "oct": 0.80, "genomics": 0.32},
    "OASIS": {"mri": 0.05, "biochem": 0.72, "oct": 0.96, "genomics": 0.82},
    "UK Biobank": {"mri": 0.34, "biochem": 0.05, "oct": 0.43, "genomics": 0.06},
    "TLSA": {"mri": 0.24, "biochem": 0.08, "oct": 0.24, "genomics": 0.34},
    "SANSCOG": {"mri": 0.54, "biochem": 0.12, "oct": 0.34, "genomics": 0.54},
}


@dataclass(frozen=True)
class DemoCohortBundle:
    """Common-model tables plus derived demo views and mapping evidence."""

    tables: CohortTables
    baseline: pd.DataFrame
    trajectories: pd.DataFrame
    harmonization_manifest: pd.DataFrame
    cohort_summary: pd.DataFrame
    seed: int


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def _cohort_tag(cohort: str) -> str:
    return cohort.lower().replace(" ", "_")


def _source_spec(cohort: str, feature: str) -> tuple[str, str, str]:
    """Return source alias, source unit, and a human-readable conversion."""

    alias = f"{_cohort_tag(cohort)}_{SHORT_NAMES[feature]}"
    canonical_unit = FEATURE_SPECS[feature][1]
    if feature == "hippocampal_volume_mm3" and cohort in {"AIBL", "OASIS"}:
        return alias, "cm3", "cm3 x 1000 -> mm3"
    if feature == "hba1c_percent" and cohort == "UK Biobank":
        return alias, "mmol/mol", "IFCC / 10.929 + 2.15 -> %"
    if feature == "rnfl_um" and cohort == "OASIS":
        return alias, "mm", "mm x 1000 -> um"
    return alias, canonical_unit, "identity"


def _round_trip_source(cohort: str, feature: str, canonical: float) -> tuple[float, str, str, str]:
    """Simulate a cohort-native value, then convert it back to canonical units."""

    alias, source_unit, conversion = _source_spec(cohort, feature)
    source_value = float(canonical)
    if feature == "hippocampal_volume_mm3" and source_unit == "cm3":
        source_value = canonical / 1000.0
        normalized = source_value * 1000.0
    elif feature == "hba1c_percent" and source_unit == "mmol/mol":
        source_value = (canonical - 2.15) * 10.929
        normalized = source_value / 10.929 + 2.15
    elif feature == "rnfl_um" and source_unit == "mm":
        source_value = canonical / 1000.0
        normalized = source_value * 1000.0
    else:
        normalized = source_value
    return float(normalized), alias, source_unit, conversion


def _diagnosis(score: float, progressed: bool) -> str:
    if score < 19.5 or (progressed and score < 22.0):
        return "dementia"
    if score < 25.5 or progressed:
        return "mci"
    return "cognitively_unimpaired"


def _participant_setting(cohort: str) -> str:
    if cohort == "TLSA":
        return "urban"
    if cohort == "SANSCOG":
        return "rural"
    return "reference"


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        for feature, (modality, canonical_unit) in FEATURE_SPECS.items():
            alias, source_unit, conversion = _source_spec(cohort, feature)
            rows.append(
                {
                    "cohort": cohort,
                    "source_variable": alias,
                    "canonical_feature": feature,
                    "modality": modality,
                    "source_unit": source_unit,
                    "canonical_unit": canonical_unit,
                    "conversion": conversion,
                    "provenance": "synthetic_demo_mapping_v1",
                }
            )
    return pd.DataFrame(rows)


def generate_demo_cohort(seed: int = 42, n_per_cohort: int = 120) -> DemoCohortBundle:
    """Create a reproducible seven-cohort longitudinal demonstration bundle.

    The default produces exactly 840 participants. ``n_per_cohort`` is exposed
    so tests and development checks can build smaller, faster cohorts.
    """

    if n_per_cohort < 4:
        raise ValueError("n_per_cohort must be at least 4")

    rng = np.random.default_rng(seed)
    participant_rows: list[dict[str, object]] = []
    visit_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    for cohort_index, cohort in enumerate(COHORTS):
        setting = _participant_setting(cohort)
        for person_index in range(n_per_cohort):
            participant_id = f"{_cohort_tag(cohort).upper()}-{person_index + 1:04d}"
            sex = "Female" if rng.random() < 0.54 else "Male"
            sex_binary = 1 if sex == "Male" else 0
            age_center = 64.0 if cohort == "UK Biobank" else 70.0
            if cohort in INDIAN_COHORTS:
                age_center -= 2.0
            age = float(np.clip(rng.normal(age_center, 7.0), 48.0, 88.0))
            education_center = 9.0 if cohort == "SANSCOG" else (12.0 if cohort == "TLSA" else 14.0)
            education = float(np.clip(rng.normal(education_center, 3.0), 0.0, 22.0))
            apoe_e4 = int(rng.choice([0, 1, 2], p=[0.69, 0.27, 0.04]))
            ancestry_pc1 = float(rng.normal(1.15 if cohort in INDIAN_COHORTS else 0.0, 0.35))
            rural_indicator = 1 if setting == "rural" else 0
            vulnerability = float(
                0.055 * (age - 65.0)
                - 0.075 * (education - 12.0)
                + 0.56 * apoe_e4
                + 0.18 * sex_binary
                + 0.28 * rural_indicator
                + rng.normal(0.0, 0.65)
            )
            baseline_cognition = float(np.clip(27.8 - 1.25 * vulnerability + rng.normal(0.0, 1.15), 15.0, 30.0))
            hippocampal_volume = float(np.clip(7100.0 - 310.0 * vulnerability - 20.0 * (age - 65) + rng.normal(0, 320), 3500, 9000))
            wmh_burden = float(np.clip(2.2 + 1.1 * vulnerability + 0.10 * (age - 60) + rng.normal(0, 1.0), 0.1, 18.0))
            hba1c = float(np.clip(5.45 + 0.17 * vulnerability + 0.18 * rural_indicator + rng.normal(0, 0.35), 4.2, 8.8))
            hs_crp = float(np.clip(np.exp(rng.normal(0.05 + 0.20 * vulnerability, 0.45)), 0.1, 12.0))
            rnfl = float(np.clip(94.0 - 2.6 * vulnerability - 0.10 * (age - 65) + rng.normal(0, 3.0), 62.0, 112.0))
            vessel_density = float(np.clip(48.5 - 1.2 * vulnerability + rng.normal(0, 2.0), 34.0, 58.0))
            memory_score = float(np.clip((baseline_cognition - 25.0) / 2.8 + rng.normal(0, 0.25), -3.0, 2.5))
            executive_score = float(np.clip((baseline_cognition - 25.0) / 3.1 + rng.normal(0, 0.3), -3.0, 2.5))

            logit_hazard = -2.75 + 0.72 * vulnerability + 0.18 * (25.0 - baseline_cognition) + 0.035 * wmh_burden
            annual_hazard = float(np.clip(_sigmoid(logit_hazard), 0.018, 0.62))
            sampled_event_year = float(rng.geometric(annual_hazard) - rng.uniform(0.05, 0.65))
            event = int(sampled_event_year <= 5.0)
            event_time_years = sampled_event_year if event else float(rng.uniform(5.05, 5.8))
            event_time_days = int(round(event_time_years * 365.25))

            modality_available = {
                modality: rng.random() >= MISSINGNESS[cohort][modality]
                for modality in ("mri", "biochem", "oct", "genomics")
            }
            n_visits = int(rng.integers(3, 7))
            start_date = date(2015 + int(rng.integers(0, 5)), 1 + int(rng.integers(0, 12)), 1 + int(rng.integers(0, 25)))

            participant_rows.append(
                {
                    "participant_id": participant_id,
                    "cohort": cohort,
                    "sex": sex,
                    "birth_year": int(start_date.year - age),
                    "education_years": round(education, 1),
                    "language": "Kannada" if cohort in INDIAN_COHORTS else "cohort_recorded",
                    "urban_rural": setting,
                    "synthetic": True,
                }
            )

            baseline_features: dict[str, float | None] = {}
            decline_rate = float(-0.14 - 0.55 * annual_hazard - 0.10 * max(vulnerability, 0.0))
            for visit_index in range(n_visits):
                year_offset = float(visit_index + rng.normal(0.0, 0.06)) if visit_index else 0.0
                baseline_days = int(round(max(year_offset, 0.0) * 365.25))
                visit_id = f"{participant_id}-V{visit_index}"
                progressed = bool(event and year_offset >= event_time_years)
                extra_decline = -0.52 * max(0.0, year_offset - event_time_years) if progressed else 0.0
                cognitive_score = float(
                    np.clip(baseline_cognition + decline_rate * year_offset + extra_decline + rng.normal(0, 0.22), 5.0, 30.0)
                )
                values = {
                    "cognitive_score": cognitive_score,
                    "memory_score": float(np.clip(memory_score + decline_rate * year_offset / 2.2 + rng.normal(0, 0.08), -4, 3)),
                    "executive_score": float(np.clip(executive_score + decline_rate * year_offset / 2.6 + rng.normal(0, 0.08), -4, 3)),
                    "hippocampal_volume_mm3": float(max(2800.0, hippocampal_volume - (32 + 35 * annual_hazard) * year_offset + rng.normal(0, 35))),
                    "wmh_burden_ml": float(max(0.1, wmh_burden + (0.13 + 0.18 * annual_hazard) * year_offset + rng.normal(0, 0.12))),
                    "hba1c_percent": float(np.clip(hba1c + rng.normal(0, 0.08), 4.0, 10.0)),
                    "hs_crp_mg_l": float(np.clip(hs_crp * rng.lognormal(0, 0.08), 0.05, 15.0)),
                    "rnfl_um": float(np.clip(rnfl - 0.16 * year_offset + rng.normal(0, 0.4), 55, 115)),
                    "vessel_density_percent": float(np.clip(vessel_density - 0.07 * year_offset + rng.normal(0, 0.3), 30, 60)),
                    "apoe_e4_count": float(apoe_e4),
                    "ancestry_pc1": ancestry_pc1,
                }
                for feature, (modality, canonical_unit) in FEATURE_SPECS.items():
                    available = modality == "cognition" or modality_available.get(modality, True)
                    if visit_index > 0 and modality in {"mri", "oct", "biochem"}:
                        available = available and rng.random() > 0.08
                    if visit_index > 0 and modality == "genomics":
                        available = False
                    if not available:
                        if visit_index == 0:
                            baseline_features[feature] = None
                        continue
                    normalized, source_variable, source_unit, conversion = _round_trip_source(cohort, feature, values[feature])
                    if visit_index == 0:
                        baseline_features[feature] = normalized
                    feature_rows.append(
                        {
                            "feature_row_id": f"{visit_id}-{feature}",
                            "participant_id": participant_id,
                            "visit_id": visit_id,
                            "cohort": cohort,
                            "modality": modality,
                            "feature_name": feature,
                            "value": normalized,
                            "unit": canonical_unit,
                            "source_variable": source_variable,
                            "source_unit": source_unit,
                            "conversion": conversion,
                            "qc_flag": "pass",
                            "derived": feature in {"memory_score", "executive_score", "ancestry_pc1"},
                        }
                    )

                diagnosis = _diagnosis(cognitive_score, progressed)
                visit_rows.append(
                    {
                        "visit_id": visit_id,
                        "participant_id": participant_id,
                        "cohort": cohort,
                        "visit_index": visit_index,
                        "age_at_visit": age + max(year_offset, 0.0),
                        "visit_date": start_date + timedelta(days=baseline_days),
                        "baseline_days": baseline_days,
                        "diagnosis": diagnosis,
                        "cdr_global": 0.0 if diagnosis == "cognitively_unimpaired" else (0.5 if diagnosis == "mci" else 1.0),
                        "cognitive_status": diagnosis,
                    }
                )
                trajectory_rows.append(
                    {
                        "participant_id": participant_id,
                        "cohort": cohort,
                        "urban_rural": setting,
                        "year": max(year_offset, 0.0),
                        "visit_index": visit_index,
                        "cognitive_score": cognitive_score,
                        "diagnosis": diagnosis,
                    }
                )

            for horizon in (1, 3, 5):
                event_by_horizon = int(event and event_time_years <= horizon)
                outcome_rows.append(
                    {
                        "outcome_id": f"{participant_id}-risk-{horizon}y",
                        "participant_id": participant_id,
                        "anchor_visit_id": f"{participant_id}-V0",
                        "endpoint": f"incident_progression_{horizon}y",
                        "horizon_days": int(round(horizon * 365.25)),
                        "event": event_by_horizon,
                        "event_time_days": event_time_days,
                        "future_score": np.nan,
                        "censoring_reason": "study_end" if not event else "",
                    }
                )

            baseline_rows.append(
                {
                    "participant_id": participant_id,
                    "cohort": cohort,
                    "urban_rural": setting,
                    "sex": sex,
                    "sex_binary": sex_binary,
                    "age": age,
                    "education_years": education,
                    "event": event,
                    "event_time_days": event_time_days,
                    "event_by_1y": int(event and event_time_years <= 1),
                    "event_by_3y": int(event and event_time_years <= 3),
                    "event_by_5y": int(event and event_time_years <= 5),
                    "annual_hazard_latent": annual_hazard,
                    **{feature: baseline_features.get(feature) for feature in FEATURE_SPECS},
                }
            )

    participants = pd.DataFrame(participant_rows)
    visits = pd.DataFrame(visit_rows).sort_values(["participant_id", "visit_index"]).reset_index(drop=True)
    modality_features = pd.DataFrame(feature_rows).sort_values(["participant_id", "visit_id", "feature_name"]).reset_index(drop=True)
    outcomes = pd.DataFrame(outcome_rows).sort_values(["participant_id", "horizon_days"]).reset_index(drop=True)
    baseline = pd.DataFrame(baseline_rows).sort_values(["cohort", "participant_id"]).reset_index(drop=True)
    trajectories = pd.DataFrame(trajectory_rows).sort_values(["participant_id", "year"]).reset_index(drop=True)

    summary_rows = []
    for cohort, group in baseline.groupby("cohort", sort=False):
        summary_rows.append(
            {
                "cohort": cohort,
                "participants": len(group),
                "role": "Development" if cohort in PUBLIC_COHORTS else ("India adaptation" if cohort == "TLSA" else "External India validation"),
                "setting": _participant_setting(cohort),
                "five_year_event_rate": float(group["event_by_5y"].mean()),
            }
        )

    return DemoCohortBundle(
        tables=CohortTables(
            participants=participants,
            visits=visits,
            modality_features=modality_features,
            outcomes=outcomes,
        ),
        baseline=baseline,
        trajectories=trajectories,
        harmonization_manifest=_manifest(),
        cohort_summary=pd.DataFrame(summary_rows),
        seed=seed,
    )
