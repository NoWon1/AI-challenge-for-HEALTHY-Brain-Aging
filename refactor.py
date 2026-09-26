import re

with open("neurosaarthi-ad/demo/synthetic.py", "r") as f:
    content = f.read()

# Let's insert some helper classes and methods before generate_demo_cohort
helpers = """
@dataclass
class ParticipantBaseline:
    sex: str
    sex_binary: int
    age: float
    education: float
    apoe_e4: int
    ancestry_pc1: float
    vulnerability: float
    baseline_cognition: float
    hippocampal_volume: float
    wmh_burden: float
    hba1c: float
    hs_crp: float
    rnfl: float
    vessel_density: float
    memory_score: float
    executive_score: float
    entorhinal_thickness: float
    ventricular_volume: float
    cortical_thickness_mean: float
    total_cholesterol: float
    fasting_glucose: float
    gfaz_area: float

def _generate_baseline_traits(rng: np.random.Generator, cohort: str, setting: str) -> ParticipantBaseline:
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
    entorhinal_thickness = float(np.clip(3.6 - 0.12 * vulnerability - 0.008 * (age - 65) + rng.normal(0, 0.15), 1.5, 4.5))
    ventricular_volume = float(np.clip(28000 + 3400 * vulnerability + 200 * (age - 65) + rng.normal(0, 3000), 12000, 65000))
    cortical_thickness_mean = float(np.clip(2.65 - 0.04 * vulnerability - 0.003 * (age - 65) + rng.normal(0, 0.08), 1.8, 3.2))
    total_cholesterol = float(np.clip(195 + 5 * vulnerability + rng.normal(0, 25), 110, 320))
    fasting_glucose = float(np.clip(95 + 4 * vulnerability + 6 * rural_indicator + rng.normal(0, 12), 65, 200))
    gfaz_area = float(np.clip(0.27 + 0.015 * vulnerability + rng.normal(0, 0.04), 0.1, 0.6))

    return ParticipantBaseline(
        sex=sex, sex_binary=sex_binary, age=age, education=education, apoe_e4=apoe_e4,
        ancestry_pc1=ancestry_pc1, vulnerability=vulnerability, baseline_cognition=baseline_cognition,
        hippocampal_volume=hippocampal_volume, wmh_burden=wmh_burden, hba1c=hba1c, hs_crp=hs_crp,
        rnfl=rnfl, vessel_density=vessel_density, memory_score=memory_score, executive_score=executive_score,
        entorhinal_thickness=entorhinal_thickness, ventricular_volume=ventricular_volume,
        cortical_thickness_mean=cortical_thickness_mean, total_cholesterol=total_cholesterol,
        fasting_glucose=fasting_glucose, gfaz_area=gfaz_area
    )

def _generate_event(rng: np.random.Generator, base: ParticipantBaseline) -> tuple[float, int, float, int]:
    logit_hazard = (
        -2.75 + 0.72 * base.vulnerability + 0.18 * (25.0 - base.baseline_cognition) + 0.035 * base.wmh_burden
        + 0.025 * base.apoe_e4 * max(0, base.age - 65)
        + 0.08 * max(0, base.hba1c - 5.7) * max(0, base.wmh_burden - 3.0)
        - 0.15 * max(0, base.education - 14)
        + 0.04 * max(0, 90 - base.rnfl)
    )
    annual_hazard = float(np.clip(_sigmoid(logit_hazard), 0.018, 0.62))
    sampled_event_year = float(rng.geometric(annual_hazard) - rng.uniform(0.05, 0.65))
    event = int(sampled_event_year <= 5.0)
    event_time_years = sampled_event_year if event else float(rng.uniform(5.05, 5.8))
    event_time_days = int(round(event_time_years * 365.25))
    return annual_hazard, event, event_time_years, event_time_days

def _generate_visit_features(rng: np.random.Generator, base: ParticipantBaseline, decline_rate: float, year_offset: float, progressed: bool, event_time_years: float) -> tuple[float, dict[str, float]]:
    extra_decline = -0.52 * max(0.0, year_offset - event_time_years) if progressed else 0.0
    cognitive_score = float(
        np.clip(base.baseline_cognition + decline_rate * year_offset + extra_decline + rng.normal(0, 0.22), 5.0, 30.0)
    )
    values = {
        "cognitive_score": cognitive_score,
        "memory_score": float(np.clip(base.memory_score + decline_rate * year_offset / 2.2 + rng.normal(0, 0.08), -4, 3)),
        "executive_score": float(np.clip(base.executive_score + decline_rate * year_offset / 2.6 + rng.normal(0, 0.08), -4, 3)),
        "hippocampal_volume_mm3": float(max(2800.0, base.hippocampal_volume - (32 + 35 * base.annual_hazard) * year_offset + rng.normal(0, 35))), # Wait, this needs annual_hazard which is not in base. I'll add annual_hazard to _generate_visit_features args
        "wmh_burden_ml": float(max(0.1, base.wmh_burden + (0.13 + 0.18 * base.annual_hazard) * year_offset + rng.normal(0, 0.12))),
        "entorhinal_thickness_mm": float(max(1.0, base.entorhinal_thickness - (0.02 + 0.05 * base.annual_hazard) * year_offset + rng.normal(0, 0.05))),
        "ventricular_volume_mm3": float(min(70000, base.ventricular_volume + (400 + 800 * base.annual_hazard) * year_offset + rng.normal(0, 500))),
        "cortical_thickness_mean_mm": float(max(1.0, base.cortical_thickness_mean - (0.01 + 0.03 * base.annual_hazard) * year_offset + rng.normal(0, 0.03))),
        "hba1c_percent": float(np.clip(base.hba1c + rng.normal(0, 0.08), 4.0, 10.0)),
        "hs_crp_mg_l": float(np.clip(base.hs_crp * rng.lognormal(0, 0.08), 0.05, 15.0)),
        "total_cholesterol_mg_dl": float(np.clip(base.total_cholesterol + rng.normal(0, 5), 100, 350)),
        "fasting_glucose_mg_dl": float(np.clip(base.fasting_glucose + rng.normal(0, 3), 60, 220)),
        "rnfl_um": float(np.clip(base.rnfl - 0.16 * year_offset + rng.normal(0, 0.4), 55, 115)),
        "vessel_density_percent": float(np.clip(base.vessel_density - 0.07 * year_offset + rng.normal(0, 0.3), 30, 60)),
        "gfaz_area_mm2": float(np.clip(base.gfaz_area + 0.005 * year_offset + rng.normal(0, 0.01), 0.05, 0.7)),
        "apoe_e4_count": float(base.apoe_e4),
        "ancestry_pc1": base.ancestry_pc1,
    }
    return cognitive_score, values

def generate_demo_cohort(
"""

# Let's use sed/awk to replace it safely, or a proper python string replace script.
