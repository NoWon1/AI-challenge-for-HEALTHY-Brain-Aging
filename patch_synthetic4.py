import re

with open("neurosaarthi-ad/demo/synthetic.py", "r") as f:
    content = f.read()

# Add the missing _generate_event and _generate_visit_features that I must have overridden accidentally

def_content = """def _generate_event(rng: np.random.Generator, base: ParticipantBaseline) -> tuple[float, int, float, int]:
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

def _generate_visit_features(rng: np.random.Generator, base: ParticipantBaseline, decline_rate: float, year_offset: float, progressed: bool, event_time_years: float, annual_hazard: float) -> tuple[float, dict[str, float]]:
    extra_decline = -0.52 * max(0.0, year_offset - event_time_years) if progressed else 0.0
    cognitive_score = float(
        np.clip(base.baseline_cognition + decline_rate * year_offset + extra_decline + rng.normal(0, 0.22), 5.0, 30.0)
    )
    values = {
        "cognitive_score": cognitive_score,
        "memory_score": float(np.clip(base.memory_score + decline_rate * year_offset / 2.2 + rng.normal(0, 0.08), -4, 3)),
        "executive_score": float(np.clip(base.executive_score + decline_rate * year_offset / 2.6 + rng.normal(0, 0.08), -4, 3)),
        "hippocampal_volume_mm3": float(max(2800.0, base.hippocampal_volume - (32 + 35 * annual_hazard) * year_offset + rng.normal(0, 35))),
        "wmh_burden_ml": float(max(0.1, base.wmh_burden + (0.13 + 0.18 * annual_hazard) * year_offset + rng.normal(0, 0.12))),
        "entorhinal_thickness_mm": float(max(1.0, base.entorhinal_thickness - (0.02 + 0.05 * annual_hazard) * year_offset + rng.normal(0, 0.05))),
        "ventricular_volume_mm3": float(min(70000, base.ventricular_volume + (400 + 800 * annual_hazard) * year_offset + rng.normal(0, 500))),
        "cortical_thickness_mean_mm": float(max(1.0, base.cortical_thickness_mean - (0.01 + 0.03 * annual_hazard) * year_offset + rng.normal(0, 0.03))),
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

def generate_demo_cohort("""

content = content.replace("def generate_demo_cohort(", def_content)

with open("neurosaarthi-ad/demo/synthetic.py", "w") as f:
    f.write(content)
