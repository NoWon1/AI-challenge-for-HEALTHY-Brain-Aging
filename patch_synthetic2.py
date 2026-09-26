import re

with open("neurosaarthi-ad/demo/synthetic.py", "r") as f:
    content = f.read()

# Let's insert the new classes and functions before the generate_demo_cohort method

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

def generate_demo_cohort(seed: int = 42, n_per_cohort: int = 120) -> DemoCohortBundle:"""

content = content.replace("def generate_demo_cohort(seed: int = 42, n_per_cohort: int = 120) -> DemoCohortBundle:", helpers)

# Now we need to replace the long body of generate_demo_cohort

body_regex_to_replace = r"""            sex = "Female" if rng\.random\(\) < 0\.54 else "Male"
.*?
            gfaz_area = float\(np\.clip\(0\.27 \+ 0\.015 \* vulnerability \+ rng\.normal\(0, 0\.04\), 0\.1, 0\.6\)\)"""

replacement_for_body = """            base = _generate_baseline_traits(rng, cohort, setting)"""

content = re.sub(body_regex_to_replace, replacement_for_body, content, flags=re.DOTALL)

with open("neurosaarthi-ad/demo/synthetic.py", "w") as f:
    f.write(content)
