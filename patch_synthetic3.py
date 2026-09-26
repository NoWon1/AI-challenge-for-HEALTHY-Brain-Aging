import re

with open("neurosaarthi-ad/demo/synthetic.py", "r") as f:
    content = f.read()

# Fix the hazard calculation and other unextracted variables that were relying on locals
body_regex_to_replace = r"""            logit_hazard = \(
                -2\.75 \+ 0\.72 \* vulnerability \+ 0\.18 \* \(25\.0 - baseline_cognition\) \+ 0\.035 \* wmh_burden
                \+ 0\.025 \* apoe_e4 \* max\(0, age - 65\)
                \+ 0\.08 \* max\(0, hba1c - 5\.7\) \* max\(0, wmh_burden - 3\.0\)
                - 0\.15 \* max\(0, education - 14\)
                \+ 0\.04 \* max\(0, 90 - rnfl\)
            \)
            annual_hazard = float\(np\.clip\(_sigmoid\(logit_hazard\), 0\.018, 0\.62\)\)
            sampled_event_year = float\(rng\.geometric\(annual_hazard\) - rng\.uniform\(0\.05, 0\.65\)\)
            event = int\(sampled_event_year <= 5\.0\)
            event_time_years = sampled_event_year if event else float\(rng\.uniform\(5\.05, 5\.8\)\)
            event_time_days = int\(round\(event_time_years \* 365\.25\)\)

            modality_available = \{
                modality: rng\.random\(\) >= MISSINGNESS\[cohort\]\[modality\]
                for modality in \("mri", "biochem", "oct", "genomics"\)
            \}
            n_visits = int\(rng\.integers\(3, 7\)\)
            start_date = date\(2015 \+ int\(rng\.integers\(0, 5\)\), 1 \+ int\(rng\.integers\(0, 12\)\), 1 \+ int\(rng\.integers\(0, 25\)\)\)

            participant_rows\.append\(
                \{
                    "participant_id": participant_id,
                    "cohort": cohort,
                    "sex": sex,
                    "birth_year": int\(start_date\.year - age\),
                    "education_years": round\(education, 1\),
                    "language": "Kannada" if cohort in INDIAN_COHORTS else "cohort_recorded",
                    "urban_rural": setting,
                    "synthetic": True,
                \}
            \)

            baseline_features: dict\[str, float \| None\] = \{\}
            decline_rate = float\(-0\.14 - 0\.55 \* annual_hazard - 0\.10 \* max\(vulnerability, 0\.0\)\)
            for visit_index in range\(n_visits\):
                year_offset = float\(visit_index \+ rng\.normal\(0\.0, 0\.06\)\) if visit_index else 0\.0
                baseline_days = int\(round\(max\(year_offset, 0\.0\) \* 365\.25\)\)
                visit_id = f"\{participant_id\}-V\{visit_index\}"
                progressed = bool\(event and year_offset >= event_time_years\)
                extra_decline = -0\.52 \* max\(0\.0, year_offset - event_time_years\) if progressed else 0\.0
                cognitive_score = float\(
                    np\.clip\(baseline_cognition \+ decline_rate \* year_offset \+ extra_decline \+ rng\.normal\(0, 0\.22\), 5\.0, 30\.0\)
                \)
                values = \{
                    "cognitive_score": cognitive_score,
                    "memory_score": float\(np\.clip\(memory_score \+ decline_rate \* year_offset / 2\.2 \+ rng\.normal\(0, 0\.08\), -4, 3\)\),
                    "executive_score": float\(np\.clip\(executive_score \+ decline_rate \* year_offset / 2\.6 \+ rng\.normal\(0, 0\.08\), -4, 3\)\),
                    "hippocampal_volume_mm3": float\(max\(2800\.0, hippocampal_volume - \(32 \+ 35 \* annual_hazard\) \* year_offset \+ rng\.normal\(0, 35\)\)\),
                    "wmh_burden_ml": float\(max\(0\.1, wmh_burden \+ \(0\.13 \+ 0\.18 \* annual_hazard\) \* year_offset \+ rng\.normal\(0, 0\.12\)\)\),
                    "entorhinal_thickness_mm": float\(max\(1\.0, entorhinal_thickness - \(0\.02 \+ 0\.05 \* annual_hazard\) \* year_offset \+ rng\.normal\(0, 0\.05\)\)\),
                    "ventricular_volume_mm3": float\(min\(70000, ventricular_volume \+ \(400 \+ 800 \* annual_hazard\) \* year_offset \+ rng\.normal\(0, 500\)\)\),
                    "cortical_thickness_mean_mm": float\(max\(1\.0, cortical_thickness_mean - \(0\.01 \+ 0\.03 \* annual_hazard\) \* year_offset \+ rng\.normal\(0, 0\.03\)\)\),
                    "hba1c_percent": float\(np\.clip\(hba1c \+ rng\.normal\(0, 0\.08\), 4\.0, 10\.0\)\),
                    "hs_crp_mg_l": float\(np\.clip\(hs_crp \* rng\.lognormal\(0, 0\.08\), 0\.05, 15\.0\)\),
                    "total_cholesterol_mg_dl": float\(np\.clip\(total_cholesterol \+ rng\.normal\(0, 5\), 100, 350\)\),
                    "fasting_glucose_mg_dl": float\(np\.clip\(fasting_glucose \+ rng\.normal\(0, 3\), 60, 220\)\),
                    "rnfl_um": float\(np\.clip\(rnfl - 0\.16 \* year_offset \+ rng\.normal\(0, 0\.4\), 55, 115\)\),
                    "vessel_density_percent": float\(np\.clip\(vessel_density - 0\.07 \* year_offset \+ rng\.normal\(0, 0\.3\), 30, 60\)\),
                    "gfaz_area_mm2": float\(np\.clip\(gfaz_area \+ 0\.005 \* year_offset \+ rng\.normal\(0, 0\.01\), 0\.05, 0\.7\)\),
                    "apoe_e4_count": float\(apoe_e4\),
                    "ancestry_pc1": ancestry_pc1,
                \}"""

replacement_for_body = """            annual_hazard, event, event_time_years, event_time_days = _generate_event(rng, base)

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
                    "sex": base.sex,
                    "birth_year": int(start_date.year - base.age),
                    "education_years": round(base.education, 1),
                    "language": "Kannada" if cohort in INDIAN_COHORTS else "cohort_recorded",
                    "urban_rural": setting,
                    "synthetic": True,
                }
            )

            baseline_features: dict[str, float | None] = {}
            decline_rate = float(-0.14 - 0.55 * annual_hazard - 0.10 * max(base.vulnerability, 0.0))
            for visit_index in range(n_visits):
                year_offset = float(visit_index + rng.normal(0.0, 0.06)) if visit_index else 0.0
                baseline_days = int(round(max(year_offset, 0.0) * 365.25))
                visit_id = f"{participant_id}-V{visit_index}"
                progressed = bool(event and year_offset >= event_time_years)

                cognitive_score, values = _generate_visit_features(rng, base, decline_rate, year_offset, progressed, event_time_years, annual_hazard)"""

content = re.sub(body_regex_to_replace, replacement_for_body, content, flags=re.DOTALL)

# Also fix the annual_hazard parameter in _generate_visit_features
def_replace = r"def _generate_visit_features\(rng: np\.random\.Generator, base: ParticipantBaseline, decline_rate: float, year_offset: float, progressed: bool, event_time_years: float\) -> tuple\[float, dict\[str, float\]\]:"
def_with = "def _generate_visit_features(rng: np.random.Generator, base: ParticipantBaseline, decline_rate: float, year_offset: float, progressed: bool, event_time_years: float, annual_hazard: float) -> tuple[float, dict[str, float]]:"
content = re.sub(def_replace, def_with, content)


# Also need to fix baseline_rows.append because it still uses base.*
baseline_rows_replace = r"""            baseline_rows\.append\(
                \{
                    "participant_id": participant_id,
                    "cohort": cohort,
                    "urban_rural": setting,
                    "sex": sex,
                    "sex_binary": sex_binary,
                    "age": age,
                    "education_years": education,"""

baseline_rows_with = """            baseline_rows.append(
                {
                    "participant_id": participant_id,
                    "cohort": cohort,
                    "urban_rural": setting,
                    "sex": base.sex,
                    "sex_binary": base.sex_binary,
                    "age": base.age,
                    "education_years": base.education,"""
content = re.sub(baseline_rows_replace, baseline_rows_with, content)

with open("neurosaarthi-ad/demo/synthetic.py", "w") as f:
    f.write(content)
