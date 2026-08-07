# Architecture

NeuroSaarthi-AD has five layers:

1. Secure cohort ingestion into the common data model.
2. Train-only ComBat batch correction harmonisation and leakage checks.
3. Modality feature blocks for cognition, MRI, OCT/OCTA, biochemistry, and genomics.
4. Upgraded v0.2 Models: LightGBM classification, scikit-survival (RSF/CoxBoost), mixed-effects progression, late fusion, and twin-lite.
5. Dashboard and reports for calibration, SHAP explainability, uncertainty, subgroup evaluation, and explanations.

## Local demonstration runtime

The judge-facing prototype exercises those layers using deterministic synthetic data:

1. Cohort-native aliases and units are generated for ADNI-, NACC-, AIBL-, OASIS-, UK Biobank-, TLSA-, and SANSCOG-style records.
2. Records are mapped to participant, visit, modality-feature, and outcome tables with provenance.
3. Public cohorts plus a TLSA adaptation subset fit train-only preprocessing and modality-specific discrete-time hazard models.
4. SANSCOG remains fully held out for rural Indian external validation.
5. Missing-aware late fusion, bootstrapped uncertainty, horizon-aware cognitive regression, and standardised nearest-neighbour retrieval power the Streamlit views.

All participants, model results, and validation metrics in this runtime are synthetic and non-clinical.
