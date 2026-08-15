# NeuroSaarthi-AD: Methods Report
## A Multimodal Longitudinal Dementia Risk and Progression Engine

### 1. Background and Objectives
- Brief on the healthy brain aging challenge
- The need for cross-cohort, multimodal, India-relevant dementia prediction
- Primary objectives: incident risk prediction, cognitive trajectory forecasting, digital twin lite

### 2. Data Sources
Table of all cohorts: ADNI, NACC, AIBL, OASIS, UK Biobank, TLSA, SANSCOG, GenomeIndia
- For each: n, modalities, visit cadence, role in training/validation

### 3. Study Design
- Longitudinal observational study design
- Prediction targets: CU→MCI, MCI→dementia at 1/3/5 years
- Cohort roles: development (ADNI, NACC, AIBL, OASIS, UKB), Indian adaptation (TLSA), external validation (SANSCOG)

### 4. Feature Engineering
- Modality groups: cognition, MRI, blood biochemistry, OCT/OCTA, genomics
- Derived features: hippocampal atrophy z-scores, metabolic risk index, retinal composite, APOE haplotype encoding
- Missingness handling: explicit indicators, native NaN support in LightGBM

### 5. Harmonization
- Common data model (participant, visit, modality_features, outcomes)
- Variable mapping with cohort-specific aliases and unit conversions
- Train-only ComBat batch correction (Johnson et al., 2007)
- Leakage prevention: subject-level splits before preprocessing

### 6. Model Architecture
- Classification: LightGBM gradient-boosted classifier (with LogisticRegression fallback)
- Survival: Random Survival Forest, Gradient-Boosted Cox (scikit-survival)
- Progression: Mixed-effects trajectory model (random intercept + slope)
- Fusion: Late fusion with learned stacking meta-learner
- Digital twin lite: Mahalanobis-distance nearest neighbor retrieval

### 7. Evaluation Protocol
- Internal: subject-level 5-fold CV, temporal validation
- External: held-out SANSCOG rural Indian cohort
- Metrics: AUROC, AUPRC, Brier score, C-index, time-dependent AUC, integrated Brier score
- Calibration: calibration slope, reliability diagrams, ECE
- Subgroup fairness: sex, age band, education, urban/rural
- Uncertainty: bootstrap CIs, conformal prediction intervals

### 8. Results
[Synthetic demonstration results — to be populated with real cohort results]

### 9. Limitations
- Current results are from synthetic data only
- OCT/OCTA evidence base is still emerging
- Indian cohort sample sizes may limit subgroup analysis
- GenomeIndia used only for ancestry context, not outcome labels

### 10. Reproducibility
- Locked random seeds throughout
- DVC-ready data pipeline
- Full environment specification (pyproject.toml)
- Model cards for each component

### References
[Key citations]
