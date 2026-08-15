# Model Card: Survival Analysis (RSF + CoxBoost)

## Intended Use
Research prototype for time-to-event analysis of cognitive decline and dementia onset.

## Model Details
- **Architectures**: Random Survival Forest (RSF) and Gradient-Boosted Cox (CoxBoost) using `scikit-survival`.
- **Outputs**: Continuous hazard scores and survival probabilities over time.

## Training Data
Synthetic demo cohort of longitudinal records (840 participants).

## Evaluation
- **Metrics**: Concordance Index (C-index), time-dependent AUC, Integrated Brier Score (IBS).

## Limitations
- Trained on synthetic data only.
- Assumes proportional hazards (CoxBoost) which may not hold for all multimodal covariates.
