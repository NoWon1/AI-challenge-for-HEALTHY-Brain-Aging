# Agent Instructions for NeuroSaarthi-AD

Welcome to the **NeuroSaarthi-AD** codebase. This file provides guidelines and context for AI coding assistants (like Jules) working on this repository.

## Overview
NeuroSaarthi-AD is a judge-facing research prototype for the CBR Healthy Brain Aging AI Challenge. It provides a secure, harmonised, longitudinal multimodal workflow with an interactive Streamlit studio for progression risk, cognitive trajectories, modality-level drivers, and digital twin lite retrieval.

**Key constraint:** The prototype operates locally on deterministic synthetic data. It should never upload participant-level data. The local demo generates synthetic participants for demonstration.

## Project Structure & Navigation
When exploring or modifying code, keep this directory structure in mind:

- `data_contracts/`: Common schema and validation contracts. Modify this if you add new features or change data expectations.
- `etl/`: Cohort-specific ingestion adapters.
- `harmonization/`: Contains logic for visit alignment, train-only transforms, leakage guards, and ComBat batch correction.
- `features/`: Modality feature builders.
- `models/`: Implementations for Classification, Survival (e.g. Random Survival Forests, CoxBoost via `scikit-survival`), Progression, Fusion, and Twin-lite.
- `evaluation/`: Scripts for splits, metrics, calibration, and validation reports.
- `dashboards/`: Streamlit demo entry points (e.g., `streamlit_app.py`).
- `demo/`: Logic to generate and run deterministic synthetic cohorts.
- `configs/`: Configuration for endpoints, cohorts, and features.
- `tests/`: Lightweight regression tests using `pytest`.

## Tools & Commands
To verify your work, use the following commands:
- **Environment**: Ensure you are using the `.venv` virtual environment.
- **Testing**: Run tests using `python -m pytest`.
- **Demo Dashboard**: Run the UI locally using `streamlit run dashboards\streamlit_app.py`.

## Coding Conventions & Constraints
- **Privacy First**: Maintain participant-level privacy. Do not add any logic that uploads, connects to a network, requires accounts, or persists sensitive participant data.
- **Model Frameworks**: The project uses `LightGBM` for classification and `scikit-survival` for survival models.
- **Explainability**: We use SHAP for explainability (modality drivers). Ensure SHAP logic is maintained or extended appropriately when modifying models.
- **Data & Harmonization**: 
  - Ensure missing-modality-aware modelling instead of complete-case-only modelling.
  - Imputation, scaling, and harmonisation must remain train-only to prevent data leakage.
  - Default splits must be subject-level and time-aware.
  - External validation and subgroup reporting are first-class outputs.

## Model Validation & Testing
When creating or modifying predictive models, basic syntax checks are insufficient. Agents MUST create or update comprehensive `pytest` suites to verify correctness, ensuring the following constraints are met and tested. Always run the **full `pytest` suite** to catch integration regressions.

### General Model Validation Rules
- **Data Leakage & Imputation**: Transformers/Imputers must be fitted *only* during `fit()` and the fitted instance reused unchanged during prediction.
- **Edge-Case Handling**: Ensure robust `NaN` handling during both training and prediction.
- **Model State Validation**: Validate missing columns cleanly and ensure `NotFittedError` is raised if predictions are attempted before `fit()`.

### Additional Survival-Model Validation Rules
- **Survival Targets**: `event_col` must be correctly formatted as boolean, and survival targets must be constructed robustly (e.g., using `sksurv.util.Surv.from_arrays()`).
- **Censoring**: Explicitly test behavior on both censored and uncensored observations.
- **Curve Correctness**: Verify that survival probabilities are monotonic and bound within `[0,1]`.
- **Time Extrapolation**: Requested `time_points` outside the model-supported time range must follow an explicit documented policy—preferably clipping or raising `ValueError`; silent extrapolation is prohibited.
- **Metrics & Explainability**: Permutation importance must use a survival-aware scoring mechanism (like C-index).
