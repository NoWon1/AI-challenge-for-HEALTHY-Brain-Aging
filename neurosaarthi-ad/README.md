# NeuroSaarthi-AD

NeuroSaarthi-AD is a competition-ready scaffold for the CBR Healthy Brain Aging AI Challenge. It is built around a secure, harmonised, longitudinal multimodal workflow for MCI/dementia risk prediction, disease progression modelling, and digital twin lite trajectory retrieval.

The initial codebase is intentionally feature-first rather than raw-image-first. It assumes cohort data can be represented as participant, visit, modality-feature, and outcome tables, with modality blocks for MRI, OCT/OCTA, cognition, blood biochemistry, and genomics.

## Quick Start

```powershell
cd neurosaarthi-ad
python -m pytest
```

## Project Shape

```text
data_contracts/      Common schema and validation contracts
cohort_cards/        Human-readable cohort notes and intended uses
etl/                 Cohort-specific ingestion adapters
harmonization/       Visit alignment, train-only transforms, leakage guards
features/            Modality feature builders
models/              Classification, survival, progression, fusion, twin-lite
evaluation/          Splits, metrics, calibration, validation reports
dashboards/          Streamlit or API demo entry points
configs/             Endpoint, cohort, and feature configuration
docs/                Architecture, governance, roadmap
tests/               Lightweight regression tests
model_cards/         Model-card templates and final cards
```

## Design Principles

- Participant-level privacy and no public-tool upload of participant-level CBR data.
- Subject-level and time-aware splits by default.
- Train-only imputation, scaling, and harmonisation.
- Missing-modality-aware modelling instead of complete-case-only modelling.
- External validation and subgroup reporting as first-class outputs.

