# NeuroSaarthi-AD

NeuroSaarthi-AD is a judge-facing research prototype for the CBR Healthy Brain Aging AI Challenge. It combines a secure, harmonised, longitudinal multimodal workflow with an interactive Streamlit studio for progression risk, cognitive trajectories, modality-level drivers, and digital twin lite retrieval.

The prototype is intentionally feature-first rather than raw-image-first. Its local demo generates 840 deterministic synthetic participants following the modality and visit patterns of seven named cohorts. These records are fictional and every displayed metric is calculated from synthetic predictions; no result represents real cohort performance or clinical validity.

## Quick Start

```powershell
cd neurosaarthi-ad
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,dashboard]"
python -m pytest
streamlit run dashboards\streamlit_app.py
```

The app opens on an editable synthetic participant and includes three views:

- **Participant studio** — 1-, 3-, and 5-year risk, uncertainty, cognitive trajectory, modality drivers, and five matched synthetic trajectories.
- **India-first validation** — public-cohort validation, TLSA-style adaptation, fully held-out SANSCOG-style evaluation, calibration, subgroups, and ablation.
- **Harmonisation audit** — source mappings, unit conversions, missingness, provenance, and leakage safeguards.

This is a research demonstration only. It is not intended for diagnosis, screening, treatment decisions, or claims about real ADNI, NACC, AIBL, OASIS, UK Biobank, TLSA, or SANSCOG performance.

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
demo/                Deterministic synthetic cohorts and fitted demo runtime
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
- Synthetic-only local operation with no upload, network, account, or persistence path.
