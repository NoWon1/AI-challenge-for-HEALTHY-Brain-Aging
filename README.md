<div align="center">
  <h1> NeuroSaarthi-AD</h1>
  <p><strong>Multimodal dementia risk, progression, and twin-lite scaffold for the CBR Healthy Brain Aging AI Challenge</strong></p>
  
  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version"></a>
    <a href="#"><img src="https://img.shields.io/badge/Version-0.2.0-success.svg" alt="Version"></a>
    <a href="#"><img src="https://img.shields.io/badge/Status-Research_Prototype-orange.svg" alt="Status"></a>
  </p>
</div>

---

##  Table of Contents

- [About the Project](#-about-the-project)
- [Core Features](#-core-features)
- [What's New in v0.2.0](#-whats-new-in-v020)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage & Dashboard](#-usage--dashboard)
- [Project Architecture](#-project-architecture)
- [Design Principles](#-design-principles)
- [Important Disclaimer](#-important-disclaimer)

---

##  About the Project

**NeuroSaarthi-AD** is a judge-facing research prototype designed for the **CBR Healthy Brain Aging AI Challenge**. It provides a secure, harmonized, and longitudinal multimodal workflow to predict dementia progression risk and cognitive trajectories.

The prototype includes an interactive **Streamlit studio** that allows users to explore progression risks, understand modality-level drivers through explainable AI (XAI), and retrieve matched synthetic trajectories using a *digital twin-lite* approach.

> **Note:** The current local demo is feature-first rather than raw-image-first. It generates **840 deterministic synthetic participants**, mimicking the modality and visit patterns of seven named cohorts. All records are fictional and do not represent real clinical data or true cohort performance.

---

##  Core Features

- **Multimodal Data Harmonization:** Securely align visits and correct batches (via ComBat) across multi-cohort data.
- **Advanced Predictive Modeling:** Leverages LightGBM classifiers, Random Survival Forests, and CoxBoost (`scikit-survival`) for robust time-to-event and risk prediction.
- **Digital Twin-Lite Retrieval:** Find and match synthetic patient trajectories to understand potential disease progression paths.
- **Explainability (XAI):** Integrated SHAP values to identify and explain the top modality drivers influencing risk scores.
- **Interactive Dashboard:** A comprehensive Streamlit application to visualize risk factors, patient trajectories, and model evaluations.
- **Strict Data Privacy:** Participant-level privacy by design with train-only transformations and no unauthorized data uploads.

---

##  What's New in v0.2.0

-  **Advanced Modeling:** Introduction of LightGBM, Random Survival Forests, and CoxBoost models.
-  **Enhanced Harmonization:** Added ComBat batch correction for seamless multi-cohort data integration.
-  **Explainability:** Deep integration of SHAP values to demystify modality drivers for individual predictions.

---

##  Getting Started

### Prerequisites

- **Python:** `3.10` or higher
- **Virtual Environment:** Recommended (e.g., `venv`, `conda`)

### Installation

Clone the repository (if applicable) and navigate to the project directory:

```powershell
cd neurosaarthi-ad
```

Create and activate a virtual environment:

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

Install the package with all optional dependencies (Advanced modeling, Development tools, and Dashboard requirements):

```bash
python -m pip install -e ".[advanced,dev,dashboard]"
```

Run the lightweight regression tests to ensure everything is set up correctly:

```bash
python -m pytest
```

---

##  Usage & Dashboard

NeuroSaarthi-AD comes with an interactive Streamlit application that serves as a participant studio and validation dashboard.

To launch the dashboard, run:

```bash
streamlit run dashboards\streamlit_app.py
```

### Dashboard Views

1. **Participant Studio:** Analyze 1-, 3-, and 5-year progression risks, uncertainty bounds, cognitive trajectories, modality drivers (SHAP), and view up to five matched synthetic trajectories.
2. **India-First Validation:** Explore public-cohort validation, TLSA-style adaptation, fully held-out SANSCOG-style evaluation, model calibration, subgroup analysis, and ablation studies.
3. **Harmonisation Audit:** Review source data mappings, unit conversions, missingness reports, data provenance, and leakage safeguard metrics.

---

## Project Architecture

The repository is structured to maintain a clear separation between data contracts, feature engineering, modeling, and presentation.

```text
neurosaarthi-ad/
├── configs/             # Endpoint, cohort, and feature configuration
├── data_contracts/      # Common schema definitions and validation contracts
├── cohort_cards/        # Human-readable cohort notes and intended uses
├── etl/                 # Cohort-specific ingestion adapters
├── harmonization/       # Visit alignment, train-only transforms, and leakage guards
├── features/            # Modality feature builders
├── models/              # Classification, survival, progression, fusion, and twin-lite logic
├── evaluation/          # Splits, metrics, calibration, and validation reports
├── dashboards/          # Streamlit apps and API demo entry points
├── demo/                # Deterministic synthetic cohorts and fitted demo runtime
├── docs/                # Architecture, governance, and project roadmap
├── tests/               # Lightweight regression tests
└── model_cards/         # Model-card templates and final documentation
```

---

##  Design Principles

NeuroSaarthi-AD is built with strict adherence to the following principles:

1. **Privacy First:** Participant-level privacy is guaranteed. There is no public-tool upload of participant-level CBR data. Synthetic-only local operation ensures no network, account, or persistence path vulnerabilities.
2. **Robust Splitting:** Subject-level and time-aware data splits are enforced by default to prevent temporal leakage.
3. **Leakage Prevention:** Imputation, scaling, and harmonisation are strictly designated as **train-only** operations.
4. **Missingness-Aware:** The modeling pipeline natively handles missing modalities, avoiding the pitfalls of complete-case-only filtering.
5. **Validation as a First-Class Citizen:** External validation and subgroup reporting are primary outputs, ensuring the model's reliability across diverse demographics.

---

##  Important Disclaimer

> **This project is a research demonstration only.** 
> It is **not** intended for diagnosis, screening, treatment decisions, or clinical use. None of the results represent real performance or clinical validity for cohorts such as ADNI, NACC, AIBL, OASIS, UK Biobank, TLSA, or SANSCOG.
