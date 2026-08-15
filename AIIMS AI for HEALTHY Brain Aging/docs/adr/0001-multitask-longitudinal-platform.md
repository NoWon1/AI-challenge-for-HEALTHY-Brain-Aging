# ADR 0001: Multitask longitudinal platform over a monolithic classifier

- Status: accepted
- Date: 2026-08-15

## Decision

Build segmentation -> quantitative biomarkers -> classification/regression/survival -> missing-aware multimodal fusion -> longitudinal forecasting -> uncertainty -> Digital Twin Lite. The flagship endpoints are future cognitive decline/time to progression, not same-visit disease classification.

## Consequences

The system requires explicit visits, censoring, feature timing, participant-isolated splits, modality provenance, and external cohorts. Transparent baselines are mandatory. MRI remains a major subsystem but its incremental value is tested against cognition/clinical context. Development is slower than a single classifier but scientific claims are more credible and fit the cohort design.
