# ADR 0002: Protected data remains inside an offline approved environment

- Status: accepted
- Date: 2026-08-15

## Decision

CBR participant-level data and derivatives run only in `secure_cbr`, where network access is disabled. The public repository contains code/config/schema/synthetic fixtures only. Exports are aggregate, DUA-authorised, locally audited, and human reviewed.

## Consequences

Remote experiment tracking, telemetry, public AI services, and public cloud inference are disabled for protected workflows. Dependency artefacts must be approved before data are mounted. Protected-data-trained checkpoints are restricted by default. Public cohorts develop and freeze pipelines before secure adaptation.
