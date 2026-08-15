# Data governance and secure-computing policy

## Scope and non-negotiable rule

NeuroSaarthi-AD is a research decision-support and cohort-analysis platform. Participant-level CBR-SANSCOG or CBR-TLSA information must never be uploaded to public ChatGPT, another external AI service, a public repository, a public experiment tracker, or an unapproved cloud endpoint. The approved CBR environment must provide zero retention/no external transmission consistent with the applicable DUA.

This policy covers raw images, DICOM headers, clinical/cognitive/biochemical/eye/gait/omics tables, direct and pseudonymous IDs, dates, free text, participant-level derived biomarkers, per-participant predictions, restricted logs, and checkpoints trained on protected data. Pseudonymisation does not make participant-level data public.

## Architecture boundary

```text
Public repository and public development environment
  source code, schemas, synthetic fixtures, aggregate report templates,
  public-data configuration without credentials

Approved secure data environment
  raw protected data, linkage keys, participant tables/images,
  derivatives, restricted logs, split registry, protected checkpoints

Export gate
  DUA-authorised aggregate metrics/model documentation only
  -> local metadata audit -> governance review -> release
```

CBR adapters must call the environment policy before reading data. `secure_cbr` rejects network enablement. Do not mount protected directories into public containers or CI. Do not use remote MLflow/W&B/telemetry/crash reporting in the secure environment. Dependencies are mirrored/approved before data are mounted.

## Dataset register and licence/DUA review

Before access, a dataset steward records the release/version, licence or DUA, permitted purposes, prohibited purposes, geography/hosting constraints, retention period, publication review, derived-data definition, trained-model status, linkage restrictions, access roster, and deletion/return requirement. Controlled datasets (OASIS-3, ADNI, NACC, AIBL, UK Biobank and CBR) are not assumed redistributable merely because code supports them.

No adapter field is activated until its release-specific data dictionary and endpoint meaning are reviewed. Cognitive instruments are not treated as interchangeable without documented evidence and an equivalence level.

## Roles and least privilege

- **Data steward:** approves access, mappings, exports, retention, and deletion.
- **Secure-environment administrator:** provisions accounts, storage, network controls, audit logs, and dependency mirrors; cannot approve scientific exports alone.
- **Data engineer:** ingests and maps only authorised fields; has no publication authority.
- **Imaging/modelling researcher:** accesses the minimum pseudonymised data required; does not access linkage keys.
- **Validation lead:** freezes endpoints/splits, reviews leakage, subgroup reporting, and model cards.
- **Export reviewer:** reviews aggregate disclosure risk and runs the export audit; must be independent of the export author where practical.

Access is named, time-bounded, reviewed at least quarterly, and revoked promptly when a member leaves the project. Shared accounts are prohibited.

## Data minimisation and identifiers

Keep source identifiers only in a steward-controlled linkage area. Analysis tables use keyed, dataset-scoped pseudonyms. The pseudonymisation key comes from the secure secret store or environment and is never committed. Exact dates are replaced with time-from-baseline where scientifically adequate. Do not rewrite identifiers into additional files or logs.

Free-text fields are excluded by default. Genomic variants and retinal/facial images receive heightened review because they may be identifying. Data joins across cohorts require explicit approval and must not attempt re-identification.

## Retention, backup, and deletion

Raw and derived protected data inherit the most restrictive applicable DUA retention term. Storage and backups are encrypted, access-controlled, and located only in approved regions/systems. Temporary files use secure local scratch and are deleted after successful pipeline completion. Container layers must not contain data.

On expiry, withdrawal, or project closure, the steward coordinates deletion/return of raw data, participant derivatives, caches, logs, split registries, protected checkpoints, and backups as required. Completion is recorded in a deletion certificate/audit entry. Git history is not an acceptable place for protected data; if accidental inclusion occurs, stop work and follow the incident procedure.

## Audit logging

Log access and security-relevant operations: authenticated user, approved project, dataset release, command/version, aggregate row counts, timestamp, success/failure, and export decision. Never log participant IDs, paths containing identifiers, request bodies, clinical values, or per-participant predictions. Local logs are access-controlled and retained according to the DUA.

## Derived data and model leakage

Participant-level morphometry, embeddings, attention maps, predictions, nearest neighbours, synthetic images derived from participants, gradients, and explanation artefacts remain protected. Checkpoints may memorise or enable reconstruction/membership inference and are restricted unless the DUA and privacy review explicitly authorise release.

Before any model/checkpoint export, document training cohorts, privacy attack assessment proportional to risk, intended recipients, licence compatibility, and approval. Generative models receive additional reconstruction and nearest-neighbour analysis. “De-identified” is not a blanket export permission.

## Publication and export

Exports default to aggregate results with adequate cell sizes, confidence intervals, model/data cards, and code/configuration. Small cells, rare combinations, exact dates, free text, participant examples, images, and participant-level tables are prohibited unless specifically approved. Any demonstration participant is public and authorised or fully synthetic and clearly labelled.

Run locally before review:

```bash
python -m neurosaarthi.security.audit_dataset \
  --config configs/security/export_audit.yaml \
  --output audit-report.json
```

The audit inspects names, schemas, metadata keys, and optional DICOM tag presence. It does not read/report participant values or attempt re-identification. A passing automated audit is necessary but not sufficient; the data steward/export reviewer makes the final decision.

## Secure AI use

Public AI tools may receive public code, synthetic fixtures, empty schemas, public documentation, and aggregate results only after export approval. They may not receive CBR participant-level data, screenshots, file paths that reveal identifiers, schema samples populated from participants, error traces with values, or protected checkpoints. If an AI capability cannot be proven zero-retention/no-external-transmission and approved under the DUA, it is disabled.

## Incident response

If protected data may have left the approved boundary:

1. Stop the transfer/process without deleting evidence.
2. Disconnect the affected integration and preserve security logs.
3. Notify the data steward and security/IR contact immediately.
4. Record what data, destination, time, credentials, and retention policy were involved.
5. Rotate exposed credentials/keys and request provider deletion only through the approved incident process.
6. Follow DUA/institutional notification deadlines and document remediation.

Do not paste participant data or sensitive traces into an issue, chat, or external support ticket while investigating.

## Quality, fairness, and clinical-language controls

Governance includes scientific misuse prevention. All outputs state research/internal/external/prospective validation level. Use “research prediction”, “candidate imaging biomarker”, or “retrospective classification”; do not claim diagnosis, screening suitability, clinician replacement, or clinical proof. The 2024 biological AD criteria do not justify routine biomarker testing of asymptomatic people in clinical care.

Subgroup results require sample counts/uncertainty and must not be used to stigmatise communities. Rural/urban comparisons test generalisation and access-related domain shift, not biological essentialism.

## Repository safeguards

The root `.gitignore` and `.dockerignore` block common imaging, derivative, checkpoint, and prediction artefacts. Pre-commit uses Ruff and Gitleaks. CI has no protected credentials. These patterns reduce accidents but do not replace secure storage, access control, DUA review, or human export review.
