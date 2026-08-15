# NeuroSaarthi-AD implementation blueprint

Status date: 2026-08-15. This document is the Phase 0 architecture and execution contract. It does not report model performance.

## A. Architecture decision

NeuroSaarthi-AD is a multitask, longitudinal research platform, not a monolithic image classifier. The primary chain is:

```text
approved local cohort data
  -> explicit cohort adapter + provenance
  -> participant-isolated/time-safe split assignment
  -> modality-specific QC and preprocessing
  -> validated anatomical masks and quantitative biomarkers
  -> transparent task baselines
  -> calibrated multimodal/longitudinal models
  -> external validation and subgroup/error analysis
  -> digital-twin-lite retrospective forecasts
  -> aggregate research reports and authorised local dashboard
```

The architecture has six bounded layers:

1. **Governance boundary.** Public development environments accept public/controlled-access data only. CBR-SANSCOG and CBR-TLSA adapters can run only inside the approved `secure_cbr` environment. Network access is disabled there. Participant rows, images, derived features, logs, predictions, and protected-data-trained checkpoints never enter the public repository or an external AI service.
2. **Canonical longitudinal data layer.** Participants, visits, imaging, cognition, clinical, biochemistry, ophthalmology, gait/balance, genomics, outcomes, and feature provenance retain cohort-native meaning. Every cross-cohort mapping declares `EXACT`, `COMPATIBLE`, `DERIVED`, `APPROXIMATE`, or `NON_EQUIVALENT` plus evidence.
3. **Quantitative imaging layer.** DICOM/NIfTI/BIDS ingestion, structural QC, transform-tracked T1 preprocessing, independently validated hippocampal/whole-brain segmentation, and native-space morphometry create interpretable imaging biomarkers.
4. **Task layer.** Classification, continuous regression, brain age, survival, and longitudinal mixed-effects models start with required transparent baselines. Advanced models are accepted only when they improve frozen external endpoints.
5. **Fusion and forecasting layer.** Missingness-aware late fusion is the first multimodal method; gated/intermediate fusion follows. Digital Twin Lite retrieves comparable observed trajectories and forecasts future scores with calibrated intervals. It does not simulate treatment effects or causal counterfactuals.
6. **Validation and delivery layer.** Calibration, uncertainty, subgroup analysis, robustness, external cohorts, model cards, aggregate reporting, local CLI/API/dashboard, and reproducible model packages remain separated from training code.

MONAI components have distinct roles: Bundles package model configuration/weights/metadata; MONAI Label provides expert correction and active learning; MONAI Deploy packages inference applications for interoperability. No component implies medical-device or clinical readiness. Releases/tags are pinned; the moving `dev` branch is not a dependency.

This is appropriate because OASIS-3, ADNI, and the intended Indian cohorts are longitudinal and multimodal. It tests how much imaging adds beyond age, cognition, and clinical context while preserving interpretable anatomical measurements, censoring, missing modalities, uncertainty, and unseen-cohort generalisation.

## B. Scope boundary

| Boundary | Included work |
|---|---|
| Competition-critical | Governance gate; common data model; participant/time-safe splits; OASIS-3 T1 QC to morphometry to brain-age/cognitive baselines; ADNI progression/survival and multimodal validation; hippocampal segmentation baseline; missingness-aware fusion; calibration/uncertainty; held-out public cohort; aggregate dashboard; Digital Twin Lite backtesting; reproducible CLI/tests/model cards. |
| Research-enhancing | Whole Brain Large UNEST feature extraction after regional validation; NACC/AIBL structured external validation; scanner/site harmonisation; representation learning; longitudinal deep models; MONAI Label expert correction; research API and MONAI Bundle. |
| Stretch | UK Biobank-scale pretraining; SwinUNETR/large transformer sweeps; self-supervised MRI pretraining; diffusion/synthesis quality experiments; multi-GPU domain adaptation; MONAI Deploy application package. None is a core dependency. |
| Inappropriate/unsupported | Autonomous diagnosis/screening; “neurologist-level” or “clinical-grade” claims; routine biomarker evaluation of asymptomatic people; causal treatment counterfactuals; pooling BraTS tumour biology into Alzheimer’s modelling; silent cognitive-test equivalence; public CBR uploads; participant redistribution; test-set tuning; final-model selection on internal AUROC alone. |

## C. Task matrix

| Task | Target | Dataset | Input modalities | Required baseline | Advanced model | Primary metric | External validation |
|---|---|---|---|---|---|---|---|
| T1 QC/preprocessing | Model-ready, invertible T1 | OASIS-3 | T1 MRI + acquisition metadata | Deterministic rule QC | Learned QC only after labels | QC failure sensitivity + review agreement | ADNI/site-held-out |
| Hippocampal segmentation | Left/right mask | MSD Hippocampus, compatible labels | T1 MRI | 3D U-Net | SegResNet/UNETR/SwinUNETR | Per-case Dice + HD95 | OASIS/ADNI manual subset |
| Whole-brain morphometry | Regional volumes | OASIS-3 | T1 MRI | Validated pretrained bundle + rules | Fine-tuned Large UNEST | Regional volume error + surface Dice | ADNI/manual subset |
| Brain age | Chronological age; brain-age gap | OASIS-3 | Morphometry + demographics; optional T1 embedding | Elastic net | 3D CNN/transformer | MAE years + calibration | ADNI and cohort-held-out |
| Cognitive prediction | Future continuous score | OASIS-3, ADNI | Baseline cognition + clinical + imaging | Elastic net/mixed effects | Temporal transformer/GRU-D | MAE/RMSE at frozen horizon | ADNI to OASIS and reverse |
| State classification | Cohort-native CU/MCI/dementia state | ADNI/OASIS-3 | Demographics, cognition, MRI | Logistic regression | Boosting/3D CNN/fusion | AUROC/AUPRC + calibration | Cohort-held-out |
| Progression survival | Time to predefined progression | ADNI | Longitudinal clinical/cognition/MRI/biomarkers | Cox regression | RSF/DeepSurv/discrete hazard | C-index + integrated Brier | OASIS/NACC/AIBL where labels align |
| Longitudinal trajectory | Individual cognitive slope/forecast | OASIS-3, ADNI | Repeated cognition + covariates | Linear mixed model | GRU-D/temporal transformer | Horizon MAE + interval coverage | Cohort-held-out |
| Missing-aware fusion | Progression/score with absent modalities | ADNI | MRI, cognition, clinical, blood/PET/CSF as allowed | Late fusion | Gated/intermediate fusion | Delta under modality ablation | OASIS/AIBL |
| Domain shift | Cohort/scanner robustness | Public cohorts then CBR-secure | Features + site/scanner | Stratified recalibration | ComBat/domain-adversarial model | External delta + calibration | Rural/urban reciprocal holdout |
| Digital Twin Lite | Future observed trajectory | Public longitudinal cohorts | Validated multimodal state/history | Nearest-neighbour retrieval + mixed model | Learned trajectory embedding | Retrospective horizon MAE/coverage | Entire held-out cohort |
| BraTS engineering | Tumour masks only | BraTS | T1/T1c/T2/FLAIR | MONAI bundle reproduction | No dementia transfer claim | Dice/HD95/inference reproducibility | Official BraTS split only |

## D. Dataset matrix

| Dataset | Will be used for | Will not be used for |
|---|---|---|
| OASIS-3 | First public vertical slice; longitudinal T1 QC/preprocessing; morphometry; brain age; cognitive prediction; ageing/decline development. | Casual mixing with other OASIS releases; invented field definitions; final claims without external validation. |
| ADNI | Progression/survival; multimodal longitudinal modelling; MRI/PET/CSF/blood/genetic experiments within permissions; quantitative-summary validation. | Unauthorised redistribution; treating clinical state as a biological gold standard; tuning after frozen external evaluation. |
| NACC | Large structured phenotype modelling and external clinical generalisation where endpoint definitions align. | MRI segmentation ground truth; silent equivalence with ADNI/OASIS cognitive tests. |
| AIBL | Independent external validation and compatible multimodal ageing/AD experiments. | Development leakage or test-driven feature selection. |
| UK Biobank | Population brain-age development, preclinical brain-health representation learning, lifestyle/genetic research when approved. | Alzheimer’s diagnosis labels by proxy; required core compute dependency. |
| MSD Hippocampus | Small-structure segmentation engineering, per-case surface/volume metrics, baseline comparison. | Alzheimer’s outcome training or claims about progression. |
| BraTS | Auxiliary multimodal loading, patch training, sliding-window inference, segmentation metrics, Bundle/visualisation engineering. | Dementia/Alzheimer’s biological modelling; tumour labels as ageing labels. |
| CBR-SANSCOG | Secure-only rural prospective validation/adaptation; rural-to-urban generalisation and longitudinal endpoints after DUA review. | Public development, public AI upload, participant export, unrestricted checkpoint publication, routine clinical screening. |
| CBR-TLSA | Secure-only urban longitudinal validation/adaptation; urban-to-rural generalisation after DUA review. | Public development, public AI upload, participant export, assumed visit-frequency match with SANSCOG. |
| GenomeIndia/other omics | Population reference or approved feature research with documented consent/ancestry limitations. | Individual risk inference, causal claims, or cross-study joining without explicit approval. |

Actual training remains blocked until each controlled dataset is locally authorised and its release-specific data dictionary is mapped. Synthetic fixtures test software contracts only and never produce claimed cohort performance.

## E. Proposed repository tree

```text
neurosaarthi-ad/
|-- README.md, LICENSE, CITATION.cff, SECURITY.md, CONTRIBUTING.md
|-- CODE_OF_CONDUCT.md, CHANGELOG.md, pyproject.toml, Makefile
|-- Dockerfile, docker-compose.yml, .dockerignore, .pre-commit-config.yaml
|-- .github/workflows/ci.yml
|-- configs/
|   |-- data/ preprocessing/ segmentation/ morphometry/ classification/
|   |-- regression/ survival/ longitudinal/ multimodal/ evaluation/ deployment/ security/
|-- neurosaarthi/
|   |-- cli.py, core/, security/
|   |-- data/{schemas,adapters,harmonization,validation,splitters}/
|   |-- imaging/{io,dicom,nifti,bids,qc,preprocessing,registration,segmentation,morphometry}/
|   |-- models/{baselines,segmentation,cnn3d,survival,temporal,fusion,generative}/
|   |-- training/, inference/, evaluation/, calibration/, uncertainty/
|   |-- explainability/, fairness/, domain_adaptation/, digital_twin/
|   |-- reporting/, visualization/, deployment/
|-- existing compatibility modules:
|   |-- data_contracts/, etl/, harmonization/, features/, models/, evaluation/, demo/
|-- monai_bundles/{hippocampus,whole_brain,brats}/
|-- deployment/{api,monai_deploy}/ dashboards/ scripts/
|-- docs/{GOVERNANCE.md,IMPLEMENTATION_BLUEPRINT.md,adr,model_cards,reproduction}/
|-- tests/{unit,integration,regression,smoke}/
|-- reports/  # aggregate templates only; participant outputs ignored
`-- data/, derivatives/, checkpoints/, artifacts/  # ignored; secure/local only
```

Migration is incremental: the established root compatibility modules remain importable while completed production slices enter the `neurosaarthi` namespace. Essential logic is never notebook-only.

## F. Technical dependency stack

Python 3.11 and 3.12 are supported. Core tabular code uses NumPy, pandas, scikit-learn, and PyYAML. Imaging uses NiBabel, SciPy, pydicom, and optional SimpleITK. The MONAI GPU environment is separate so a laptop installation does not download a large PyTorch stack.

Verified release pins on 2026-08-15:

- `monai==1.6.0`, released 2026-06-22, Python >=3.10. Official source: <https://pypi.org/project/monai/1.6.0/>.
- `torch==2.13.0`, released 2026-07-08 with Python 3.10-3.14 wheels. Official source: <https://pypi.org/project/torch/2.13.0/>.
- `nibabel==5.4.2`, released 2026-03-11. Official source: <https://pypi.org/project/nibabel/5.4.2/>.

MONAI 1.6 documents support for the current PyTorch version plus three previous minor versions, but the exact CUDA build must be locked per deployment hardware. Production images use a versioned MONAI image/tag, never `latest` or the GitHub `dev` branch. Core library ranges in `pyproject.toml` allow security patches; a lock/constraints file is produced and frozen separately for each CPU and CUDA profile after CI compatibility resolution.

Roles are deliberately non-duplicative: MONAI for medical imaging transforms/models/metrics and Bundles; scikit-learn for transparent tabular baselines; scikit-survival/lifelines for survival; statsmodels for mixed effects; MLflow local-file mode only; Optuna for development folds only; FastAPI for a later local research API; Streamlit for the authorised research dashboard.

## G. Sixteen-week execution plan

| Week | Objective and tasks | Owner | Required data | Output artefacts | Acceptance test | Failure fallback |
|---|---|---|---|---|---|---|
| 1 | Freeze endpoints, DUA checklist, secure/public boundary, terminology, ADRs. | Lead + validation + data | DUA summaries; no participant data | Governance, endpoint registry, threat model | Governance sign-off before protected access | Continue synthetic/public schema work only |
| 2 | Package, CLI, schemas, OASIS manifest adapter, split/leakage/security tests, CI. | Data + MLOps | Synthetic manifests | Installable package, audit CLI, CI | `make lint test`; participant/time leakage regression passes | Reduce to core dependencies; no imaging data |
| 3 | OASIS-3 T1 local ingestion, NIfTI geometry, automated QC, secure montage plan. | Imaging + data | Authorised OASIS-3 | QC tables and local montages | 100+ scans processed; failures manually sampled | Use smaller approved subset; fix protocol mapping |
| 4 | T1 resampling/crop/normalisation, inverse mask transforms, cache lineage. | Imaging | OASIS-3 T1 | Versioned derivatives + transform records | Native-space round-trip tests; deterministic hashes | Disable optional N4/registration and keep transparent baseline |
| 5 | MSD hippocampal 3D U-Net then SegResNet; per-case/surface metrics. | Imaging | MSD Hippocampus | Cross-validation masks/report | No subject leakage; Dice/HD95 CI; 2-sample CPU smoke | 3D U-Net only; patch-size reduction |
| 6 | Whole-brain bundle evaluation, morphometry, brain-age elastic net and 3D baseline. | Imaging + modelling | OASIS-3; label subset | Feature table, QC audit, brain-age report | Native-space volumes; held-out MAE/calibration; subgroup table | Use validated hippocampal/ICV features only |
| 7 | Cohort-native logistic/elastic-net/boosting classification and cognitive regression. | Modelling + statistics | OASIS-3/ADNI | Baseline benchmark | Nested participant CV; frozen endpoint definitions | Report insufficient label alignment; single-cohort analysis |
| 8 | Cox and mixed-effects baselines; censoring and visit-frequency checks. | Survival + statistics | ADNI longitudinal | Survival/trajectory reports | C-index, Brier, calibration; censored tests | Cox/mixed effects only; no deep model |
| 9 | Missingness patterns, modality masks, late fusion and modality dropout. | Fusion + data | ADNI modalities | Missingness report, fusion baseline | Missing-modality stress test; no complete-case-only result | Late fusion over reliable modalities only |
| 10 | Gated/intermediate fusion and temporal models; mandatory ablations. | Fusion + longitudinal | ADNI | Ablation table | Each added method beats or is rejected against baseline | Retain best calibrated simpler model |
| 11 | OASIS/ADNI reciprocal validation, site/scanner shift, train-only ComBat. | Validation + data | Public cohorts | External/domain-shift report | No refit on external set except declared recalibration subset | Stratified report without harmonisation |
| 12 | NACC/AIBL compatible external checks; robustness, fairness, error taxonomy. | Validation + statistics | Approved NACC/AIBL | Model/data cards, error report | Bootstrap CIs; subgroup Ns; multiple-comparison plan | Limit to endpoint-compatible cohorts |
| 13 | Secure CBR schema mapping, local QC, no-export audit, locked split registry. | Secure data + governance | Approved CBR only | Secure mappings/aggregate QC | Zero network; audit log; no participant exports | Remain public-data-only and mark CBR blocked |
| 14 | Public-to-CBR transfer, calibration, rural/urban reciprocal experiments. | Domain + validation | Secure SANSCOG/TLSA | Aggregate generalisation table | Frozen public model; held-out rural/urban protocol | Recalibration-only baseline; report shift transparently |
| 15 | Digital Twin Lite backtesting and authorised local dashboard demo story. | Longitudinal + MLOps | Public example; secure aggregate results | Forecasts, intervals, dashboard | Retrospective horizon coverage/MAE; no causal claims | Retrieval + mixed-effects baseline only |
| 16 | Freeze commit/config/splits; final external run once; ablations; Bundle/container/docs. | Entire team | Frozen approved releases | Release candidate and reproduction pack | Fresh-environment install; tests; model cards; final audit | Ship validated smaller scope; list blocked stretch work |

## H. Compute plan

| Tier | Hardware | Supported work | Not required/appropriate |
|---|---|---|---|
| Minimum laptop/dev | 4-8 CPU cores, 16 GB RAM, 50 GB free disk | Schemas, QC unit fixtures, tabular baselines, security audit, dashboard, CPU smoke tests | Full 3D training, diffusion, large cohort preprocessing |
| Recommended core | 16-32 CPU cores, 64-128 GB RAM, fast encrypted 2-4 TB SSD; one 24-48 GB CUDA GPU | OASIS/ADNI preprocessing, hippocampal segmentation, 3D CNN/SegResNet, modest transformer inference/fine-tuning | Large foundation pretraining |
| Stretch | 4+ GPUs with 48-80 GB each, 256+ GB RAM, encrypted high-throughput storage | Self-supervised pretraining, large transformer sweeps, diffusion fine-tuning, broad Optuna studies | Competition-critical path |

Patch training, cached deterministic transforms, persistent datasets, mixed precision, sliding-window inference, and gradient checkpointing are enabled only after correctness tests. Tier 3 is never mandatory.

## I. Validation protocol

1. Register task labels, prediction origin, horizons, exclusions, primary metric, calibration metric, and subgroup list before training.
2. Assign each participant to exactly one of train, calibration, validation, or test. All scans and visits follow the participant. The split registry is immutable and versioned locally.
3. For time-`t` prediction, every feature has `feature_time <= prediction_origin`; outcomes begin after the origin. Baseline/future duplicates and derivative provenance are checked automatically.
4. Development uses participant-level nested cross-validation. Inner folds select hyperparameters. Outer folds estimate development performance. Imputation, scaling, ComBat, feature selection, augmentation statistics, and calibration fit on the applicable training partition only.
5. Reserve a participant-disjoint calibration set for temperature/isotonic/conformal methods. Never fit calibration on the external test cohort.
6. Primary public internal test: OASIS participant holdout for the vertical slice and ADNI participant holdout for survival/fusion. Secondary site-held-out and scanner-vendor-held-out analyses test shortcut risk.
7. Public external hierarchy: internal participant -> internal site -> public cohort -> Indian secure cohort -> prospective study. Report the achieved level explicitly.
8. CBR protocol, after governance approval: freeze the public model; use a declared TLSA/SANSCOG adaptation subset if permitted; keep the reciprocal rural/urban cohort fully held out. Export aggregates only.
9. Primary endpoints: time to predefined cognitive progression and future cognitive score. Secondary endpoints: brain-age gap and validated structural phenotype. Engineering endpoints: MSD hippocampus and BraTS segmentation.
10. Report per-case estimates, participant-clustered bootstrap 95% CIs, AUROC/AUPRC/sensitivity/specificity for classification, MAE/RMSE/R2 for regression, C-index/time-dependent AUC/Brier/calibration for survival, and Dice/HD95/ASSD/surface Dice/volume error for segmentation.
11. Perform prespecified ablations without MRI/cognition/clinical/blood/harmonisation/pretraining/temporal modelling/modality dropout/calibration. Select models by external generalisation, calibration, uncertainty, relevant operating point, robustness, interpretability, compute, then headline discrimination.
12. Run the frozen final external test once. Any post-test change creates a new version and cannot reuse the old test as untouched evidence.

No current file reports real-cohort performance; software fixtures and synthetic dashboard metrics are explicitly non-clinical.

## J. First 20 implementation tickets

| # | Ticket | Acceptance criterion | Status |
|---:|---|---|---|
| 1 | Governance and no-external-transmission policy | Roles, retention, export, deletion, AI restriction documented | Complete |
| 2 | Architecture and scientific scope ADR | Multitask/longitudinal chain and exclusions frozen | Complete |
| 3 | Installable package + CLI namespace | `neurosaarthi --version` and help work | Complete |
| 4 | Privacy-safe settings and logging | Secure network enablement rejected; identifiers redacted | Complete |
| 5 | Dataset export metadata audit | Blocks image/ID/date schemas without reading values | Complete |
| 6 | Canonical longitudinal record types | Invalid age/time/remote URI tests pass | Complete |
| 7 | OASIS-3 explicit manifest adapter | Keyed pseudonyms; no guessed field equivalence | Complete |
| 8 | Participant/group/site/cohort splitters | Pairwise participant-isolation tests pass | Complete |
| 9 | Future-feature leakage guard | Regression test rejects `feature_time > origin` | Complete |
| 10 | NIfTI native/canonical I/O | 3D/affine checks and local-only paths | Complete |
| 11 | Structural MRI automated QC | Geometry/finite/foreground/slice/FOV flags tested | Complete |
| 12 | Transform-tracked T1 preprocessing | Mask inversion preserves discrete labels and native shape | Complete |
| 13 | Native-space morphometry | Physical volumes, asymmetry, ICV normalisation tested | Complete |
| 14 | Longitudinal atrophy features | Annualised loss sign and interval validation tested | Complete |
| 15 | Brain-age elastic-net baseline | Participant train/cal/test; bias/conformal calibration | Complete |
| 16 | Future cognition elastic-net baseline | Explicit positive horizon and fit-state tests | Complete |
| 17 | CI/lint/container/reproducibility shell | Full suite passes in clean Python 3.11/3.12 | Implemented; remote matrix pending |
| 18 | OASIS-3 100-scan local run | Aggregate QC report; no participant export | Blocked on authorised data |
| 19 | MSD hippocampal 3D U-Net Bundle | CPU smoke + cross-validation metrics | Planned (Phase 3) |
| 20 | ADNI Cox/progression vertical slice | Frozen endpoint and external-ready report | Blocked on authorised data |

The live checklist is updated only when executable tests or authorised-data evidence meet the acceptance criterion.
