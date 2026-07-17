# BRAIN-INDIA Progression Studio: Deep-Research Strategy Report for the CBR AI Challenge for Healthy Brain Aging

## Executive recommendation

Build one flagship system: **BRAIN-INDIA Progression Studio**, a multimodal, longitudinal dementia-risk and progression workbench that predicts future MCI/dementia conversion, forecasts cognitive trajectories, and presents a "digital twin lite" view using matched historical trajectories and calibrated uncertainty. The core should use **tabulated and extracted features** from MRI, OCT/OCTA, cognitive assessments, blood biochemistry, and targeted omics/genomics. Do not make a generic chatbot, a raw-image foundation model, or a genetics-only score the center of the submission. [S1, S2, S3]

The sharp interpretation of the challenge is this: CBR is asking for **research acceleration infrastructure plus credible prediction**, not just an isolated classifier. The wording explicitly names prediction of MCI/dementia, disease progression, digital twins, and data processing or harmonisation tools; the local inventory confirms that the provided CBR data are already multimodal and substantially tabulated across clinical, blood, cognitive, MRI, eye/vision, autonomic, gait, balance, audiometry, and spirometry streams. [S1, S2]

This recommendation fits a B.Tech CS + bioinformatics team because it leans into Python ML, data engineering, survival modelling, multimodal tabular learning, explainability, and genomics-aware QC. It avoids the least feasible path: training a massive raw MRI/OCT foundation model from scratch under a 12-week competition schedule. [S1, S2, S4, S9]

The single best project name and scope are:

**BRAIN-INDIA Progression Studio: a harmonised multimodal risk, progression, and twin-lite workbench for Indian brain-aging research.**

The highest-scoring story is: train and benchmark on public longitudinal cohorts; adapt and validate on CBR's rural and urban Indian cohorts; show that the system works when modalities are missing; and report calibration, survival metrics, fairness, uncertainty, and leakage controls instead of only AUROC. [S4, S6, S7, S8, S9, S10, S11, S12, S17]

## Dataset landscape

### CBR data inventory facts that should drive the design

The provided CBR inventory is more specific than the public cohort webpages and should be treated as the planning source for the challenge data. It describes SANSCOG as a rural Karnataka cohort and TLSA as an urban Bangalore cohort, both prospective and ongoing. It lists SANSCOG at **10,013 participants**, average age about **59**, with about **4 visits** and **2-year follow-up**, while TLSA is listed at **1,670 participants**, average age about **62**, with about **9 visits** and **yearly follow-up**. [S2]

| Data stream | SANSCOG data points | TLSA data points | Design implication |
|---|---:|---:|---|
| Clinical data | 17,561 | 5,534 | Strong base for risk factors, comorbidity, CDR/HMSE, function, family history, substance use. |
| Blood biochemistry | 15,314 | 5,222 | Feasible low-cost biomarker layer: lipids, glucose, CBC, CRP, liver/kidney function, HbA1c. |
| Cognitive data | 16,134 | 4,478 | Core outcome and progression modelling substrate. |
| MRI | 3,330 | 3,345 | Analysis-ready structured neuroimaging layer, not necessarily raw foundation-model training. |
| Eye and vision | 2,436 | 4,012 | OCT/OCTA/fundus should be an optional augmentation channel, especially strong in TLSA. |
| Autonomic function | 92 | 477 | Exploratory only; too sparse for a core challenge claim. |
| Audiometry | 714 | 2,013 | Useful for multimorbidity and aging phenotype; secondary. |
| Gait | 590 | 1,359 | Secondary mobility/frailty feature block. |
| Balance | 1,850 | 4,020 | Potentially valuable for aging phenotype and fall/frailty modelling. |
| Spirometry | 5,123 | 4,601 | Cardiometabolic/vascular-aging context; useful but not dementia-specific by itself. |

The CBR Data Use Agreement materially affects architecture: participant-level CBR data must stay in secure, approved environments; public-facing AI tools without guaranteed data containment must not process sensitive or participant-level data; outputs must not enable re-identification or reconstruction; and public dissemination using the data requires CBR review and written approval. This means the competition build should be privacy-preserving by design: local pipelines, aggregate demos, synthetic or de-identified demo rows, model cards, and no uploading raw CBR data to public tools. [S3]

### Dataset-by-dataset analysis

| Dataset | Best use in this challenge | Do not use it for | Why it matters |
|---|---|---|---|
| **ADNI** | Biomarker-rich AD progression modelling; CU/MCI/dementia cohort definitions; MRI/PET/cognition/biofluid/genetics benchmark; model baselines and longitudinal endpoints. | India-specific calibration; retinal/OCT core modelling; population prevalence. | ADNI is a longitudinal multicenter observational study built to validate AD biomarkers for clinical trials, with clinical, imaging, genetic, biofluid, cognitive, and demographic data. [S4, S5] |
| **UK Biobank** | Population-scale preclinical risk, brain/health/lifestyle/genomics context, blood and imaging-derived features, long-term linked outcomes. | High-resolution symptomatic AD staging by itself; fast access if RAP is unavailable; India-specific transfer without recalibration. | UKB follows 500,000 participants recruited age 40-69, includes broad health, imaging, biomarker, genetic, records, questionnaire, physical, lifestyle, and environmental data; UKB-RAP was listed as closed with phased reopening from September 2026 on the page accessed July 9, 2026. [S6] |
| **OASIS** | Open MRI-first proof-of-concept, reproducible imaging benchmarks, FreeSurfer/PET processed outputs, rapid demos. | A full competition flagship alone; Indian calibration; blood/omics-heavy modelling. | OASIS provides open neuroimaging datasets; OASIS-3 has 1,378 participants, 2,842 MR sessions, PET scans, clinical/cognitive data, and biomarker context. [S7] |
| **ADDI / AD Workbench** | Data discovery, secure cloud workspaces, interoperability, harmonisation inspiration, federated/distributed data sharing design. | Treating it as a single cohort to train the model. | ADDI describes AD Workbench as a secure cloud-based interoperability layer with 340+ datasets and tools to access, combine, analyze, and visualize ADRD data. [S8] |
| **NACC** | External validation for longitudinal clinical prediction; UDS annual visits; multimodal linkage across imaging, genetics, fluid biomarkers, neuropathology. | Estimating U.S. population prevalence/incidence; naive pooling of CSF assays; assuming all imaging is homogeneous. | NACC contains 56,000+ participants, 217,000 clinical assessments, annual UDS follow-up, imaging, genetic/genomic, neuropathology, and biomarker streams; NACC explicitly warns its sample is not population-based and that CSF values need additional harmonisation across assays/centers. [S9] |
| **AIBL** | Early detection, long follow-up, external validation, cognition/mood/lifestyle plus blood/CSF and neuroimaging. | Indian calibration; retinal-first claims; very large-scale genomics discovery. | AIBL is an ongoing observational cohort launched in 2006, collecting data every 18 months across biomarkers, neuroimaging, cognition, mood, health, and lifestyle, with 3,045 participants as of February 2023. [S10] |
| **TLSA** | Urban Indian validation/adaptation; frequent annual visits; strong OCT/OCTA, MRI, clinical, cognitive, blood, balance, spirometry, genomics streams. | Training massive raw-image models; discovery-scale genomics alone; assuming it represents rural India. | CBR describes TLSA as a long-term Bangalore aging cohort with clinical, neurocognitive, lifestyle, anthropometric, biochemical, genetic, and multimodal neuroimaging measures; the local inventory lists 1,670 participants and about 9 annual visits. [S2, S11] |
| **SANSCOG** | Rural Indian generalisation; fairness across rural/urban context; community-based risk/protective factor analysis; large CBR anchor cohort. | Assuming dense MRI/OCT for every participant; short-term event-rich dementia modelling without checking incident counts; raw-image training. | CBR describes SANSCOG as a prospective rural cohort in Karnataka for adults 45+ with projected n=10,000, multimodal assessments, and at least 10 years of follow-up; the inventory lists 10,013 participants. [S2, S12] |
| **GenomeIndia** | Indian ancestry-aware genomics QC, allele-frequency reference, variant interpretation, ancestry PCs, conservative genetic-risk calibration for Indian cohorts. | Dementia outcome modelling by itself; direct AD progression labels; unvalidated European PRS transfer. | GenomeIndia is designed to catalogue Indian genetic variation from 10,000 representative individuals; the 2025 Nature Genetics paper reports 10,000 healthy unrelated Indians from 83 populations, addressing underrepresentation in global genomics. [S13] |
| **YLOPD** | Reusable neurodegeneration pipeline ideas: multimodal visit handling, brain/retinal imaging, genomics, biomarkers, longitudinal architecture. | Alzheimer's/dementia model training labels; AD outcome validation. | CBR's YLOPD is a Parkinson's disease longitudinal study with multimodal clinical, cognitive, brain/retinal imaging, GWAS/WGS, blood biomarkers, and exposure data; it is infrastructure-relevant, not a dementia target cohort. [S14] |

The practical dataset strategy is: **OASIS for quick open imaging prototypes; ADNI for biomarker-rich modelling; NACC and AIBL for external validation; UK Biobank for population-risk and blood/genetic context if access permits; ADDI for data-discovery and harmonisation logic; TLSA/SANSCOG for the Indian core; GenomeIndia for ancestry-aware genomics; YLOPD only as reusable neurodegeneration infrastructure inspiration.**

## Ranked project ideas

1. **BRAIN-INDIA Progression Studio - flagship recommendation.**  
   Build a harmonised multimodal system that predicts 1-year, 3-year, and 5-year risk of MCI/dementia conversion; forecasts cognitive trajectories; retrieves matched historical trajectories; and reports uncertainty, calibration, interpretability, and subgroup performance. This directly maps to the challenge brief and CBR inventory. [S1, S2]

2. **Cross-cohort harmonisation toolkit for dementia AI.**  
   Build a common data model, data dictionary mapper, visit aligner, unit normalizer, missingness dashboard, train-only harmonisation pipeline, and cohort comparability reports. This is scientifically valuable but weaker as a standalone unless paired with prediction outputs. [S8, S9, S19]

3. **OCT/OCTA-augmented low-cost triage model.**  
   Use retinal OCT/OCTA and fundus-derived tabular parameters as an augmentation layer beside cognition, blood biochemistry, demographics, and MRI. This is attractive for scalable public-health screening, especially because the CBR inventory includes substantial eye/vision data, but retinal biomarkers should not be the sole flagship claim. [S2]

4. **GenomeIndia-aware dementia genomics sidecar.**  
   Build a genomics module with APOE, ancestry PCs, curated AD loci, pathway/burden features, and Indian allele-frequency annotation. This is a strong differentiator for a bioinformatics team but should support the multimodal model rather than replace it. [S13]

5. **Federated/private analytics template for CBR data.**  
   Package the workflow so raw CBR data never leave approved environments: local training scripts, aggregate-only exports, synthetic demo rows, privacy checks, and data lineage. This aligns with the DUA and would impress on governance, but needs a modelling layer to feel complete. [S3, S8]

6. **Long-term frontier: raw multimodal foundation model or generative patient twin.**  
   This is interesting but not the right core for a B.Tech competition team. It requires large curated raw imaging data, heavy compute, rigorous privacy controls, and external clinical validation. Treat it as a stretch goal after the tabular longitudinal platform is working. [S3, S17]

### Evidence strength map

| Evidence tier | What is strongest | What this means for the build |
|---|---|---|
| **Strong, near-term** | Longitudinal cognition, diagnosis/state transitions, structured clinical variables, extracted MRI phenotypes, blood biochemistry, and standard risk factors. | Make these the base model and primary validation target. [S2, S4, S5, S9, S10] |
| **Moderate, near-term** | Multimodal late fusion, survival prediction, dynamic landmarking, calibration, subgroup reporting, and external cohort validation. | Use these to make the project scientifically credible and competition-ready. [S9, S17] |
| **Promising but emerging** | OCT/OCTA as a retinal augmentation channel; omics beyond APOE/ancestry/pathway summaries; twin-lite trajectory retrieval. | Include as differentiators, but avoid standalone diagnostic claims. [S2, S13] |
| **Long-term frontier** | Raw 3D MRI/OCT foundation models, causal treatment simulators, fully generative patient twins, and genome-wide discovery in small Indian dementia cohorts. | Present only as stretch goals or future work unless validation evidence is unusually strong. [S3, S17, S18] |

## Deep dive on the best flagship project

### Why this project is the best technical fit

BRAIN-INDIA Progression Studio is feasible because the CBR data are already described as extracted/tabulated multimodal parameters, not a mandate to process every MRI/OCT frame from raw pixels. That lets the team spend the competition window on higher-value work: defining outcomes, harmonising features, preventing leakage, training strong baselines, adding survival/progression models, and building an interpretable demo. [S1, S2]

The best endpoint set is:

- **Incident MCI** from cognitively unimpaired baseline.
- **Incident dementia** or AD-type dementia from MCI or high-risk baseline.
- **Time-to-conversion** when event dates or visit intervals support survival analysis.
- **Future cognitive trajectory** over 1, 3, and 5 visits using HMSE/MMSE/MoCA/CDR-style or harmonized cognitive composites.
- **Meaningful decline** thresholds defined before model training, such as drop in a cognitive composite or transition in clinical state.

The key engineering insight: do not force a single "complete-case" model. The winning model should accept missing modalities gracefully. A participant may have cognition and blood data but no MRI; another may have OCT/OCTA and cognitive data; another may have genomics but sparse imaging. Late fusion, modality-specific encoders, missingness indicators, and calibrated model stacking are more robust than an all-or-nothing deep net.

### Why this project is the best scientific fit

Dementia is longitudinal, heterogeneous, and multimodal. ADNI, NACC, AIBL, OASIS, and the CBR cohorts all reinforce the same lesson: clinically meaningful AI should model change over time, not only classify AD versus control at one visit. ADNI was designed around biomarker validation for AD clinical trials; NACC's UDS is annual and longitudinal; AIBL collects repeated biomarker, imaging, cognitive, mood, and lifestyle data; TLSA/SANSCOG are explicitly built around long-term cognitive aging and dementia risk factors in India. [S4, S5, S9, S10, S11, S12]

The scientifically honest version of a digital twin is **not** a claim that the system can simulate a person's causal future under arbitrary treatments. The competition-ready version is a **digital twin lite**: a personalized forecast plus matched real trajectories from similar historical participants. This gives the demo intuitive power while staying defensible.

### Why this project is the best strategic fit

Judges are likely to value a project that makes the CBR cohorts central instead of treating them as a final leaderboard dataset. TLSA and SANSCOG let the team make an India-relevant contribution: urban-rural transfer, education/language-aware cognitive modelling, cardiometabolic risk, ancestry-aware genomics, and privacy-respecting deployment. GenomeIndia strengthens this story because it makes naive European-default genetics less acceptable for Indian cohorts. [S2, S11, S12, S13]

The clear call: **do global benchmark training and method development, then demonstrate Indian adaptation and validation.** That is more compelling than a single ADNI classifier with a polished UI.

## Technical architecture

### System overview

BRAIN-INDIA should have five layers:

1. **Secure data layer:** local data loaders, data dictionaries, provenance, cohort-specific ingestion, DUA-compliant no-upload workflow.
2. **Harmonisation layer:** common data model, visit alignment, unit normalization, feature mapping, cohort/site/scanner/device metadata, train-only harmonisation.
3. **Modality encoders:** clinical/cognitive, MRI, OCT/OCTA/fundus, blood biochemistry, genomics/omics, aging-phenotype modules.
4. **Prediction layer:** fixed-horizon risk models, survival models, trajectory forecasts, matched-trajectory retrieval, uncertainty estimation.
5. **Research workbench UI:** participant-level risk explanation, cohort dashboard, missingness view, calibration plots, subgroup fairness, and demo twin-lite panel.

### Common data model

Use a relational or parquet-backed schema, not one giant spreadsheet:

- `participants`: person ID, cohort, recruitment context, sex, age/birth proxy, education, language, socioeconomic fields if allowed, ancestry fields if available.
- `visits`: visit ID, person ID, visit date or offset, wave number, follow-up interval, diagnosis/state, censoring/event status.
- `modality_features`: person ID, visit ID, modality, feature name, value, unit, source table, derivation status, QC flag.
- `outcomes`: endpoint, anchor visit, prediction horizon, event indicator, event time/window, future cognitive score.
- `data_dictionary`: variable definitions, units, allowed values, missing codes, cohort-specific mappings.
- `provenance`: extraction version, script hash, data version/date, cohort, access constraints.

### Modelling blueprint

**Structured MRI.** Start with extracted imaging phenotypes: hippocampal volume, entorhinal/cortical thickness, ventricular volume, WMH burden, regional volumes, diffusion summaries, fMRI summaries if available, PET SUVR/centiloid only in public cohorts where available. Use raw MRI only as a stretch goal. NACC SCAN's analysis results include brain volumes, cortical thickness, surface area, SUVRs, QC summaries, and defaced images, which validates an extracted-feature-first approach. [S9]

**OCT/OCTA and fundus.** Use retinal thickness, layer measures, vessel density, FAZ, angiography summaries, fundus-derived features, eye-level QA, and device metadata. Treat this as an augmentation channel that may improve low-cost triage, not as a standalone dementia diagnosis engine. The CBR inventory makes this attractive because eye/vision data are substantial, especially in TLSA. [S2]

**Cognition.** Build both cohort-specific and harmonized features. Keep raw test scores where interpretable, but also create domain composites: global cognition, attention, executive function, language, memory, visuospatial ability, functional status, and CDR/HMSE/MMSE/MoCA-style state. Adjust interpretation for education, language, and rural/urban context.

**Blood biochemistry.** Use fasting lipid profile, glucose, CBC, CRP, liver/kidney function, HbA1c, and cardiometabolic composites. These should be framed as scalable risk and comorbidity features, not AD-specific biomarkers unless AD-specific assays are present. NACC's CSF warning is a useful principle: assay values from different centers and methods cannot be blindly pooled without harmonisation. [S2, S9]

**Omics and genomics.** Use a conservative layer: APOE if available; ancestry PCs; curated AD loci; pathway/burden scores; variant QC; GenomeIndia-aware allele-frequency annotation; and omics modules only when sample size supports them. Do not lead with a genome-wide black-box model. GenomeIndia's value is reference and calibration, not direct dementia labels. [S13]

### Harmonisation plan

1. **Define target outcomes before harmonisation.** Decide the anchor visits, horizons, conversion definitions, and cognitive decline thresholds before feature engineering to reduce label leakage.
2. **Create a variable mapping book.** Each feature gets a canonical name, unit, cohort-specific source variable, missing-code logic, and allowed transforms.
3. **Align visits by anchor time.** Convert dates into visit offsets from baseline, age at visit, and prediction windows; never let post-anchor data enter pre-anchor features.
4. **Normalize units and ranges.** Explicitly convert lab units, imaging volumes, retinal measures, and cognitive scales; retain original values for audit.
5. **Model missingness, do not erase it.** Use missingness indicators, modality availability flags, and model architectures that degrade gracefully.
6. **Handle site/scanner/device effects.** Use metadata-aware residualization or ComBat-style harmonisation fitted only on training data. Preserve biological covariates such as age and sex. [S19]
7. **Perform cohort-aware validation.** A harmonisation method is only credible if it improves or preserves performance on held-out cohorts, not only internal cross-validation.
8. **Expose diagnostics.** The demo should show missingness heatmaps, distribution shifts, feature overlap, and post-harmonisation checks.

### Risk prediction plan

Use three tiers:

- **Tier 1 transparent baselines:** logistic regression, elastic net, Cox proportional hazards, simple landmark models.
- **Tier 2 strong tabular ML:** LightGBM/XGBoost/CatBoost, random survival forests, gradient boosting survival models.
- **Tier 3 longitudinal models:** mixed-effects models, dynamic landmarking, GRU-D or small temporal transformer only after baselines are strong.

Fixed-horizon outputs should include risk at 1, 3, and 5 years or visits, depending on available follow-up. Survival outputs should include time-to-conversion and censoring-aware metrics.

### Progression modelling plan

Forecast future cognitive trajectory as a band, not a single number. Start with mixed-effects models and gradient-boosted regressors for future cognitive composite scores. Then add a sequence model if the visit density supports it. The result should show:

- expected future cognitive score;
- 50% and 90% uncertainty intervals;
- top contributing modalities/features;
- matched historical trajectories;
- warnings when the participant is outside the training distribution.

### Digital twin lite plan

The twin-lite module should be retrieval plus calibrated forecasting:

1. Embed each anchor visit into a harmonized latent space using selected multimodal features.
2. Retrieve top-k similar historical participants from training cohorts using strict train/test separation.
3. Display their observed downstream trajectories as "matched histories."
4. Overlay the model's predicted risk and cognitive trajectory band.
5. Report similarity confidence and missing-modality caveats.

This is much safer than a generative synthetic patient twin. It is also easier to validate: the model can be judged by calibration, nearest-neighbor outcome consistency, and external trajectory error.

## Genomics and the GenomeIndia advantage

GenomeIndia changes the design in three ways. First, it makes Indian ancestry and population structure a first-class modelling issue. Second, it gives a principled reason not to blindly import European-derived polygenic risk scores. Third, it lets the team build a bioinformatics differentiator without overclaiming causal genetics. [S13]

The near-term genomics module should include:

- ancestry-aware QC: call rate, relatedness, sex check, heterozygosity, ancestry PCs;
- APOE encoding where available;
- curated AD risk loci and pathway-level summaries;
- rare-variant burden only for genes/pathways with enough sample support;
- Indian allele-frequency annotation using GenomeIndia resources when allowed;
- interaction checks with age, sex, education, rural/urban status, and cardiometabolic risk;
- calibration comparison: model without genetics, model with APOE only, model with conservative genomics, and model with omics sidecars.

Avoid claiming that GenomeIndia is an Alzheimer's cohort. It is a population genomics reference. Its role is to make Indian cohort genomics more accurate, interpretable, and ethically defensible. [S13]

## Validation and judging risks

### Evaluation design

Use a biomedical AI evaluation panel, not a leaderboard-only panel:

- **Fixed-horizon prediction:** AUROC, AUPRC, sensitivity/specificity at clinically meaningful thresholds, balanced accuracy, decision-curve analysis if feasible.
- **Survival modelling:** C-index, time-dependent AUC, integrated Brier score, calibration by time horizon.
- **Trajectory forecasting:** MAE/RMSE, correlation, prediction interval coverage, calibration of uncertainty intervals.
- **Calibration:** calibration slope, calibration-in-the-large, ECE/Brier score, reliability plots by cohort and horizon.
- **External validation:** hold out at least one public cohort and, when allowed, report TLSA/SANSCOG separately rather than only pooled CBR results.
- **Subgroup fairness:** sex, age band, education, cohort, rural/urban, baseline cognitive state, modality availability, and ancestry/genetic strata where allowed.
- **Interpretability:** SHAP or permutation importance by feature and modality; patient-level explanation; matched-trajectory evidence.
- **Uncertainty:** confidence intervals, conformal/quantile bands, out-of-distribution warnings.
- **Leakage prevention:** participant-level splits, anchor-time feature windows, no future visits in features, train-only imputation/harmonisation, no scaling on full data, and no near-duplicate leakage across visits.

TRIPOD+AI-style reporting is useful as a checklist for transparent prediction-model development and validation, especially around data sources, participants, outcomes, predictors, sample size, missing data, model specification, performance, calibration, and code reproducibility. [S17]

### What judges will likely value most

Judges will likely reward:

- a clear India-first reason the project matters;
- serious use of TLSA/SANSCOG rather than public data only;
- multimodal longitudinal modelling that matches the challenge wording;
- strong baselines and external validation;
- calibration and uncertainty, not just headline accuracy;
- privacy and DUA compliance;
- a clean demo that explains "why this risk, why this trajectory, and what similar participants did";
- a reproducible pipeline with documented data dictionaries, model cards, and benchmark reports.

### What to avoid

Avoid these as the core submission:

- a generic dementia chatbot;
- AD-vs-control classifier on one public dataset;
- raw MRI/OCT foundation model from scratch;
- full causal digital twin or treatment simulator without validation;
- European PRS transplanted to Indian cohorts without recalibration;
- complete-case-only modelling that discards most participants;
- leakage through future visits, full-data normalization, or subject-overlapping splits;
- claims of clinical diagnosis or screening deployment when the system is a research tool;
- uploading participant-level CBR data to public AI services or any non-approved environment. [S3]

## Execution roadmap

Assume a 5-person B.Tech team:

- **Lead/data architect:** common data model, ingestion, data dictionary, reproducibility.
- **ML lead:** baselines, boosting, survival models, calibration.
- **Longitudinal lead:** trajectory forecasting, dynamic landmarking, twin-lite retrieval.
- **Bioinformatics lead:** genomics QC, APOE/loci/pathways, GenomeIndia-aware annotation.
- **Product/evaluation lead:** dashboard, model cards, subgroup/fairness reports, final demo.

| Week | Milestone | Deliverables |
|---|---|---|
| 1 | Scope lock and ethics setup | Final endpoints, DUA-safe workflow, repository skeleton, source/data inventory, risk register. |
| 2 | Data model and dictionaries | Participant/visit/modality/outcome schema, variable mapping template, missingness codebook. |
| 3 | Public cohort prototype | OASIS/ADNI starter extraction, baseline cognitive/MRI model, subject-level split tests. |
| 4 | CBR schema adapter | TLSA/SANSCOG ingestion adapters, inventory-aligned feature groups, secure local run instructions. |
| 5 | Harmonisation v1 | Unit normalization, visit alignment, train-only imputation/scaling, missingness dashboard. |
| 6 | Baseline risk models | Logistic/elastic-net/Cox/LightGBM models, AUROC/AUPRC/C-index, first calibration plots. |
| 7 | Progression models | Mixed-effects and gradient-boosted trajectory forecasts, interval coverage reports. |
| 8 | Multimodal fusion | Modality-specific encoders, missing-aware stacking, ablation by modality and cohort. |
| 9 | Twin-lite retrieval | Similar-patient embedding, top-k trajectory panel, retrieval validation, OOD warnings. |
| 10 | Genomics sidecar | APOE/PC/pathway module, GenomeIndia-aware notes, genetics ablation and calibration check. |
| 11 | External validation and fairness | Held-out cohort tests, TLSA vs SANSCOG reporting, sex/age/education/rural-urban subgroup panels. |
| 12 | Demo and final package | Streamlit/FastAPI demo, model card, benchmark checklist, slide-ready figures, reproducible command script. |

## Winning Build Recommendation

**Exact project name:** BRAIN-INDIA Progression Studio.

**One-line scope:** A privacy-aware, harmonised multimodal workbench that predicts dementia/MCI risk, forecasts cognitive progression, and displays matched longitudinal "twin-lite" trajectories for Indian brain-aging research.

**System architecture:** Local secure data ingestion; common data model; train-only harmonisation; modality-specific feature encoders; fixed-horizon risk models; survival models; trajectory forecasting; similar-participant retrieval; dashboard with calibration, explanations, uncertainty, and subgroup reports.

**Novelty angle:** The novelty is not "AI predicts dementia." The novelty is **Indian-cohort-aware longitudinal multimodal dementia modelling with harmonisation, uncertainty, and twin-lite retrieval**, grounded in TLSA/SANSCOG and strengthened by public cohorts and GenomeIndia-aware genomics.

**Demo plan:** Show three synthetic/de-identified participant profiles:

1. Urban TLSA-like participant with cognition, blood, MRI, OCT/OCTA, and genomics.
2. Rural SANSCOG-like participant with cognition, clinical, blood, spirometry, and partial imaging.
3. Public-cohort ADNI/NACC-like participant for external benchmark comparison.

For each, the demo should show current risk, 1/3/5-year conversion risk, predicted cognitive trajectory, uncertainty band, top feature drivers, missing-modality caveats, and matched historical trajectories.

**Minimum viable version:** A working local pipeline that:

- ingests at least one public cohort plus a CBR-schema mock or approved sample;
- maps features into the common data model;
- trains baseline fixed-horizon and survival models;
- reports calibration and subgroup metrics;
- produces a twin-lite matched-trajectory panel;
- includes DUA-safe demo data and documentation.

**Stretch goals:** OCT/OCTA ablation, genomics sidecar, federated/ADDI-compatible packaging, advanced temporal model, raw-image feature extraction, and richer dashboard deployment.

## Appendix: tool stack, repo structure, and benchmark checklist

### Suggested tool stack

- **Data:** Python, pandas, polars, pyarrow, pydantic, pandera, DuckDB.
- **ML:** scikit-learn, LightGBM/XGBoost/CatBoost, lifelines, scikit-survival, statsmodels.
- **Longitudinal:** mixed-effects models, landmarking utilities, PyTorch for GRU-D/temporal transformer stretch.
- **Harmonisation:** neuroCombat/neuroHarmonize-style ComBat workflows, custom residualization, cohort shift diagnostics. [S19]
- **Genomics:** PLINK2, bcftools, Hail for larger-scale public genomics, pandas/pyarrow for feature sidecars.
- **Explainability:** SHAP, permutation importance, partial dependence, feature-group attribution.
- **Uncertainty:** bootstrapping, conformal prediction, quantile regression, calibration curves.
- **Dashboard:** Streamlit or Dash for speed; FastAPI if an API demo is needed.
- **Reproducibility:** DVC or git-annex for metadata, Docker/conda/uv, Makefile or nox, model cards, data cards.

### Suggested repo structure

```text
brain-india-progression-studio/
  README.md
  configs/
    cohorts/
    features/
    endpoints/
  data_dictionary/
    common_data_model.yaml
    variable_mapping_template.csv
  src/
    ingest/
    harmonize/
    features/
    models/
    evaluation/
    twin_lite/
    privacy/
    dashboard/
  notebooks/
    01_data_audit.ipynb
    02_baseline_models.ipynb
    03_external_validation.ipynb
  reports/
    data_inventory_audit.md
    model_card.md
    benchmark_report.md
  tests/
    test_no_future_leakage.py
    test_subject_split.py
    test_harmonization_train_only.py
    test_schema_validation.py
  app/
    streamlit_app.py
```

### Benchmark checklist

- [ ] Participant-level splits, no visit leakage.
- [ ] Prediction anchor and horizon defined before feature extraction.
- [ ] Imputation/scaling/harmonisation fitted only on training data.
- [ ] Missingness and modality availability reported.
- [ ] AUROC/AUPRC for fixed horizons.
- [ ] C-index, time-dependent AUC, integrated Brier score for survival.
- [ ] Calibration slope/intercept and reliability plots.
- [ ] External cohort validation.
- [ ] TLSA and SANSCOG reported separately when available.
- [ ] Sex, age, education, cohort, rural/urban subgroup metrics.
- [ ] Feature and modality ablations.
- [ ] SHAP/permutation explanations.
- [ ] Uncertainty intervals and OOD warnings.
- [ ] DUA-compliant data handling.
- [ ] Model card and data card.
- [ ] Reproducible run command.

## Sources

**Local challenge and CBR-provided files**

- **S1.** Challenge brief and goal objective, local attachment: `C:\Users\adity\.codex\attachments\2327bc2b-a372-419d-8ff6-a57c65ef8bba\goal-objective.md` and `pasted-text-1.txt`.
- **S2.** CBR AI Challenge Dataset Inventory, local PDF: `C:\Users\adity\OneDrive\Desktop\IISc\CBR\Hackathon\Dataset_Inventory_AI_Challenge.pdf`.
- **S3.** CBR Data Use Agreement, local PDF: `C:\Users\adity\OneDrive\Desktop\IISc\CBR\Hackathon\Data_Use_Agreement.pdf`.

**Official cohort and platform sources**

- **S4.** ADNI home page and data overview: [ADNI](https://adni.loni.usc.edu/) and [ADNI Data](https://adni.loni.usc.edu/data-samples/adni-data/).
- **S5.** ADNI cohort definitions: [Study Cohort Information](https://adni.loni.usc.edu/data-samples/adni-data/study-cohort-information/).
- **S6.** UK Biobank: [home](https://www.ukbiobank.ac.uk/), [about data](https://www.ukbiobank.ac.uk/about-our-data/), [types of data](https://www.ukbiobank.ac.uk/about-our-data/types-of-data/), [Research Analysis Platform](https://www.ukbiobank.ac.uk/use-our-data/research-analysis-platform/).
- **S7.** OASIS Brains: [Open Access Series of Imaging Studies](https://sites.wustl.edu/oasisbrains/).
- **S8.** Alzheimer's Disease Data Initiative: [ADDI home](https://www.alzheimersdata.org/).
- **S9.** NACC: [home](https://www.naccdata.org/), [longitudinal UDS data](https://www.naccdata.org/about-nacc-data/longitudinal-neurocognitive-and-clinical-phenotype-data/), [imaging data](https://www.naccdata.org/about-nacc-data/imaging-data/), [fluid biomarker data](https://www.naccdata.org/about-nacc-data/fluid-biomarker-data/), [genetic/genomic data](https://www.naccdata.org/about-nacc-data/genetic-and-genomic-data/).
- **S10.** AIBL: [Australian Imaging, Biomarker and Lifestyle Study of Ageing](https://aibl.org.au/).
- **S11.** CBR-TLSA: [CBR-TATA Longitudinal Study of Aging](https://cbr-iisc.ac.in/tlsa/).
- **S12.** CBR-SANSCOG: [Srinivaspura Aging, Neuro Senescence and Cognition Study](https://cbr-iisc.ac.in/sanscog/).
- **S13.** GenomeIndia: [CBR GenomeIndia page](https://cbr-iisc.ac.in/genomeindia/) and Bhattacharyya et al., 2025, [Nature Genetics: Mapping genetic diversity with the GenomeIndia project](https://www.nature.com/articles/s41588-025-02153-x).
- **S14.** CBR-YLOPD: [Young and Late-onset Parkinson's Disease Study](https://cbr-iisc.ac.in/ylopd-study/).

**Clinical, reporting, and methods guidance**

- **S15.** Alzheimer's Association Workgroup, 2024: [Revised criteria for diagnosis and staging of Alzheimer's disease](https://alz-journals.onlinelibrary.wiley.com/doi/10.1002/alz.13859).
- **S16.** Alzheimer's Association diagnostic criteria portal: [Criteria for diagnosis and staging of Alzheimer's disease](https://www.alz.org/research/for_researchers/diagnostic-criteria-guidelines).
- **S17.** TRIPOD+AI Statement, BMJ 2024: [Transparent reporting of prediction models using regression or machine learning](https://www.bmj.com/content/385/bmj-2023-078378).
- **S18.** Hancerliogullari Koksalmis et al., 2025 preprint survey: [Artificial Intelligence for Personalized Prediction of Alzheimer's Disease Progression](https://arxiv.org/abs/2504.21189).
- **S19.** Fortin et al., 2018, NeuroImage: [Harmonization of cortical thickness measurements across scanners and sites](https://doi.org/10.1016/j.neuroimage.2017.11.024).
