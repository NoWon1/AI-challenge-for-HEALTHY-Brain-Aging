# Model Card: Risk Classification (LightGBM)

## Intended Use
Research prototype for incident dementia risk prediction at 1, 3, and 5-year horizons. Not for clinical use.

## Model Details
- **Architecture**: LightGBM Gradient-Boosted Classifier
- **Hyperparameters**: 500 trees, learning rate = 0.05
- **Input**: Multimodal feature vectors (cognition, MRI, blood, OCT/OCTA, genomics)

## Training Data
Synthetic demo cohort consisting of 840 participants across 7 cohort patterns (ADNI, NACC, AIBL, OASIS, UKB, TLSA, SANSCOG).

## Evaluation
- **Metrics**: AUROC, AUPRC, Brier score
- **Analysis**: Calibration checks and subgroup fairness analysis (sex, age band, education, urban/rural)

## Limitations and Risks
- **Ethical Considerations**: This is a research tool evaluated on synthetic data only. Not intended for diagnosis, screening, treatment decisions, or claims about real cohort performance.
- **Risks**: Potential bias in missing modality imputation.
