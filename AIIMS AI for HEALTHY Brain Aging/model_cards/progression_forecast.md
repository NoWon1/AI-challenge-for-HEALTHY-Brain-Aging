# Model Card: Progression Forecast

## Intended Use
Forecasting longitudinal cognitive trajectories for research and hypothesis generation.

## Model Details
- **Architecture**: Mixed-effects trajectory model (random intercept and slope).
- **Features**: Modality-specific drivers and temporal visit anchors.

## Training Data
Synthetic longitudinal visits from the demo cohort.

## Evaluation
- **Metrics**: RMSE, MAE for cognitive composite score predictions at future timepoints.

## Limitations
- Linear mixed-effects assumptions may fail to capture non-linear decline phases.
- Trained on synthetic trajectories.
