# Model Card: Digital Twin Lite

## Intended Use
Retrieval of matched synthetic trajectories for a given baseline profile to aid visualization and case comparison.

## Model Details
- **Architecture**: Mahalanobis-distance nearest neighbor retrieval.
- **Input**: Baseline multimodal feature vectors.

## Training Data
Indexed from the synthetic demo cohort database.

## Evaluation
- **Metrics**: Retrieval latency, matching quality (distance distribution of retrieved neighbors).

## Limitations
- Mahalanobis distance requires robust covariance estimation, which can be sensitive to outliers.
- Only retrieves synthetic matches; not for clinical matching.
