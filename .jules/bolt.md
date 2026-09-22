## Vectorized Late Fusion Performance Improvement

**What**:
Replaced nested python `for` loops and heavy pandas operations in `_risk_distribution` and `_risk_distribution_baseline` with a fully vectorized numpy implementation in `_vectorized_score_fusion`.

**Why**:
The original implementation looped over each bootstrap iteration and horizon, creating new Pandas DataFrames in memory and using `weighted_score_fusion` which itself did more Pandas operations. This was O(N) complexity in python with large overhead. By stacking the modality distributions and using numpy broadcasting, we do the weighted sum across all modalities, samples, bootstraps, and horizons simultaneously without Python looping or Pandas overhead.

**Performance Gain**:
Benchmarked a mock distribution of 5 modalities over 20 bootstraps, 5000 samples, and 3 horizons. The vectorized implementation executed in 0.0616s compared to the original 0.5666s, achieving a **9.20x speedup** on the fusion logic.

