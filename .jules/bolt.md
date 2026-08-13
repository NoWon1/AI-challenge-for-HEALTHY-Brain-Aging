## 2025-02-18 - ETL Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in the ETL adapters (e.g., `etl/base.py`, cohort adapters). Iterating over DataFrames is extremely slow in python compared to utilizing numpy underneath.
**Action:** Replace iteration with idiomatic, vectorized Pandas operations (like `.merge()`, `combine_first()`, and boolean masking) to prevent severe performance bottlenecks.

## 2026-08-13 - Fast Row-Wise Top-K Extraction
**Learning:** Found an anti-pattern in `evaluation/explainability.py` where SHAP top driver extraction used a Pandas `.iterrows()` loop combined with per-row sorting (`row.abs().sort_values(ascending=False).head(k)`). This approach is O(N * M log M) in Python overhead and is severely inefficient (tested to be ~95x slower) for extracting top features from many samples.
**Action:** Always replace row-wise Pandas operations for Top-K extraction with vectorized NumPy operations. Extract `.values`, compute `np.abs()`, use `np.argsort(-abs_vals, axis=1)[:, :k]` to get top column indices, and use advanced indexing (`shap_vals[row_indices, sorted_indices]`) to pull out the actual feature values and names. This avoids the Python loop and is nearly instant.
