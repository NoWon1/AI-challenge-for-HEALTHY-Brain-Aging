## 2025-02-18 - ETL Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in the ETL adapters (e.g., `etl/base.py`, cohort adapters). Iterating over DataFrames is extremely slow in python compared to utilizing numpy underneath.
**Action:** Replace iteration with idiomatic, vectorized Pandas operations (like `.merge()`, `combine_first()`, and boolean masking) to prevent severe performance bottlenecks.

## 2026-08-13 - Fast Row-Wise Top-K Extraction
**Learning:** Found an anti-pattern in `evaluation/explainability.py` where SHAP top driver extraction used a Pandas `.iterrows()` loop combined with per-row sorting (`row.abs().sort_values(ascending=False).head(k)`). This approach is O(N * M log M) in Python overhead and is severely inefficient (tested to be ~95x slower) for extracting top features from many samples.
**Action:** Always replace row-wise Pandas operations for Top-K extraction with vectorized NumPy operations. Extract `.values`, compute `np.abs()`, use `np.argsort(-abs_vals, axis=1)[:, :k]` to get top column indices, and use advanced indexing (`shap_vals[row_indices, sorted_indices]`) to pull out the actual feature values and names. This avoids the Python loop and is nearly instant.
## 2025-02-18 - Explainability Evaluation Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in evaluation scripts, particularly for SHAP top-K extraction (`evaluation/explainability.py`). Iterating row-by-row on large SHAP DataFrames is highly inefficient.
**Action:** Replace `iterrows()` with vectorized NumPy operations such as `np.argsort()` and advanced array indexing, which proved to be ~100x faster when sorting feature attributions across many samples.
## 2025-02-20 - [Performance] Vectorize Pandas top-K extraction in evaluation
**Learning:** Using pandas `.iterrows()` for top-K extraction in `evaluation/explainability.py` was a significant performance bottleneck.
**Action:** Replace `.iterrows()` loops in dataframe processing with vectorized NumPy operations. We used `np.argpartition` combined with `np.argsort` to efficiently extract top K features, leading to a massive speedup (~240x in isolated tests).
## 2026-08-17 - Vectorize Pandas iterrows in ADNI ETL Adapter
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` across the codebase, specifically in ETL adapters like `AdniAdapter`.
**Action:** Replaced `.iterrows()` loops in `neurosaarthi-ad/etl/adni/adapter.py` with vectorized `pd.concat` operations, while ensuring original insertion order is preserved by tracking original row index (`_row_idx`) and feature index (`_feat_idx`). This produced an estimated ~16x speedup.
