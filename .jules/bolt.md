## 2025-02-18 - ETL Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in the ETL adapters (e.g., `etl/base.py`, cohort adapters). Iterating over DataFrames is extremely slow in python compared to utilizing numpy underneath.
**Action:** Replace iteration with idiomatic, vectorized Pandas operations (like `.merge()`, `combine_first()`, and boolean masking) to prevent severe performance bottlenecks.

## 2025-02-18 - Explainability Evaluation Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in evaluation scripts, particularly for SHAP top-K extraction (`evaluation/explainability.py`). Iterating row-by-row on large SHAP DataFrames is highly inefficient.
**Action:** Replace `iterrows()` with vectorized NumPy operations such as `np.argsort()` and advanced array indexing, which proved to be ~100x faster when sorting feature attributions across many samples.
## 2025-02-20 - [Performance] Vectorize Pandas top-K extraction in evaluation
**Learning:** Using pandas `.iterrows()` for top-K extraction in `evaluation/explainability.py` was a significant performance bottleneck.
**Action:** Replace `.iterrows()` loops in dataframe processing with vectorized NumPy operations. We used `np.argpartition` combined with `np.argsort` to efficiently extract top K features, leading to a massive speedup (~240x in isolated tests).
## 2025-02-23 - [Performance] Vectorize Pandas ETL feature mapping
**Learning:** Using `.iterrows()` to parse rows and then loop through a feature mapping dict in the ETL adapters was a significant performance bottleneck (taking O(N*M) pure Python operations instead of utilizing NumPy/Pandas underlying C arrays).
**Action:** Replace nested `.iterrows()` loops when melting/extracting dataset features with a column-based approach: loop through the feature dictionary and create sub-DataFrames for each mapped feature directly using `raw[source_col].notna()` masks, then concat at the end. By keeping track of an initial index (`_row_idx`), you can sort the concatenated frame back exactly into the original insertion order, giving an order of magnitude speedup.
## 2026-08-17 - Vectorize Pandas iterrows in ADNI ETL Adapter
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` across the codebase, specifically in ETL adapters like `AdniAdapter`.
**Action:** Replaced `.iterrows()` loops in `neurosaarthi-ad/etl/adni/adapter.py` with vectorized `pd.concat` operations, while ensuring original insertion order is preserved by tracking original row index (`_row_idx`) and feature index (`_feat_idx`). This produced an estimated ~16x speedup.
## 2026-08-18 - Vectorize Pandas iterrows in NACC ETL Adapter
**Learning:** Codebase Anti-Pattern/Convention: Using Pandas `.iterrows()` for processing features in ETL adapters like `NaccAdapter` causes significant performance bottlenecks.
**Action:** Replaced the nested `.iterrows()` loops in `neurosaarthi-ad/etl/nacc/adapter.py`'s `_build_features` with vectorized `pd.concat` operations. Maintained strict DataFrame equality checks by tracking original row and feature indices (`_row_idx`, `_feat_idx`). Achieved an estimated ~7x speedup.
## 2024-05-14 - Vectorized feature extraction in AIBL and OASIS adapters
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in ETL adapters. AIBL and OASIS adapters used `.iterrows()` to parse rows and loop through a feature mapping dict, which is O(N*M) pure Python operations instead of using Pandas/NumPy C arrays underneath.
**Action:** Replaced `.iterrows()` in `AiblAdapter` and `OasisAdapter`'s `_build_features` methods with `pd.concat` loops across the `feature_map`. To pass strict dataframe tests, I preserved original sorting by maintaining the original row index (`_row_idx`) and mapping index (`_feat_idx`). Cast identifier `visit_id` to `str` to avoid subtle type coercion errors.
