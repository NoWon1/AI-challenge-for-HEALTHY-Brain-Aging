## 2024-05-18 - Replacing Pandas `.iterrows()` in cohort event building
**Learning:** Found a major bottleneck in `etl/base.py`'s `_build_outcomes` method, where `.iterrows()` was used to simulate time-series progression logic inside a `.groupby()`. This O(n²) anti-pattern made dataset extraction extremely slow.
**Action:** Replaced `.iterrows()` with vectorized sorting, boolean masking, and cross joins for horizons. This pattern can be applied to other adapters if `.iterrows()` usage is found.
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
