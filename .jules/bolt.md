## 2025-02-18 - ETL Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in the ETL adapters (e.g., `etl/base.py`, cohort adapters). Iterating over DataFrames is extremely slow in python compared to utilizing numpy underneath.
**Action:** Replace iteration with idiomatic, vectorized Pandas operations (like `.merge()`, `combine_first()`, and boolean masking) to prevent severe performance bottlenecks.
## 2025-02-20 - [Performance] Vectorize Pandas top-K extraction in evaluation
**Learning:** Using pandas `.iterrows()` for top-K extraction in `evaluation/explainability.py` was a significant performance bottleneck.
**Action:** Replace `.iterrows()` loops in dataframe processing with vectorized NumPy operations. We used `np.argpartition` combined with `np.argsort` to efficiently extract top K features, leading to a massive speedup (~240x in isolated tests).
## 2026-08-17 - Vectorize Pandas iterrows in ADNI ETL Adapter
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` across the codebase, specifically in ETL adapters like `AdniAdapter`.
**Action:** Replaced `.iterrows()` loops in `neurosaarthi-ad/etl/adni/adapter.py` with vectorized `pd.concat` operations, while ensuring original insertion order is preserved by tracking original row index (`_row_idx`) and feature index (`_feat_idx`). This produced an estimated ~16x speedup.
