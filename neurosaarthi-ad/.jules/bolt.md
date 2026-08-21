
## 2026-08-21 - Vectorized Feature Extraction in ETL Adapters
**Learning:** Using `pandas.iterrows()` in ETL adapters to dynamically generate feature rows is a significant codebase-specific performance bottleneck. While `pd.concat` is a common fix, preserving exact row and feature insertion order requires explicitly tracking `_row_idx` and `_feat_idx` within the vectorized frames. Otherwise, strict dataframe equality tests relying on implicit row generation order will fail.
**Action:** When converting `.iterrows()` row generation patterns to vectorized `pd.concat` sequences, explicitly track original source row indices and column evaluation indices to perfectly match the original deterministic output order.
