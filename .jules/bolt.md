## 2025-02-18 - ETL Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in the ETL adapters (e.g., `etl/base.py`, cohort adapters). Iterating over DataFrames is extremely slow in python compared to utilizing numpy underneath.
**Action:** Replace iteration with idiomatic, vectorized Pandas operations (like `.merge()`, `combine_first()`, and boolean masking) to prevent severe performance bottlenecks.

## 2025-02-18 - Explainability Evaluation Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in evaluation scripts, particularly for SHAP top-K extraction (`evaluation/explainability.py`). Iterating row-by-row on large SHAP DataFrames is highly inefficient.
**Action:** Replace `iterrows()` with vectorized NumPy operations such as `np.argsort()` and advanced array indexing, which proved to be ~100x faster when sorting feature attributions across many samples.
## 2025-02-20 - [Performance] Vectorize Pandas top-K extraction in evaluation
**Learning:** Using pandas `.iterrows()` for top-K extraction in `evaluation/explainability.py` was a significant performance bottleneck.
**Action:** Replace `.iterrows()` loops in dataframe processing with vectorized NumPy operations. We used `np.argpartition` combined with `np.argsort` to efficiently extract top K features, leading to a massive speedup (~240x in isolated tests).
