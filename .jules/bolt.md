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
## 2026-08-26 - Vectorize Pandas iterrows in AIBL & OASIS ETL Adapters
**Learning:** Codebase Anti-Pattern/Convention: Using Pandas `.iterrows()` in ETL adapters (`AiblAdapter` and `OasisAdapter`) creates a significant performance bottleneck.
**Action:** Replaced `.iterrows()` loops with vectorized `.notna()` masking and `pd.concat` operations while tracking `_row_idx` and `_feat_idx` to maintain strict original row ordering. This optimization provides roughly a 10x speedup in parsing clinical feature rows.
## 2026-09-01 - [O(N^2) Kaplan-Meier computation]
**Learning:** Codebase Anti-Pattern/Convention: Avoid computing cumulative metrics (like `at_risk` counts) inside a groupby loop by repeatedly filtering the original dataframe. In `neurosaarthi-ad/models/survival/baseline.py`, the Kaplan-Meier baseline was taking O(N^2) time.
**Action:** Replace `groupby` loops over `time` with a fully vectorized pandas `.agg()` followed by a reverse cumulative sum (`.iloc[::-1].cumsum().iloc[::-1]`). This computes `at_risk` and survival probabilities in O(N) time without Python-level loops, yielding a ~30x speedup for 10k rows.
## 2026-09-02 - [Vectorize pandas apply and loops for grouped monotonicity checks]
**Learning:** Codebase Anti-Pattern/Convention: Avoid using pandas `.apply()` with custom lambdas (like `lambda values: values.is_monotonic_increasing`) or `.tolist()` comparisons in python loops to check for monotonicity over groups. These operations are extremely slow because they drop into pure python for every single group.
**Action:** Replace `df.groupby(...)['col'].apply(...)` or loops checking `values != sorted(values)` with a vectorized approach: `df.groupby(..., sort=False)['col'].diff().fillna(0).ge(0).all()`. This runs in C via numpy/pandas and provides a ~100x+ speedup. In validation contexts where the specific violating group must be identified, use the fast vectorized check as a guard, and only fall back to the slow loop if a violation is detected.
## 2026-09-03 - [Vectorize pandas to_datetime parsing]
**Learning:** Codebase Anti-Pattern/Convention: Avoid using pandas `.apply(lambda x: pd.to_datetime(x))` for parsing dates in dataframes (e.g., in `neurosaarthi-ad/etl/adni/adapter.py`). Iterating row-by-row drops down to Python context and is extremely slow.
**Action:** Replace `df['col'].apply(lambda x: pd.to_datetime(x))` with the vectorized approach `pd.to_datetime(df['col'])`. This runs in C via numpy/pandas, leading to massive speedups (~1100x speedup observed for 100k rows in isolated benchmarking).
