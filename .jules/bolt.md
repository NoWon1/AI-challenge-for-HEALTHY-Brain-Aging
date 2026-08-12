## 2026-08-07 - [Vectorizing MixedEffectsTrajectory]
**Learning:** Iterating over Pandas dataframes using iterrows is a huge anti-pattern, especially inside heavily-used prediction loops like MixedEffectsTrajectory.predict. The slowness multiplies when used in combination with row-by-row lookups of statsmodels random effects and dictionary .loc operations.
**Action:** When working with statsmodels output (result.predict) or scikit-learn models applied to Pandas dataframes grouped by participants, always extract the base prediction matrix first (model.predict(X_all)), and then update the specific indices for each group using vectorized operations.
## 2023-10-25 - [Vectorizing CohortAdapter Outcomes]
**Learning:** Using `.iterrows()` combined with nested loops over `.groupby()` to calculate event horizons is a severe O(N²) bottleneck when processing tens of thousands of visits across cohort adapters (AIBL, ADNI, etc).
**Action:** Replace `.iterrows()` with fully vectorized operations for longitudinal event logic. Use `.groupby().first()`, `.merge()`, and boolean masks to locate events, and `.combine_first()` to fall back to follow-up times when events are absent.

## 2025-02-18 - ETL Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in the ETL adapters (e.g., `etl/base.py`, cohort adapters). Iterating over DataFrames is extremely slow in python compared to utilizing numpy underneath.
**Action:** Replace iteration with idiomatic, vectorized Pandas operations (like `.merge()`, `combine_first()`, and boolean masking) to prevent severe performance bottlenecks.
