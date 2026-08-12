## 2024-05-18 - Replacing Pandas `.iterrows()` in cohort event building
**Learning:** Found a major bottleneck in `etl/base.py`'s `_build_outcomes` method, where `.iterrows()` was used to simulate time-series progression logic inside a `.groupby()`. This O(n²) anti-pattern made dataset extraction extremely slow.
**Action:** Replaced `.iterrows()` with vectorized sorting, boolean masking, and cross joins for horizons. This pattern can be applied to other adapters if `.iterrows()` usage is found.

## 2025-02-18 - ETL Anti-Pattern
**Learning:** Codebase Anti-Pattern/Convention: Avoid using Pandas `.iterrows()` in the ETL adapters (e.g., `etl/base.py`, cohort adapters). Iterating over DataFrames is extremely slow in python compared to utilizing numpy underneath.
**Action:** Replace iteration with idiomatic, vectorized Pandas operations (like `.merge()`, `combine_first()`, and boolean masking) to prevent severe performance bottlenecks.
