## 2024-05-18 - Replacing Pandas `.iterrows()` in cohort event building
**Learning:** Found a major bottleneck in `etl/base.py`'s `_build_outcomes` method, where `.iterrows()` was used to simulate time-series progression logic inside a `.groupby()`. This O(n²) anti-pattern made dataset extraction extremely slow.
**Action:** Replaced `.iterrows()` with vectorized sorting, boolean masking, and cross joins for horizons. This pattern can be applied to other adapters if `.iterrows()` usage is found.
