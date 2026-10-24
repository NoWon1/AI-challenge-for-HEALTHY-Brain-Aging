## Performance Optimizations

* **Groupby Iteration Avoidance**: When building metrics or iterating over Pandas `groupby` objects (e.g., in `_build_validation_summary`), avoid calling aggregating methods like `.mean()` or `.size()` inside the python loop. Extract them out using `event_rates = grouped["event_by_3y"].mean()` and query the returned Series by its index within the loop. Doing so on a ~10k row dataset over 100 groups can result in a ~2x performance speedup as it avoids repeatedly dropping into python and re-evaluating DataFrame slices.
## 2024-05-18 - Avoid iterative random generation in bootstrap loops
**Learning:** Calling `rng.choice()` sequentially inside a bootstrap loop to generate resample indices introduces an O(B) python interpreter overhead and significantly slows down bootstrap confidence interval calculations and comparisons.
**Action:** Replace with a single vectorized numpy call to pre-allocate the entire array of indices (e.g., `all_indices = rng.integers(0, n_samples, size=(n_bootstrap, n_samples))`) before the loop, and slice from it during the loop.

## 2026-10-15 - Vectorized structured array creation for survival models
**Learning:** Creating `sksurv` structured arrays (e.g., `dtype=[("event", bool), ("time", float)]`) by combining parallel NumPy arrays with a Python list comprehension and `zip()` inside `np.array()` introduces a significant O(N) overhead in survival model data preparation.
**Action:** Replace `list(zip(...))` inside `np.array()` with vectorized pre-allocation (`np.empty(N, dtype=...)`) and direct column assignments (`y["event"] = ...`) for massive performance speedups.

## 2024-10-24 - Vectorize repeated 1D interpolation
**Learning:** Using `np.interp` inside a Python loop or list comprehension over multiple evaluation points (e.g., target years for trajectory interpolation) incurs significant interpreter overhead.
**Action:** Vectorize the 1D interpolation by passing the entire array of evaluation points as the first argument to `np.interp`, and combine with vectorized reductions like `np.std(axis=0)` to process all targets simultaneously, eliminating the loop entirely.
