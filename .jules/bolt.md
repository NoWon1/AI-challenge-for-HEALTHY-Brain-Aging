## Performance Optimizations

* **Groupby Iteration Avoidance**: When building metrics or iterating over Pandas `groupby` objects (e.g., in `_build_validation_summary`), avoid calling aggregating methods like `.mean()` or `.size()` inside the python loop. Extract them out using `event_rates = grouped["event_by_3y"].mean()` and query the returned Series by its index within the loop. Doing so on a ~10k row dataset over 100 groups can result in a ~2x performance speedup as it avoids repeatedly dropping into python and re-evaluating DataFrame slices.

## 2026-10-05 - Vectorizing index sampling in Bootstrap CI
**Learning:** Calling `rng.choice()` sequentially inside a bootstrap loop to generate resample indices introduces an O(B) Python interpreter overhead, which can be costly for large numbers of bootstrap iterations (e.g., B=1000).
**Action:** Pre-allocate the entire (B x N) array of indices with a single NumPy vectorized call: `all_indices = rng.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)`, and index into this array within the loop. This reduces the sampling time by roughly 4x.
