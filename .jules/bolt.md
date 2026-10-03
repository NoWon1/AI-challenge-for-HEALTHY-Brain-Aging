## Performance Optimizations

* **Groupby Iteration Avoidance**: When building metrics or iterating over Pandas `groupby` objects (e.g., in `_build_validation_summary`), avoid calling aggregating methods like `.mean()` or `.size()` inside the python loop. Extract them out using `event_rates = grouped["event_by_3y"].mean()` and query the returned Series by its index within the loop. Doing so on a ~10k row dataset over 100 groups can result in a ~2x performance speedup as it avoids repeatedly dropping into python and re-evaluating DataFrame slices.
