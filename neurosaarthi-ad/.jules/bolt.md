# Performance Learnings

## Repeated DataFrame Instantiation & Iterative Mutation
Avoid instantiating empty Pandas DataFrames and mutating them incrementally (e.g., `df[col] = values`) inside tight inner loops (like bootstrap iterations or cross-validation folds). Pandas triggers expensive index alignment, metadata tracking, and block consolidation on every insertion.

Instead:
1. Hoist redundant calculations (like generating boolean masks) outside the loop.
2. Accumulate values into a standard Python dictionary (`score_dict[col] = values`) inside the loop.
3. Instantiate the DataFrame exactly once per iteration with the completed dictionary (`pd.DataFrame(score_dict, index=...)`).

In our demonstration runtime, this pattern reduced late-fusion processing overhead by ~25% (e.g., from ~1.63s to ~1.19s for large simulated batch iterations) while preserving perfect numerical equivalence and skipping the need for a complex full-NumPy rewrite.
