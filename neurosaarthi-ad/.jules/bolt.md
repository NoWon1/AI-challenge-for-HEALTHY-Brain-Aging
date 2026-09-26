## Repeated DataFrame Instantiation in Tight Loops

**Anti-Pattern:** Repeatedly creating an empty pandas DataFrame inside a loop and populating it column-by-column causes massive CPU and memory allocation overhead.
```python
for _ in range(bootstrap_iterations):
    for _ in range(horizons):
        score_frame = pd.DataFrame(index=frame.index)
        for modality, distribution in modality_distributions.items():
            score_frame[modality] = values # Adding columns one by one
```

**Optimization:** Accumulate the column data inside a standard Python dictionary of NumPy arrays, then instantiate the pandas DataFrame only once before it is needed.

```python
for _ in range(bootstrap_iterations):
    for _ in range(horizons):
        score_dict = {}
        for modality, distribution in modality_distributions.items():
            score_dict[modality] = values
        score_frame = pd.DataFrame(score_dict, index=frame.index) # Create once
```
*Measurements*: In an isolated micro-benchmark tracking runtime fusion, moving DataFrame instantiation outside the inner modality loop yielded a >20% reduction in loop time (e.g., 0.70s -> 0.55s), bypassing significant Pandas validation overhead.
