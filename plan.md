1. **Analyze Security Issue**: In `dashboards/streamlit_app.py`, the `ImmutableRuntimeProxy` uses `_secure_value` to secure mutable types like dictionaries, lists, sets, and Pandas DataFrames against being modified. However, `tuple`s are not handled. While tuples are immutable sequences, they can contain mutable elements (e.g., `( [1,2], {'a':1} )`). If a tuple contains a list, returning it as is allows the caller to extract the list and mutate it, which circumvents the cache's immutability protection.
2. **Implement Fix in `_secure_value`**:
   - Add a check for `isinstance(val, tuple)` in `_secure_value`.
   - If it's a namedtuple (`hasattr(val, "_fields")`), reconstruct it with secured elements via `type(val)(*(_secure_value(v) for v in val))`. (Wait, let's use `val._make` if possible, but `type(val)(...)` works as tested).
   - If it's a regular tuple, reconstruct it using `tuple(_secure_value(v) for v in val)`.
3. **Verify with Tests**:
   - Update `tests/test_streamlit_app.py` or run local tests.
4. **Follow pre-commit steps** to ensure all linting and testing passes.
