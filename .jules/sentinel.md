## 2026-08-15 - [Preventing XSS in Streamlit's Markdown]
**Vulnerability:** Found dynamic variables being injected into `st.markdown(..., unsafe_allow_html=True)` without explicit sanitization, which can expose the application to Cross-Site Scripting (XSS) if the variables were controlled or influenced by users.
**Learning:** Even if the expected types for these variables are numbers (e.g. floats converted to percentages, integers for horizons), failing to explicitly escape them with `html.escape` creates a poor defense-in-depth posture, violating codebase security conventions. Streamlit's `unsafe_allow_html=True` is dangerous without strict controls.
**Prevention:** Always wrap dynamic variables in `html.escape(str(...))` before formatting them into a string to be passed to `st.markdown(..., unsafe_allow_html=True)`, enforcing a consistent XSS prevention pattern.

## 2026-08-16 - [Fixing redundant unescaped string injection in Streamlit]
**Vulnerability:** In `_risk_card` and `_participant_view`, dynamically loaded values from DataFrames were converted to escaped and non-escaped local variables. The HTML strings passed to `st.markdown(..., unsafe_allow_html=True)` interpolated the unescaped variables directly in addition to the escaped ones.
**Learning:** Having both unescaped and escaped variables available locally creates a risk of accidentally using the unescaped versions in the string templates for HTML injection. Redundant unescaped variables should be removed, leaving only the properly sanitized values.
**Prevention:** Avoid defining both unescaped and escaped versions of variables if they are only needed for HTML generation. Explicitly use `html.escape` to define a single set of sanitized variables, and only reference those safe variables during string interpolation.

## 2026-08-22 - [Defense-in-depth Path Traversal Prevention]
**Vulnerability:** The generic CSV data adapter dynamically constructed file paths for loading CSV files using `Path(self.raw_dir) / f"{table_name}.csv"`. While `table_name` is currently static, this pattern lacked explicit containment verification.
**Learning:** Hardcoded strings or fixed loops can still be refactored or exposed in the future. Constructing file paths dynamically without containment checks violates the explicit codebase convention to use `Path.resolve().is_relative_to(base_dir)`. This is a classic defense-in-depth failure.
**Prevention:** Always resolve the base directory and the target path, then explicitly verify containment using `target_path.is_relative_to(base_dir)` before performing file I/O operations in ETL loaders.
