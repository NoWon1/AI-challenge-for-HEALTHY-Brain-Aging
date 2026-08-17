## 2026-08-15 - [Preventing XSS in Streamlit's Markdown]
**Vulnerability:** Found dynamic variables being injected into `st.markdown(..., unsafe_allow_html=True)` without explicit sanitization, which can expose the application to Cross-Site Scripting (XSS) if the variables were controlled or influenced by users.
**Learning:** Even if the expected types for these variables are numbers (e.g. floats converted to percentages, integers for horizons), failing to explicitly escape them with `html.escape` creates a poor defense-in-depth posture, violating codebase security conventions. Streamlit's `unsafe_allow_html=True` is dangerous without strict controls.
**Prevention:** Always wrap dynamic variables in `html.escape(str(...))` before formatting them into a string to be passed to `st.markdown(..., unsafe_allow_html=True)`, enforcing a consistent XSS prevention pattern.
## 2026-08-17 - [Defense-in-Depth Path Traversal Protection]
**Vulnerability:** The ETL adapters read files (e.g. `pd.read_csv(path)`) constructed via string concatenation/formatting. Even if currently hardcoded, any future switch to dynamic names could result in arbitrary file read vulnerabilities.
**Learning:** Implementing `Path.resolve().is_relative_to(base_dir)` provides a lightweight, critical defense-in-depth barrier against path traversal without breaking existing logic.
**Prevention:** Enforce strict root-containment validation on all dynamically generated paths before accessing the filesystem.
