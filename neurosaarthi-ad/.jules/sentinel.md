## 2023-10-25 - Prevented XSS in Streamlit Markdown Cards
**Vulnerability:** The `_risk_card` component in `dashboards/streamlit_app.py` rendered dynamic patient risk factors (like `horizon`, `risk`, `lower`, `upper`) into HTML structures using Streamlit's `unsafe_allow_html=True` without prior escaping.
**Learning:** `unsafe_allow_html=True` acts as a direct injection vector. While the synthetic runtime typically outputs controlled floats, any upstream validation failure or adversarial input modifying these values in a dashboard state could execute arbitrary HTML/JavaScript (CWE-79).
**Prevention:** Always wrap dynamically interpolated string variables with `html.escape()` before injecting them into HTML templates rendered by `st.markdown(..., unsafe_allow_html=True)`, even if the intended type is numeric.

## 2024-05-20 - Prevented Cross-Session State Corruption in Streamlit Cache (CWE-374 / CWE-662)
**Vulnerability:** The `ImmutableRuntimeProxy` for cached objects in `dashboards/streamlit_app.py` previously only protected dictionaries using `types.MappingProxyType`. This allowed lists, sets, and Pandas DataFrames to be modified in-place within the global Streamlit cache by any session.
**Learning:** Shared mutable state in Streamlit's `@st.cache_resource` leads to severe cross-session contamination (CWE-374) and data leakage between concurrent users, as one user's action could mutate the globally cached Pandas DataFrame or list.
**Prevention:** Extend defensive proxies to intercept multiple types of mutable sequences/collections. Always return immutable equivalents (like `tuple` or `frozenset`) or a deep copy (`df.copy(deep=True)`) when exposing cached DataFrames or lists to untrusted contexts.

## 2025-02-23 - Prevented Path Traversal in Export Scripts
**Vulnerability:** The `scripts/export_models.py` script constructed model export paths using unvalidated dynamic inputs (`modality` and `model_name`), leaving it vulnerable to Path Traversal (CWE-22) if untrusted configurations were processed.
**Learning:** Even internal helper scripts must sanitize dynamically generated file paths (using `Path(name).name`) and perform strict bounds-checking against the intended target directory to prevent adversarial configurations from writing artifacts outside of restricted scopes.
**Prevention:** Sanitize dynamic filename components by stripping directory parts using `Path(name).name`, and enforce explicit path resolution validation by ensuring the resolved path is within the intended base directory using `path.resolve().is_relative_to(base_dir)`.

## 2024-10-25 - Securing Streamlit Configuration
**Vulnerability:** The application was lacking a `.streamlit/config.toml` file to harden the application. Specifically, XSRF protection was not explicitly enabled and client-side error details (which could leak stack traces and server-side information) were not disabled.
**Learning:** In production, Streamlit apps can expose internal error tracebacks to the client if not explicitly configured to hide them. Also, XSRF protection is a critical defense mechanism that should be explicitly enabled.
**Prevention:** Always include a `.streamlit/config.toml` in Streamlit deployments with `enableXsrfProtection = true` and `showErrorDetails = "none"`. Do not explicitly disable CORS.
