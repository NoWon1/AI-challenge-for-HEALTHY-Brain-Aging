## 2023-10-25 - Prevented XSS in Streamlit Markdown Cards
**Vulnerability:** The `_risk_card` component in `dashboards/streamlit_app.py` rendered dynamic patient risk factors (like `horizon`, `risk`, `lower`, `upper`) into HTML structures using Streamlit's `unsafe_allow_html=True` without prior escaping.
**Learning:** `unsafe_allow_html=True` acts as a direct injection vector. While the synthetic runtime typically outputs controlled floats, any upstream validation failure or adversarial input modifying these values in a dashboard state could execute arbitrary HTML/JavaScript (CWE-79).
**Prevention:** Always wrap dynamically interpolated string variables with `html.escape()` before injecting them into HTML templates rendered by `st.markdown(..., unsafe_allow_html=True)`, even if the intended type is numeric.

## 2024-05-20 - Prevented Cross-Session State Corruption in Streamlit Cache (CWE-374 / CWE-662)
**Vulnerability:** The `ImmutableRuntimeProxy` for cached objects in `dashboards/streamlit_app.py` previously only protected dictionaries using `types.MappingProxyType`. This allowed lists, sets, and Pandas DataFrames to be modified in-place within the global Streamlit cache by any session.
**Learning:** Shared mutable state in Streamlit's `@st.cache_resource` leads to severe cross-session contamination (CWE-374) and data leakage between concurrent users, as one user's action could mutate the globally cached Pandas DataFrame or list.
**Prevention:** Extend defensive proxies to intercept multiple types of mutable sequences/collections. Always return immutable equivalents (like `tuple` or `frozenset`) or a deep copy (`df.copy(deep=True)`) when exposing cached DataFrames or lists to untrusted contexts.
## 2024-10-26 - Deep Immutability for Streamlit Cache
**Vulnerability:** The `ImmutableRuntimeProxy` for `@st.cache_resource` only provided shallow immutability via `MappingProxyType`. Nested mutable objects (like nested dictionaries or lists within configs) were still fully susceptible to in-place mutation by any user, leading to cross-session state corruption (CWE-374).
**Learning:** Python's `types.MappingProxyType` and standard proxy wrappers do not recurse. Relying on them for complex nested objects in shared application memory leaves the inner layers vulnerable.
**Prevention:** When protecting shared mutable state, implement deep recursive transformations (e.g., converting nested dicts to proxies, lists to tuples) and ensure NumPy arrays return read-only views via `.view()` and `.flags.writeable = False`.
