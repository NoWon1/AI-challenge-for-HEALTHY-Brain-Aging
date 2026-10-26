## 2023-10-25 - Prevented XSS in Streamlit Markdown Cards
**Vulnerability:** The `_risk_card` component in `dashboards/streamlit_app.py` rendered dynamic patient risk factors (like `horizon`, `risk`, `lower`, `upper`) into HTML structures using Streamlit's `unsafe_allow_html=True` without prior escaping.
**Learning:** `unsafe_allow_html=True` acts as a direct injection vector. While the synthetic runtime typically outputs controlled floats, any upstream validation failure or adversarial input modifying these values in a dashboard state could execute arbitrary HTML/JavaScript (CWE-79).
**Prevention:** Always wrap dynamically interpolated string variables with `html.escape()` before injecting them into HTML templates rendered by `st.markdown(..., unsafe_allow_html=True)`, even if the intended type is numeric.

## 2024-02-23 - Prevent Cross-Session State Corruption in Streamlit Cache
**Vulnerability:** The `ImmutableRuntimeProxy` in `dashboards/streamlit_app.py` only protected dictionary attributes, leaving pandas DataFrames, lists, and sets vulnerable to modification by one user, affecting all users sharing the cached object (CWE-374 / CWE-662).
**Learning:** Streamlit's `@st.cache_resource` is a global cache across all sessions. If any mutable object returned by this cache is modified, it causes cross-session data leakage and state corruption.
**Prevention:** Always comprehensively wrap cached singletons in an immutable facade. For custom proxies, intercept `__getattr__` and recursively cast all mutable attributes: `list` to `tuple`, `set` to `frozenset`, `dict` to `types.MappingProxyType`, and return `.copy(deep=True)` for pandas DataFrames.
