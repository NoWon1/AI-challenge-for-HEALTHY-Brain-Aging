### Sentinel Security Learnings
* **Exception Handling:** Replaced `except Exception:` with specific `except (ImportError, ValueError):` in `demo/runtime.py`. Broad `except Exception:` blocks can mask critical errors including syntax bugs, logic bugs, or memory issues which can make auditing and troubleshooting harder. We specifically target the exceptions related to missing modules or data structure values.

## 2026-10-25 - [Routine Security Audit Sign-off]
**Vulnerability:** No new vulnerabilities found during routine security audit.
**Learning:** Evaluated the application across multiple potential vulnerability domains including deserialization (joblib/pickle), archive extraction (zipfile/tarfile), configuration parsing (yaml.safe_load), and UI rendering. The codebase proved robust in these areas. Static file paths loaded in ETL adapters do not require defense-in-depth path traversal checks because they are immune to external traversal attacks by design, and enforcing resolution checks actively breaks legitimate dataset symlinking workflows.
**Prevention:** Maintained strict hygiene rules by explicitly adding dataset (`*.csv`, `*.tsv`, `*.nii`, `*.nii.gz`) and environment variable (`.env`, `.env.local`) exclusions to `.gitignore` to prevent accidental PII/secret leaks. The Sentinel audit confirmed compliance with existing security baselines.

## 2026-10-26 - [Secure Exception Handling]
**Vulnerability:** Exception messages in `harmonization/leakage.py` were leaking raw participant IDs when raising a `ValueError`.
**Learning:** Found a specific pattern where primary identifiers (`participant_id`) were embedded in exception messages, increasing the risk of accidental data leakage into logs or dashboards.
**Prevention:** Ensured exception messages "fail securely" by providing actionable debugging context (like overlap counts) instead of raw PII.

## 2026-10-27 - [Preventing Markdown Injection in Streamlit Warnings]
**Vulnerability:** Unescaped dynamic strings in `st.warning` allowed Markdown injection (hyperlinks, image tracking pixels).
**Learning:** Streamlit's `st.warning` parses Markdown by default. While it doesn't execute `<script>` tags, it is vulnerable to Markdown injection through unescaped dynamic strings.
**Prevention:** Sanitize text dynamically inserted into warnings at the source boundary using regex to escape CommonMark structural tokens (`\`, `` ` ``, `*`, `_`, `{`, `}`, `[`, `]`, `(`, `)`, `#`, `+`, `-`, `.`, `!`, `~`, `|`, `<`, `>`). Enclose dynamic identifiers in monospace code spans to neutralize hyperlink and tracking-pixel injection.

## 2026-10-28 - [Preventing Cross-Session State Corruption (CWE-374 / CWE-662)]
**Vulnerability:** Cached runtime (`DemoRuntime`) created by `@st.cache_resource` was mutable, which could allow one concurrent user to corrupt the singleton state (e.g., sorting, assignments, filtering) for all active sessions.
**Learning:** `copy.deepcopy()` is too resource-intensive and might fail on un-serializable objects (C-extensions, file descriptors) in `@st.cache_resource`. Streamlit's `@st.cache_resource` shares objects in memory, requiring defensive wrappers.
**Prevention:** Used an immutable facade (`ImmutableRuntimeProxy`) implementing `__setattr__`, `__delattr__`, and `MappingProxyType` to intercept mutation attempts at O(1) overhead. All in-place modifications to objects fetched from the proxy must first `.copy()` them.
