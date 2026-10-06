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

## 2026-10-28 - [TOCTOU File Permission Race Condition]
**Vulnerability:** A Time-of-Check to Time-of-Use (TOCTOU) file permission race condition existed in `export_models.py` where a directory was created with default permissions before `os.chmod()` was applied.
**Learning:** In the split-second between `mkdir(exist_ok=True)` and `os.chmod(OUT_DIR, 0o700)`, the directory is accessible with potentially overly permissive default permissions, creating a window for data exposure or tampering if sensitive artifacts are written concurrently or if the script crashes before `chmod`.
**Prevention:** Always enforce access restrictions exactly at the time of creation by setting the permission mode natively during the filesystem operation (e.g., using `Path.mkdir(mode=0o700, exist_ok=True)` or passing restrictive flags and modes to `os.open()`).

## 2026-10-31 - [Preventing CSV Spreadsheet Formula Injection]
**Vulnerability:** Calling `st.dataframe` allows users to export the grid as a CSV file. Without sanitization, any string column containing formula-trigger characters (`=`, `+`, `-`, `@`, `\t`, `\r`) could be automatically evaluated as dynamic expressions or DDE commands when opened in a spreadsheet program, leading to CSV Formula Injection (CWE-1236).
**Learning:** Even internal prototype UI tools must prevent untrusted or dynamic text data (such as participant IDs, labels, or cohorts) from inadvertently turning into executable formulas during standard export operations.
**Prevention:** Intercept the DataFrame just before rendering in the UI with a targeted sanitization utility (e.g. `_sanitize_dataframe`) that iterates over object/string columns and defensibly prepends a single quote (`'`) to values starting with formula trigger prefixes.
