### Sentinel Security Learnings
* **Exception Handling:** Replaced `except Exception:` with specific `except (ImportError, ValueError):` in `demo/runtime.py`. Broad `except Exception:` blocks can mask critical errors including syntax bugs, logic bugs, or memory issues which can make auditing and troubleshooting harder. We specifically target the exceptions related to missing modules or data structure values.

## 2026-10-25 - [Routine Security Audit Sign-off]
**Vulnerability:** No new vulnerabilities found during routine security audit.
**Learning:** Evaluated the application across multiple potential vulnerability domains including deserialization (joblib/pickle), archive extraction (zipfile/tarfile), configuration parsing (yaml.safe_load), and UI rendering. The codebase proved robust in these areas. Static file paths loaded in ETL adapters do not require defense-in-depth path traversal checks because they are immune to external traversal attacks by design, and enforcing resolution checks actively breaks legitimate dataset symlinking workflows.
**Prevention:** Maintained strict hygiene rules by explicitly adding dataset (`*.csv`, `*.tsv`, `*.nii`, `*.nii.gz`) and environment variable (`.env`, `.env.local`) exclusions to `.gitignore` to prevent accidental PII/secret leaks. The Sentinel audit confirmed compliance with existing security baselines.
