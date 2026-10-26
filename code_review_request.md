# Request for Code Review

I have added tests for the `time_dependent_auc` function in `tests/test_survival_metrics.py`.

The changes:
- Created 4 test cases for perfect predictor, random predictor, output shape/bounds validation, and a fallback branch when `HAS_SKSURV` is unavailable.
- Ran tests locally which pass successfully.

Please review the changes.
