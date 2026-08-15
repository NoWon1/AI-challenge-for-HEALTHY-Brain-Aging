# Contributing

NeuroSaarthi-AD welcomes research-software contributions that preserve participant privacy, cohort-native semantics, and external-validation discipline.

1. Work only with public, authorised local, or synthetic data. Never commit participant images, rows, identifiers, dates, derivatives, predictions, restricted logs, credentials, or protected checkpoints.
2. Create a focused branch and add tests for scientific invariants, edge cases, fit-state behaviour, missingness, censoring, and leakage where applicable.
3. Run `make lint` and `make test`. Imaging changes must also run geometry/label round-trip tests. Model changes must prove preprocessing is fit on training data only.
4. Update model/data cards and documentation. Do not report synthetic metrics as real performance or make diagnostic/clinical-grade claims.
5. Describe the endpoint, prediction origin, split, calibration set, external set, and expected failure mode in the pull request.

Controlled-data results may enter a review only after the applicable steward/export process approves aggregate disclosure.
