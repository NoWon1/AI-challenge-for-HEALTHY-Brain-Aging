.PHONY: install test lint typecheck build audit-demo

install:
	python -m pip install -e ".[advanced,dev,imaging]"

test:
	python -m pytest

lint:
	python -m ruff check neurosaarthi tests models/survival/rsf.py models/classification/lightgbm_risk.py evaluation/uncertainty.py
	python -m ruff format --check neurosaarthi tests models/survival/rsf.py models/classification/lightgbm_risk.py evaluation/uncertainty.py

typecheck:
	python -m mypy neurosaarthi/core neurosaarthi/security neurosaarthi/data/splitters

build:
	python -m build

audit-demo:
	python -m neurosaarthi.security.audit_dataset --root reports/export_candidate --mode export
