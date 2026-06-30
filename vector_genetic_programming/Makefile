PYTHON := .venv/bin/python

.PHONY: start test smoke lint approve

start:
	$(PYTHON) scripts/run.py

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

smoke:
	$(PYTHON) -m pytest tests/test_smoke.py -v --tb=short

lint:
	$(PYTHON) -m ruff check vgp/ tests/ scripts/

approve:
	$(PYTHON) scripts/approve.py
