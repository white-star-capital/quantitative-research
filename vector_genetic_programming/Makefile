PYTHON := .venv/bin/python

.PHONY: test smoke lint approve

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

smoke:
	$(PYTHON) -m pytest tests/test_smoke.py -v --tb=short

lint:
	$(PYTHON) -m ruff check vgp/ tests/ scripts/

approve:
	$(PYTHON) scripts/approve.py
