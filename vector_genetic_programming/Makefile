PYTHON := .venv/bin/python

.PHONY: start test smoke lint approve verify-data

start:
	$(PYTHON) scripts/run.py

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Checks data/ against data/MANIFEST.json. Run this first when a re-run does not
# reproduce results/ — it separates "wrong data" from "wrong code" in seconds.
# Uses system python so it works before the venv exists.
verify-data:
	python3 scripts/verify_data.py

smoke:
	$(PYTHON) -m pytest tests/test_smoke.py -v --tb=short

lint:
	$(PYTHON) -m ruff check vgp/ tests/ scripts/

approve:
	$(PYTHON) scripts/approve.py
