PYTHON := .venv/bin/python
IMAGE  := vgp-dashboard
PORT   ?= 8000

.PHONY: help start stop serve vgp dashboard check-dashboard test smoke lint verify-data approve

# `make` on its own lists the targets rather than silently running the first one.
help:
	@echo "VGP targets"
	@echo "  make start         build the dashboard image and serve the committed results"
	@echo "                     (Docker required; takes seconds — no vectorbt, no compiler)"
	@echo "  make stop          stop the dashboard container"
	@echo "  make serve         serve the dashboard locally WITH run launching enabled"
	@echo "                     (needs .venv; the New Run panel can start experiments)"
	@echo "  make vgp           run the full experiment — 3 windows x 3 seeds + 99 null runs"
	@echo "                     (needs .venv; TAKES SEVERAL HOURS)"
	@echo "  make dashboard     rebuild dashboard/index.html from results/ without Docker"
	@echo "  make verify-data   check data/ against data/MANIFEST.json"
	@echo "  make test          full test suite"
	@echo "  make smoke         install/import smoke test only"
	@echo "  make lint          ruff"
	@echo ""
	@echo "  Override the port with: make start PORT=9000"

# --- Seeing the results -----------------------------------------------------
# Builds the image and serves it in the foreground, so Ctrl-C stops it and
# nothing is left running. The image build regenerates the page from results/,
# so what you see is the committed run, not a snapshot someone pasted in.
start:
	@command -v docker >/dev/null 2>&1 || { \
	  echo "ERROR: docker is not installed."; \
	  echo "Without it, 'make dashboard' writes the same page to dashboard/index.html"; \
	  echo "and you can open that file directly."; exit 1; }
	@docker info >/dev/null 2>&1 || { \
	  echo "ERROR: the Docker daemon is not running. Start Docker and retry."; exit 1; }
	docker build -t $(IMAGE) .
	@echo ""
	@echo "  Dashboard -> http://localhost:$(PORT)"
	@echo "  Ctrl-C to stop."
	@echo ""
	docker run --rm --name $(IMAGE) -p $(PORT):8000 $(IMAGE)

stop:
	-docker rm -f $(IMAGE) 2>/dev/null || true

# Same page without Docker. Useful after a run, and what CI checks for staleness.
dashboard:
	python3 scripts/build_dashboard.py

check-dashboard:
	python3 scripts/build_dashboard.py --check

# Serves from the project venv with launching enabled, so the New Run panel can
# actually start an experiment. The Docker image deliberately cannot: it has no
# vectorbt, and a container that can spend eight hours of CPU should be an
# explicit choice rather than the default.
serve: dashboard
	VGP_ALLOW_RUN=1 $(PYTHON) scripts/serve_dashboard.py

# --- Producing the results --------------------------------------------------
# The experiment itself. Was `make start` until the dashboard took that name.
vgp:
	$(PYTHON) scripts/run.py

# --- Checks -----------------------------------------------------------------
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

smoke:
	$(PYTHON) -m pytest tests/test_smoke.py -v --tb=short

lint:
	$(PYTHON) -m ruff check vgp/ tests/ scripts/

# Checks data/ against data/MANIFEST.json. Run this first when a re-run does not
# reproduce results/ — it separates "wrong data" from "wrong code" in seconds.
# Uses system python so it works before the venv exists.
verify-data:
	python3 scripts/verify_data.py

approve:
	$(PYTHON) scripts/approve.py
