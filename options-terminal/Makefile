# Options terminal, local stack in Docker (docker-compose.yml).
#   make start          build and start everything, wait until healthy
#   make stop           stop and remove this project's containers, network and images (data kept)
#   make clean          stop + delete the local database and snapshot volumes
#   make probe          check the venue facts vrp_signal assumes (read-only calls, report file)
# Database: Supabase when .env sets SUPABASE_DB_URL, else a local Postgres container.
# Only resources labelled with this compose project are touched; nothing else on the machine.

SHELL := /bin/bash
PROJECT := options-terminal
GIT_SHA := $(shell git rev-parse HEAD 2>/dev/null || echo 0000000)

# .env is read by docker compose, not by make (a pasted URL may contain $ or #); make only
# checks whether SUPABASE_DB_URL is set to pick the mode.
ifeq ($(shell grep -qE '^SUPABASE_DB_URL=.+' .env 2>/dev/null && echo yes),yes)
  # Supabase > Connect > Session pooler URI (port 5432): the API LISTENs and the runner holds
  # advisory locks, both need session connections (never the 6543 transaction pooler).
  DB_MODE := supabase
  COMPOSE := docker compose
else
  DB_MODE := local
  COMPOSE := docker compose --profile local
endif
COMPOSE_ALL := docker compose --profile local  # stop/clean cover both modes

.DEFAULT_GOAL := help
.PHONY: help start stop clean restart logs ps probe check-docker env check-url

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-14s %s\n", $$1, $$2}'
	@echo "  database: $(DB_MODE)"

check-docker:
	@docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop (or the daemon) and retry."; exit 1; }

env: ## create .env with random database passwords (first run only)
	@if [ ! -f .env ]; then \
		umask 077; \
		{ echo "POSTGRES_PASSWORD=$$(openssl rand -hex 24)"; \
		  echo "FRONTEND_PORT=3000"; } > .env; \
		echo "created .env (gitignored). For Supabase add one line: SUPABASE_DB_URL=<Session pooler URI>"; \
	fi

start: check-docker env check-url ## build and start api, runner and frontend (+ local db unless Supabase)
	@echo "database: $(DB_MODE)"
	TERMINAL_GIT_SHA=$(GIT_SHA) $(COMPOSE) up -d --build --wait --remove-orphans
	@echo ""
	@echo "Options terminal: http://localhost:$$(sed -n 's/^FRONTEND_PORT=//p' .env | grep . || echo 3000)"
	@echo "Runs appear within a few minutes once Deribit answers. Follow with: make logs"

check-url:
	@if grep -qE '^SUPABASE_DB_URL=.*\[YOUR-PASSWORD\]' .env 2>/dev/null; then \
		echo "SUPABASE_DB_URL still contains [YOUR-PASSWORD]: put your database password in it"; exit 1; fi
	@if grep -qE '^SUPABASE_DB_URL=.*:6543/' .env 2>/dev/null; then \
		echo "SUPABASE_DB_URL is the transaction pooler (6543): copy the Session pooler URI (5432)"; exit 1; fi

stop: check-docker env ## stop everything and remove this project's containers, network and images (keeps data)
	$(COMPOSE_ALL) down --remove-orphans --rmi local
	docker image prune -f --filter label=com.options-terminal.project=$(PROJECT)

clean: stop ## stop + delete the local database and snapshot volumes (never touches Supabase)
	$(COMPOSE_ALL) down --volumes --remove-orphans

restart: stop start ## rebuild and restart

logs: env ## follow all service logs
	$(COMPOSE) logs -f --tail=100

ps: env ## show service status
	$(COMPOSE) ps

# Read-only Deribit/Hyperliquid calls from the runner image (tools/probe_venues.py); runs as the
# host user so the report lands in the bind-mounted folder. No database or snapshot dir needed.
PROBE_DIR := models/terminal_models/library/rv_har/validation
probe: check-docker env ## check Deribit/Hyperliquid candle caps, history, DVOL hours; writes venue_probe.json
	@mkdir -p $(PROBE_DIR)
	$(COMPOSE) run --rm --no-deps --build --user "$$(id -u):$$(id -g)" -v "$(CURDIR)/$(PROBE_DIR):/out" \
		runner python -m terminal_models.tools.probe_venues --out /out/venue_probe.json
