# Options terminal, local stack in Docker (docker-compose.yml).
#   make start          build and start everything, wait until healthy
#   make stop           stop and remove this project's containers, network and images (data kept)
#   make clean          stop + delete the local database and snapshot volumes
#   make supabase-init  one-time: schema, logins and lockdown in the Supabase project
# Database: Supabase when .env sets SUPABASE_PROJECT_REF, else a local Postgres container.
# Only resources labelled with this compose project are touched; nothing else on the machine.

SHELL := /bin/bash
PROJECT := options-terminal
GIT_SHA := $(shell git rev-parse HEAD 2>/dev/null || echo 0000000)

-include .env

ifdef SUPABASE_PROJECT_REF
  # Session pooler (port 5432): the API LISTENs and the runner holds advisory locks, both of
  # which need session connections (never the 6543 transaction pooler).
  DB_MODE := supabase
  export SUPABASE_PROJECT_REF SUPABASE_POOLER_HOST SUPABASE_DB_PASSWORD RUNNER_DB_PASSWORD API_DB_PASSWORD
  SUPABASE_URL = $(1).$(SUPABASE_PROJECT_REF):$(2)@$(SUPABASE_POOLER_HOST):5432/postgres?sslmode=require
  export RUNNER_DATABASE_URL := postgresql://$(call SUPABASE_URL,runner,$(RUNNER_DB_PASSWORD))
  export API_DATABASE_URL := postgres://$(call SUPABASE_URL,api,$(API_DB_PASSWORD))
  COMPOSE := docker compose
else
  DB_MODE := local
  COMPOSE := docker compose --profile local
endif
COMPOSE_ALL := docker compose --profile local  # stop/clean cover both modes

.DEFAULT_GOAL := help
.PHONY: help start stop clean restart logs ps check-docker env supabase-init

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-14s %s\n", $$1, $$2}'
	@echo "  database: $(DB_MODE)"

check-docker:
	@docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop (or the daemon) and retry."; exit 1; }

env: ## create .env with random database passwords (first run only)
	@if [ ! -f .env ]; then \
		umask 077; \
		{ echo "POSTGRES_PASSWORD=$$(openssl rand -hex 24)"; \
		  echo "RUNNER_DB_PASSWORD=$$(openssl rand -hex 24)"; \
		  echo "API_DB_PASSWORD=$$(openssl rand -hex 24)"; \
		  echo "FRONTEND_PORT=3000"; } > .env; \
		echo "created .env (random passwords; gitignored). For Supabase, add the SUPABASE_* lines from .env.example."; \
	fi

start: check-docker env ## build and start api, runner and frontend (+ local db unless Supabase)
	@echo "database: $(DB_MODE)"
	TERMINAL_GIT_SHA=$(GIT_SHA) $(COMPOSE) up -d --build --wait --remove-orphans
	@echo ""
	@echo "Options terminal: http://localhost:$$(sed -n 's/^FRONTEND_PORT=//p' .env)"
	@echo "Runs appear within a few minutes once Deribit answers. Follow with: make logs"

supabase-init: check-docker env ## one-time Supabase setup: migrations, runner/api logins, anon lockdown + RLS
	@test -n "$(SUPABASE_PROJECT_REF)" -a -n "$(SUPABASE_POOLER_HOST)" -a -n "$(SUPABASE_DB_PASSWORD)" || \
		{ echo "set SUPABASE_PROJECT_REF, SUPABASE_POOLER_HOST and SUPABASE_DB_PASSWORD in .env (see .env.example)"; exit 1; }
	docker run --rm --label com.options-terminal.project=$(PROJECT) \
		-e SUPABASE_PROJECT_REF -e SUPABASE_POOLER_HOST -e SUPABASE_DB_PASSWORD \
		-e RUNNER_DB_PASSWORD -e API_DB_PASSWORD \
		-v "$(CURDIR)/contracts/db/migrations:/migrations:ro" -v "$(CURDIR)/docker/db:/setup:ro" \
		postgres:17 sh /setup/supabase_init.sh

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
