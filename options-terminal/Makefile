# Options terminal, local stack in Docker (docker-compose.yml).
#   make start   build and start everything, wait until healthy
#   make stop    stop and remove this project's containers, network and images (data kept)
#   make clean   stop + delete the database and snapshot volumes
# Only resources labelled with this compose project are touched; nothing else on the machine.

SHELL := /bin/bash
COMPOSE := docker compose
PROJECT := options-terminal
GIT_SHA := $(shell git rev-parse HEAD 2>/dev/null || echo 0000000)

.DEFAULT_GOAL := help
.PHONY: help start stop clean restart logs ps check-docker env

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-8s %s\n", $$1, $$2}'

check-docker:
	@docker info >/dev/null 2>&1 || { echo "Docker is not running. Start Docker Desktop (or the daemon) and retry."; exit 1; }

env: ## create .env with random database passwords (first run only)
	@if [ ! -f .env ]; then \
		umask 077; \
		{ echo "POSTGRES_PASSWORD=$$(openssl rand -hex 24)"; \
		  echo "RUNNER_DB_PASSWORD=$$(openssl rand -hex 24)"; \
		  echo "API_DB_PASSWORD=$$(openssl rand -hex 24)"; \
		  echo "FRONTEND_PORT=3000"; } > .env; \
		echo "created .env (random passwords; gitignored)"; \
	fi

start: check-docker env ## build and start db, api, runner and frontend
	TERMINAL_GIT_SHA=$(GIT_SHA) $(COMPOSE) up -d --build --wait --remove-orphans
	@echo ""
	@echo "Options terminal: http://localhost:$$(sed -n 's/^FRONTEND_PORT=//p' .env)"
	@echo "Runs appear within a minute or two once Deribit answers. Follow with: make logs"

stop: check-docker env ## stop everything and remove this project's containers, network and images (keeps data)
	$(COMPOSE) down --remove-orphans --rmi local
	docker image prune -f --filter label=com.options-terminal.project=$(PROJECT)

clean: stop ## stop + delete the database and snapshot volumes
	$(COMPOSE) down --volumes --remove-orphans

restart: stop start ## rebuild and restart

logs: env ## follow all service logs
	$(COMPOSE) logs -f --tail=100

ps: env ## show service status
	$(COMPOSE) ps
