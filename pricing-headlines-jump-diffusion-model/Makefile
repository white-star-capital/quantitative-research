# Pricing agent + dashboard control. Two commands:
#   make start  - create the venv on first run (installs deps), then launch BOTH the
#                 hourly pricing agent and the Streamlit dashboard in the background.
#   make stop   - gracefully shut down both the agent and the dashboard.

VENV      := .venv
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit

AGENT_PID := agent.pid
AGENT_LOG := agent.log
DASH_PID  := dashboard.pid
DASH_LOG  := dashboard.log
DASH_PORT := 8501

.PHONY: start stop

start:
	@if [ ! -x "$(PY)" ]; then \
		echo "📦 Creating virtualenv in $(VENV) ..."; \
		python3 -m venv $(VENV); \
		$(PY) -m pip install --upgrade pip >/dev/null; \
	fi
	@echo "📦 Ensuring dependencies are installed ..."
	@$(PIP) install -q -r requirements.txt
	@if [ -f "$(AGENT_PID)" ] && kill -0 "`cat $(AGENT_PID)`" 2>/dev/null; then \
		echo "⚠️  Agent already running (PID `cat $(AGENT_PID)`)."; \
	else \
		nohup $(PY) start_agent.py >> $(AGENT_LOG) 2>&1 & echo $$! > $(AGENT_PID); \
		echo "✅ Started agent (PID `cat $(AGENT_PID)`). Logs: $(AGENT_LOG)"; \
	fi
	@if [ -f "$(DASH_PID)" ] && kill -0 "`cat $(DASH_PID)`" 2>/dev/null; then \
		echo "⚠️  Dashboard already running (PID `cat $(DASH_PID)`)."; \
	else \
		nohup $(STREAMLIT) run dashboard.py --server.headless true --server.port $(DASH_PORT) >> $(DASH_LOG) 2>&1 & echo $$! > $(DASH_PID); \
		echo "✅ Started dashboard (PID `cat $(DASH_PID)`). URL: http://localhost:$(DASH_PORT)  Logs: $(DASH_LOG)"; \
	fi

stop:
	@if [ -f "$(AGENT_PID)" ] && kill -0 "`cat $(AGENT_PID)`" 2>/dev/null; then \
		PID=`cat $(AGENT_PID)`; \
		echo "🛑 Stopping agent (PID $$PID) ..."; \
		kill -TERM $$PID; \
		for i in `seq 1 40`; do kill -0 $$PID 2>/dev/null || break; sleep 1; done; \
		if kill -0 $$PID 2>/dev/null; then echo "   forcing ..."; kill -KILL $$PID; fi; \
		rm -f $(AGENT_PID); \
		echo "✅ Agent stopped."; \
	else \
		echo "ℹ️  Agent not running."; rm -f $(AGENT_PID); \
	fi
	@if [ -f "$(DASH_PID)" ] && kill -0 "`cat $(DASH_PID)`" 2>/dev/null; then \
		PID=`cat $(DASH_PID)`; \
		echo "🛑 Stopping dashboard (PID $$PID) ..."; \
		kill -TERM $$PID; \
		for i in `seq 1 15`; do kill -0 $$PID 2>/dev/null || break; sleep 1; done; \
		if kill -0 $$PID 2>/dev/null; then echo "   forcing ..."; kill -KILL $$PID; fi; \
		rm -f $(DASH_PID); \
		echo "✅ Dashboard stopped."; \
	else \
		echo "ℹ️  Dashboard not running."; rm -f $(DASH_PID); \
	fi
	@pkill -f "streamlit run dashboard.py" 2>/dev/null || true
