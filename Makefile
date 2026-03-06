# =============================================================================
#  Makefile — Arbitrage Bots convenience commands
#  Usage: make <target>
#
#  Requires: make (Linux/macOS built-in; Windows: winget install GnuWin32.Make)
# =============================================================================

# ── Auto-detect Python in venv or system ─────────────────────────────────────
VENV_PYTHON := python/.venv/bin/python
SYS_PYTHON  := python3
PYTHON      := $(shell [ -f "$(VENV_PYTHON)" ] && echo "$(VENV_PYTHON)" || echo "$(SYS_PYTHON)")
PYTHON_DIR  := python

.DEFAULT_GOAL := help

# ── Help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  Arbitrage Bots — available commands"
	@echo ""
	@echo "  Setup"
	@echo "    make setup          One-command setup (venv + deps + config copy)"
	@echo ""
	@echo "  Run"
	@echo "    make menu           Open the interactive launcher menu"
	@echo "    make simulate       Run paper-trading simulator (no API keys needed)"
	@echo "    make backtest       Run strategy backtester (no API keys needed)"
	@echo "    make wizard         Run the interactive config wizard"
	@echo ""
	@echo "  Bots (require config.yaml with API keys)"
	@echo "    make bot-cross      Run cross-exchange arbitrage bot"
	@echo "    make bot-tri        Run triangular arbitrage bot"
	@echo "    make bot-ai         Run AI arbitrage bot"
	@echo ""
	@echo "  Quality"
	@echo "    make test           Run the test suite"
	@echo "    make lint           Lint Python code (flake8)"
	@echo "    make fmt            Auto-format Python code (black)"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-build   Build the Docker image"
	@echo "    make docker-sim     Run the simulator inside Docker"
	@echo "    make docker-up      Start all services with docker-compose"
	@echo "    make docker-down    Stop docker-compose services"
	@echo ""
	@echo "  Misc"
	@echo "    make clean          Remove build/cache artefacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	@./setup.sh

# ── Run ──────────────────────────────────────────────────────────────────────
.PHONY: menu
menu:
	$(PYTHON) arb.py

.PHONY: simulate
simulate:
	cd $(PYTHON_DIR) && $(PYTHON) -m simulator.run

.PHONY: backtest
backtest:
	cd $(PYTHON_DIR) && $(PYTHON) -m backtest.backtest

.PHONY: wizard
wizard:
	$(PYTHON) arb.py wizard

# ── Bots ─────────────────────────────────────────────────────────────────────
.PHONY: bot-cross
bot-cross:
	cd $(PYTHON_DIR) && $(PYTHON) -m cross_exchange_arb.bot --config config/config.yaml

.PHONY: bot-tri
bot-tri:
	cd $(PYTHON_DIR) && $(PYTHON) -m triangular_arb.bot --config config/config.yaml

.PHONY: bot-ai
bot-ai:
	cd $(PYTHON_DIR) && $(PYTHON) -m ai_arb.bot --config config/config.yaml --strategy both

# ── Quality ──────────────────────────────────────────────────────────────────
.PHONY: test
test:
	cd $(PYTHON_DIR) && $(PYTHON) -m pytest tests/ -v --tb=short

.PHONY: lint
lint:
	cd $(PYTHON_DIR) && $(PYTHON) -m flake8 . \
		--max-line-length=120 \
		--exclude=.venv,__pycache__,*.egg-info,models \
		--ignore=E501,W503

.PHONY: fmt
fmt:
	cd $(PYTHON_DIR) && $(PYTHON) -m black . --line-length=120 \
		--exclude='/(\.venv|models|__pycache__)/'

# ── Docker ───────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:
	docker build -t arb-bots:latest .

.PHONY: docker-sim
docker-sim:
	docker run --rm -it arb-bots:latest simulate

.PHONY: docker-up
docker-up:
	docker-compose up --build

.PHONY: docker-down
docker-down:
	docker-compose down

# ── Misc ─────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name "*.pyo" -delete 2>/dev/null; true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	rm -f simulation_report.html *_sim.html *_report.html
	@echo "  Cleaned."
