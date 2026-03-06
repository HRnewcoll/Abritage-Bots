#!/usr/bin/env bash
# =============================================================================
#  setup.sh — One-command setup for Arbitrage Bots (Linux / macOS)
# =============================================================================
#
#  Usage:
#    chmod +x setup.sh && ./setup.sh
#
#  What it does:
#    1. Checks Python 3.9+
#    2. Creates a virtual environment in python/.venv
#    3. Installs all Python dependencies
#    4. Copies config template → python/config/config.yaml (if not present)
#    5. Prints next steps
# =============================================================================
set -e

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}  ▶  $*${RESET}"; }
success() { echo -e "${GREEN}  ✔  $*${RESET}"; }
warn()    { echo -e "${YELLOW}  ⚠  $*${RESET}"; }
error()   { echo -e "${RED}  ✘  $*${RESET}"; exit 1; }

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║   Arbitrage Bots — Quick Setup               ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════╝${RESET}"
echo ""

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_DIR="$REPO_ROOT/python"

# ── 1. Check Python ───────────────────────────────────────────────────────────
info "Checking Python version…"
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    error "Python not found. Install Python 3.9+ from https://python.org"
fi

PY_VER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    error "Python 3.9+ required (found $PY_VER). Please upgrade: https://python.org"
fi
success "Python $PY_VER found"

# ── 2. Create virtual environment ─────────────────────────────────────────────
VENV_DIR="$PYTHON_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists — skipping creation"
else
    info "Creating virtual environment at python/.venv …"
    $PYTHON -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

# ── 3. Activate and install dependencies ──────────────────────────────────────
info "Installing Python dependencies…"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$PYTHON_DIR/requirements.txt" --quiet
success "Dependencies installed"

# ── 4. Copy config template ───────────────────────────────────────────────────
CONFIG_FILE="$PYTHON_DIR/config/config.yaml"
CONFIG_EXAMPLE="$PYTHON_DIR/config/config.example.yaml"

if [ -f "$CONFIG_FILE" ]; then
    warn "python/config/config.yaml already exists — not overwritten"
else
    cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
    success "Created python/config/config.yaml from example template"
fi

# ── 5. Copy .env template ─────────────────────────────────────────────────────
ENV_FILE="$REPO_ROOT/.env"
ENV_EXAMPLE="$REPO_ROOT/.env.example"
if [ -f "$ENV_FILE" ]; then
    warn ".env already exists — not overwritten"
elif [ -f "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    success "Created .env from .env.example"
fi

# ── 6. Done — print next steps ────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}  ✔  Setup complete!${RESET}"
echo ""
echo -e "${BOLD}  Next steps:${RESET}"
echo ""
echo -e "  ${CYAN}Option A — Try the simulator right now (no API keys needed):${RESET}"
echo ""
echo -e "    source python/.venv/bin/activate"
echo -e "    python arb.py"
echo ""
echo -e "  ${CYAN}Option B — Configure API keys and run a live/sandbox bot:${RESET}"
echo ""
echo -e "    source python/.venv/bin/activate"
echo -e "    python arb.py           # choose 'Setup Wizard' from the menu"
echo ""
echo -e "  ${YELLOW}  Tip: All bots start in dry_run mode — no real orders are placed.${RESET}"
echo ""
