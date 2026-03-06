# =============================================================================
#  Dockerfile — Arbitrage Bots
# =============================================================================
#
#  Build:   docker build -t arb-bots:latest .
#  Run:     docker run --rm -it arb-bots:latest simulate
#           docker run --rm -it arb-bots:latest backtest
#           docker run --rm -it arb-bots:latest menu
#           docker run --rm -it -v $(pwd)/config:/app/python/config arb-bots:latest bot-cross
#
#  Or with docker-compose:
#           docker-compose up
# =============================================================================

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy dependency manifest first (layer-cache friendly) ────────────────────
COPY python/requirements.txt python/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r python/requirements.txt

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Create config directory (user can volume-mount their own config.yaml) ─────
RUN mkdir -p python/config \
 && if [ ! -f python/config/config.yaml ]; then \
        cp python/config/config.example.yaml python/config/config.yaml; \
    fi

# ── Health check: verify the launcher script is importable ───────────────────
RUN python -c "import ast; ast.parse(open('arb.py').read()); print('arb.py OK')"

# ── Default environment ───────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/python

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Pass a command as the first argument:
#   simulate  → paper-trading simulator (default, no keys needed)
#   backtest  → backtester
#   menu      → interactive menu (requires a TTY: docker run -it ...)
#   bot-cross / bot-tri / bot-ai → live bots (need config.yaml with keys)

ENTRYPOINT ["python", "arb.py"]
CMD ["simulate"]
