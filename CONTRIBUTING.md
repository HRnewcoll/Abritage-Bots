# Contributing to Arbitrage Bots

Thank you for your interest in contributing! 🎉  
This document explains how to set up your development environment and the conventions used in this project.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Running Tests](#running-tests)
5. [Code Style](#code-style)
6. [Adding a New Strategy](#adding-a-new-strategy)
7. [Adding a New Exchange](#adding-a-new-exchange)
8. [Pull Request Guidelines](#pull-request-guidelines)
9. [Security](#security)

---

## Getting Started

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/Arbitrage-Bots.git
cd Arbitrage-Bots

# 2. One-command setup
./setup.sh          # Linux/macOS
setup.bat           # Windows

# 3. Activate the virtual environment
source python/.venv/bin/activate   # Linux/macOS
python\.venv\Scripts\activate      # Windows

# 4. Verify everything works (no API keys needed)
make simulate
```

---

## Project Structure

```
Arbitrage-Bots/
├── arb.py                      Unified interactive launcher
├── setup.sh / setup.bat        One-command environment setup
├── Makefile                    Shorthand for common tasks
├── Dockerfile / docker-compose.yml  Containerised deployment
├── python/
│   ├── requirements.txt        Runtime dependencies
│   ├── setup_wizard.py         Interactive config wizard
│   ├── config/
│   │   └── config.example.yaml Config template (copy → config.yaml)
│   ├── utils/exchange.py       Shared ccxt helpers (used by all bots)
│   ├── cross_exchange_arb/     Cross-exchange arbitrage strategy
│   ├── triangular_arb/         Triangular arbitrage strategy
│   ├── ai_arb/                 AI/ML strategies (GB predictor + RL agent)
│   ├── nn_arb/                 Neural-network bots (LSTM + DQN, pure NumPy)
│   ├── simulator/              Paper-trading engine + report generator
│   ├── backtest/               Historical backtester
│   ├── dashboard/              Live terminal dashboard
│   └── tests/                  Test suite (pytest)
└── rust/                       High-performance Rust bot
```

---

## Development Workflow

1. **Create a branch** for your change:
   ```bash
   git checkout -b feature/my-new-strategy
   ```

2. **Make your changes** and run the tests:
   ```bash
   make test
   ```

3. **Run the linter** and fix any issues:
   ```bash
   make lint
   ```

4. **Manually verify** your change with the simulator:
   ```bash
   make simulate
   ```

5. **Commit** using a descriptive message:
   ```bash
   git commit -m "feat: add XYZ strategy"
   ```

6. **Open a pull request** against the `main` branch.

---

## Running Tests

The test suite lives in `python/tests/` and uses **pytest**.

```bash
# Run all tests
make test

# Or manually:
cd python
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_portfolio.py -v

# Run tests matching a keyword
python -m pytest tests/ -k "sharpe"
```

Tests do **not** require any API keys or network access — they use mocked or
pre-generated data.

---

## Code Style

- **Python:** Follow [PEP 8](https://pep8.org/) with a line length of **120 characters**.
- **Formatting:** We use [black](https://black.readthedocs.io/):
  ```bash
  make fmt
  ```
- **Type hints:** Add type hints to all public functions and method signatures.
- **Docstrings:** Use standard Python docstrings for all modules and public APIs.
- **Logging:** Use `logging.getLogger(__name__)` — never print inside library code.

---

## Adding a New Strategy

1. Create a new directory under `python/` (e.g. `python/my_strategy/`).
2. Add `__init__.py` and `bot.py` following the pattern in `cross_exchange_arb/bot.py`.
3. Hook your strategy into the simulator by adding a new strategy key in  
   `simulator/engine.py → SimulationEngine.run()`.
4. Expose it in the launcher by adding an entry to `arb.py → main()`.
5. Write tests in `python/tests/test_my_strategy.py`.
6. Document it in `README.md`.

---

## Adding a New Exchange

All Python bots use [ccxt](https://github.com/ccxt/ccxt) which supports 100+ exchanges.

1. Add the new exchange credentials to `python/config/config.example.yaml`.
2. Update `python/setup_wizard.py → EXCHANGES` to include the new exchange.
3. Update the "Supported Exchanges" section in `README.md`.

For the Rust bot, add a new exchange client in `rust/src/exchange.rs` following the pattern for Binance/Bybit.

---

## Pull Request Guidelines

- Keep PRs focused — one logical change per PR.
- Include tests for new functionality.
- Update `README.md` if you change any public-facing interface.
- Ensure `make test` and `make lint` both pass before opening a PR.
- All bots must default to `dry_run: true` — never make PRs that default to live trading.

---

## Security

- **Never commit API keys.** `python/config/config.yaml` and `.env` are in `.gitignore`.
- If you accidentally commit a secret, immediately revoke the key on the exchange.
- Report security vulnerabilities privately by opening a GitHub issue marked **[SECURITY]**.

---

Thanks for helping make Arbitrage Bots better! 🚀
