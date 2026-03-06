# Arbitrage Bots 🤖💹

A collection of **cryptocurrency arbitrage bots** written in **Python** and **Rust**, including two **AI/ML-powered** strategies — plus a full **paper-trading simulator** that uses real historical market prices so you can test everything without spending a single dollar.

> ⚠️ **All bots run in `dry_run: true` mode by default — no real orders are placed until you explicitly disable it and provide live API credentials.**

---

## ⚡ 30-Second Quick Start

**No API keys needed to try the simulator!**

```bash
# 1. Clone the repo
git clone https://github.com/HRnewcoll/Arbitrage-Bots.git
cd Arbitrage-Bots

# 2. One-command setup (creates venv, installs deps, copies config)
./setup.sh          # Linux / macOS
setup.bat           # Windows (double-click or run in Command Prompt)

# 3. Activate the environment and launch the interactive menu
source python/.venv/bin/activate   # Linux/macOS
python\.venv\Scripts\activate      # Windows (in Command Prompt)

python arb.py
```

The launcher opens a numbered menu — type `1` and press Enter to run a full
paper-trading simulation on real BTC/USDT price history. No config required.

### Direct shortcuts (bypass the menu)

```bash
python arb.py simulate          # paper-trading sim  — no API keys needed
python arb.py backtest          # strategy backtest  — no API keys needed
python arb.py wizard            # interactive API key & settings wizard
python arb.py bot cross         # run cross-exchange bot (needs config)
python arb.py bot tri           # run triangular bot (needs config)
python arb.py bot ai            # run AI bot (needs config)
python arb.py help              # show all commands
```

---

## ✨ What's Inside

| Component | What it does |
|-----------|-------------|
| 🔄 **Cross-Exchange Bot** | Buys low on one exchange, sells high on another |
| 🔺 **Triangular Arb Bot** | Exploits 3-currency cycles on a single exchange |
| 🤖 **AI Spread Predictor** | Gradient Boosting model predicts the next price spread |
| 🧠 **AI RL Agent** | Q-learning agent that learns optimal entry/exit over time |
| ⚡ **Rust Bot** | High-performance async scanner (Binance + Bybit) |
| 📊 **Paper-Trading Simulator** | Replays **real** historical prices, no API keys, no money |
| 📈 **Backtester** | Measures Sharpe, Sortino, Calmar, drawdown, win rate |
| 🖥️ **Live Dashboard** | Colour terminal dashboard for real bot monitoring |

---

## 🚀 Paper-Trading Simulator — Test Without Any Money

The simulator fetches **real** historical OHLCV data from Binance's public API (no API key required) and creates a realistic multi-exchange environment with synthetic bid/ask spreads.  You get:

- ✅ Real price levels (BTC, ETH, SOL — whatever you pick)
- ✅ Simulated per-exchange spread noise (AR(1) correlated, realistic)
- ✅ Virtual portfolio with P&L tracking
- ✅ Rich colour terminal report
- ✅ Interactive HTML report with charts (opens in any browser)
- ✅ Zero setup — just install requirements and run

### Run a simulation instantly

```bash
cd python
pip install -r requirements.txt

# Default: BTC/USDT, 500 hourly candles, $10 000 starting balance
python -m simulator.run

# Customise the symbol, timeframe and balance
python -m simulator.run --symbol ETH/USDT --candles 720 --balance 5000
python -m simulator.run --symbol SOL/USDT --timeframe 15m --candles 1000

# Save the HTML report to a specific file
python -m simulator.run --symbol BTC/USDT --output btc_sim.html
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | `BTC/USDT` | Any coin pair (e.g. ETH/USDT, SOL/USDT, XRP/USDT) |
| `--timeframe` | `1h` | Candle size: `1m` `5m` `15m` `1h` `4h` `1d` |
| `--candles` | `500` | How many historical bars to replay (max 1000) |
| `--balance` | `10000` | Starting virtual USDT balance |
| `--strategy` | `all` | `cross_exchange` / `ai_spread` / `all` |
| `--exchanges` | `3` | Number of simulated exchanges |
| `--min-profit` | `0.15` | Minimum spread % to trigger a paper trade |
| `--fee` | `0.001` | Taker fee per leg (0.001 = 0.1%) |
| `--output` | `simulation_report.html` | HTML report output path |
| `--no-html` | — | Skip HTML report |

---

## 📈 Backtester

Compare strategy performance side-by-side with full risk metrics:

```bash
cd python

# Backtest all strategies on BTC/USDT
python -m backtest.backtest --symbol BTC/USDT --candles 1000

# Compare cross_exchange vs ai_spread side by side
python -m backtest.backtest --symbol ETH/USDT --compare

# Customise timeframe and starting balance
python -m backtest.backtest --symbol SOL/USDT --timeframe 4h --balance 50000
```

**Metrics reported:**
- Total & annualised return
- Sharpe ratio, Sortino ratio, Calmar ratio
- Max drawdown
- Win rate & profit factor
- Average trade P&L and duration

---

## 🖥️ Live Terminal Dashboard

Attach a colour dashboard to any running bot:

```python
from dashboard.terminal import LiveDashboard

dash = LiveDashboard(symbols=["BTC/USDT", "ETH/USDT"], bot_name="My Arb Bot")
dash.set_initial_balance(10_000.0)
dash.start()

# In your bot loop:
dash.update_price("binance", "BTC/USDT", 65000.0)
dash.update_price("kraken",  "BTC/USDT", 65120.0)
dash.add_opportunity("BTC/USDT", "binance", "kraken", 65000, 65120, 0.18)
dash.update_balance(10_023.50)
dash.add_trade("BTC/USDT", "binance", "kraken", 65000, 65120, 1.20)
```

The dashboard renders live prices, detected opportunities, trade history, and P&L — all updating in real time in your terminal.

---

## Bots at a Glance

| Bot | Language | Strategy | File |
|-----|----------|----------|------|
| Cross-Exchange Arbitrage | Python | Buy low on one exchange, sell high on another | `python/cross_exchange_arb/bot.py` |
| Triangular Arbitrage | Python | 3-currency circular trade on a single exchange | `python/triangular_arb/bot.py` |
| **AI Spread Predictor** | **Python** | **ML (Gradient Boosting) predicts next spread** | `python/ai_arb/bot.py` |
| **AI RL Trading Agent** | **Python** | **Q-learning agent learns optimal entry/exit** | `python/ai_arb/bot.py` |
| Cross-Exchange Arbitrage | Rust | Fast async order-book scanner (Binance + Bybit) | `rust/src/main.rs` |

---

## Supported Exchanges

All Python bots use [ccxt](https://github.com/ccxt/ccxt) and support **100+ exchanges** out of the box, including:

- Binance / Binance Testnet
- Kraken
- Coinbase (Advanced Trade)
- Bybit
- KuCoin
- OKX
- Gate.io
- Huobi / HTX
- …and many more — just add them to `config.yaml`

The Rust bot currently supports **Binance** and **Bybit** directly (testnet-ready).

---

## Repository Structure

```
Arbitrage-Bots/
├── arb.py                            # ★ Unified interactive launcher — start here!
├── setup.sh                          # ★ One-command setup (Linux/macOS)
├── setup.bat                         # ★ One-command setup (Windows)
├── .env.example                      # Environment-variable API key template
├── python/
│   ├── requirements.txt              # Python dependencies
│   ├── setup_wizard.py               # ★ Interactive config wizard
│   ├── config/
│   │   └── config.example.yaml      # Config template (copy → config.yaml)
│   ├── utils/
│   │   └── exchange.py              # Shared ccxt helpers
│   ├── cross_exchange_arb/
│   │   └── bot.py                   # Cross-exchange arbitrage bot
│   ├── triangular_arb/
│   │   └── bot.py                   # Triangular arbitrage bot
│   ├── ai_arb/
│   │   └── bot.py                   # AI/ML arbitrage bot (spread predictor + RL)
│   ├── simulator/                   # ★ Paper-trading simulator (no API keys needed)
│   │   ├── market_data.py           #   Real OHLCV fetcher + multi-exchange spread synthesiser
│   │   ├── portfolio.py             #   Virtual portfolio — balance, positions, P&L
│   │   ├── engine.py                #   Core simulation engine
│   │   ├── report.py                #   Rich terminal + HTML report generator
│   │   └── run.py                   #   CLI entry point
│   ├── backtest/                    # ★ Historical strategy backtester
│   │   └── backtest.py              #   Sharpe, Sortino, Calmar, drawdown, win rate
│   └── dashboard/                   # ★ Live terminal dashboard
│       └── terminal.py              #   Rich colour live display for real bot runs
└── rust/
    ├── Cargo.toml
    └── src/
        ├── main.rs                  # Entry point + main loop
        ├── exchange.rs              # Exchange clients (Binance, Bybit)
        ├── arbitrage.rs             # Opportunity scanner
        └── config.rs                # Config deserialization
```

---

## Quick Start (Python — manual, without setup.sh)

### 1. Install dependencies

```bash
cd python
pip install -r requirements.txt
```

### 2. Configure

```bash
# Option A — Interactive wizard (recommended)
python setup_wizard.py

# Option B — Copy and edit manually
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml and add your exchange API keys.
# Leave  sandbox: true  and  dry_run: true  until you are confident.
```

### 3. Run a bot

```bash
# Use the interactive menu (easiest)
cd ..         # back to repo root
python arb.py

# Or run directly with module paths:
cd python

# Cross-exchange arbitrage
python -m cross_exchange_arb.bot --config config/config.yaml

# Triangular arbitrage
python -m triangular_arb.bot --config config/config.yaml

# AI bot — both strategies (default)
python -m ai_arb.bot --config config/config.yaml --strategy both

# AI bot — spread predictor only
python -m ai_arb.bot --config config/config.yaml --strategy spread

# AI bot — RL agent only
python -m ai_arb.bot --config config/config.yaml --strategy rl
```

---

## Quick Start (Rust)

```bash
cd rust

# Debug build (faster compile)
cargo run -- --config ../python/config/config.yaml

# Optimised release build (faster execution — recommended for live trading)
cargo build --release
./target/release/arb-bot --config ../python/config/config.yaml
```

Set `RUST_LOG=debug` for verbose per-request logging:

```bash
RUST_LOG=debug cargo run -- --config ../python/config/config.yaml
```

---

## Bot Details

### Cross-Exchange Arbitrage (Python & Rust)

Continuously fetches order books from every configured exchange and checks
all pairs. When:

```
(best_bid_on_exchange_B / best_ask_on_exchange_A) - 1 - fees  ≥  min_profit_pct
```

…the bot logs (dry-run) or executes simultaneous market buy + sell orders.

### Triangular Arbitrage (Python)

On a single exchange, enumerates every 3-currency cycle involving the
`base_currency` (default `USDT`):

```
USDT → BTC → ETH → USDT
```

Calculates the net rate product after fees. Trades when profit > threshold.

### AI Spread Predictor (Python)

- Fetches `history_candles` minutes of OHLCV from both exchanges.
- Engineers features: RSI, EMA, spread z-score, correlation, etc.
- Trains a **Gradient Boosting regressor** to predict the next-step spread.
- Retrains every 100 polling cycles on fresh data.
- Model persisted to `python/models/spread_predictor.pkl`.

### AI RL Trading Agent (Python)

- Discretises the spread z-score into buckets to form a compact state space.
- Uses **tabular Q-learning** (ε-greedy) with actions: Hold / Open / Close.
- Trains through simulated episodes on historical spread data.
- Q-table persisted to `python/models/rl_q_table.pkl` — grows smarter over time.

---

## Configuration Reference

See [`python/config/config.example.yaml`](python/config/config.example.yaml) for the full annotated example.

Key settings:

| Key | Default | Description |
|-----|---------|-------------|
| `arbitrage.dry_run` | `true` | Log opportunities only — **no real orders** |
| `arbitrage.min_profit_pct` | `0.5` | Minimum net profit % before trading |
| `arbitrage.max_trade_size_usdt` | `100` | Maximum position size per trade |
| `arbitrage.poll_interval_seconds` | `5` | How often to scan prices |
| `ai.prediction_threshold` | `0.4` | Min predicted spread % for AI signal |
| `ai.history_candles` | `500` | Candles fetched for model training |

---

## Security

- **Never commit `config.yaml`** — it is listed in `.gitignore`.
- API keys are read from `config.yaml` or from environment variables  
  (`BINANCE_API_KEY`, `BINANCE_API_SECRET`, `KRAKEN_API_KEY`, …).
- Always test with `sandbox: true` and small `max_trade_size_usdt` first.
- The bots do not withdraw funds; only spot trading permissions are needed.

---

## Disclaimer

These bots are provided for **educational purposes**. Cryptocurrency trading
carries significant financial risk. Past performance does not guarantee future
results. Always test thoroughly in sandbox/paper-trading mode before risking
real funds.
