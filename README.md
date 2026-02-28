# Abritage-Bots 🤖💹

A collection of **cryptocurrency arbitrage bots** written in **Python** and **Rust**, including two **AI/ML-powered** strategies.

> ⚠️ **All bots run in `dry_run: true` mode by default — no real orders are placed until you explicitly disable it and provide live API credentials.**

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
Abritage-Bots/
├── python/
│   ├── requirements.txt              # Python dependencies
│   ├── config/
│   │   └── config.example.yaml      # Copy → config.yaml, fill in your keys
│   ├── utils/
│   │   └── exchange.py              # Shared ccxt helpers
│   ├── cross_exchange_arb/
│   │   └── bot.py                   # Cross-exchange arbitrage bot
│   ├── triangular_arb/
│   │   └── bot.py                   # Triangular arbitrage bot
│   └── ai_arb/
│       └── bot.py                   # AI/ML arbitrage bot (spread predictor + RL)
└── rust/
    ├── Cargo.toml
    └── src/
        ├── main.rs                  # Entry point + main loop
        ├── exchange.rs              # Exchange clients (Binance, Bybit)
        ├── arbitrage.rs             # Opportunity scanner
        └── config.rs                # Config deserialization
```

---

## Quick Start (Python)

### 1. Install dependencies

```bash
cd python
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml and add your exchange API keys.
# Leave  sandbox: true  and  dry_run: true  until you are confident.
```

### 3. Run a bot

```bash
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
