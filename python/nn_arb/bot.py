"""
bot.py — Neural-Network Arbitrage Bot runner
=============================================

Runs the LSTM spread predictor and/or the DQN trading agent against either:

  • Simulated order books from real historical Binance OHLCV data
    (default — no API keys needed)

  • Live exchange order books via ccxt (requires python/config/config.yaml)

Usage
-----
    # From the python/ directory:
    python -m nn_arb.bot                        # simulate both NN strategies
    python -m nn_arb.bot --strategy lstm        # LSTM predictor only
    python -m nn_arb.bot --strategy dqn         # DQN agent only
    python -m nn_arb.bot --symbol ETH/USDT --candles 720
    python -m nn_arb.bot --symbol BTC/USDT --balance 50000

Options
-------
    --symbol     Trading pair          (default: BTC/USDT)
    --timeframe  Candle size           (default: 1h)
    --candles    Historical bars        (default: 500)
    --balance    Starting balance USDT (default: 10000)
    --strategy   lstm | dqn | both     (default: both)
    --min-spread Minimum spread % to trade on (default: 0.10)
    --fee        Taker fee rate per leg   (default: 0.001)
    --exchanges  Number of simulated exchanges (default: 3)
    --seed       Random seed               (default: 42)
    --save-dir   Directory to save trained model weights (default: models/)
    --load       Load pre-trained weights before running
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_arb.dqn_agent import (
    ACTION_ENTER, ACTION_EXIT, ACTION_HOLD,
    DQNAgent, compute_reward,
    STATE_DIM,
)
from nn_arb.lstm_predictor import SpreadLSTMPredictor
from simulator.market_data import Candle, MultiExchangeSimulator, SimOrderBook, fetch_ohlcv_binance
from simulator.portfolio import PaperPortfolio

logger = logging.getLogger("nn_arb")


# ---------------------------------------------------------------------------
# Feature builder for DQN state
# ---------------------------------------------------------------------------

class _DQNFeatureBuilder:
    """Builds DQN state vectors from streaming order-book data."""

    def __init__(self, window: int = 20) -> None:
        self._spreads:  List[float] = []
        self._mids:     List[float] = []
        self._window    = window
        self._position  = 0
        self._steps_held = 0
        self._entry_spread: float = 0.0

    def update_position(self, position: int, entry_spread: float = 0.0) -> None:
        self._position = position
        if position == 1:
            self._entry_spread = entry_spread
            self._steps_held   = 0
        else:
            self._steps_held  = 0
            self._entry_spread = 0.0

    def step_held(self) -> None:
        if self._position == 1:
            self._steps_held += 1

    def build(self, books: List[SimOrderBook]) -> Optional[np.ndarray]:
        if len(books) < 2:
            return None

        best_ask = min(b.ask for b in books)
        best_bid = max(b.bid for b in books)
        avg_mid  = sum(b.mid for b in books) / len(books)

        spread_pct = (best_bid - best_ask) / best_ask * 100.0 if best_ask > 0 else 0.0

        self._spreads.append(spread_pct)
        self._mids.append(avg_mid)

        spread_arr = np.array(self._spreads[-max(self._window, 15):])
        mid_arr    = np.array(self._mids[-max(self._window, 15):])

        # Feature 0: current spread
        # Feature 1: 10-step MA of spread
        spread_ma10 = float(spread_arr[-min(10, len(spread_arr)):].mean())

        # Feature 2: % change in mid price
        mid_change = 0.0
        if len(mid_arr) >= 2 and mid_arr[-2] > 0:
            mid_change = (mid_arr[-1] - mid_arr[-2]) / mid_arr[-2] * 100.0

        # Feature 3: volume imbalance
        asks = sorted(books, key=lambda b: b.ask)
        bids = sorted(books, key=lambda b: b.bid, reverse=True)
        v_ask = asks[0].ask_qty
        v_bid = bids[0].bid_qty
        vol_imbalance = (v_bid - v_ask) / (v_bid + v_ask + 1e-9)

        # Feature 4: RSI (scaled to 0–1)
        rsi = 50.0
        if len(mid_arr) >= 15:
            deltas = np.diff(mid_arr[-15:])
            up   = deltas[deltas > 0]
            down = deltas[deltas < 0]
            ag = up.mean()    if len(up)   > 0 else 0.0
            al = -down.mean() if len(down) > 0 else 0.0
            if al > 0:
                rsi = 100.0 - 100.0 / (1.0 + ag / al)
        rsi_norm = rsi / 100.0

        # Feature 5: position flag
        position_flag = float(self._position)

        # Feature 6: steps held (normalised)
        steps_norm = min(self._steps_held / 100.0, 1.0)

        # Feature 7: unrealised P&L % (0 if flat)
        unrealised = 0.0
        if self._position == 1 and self._entry_spread > 0:
            unrealised = (spread_pct - self._entry_spread) * 0.5

        return np.array([
            spread_pct * 10.0,    # scale for better NN conditioning
            spread_ma10 * 10.0,
            mid_change,
            vol_imbalance,
            rsi_norm,
            position_flag,
            steps_norm,
            unrealised,
        ], dtype=float)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    symbol:       str   = "BTC/USDT",
    timeframe:    str   = "1h",
    candles:      int   = 500,
    balance:      float = 10_000.0,
    strategies:   List[str] = None,
    min_spread:   float = 0.10,
    fee_rate:     float = 0.001,
    n_exchanges:  int   = 3,
    seed:         int   = 42,
    save_dir:     Optional[str] = None,
    load_weights: bool  = False,
) -> dict:
    """
    Run the NN bots against simulated order books built from real OHLCV data.

    Returns a summary dict with P&L statistics.
    """
    strategies = strategies or ["lstm", "dqn"]

    logger.info("Fetching %d %s candles for %s …", candles, timeframe, symbol)
    candle_list = fetch_ohlcv_binance(symbol, timeframe, candles)
    if not candle_list:
        raise RuntimeError(
            f"Could not fetch OHLCV data for {symbol}. "
            "Check your internet connection."
        )

    market_sim    = MultiExchangeSimulator(n_exchanges=n_exchanges, seed=seed)
    portfolio_lstm = PaperPortfolio(balance)
    portfolio_dqn  = PaperPortfolio(balance)

    lstm_predictor = SpreadLSTMPredictor(seq_len=20, hidden_size=32, seed=seed)
    dqn_agent      = DQNAgent(seed=seed)

    # Optionally load pre-trained weights
    if load_weights and save_dir:
        lstm_path = Path(save_dir) / f"lstm_{symbol.replace('/', '')}.npz"
        dqn_path  = Path(save_dir) / f"dqn_{symbol.replace('/', '')}.npz"
        if lstm_path.exists():
            lstm_predictor.load(str(lstm_path))
            logger.info("Loaded LSTM weights from %s", lstm_path)
        if dqn_path.exists():
            dqn_agent.load(str(dqn_path))
            logger.info("Loaded DQN weights from %s", dqn_path)

    dqn_builder = _DQNFeatureBuilder()

    # DQN tracking state
    dqn_position   = 0
    dqn_trade_id   = -1
    dqn_entry_ask  = 0.0
    dqn_entry_bid  = 0.0
    dqn_prev_state: Optional[np.ndarray] = None
    dqn_prev_action = ACTION_HOLD

    # LSTM tracking state
    lstm_trade_id     = -1
    lstm_in_trade     = False
    lstm_entry_ask    = 0.0
    lstm_entry_bid    = 0.0
    lstm_buy_exchange = ""
    lstm_sell_exchange = ""

    results: dict = {
        "candles": len(candle_list),
        "lstm_trades": 0, "lstm_pnl": 0.0,
        "dqn_trades":  0, "dqn_pnl":  0.0,
    }

    try:
        from rich.progress import Progress, BarColumn, MofNCompleteColumn, SpinnerColumn, TextColumn, TimeRemainingColumn
        _progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Training NN bots…"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("candles"),
            TimeRemainingColumn(),
        )
        _task = _progress.add_task("nn", total=len(candle_list))
        _progress.start()
    except ImportError:
        _progress = None
        _task = None

    for step_idx, candle in enumerate(candle_list):
        books = market_sim.step(candle)
        for b in books:
            b.symbol = symbol

        if _progress:
            _progress.update(_task, completed=step_idx + 1)

        best_ask = min(b.ask for b in books)
        best_bid = max(b.bid for b in books)
        buy_book  = min(books, key=lambda b: b.ask)
        sell_book = max(books, key=lambda b: b.bid)
        spread_pct = (best_bid - best_ask) / best_ask * 100.0 if best_ask > 0 else 0.0

        # ── LSTM strategy ──────────────────────────────────────────────────
        if "lstm" in strategies:
            lstm_predictor.observe(books)

            if lstm_predictor.can_train():
                lstm_predictor.train()

            pred = lstm_predictor.predict()

            # Decision: predicted spread > min_spread → enter; in trade → check exit
            if not lstm_in_trade and pred is not None and pred >= min_spread:
                size = min(500.0, portfolio_lstm.balance * 0.25)
                trade_id = portfolio_lstm.open_trade(
                    symbol=symbol,
                    strategy="lstm_nn",
                    buy_exchange=buy_book.exchange,
                    sell_exchange=sell_book.exchange,
                    buy_price=buy_book.ask,
                    sell_price=sell_book.bid,
                    size_usdt=size,
                    fee_rate=fee_rate,
                    timestamp=candle.timestamp,
                )
                if trade_id > 0:
                    lstm_trade_id      = trade_id
                    lstm_in_trade      = True
                    lstm_entry_ask     = buy_book.ask
                    lstm_entry_bid     = sell_book.bid
                    lstm_buy_exchange  = buy_book.exchange
                    lstm_sell_exchange = sell_book.exchange

            elif lstm_in_trade:
                # Exit when the predicted spread has collapsed below 30% of the entry threshold
                exit_condition = (pred is not None and pred < min_spread * 0.3)
                if exit_condition and lstm_trade_id in portfolio_lstm._open_trades:
                    gross_pnl = portfolio_lstm._open_trades[lstm_trade_id].size_usdt / lstm_entry_ask * (sell_book.bid - lstm_entry_ask)
                    portfolio_lstm.close_trade(lstm_trade_id, gross_pnl, timestamp=candle.timestamp)
                    lstm_in_trade = False

        # ── DQN strategy ───────────────────────────────────────────────────
        if "dqn" in strategies:
            dqn_builder.step_held()
            state = dqn_builder.build(books)

            if state is not None:
                action = dqn_agent.select_action(state, explore=True)

                # Execute action
                reward_now = 0.0
                if action == ACTION_ENTER and dqn_position == 0:
                    size = min(500.0, portfolio_dqn.balance * 0.25)
                    if spread_pct >= min_spread:
                        trade_id = portfolio_dqn.open_trade(
                            symbol=symbol,
                            strategy="dqn_nn",
                            buy_exchange=buy_book.exchange,
                            sell_exchange=sell_book.exchange,
                            buy_price=buy_book.ask,
                            sell_price=sell_book.bid,
                            size_usdt=size,
                            fee_rate=fee_rate,
                            timestamp=candle.timestamp,
                        )
                        if trade_id > 0:
                            dqn_trade_id  = trade_id
                            dqn_position  = 1
                            dqn_entry_ask = buy_book.ask
                            dqn_entry_bid = sell_book.bid
                            dqn_builder.update_position(1, spread_pct)
                            reward_now, _ = compute_reward(ACTION_ENTER, 0, fee_pct=fee_rate * 100)
                        else:
                            reward_now = -0.001
                    else:
                        reward_now = -0.001  # tried to enter but spread too tight

                elif action == ACTION_EXIT and dqn_position == 1 and dqn_trade_id > 0:
                    trade = portfolio_dqn._open_trades.get(dqn_trade_id)
                    if trade:
                        gross_pnl = trade.size_usdt / dqn_entry_ask * (sell_book.bid - dqn_entry_ask)
                        realised_pct = (sell_book.bid - dqn_entry_ask) / dqn_entry_ask * 100.0
                        portfolio_dqn.close_trade(dqn_trade_id, gross_pnl, timestamp=candle.timestamp)
                        reward_now, _ = compute_reward(ACTION_EXIT, 1, realised_pct, fee_rate * 100)
                        dqn_position = 0
                        dqn_trade_id = -1
                        dqn_builder.update_position(0)
                    else:
                        dqn_position = 0

                else:
                    reward_now, _ = compute_reward(action, dqn_position)

                # Store transition and train
                if dqn_prev_state is not None:
                    dqn_agent.store(dqn_prev_state, dqn_prev_action, reward_now, state)
                    dqn_agent.train_step()

                dqn_prev_state  = state.copy()
                dqn_prev_action = action

    if _progress:
        _progress.stop()

    # Close any open trades at end of simulation (flat exit, 0 gross PnL)
    last_ts = candle_list[-1].timestamp if candle_list else 0
    for portfolio in [portfolio_lstm, portfolio_dqn]:
        for tid in list(portfolio._open_trades.keys()):
            portfolio.close_trade(tid, 0.0, timestamp=last_ts)

    # Optionally save trained weights
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        sym_clean = symbol.replace("/", "")
        if "lstm" in strategies:
            lstm_predictor.save(str(Path(save_dir) / f"lstm_{sym_clean}.npz"))
        if "dqn" in strategies:
            dqn_agent.save(str(Path(save_dir) / f"dqn_{sym_clean}.npz"))
        logger.info("Saved model weights to %s/", save_dir)

    results.update({
        "lstm_trades":        len(portfolio_lstm.closed_trades),
        "lstm_pnl":           portfolio_lstm.total_pnl,
        "lstm_final_balance": portfolio_lstm.balance,
        "lstm_win_rate":      portfolio_lstm.win_rate,
        "dqn_trades":         len(portfolio_dqn.closed_trades),
        "dqn_pnl":            portfolio_dqn.total_pnl,
        "dqn_final_balance":  portfolio_dqn.balance,
        "dqn_win_rate":       portfolio_dqn.win_rate,
        "lstm_n_trained":     lstm_predictor.n_trained,
        "dqn_epsilon_final":  dqn_agent.epsilon,
    })
    return results


# ---------------------------------------------------------------------------
# Terminal results printer
# ---------------------------------------------------------------------------

def _print_results(results: dict, balance: float, strategies: List[str]) -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        t = Table(box=box.ROUNDED, show_header=True,
                  header_style="bold magenta", padding=(0, 1))
        t.add_column("Metric",       style="cyan",  min_width=28)
        t.add_column("LSTM Spread NN", justify="right", min_width=16)
        t.add_column("DQN RL Agent",   justify="right", min_width=16)

        def _fmt(val, is_pct=False, is_money=False):
            if val is None:
                return "—"
            if is_money:
                color = "green" if val >= 0 else "red"
                return f"[{color}]${val:+,.2f}[/]"
            if is_pct:
                color = "green" if val >= 0 else "red"
                return f"[{color}]{val:+.2f}%[/]"
            return str(val)

        rows = [
            ("Starting balance",     f"${balance:,.0f}",                    f"${balance:,.0f}"),
            ("Final balance",
             _fmt(results.get("lstm_final_balance"), is_money=True),
             _fmt(results.get("dqn_final_balance"),  is_money=True)),
            ("Total P&L",
             _fmt(results.get("lstm_pnl"),           is_money=True),
             _fmt(results.get("dqn_pnl"),            is_money=True)),
            ("Trades executed",
             str(results.get("lstm_trades", 0)),
             str(results.get("dqn_trades", 0))),
            ("Win rate",
             _fmt(results.get("lstm_win_rate"),      is_pct=True),
             _fmt(results.get("dqn_win_rate"),       is_pct=True)),
            ("LSTM training steps",
             str(results.get("lstm_n_trained", 0)),  "—"),
            ("DQN ε (final)",
             "—",
             f"{results.get('dqn_epsilon_final', 0):.3f}"),
        ]
        for row in rows:
            t.add_row(*row)

        console.print()
        console.print(Panel("[bold cyan]🧠  Neural Network Arbitrage Bot — Results[/]",
                            border_style="cyan"))
        console.print(t)
        console.print()

    except ImportError:
        print("\n=== Neural Network Arbitrage Bot Results ===")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nn_arb.bot",
        description=(
            "Neural-Network Arbitrage Bot — LSTM + DQN strategies.\n"
            "Uses real historical OHLCV prices. No API keys required!"
        ),
    )
    p.add_argument("--symbol",     default="BTC/USDT")
    p.add_argument("--timeframe",  default="1h")
    p.add_argument("--candles",    type=int,   default=500)
    p.add_argument("--balance",    type=float, default=10_000.0)
    p.add_argument("--strategy",   default="both",
                   help="lstm | dqn | both (default: both)")
    p.add_argument("--min-spread", type=float, default=0.10, dest="min_spread")
    p.add_argument("--fee",        type=float, default=0.001)
    p.add_argument("--exchanges",  type=int,   default=3)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save-dir",   default="models", dest="save_dir",
                   help="Directory to save trained model weights (default: models/)")
    p.add_argument("--load",       action="store_true",
                   help="Load pre-trained weights before running")
    return p


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.WARNING,
    )
    logging.getLogger("nn_arb").setLevel(logging.INFO)

    args = _build_parser().parse_args()

    strategies = {
        "both": ["lstm", "dqn"],
        "lstm": ["lstm"],
        "dqn":  ["dqn"],
    }.get(args.strategy.lower(), ["lstm", "dqn"])

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        c = Console()
        banner = Text.assemble(
            ("🧠  Neural-Network Arbitrage Bot\n", "bold cyan"),
            ("   Pure-NumPy LSTM · Deep Q-Network · Real OHLCV prices\n\n", "dim"),
            (f"   Symbol    : {args.symbol}\n", "white"),
            (f"   Timeframe : {args.timeframe}\n", "white"),
            (f"   Candles   : {args.candles}\n", "white"),
            (f"   Balance   : ${args.balance:,.0f} USDT\n", "white"),
            (f"   Strategies: {', '.join(strategies)}\n", "white"),
        )
        c.print(Panel(banner, border_style="cyan"))
    except ImportError:
        print(f"\nNN Arb Bot | {args.symbol} | {args.candles} candles")

    t0 = time.time()
    try:
        results = run_simulation(
            symbol=args.symbol,
            timeframe=args.timeframe,
            candles=min(args.candles, 1000),
            balance=args.balance,
            strategies=strategies,
            min_spread=args.min_spread,
            fee_rate=args.fee,
            n_exchanges=args.exchanges,
            seed=args.seed,
            save_dir=args.save_dir,
            load_weights=args.load,
        )
    except RuntimeError as e:
        print(f"\n❌  Error: {e}")
        sys.exit(1)

    print(f"\n  ✓ Finished in {time.time() - t0:.1f}s\n")
    _print_results(results, args.balance, strategies)


if __name__ == "__main__":
    main()
