"""
engine.py — Core paper-trading simulation engine
=================================================

Wires together:
  - MultiExchangeSimulator  (realistic synthetic order books from real OHLCV)
  - PaperPortfolio          (virtual account)
  - Strategy implementations (cross-exchange, triangular, AI spread)

Running the simulation requires NO API keys — it fetches public historical
data from Binance and synthesises multi-exchange order books locally.

Example
-------
    from simulator.engine import SimulationEngine, SimulationConfig

    cfg = SimulationConfig(symbol="BTC/USDT", initial_balance=10_000.0)
    result = SimulationEngine(cfg).run()
    print(result.portfolio.summary())
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator.market_data import (
    Candle,
    MultiExchangeSimulator,
    SimOrderBook,
    fetch_ohlcv_binance,
)
from simulator.portfolio import PaperPortfolio, Trade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """All configurable parameters for a simulation run."""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    candles: int = 500
    initial_balance: float = 10_000.0
    fee_rate: float = 0.001           # 0.1% taker fee per leg
    min_profit_pct: float = 0.15      # lower threshold for paper-trading (spreads are realistic)
    max_trade_size_usdt: float = 500.0
    n_exchanges: int = 3              # number of simulated exchanges
    spread_bps: float = 8.0           # typical half-spread
    strategies: List[str] = field(default_factory=lambda: ["cross_exchange", "ai_spread"])
    seed: int = 42


# ---------------------------------------------------------------------------
# Strategy detectors (strategy-specific signal logic)
# ---------------------------------------------------------------------------

def _calc_profit_pct(buy_price: float, sell_price: float, fee: float) -> float:
    eff_buy  = buy_price  * (1.0 + fee)
    eff_sell = sell_price * (1.0 - fee)
    if eff_buy <= 0:
        return 0.0
    return (eff_sell / eff_buy - 1.0) * 100.0


def detect_cross_exchange(
    books: List[SimOrderBook],
    min_profit_pct: float,
    fee: float,
) -> Optional[Tuple[SimOrderBook, SimOrderBook, float]]:
    """
    Find the best buy/sell pair across all simulated exchanges.

    Returns (buy_book, sell_book, profit_pct) or None.
    """
    best = None
    best_profit = min_profit_pct

    for buy_b, sell_b in itertools.permutations(books, 2):
        profit = _calc_profit_pct(buy_b.ask, sell_b.bid, fee)
        if profit >= best_profit:
            best_profit = profit
            best = (buy_b, sell_b, profit)

    return best


# ---------------------------------------------------------------------------
# Simple AI spread-predictor (lightweight version for simulation)
# ---------------------------------------------------------------------------

class SimpleSpreadPredictor:
    """
    Lightweight rolling-window spread predictor used in the simulation.
    Uses a weighted moving average of recent spreads plus a momentum term.
    No external ML dependencies needed for the simulation.
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._spread_history: List[float] = []

    def update(self, spread: float) -> None:
        self._spread_history.append(spread)
        if len(self._spread_history) > self.window * 2:
            self._spread_history = self._spread_history[-self.window:]

    def predict(self) -> Optional[float]:
        """Return predicted next spread or None if not enough data."""
        h = self._spread_history
        if len(h) < self.window:
            return None
        recent = h[-self.window:]
        # Weighted mean (more weight to recent)
        weights = np.linspace(0.5, 1.5, self.window)
        weighted_mean = float(np.average(recent, weights=weights))
        # Momentum: if last spread > mean, predict slightly higher
        momentum = (recent[-1] - recent[-2]) * 0.3 if len(recent) >= 2 else 0.0
        return weighted_mean + momentum


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    config: SimulationConfig
    portfolio: PaperPortfolio
    candles_processed: int = 0
    opportunities_found: int = 0
    trades_executed: int = 0
    skipped_no_balance: int = 0
    equity_curve: List[float] = field(default_factory=list)
    candle_timestamps: List[int] = field(default_factory=list)
    # Per-candle spread for the best pair (for charting)
    spread_pct_series: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Runs a full paper-trading back-simulation using real OHLCV data.

    Parameters
    ----------
    config: SimulationConfig instance.
    progress_callback: Optional callable(candle_index, total) for progress updates.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        progress_callback=None,
    ) -> None:
        self.cfg = config or SimulationConfig()
        self.progress_callback = progress_callback

    # ------------------------------------------------------------------
    def run(self) -> SimulationResult:
        """
        Execute the full simulation.

        Steps:
        1. Fetch real OHLCV history (public Binance API, no keys needed).
        2. Synthesise multi-exchange order books for each candle.
        3. Run configured strategies on each step.
        4. Execute paper trades via PaperPortfolio.
        5. Return SimulationResult with full statistics.
        """
        cfg = self.cfg
        logger.info(
            "Starting simulation: %s | %s bars | %d exchanges | Balance: $%.0f",
            cfg.symbol, cfg.candles, cfg.n_exchanges, cfg.initial_balance,
        )

        # 1. Fetch real data
        candles = fetch_ohlcv_binance(cfg.symbol, cfg.timeframe, cfg.candles)
        if not candles:
            raise RuntimeError(
                f"Could not fetch OHLCV data for {cfg.symbol}. "
                "Check your internet connection."
            )

        # 2. Set up components
        portfolio   = PaperPortfolio(initial_balance=cfg.initial_balance)
        market_sim  = MultiExchangeSimulator(
            n_exchanges=cfg.n_exchanges,
            spread_bps=cfg.spread_bps,
            seed=cfg.seed,
        )
        predictor   = SimpleSpreadPredictor(window=20)

        result = SimulationResult(config=cfg, portfolio=portfolio)
        total = len(candles)

        # 3. Replay candles
        for i, candle in enumerate(candles):
            if self.progress_callback:
                self.progress_callback(i, total)

            books = market_sim.step(candle)
            for b in books:
                b.symbol = cfg.symbol

            result.candles_processed += 1
            result.candle_timestamps.append(candle.timestamp)
            result.equity_curve.append(portfolio.balance)

            # Track best inter-exchange spread for reporting
            if len(books) >= 2:
                best_ask = min(b.ask for b in books)
                best_bid = max(b.bid for b in books)
                if best_ask > 0:
                    spread_pct = (best_bid - best_ask) / best_ask * 100.0
                    result.spread_pct_series.append(spread_pct)
                    predictor.update(spread_pct)

            # --- Cross-exchange strategy ---
            if "cross_exchange" in cfg.strategies:
                opp = detect_cross_exchange(books, cfg.min_profit_pct, cfg.fee_rate)
                if opp:
                    buy_b, sell_b, profit_pct = opp
                    result.opportunities_found += 1
                    self._execute_cross_exchange(
                        portfolio, candle, cfg, buy_b, sell_b, profit_pct
                    )
                    if result.trades_executed < len(portfolio.closed_trades):
                        result.trades_executed = len(portfolio.closed_trades)

            # --- AI spread strategy ---
            if "ai_spread" in cfg.strategies:
                pred = predictor.predict()
                if pred is not None and pred >= cfg.min_profit_pct:
                    if len(books) >= 2:
                        sorted_by_ask = sorted(books, key=lambda b: b.ask)
                        buy_b  = sorted_by_ask[0]
                        sell_b = sorted(books, key=lambda b: b.bid, reverse=True)[0]
                        profit = _calc_profit_pct(buy_b.ask, sell_b.bid, cfg.fee_rate)
                        if profit > 0:
                            result.opportunities_found += 1
                            self._execute_cross_exchange(
                                portfolio, candle, cfg, buy_b, sell_b, profit,
                                strategy="ai_spread",
                            )

        # Close any trades still open at simulation end
        for trade_id, trade in list(portfolio._open_trades.items()):
            gross = trade.size_usdt * 0.0  # assume flat exit
            portfolio.close_trade(trade_id, gross, timestamp=candles[-1].timestamp)

        result.trades_executed = len(portfolio.closed_trades)
        result.equity_curve.append(portfolio.balance)

        logger.info(
            "Simulation complete: %d candles | %d trades | PnL: $%.2f (%.3f%%)",
            result.candles_processed,
            result.trades_executed,
            portfolio.total_pnl,
            portfolio.total_return_pct,
        )
        return result

    # ------------------------------------------------------------------
    def _execute_cross_exchange(
        self,
        portfolio: PaperPortfolio,
        candle: Candle,
        cfg: SimulationConfig,
        buy_b: SimOrderBook,
        sell_b: SimOrderBook,
        profit_pct: float,
        strategy: str = "cross_exchange",
    ) -> None:
        """
        Paper-execute one cross-exchange arbitrage round trip.

        The gross PnL is computed from the actual buy/sell prices in the
        simulated order books, making the simulation realistic.
        """
        if portfolio.balance < 10:
            return

        size = min(cfg.max_trade_size_usdt, portfolio.balance * 0.25)
        amount = size / buy_b.ask

        # Gross PnL = coins_bought * (sell_price - buy_price)
        gross_pnl = amount * (sell_b.bid - buy_b.ask)

        trade_id = portfolio.open_trade(
            symbol=cfg.symbol,
            strategy=strategy,
            buy_exchange=buy_b.exchange,
            sell_exchange=sell_b.exchange,
            buy_price=buy_b.ask,
            sell_price=sell_b.bid,
            size_usdt=size,
            fee_rate=cfg.fee_rate,
            timestamp=candle.timestamp,
        )

        if trade_id > 0:
            portfolio.close_trade(trade_id, gross_pnl, timestamp=candle.timestamp)
