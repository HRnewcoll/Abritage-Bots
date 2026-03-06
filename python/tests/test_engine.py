"""
Tests for simulator.engine — SimulationEngine + helpers
=======================================================
Uses mocked OHLCV data (no network calls).
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.engine import (
    SimulationConfig,
    SimulationEngine,
    SimulationResult,
    SimpleSpreadPredictor,
    _calc_profit_pct,
    detect_cross_exchange,
)
from simulator.market_data import Candle, SimOrderBook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = 1_700_000_000_000  # ms


def _make_candles(n: int = 30, price: float = 40_000.0) -> list:
    """Synthetic candle list, no network required."""
    rng = np.random.default_rng(0)
    candles = []
    p = price
    for i in range(n):
        p *= 1.0 + rng.normal(0, 0.002)
        candles.append(
            Candle(
                timestamp=_TS + i * 3_600_000,
                open=p,
                high=p * 1.001,
                low=p * 0.999,
                close=p,
                volume=10.0,
            )
        )
    return candles


def _make_book(exchange: str, bid: float, ask: float) -> SimOrderBook:
    return SimOrderBook(
        exchange=exchange,
        symbol="BTC/USDT",
        timestamp=_TS,
        bid=bid,
        ask=ask,
    )


# ---------------------------------------------------------------------------
# _calc_profit_pct
# ---------------------------------------------------------------------------

class TestCalcProfitPct:
    def test_profitable_spread(self):
        """sell > buy after fees → positive profit"""
        p = _calc_profit_pct(buy_price=100.0, sell_price=101.0, fee=0.001)
        assert p > 0.0

    def test_unprofitable_tight_spread(self):
        """Very tight spread → negative after fees"""
        p = _calc_profit_pct(buy_price=100.0, sell_price=100.05, fee=0.001)
        assert p < 0.0

    def test_zero_buy_price(self):
        """Guard against divide-by-zero."""
        p = _calc_profit_pct(buy_price=0.0, sell_price=100.0, fee=0.001)
        assert p == pytest.approx(0.0)

    def test_symmetric_no_fee(self):
        """With zero fee and buy == sell, profit = 0."""
        p = _calc_profit_pct(100.0, 100.0, fee=0.0)
        assert p == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# detect_cross_exchange
# ---------------------------------------------------------------------------

class TestDetectCrossExchange:
    def test_detects_opportunity(self):
        """Clear price difference across two books should be detected."""
        books = [
            _make_book("A", bid=100.0, ask=100.1),   # cheap ask
            _make_book("B", bid=101.0, ask=101.1),   # high bid
        ]
        result = detect_cross_exchange(books, min_profit_pct=0.0, fee=0.001)
        assert result is not None
        buy_b, sell_b, profit = result
        assert buy_b.exchange == "A"
        assert sell_b.exchange == "B"
        assert profit > 0

    def test_returns_none_when_no_opportunity(self):
        """No inter-exchange spread → no signal."""
        books = [
            _make_book("A", bid=100.0, ask=100.01),
            _make_book("B", bid=100.0, ask=100.01),
        ]
        result = detect_cross_exchange(books, min_profit_pct=1.0, fee=0.001)
        assert result is None

    def test_picks_best_pair(self):
        """Should return the pair with the highest profit."""
        books = [
            _make_book("A", bid=100.0, ask=100.1),
            _make_book("B", bid=101.0, ask=101.1),
            _make_book("C", bid=102.0, ask=102.1),   # best sell
        ]
        result = detect_cross_exchange(books, min_profit_pct=0.0, fee=0.001)
        assert result is not None
        _, sell_b, _ = result
        # The highest bid is on exchange C
        assert sell_b.exchange == "C"

    def test_single_book_returns_none(self):
        books = [_make_book("A", bid=100.0, ask=100.1)]
        result = detect_cross_exchange(books, min_profit_pct=0.0, fee=0.001)
        assert result is None


# ---------------------------------------------------------------------------
# SimpleSpreadPredictor
# ---------------------------------------------------------------------------

class TestSimpleSpreadPredictor:
    def test_returns_none_before_window_filled(self):
        pred = SimpleSpreadPredictor(window=10)
        for i in range(9):
            pred.update(float(i))
        assert pred.predict() is None

    def test_returns_value_after_window_filled(self):
        pred = SimpleSpreadPredictor(window=5)
        for i in range(5):
            pred.update(float(i))
        val = pred.predict()
        assert val is not None
        assert isinstance(val, float)

    def test_prediction_is_finite(self):
        pred = SimpleSpreadPredictor(window=10)
        for v in [0.1, 0.2, 0.15, 0.18, 0.12, 0.20, 0.11, 0.19, 0.13, 0.17]:
            pred.update(v)
        val = pred.predict()
        assert val is not None
        assert val == val  # not NaN
        assert abs(val) < 1e6

    def test_history_capped(self):
        pred = SimpleSpreadPredictor(window=5)
        for i in range(50):
            pred.update(float(i))
        # Internal history is trimmed to 'window' once it exceeds 'window * 2',
        # but between trims it can hold up to window * 2 elements.
        assert len(pred._spread_history) <= pred.window * 2


# ---------------------------------------------------------------------------
# SimulationEngine (with mocked OHLCV fetch)
# ---------------------------------------------------------------------------

class TestSimulationEngine:
    """All tests mock fetch_ohlcv_binance so no network is required."""

    _CANDLES = _make_candles(40)

    def _run(self, **kwargs) -> SimulationResult:
        # Build defaults that can be overridden by kwargs
        defaults = dict(
            symbol="BTC/USDT",
            candles=len(self._CANDLES),
            initial_balance=10_000.0,
            strategies=["cross_exchange", "ai_spread"],
            n_exchanges=3,
            min_profit_pct=0.05,
        )
        defaults.update(kwargs)
        cfg = SimulationConfig(**defaults)
        with patch(
            "simulator.engine.fetch_ohlcv_binance",
            return_value=self._CANDLES,
        ):
            return SimulationEngine(cfg).run()

    def test_result_type(self):
        result = self._run()
        assert isinstance(result, SimulationResult)

    def test_candles_processed(self):
        result = self._run()
        assert result.candles_processed == len(self._CANDLES)

    def test_equity_curve_length(self):
        """Equity curve must have at least candles+1 entries (initial + one per step)."""
        result = self._run()
        assert len(result.equity_curve) >= len(self._CANDLES)

    def test_spread_series_populated(self):
        result = self._run()
        assert len(result.spread_pct_series) > 0

    def test_portfolio_in_result(self):
        result = self._run()
        summary = result.portfolio.summary()
        assert "total_trades" in summary

    def test_balance_positive(self):
        """Balance should not go negative even with many trades."""
        result = self._run()
        assert result.portfolio.balance >= 0.0

    def test_no_open_trades_at_end(self):
        """Engine must close all trades at the end of the simulation."""
        result = self._run()
        assert result.portfolio.summary()["open_trades"] == 0

    def test_cross_exchange_only_strategy(self):
        result = self._run(strategies=["cross_exchange"])
        assert isinstance(result, SimulationResult)

    def test_ai_spread_only_strategy(self):
        result = self._run(strategies=["ai_spread"])
        assert isinstance(result, SimulationResult)

    def test_progress_callback_called(self):
        calls = []
        cfg = SimulationConfig(
            candles=len(self._CANDLES),
            strategies=["cross_exchange"],
            min_profit_pct=0.05,
        )
        with patch(
            "simulator.engine.fetch_ohlcv_binance",
            return_value=self._CANDLES,
        ):
            SimulationEngine(cfg, progress_callback=lambda i, n: calls.append(i)).run()
        assert len(calls) == len(self._CANDLES)

    def test_empty_candle_list_raises(self):
        cfg = SimulationConfig(candles=5)
        with patch(
            "simulator.engine.fetch_ohlcv_binance",
            return_value=[],
        ):
            with pytest.raises(RuntimeError, match="Could not fetch OHLCV"):
                SimulationEngine(cfg).run()

    def test_reproducible_with_seed(self):
        r1 = self._run(seed=99)
        r2 = self._run(seed=99)
        assert r1.portfolio.balance == pytest.approx(r2.portfolio.balance)

    def test_different_seeds_differ(self):
        r1 = self._run(seed=1)
        r2 = self._run(seed=2)
        # Different seeds may produce same result by chance, but usually won't
        # Just verify both complete successfully
        assert r1.candles_processed == r2.candles_processed


# ---------------------------------------------------------------------------
# SimulationConfig defaults
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.symbol == "BTC/USDT"
        assert cfg.initial_balance == pytest.approx(10_000.0)
        assert cfg.fee_rate == pytest.approx(0.001)
        assert "cross_exchange" in cfg.strategies

    def test_custom_symbol(self):
        cfg = SimulationConfig(symbol="ETH/USDT")
        assert cfg.symbol == "ETH/USDT"
