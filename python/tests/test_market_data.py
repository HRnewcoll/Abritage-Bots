"""
Tests for simulator.market_data — MultiExchangeSimulator + helpers
===================================================================
No network calls — all tests use synthetic data.
"""

from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.market_data import (
    Candle,
    MultiExchangeSimulator,
    SimOrderBook,
    _binance_symbol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(price: float = 40_000.0, ts: int = 1_700_000_000_000) -> Candle:
    return Candle(
        timestamp=ts,
        open=price,
        high=price * 1.001,
        low=price * 0.999,
        close=price,
        volume=10.0,
    )


# ---------------------------------------------------------------------------
# _binance_symbol
# ---------------------------------------------------------------------------

class TestBinanceSymbol:
    def test_standard(self):
        assert _binance_symbol("BTC/USDT") == "BTCUSDT"

    def test_no_slash(self):
        assert _binance_symbol("ETHUSDT") == "ETHUSDT"

    def test_multiple_slashes(self):
        assert _binance_symbol("A/B/C") == "ABC"


# ---------------------------------------------------------------------------
# Candle
# ---------------------------------------------------------------------------

class TestCandle:
    def test_fields(self):
        c = _make_candle(50_000.0, 1_000)
        assert c.close == 50_000.0
        assert c.timestamp == 1_000

    def test_high_gte_low(self):
        c = _make_candle(40_000.0)
        assert c.high >= c.low


# ---------------------------------------------------------------------------
# SimOrderBook
# ---------------------------------------------------------------------------

class TestSimOrderBook:
    def _book(self, bid: float = 100.0, ask: float = 101.0) -> SimOrderBook:
        return SimOrderBook(
            exchange="test",
            symbol="BTC/USDT",
            timestamp=1_000,
            bid=bid,
            ask=ask,
        )

    def test_mid(self):
        b = self._book(100.0, 102.0)
        assert b.mid == pytest.approx(101.0)

    def test_spread_pct_positive(self):
        b = self._book(100.0, 101.0)
        assert b.spread_pct > 0.0

    def test_spread_pct_symmetric(self):
        """ask > bid → spread_pct > 0"""
        b = self._book(99.0, 101.0)
        assert b.spread_pct == pytest.approx(2.0 / 100.0 * 100.0)


# ---------------------------------------------------------------------------
# MultiExchangeSimulator
# ---------------------------------------------------------------------------

class TestMultiExchangeSimulator:
    def test_returns_correct_number_of_books(self):
        sim = MultiExchangeSimulator(n_exchanges=4, seed=42)
        books = sim.step(_make_candle())
        assert len(books) == 4

    def test_default_three_exchanges(self):
        sim = MultiExchangeSimulator(seed=99)
        books = sim.step(_make_candle())
        assert len(books) == 3

    def test_exchange_names_unique(self):
        sim = MultiExchangeSimulator(n_exchanges=5, seed=1)
        books = sim.step(_make_candle())
        names = [b.exchange for b in books]
        assert len(set(names)) == 5

    def test_bid_below_ask(self):
        sim = MultiExchangeSimulator(seed=0)
        for _ in range(50):
            books = sim.step(_make_candle())
            for b in books:
                assert b.bid < b.ask, f"bid={b.bid} >= ask={b.ask}"

    def test_prices_near_candle_close(self):
        """All exchange mid-prices must stay within ±1% of real close."""
        sim = MultiExchangeSimulator(n_exchanges=3, seed=7)
        price = 40_000.0
        candle = _make_candle(price)
        for _ in range(100):
            books = sim.step(candle)
            for b in books:
                assert abs(b.mid - price) / price <= 0.01, (
                    f"mid {b.mid:.2f} drifted more than 1% from {price}"
                )

    def test_reproducible_with_seed(self):
        candle = _make_candle()
        sim1 = MultiExchangeSimulator(seed=42)
        sim2 = MultiExchangeSimulator(seed=42)
        for _ in range(10):
            books1 = sim1.step(candle)
            books2 = sim2.step(candle)
        bids1 = [b.bid for b in books1]
        bids2 = [b.bid for b in books2]
        assert bids1 == pytest.approx(bids2)

    def test_different_seeds_differ(self):
        candle = _make_candle()
        sim1 = MultiExchangeSimulator(seed=1)
        sim2 = MultiExchangeSimulator(seed=2)
        for _ in range(5):
            books1 = sim1.step(candle)
            books2 = sim2.step(candle)
        assert books1[0].bid != pytest.approx(books2[0].bid)

    def test_simulate_all_length(self):
        sim = MultiExchangeSimulator(seed=0)
        candles = [_make_candle(40_000.0 + i, ts=i * 3600) for i in range(20)]
        results = sim.simulate_all(candles, "BTC/USDT")
        assert len(results) == 20

    def test_simulate_all_symbol_set(self):
        sim = MultiExchangeSimulator(seed=0)
        candles = [_make_candle() for _ in range(5)]
        results = sim.simulate_all(candles, "ETH/USDT")
        for _, books in results:
            for b in books:
                assert b.symbol == "ETH/USDT"

    def test_spread_bps_scales_spread(self):
        """Wider spread_bps → wider typical bid/ask gap."""
        candle = _make_candle(10_000.0)
        tight = MultiExchangeSimulator(n_exchanges=1, spread_bps=1.0,  seed=5)
        wide  = MultiExchangeSimulator(n_exchanges=1, spread_bps=50.0, seed=5)

        tight_spreads, wide_spreads = [], []
        for _ in range(100):
            t = tight.step(candle)[0].spread_pct
            w = wide.step(candle)[0].spread_pct
            tight_spreads.append(t)
            wide_spreads.append(w)

        assert sum(wide_spreads) > sum(tight_spreads)

    def test_qty_fields_positive(self):
        sim = MultiExchangeSimulator(seed=3)
        books = sim.step(_make_candle())
        for b in books:
            assert b.bid_qty > 0
            assert b.ask_qty > 0
