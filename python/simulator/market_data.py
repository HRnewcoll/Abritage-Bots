"""
market_data.py — Real price data fetcher + multi-exchange spread synthesiser
=============================================================================

Fetches genuine historical OHLCV candles from Binance's **public** REST API
(no API key required).  Then synthesises N virtual exchange price series by
adding small, realistic spread noise to the real prices — giving the
simulation realistic absolute price levels while still producing the kind of
micro-spread discrepancies that arbitrage bots exploit.

The synthetic spread model
--------------------------
For each simulated exchange i the mid-price at time t is:

    price_i(t) = real_price(t) * (1 + ε_i(t))

where ε_i(t) is drawn from a correlated AR(1) process so that spreads drift
slowly (as they do in real markets) rather than being pure white noise.

Public data sources (no API key required)
------------------------------------------
  - Binance  : https://api.binance.com
  - Bybit    : https://api.bybit.com (fallback)
  - CoinGecko: https://api.coingecko.com (fallback, slower)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """One OHLCV bar."""
    timestamp: int   # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SimOrderBook:
    """Thin synthetic order-book snapshot produced by the spread model."""
    exchange: str
    symbol: str
    timestamp: int
    bid: float
    ask: float
    bid_qty: float = 1.0
    ask_qty: float = 1.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid * 100.0


# ---------------------------------------------------------------------------
# Public OHLCV fetcher — no API keys required
# ---------------------------------------------------------------------------

_BINANCE_BASE = "https://api.binance.com/api/v3"
_BYBIT_BASE   = "https://api.bybit.com/v5/market"


def _binance_symbol(symbol: str) -> str:
    """BTC/USDT → BTCUSDT"""
    return symbol.replace("/", "")


def fetch_ohlcv_binance(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500,
) -> List[Candle]:
    """
    Fetch OHLCV candles from Binance's public endpoint.
    No API key required.

    Args:
        symbol:    e.g. "BTC/USDT"
        timeframe: e.g. "1m", "5m", "1h", "1d"
        limit:     number of bars (max 1000)

    Returns:
        List of Candle objects, oldest first.
    """
    sym = _binance_symbol(symbol)
    url = f"{_BINANCE_BASE}/klines"
    params = {"symbol": sym, "interval": timeframe, "limit": min(limit, 1000)}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        candles = [
            Candle(
                timestamp=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in raw
        ]
        logger.info("Fetched %d %s candles for %s from Binance public API.", len(candles), timeframe, symbol)
        return candles
    except Exception as exc:
        logger.warning("Binance OHLCV fetch failed (%s) — trying Bybit.", exc)
        return _fetch_ohlcv_bybit(symbol, timeframe, limit)


def _fetch_ohlcv_bybit(
    symbol: str,
    timeframe: str = "60",
    limit: int = 500,
) -> List[Candle]:
    """Fallback: fetch from Bybit public endpoint."""
    _tf_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    tf = _tf_map.get(timeframe, timeframe)
    sym = symbol.replace("/", "")
    url = f"{_BYBIT_BASE}/kline"
    params = {"category": "spot", "symbol": sym, "interval": tf, "limit": min(limit, 1000)}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("result", {}).get("list", [])
        candles = sorted(
            [
                Candle(
                    timestamp=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
                for row in data
            ],
            key=lambda c: c.timestamp,
        )
        logger.info("Fetched %d candles for %s from Bybit public API.", len(candles), symbol)
        return candles
    except Exception as exc:
        logger.error("Bybit OHLCV fetch also failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Multi-exchange spread synthesiser
# ---------------------------------------------------------------------------

class MultiExchangeSimulator:
    """
    Given one real price series, produces N synthetic exchange price series
    with realistic correlated spread noise.

    Each synthetic exchange has:
    - A slowly drifting mid-price offset (AR(1) process)
    - A half-spread that determines bid/ask around the mid-price
    - Occasional liquidity gaps (wider spreads) simulating real conditions

    Parameters
    ----------
    n_exchanges:   Number of virtual exchanges to simulate (default 3).
    spread_bps:    Typical half-spread in basis points (default 5 bps = 0.05%).
    drift_vol:     Volatility of the inter-exchange drift process (default 0.001).
    drift_mean_rev: Mean-reversion speed of the drift (default 0.05).
    seed:          Random seed for reproducibility.
    """

    def __init__(
        self,
        n_exchanges: int = 3,
        spread_bps: float = 5.0,
        drift_vol: float = 0.0008,
        drift_mean_rev: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_exchanges = n_exchanges
        self.half_spread = spread_bps / 10_000.0
        self.drift_vol = drift_vol
        self.drift_mean_rev = drift_mean_rev
        self.exchange_names = [f"exchange_{chr(65 + i)}" for i in range(n_exchanges)]
        rng = np.random.default_rng(seed)
        # Initial drifts (small random offsets around 0)
        self._drifts = rng.normal(0, drift_vol * 2, n_exchanges)
        self._rng = rng

    def step(self, candle: Candle) -> List[SimOrderBook]:
        """
        Advance one candle and return a list of SimOrderBook objects,
        one per simulated exchange.
        """
        price = candle.close
        books: List[SimOrderBook] = []

        for i in range(self.n_exchanges):
            # AR(1) drift update
            noise = self._rng.normal(0, self.drift_vol)
            self._drifts[i] = (
                self._drifts[i] * (1.0 - self.drift_mean_rev) + noise
            )
            # Clamp drift to ±1% so prices stay realistic
            self._drifts[i] = max(-0.01, min(0.01, self._drifts[i]))

            mid = price * (1.0 + self._drifts[i])

            # Occasionally widen the spread (liquidity gap)
            spread_mult = 1.0
            if self._rng.random() < 0.05:  # 5% chance of wide spread
                spread_mult = self._rng.uniform(2.0, 5.0)

            half = mid * self.half_spread * spread_mult
            bid = mid - half
            ask = mid + half

            # Simulate varying queue sizes
            bid_qty = float(self._rng.uniform(0.1, 5.0))
            ask_qty = float(self._rng.uniform(0.1, 5.0))

            books.append(
                SimOrderBook(
                    exchange=self.exchange_names[i],
                    symbol="",  # filled in by caller via simulate_all() / engine
                    timestamp=candle.timestamp,
                    bid=round(bid, 8),
                    ask=round(ask, 8),
                    bid_qty=round(bid_qty, 4),
                    ask_qty=round(ask_qty, 4),
                )
            )

        return books

    def simulate_all(
        self, candles: List[Candle], symbol: str
    ) -> List[Tuple[Candle, List[SimOrderBook]]]:
        """
        Process an entire candle list and return (candle, [order_books]) pairs.
        """
        result = []
        for candle in candles:
            books = self.step(candle)
            # Fix the symbol placeholder
            for b in books:
                b.symbol = symbol
            result.append((candle, books))
        return result
