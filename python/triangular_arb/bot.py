"""
Triangular Arbitrage Bot
========================
Detects and (optionally) executes triangular arbitrage opportunities on a
single exchange.

A triangular cycle converts:
    base  ──► coin_a  ──► coin_b  ──► base

If the product of the three exchange rates (minus fees) is > 1, a profit
exists and the bot executes the three trades in rapid succession.

Usage
-----
    python -m triangular_arb.bot [--config path/to/config.yaml]

Always start with  dry_run: true  in config until you have verified the bot
is operating as expected on your account.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import ccxt
import colorlog

sys.path.insert(0, "..")
from utils.exchange import build_exchange, load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_handler = colorlog.StreamHandler()
_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logging.basicConfig(handlers=[_handler], level=logging.INFO)
logger = logging.getLogger("triangular_arb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_rate(exchange: ccxt.Exchange, base: str, quote: str) -> Optional[float]:
    """
    Return the mid-price rate for converting `base` → `quote`.

    Tries both base/quote and quote/base market directions.
    Returns None if neither market exists.
    """
    direct = f"{base}/{quote}"
    inverse = f"{quote}/{base}"

    try:
        if direct in exchange.markets:
            ticker = exchange.fetch_ticker(direct)
            mid = (ticker["bid"] + ticker["ask"]) / 2.0
            return mid  # 1 base = mid quote

        if inverse in exchange.markets:
            ticker = exchange.fetch_ticker(inverse)
            mid = (ticker["bid"] + ticker["ask"]) / 2.0
            return 1.0 / mid  # 1 base = (1/mid) quote
    except Exception as exc:
        logger.debug("Could not fetch rate %s/%s: %s", base, quote, exc)

    return None


def _get_taker_fee(exchange: ccxt.Exchange, symbol: str) -> float:
    try:
        return float(exchange.market(symbol).get("taker", 0.001))
    except Exception:
        return 0.001


# ---------------------------------------------------------------------------
# Triangular opportunity detection
# ---------------------------------------------------------------------------

def find_triangles(exchange: ccxt.Exchange, base: str) -> List[Tuple]:
    """
    Enumerate all unique 3-currency cycles involving `base` and return those
    where the net profit (after fees) is positive.

    Returns a list of tuples:
        (coin_a, coin_b, profit_pct, rate_1, rate_2, rate_3)
    sorted by profit_pct descending.
    """
    # Collect all currencies that trade against `base`
    base_pairs = [
        sym for sym in exchange.markets
        if sym.endswith(f"/{base}") or sym.startswith(f"{base}/")
    ]
    coins = set()
    for sym in base_pairs:
        a, b = sym.split("/")
        coins.add(a if b == base else b)
    coins.discard(base)

    opportunities = []
    fee = 0.001  # approximate taker fee for all legs

    for coin_a, coin_b in permutations(coins, 2):
        # Path: base → coin_a → coin_b → base
        r1 = _get_rate(exchange, base, coin_a)    # base   → coin_a
        r2 = _get_rate(exchange, coin_a, coin_b)  # coin_a → coin_b
        r3 = _get_rate(exchange, coin_b, base)    # coin_b → base

        if r1 is None or r2 is None or r3 is None:
            continue

        # Apply fee on each leg
        net = r1 * r2 * r3 * ((1 - fee) ** 3)
        profit_pct = (net - 1.0) * 100.0

        if profit_pct > 0:
            opportunities.append((coin_a, coin_b, profit_pct, r1, r2, r3))

    opportunities.sort(key=lambda x: x[2], reverse=True)
    return opportunities


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class TriangularArbitrageBot:
    """
    Runs a continuous scan loop on a single exchange, looking for profitable
    triangular cycles and optionally executing them.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.tri_cfg = config.get("triangular", {})
        self.dry_run: bool = self.tri_cfg.get("dry_run", True)
        self.min_profit_pct: float = self.tri_cfg.get("min_profit_pct", 0.3)
        self.base_currency: str = self.tri_cfg.get("base_currency", "USDT")
        self.poll_interval: int = int(self.cfg.get("arbitrage", {}).get("poll_interval_seconds", 5))

        exchange_name: str = self.tri_cfg.get("exchange", "binance")
        self.exchange = build_exchange(exchange_name, config)
        logger.info("Connected to %s for triangular arbitrage.", exchange_name)

    # ------------------------------------------------------------------
    def _execute_triangle(
        self,
        base: str,
        coin_a: str,
        coin_b: str,
        start_amount: float,
    ) -> None:
        """
        Execute the three legs of a triangular trade.
        start_amount is denominated in `base` currency.
        """
        ex = self.exchange

        def _market_buy(symbol: str, quote_amount: float) -> Optional[float]:
            """Buy as much `base_of_symbol` as possible with quote_amount."""
            try:
                ticker = ex.fetch_ticker(symbol)
                price = ticker["ask"]
                amount = quote_amount / price
                order = ex.create_market_buy_order(symbol, amount)
                logger.info("Leg BUY  %s – order id: %s", symbol, order.get("id"))
                return amount * (1 - _get_taker_fee(ex, symbol))
            except Exception as exc:
                logger.error("Leg BUY  %s FAILED: %s", symbol, exc)
                return None

        def _market_sell(symbol: str, amount: float) -> Optional[float]:
            """Sell `amount` of base currency of `symbol`."""
            try:
                ticker = ex.fetch_ticker(symbol)
                price = ticker["bid"]
                order = ex.create_market_sell_order(symbol, amount)
                logger.info("Leg SELL %s – order id: %s", symbol, order.get("id"))
                return amount * price * (1 - _get_taker_fee(ex, symbol))
            except Exception as exc:
                logger.error("Leg SELL %s FAILED: %s", symbol, exc)
                return None

        # Leg 1: base → coin_a
        sym1 = f"{coin_a}/{base}"
        holding_a = _market_buy(sym1, start_amount)
        if holding_a is None:
            return

        # Leg 2: coin_a → coin_b
        sym2 = f"{coin_b}/{coin_a}" if f"{coin_b}/{coin_a}" in ex.markets else f"{coin_a}/{coin_b}"
        if sym2 == f"{coin_a}/{coin_b}":
            # We're selling coin_a for coin_b
            holding_b = _market_sell(sym2, holding_a)
        else:
            holding_b = _market_buy(sym2, holding_a)
        if holding_b is None:
            return

        # Leg 3: coin_b → base
        sym3 = f"{coin_b}/{base}"
        final = _market_sell(sym3, holding_b)
        if final:
            profit = final - start_amount
            logger.info("Triangle complete. Started: %.4f %s | Ended: %.4f %s | Profit: %.6f %s",
                        start_amount, base, final, base, profit, base)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main event loop."""
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        logger.info("=== Triangular Arbitrage Bot started [%s] ===", mode)
        logger.info("Exchange: %s | Base: %s | Min profit: %.2f%%",
                    self.exchange.id, self.base_currency, self.min_profit_pct)

        while True:
            opps = find_triangles(self.exchange, self.base_currency)
            filtered = [o for o in opps if o[2] >= self.min_profit_pct]

            if filtered:
                for coin_a, coin_b, profit_pct, r1, r2, r3 in filtered[:3]:
                    logger.info(
                        "TRIANGLE OPPORTUNITY | %s → %s → %s → %s | "
                        "Rates: %.6f / %.6f / %.6f | Profit: %.4f%%",
                        self.base_currency, coin_a, coin_b, self.base_currency,
                        r1, r2, r3, profit_pct,
                    )
                    if not self.dry_run:
                        self._execute_triangle(
                            self.base_currency, coin_a, coin_b, start_amount=100.0
                        )
            else:
                logger.debug("No triangular opportunities above %.2f%%", self.min_profit_pct)

            time.sleep(self.poll_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Triangular Crypto Arbitrage Bot")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    bot = TriangularArbitrageBot(cfg)
    bot.run()


if __name__ == "__main__":
    main()
