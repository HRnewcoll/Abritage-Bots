"""
Cross-Exchange Arbitrage Bot
============================
Monitors the same trading pair across multiple exchanges and executes
simultaneous buy/sell orders when the price spread exceeds the configured
minimum profit threshold (after estimated trading fees).

Supported exchanges (via ccxt): Binance, Kraken, Coinbase, Bybit, KuCoin,
OKX – and virtually any other ccxt-supported exchange you add to config.yaml.

Usage
-----
    python -m cross_exchange_arb.bot [--config path/to/config.yaml]

Always start with  dry_run: true  in config until you have verified the bot
is operating as expected on your account.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import ccxt
import colorlog

sys.path.insert(0, "..")
from utils.exchange import build_exchange, fetch_order_book, load_config

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
logger = logging.getLogger("cross_exchange_arb")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def estimate_fee(exchange: ccxt.Exchange, symbol: str) -> float:
    """Return taker fee as a decimal (e.g. 0.001 for 0.1%)."""
    try:
        market = exchange.market(symbol)
        return float(market.get("taker", 0.001))
    except Exception:
        return 0.001  # conservative default


def best_ask(order_book: dict) -> Optional[float]:
    """Cheapest sell price available on an order book."""
    asks = order_book.get("asks", [])
    return float(asks[0][0]) if asks else None


def best_bid(order_book: dict) -> Optional[float]:
    """Highest buy price available on an order book."""
    bids = order_book.get("bids", [])
    return float(bids[0][0]) if bids else None


def calculate_profit_pct(
    buy_price: float,
    sell_price: float,
    buy_fee: float,
    sell_fee: float,
) -> float:
    """
    Net profit percentage accounting for taker fees on both legs.

        profit % = ((sell_price * (1 - sell_fee)) / (buy_price * (1 + buy_fee)) - 1) * 100
    """
    effective_buy = buy_price * (1.0 + buy_fee)
    effective_sell = sell_price * (1.0 - sell_fee)
    return (effective_sell / effective_buy - 1.0) * 100.0


# ---------------------------------------------------------------------------
# Arbitrage scanner
# ---------------------------------------------------------------------------

class CrossExchangeArbitrageBot:
    """
    Scans all pairs of exchanges for price discrepancies and optionally
    places matched buy/sell orders when profit exceeds the threshold.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.arb_cfg = config.get("arbitrage", {})
        self.dry_run: bool = self.arb_cfg.get("dry_run", True)
        self.min_profit_pct: float = self.arb_cfg.get("min_profit_pct", 0.5)
        self.max_trade_usdt: float = self.arb_cfg.get("max_trade_size_usdt", 100.0)
        self.poll_interval: int = int(self.arb_cfg.get("poll_interval_seconds", 5))
        self.symbols: List[str] = self.arb_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])

        # Build exchange objects for every exchange defined in config
        exchange_names = list(config.get("exchanges", {}).keys())
        if not exchange_names:
            raise ValueError("No exchanges defined in config.")

        self.exchanges: Dict[str, ccxt.Exchange] = {}
        for name in exchange_names:
            try:
                self.exchanges[name] = build_exchange(name, config)
                logger.info("Connected to exchange: %s", name)
            except Exception as exc:
                logger.warning("Could not connect to %s: %s – skipping.", name, exc)

        if len(self.exchanges) < 2:
            raise RuntimeError("Need at least 2 working exchanges for cross-exchange arbitrage.")

    # ------------------------------------------------------------------
    def _scan_symbol(self, symbol: str) -> Optional[Tuple]:
        """
        For a given symbol, fetch order books from all exchanges and find
        the best buy/sell pair.

        Returns (buy_exchange, sell_exchange, profit_pct, buy_price, sell_price)
        or None if no opportunity found.
        """
        order_books: Dict[str, dict] = {}
        for name, ex in self.exchanges.items():
            if symbol not in ex.markets:
                continue
            ob = fetch_order_book(ex, symbol)
            if ob:
                order_books[name] = ob

        if len(order_books) < 2:
            return None

        best: Optional[Tuple] = None
        for buy_name, sell_name in itertools.permutations(order_books.keys(), 2):
            ask = best_ask(order_books[buy_name])
            bid = best_bid(order_books[sell_name])
            if ask is None or bid is None:
                continue

            buy_fee = estimate_fee(self.exchanges[buy_name], symbol)
            sell_fee = estimate_fee(self.exchanges[sell_name], symbol)
            profit = calculate_profit_pct(ask, bid, buy_fee, sell_fee)

            if profit >= self.min_profit_pct:
                if best is None or profit > best[2]:
                    best = (buy_name, sell_name, profit, ask, bid)

        return best

    # ------------------------------------------------------------------
    def _execute_trade(
        self,
        symbol: str,
        buy_exchange: ccxt.Exchange,
        sell_exchange: ccxt.Exchange,
        buy_price: float,
        sell_price: float,
    ) -> None:
        """Place simultaneous market orders on both legs."""
        amount = self.max_trade_usdt / buy_price
        logger.info(
            "EXECUTING: Buy %.6f %s @ %.4f on %s | Sell @ %.4f on %s",
            amount, symbol, buy_price, buy_exchange.id, sell_price, sell_exchange.id,
        )
        try:
            buy_order = buy_exchange.create_market_buy_order(symbol, amount)
            logger.info("Buy order placed: %s", buy_order.get("id"))
        except Exception as exc:
            logger.error("Buy order FAILED on %s: %s", buy_exchange.id, exc)
            return

        try:
            sell_order = sell_exchange.create_market_sell_order(symbol, amount)
            logger.info("Sell order placed: %s", sell_order.get("id"))
        except Exception as exc:
            logger.error("Sell order FAILED on %s: %s – WARNING: position may be unhedged!", sell_exchange.id, exc)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main event loop."""
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        logger.info("=== Cross-Exchange Arbitrage Bot started [%s] ===", mode)
        logger.info("Watching %d symbol(s) across %d exchange(s)", len(self.symbols), len(self.exchanges))

        while True:
            for symbol in self.symbols:
                result = self._scan_symbol(symbol)
                if result:
                    buy_name, sell_name, profit_pct, buy_price, sell_price = result
                    logger.info(
                        "OPPORTUNITY | %s | Buy on %-10s @ %.4f | Sell on %-10s @ %.4f | Profit: %.3f%%",
                        symbol, buy_name, buy_price, sell_name, sell_price, profit_pct,
                    )
                    if not self.dry_run:
                        self._execute_trade(
                            symbol,
                            self.exchanges[buy_name],
                            self.exchanges[sell_name],
                            buy_price,
                            sell_price,
                        )
                else:
                    logger.debug("No opportunity for %s", symbol)

            time.sleep(self.poll_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-Exchange Crypto Arbitrage Bot")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    bot = CrossExchangeArbitrageBot(cfg)
    bot.run()


if __name__ == "__main__":
    main()
