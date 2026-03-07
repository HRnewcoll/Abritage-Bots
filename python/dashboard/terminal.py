"""
terminal.py — Live Rich terminal dashboard for real bot runs
=============================================================

Displays a continuously-updating dashboard in the terminal while any of the
arbitrage bots is running.  Shows:
  - Live clock and bot status
  - Per-exchange last price for each watched symbol
  - Detected opportunities (with profit %)
  - Paper / live portfolio balance
  - Last 10 trades

Usage
-----
    # Import and use in a bot:
    from dashboard.terminal import LiveDashboard

    dash = LiveDashboard(symbols=["BTC/USDT", "ETH/USDT"])
    dash.start()

    # In your bot loop:
    dash.update_price("binance", "BTC/USDT", 65000.0)
    dash.update_price("kraken",  "BTC/USDT", 65100.0)
    dash.add_opportunity("BTC/USDT", "binance", "kraken", 65000, 65100, 0.12)
    dash.update_balance(10_523.41)
    dash.add_trade("BTC/USDT", "binance", "kraken", 65000, 65100, 1.53)

    dash.stop()
"""

from __future__ import annotations

import datetime
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_RICH_AVAILABLE = False
try:
    from rich import box
    from rich.align import Align
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PriceSnapshot:
    exchange: str
    symbol: str
    price: float
    updated_at: float = field(default_factory=time.time)


@dataclass
class OpportunityRecord:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_pct: float
    detected_at: float = field(default_factory=time.time)


@dataclass
class TradeRecord:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    net_pnl: float
    executed_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class LiveDashboard:
    """
    Rich live terminal dashboard for arbitrage bot monitoring.

    Safe to update from multiple threads (uses a lock).
    Falls back to plain-text logging if `rich` is not installed.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        bot_name: str = "Arbitrage Bot",
        refresh_rate: float = 1.0,
    ) -> None:
        self.symbols = symbols or ["BTC/USDT"]
        self.bot_name = bot_name
        self.refresh_rate = refresh_rate
        self._lock = threading.Lock()

        # State
        self._prices: Dict[Tuple[str, str], PriceSnapshot] = {}
        self._opps: deque = deque(maxlen=50)
        self._trades: deque = deque(maxlen=50)
        self._balance: float = 0.0
        self._start_balance: float = 0.0
        self._status: str = "Starting…"
        self._running = False

        self._live: Optional["Live"] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def set_initial_balance(self, balance: float) -> None:
        with self._lock:
            self._balance = balance
            self._start_balance = balance

    def update_balance(self, balance: float) -> None:
        with self._lock:
            self._balance = balance

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def update_price(self, exchange: str, symbol: str, price: float) -> None:
        with self._lock:
            self._prices[(exchange, symbol)] = PriceSnapshot(exchange, symbol, price)

    def add_opportunity(
        self,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: float,
        sell_price: float,
        profit_pct: float,
    ) -> None:
        with self._lock:
            self._opps.appendleft(
                OpportunityRecord(symbol, buy_exchange, sell_exchange,
                                  buy_price, sell_price, profit_pct)
            )

    def add_trade(
        self,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: float,
        sell_price: float,
        net_pnl: float,
    ) -> None:
        with self._lock:
            self._trades.appendleft(
                TradeRecord(symbol, buy_exchange, sell_exchange,
                            buy_price, sell_price, net_pnl)
            )

    # ------------------------------------------------------------------
    # Rich rendering
    # ------------------------------------------------------------------

    def _build_layout(self) -> "Panel":
        if not _RICH_AVAILABLE:
            raise RuntimeError("rich not installed")

        with self._lock:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
            pnl     = self._balance - self._start_balance
            pnl_col = "green" if pnl >= 0 else "red"

            # ── Header ──────────────────────────────────────────────────
            header = Text.assemble(
                ("⚡  ", "yellow"),
                (self.bot_name, "bold cyan"),
                ("  │  ", "dim"),
                (now_str, "dim"),
                ("  │  Status: ", "dim"),
                (self._status, "bold green"),
            )

            # ── Prices table ─────────────────────────────────────────────
            price_table = Table(box=box.SIMPLE, show_header=True, header_style="bold blue",
                                title="[bold]Live Prices[/]", title_style="cyan")
            price_table.add_column("Exchange",  width=14)
            price_table.add_column("Symbol",    width=12)
            price_table.add_column("Price",     justify="right", width=14)
            price_table.add_column("Age (s)",   justify="right", width=9)
            now_ts = time.time()
            for (exch, sym), snap in list(self._prices.items()):
                age  = now_ts - snap.updated_at
                col  = "green" if age < 5 else ("yellow" if age < 15 else "red")
                price_table.add_row(exch, sym, f"${snap.price:,.2f}", f"[{col}]{age:.0f}s[/]")

            # ── Opportunities table ───────────────────────────────────────
            opp_table = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow",
                              title="[bold]Recent Opportunities[/]", title_style="yellow")
            opp_table.add_column("Symbol",   width=10)
            opp_table.add_column("Buy",      width=14)
            opp_table.add_column("Sell",     width=14)
            opp_table.add_column("Profit %", justify="right", width=10)
            opp_table.add_column("Age (s)",  justify="right", width=9)
            for opp in list(self._opps)[:8]:
                age = now_ts - opp.detected_at
                opp_table.add_row(
                    opp.symbol,
                    f"{opp.buy_exchange} ${opp.buy_price:,.2f}",
                    f"{opp.sell_exchange} ${opp.sell_price:,.2f}",
                    f"[green]{opp.profit_pct:+.4f}%[/]",
                    f"{age:.0f}s",
                )

            # ── Trade history ─────────────────────────────────────────────
            trade_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta",
                                title="[bold]Recent Trades[/]", title_style="magenta")
            trade_table.add_column("Symbol",  width=10)
            trade_table.add_column("Buy",     width=14)
            trade_table.add_column("Sell",    width=14)
            trade_table.add_column("Net P&L", justify="right", width=12)
            for t in list(self._trades)[:8]:
                c = "green" if t.net_pnl > 0 else "red"
                trade_table.add_row(
                    t.symbol,
                    f"{t.buy_exchange} ${t.buy_price:,.2f}",
                    f"{t.sell_exchange} ${t.sell_price:,.2f}",
                    f"[{c}]${t.net_pnl:+.4f}[/]",
                )

            # ── Balance panel ─────────────────────────────────────────────
            bal_text = Text.assemble(
                ("  Balance: ", "dim"),
                (f"${self._balance:,.2f}", "bold white"),
                ("  │  P&L: ", "dim"),
                (f"${pnl:+,.2f}", f"bold {pnl_col}"),
                ("  │  Return: ", "dim"),
                (
                    f"{(pnl / self._start_balance * 100):+.3f}%" if self._start_balance else "0%",
                    f"bold {pnl_col}",
                ),
            )

            from rich.columns import Columns

        # Compose final panel outside the lock
        from rich.console import Group
        group = Group(
            Panel(bal_text, border_style="dim"),
            Columns([price_table, opp_table, trade_table], equal=False, expand=True),
        )
        return Panel(group, title=Align.center(header), border_style="cyan")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the live dashboard in a background thread."""
        if not _RICH_AVAILABLE:
            import logging
            logging.getLogger("dashboard").warning(
                "rich not installed — dashboard disabled. "
                "Install with: pip install rich"
            )
            return

        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._live:
            self._live.stop()

    def _render_loop(self) -> None:
        console = Console()
        with Live(console=console, refresh_per_second=int(1 / self.refresh_rate),
                  screen=True) as live:
            self._live = live
            while self._running:
                try:
                    live.update(self._build_layout())
                except Exception:
                    pass
                time.sleep(self.refresh_rate)
