"""
portfolio.py — Paper-trading portfolio for simulation
======================================================

Tracks a virtual account with USDT balance and open/closed positions.
All values are in USDT.  No real orders are ever placed.

Usage
-----
    from simulator.portfolio import PaperPortfolio

    port = PaperPortfolio(initial_balance=10_000.0)
    trade_id = port.open_trade("BTC/USDT", "cross_exchange",
                                buy_exchange="exchange_A",
                                sell_exchange="exchange_B",
                                buy_price=30000.0, sell_price=30150.0,
                                size_usdt=200.0, fee_rate=0.001)
    port.close_trade(trade_id, realised_pnl=2.50)
    print(port.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Record of a single completed or open round-trip arbitrage trade."""
    trade_id: int
    symbol: str
    strategy: str
    open_ts: int          # Unix ms
    close_ts: Optional[int] = None

    buy_exchange: str = ""
    sell_exchange: str = ""
    buy_price: float = 0.0
    sell_price: float = 0.0
    size_usdt: float = 0.0
    fee_rate: float = 0.001

    # Realised values (filled when trade closes)
    gross_pnl: float = 0.0     # sell - buy in USDT, before fees
    fee_cost: float = 0.0      # total fee in USDT
    net_pnl: float = 0.0       # gross - fees

    is_open: bool = True

    @property
    def profit_pct(self) -> float:
        if self.size_usdt == 0:
            return 0.0
        return self.net_pnl / self.size_usdt * 100.0

    @property
    def duration_s(self) -> float:
        if self.close_ts is None:
            return 0.0
        return (self.close_ts - self.open_ts) / 1000.0


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class PaperPortfolio:
    """
    Simulated trading account.

    Parameters
    ----------
    initial_balance: Starting USDT balance (default: 10 000 USDT).
    """

    def __init__(self, initial_balance: float = 10_000.0) -> None:
        self.initial_balance: float = initial_balance
        self.balance: float = initial_balance
        self._trades: List[Trade] = []
        self._open_trades: Dict[int, Trade] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def open_trade(
        self,
        symbol: str,
        strategy: str,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: float,
        sell_price: float,
        size_usdt: float,
        fee_rate: float = 0.001,
        timestamp: Optional[int] = None,
    ) -> int:
        """
        Open a paper trade.  Deducts size_usdt from the balance.

        Returns the trade ID for later reference.
        """
        if size_usdt > self.balance:
            size_usdt = self.balance  # never over-commit
        if size_usdt <= 0:
            return -1

        self.balance -= size_usdt

        trade = Trade(
            trade_id=self._next_id,
            symbol=symbol,
            strategy=strategy,
            open_ts=timestamp or int(time.time() * 1000),
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            size_usdt=size_usdt,
            fee_rate=fee_rate,
        )
        self._open_trades[self._next_id] = trade
        self._trades.append(trade)
        self._next_id += 1
        return trade.trade_id

    def close_trade(
        self,
        trade_id: int,
        realised_pnl: float,
        timestamp: Optional[int] = None,
    ) -> Optional[Trade]:
        """
        Close an open trade and credit the result to the balance.

        realised_pnl is the gross PnL (before fees).  Fees are calculated
        automatically from fee_rate * size_usdt * 2 (buy + sell legs).
        """
        trade = self._open_trades.pop(trade_id, None)
        if trade is None:
            return None

        fee_cost = trade.size_usdt * trade.fee_rate * 2.0
        net_pnl = realised_pnl - fee_cost

        trade.gross_pnl = realised_pnl
        trade.fee_cost = fee_cost
        trade.net_pnl = net_pnl
        trade.is_open = False
        trade.close_ts = timestamp or int(time.time() * 1000)

        # Return capital + net profit to balance
        self.balance += trade.size_usdt + net_pnl
        return trade

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    @property
    def closed_trades(self) -> List[Trade]:
        return [t for t in self._trades if not t.is_open]

    @property
    def total_pnl(self) -> float:
        return sum(t.net_pnl for t in self.closed_trades)

    @property
    def total_return_pct(self) -> float:
        return (self.balance - self.initial_balance) / self.initial_balance * 100.0

    @property
    def win_rate(self) -> float:
        closed = self.closed_trades
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.net_pnl > 0)
        return wins / len(closed) * 100.0

    @property
    def avg_profit_per_trade(self) -> float:
        closed = self.closed_trades
        if not closed:
            return 0.0
        return sum(t.net_pnl for t in closed) / len(closed)

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum peak-to-trough drawdown over the simulation."""
        if not self._trades:
            return 0.0
        equity_curve = self._equity_curve()
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for v in equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _equity_curve(self) -> List[float]:
        """Reconstruct equity curve from closed trades."""
        curve = [self.initial_balance]
        running = self.initial_balance
        for t in sorted(self.closed_trades, key=lambda x: x.close_ts or 0):
            running += t.net_pnl
            curve.append(running)
        return curve

    def sharpe_ratio(self, risk_free_rate: float = 0.0, trades_per_year: float = 0.0) -> float:
        """
        Annualised Sharpe ratio based on per-trade returns.

        Args:
            risk_free_rate: Per-trade risk-free rate (default 0).
            trades_per_year: Annualisation factor.  If 0 (default), it is
                estimated from the actual trade timestamps when available,
                falling back to 1460 (≈4 trades/day) as a conservative guess.
        """
        closed = self.closed_trades
        if len(closed) < 2:
            return 0.0
        returns = [t.net_pnl / t.size_usdt for t in closed if t.size_usdt > 0]
        if not returns:
            return 0.0
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = variance ** 0.5
        if std_r == 0:
            return 0.0
        # Estimate annualisation factor from trade timestamps when available
        if trades_per_year <= 0:
            ts_list = [t.open_ts for t in closed if t.open_ts]
            if len(ts_list) >= 2:
                elapsed_ms = max(ts_list) - min(ts_list)
                elapsed_years = elapsed_ms / (1000 * 60 * 60 * 24 * 365.25)
                if elapsed_years > 0:
                    trades_per_year = len(closed) / elapsed_years
                else:
                    trades_per_year = 1460.0
            else:
                trades_per_year = 1460.0
        return (mean_r - risk_free_rate) / std_r * (trades_per_year ** 0.5)

    def summary(self) -> dict:
        """Return a dictionary with key portfolio statistics."""
        return {
            "initial_balance_usdt": round(self.initial_balance, 2),
            "final_balance_usdt": round(self.balance, 2),
            "total_pnl_usdt": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "total_trades": len(self.closed_trades),
            "win_rate_pct": round(self.win_rate, 2),
            "avg_profit_per_trade_usdt": round(self.avg_profit_per_trade, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio(), 4),
            "open_trades": len(self._open_trades),
        }
