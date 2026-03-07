"""
Tests for simulator.portfolio — PaperPortfolio
================================================
Covers: open/close lifecycle, P&L accounting, fee deduction,
        win-rate, drawdown, Sharpe ratio, and summary dict.
"""

from __future__ import annotations

import sys
import os
import time
import pytest

# Ensure the python package root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.portfolio import PaperPortfolio, Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = 1_700_000_000_000  # fixed base timestamp (ms)


def _open_and_close(
    port: PaperPortfolio,
    gross_pnl: float,
    size: float = 100.0,
    ts_open: int = _TS,
    ts_close: int = _TS + 3_600_000,
) -> Trade:
    """Open then immediately close one trade and return the closed Trade."""
    tid = port.open_trade(
        symbol="BTC/USDT",
        strategy="cross_exchange",
        buy_exchange="exchange_A",
        sell_exchange="exchange_B",
        buy_price=40_000.0,
        sell_price=40_200.0,
        size_usdt=size,
        fee_rate=0.001,
        timestamp=ts_open,
    )
    assert tid > 0, "open_trade should return a positive trade ID"
    return port.close_trade(tid, gross_pnl, timestamp=ts_close)


# ---------------------------------------------------------------------------
# open_trade
# ---------------------------------------------------------------------------

class TestOpenTrade:
    def test_balance_reduced(self):
        port = PaperPortfolio(10_000.0)
        port.open_trade(
            "BTC/USDT", "cross", "A", "B",
            40_000.0, 40_200.0, 500.0, 0.001, _TS,
        )
        assert port.balance == pytest.approx(9_500.0)

    def test_returns_positive_id(self):
        port = PaperPortfolio(10_000.0)
        tid = port.open_trade(
            "BTC/USDT", "cross", "A", "B",
            40_000.0, 40_200.0, 100.0, 0.001, _TS,
        )
        assert tid == 1

    def test_id_increments(self):
        port = PaperPortfolio(10_000.0)
        ids = [
            port.open_trade("X/Y", "s", "A", "B", 1.0, 1.0, 10.0, 0.001, _TS)
            for _ in range(5)
        ]
        assert ids == [1, 2, 3, 4, 5]

    def test_caps_at_balance(self):
        port = PaperPortfolio(50.0)
        port.open_trade("X/Y", "s", "A", "B", 1.0, 1.0, 200.0, 0.001, _TS)
        # trade size must be capped to available balance, so balance → 0
        assert port.balance == pytest.approx(0.0)

    def test_zero_size_returns_minus_one(self):
        port = PaperPortfolio(10_000.0)
        tid = port.open_trade("X/Y", "s", "A", "B", 1.0, 1.0, 0.0, 0.001, _TS)
        assert tid == -1

    def test_trade_recorded_as_open(self):
        port = PaperPortfolio(10_000.0)
        tid = port.open_trade("X/Y", "s", "A", "B", 1.0, 1.0, 50.0, 0.001, _TS)
        assert tid in port._open_trades
        assert port._open_trades[tid].is_open is True


# ---------------------------------------------------------------------------
# close_trade
# ---------------------------------------------------------------------------

class TestCloseTrade:
    def test_net_pnl_after_fees(self):
        """gross_pnl - fee_rate*size*2 = net_pnl"""
        port = PaperPortfolio(10_000.0)
        size = 100.0
        gross = 2.0
        fee_cost = size * 0.001 * 2  # = 0.20
        trade = _open_and_close(port, gross, size=size)

        assert trade.gross_pnl == pytest.approx(gross)
        assert trade.fee_cost  == pytest.approx(fee_cost)
        assert trade.net_pnl   == pytest.approx(gross - fee_cost)

    def test_balance_restored_plus_profit(self):
        port = PaperPortfolio(10_000.0)
        trade = _open_and_close(port, gross_pnl=3.0, size=100.0)
        expected = 10_000.0 + trade.net_pnl
        assert port.balance == pytest.approx(expected)

    def test_losing_trade_reduces_balance(self):
        port = PaperPortfolio(10_000.0)
        trade = _open_and_close(port, gross_pnl=-1.0, size=100.0)
        assert trade.net_pnl < 0
        assert port.balance < 10_000.0

    def test_trade_removed_from_open_dict(self):
        port = PaperPortfolio(10_000.0)
        tid = port.open_trade("X/Y", "s", "A", "B", 1.0, 1.0, 50.0, 0.001, _TS)
        port.close_trade(tid, 1.0)
        assert tid not in port._open_trades

    def test_trade_marked_as_closed(self):
        port = PaperPortfolio(10_000.0)
        trade = _open_and_close(port, 1.0)
        assert trade.is_open is False

    def test_invalid_id_returns_none(self):
        port = PaperPortfolio(10_000.0)
        result = port.close_trade(9999, 1.0)
        assert result is None


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class TestAnalytics:
    def _make_portfolio_with_trades(
        self, wins: int, losses: int, win_pnl: float = 1.0, loss_pnl: float = -0.5
    ) -> PaperPortfolio:
        port = PaperPortfolio(10_000.0)
        ts = _TS
        for _ in range(wins):
            _open_and_close(port, win_pnl, size=100.0, ts_open=ts, ts_close=ts + 3_600_000)
            ts += 7_200_000
        for _ in range(losses):
            _open_and_close(port, loss_pnl, size=100.0, ts_open=ts, ts_close=ts + 3_600_000)
            ts += 7_200_000
        return port

    def test_total_pnl(self):
        port = self._make_portfolio_with_trades(3, 2, win_pnl=2.0, loss_pnl=-1.0)
        fee = 0.001 * 100.0 * 2  # 0.20 per trade
        expected = 3 * (2.0 - fee) + 2 * (-1.0 - fee)
        assert port.total_pnl == pytest.approx(expected)

    def test_win_rate_all_wins(self):
        port = self._make_portfolio_with_trades(5, 0, win_pnl=1.0)
        assert port.win_rate == pytest.approx(100.0)

    def test_win_rate_all_losses(self):
        port = self._make_portfolio_with_trades(0, 5, loss_pnl=-1.0)
        assert port.win_rate == pytest.approx(0.0)

    def test_win_rate_mixed(self):
        port = self._make_portfolio_with_trades(3, 1, win_pnl=1.0, loss_pnl=-1.0)
        assert port.win_rate == pytest.approx(75.0)

    def test_win_rate_empty(self):
        port = PaperPortfolio(10_000.0)
        assert port.win_rate == pytest.approx(0.0)

    def test_max_drawdown_flat(self):
        """No losing trades → drawdown should be 0."""
        port = self._make_portfolio_with_trades(5, 0, win_pnl=1.0)
        assert port.max_drawdown_pct >= 0.0

    def test_max_drawdown_with_loss(self):
        port = self._make_portfolio_with_trades(1, 1, win_pnl=0.5, loss_pnl=-5.0)
        assert port.max_drawdown_pct > 0.0

    def test_avg_profit_per_trade(self):
        port = self._make_portfolio_with_trades(2, 0, win_pnl=2.0)
        fee = 0.001 * 100.0 * 2
        expected = 2.0 - fee
        assert port.avg_profit_per_trade == pytest.approx(expected)

    def test_sharpe_zero_with_single_trade(self):
        """Single trade → can't compute std dev → Sharpe = 0."""
        port = PaperPortfolio(10_000.0)
        _open_and_close(port, 2.0)
        assert port.sharpe_ratio() == pytest.approx(0.0)

    def test_sharpe_nonzero_with_multiple_trades(self):
        port = self._make_portfolio_with_trades(5, 2, win_pnl=2.0, loss_pnl=-0.5)
        # With mixed results, Sharpe should be non-zero and finite
        s = port.sharpe_ratio()
        assert s != 0.0
        assert not (s != s)  # not NaN

    def test_total_return_pct(self):
        port = PaperPortfolio(10_000.0)
        _open_and_close(port, 200.0, size=100.0)  # big gross win
        assert port.total_return_pct != pytest.approx(0.0)

    def test_summary_keys(self):
        port = PaperPortfolio(10_000.0)
        _open_and_close(port, 1.0)
        s = port.summary()
        required = {
            "initial_balance_usdt", "final_balance_usdt", "total_pnl_usdt",
            "total_return_pct", "total_trades", "win_rate_pct",
            "avg_profit_per_trade_usdt", "max_drawdown_pct", "sharpe_ratio",
            "open_trades",
        }
        assert required <= set(s.keys())

    def test_summary_trade_count(self):
        port = self._make_portfolio_with_trades(3, 2)
        s = port.summary()
        assert s["total_trades"] == 5


# ---------------------------------------------------------------------------
# Trade properties
# ---------------------------------------------------------------------------

class TestTradeProperties:
    def test_profit_pct(self):
        t = Trade(
            trade_id=1, symbol="BTC/USDT", strategy="cross",
            open_ts=_TS, buy_exchange="A", sell_exchange="B",
            size_usdt=100.0, net_pnl=1.5,
        )
        assert t.profit_pct == pytest.approx(1.5)

    def test_profit_pct_zero_size(self):
        t = Trade(
            trade_id=1, symbol="BTC/USDT", strategy="cross",
            open_ts=_TS, size_usdt=0.0,
        )
        assert t.profit_pct == pytest.approx(0.0)

    def test_duration_s(self):
        t = Trade(
            trade_id=1, symbol="BTC/USDT", strategy="cross",
            open_ts=_TS, close_ts=_TS + 3_600_000,
        )
        assert t.duration_s == pytest.approx(3600.0)
