"""
backtest.py — Strategy backtester with performance metrics
===========================================================

Runs any configured strategy against a full historical dataset and reports:
  - Total / annualised return
  - Sharpe ratio
  - Sortino ratio
  - Calmar ratio
  - Max drawdown
  - Win rate
  - Profit factor
  - Average trade duration

Usage
-----
    # From the python/ directory:
    python -m backtest.backtest

    python -m backtest.backtest --symbol ETH/USDT --candles 1000 --timeframe 1h
    python -m backtest.backtest --compare   # compare all strategies side by side
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.engine import SimulationConfig, SimulationEngine
from simulator.portfolio import PaperPortfolio, Trade

logger = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    strategy: str
    symbol: str
    timeframe: str
    n_candles: int
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_return_pct: float
    annualised_return_pct: float
    n_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_pnl: float
    avg_trade_duration_s: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    opportunities_found: int


def _compute_metrics(
    strategy_name: str,
    result,  # SimulationResult
    candles_per_year: float,
) -> BacktestMetrics:
    port    = result.portfolio
    closed  = port.closed_trades
    cfg     = result.config

    total_return_pct = port.total_return_pct

    # Annualised return (CAGR)
    n_periods = result.candles_processed
    if n_periods > 0 and total_return_pct > -100:
        ratio = 1.0 + total_return_pct / 100.0
        if ratio > 0:
            ann_return = (ratio ** (candles_per_year / n_periods) - 1.0) * 100.0
        else:
            ann_return = -100.0
    else:
        ann_return = 0.0

    # Win rate
    wins   = sum(1 for t in closed if t.net_pnl > 0)
    losses = sum(1 for t in closed if t.net_pnl <= 0)
    win_rate = wins / len(closed) * 100.0 if closed else 0.0

    # Profit factor
    gross_profit = sum(t.net_pnl for t in closed if t.net_pnl > 0)
    gross_loss   = abs(sum(t.net_pnl for t in closed if t.net_pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Avg trade duration
    durations = [t.duration_s for t in closed if t.duration_s > 0]
    avg_dur = sum(durations) / len(durations) if durations else 0.0

    # Sharpe (already in portfolio)
    sharpe = port.sharpe_ratio()

    # Sortino (penalises only downside volatility)
    returns_list = [t.net_pnl / t.size_usdt for t in closed if t.size_usdt > 0]
    if len(returns_list) >= 2:
        mean_r   = sum(returns_list) / len(returns_list)
        downside = [min(r, 0.0) for r in returns_list]
        downside_var = sum(r ** 2 for r in downside) / (len(downside) - 1 + 1e-9)
        downside_std = math.sqrt(downside_var)
        sortino = (mean_r / downside_std * math.sqrt(candles_per_year)) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # Calmar = annualised return / max drawdown
    mdd = port.max_drawdown_pct
    calmar = ann_return / mdd if mdd > 0 else float("inf")

    return BacktestMetrics(
        strategy=strategy_name,
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        n_candles=result.candles_processed,
        initial_balance=cfg.initial_balance,
        final_balance=port.balance,
        total_pnl=port.total_pnl,
        total_return_pct=round(total_return_pct, 4),
        annualised_return_pct=round(ann_return, 4),
        n_trades=len(closed),
        win_rate_pct=round(win_rate, 2),
        profit_factor=round(profit_factor, 4),
        avg_trade_pnl=round(port.avg_profit_per_trade, 6),
        avg_trade_duration_s=round(avg_dur, 1),
        max_drawdown_pct=round(mdd, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4) if calmar != float("inf") else float("inf"),
        opportunities_found=result.opportunities_found,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_metrics_rich(metrics_list: List[BacktestMetrics]) -> None:
    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.rule("[bold cyan]📊  Backtest Results")

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Metric",             style="cyan",     min_width=28)

        for m in metrics_list:
            table.add_column(m.strategy, justify="right", min_width=16)

        def row(label: str, values, fmt=str, colour_fn=None):
            cells = []
            for v in values:
                s = fmt(v)
                if colour_fn:
                    colour = colour_fn(v)
                    s = f"[{colour}]{s}[/]"
                cells.append(s)
            table.add_row(label, *cells)

        _green_red = lambda v: "green" if v >= 0 else "red"
        _green_red_pct = lambda v: "green" if v >= 50 else "red"

        all_returns  = [m.total_return_pct         for m in metrics_list]
        all_ann      = [m.annualised_return_pct     for m in metrics_list]
        all_pnl      = [m.total_pnl                for m in metrics_list]
        all_wr       = [m.win_rate_pct              for m in metrics_list]
        all_pf       = [m.profit_factor             for m in metrics_list]
        all_mdd      = [m.max_drawdown_pct          for m in metrics_list]
        all_sharpe   = [m.sharpe_ratio              for m in metrics_list]
        all_sortino  = [m.sortino_ratio             for m in metrics_list]
        all_calmar   = [m.calmar_ratio              for m in metrics_list]
        all_trades   = [m.n_trades                 for m in metrics_list]
        all_opps     = [m.opportunities_found       for m in metrics_list]

        row("Symbol / Timeframe",       [f"{m.symbol} {m.timeframe}" for m in metrics_list])
        row("Candles Replayed",         [str(m.n_candles)             for m in metrics_list])
        row("Starting Balance",         [f"${m.initial_balance:,.0f}" for m in metrics_list])
        row("Final Balance",            [f"${m.final_balance:,.2f}"   for m in metrics_list])
        row("Total P&L",                all_pnl,  lambda v: f"${v:+,.2f}",    _green_red)
        row("Total Return %",           all_returns, lambda v: f"{v:+.4f}%",  _green_red)
        row("Annualised Return %",      all_ann,  lambda v: f"{v:+.4f}%",     _green_red)
        row("# Trades",                 all_trades,  str)
        row("Opportunities Found",      all_opps,    str)
        row("Win Rate %",               all_wr,   lambda v: f"{v:.2f}%",      _green_red_pct)
        row("Profit Factor",            all_pf,   lambda v: f"{v:.4f}" if v != float("inf") else "∞")
        row("Avg P&L / Trade",          [m.avg_trade_pnl for m in metrics_list],
                                          lambda v: f"${v:+.5f}",             _green_red)
        row("Max Drawdown %",           all_mdd,  lambda v: f"{v:.4f}%",
                                          lambda v: "red" if v > 5 else "green")
        row("Sharpe Ratio",             all_sharpe,  lambda v: f"{v:.4f}",
                                          lambda v: "green" if v > 1 else ("yellow" if v > 0 else "red"))
        row("Sortino Ratio",            all_sortino, lambda v: f"{v:.4f}")
        row("Calmar Ratio",             all_calmar,  lambda v: f"{v:.4f}" if v != float("inf") else "∞")

        console.print(table)
        console.print()

    except ImportError:
        # Plain text fallback
        print("\n" + "=" * 70)
        print("  BACKTEST RESULTS")
        print("=" * 70)
        for m in metrics_list:
            print(f"\n  Strategy: {m.strategy}")
            for k, v in m.__dict__.items():
                print(f"    {k:<35} {v}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_backtest(
    strategy: str,
    symbol: str,
    timeframe: str,
    candles: int,
    balance: float,
    fee: float,
    min_profit: float,
) -> BacktestMetrics:
    strategy_map = {
        "cross_exchange": ["cross_exchange"],
        "ai_spread":      ["ai_spread"],
        "all":            ["cross_exchange", "ai_spread"],
    }
    strats = strategy_map.get(strategy, [strategy])

    # Candles-per-year for annualisation
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    mins = tf_minutes.get(timeframe, 60)
    candles_per_year = 525_600 / mins  # minutes in a year / candle minutes

    cfg = SimulationConfig(
        symbol=symbol,
        timeframe=timeframe,
        candles=min(candles, 1000),
        initial_balance=balance,
        fee_rate=fee,
        min_profit_pct=min_profit,
        max_trade_size_usdt=balance * 0.05,
        strategies=strats,
    )
    engine = SimulationEngine(cfg)
    result = engine.run()
    return _compute_metrics(strategy, result, candles_per_year)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m backtest.backtest",
        description="Backtest arbitrage strategies on real historical data.",
    )
    parser.add_argument("--symbol",     default="BTC/USDT")
    parser.add_argument("--timeframe",  default="1h")
    parser.add_argument("--candles",    type=int,   default=500)
    parser.add_argument("--balance",    type=float, default=10_000.0)
    parser.add_argument("--strategy",   default="all",
                        help="Strategy: cross_exchange | ai_spread | all (default: all)")
    parser.add_argument("--fee",        type=float, default=0.001)
    parser.add_argument("--min-profit", type=float, default=0.15, dest="min_profit")
    parser.add_argument("--compare",    action="store_true",
                        help="Compare cross_exchange vs ai_spread side by side")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.WARNING,
    )
    logging.getLogger("simulator").setLevel(logging.INFO)

    if args.compare:
        strategies = ["cross_exchange", "ai_spread"]
        metrics_list = []
        for s in strategies:
            print(f"  Running backtest: {s}…")
            m = _run_backtest(s, args.symbol, args.timeframe, args.candles,
                              args.balance, args.fee, args.min_profit)
            metrics_list.append(m)
    else:
        print(f"  Running backtest: {args.strategy}…")
        m = _run_backtest(args.strategy, args.symbol, args.timeframe,
                          args.candles, args.balance, args.fee, args.min_profit)
        metrics_list = [m]

    _print_metrics_rich(metrics_list)


if __name__ == "__main__":
    main()
