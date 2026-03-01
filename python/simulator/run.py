"""
run.py — One-command simulation runner with Rich live progress display
======================================================================

Usage
-----
    # From the python/ directory:
    python -m simulator.run

    # Customise:
    python -m simulator.run --symbol ETH/USDT --candles 720 --balance 5000
    python -m simulator.run --symbol BTC/USDT --candles 500 --strategy cross_exchange
    python -m simulator.run --symbol SOL/USDT --timeframe 15m --output report.html

Options
-------
  --symbol     Trading pair  (default: BTC/USDT)
  --timeframe  Candle size   (default: 1h)
  --candles    Number of bars to replay  (default: 500, max 1000)
  --balance    Starting virtual balance in USDT  (default: 10000)
  --strategy   Strategies: cross_exchange | ai_spread | all  (default: all)
  --exchanges  Number of simulated exchanges  (default: 3)
  --min-profit Minimum spread % to trigger a trade  (default: 0.15)
  --fee        Taker fee rate, e.g. 0.001 = 0.1%  (default: 0.001)
  --output     Path for HTML report  (default: simulation_report.html)
  --no-html    Skip HTML report generation
  --seed       Random seed for spread noise  (default: 42)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Ensure the parent directory is on the path when run as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.engine import SimulationConfig, SimulationEngine
from simulator.report import generate_html_report, print_terminal_report

logger = logging.getLogger("simulator.run")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m simulator.run",
        description="Paper-trading simulation using real historical market data. No API keys required!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m simulator.run
  python -m simulator.run --symbol ETH/USDT --candles 720
  python -m simulator.run --symbol BTC/USDT --balance 50000 --strategy cross_exchange
  python -m simulator.run --symbol SOL/USDT --timeframe 15m --output sol_report.html
        """,
    )
    p.add_argument("--symbol",     default="BTC/USDT",              help="Trading pair (default: BTC/USDT)")
    p.add_argument("--timeframe",  default="1h",                     help="Candle size: 1m 5m 15m 1h 4h 1d (default: 1h)")
    p.add_argument("--candles",    type=int,   default=500,          help="Number of historical candles to replay (default: 500)")
    p.add_argument("--balance",    type=float, default=10_000.0,     help="Starting virtual USDT balance (default: 10000)")
    p.add_argument("--strategy",   default="all",                    help="Strategy: cross_exchange | ai_spread | all (default: all)")
    p.add_argument("--exchanges",  type=int,   default=3,            help="Number of simulated exchanges (default: 3)")
    p.add_argument("--min-profit", type=float, default=0.15,         dest="min_profit", help="Min spread %% to trigger a trade (default: 0.15)")
    p.add_argument("--fee",        type=float, default=0.001,        help="Taker fee rate per leg (default: 0.001 = 0.1%%)")
    p.add_argument("--output",     default="simulation_report.html", help="HTML report output path")
    p.add_argument("--no-html",    action="store_true",              help="Skip HTML report generation")
    p.add_argument("--seed",       type=int,   default=42,           help="Random seed for spread noise (default: 42)")
    return p


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def _make_progress_callback(total: int):
    """Returns a callback that renders a Rich progress bar."""
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Simulating…"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("candles"),
            TimeRemainingColumn(),
        )
        task = progress.add_task("sim", total=total)
        progress.start()

        def callback(i: int, n: int) -> None:
            progress.update(task, completed=i + 1)
            if i + 1 >= n:
                progress.stop()

        return callback, progress
    except ImportError:
        # Fallback: simple print progress
        def callback(i: int, n: int) -> None:  # type: ignore[misc]
            if i % max(1, n // 20) == 0:
                pct = (i + 1) / n * 100
                print(f"\r  Progress: {pct:5.1f}%  ({i+1}/{n} candles)", end="", flush=True)
            if i + 1 >= n:
                print()

        return callback, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.WARNING,  # suppress ccxt noise; use WARNING
    )
    logging.getLogger("simulator").setLevel(logging.INFO)

    # Strategy list
    if args.strategy == "all":
        strategies = ["cross_exchange", "ai_spread"]
    else:
        strategies = [s.strip() for s in args.strategy.split(",")]

    # Build config
    cfg = SimulationConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        candles=min(args.candles, 1000),
        initial_balance=args.balance,
        fee_rate=args.fee,
        min_profit_pct=args.min_profit,
        max_trade_size_usdt=args.balance * 0.05,  # 5% of balance per trade
        n_exchanges=args.exchanges,
        strategies=strategies,
        seed=args.seed,
    )

    # Banner
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        banner = Text.assemble(
            ("🚀  Arbitrage Bot — Paper-Trading Simulator\n", "bold cyan"),
            ("   Real historical prices · No API keys · No real money\n\n", "dim"),
            (f"   Symbol    : {cfg.symbol}\n",    "white"),
            (f"   Timeframe : {cfg.timeframe}\n", "white"),
            (f"   Candles   : {cfg.candles}\n",   "white"),
            (f"   Balance   : ${cfg.initial_balance:,.0f} USDT\n", "white"),
            (f"   Strategies: {', '.join(cfg.strategies)}\n", "white"),
            (f"   Exchanges : {cfg.n_exchanges} (simulated)\n", "white"),
        )
        console.print(Panel(banner, border_style="cyan"))
    except ImportError:
        print("\n" + "=" * 60)
        print("  Arbitrage Bot — Paper-Trading Simulator")
        print(f"  Symbol: {cfg.symbol}  |  Candles: {cfg.candles}  |  Balance: ${cfg.initial_balance:,.0f}")
        print("=" * 60)

    # Run simulation with progress display
    callback, progress_obj = _make_progress_callback(cfg.candles)
    t_start = time.time()

    try:
        engine = SimulationEngine(cfg, progress_callback=callback)
        result = engine.run()
    except RuntimeError as exc:
        print(f"\n❌  Error: {exc}")
        sys.exit(1)

    elapsed = time.time() - t_start
    print(f"\n  ✓ Simulation finished in {elapsed:.1f}s\n")

    # Print terminal report
    print_terminal_report(result)

    # Generate HTML report
    if not args.no_html:
        path = generate_html_report(result, output_path=args.output)
        try:
            from rich.console import Console
            Console().print(
                f"  📄 HTML report saved → [bold underline]{path}[/]\n"
                "     Open it in your browser to view interactive charts.\n"
            )
        except ImportError:
            print(f"\n  HTML report saved → {path}")
            print("  Open it in your browser to view interactive charts.")


if __name__ == "__main__":
    main()
