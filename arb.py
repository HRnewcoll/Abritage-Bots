#!/usr/bin/env python3
"""
arb.py — Unified launcher for Arbitrage Bots
=============================================

Run this from the repository root to get an interactive menu.
No module paths, no memorising commands — just:

    python arb.py

Requirements: Python 3.9+  |  pip install -r python/requirements.txt
              (or run setup.sh / setup.bat first)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── Path wiring ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
PYTHON_DIR = ROOT / "python"

# Add python/ so relative imports inside the submodules work
sys.path.insert(0, str(PYTHON_DIR))

# ── Dependency check ─────────────────────────────────────────────────────────

def _check_deps() -> bool:
    """Return True if all required packages are importable."""
    missing = []
    for pkg in ["rich", "requests", "numpy", "yaml"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg if pkg != "yaml" else "PyYAML")
    if missing:
        print(f"\n  ⚠  Missing packages: {', '.join(missing)}")
        print(f"     Run:  pip install -r python/requirements.txt")
        print(f"     Or:   ./setup.sh    (Linux/macOS)")
        print(f"           setup.bat     (Windows)\n")
        return False
    return True


def _require_rich():
    """Import Rich or raise a helpful error."""
    try:
        import rich
    except ImportError:
        print("\n  rich not installed. Run: pip install rich\n")
        sys.exit(1)


# ── Rich helpers ──────────────────────────────────────────────────────────────

def _print_banner():
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    t = Text.assemble(
        ("⚡  Arbitrage Bots\n", "bold cyan"),
        ("   Paper-trading simulator · Backtester · Live bots\n\n", "dim"),
        ("   No API keys needed for simulation mode.\n", "green"),
        ("   All live bots default to dry_run (no real orders).", "yellow"),
    )
    console.print(Panel(t, border_style="cyan", padding=(0, 2)))


def _menu(console, choices: list[tuple[str, str]]) -> str:
    """
    Display a numbered menu and return the chosen key.

    choices: list of (key, label) pairs.
    """
    from rich.table import Table
    from rich import box

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Key",   style="bold cyan",  width=4)
    table.add_column("Label", style="white")

    for key, label in choices:
        table.add_row(key, label)

    console.print(table)
    console.print()

    valid = {k for k, _ in choices}
    while True:
        try:
            choice = console.input("  [bold cyan]Enter number →[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  Bye! 👋\n")
            sys.exit(0)
        if choice in valid:
            return choice
        console.print(f"  [red]Invalid choice '{choice}' — try again.[/]")


# ── Sub-command runners ───────────────────────────────────────────────────────

def _run_simulator(console):
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel

    console.print(Panel("[bold cyan]📊  Paper-Trading Simulator[/]\n"
                        "[dim]Uses real Binance historical prices — no API key needed.[/]",
                        border_style="cyan"))

    symbol    = Prompt.ask("  Trading pair", default="BTC/USDT")
    timeframe = Prompt.ask("  Timeframe (1m 5m 15m 1h 4h 1d)", default="1h")
    candles   = Prompt.ask("  Number of candles to replay", default="500")
    balance   = Prompt.ask("  Starting virtual balance (USDT)", default="10000")
    strategy  = Prompt.ask("  Strategy (cross_exchange / ai_spread / all)", default="all")
    exchanges = Prompt.ask("  Number of simulated exchanges", default="3")
    output    = Prompt.ask("  HTML report filename", default="simulation_report.html")

    cmd = [
        sys.executable, "-m", "simulator.run",
        "--symbol",    symbol,
        "--timeframe", timeframe,
        "--candles",   candles,
        "--balance",   balance,
        "--strategy",  strategy,
        "--exchanges", exchanges,
        "--output",    output,
    ]
    console.print()
    console.print(f"  [dim]Running:[/] [cyan]{' '.join(cmd)}[/]\n")
    subprocess.run(cmd, cwd=str(PYTHON_DIR))
    console.print(f"\n  [green]Done![/]  HTML report saved → [underline]{output}[/]")
    console.print("  Open it in your browser to see the equity curve and trade list.\n")


def _run_backtest(console):
    from rich.prompt import Prompt

    console.print("[bold cyan]📈  Backtester[/]\n")

    symbol    = Prompt.ask("  Trading pair", default="BTC/USDT")
    timeframe = Prompt.ask("  Timeframe", default="1h")
    candles   = Prompt.ask("  Candles", default="500")
    balance   = Prompt.ask("  Starting balance (USDT)", default="10000")
    compare   = Prompt.ask("  Compare strategies? (yes/no)", default="no")

    cmd = [
        sys.executable, "-m", "backtest.backtest",
        "--symbol",    symbol,
        "--timeframe", timeframe,
        "--candles",   candles,
        "--balance",   balance,
    ]
    if compare.lower() in ("yes", "y"):
        cmd.append("--compare")

    console.print()
    console.print(f"  [dim]Running:[/] [cyan]{' '.join(cmd)}[/]\n")
    subprocess.run(cmd, cwd=str(PYTHON_DIR))
    console.print()


def _run_bot(console, bot_key: str):
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel

    bot_names = {
        "cross": ("cross_exchange_arb.bot", "Cross-Exchange Arbitrage Bot"),
        "tri":   ("triangular_arb.bot",     "Triangular Arbitrage Bot"),
        "ai":    ("ai_arb.bot",             "AI Arbitrage Bot"),
    }
    module, name = bot_names[bot_key]

    config = str(PYTHON_DIR / "config" / "config.yaml")
    if not os.path.exists(config):
        console.print(f"\n  [yellow]⚠  Config file not found: {config}[/]")
        console.print("     Run the [bold]Setup Wizard[/] first, or copy the example:\n")
        console.print(f"     [dim]cp python/config/config.example.yaml python/config/config.yaml[/]\n")
        return

    console.print(Panel(f"[bold cyan]🤖  {name}[/]\n"
                        "[yellow]All bots run in dry_run mode by default — no real orders.[/]",
                        border_style="cyan"))

    cmd = [sys.executable, "-m", module, "--config", config]

    if bot_key == "ai":
        strategy = Prompt.ask("  Strategy (spread / rl / both)", default="both")
        cmd += ["--strategy", strategy]

    console.print(f"\n  [dim]Running:[/] [cyan]{' '.join(cmd)}[/]")
    console.print("  Press [bold]Ctrl+C[/] to stop.\n")
    try:
        subprocess.run(cmd, cwd=str(PYTHON_DIR))
    except KeyboardInterrupt:
        console.print("\n  [yellow]Bot stopped.[/]\n")


def _run_setup_wizard(console):
    """Import and run the interactive setup wizard."""
    try:
        import setup_wizard
        setup_wizard.run()
    except ImportError:
        # Fallback: run as subprocess
        wiz = PYTHON_DIR / "setup_wizard.py"
        if wiz.exists():
            subprocess.run([sys.executable, str(wiz)], cwd=str(PYTHON_DIR))
        else:
            console.print("[red]setup_wizard.py not found.[/]")


def _show_help(console):
    from rich.table import Table
    from rich import box
    from rich.panel import Panel

    console.print(Panel("[bold cyan]❓  Help & Quick Reference[/]", border_style="cyan"))

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta", padding=(0, 1))
    t.add_column("Command",     style="cyan",  min_width=20)
    t.add_column("What it does",               min_width=50)

    rows = [
        ("python arb.py",                         "Open this interactive menu"),
        ("python arb.py simulate",                "Run simulator directly (skips menu)"),
        ("python arb.py backtest",                "Run backtester directly"),
        ("python arb.py bot cross",               "Run cross-exchange bot"),
        ("python arb.py bot tri",                 "Run triangular bot"),
        ("python arb.py bot ai",                  "Run AI bot"),
        ("python arb.py wizard",                  "Run setup wizard"),
        ("./setup.sh  (setup.bat on Windows)",    "One-command environment setup"),
        ("python -m simulator.run --help",         "All simulator CLI options"),
        ("python -m backtest.backtest --help",     "All backtest CLI options"),
    ]
    for cmd, desc in rows:
        t.add_row(cmd, desc)

    console.print(t)
    console.print()
    console.print("[dim]  Tip: The simulator needs no API keys — try it first![/]")
    console.print("[dim]  Tip: All bots default to dry_run=true in config.yaml.[/]\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    _require_rich()

    from rich.console import Console
    console = Console()

    # ── Command-line shortcuts (bypass menu) ─────────────────────────────────
    args = sys.argv[1:]
    if args:
        cmd = args[0].lower()
        if cmd == "simulate":
            _run_simulator(console)
        elif cmd == "backtest":
            _run_backtest(console)
        elif cmd == "bot" and len(args) >= 2:
            _run_bot(console, args[1].lower())
        elif cmd == "wizard":
            _run_setup_wizard(console)
        elif cmd in ("help", "--help", "-h"):
            _show_help(console)
        return

    # ── Check dependencies before showing menu ───────────────────────────────
    if not _check_deps():
        sys.exit(1)

    # ── Interactive menu loop ────────────────────────────────────────────────
    CHOICES = [
        ("1", "📊  Run Paper-Trading Simulator    [green](no API keys needed)[/]"),
        ("2", "📈  Run Backtester                 [green](no API keys needed)[/]"),
        ("3", "🔄  Run Cross-Exchange Bot         [yellow](needs API keys + config)[/]"),
        ("4", "🔺  Run Triangular Arbitrage Bot   [yellow](needs API keys + config)[/]"),
        ("5", "🧠  Run AI Arbitrage Bot           [yellow](needs API keys + config)[/]"),
        ("6", "⚙️   Setup Wizard                  [cyan](configure API keys)[/]"),
        ("7", "❓  Help & quick-reference"),
        ("0", "🚪  Exit"),
    ]

    while True:
        console.clear()
        _print_banner()
        console.print("[bold]  What would you like to do?[/]\n")
        choice = _menu(console, CHOICES)

        if   choice == "1": _run_simulator(console)
        elif choice == "2": _run_backtest(console)
        elif choice == "3": _run_bot(console, "cross")
        elif choice == "4": _run_bot(console, "tri")
        elif choice == "5": _run_bot(console, "ai")
        elif choice == "6": _run_setup_wizard(console)
        elif choice == "7": _show_help(console)
        elif choice == "0":
            console.print("\n  Bye! 👋\n")
            sys.exit(0)

        try:
            console.input("  [dim]Press Enter to return to menu…[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n  Bye! 👋\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
