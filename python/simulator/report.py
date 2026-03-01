"""
report.py — Rich terminal report + HTML report generator
=========================================================

Produces:
1. A colour-coded summary table printed to the terminal using the `rich`
   library.
2. An interactive HTML file (no external JS CDN required) with:
   - Equity curve chart (inline SVG)
   - Trade history table
   - Key statistics panel
"""

from __future__ import annotations

import datetime
import html
import os
from typing import List, Optional

from simulator.engine import SimulationResult
from simulator.portfolio import Trade


# ---------------------------------------------------------------------------
# Terminal report (rich)
# ---------------------------------------------------------------------------

def print_terminal_report(result: SimulationResult) -> None:
    """Print a rich, colour-coded summary to the terminal."""
    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        _rich = True
    except ImportError:
        _rich = False

    port  = result.portfolio
    stats = port.summary()
    cfg   = result.config

    if not _rich:
        # Plain-text fallback
        print("\n" + "=" * 60)
        print("  SIMULATION RESULTS")
        print("=" * 60)
        for k, v in stats.items():
            print(f"  {k:<35} {v}")
        print("=" * 60)
        return

    console = Console()
    console.print()

    # ── Header ──────────────────────────────────────────────────────────
    console.rule("[bold cyan]📊  Simulation Results")
    console.print(
        f"  Symbol: [bold]{cfg.symbol}[/]  |  "
        f"Timeframe: [bold]{cfg.timeframe}[/]  |  "
        f"Candles: [bold]{result.candles_processed}[/]  |  "
        f"Exchanges: [bold]{cfg.n_exchanges}[/]"
    )
    console.print()

    # ── Stats panel ──────────────────────────────────────────────────────
    pnl = stats["total_pnl_usdt"]
    ret = stats["total_return_pct"]
    pnl_colour = "green" if pnl >= 0 else "red"
    ret_str = f"[{pnl_colour}]{ret:+.4f}%[/]"

    stats_table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    stats_table.add_column("Metric", style="bold cyan", min_width=32)
    stats_table.add_column("Value", min_width=18)

    rows = [
        ("💰 Starting Balance",     f"${stats['initial_balance_usdt']:,.2f}"),
        ("💵 Final Balance",         f"[bold]${stats['final_balance_usdt']:,.2f}[/bold]"),
        ("📈 Total P&L",             f"[{pnl_colour}]${pnl:+,.2f}[/]"),
        ("📊 Total Return",          ret_str),
        ("🔢 Total Trades",          str(stats["total_trades"])),
        ("✅ Win Rate",               f"{stats['win_rate_pct']:.2f}%"),
        ("📉 Avg Profit / Trade",    f"${stats['avg_profit_per_trade_usdt']:+.4f}"),
        ("🕳  Max Drawdown",          f"{stats['max_drawdown_pct']:.4f}%"),
        ("⚡ Sharpe Ratio",           f"{stats['sharpe_ratio']:.4f}"),
        ("🔍 Opportunities Found",   str(result.opportunities_found)),
    ]
    for label, value in rows:
        stats_table.add_row(label, value)

    console.print(Panel(stats_table, title="[bold]Portfolio Statistics[/]", border_style="cyan"))

    # ── Last 10 trades table ──────────────────────────────────────────────
    closed = port.closed_trades[-10:]
    if closed:
        console.print()
        trade_table = Table(
            title="Last 10 Trades",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
        )
        trade_table.add_column("#",           style="dim",       width=5)
        trade_table.add_column("Symbol",      width=12)
        trade_table.add_column("Strategy",    width=16)
        trade_table.add_column("Buy Ex.",     width=14)
        trade_table.add_column("Sell Ex.",    width=14)
        trade_table.add_column("Buy $",       justify="right",  width=12)
        trade_table.add_column("Sell $",      justify="right",  width=12)
        trade_table.add_column("Net PnL",     justify="right",  width=12)
        trade_table.add_column("Return %",    justify="right",  width=10)

        for t in closed:
            c = "green" if t.net_pnl > 0 else "red"
            trade_table.add_row(
                str(t.trade_id),
                t.symbol,
                t.strategy,
                t.buy_exchange,
                t.sell_exchange,
                f"${t.buy_price:,.2f}",
                f"${t.sell_price:,.2f}",
                f"[{c}]${t.net_pnl:+.4f}[/]",
                f"[{c}]{t.profit_pct:+.4f}%[/]",
            )
        console.print(trade_table)

    console.print()
    console.rule("[dim]End of Simulation Report")
    console.print()


# ---------------------------------------------------------------------------
# HTML report with inline SVG charts
# ---------------------------------------------------------------------------

def _svg_line_chart(
    values: List[float],
    width: int = 800,
    height: int = 250,
    colour: str = "#00b4d8",
    label: str = "",
    fill: bool = True,
) -> str:
    """Render a simple line chart as inline SVG (no external deps)."""
    if not values or len(values) < 2:
        return "<p style='color:#888'>Not enough data for chart.</p>"

    vmin = min(values)
    vmax = max(values)
    vrange = (vmax - vmin) or 1.0
    pad = 40

    def x_coord(i: int) -> float:
        return pad + (i / (len(values) - 1)) * (width - 2 * pad)

    def y_coord(v: float) -> float:
        return pad + (1.0 - (v - vmin) / vrange) * (height - 2 * pad)

    points = [(x_coord(i), y_coord(v)) for i, v in enumerate(values)]
    polyline = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)

    fill_path = ""
    if fill:
        pts = f"M{points[0][0]:.1f},{points[0][1]:.1f} "
        pts += " ".join(f"L{px:.1f},{py:.1f}" for px, py in points)
        pts += f" L{points[-1][0]:.1f},{height - pad:.1f} L{points[0][0]:.1f},{height - pad:.1f} Z"
        fill_path = f'<path d="{pts}" fill="{colour}" fill-opacity="0.15"/>'

    # Y-axis labels
    y_labels = ""
    for tick in [vmin, (vmin + vmax) / 2, vmax]:
        yc = y_coord(tick)
        y_labels += f'<text x="{pad - 4}" y="{yc:.1f}" font-size="10" fill="#aaa" text-anchor="end">${tick:,.0f}</text>'

    svg = f"""
<svg width="{width}" height="{height}" style="background:#1a1a2e;border-radius:8px">
  <text x="{width/2}" y="18" font-size="13" fill="#ccc" text-anchor="middle">{html.escape(label)}</text>
  {fill_path}
  <polyline points="{polyline}" fill="none" stroke="{colour}" stroke-width="2"/>
  {y_labels}
  <!-- X axis -->
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#444"/>
  <!-- Y axis -->
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#444"/>
</svg>"""
    return svg


def generate_html_report(result: SimulationResult, output_path: str = "simulation_report.html") -> str:
    """
    Generate a self-contained HTML report and write it to output_path.

    Returns the path of the written file.
    """
    port  = result.portfolio
    stats = port.summary()
    cfg   = result.config
    now   = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Charts
    equity_svg = _svg_line_chart(
        result.equity_curve,
        label=f"Equity Curve — {cfg.symbol}",
        colour="#00b4d8",
    )
    spread_svg = _svg_line_chart(
        result.spread_pct_series,
        label="Best Cross-Exchange Spread % over time",
        colour="#f77f00",
        fill=False,
        height=200,
    )

    # Stats cards
    pnl      = stats["total_pnl_usdt"]
    pnl_col  = "#4caf50" if pnl >= 0 else "#f44336"
    ret_col  = pnl_col
    wr_col   = "#4caf50" if stats["win_rate_pct"] >= 50 else "#f44336"

    def card(label: str, value: str, colour: str = "#ccc") -> str:
        return f"""
        <div class="card">
          <div class="card-label">{html.escape(label)}</div>
          <div class="card-value" style="color:{colour}">{value}</div>
        </div>"""

    cards = "".join([
        card("Starting Balance",       f"${stats['initial_balance_usdt']:,.2f}"),
        card("Final Balance",           f"${stats['final_balance_usdt']:,.2f}", "#00b4d8"),
        card("Total P&L",              f"${pnl:+,.2f}",                         pnl_col),
        card("Total Return",           f"{stats['total_return_pct']:+.4f}%",    ret_col),
        card("Total Trades",           str(stats["total_trades"])),
        card("Win Rate",               f"{stats['win_rate_pct']:.2f}%",         wr_col),
        card("Avg Profit / Trade",     f"${stats['avg_profit_per_trade_usdt']:+.4f}"),
        card("Max Drawdown",           f"{stats['max_drawdown_pct']:.4f}%",    "#f44336"),
        card("Sharpe Ratio",           f"{stats['sharpe_ratio']:.4f}"),
        card("Opportunities Found",    str(result.opportunities_found)),
    ])

    # Trade rows
    trade_rows = ""
    for t in port.closed_trades:
        c   = "#4caf50" if t.net_pnl > 0 else "#f44336"
        ts  = datetime.datetime.utcfromtimestamp(t.open_ts / 1000).strftime("%Y-%m-%d %H:%M")
        trade_rows += f"""
        <tr>
          <td>{t.trade_id}</td>
          <td>{ts}</td>
          <td>{html.escape(t.symbol)}</td>
          <td>{html.escape(t.strategy)}</td>
          <td>{html.escape(t.buy_exchange)}</td>
          <td>{html.escape(t.sell_exchange)}</td>
          <td>${t.buy_price:,.2f}</td>
          <td>${t.sell_price:,.2f}</td>
          <td style="color:{c}">${t.net_pnl:+.4f}</td>
          <td style="color:{c}">{t.profit_pct:+.4f}%</td>
        </tr>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Arbitrage Bot — Simulation Report</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0d0d1a;
      color: #e0e0e0;
      padding: 24px;
    }}
    h1 {{ font-size: 1.8rem; color: #00b4d8; margin-bottom: 4px; }}
    .meta {{ color: #888; font-size: 0.85rem; margin-bottom: 24px; }}
    h2 {{ font-size: 1.1rem; color: #90caf9; margin: 28px 0 12px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 28px;
    }}
    .card {{
      background: #1a1a2e;
      border: 1px solid #2a2a4a;
      border-radius: 8px;
      padding: 14px 18px;
    }}
    .card-label {{ font-size: 0.75rem; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .05em; }}
    .card-value  {{ font-size: 1.3rem; font-weight: 700; }}
    .chart-wrap  {{ overflow-x: auto; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #2a2a4a; }}
    th {{ background: #1a1a2e; color: #90caf9; position: sticky; top: 0; }}
    tr:hover {{ background: #1a1a2e88; }}
    .tbl-wrap {{ overflow-x: auto; max-height: 480px; overflow-y: auto; }}
    footer {{ margin-top: 40px; color: #555; font-size: 0.75rem; text-align: center; }}
  </style>
</head>
<body>
  <h1>📊 Arbitrage Bot — Simulation Report</h1>
  <p class="meta">
    Symbol: <strong>{html.escape(cfg.symbol)}</strong> &nbsp;|&nbsp;
    Timeframe: <strong>{html.escape(cfg.timeframe)}</strong> &nbsp;|&nbsp;
    Candles processed: <strong>{result.candles_processed}</strong> &nbsp;|&nbsp;
    Simulated exchanges: <strong>{cfg.n_exchanges}</strong> &nbsp;|&nbsp;
    Generated: {now}
  </p>

  <h2>Key Statistics</h2>
  <div class="cards">{cards}</div>

  <h2>Equity Curve</h2>
  <div class="chart-wrap">{equity_svg}</div>

  <h2>Inter-Exchange Spread %</h2>
  <div class="chart-wrap">{spread_svg}</div>

  <h2>All Trades ({len(port.closed_trades)})</h2>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Opened</th><th>Symbol</th><th>Strategy</th>
          <th>Buy Exchange</th><th>Sell Exchange</th>
          <th>Buy Price</th><th>Sell Price</th>
          <th>Net P&amp;L</th><th>Return %</th>
        </tr>
      </thead>
      <tbody>
        {trade_rows if trade_rows else '<tr><td colspan="10" style="text-align:center;color:#666">No closed trades.</td></tr>'}
      </tbody>
    </table>
  </div>

  <footer>Arbitrage Bots simulation — all figures are paper-trading results using real historical prices. Not financial advice.</footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    return output_path
