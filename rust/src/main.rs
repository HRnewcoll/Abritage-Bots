//! Abritage-Bots — High-performance cross-exchange crypto arbitrage bot (Rust)
//!
//! # Usage
//!
//! ```bash
//! cargo run -- --config ../python/config/config.yaml
//! ```
//!
//! Configuration is shared with the Python bots via the same YAML file.
//! Set `dry_run: true` in config until you have verified correct behaviour.

mod arbitrage;
mod config;
mod exchange;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use rust_decimal::Decimal;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use arbitrage::scan_symbol;
use config::BotConfig;
use exchange::build_exchange;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "arb-bot", about = "High-performance cross-exchange crypto arbitrage bot")]
struct Cli {
    /// Path to YAML config file
    #[arg(short, long, default_value = "../python/config/config.yaml")]
    config: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise logging (RUST_LOG=debug for verbose output)
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let cfg = BotConfig::from_file(&cli.config)?;

    let dry_run = cfg.arbitrage.dry_run;
    let min_profit_pct = cfg.arbitrage.min_profit_pct;
    let max_trade_usdt = cfg.arbitrage.max_trade_size_usdt;
    let poll_interval = Duration::from_secs(cfg.arbitrage.poll_interval_seconds);
    let symbols = cfg.arbitrage.symbols.clone();

    // Build exchange clients
    let mut exchanges: Vec<Arc<dyn exchange::Exchange>> = Vec::new();
    for (name, ex_cfg) in &cfg.exchanges {
        if let Some(client) = build_exchange(name, ex_cfg) {
            info!("Connected to exchange: {}", name);
            exchanges.push(Arc::from(client));
        }
    }

    if exchanges.len() < 2 {
        anyhow::bail!("Need at least 2 working exchanges. Check config.");
    }

    let mode = if dry_run { "DRY-RUN" } else { "LIVE" };
    info!(
        "=== Rust Cross-Exchange Arbitrage Bot started [{}] ===",
        mode
    );
    info!(
        "Watching {} symbol(s) across {} exchange(s)",
        symbols.len(),
        exchanges.len()
    );

    // Main event loop
    loop {
        for symbol in &symbols {
            match scan_symbol(symbol, &exchanges, min_profit_pct).await {
                Some(opp) => {
                    info!(
                        "OPPORTUNITY | {} | Buy on {:10} @ {} | Sell on {:10} @ {} | Profit: {:.4}%",
                        opp.symbol,
                        opp.buy_exchange,
                        opp.buy_price,
                        opp.sell_exchange,
                        opp.sell_price,
                        opp.profit_pct,
                    );

                    if !dry_run {
                        // Find the actual exchange objects
                        let buy_ex = exchanges.iter().find(|e| e.name() == opp.buy_exchange);
                        let sell_ex = exchanges.iter().find(|e| e.name() == opp.sell_exchange);

                        if let (Some(buy), Some(sell)) = (buy_ex, sell_ex) {
                            let amount: Decimal =
                                (max_trade_usdt / opp.buy_price.to_string().parse::<f64>().unwrap_or(1.0))
                                    .to_string()
                                    .parse()
                                    .unwrap_or(Decimal::ONE);

                            match buy.place_market_buy(&opp.symbol, amount).await {
                                Ok(id) => info!("Buy order placed: {}", id),
                                Err(e) => warn!("Buy order FAILED on {}: {}", opp.buy_exchange, e),
                            }
                            match sell.place_market_sell(&opp.symbol, amount).await {
                                Ok(id) => info!("Sell order placed: {}", id),
                                Err(e) => warn!(
                                    "Sell order FAILED on {} – WARNING: position may be unhedged!: {}",
                                    opp.sell_exchange, e
                                ),
                            }
                        }
                    }
                }
                None => {
                    tracing::debug!("No opportunity for {}", symbol);
                }
            }
        }

        tokio::time::sleep(poll_interval).await;
    }
}
