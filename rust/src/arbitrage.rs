use std::sync::Arc;

use rust_decimal::Decimal;
use tracing::debug;

use crate::exchange::Exchange;

/// A detected arbitrage opportunity.
#[derive(Debug)]
pub struct Opportunity {
    pub symbol: String,
    pub buy_exchange: String,
    pub sell_exchange: String,
    pub buy_price: Decimal,
    pub sell_price: Decimal,
    pub profit_pct: f64,
}

/// Calculate net profit percentage after taker fees on both legs.
///
/// ```text
/// profit% = (sell * (1 - sell_fee)) / (buy * (1 + buy_fee)) - 1) * 100
/// ```
pub fn calc_profit_pct(
    buy_price: Decimal,
    sell_price: Decimal,
    buy_fee: Decimal,
    sell_fee: Decimal,
) -> f64 {
    let one = Decimal::ONE;
    let eff_buy = buy_price * (one + buy_fee);
    let eff_sell = sell_price * (one - sell_fee);
    if eff_buy.is_zero() {
        return 0.0;
    }
    let ratio = eff_sell / eff_buy;
    (ratio.to_string().parse::<f64>().unwrap_or(1.0) - 1.0) * 100.0
}

/// Scan a single symbol across all exchange pairs and return the best
/// opportunity found, or None if no profitable spread exists.
pub async fn scan_symbol(
    symbol: &str,
    exchanges: &[Arc<dyn Exchange>],
    min_profit_pct: f64,
) -> Option<Opportunity> {
    // Fetch all order books concurrently.
    let mut order_books = Vec::new();
    for ex in exchanges {
        match ex.fetch_order_book(symbol).await {
            Ok(ob) => order_books.push((ex.name().to_string(), ob, ex.taker_fee())),
            Err(e) => {
                debug!("Could not fetch order book [{}/{}]: {}", ex.name(), symbol, e);
            }
        }
    }

    if order_books.len() < 2 {
        return None;
    }

    let mut best: Option<Opportunity> = None;

    for i in 0..order_books.len() {
        for j in 0..order_books.len() {
            if i == j {
                continue;
            }
            let (buy_name, buy_ob, buy_fee) = &order_books[i];
            let (sell_name, sell_ob, sell_fee) = &order_books[j];

            let ask = buy_ob.best_ask()?;
            let bid = sell_ob.best_bid()?;

            let profit = calc_profit_pct(ask, bid, *buy_fee, *sell_fee);
            if profit >= min_profit_pct {
                let replace = best.as_ref().map_or(true, |b| profit > b.profit_pct);
                if replace {
                    best = Some(Opportunity {
                        symbol: symbol.to_string(),
                        buy_exchange: buy_name.clone(),
                        sell_exchange: sell_name.clone(),
                        buy_price: ask,
                        sell_price: bid,
                        profit_pct: profit,
                    });
                }
            }
        }
    }

    best
}
