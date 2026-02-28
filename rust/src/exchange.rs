use anyhow::Result;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::config::ExchangeConfig;

/// A single level in an order book (price, quantity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level {
    pub price: Decimal,
    pub qty: Decimal,
}

/// Minimal order book snapshot.
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: Vec<Level>, // sorted descending
    pub asks: Vec<Level>, // sorted ascending
}

impl OrderBook {
    /// Best ask price (cheapest offer).
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.first().map(|l| l.price)
    }

    /// Best bid price (highest buy).
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.first().map(|l| l.price)
    }
}

/// Generic exchange client trait.
#[async_trait::async_trait]
pub trait Exchange: Send + Sync {
    fn name(&self) -> &str;
    async fn fetch_order_book(&self, symbol: &str) -> Result<OrderBook>;
    async fn place_market_buy(&self, symbol: &str, amount: Decimal) -> Result<String>;
    async fn place_market_sell(&self, symbol: &str, amount: Decimal) -> Result<String>;
    /// Taker fee as a decimal fraction (e.g. 0.001 for 0.1%).
    fn taker_fee(&self) -> Decimal;
}

// ---------------------------------------------------------------------------
// Generic REST exchange (works with Binance, Bybit, KuCoin, etc.)
// ---------------------------------------------------------------------------

/// Binance order-book response shape.
#[derive(Deserialize)]
struct BinanceOrderBook {
    bids: Vec<[String; 2]>,
    asks: Vec<[String; 2]>,
}

/// Bybit order-book response shape.
#[derive(Deserialize)]
struct BybitResult {
    b: Vec<[String; 2]>, // bids
    a: Vec<[String; 2]>, // asks
}

#[derive(Deserialize)]
struct BybitResponse {
    result: BybitResult,
}

/// Kraken order-book response (kept for future Kraken client).
#[allow(dead_code)]
#[derive(Deserialize)]
struct KrakenOrderBookInner {
    bids: Vec<(String, String, u64)>,
    asks: Vec<(String, String, u64)>,
}

fn parse_levels(raw: &[[String; 2]]) -> Vec<Level> {
    raw.iter()
        .filter_map(|pair| {
            let price = pair[0].parse::<Decimal>().ok()?;
            let qty = pair[1].parse::<Decimal>().ok()?;
            Some(Level { price, qty })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Binance Client
// ---------------------------------------------------------------------------

// Fields used when signing authenticated order requests (live mode).
#[allow(dead_code)]
pub struct BinanceClient {
    http: reqwest::Client,
    api_key: String,
    api_secret: String,
    sandbox: bool,
    fee: Decimal,
}

impl BinanceClient {
    pub fn new(cfg: &ExchangeConfig) -> Self {
        let http = reqwest::Client::builder()
            .use_rustls_tls()
            .build()
            .expect("Failed to build HTTP client");
        Self {
            http,
            api_key: cfg.api_key.clone(),
            api_secret: cfg.api_secret.clone(),
            sandbox: cfg.sandbox,
            fee: "0.001".parse().unwrap(),
        }
    }

    fn base_url(&self) -> &str {
        if self.sandbox {
            "https://testnet.binance.vision/api/v3"
        } else {
            "https://api.binance.com/api/v3"
        }
    }

    fn normalise_symbol(symbol: &str) -> String {
        symbol.replace('/', "")
    }
}

#[async_trait::async_trait]
impl Exchange for BinanceClient {
    fn name(&self) -> &str {
        "binance"
    }

    fn taker_fee(&self) -> Decimal {
        self.fee
    }

    async fn fetch_order_book(&self, symbol: &str) -> Result<OrderBook> {
        let sym = Self::normalise_symbol(symbol);
        let url = format!("{}/depth?symbol={}&limit=5", self.base_url(), sym);
        let resp: BinanceOrderBook = self.http.get(&url).send().await?.json().await?;
        let mut bids = parse_levels(&resp.bids);
        let mut asks = parse_levels(&resp.asks);
        bids.sort_by(|a, b| b.price.cmp(&a.price));
        asks.sort_by(|a, b| a.price.cmp(&b.price));
        Ok(OrderBook { bids, asks })
    }

    async fn place_market_buy(&self, symbol: &str, amount: Decimal) -> Result<String> {
        // In dry-run / testnet mode the bot calls this but we log instead.
        let sym = Self::normalise_symbol(symbol);
        tracing::info!("[binance] MARKET BUY {} qty={}", sym, amount);
        // Real implementation: sign request with api_key/api_secret and POST to /order
        Ok("dry-run-order-id".to_string())
    }

    async fn place_market_sell(&self, symbol: &str, amount: Decimal) -> Result<String> {
        let sym = Self::normalise_symbol(symbol);
        tracing::info!("[binance] MARKET SELL {} qty={}", sym, amount);
        Ok("dry-run-order-id".to_string())
    }
}

// ---------------------------------------------------------------------------
// Bybit Client
// ---------------------------------------------------------------------------

// Fields used when signing authenticated order requests (live mode).
#[allow(dead_code)]
pub struct BybitClient {
    http: reqwest::Client,
    api_key: String,
    api_secret: String,
    sandbox: bool,
    fee: Decimal,
}

impl BybitClient {
    pub fn new(cfg: &ExchangeConfig) -> Self {
        let http = reqwest::Client::builder()
            .use_rustls_tls()
            .build()
            .expect("Failed to build HTTP client");
        Self {
            http,
            api_key: cfg.api_key.clone(),
            api_secret: cfg.api_secret.clone(),
            sandbox: cfg.sandbox,
            fee: "0.001".parse().unwrap(),
        }
    }

    fn base_url(&self) -> &str {
        if self.sandbox {
            "https://api-testnet.bybit.com/v5"
        } else {
            "https://api.bybit.com/v5"
        }
    }
}

#[async_trait::async_trait]
impl Exchange for BybitClient {
    fn name(&self) -> &str {
        "bybit"
    }

    fn taker_fee(&self) -> Decimal {
        self.fee
    }

    async fn fetch_order_book(&self, symbol: &str) -> Result<OrderBook> {
        let sym = symbol.replace('/', "");
        let url = format!("{}/market/orderbook?category=spot&symbol={}&limit=5", self.base_url(), sym);
        let resp: BybitResponse = self.http.get(&url).send().await?.json().await?;
        let mut bids = parse_levels(&resp.result.b);
        let mut asks = parse_levels(&resp.result.a);
        bids.sort_by(|a, b| b.price.cmp(&a.price));
        asks.sort_by(|a, b| a.price.cmp(&b.price));
        Ok(OrderBook { bids, asks })
    }

    async fn place_market_buy(&self, symbol: &str, amount: Decimal) -> Result<String> {
        tracing::info!("[bybit] MARKET BUY {} qty={}", symbol, amount);
        Ok("dry-run-order-id".to_string())
    }

    async fn place_market_sell(&self, symbol: &str, amount: Decimal) -> Result<String> {
        tracing::info!("[bybit] MARKET SELL {} qty={}", symbol, amount);
        Ok("dry-run-order-id".to_string())
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Build an exchange client from config by name.
pub fn build_exchange(
    name: &str,
    cfg: &ExchangeConfig,
) -> Option<Box<dyn Exchange>> {
    match name {
        "binance" => Some(Box::new(BinanceClient::new(cfg))),
        "bybit" => Some(Box::new(BybitClient::new(cfg))),
        other => {
            warn!("Exchange '{}' not yet implemented in Rust bot – skipping.", other);
            None
        }
    }
}
