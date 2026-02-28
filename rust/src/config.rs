use std::collections::HashMap;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct ExchangeConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub api_secret: String,
    /// Used by KuCoin / OKX clients.
    #[serde(default)]
    #[allow(dead_code)]
    pub passphrase: String,
    #[serde(default = "default_true")]
    pub sandbox: bool,
}

#[derive(Debug, Deserialize)]
pub struct ArbitrageConfig {
    #[serde(default = "default_symbols")]
    pub symbols: Vec<String>,
    #[serde(default = "default_min_profit")]
    pub min_profit_pct: f64,
    #[serde(default = "default_trade_size")]
    pub max_trade_size_usdt: f64,
    #[serde(default = "default_poll_interval")]
    pub poll_interval_seconds: u64,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
pub struct BotConfig {
    #[serde(default)]
    pub exchanges: HashMap<String, ExchangeConfig>,
    pub arbitrage: ArbitrageConfig,
}

// Serde defaults
fn default_true() -> bool { true }
fn default_symbols() -> Vec<String> {
    vec!["BTC/USDT".into(), "ETH/USDT".into()]
}
fn default_min_profit() -> f64 { 0.5 }
fn default_trade_size() -> f64 { 100.0 }
fn default_poll_interval() -> u64 { 5 }

impl BotConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let cfg: BotConfig = serde_yaml::from_str(&content)?;
        Ok(cfg)
    }
}
