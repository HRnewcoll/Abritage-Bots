"""
Shared exchange utility helpers used across all bots.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import ccxt
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config, falling back to environment-variable overrides."""
    if not os.path.exists(path):
        logger.warning("Config file %s not found – using environment variables only.", path)
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def build_exchange(name: str, cfg: dict, sandbox: Optional[bool] = None) -> ccxt.Exchange:
    """
    Instantiate a ccxt exchange object from configuration.

    Args:
        name:    ccxt exchange id (e.g. 'binance', 'kraken').
        cfg:     The full config dict (output of load_config).
        sandbox: Override the sandbox flag from config when provided.

    Returns:
        Configured ccxt.Exchange instance.
    """
    exchange_cfg: dict = cfg.get("exchanges", {}).get(name, {})

    params: Dict[str, str] = {}
    api_key = exchange_cfg.get("api_key") or os.getenv(f"{name.upper()}_API_KEY", "")
    api_secret = exchange_cfg.get("api_secret") or os.getenv(f"{name.upper()}_API_SECRET", "")

    if api_key:
        params["apiKey"] = api_key
    if api_secret:
        params["secret"] = api_secret

    # KuCoin / OKX require a passphrase
    passphrase = exchange_cfg.get("passphrase") or os.getenv(f"{name.upper()}_PASSPHRASE", "")
    if passphrase:
        params["password"] = passphrase

    exchange_class = getattr(ccxt, name)
    exchange: ccxt.Exchange = exchange_class(params)

    use_sandbox = sandbox if sandbox is not None else exchange_cfg.get("sandbox", True)
    if use_sandbox and exchange.has.get("sandbox"):
        exchange.set_sandbox_mode(True)
        logger.info("[%s] Sandbox/testnet mode ENABLED.", name)

    exchange.load_markets()
    return exchange


def fetch_ticker(exchange: ccxt.Exchange, symbol: str) -> Optional[dict]:
    """Fetch ticker safely, returning None on error."""
    try:
        return exchange.fetch_ticker(symbol)
    except Exception as exc:
        logger.warning("[%s] Could not fetch ticker for %s: %s", exchange.id, symbol, exc)
        return None


def fetch_order_book(exchange: ccxt.Exchange, symbol: str, limit: int = 5) -> Optional[dict]:
    """Fetch order book safely, returning None on error."""
    try:
        return exchange.fetch_order_book(symbol, limit)
    except Exception as exc:
        logger.warning("[%s] Could not fetch order book for %s: %s", exchange.id, symbol, exc)
        return None
