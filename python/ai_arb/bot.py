"""
AI Arbitrage Bot
================
Two AI-powered strategies combined in one bot:

1. **Spread Predictor** (supervised ML)
   Trains a Random-Forest regressor on historical OHLCV data from two
   exchanges to predict the near-future price spread.  A trade is opened
   when the predicted spread exceeds the configured threshold.

2. **RL Trading Agent** (reinforcement learning, Q-learning)
   Learns, through simulated episodes on historical data, the optimal
   policy for entering and exiting arbitrage positions.  The Q-table is
   persisted to disk and updated continuously as new data arrives.

Both strategies share the same exchange connectivity layer and config file.

Usage
-----
    # Train/update models and run:
    python -m ai_arb.bot --config config/config.yaml --strategy spread
    python -m ai_arb.bot --config config/config.yaml --strategy rl
    python -m ai_arb.bot --config config/config.yaml --strategy both   (default)

Always start with  dry_run: true  in config until you have verified the bot
is operating as expected on your account.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import colorlog
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "..")
from utils.exchange import build_exchange, fetch_order_book, load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_handler = colorlog.StreamHandler()
_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logging.basicConfig(handlers=[_handler], level=logging.INFO)
logger = logging.getLogger("ai_arb")


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _rsi(prices: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index for the last `period` observations."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = deltas[deltas > 0].mean() if (deltas > 0).any() else 0.0
    losses = -deltas[deltas < 0].mean() if (deltas < 0).any() else 0.0
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(prices: np.ndarray, span: int) -> float:
    """Exponential moving average over the last `span` observations."""
    if len(prices) == 0:
        return 0.0
    alpha = 2.0 / (span + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema


def build_features(
    prices_a: np.ndarray,  # close prices from exchange A
    prices_b: np.ndarray,  # close prices from exchange B
    window: int = 20,
) -> Optional[np.ndarray]:
    """
    Build a 1-D feature vector from recent OHLCV data.
    Returns None if not enough data.
    """
    n = min(len(prices_a), len(prices_b))
    if n < window + 1:
        return None

    pa = prices_a[-window:]
    pb = prices_b[-window:]

    spread = pa - pb
    spread_mean = spread.mean()
    spread_std = spread.std() + 1e-9
    spread_z = (spread[-1] - spread_mean) / spread_std

    features = [
        spread[-1],                       # current spread
        spread_mean,                       # mean spread over window
        spread_std,                        # spread volatility
        spread_z,                          # z-score of current spread
        _rsi(pa),                          # RSI of exchange A
        _rsi(pb),                          # RSI of exchange B
        _ema(pa, 10),                      # EMA-10 of exchange A
        _ema(pb, 10),                      # EMA-10 of exchange B
        pa[-1] / (pa.mean() + 1e-9),       # price ratio A vs its mean
        pb[-1] / (pb.mean() + 1e-9),       # price ratio B vs its mean
        (pa[-1] - pa[0]) / (pa[0] + 1e-9), # return over window A
        (pb[-1] - pb[0]) / (pb[0] + 1e-9), # return over window B
        float(np.corrcoef(pa, pb)[0, 1]),  # correlation
    ]
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Spread Predictor (supervised ML)
# ---------------------------------------------------------------------------

class SpreadPredictor:
    """
    Trains a Gradient-Boosting regressor to predict the *next-step*
    spread between two exchanges for a given symbol.
    """

    def __init__(self, model_dir: str = "models/", window: int = 20) -> None:
        self.window = window
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self._model_path = os.path.join(model_dir, "spread_predictor.pkl")
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if os.path.exists(self._model_path):
            try:
                with open(self._model_path, "rb") as fh:
                    data = pickle.load(fh)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.is_trained = True
                logger.info("SpreadPredictor: loaded existing model from %s", self._model_path)
            except Exception as exc:
                logger.warning("Could not load saved model: %s", exc)

    def _save(self) -> None:
        with open(self._model_path, "wb") as fh:
            pickle.dump({"model": self.model, "scaler": self.scaler}, fh)
        logger.info("SpreadPredictor: model saved to %s", self._model_path)

    # ------------------------------------------------------------------
    def train(self, prices_a: np.ndarray, prices_b: np.ndarray) -> None:
        """
        Build training dataset from historical price arrays and fit the model.
        """
        n = min(len(prices_a), len(prices_b))
        if n < self.window + 20:
            logger.warning("SpreadPredictor: not enough data to train (%d samples).", n)
            return

        X, y = [], []
        for i in range(self.window, n - 1):
            feats = build_features(prices_a[: i + 1], prices_b[: i + 1], self.window)
            if feats is None:
                continue
            # Target: next-step spread
            target = float(prices_a[i + 1] - prices_b[i + 1])
            X.append(feats)
            y.append(target)

        if not X:
            return

        X_arr = np.array(X)
        y_arr = np.array(y)

        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled, y_arr)
        self.is_trained = True
        logger.info("SpreadPredictor: trained on %d samples.", len(X))
        self._save()

    # ------------------------------------------------------------------
    def predict(self, prices_a: np.ndarray, prices_b: np.ndarray) -> Optional[float]:
        """Predict next-step spread.  Returns None if not ready."""
        if not self.is_trained:
            return None
        feats = build_features(prices_a, prices_b, self.window)
        if feats is None:
            return None
        scaled = self.scaler.transform(feats.reshape(1, -1))
        return float(self.model.predict(scaled)[0])


# ---------------------------------------------------------------------------
# RL Trading Agent (tabular Q-learning)
# ---------------------------------------------------------------------------

class RLTradingAgent:
    """
    Tabular Q-learning agent that learns when to enter / exit arbitrage
    positions.

    State space   : (spread_bucket, position) — discretised spread z-score
    Action space  : 0 = hold, 1 = open long spread, 2 = close position
    Reward        : realised PnL per step (normalised)
    """

    ACTIONS = [0, 1, 2]  # hold, open, close

    def __init__(
        self,
        model_dir: str = "models/",
        n_buckets: int = 20,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
    ) -> None:
        self.n_buckets = n_buckets
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self._q_path = os.path.join(model_dir, "rl_q_table.pkl")
        # Q-table: defaultdict[(spread_bucket, position)] → [q0, q1, q2]
        self.Q: Dict[Tuple, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if os.path.exists(self._q_path):
            try:
                with open(self._q_path, "rb") as fh:
                    self.Q = pickle.load(fh)
                logger.info("RLAgent: loaded Q-table from %s (%d states).", self._q_path, len(self.Q))
            except Exception as exc:
                logger.warning("Could not load Q-table: %s", exc)

    def _save(self) -> None:
        with open(self._q_path, "wb") as fh:
            pickle.dump(dict(self.Q), fh)

    # ------------------------------------------------------------------
    def _discretise(self, spread_z: float) -> int:
        """Clip and bin spread z-score into n_buckets discrete levels."""
        clipped = max(-3.0, min(3.0, spread_z))
        bucket = int((clipped + 3.0) / 6.0 * (self.n_buckets - 1))
        return bucket

    def _state(self, spread_z: float, position: int) -> Tuple[int, int]:
        return (self._discretise(spread_z), position)

    # ------------------------------------------------------------------
    def select_action(self, spread_z: float, position: int) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        state = self._state(spread_z, position)
        return int(np.argmax(self.Q[state]))

    def update(
        self,
        spread_z: float,
        position: int,
        action: int,
        reward: float,
        next_spread_z: float,
        next_position: int,
    ) -> None:
        """Bellman update."""
        s = self._state(spread_z, position)
        s_next = self._state(next_spread_z, next_position)
        q_max_next = max(self.Q[s_next])
        td_target = reward + self.gamma * q_max_next
        self.Q[s][action] += self.alpha * (td_target - self.Q[s][action])

    # ------------------------------------------------------------------
    def train_episode(
        self,
        spreads: np.ndarray,
        episode_steps: int = 200,
    ) -> float:
        """
        Run one simulated episode on historical spread data.
        Returns total episode reward.
        """
        n = len(spreads)
        if n < 10:
            return 0.0

        spread_mean = spreads.mean()
        spread_std = spreads.std() + 1e-9

        position = 0   # 0 = flat, 1 = long spread
        entry_spread = 0.0
        total_reward = 0.0
        start = random.randint(0, max(0, n - episode_steps - 1))

        for t in range(start, min(start + episode_steps, n - 1)):
            z = (spreads[t] - spread_mean) / spread_std
            action = self.select_action(z, position)

            # Simulate action
            reward = 0.0
            if action == 1 and position == 0:
                # Open long spread position
                position = 1
                entry_spread = spreads[t]
            elif action == 2 and position == 1:
                # Close position
                pnl = (spreads[t] - entry_spread) / (abs(entry_spread) + 1e-9)
                reward = pnl - 0.001  # subtract fee estimate
                position = 0

            # Small penalty for holding open position (capital cost)
            if position == 1:
                reward -= 0.0001

            next_z = (spreads[t + 1] - spread_mean) / spread_std
            self.update(z, position, action, reward, next_z, position)
            total_reward += reward

        self._save()
        return total_reward


# ---------------------------------------------------------------------------
# Main AI Arbitrage Bot
# ---------------------------------------------------------------------------

class AIArbitrageBot:
    """
    Orchestrates the Spread Predictor and RL Agent against live market data.
    """

    def __init__(self, config: dict, strategy: str = "both") -> None:
        self.cfg = config
        self.ai_cfg = config.get("ai", {})
        self.arb_cfg = config.get("arbitrage", {})
        self.dry_run: bool = self.ai_cfg.get("dry_run", True)
        self.strategy = strategy
        self.model_dir: str = self.ai_cfg.get("model_dir", "models/")
        self.history_candles: int = int(self.ai_cfg.get("history_candles", 500))
        self.feature_window: int = int(self.ai_cfg.get("feature_window", 20))
        self.pred_threshold: float = float(self.ai_cfg.get("prediction_threshold", 0.4))
        self.rl_episode_steps: int = int(self.ai_cfg.get("rl_episode_steps", 200))
        self.poll_interval: int = int(self.arb_cfg.get("poll_interval_seconds", 10))
        self.symbols: List[str] = self.arb_cfg.get("symbols", ["BTC/USDT"])
        self.max_trade_usdt: float = float(self.arb_cfg.get("max_trade_size_usdt", 100.0))

        # Build two exchanges (first two in config)
        exchange_names = list(config.get("exchanges", {}).keys())
        if len(exchange_names) < 2:
            raise ValueError("AI bot needs at least 2 exchanges in config.")

        self.ex_a_name = exchange_names[0]
        self.ex_b_name = exchange_names[1]
        self.ex_a = build_exchange(self.ex_a_name, config)
        self.ex_b = build_exchange(self.ex_b_name, config)
        logger.info("AI Bot: using exchanges %s and %s", self.ex_a_name, self.ex_b_name)

        # Initialise models
        self.predictor = SpreadPredictor(self.model_dir, self.feature_window)
        self.rl_agent = RLTradingAgent(self.model_dir)

        # Live price buffers
        self._prices_a: Dict[str, List[float]] = defaultdict(list)
        self._prices_b: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    def _fetch_history(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch OHLCV history from both exchanges and return close arrays."""
        def _ohlcv(exchange, sym, limit):
            try:
                data = exchange.fetch_ohlcv(sym, timeframe="1m", limit=limit)
                return np.array([c[4] for c in data], dtype=np.float64)
            except Exception as exc:
                logger.warning("OHLCV fetch failed [%s/%s]: %s", exchange.id, sym, exc)
                return np.array([], dtype=np.float64)

        pa = _ohlcv(self.ex_a, symbol, self.history_candles)
        pb = _ohlcv(self.ex_b, symbol, self.history_candles)
        return pa, pb

    # ------------------------------------------------------------------
    def _live_prices(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """Append current mid-price to the rolling price buffers."""
        def _mid(exchange, sym):
            try:
                ob = exchange.fetch_order_book(sym, 1)
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if bids and asks:
                    return (bids[0][0] + asks[0][0]) / 2.0
            except Exception:
                pass
            return None

        pa_mid = _mid(self.ex_a, symbol)
        pb_mid = _mid(self.ex_b, symbol)

        if pa_mid:
            self._prices_a[symbol].append(pa_mid)
        if pb_mid:
            self._prices_b[symbol].append(pb_mid)

        # Keep only last history_candles prices in memory
        keep = self.history_candles
        self._prices_a[symbol] = self._prices_a[symbol][-keep:]
        self._prices_b[symbol] = self._prices_b[symbol][-keep:]

        return (
            np.array(self._prices_a[symbol], dtype=np.float64),
            np.array(self._prices_b[symbol], dtype=np.float64),
        )

    # ------------------------------------------------------------------
    def _execute_arb(
        self,
        symbol: str,
        buy_exchange,
        sell_exchange,
        buy_price: float,
        sell_price: float,
    ) -> None:
        amount = self.max_trade_usdt / buy_price
        logger.info(
            "AI EXECUTE: Buy %.6f %s @ %.4f on %s | Sell @ %.4f on %s",
            amount, symbol, buy_price, buy_exchange.id, sell_price, sell_exchange.id,
        )
        try:
            buy_exchange.create_market_buy_order(symbol, amount)
            sell_exchange.create_market_sell_order(symbol, amount)
        except Exception as exc:
            logger.error("Order execution failed: %s", exc)

    # ------------------------------------------------------------------
    def _run_spread_strategy(self, symbol: str, pa: np.ndarray, pb: np.ndarray) -> None:
        """Use the ML spread predictor to decide whether to trade."""
        predicted_spread = self.predictor.predict(pa, pb)
        if predicted_spread is None:
            logger.debug("[spread] Model not ready for %s", symbol)
            return

        current_price_a = float(pa[-1]) if len(pa) else 0.0
        current_price_b = float(pb[-1]) if len(pb) else 0.0
        spread_pct = (predicted_spread / (current_price_a + 1e-9)) * 100.0

        logger.info(
            "[spread] %s | Predicted spread: %.4f (%.4f%%) | Threshold: %.4f%%",
            symbol, predicted_spread, spread_pct, self.pred_threshold,
        )

        if abs(spread_pct) >= self.pred_threshold:
            if current_price_a < current_price_b:
                buy_ex, sell_ex = self.ex_a, self.ex_b
                buy_p, sell_p = current_price_a, current_price_b
            else:
                buy_ex, sell_ex = self.ex_b, self.ex_a
                buy_p, sell_p = current_price_b, current_price_a

            logger.info(
                "[spread] SIGNAL: Buy on %s @ %.4f | Sell on %s @ %.4f",
                buy_ex.id, buy_p, sell_ex.id, sell_p,
            )
            if not self.dry_run:
                self._execute_arb(symbol, buy_ex, sell_ex, buy_p, sell_p)

    # ------------------------------------------------------------------
    def _run_rl_strategy(self, symbol: str, pa: np.ndarray, pb: np.ndarray) -> None:
        """Use the RL agent to decide whether to trade."""
        n = min(len(pa), len(pb))
        if n < 10:
            logger.debug("[rl] Not enough data for %s", symbol)
            return

        spreads = pa[-n:] - pb[-n:]
        spread_mean = spreads.mean()
        spread_std = spreads.std() + 1e-9
        current_z = float((spreads[-1] - spread_mean) / spread_std)

        # Greedy action (epsilon=0 in production)
        action = self.rl_agent.select_action(current_z, position=0)

        action_names = {0: "HOLD", 1: "OPEN", 2: "CLOSE"}
        logger.info(
            "[rl] %s | Spread z-score: %.4f | RL action: %s",
            symbol, current_z, action_names.get(action, "?"),
        )

        if action == 1:
            if pa[-1] < pb[-1]:
                buy_ex, sell_ex = self.ex_a, self.ex_b
                buy_p, sell_p = float(pa[-1]), float(pb[-1])
            else:
                buy_ex, sell_ex = self.ex_b, self.ex_a
                buy_p, sell_p = float(pb[-1]), float(pa[-1])

            logger.info(
                "[rl] SIGNAL: Buy on %s @ %.4f | Sell on %s @ %.4f",
                buy_ex.id, buy_p, sell_ex.id, sell_p,
            )
            if not self.dry_run:
                self._execute_arb(symbol, buy_ex, sell_ex, buy_p, sell_p)

    # ------------------------------------------------------------------
    def _train_models(self, symbol: str) -> None:
        """Fetch historical data and retrain/update both models."""
        logger.info("Training models on historical data for %s …", symbol)
        pa, pb = self._fetch_history(symbol)
        if len(pa) == 0 or len(pb) == 0:
            logger.warning("No historical data available for %s", symbol)
            return

        # Seed the rolling buffers with history
        self._prices_a[symbol] = list(pa)
        self._prices_b[symbol] = list(pb)

        # Train spread predictor
        self.predictor.train(pa, pb)

        # Train RL agent for multiple episodes
        n_min = min(len(pa), len(pb))
        spreads = pa[-n_min:] - pb[-n_min:]
        total_reward = 0.0
        for ep in range(5):
            r = self.rl_agent.train_episode(spreads, self.rl_episode_steps)
            total_reward += r
        logger.info("RL training complete for %s. Avg episode reward: %.4f", symbol, total_reward / 5)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main loop: train on first run, then continuously poll."""
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        logger.info("=== AI Arbitrage Bot started [%s] | Strategy: %s ===", mode, self.strategy)

        # Initial training pass
        for symbol in self.symbols:
            self._train_models(symbol)

        retrain_counter = 0
        retrain_every = 100  # retrain every N poll cycles

        while True:
            for symbol in self.symbols:
                pa, pb = self._live_prices(symbol)

                if self.strategy in ("spread", "both"):
                    self._run_spread_strategy(symbol, pa, pb)

                if self.strategy in ("rl", "both"):
                    self._run_rl_strategy(symbol, pa, pb)

            retrain_counter += 1
            if retrain_counter >= retrain_every:
                retrain_counter = 0
                for symbol in self.symbols:
                    self._train_models(symbol)

            time.sleep(self.poll_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Powered Crypto Arbitrage Bot")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--strategy",
        choices=["spread", "rl", "both"],
        default="both",
        help="AI strategy to use: spread predictor, RL agent, or both (default: both)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    bot = AIArbitrageBot(cfg, strategy=args.strategy)
    bot.run()


if __name__ == "__main__":
    main()
