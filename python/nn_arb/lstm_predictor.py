"""
lstm_predictor.py — LSTM spread predictor with feature engineering
===================================================================

Wraps LSTMPredictor from neural_net.py in a higher-level class that:
  - Builds feature vectors from raw OHLCV candles / SimOrderBook snapshots.
  - Maintains a rolling input window.
  - Trains online as new data arrives.
  - Exposes a simple ``fit`` / ``predict_next`` API.

Features fed to the LSTM (6 per timestep)
------------------------------------------
  0  spread_pct       — best inter-exchange spread (ask_min → bid_max) as %
  1  spread_delta     — change in spread vs previous timestep
  2  mid_norm         — current average mid-price normalised by a rolling mean
                        (removes the absolute price scale)
  3  vol_imbalance    — signed volume difference between the two best books,
                        normalised by their sum
  4  rsi_scaled       — RSI(14) of the mid-price history, scaled to (−0.5, 0.5)
  5  spread_ma_ratio  — spread_pct / (rolling-20 mean of spread_pct), centred
                        around 1 — captures whether spread is high or low
                        relative to recent history

Output
------
  Predicted spread (%) for the next candle.

Usage
-----
    predictor = SpreadLSTMPredictor(seq_len=20)

    # Training (step-by-step, online):
    for books in stream_of_order_books:
        predictor.observe(books)         # add one timestep
        if predictor.can_train():
            loss = predictor.train()     # one BPTT update

    # Inference:
    pred_spread = predictor.predict()    # predicted next spread (%)
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from nn_arb.neural_net import LSTMPredictor

INPUT_SIZE = 6    # number of features per timestep


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _rsi(prices: np.ndarray, period: int = 14) -> float:
    """RSI over the last ``period`` observations."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    up   = deltas[deltas > 0]
    down = deltas[deltas < 0]
    avg_gain = up.mean()   if len(up)   > 0 else 0.0
    avg_loss = -down.mean() if len(down) > 0 else 0.0
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    return a / b if b != 0.0 else fallback


# ---------------------------------------------------------------------------
# SpreadLSTMPredictor
# ---------------------------------------------------------------------------

class SpreadLSTMPredictor:
    """
    High-level wrapper that couples feature engineering with LSTMPredictor.

    Parameters
    ----------
    seq_len     : Length of the input window (timesteps).
    hidden_size : LSTM hidden-state size.
    lr          : Adam learning rate.
    seed        : Reproducibility.
    """

    def __init__(
        self,
        seq_len:     int   = 20,
        hidden_size: int   = 32,
        lr:          float = 1e-3,
        seed:        int   = 0,
    ) -> None:
        self.seq_len  = seq_len
        self._model   = LSTMPredictor(
            input_size=INPUT_SIZE,
            hidden_size=hidden_size,
            seq_len=seq_len,
            lr=lr,
            seed=seed,
        )

        # Rolling histories for feature computation
        self._spread_hist:   Deque[float] = deque(maxlen=100)
        self._mid_hist:      Deque[float] = deque(maxlen=100)
        self._feature_window: Deque[np.ndarray] = deque(maxlen=seq_len + 1)

        self._prev_spread: float = 0.0
        self._n_trained:   int   = 0

    # ── Observation ─────────────────────────────────────────────────────────

    def observe(self, books: list) -> Optional[np.ndarray]:
        """
        Feed one step's order books into the predictor.

        books : list of SimOrderBook (at least 2 required for a spread).

        Returns the computed feature vector (or None if not enough books).
        """
        if len(books) < 2:
            return None

        best_ask = min(b.ask for b in books)
        best_bid = max(b.bid for b in books)
        avg_mid  = sum(b.mid for b in books) / len(books)

        spread_pct = _safe_div(best_bid - best_ask, best_ask) * 100.0

        # Volumes
        asks_sorted = sorted(books, key=lambda b: b.ask)
        bids_sorted = sorted(books, key=lambda b: b.bid, reverse=True)
        v_ask = asks_sorted[0].ask_qty
        v_bid = bids_sorted[0].bid_qty
        vol_imbalance = _safe_div(v_bid - v_ask, v_bid + v_ask + 1e-9)

        self._spread_hist.append(spread_pct)
        self._mid_hist.append(avg_mid)

        spread_arr = np.array(list(self._spread_hist))
        mid_arr    = np.array(list(self._mid_hist))

        spread_delta   = spread_pct - self._prev_spread
        self._prev_spread = spread_pct

        mid_mean = mid_arr.mean() if len(mid_arr) > 0 else avg_mid
        mid_norm = _safe_div(avg_mid, mid_mean, 1.0) - 1.0

        rsi_scaled = (_rsi(mid_arr) - 50.0) / 100.0

        spread_ma = spread_arr[-20:].mean() if len(spread_arr) >= 1 else spread_pct
        spread_ma_ratio = _safe_div(spread_pct, spread_ma, 1.0) - 1.0

        feat = np.array([
            spread_pct,
            spread_delta,
            mid_norm,
            vol_imbalance,
            rsi_scaled,
            spread_ma_ratio,
        ], dtype=float)

        self._feature_window.append(feat)
        return feat

    # ── Can we train? ────────────────────────────────────────────────────────

    def can_train(self) -> bool:
        """True once the feature window is full."""
        return len(self._feature_window) >= self.seq_len + 1

    def can_predict(self) -> bool:
        """True once the window holds at least seq_len features."""
        return len(self._feature_window) >= self.seq_len

    # ── Train one step ───────────────────────────────────────────────────────

    def train(self) -> float:
        """
        Perform one BPTT training step.

        Uses the first ``seq_len`` features as input and the LAST feature's
        spread_pct (index 0) as the regression target.

        Returns the training loss.
        """
        window = list(self._feature_window)
        X = np.array(window[:self.seq_len])         # (seq_len, INPUT_SIZE)
        y = float(window[self.seq_len][0])           # target = next spread_pct

        loss = self._model.train_step(X, y)
        self._n_trained += 1
        return loss

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self) -> Optional[float]:
        """
        Predict the next-step spread (%) using the last ``seq_len`` features.

        Returns None if the window is not yet full.
        """
        if not self.can_predict():
            return None
        window = list(self._feature_window)
        X = np.array(window[-self.seq_len:])         # (seq_len, INPUT_SIZE)
        return self._model.predict(X)

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save LSTM weights to a .npz file."""
        np.savez(path, **self._model.get_weights())

    def load(self, path: str) -> None:
        """Load LSTM weights from a .npz file."""
        data = np.load(path)
        self._model.set_weights(dict(data))

    @property
    def n_trained(self) -> int:
        return self._n_trained
