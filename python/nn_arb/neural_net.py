"""
neural_net.py — Pure-NumPy neural network building blocks
==========================================================

No PyTorch / TensorFlow required.  All operations use NumPy linear algebra
so the code is transparent, auditable, and has zero heavy dependencies.

Implements
----------
FeedForwardNet   Configurable multi-layer perceptron (MLP) with ReLU hidden
                 activations and a linear output.  Used by the DQN agent to
                 approximate Q(s, a).

LSTMPredictor    Single-layer LSTM followed by a linear dense head for
                 single-step regression.  Used by the spread predictor.
                 Trained with full Backpropagation Through Time (BPTT).

Both models share the Adam optimiser for gradient updates.

References
----------
- Kingma & Ba (2014) Adam: https://arxiv.org/abs/1412.6980
- Hochreiter & Schmidhuber (1997) Long Short-Term Memory.
  Neural Computation 9(8), 1735-1780.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


# ---------------------------------------------------------------------------
# Adam optimiser state (per parameter tensor)
# ---------------------------------------------------------------------------

@dataclass
class _AdamState:
    """Tracks the first and second moment estimates for one parameter tensor."""
    lr:      float = 1e-3
    beta1:   float = 0.9
    beta2:   float = 0.999
    eps:     float = 1e-8
    t:       int   = 0
    m:       Optional[np.ndarray] = field(default=None, repr=False)
    v:       Optional[np.ndarray] = field(default=None, repr=False)

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return updated parameter value."""
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad ** 2
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# FeedForwardNet — Multi-Layer Perceptron
# ---------------------------------------------------------------------------

class FeedForwardNet:
    """
    Configurable multi-layer perceptron.

    Architecture: Linear → ReLU (×n_hidden) → Linear (output, no activation).

    Parameters
    ----------
    layer_sizes : e.g. [8, 64, 64, 3]  — first is input, last is output.
    lr          : Adam learning rate (default 1e-3).
    seed        : For reproducible weight initialisation.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input + output dimensions.")
        self.layer_sizes = layer_sizes
        rng = np.random.default_rng(seed)

        self._W: List[np.ndarray] = []
        self._b: List[np.ndarray] = []
        self._adam_W: List[_AdamState] = []
        self._adam_b: List[_AdamState] = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialisation (good for ReLU)
            scale = np.sqrt(2.0 / fan_in)
            self._W.append(rng.standard_normal((fan_out, fan_in)) * scale)
            self._b.append(np.zeros(fan_out))
            self._adam_W.append(_AdamState(lr=lr))
            self._adam_b.append(_AdamState(lr=lr))

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Forward pass.

        Returns
        -------
        output  : Final layer output (linear, no activation).
        cache   : List of (z, a, x_in) per layer — needed for backprop.
        """
        cache = []
        a = x.copy()
        for l_idx, (W, b) in enumerate(zip(self._W, self._b)):
            x_in = a.copy()
            z = W @ a + b
            # Last layer is linear; all others use ReLU
            a = z if l_idx == len(self._W) - 1 else _relu(z)
            cache.append((z, a, x_in))
        return a, cache

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict without storing cache (inference only)."""
        a = x.copy()
        for l_idx, (W, b) in enumerate(zip(self._W, self._b)):
            z = W @ a + b
            a = z if l_idx == len(self._W) - 1 else _relu(z)
        return a

    # ── Backward pass + update ───────────────────────────────────────────────

    def backward_and_update(
        self,
        cache: list,
        dL_dout: np.ndarray,
        grad_clip: float = 5.0,
    ) -> None:
        """
        Backpropagate gradients and apply Adam updates.

        Parameters
        ----------
        cache     : From forward().
        dL_dout   : Gradient of loss w.r.t. network output.
        grad_clip : Gradient clipping threshold (L-inf norm per layer).
        """
        delta = dL_dout.copy()
        for l_idx in reversed(range(len(self._W))):
            z, a, x_in = cache[l_idx]
            W = self._W[l_idx]

            # Gradient through activation (ReLU, except last layer is linear)
            if l_idx < len(self._W) - 1:
                delta = delta * _relu_grad(z)

            # Weight / bias gradients
            dW = np.outer(delta, x_in)
            db = delta.copy()

            # Gradient to pass to previous layer
            delta = W.T @ delta

            # Clip
            np.clip(dW, -grad_clip, grad_clip, out=dW)
            np.clip(db, -grad_clip, grad_clip, out=db)

            # Adam update
            self._W[l_idx] = self._adam_W[l_idx].step(W, dW)
            self._b[l_idx] = self._adam_b[l_idx].step(self._b[l_idx], db)

    # ── Serialisation ────────────────────────────────────────────────────────

    def get_weights(self) -> dict:
        return {"W": [w.copy() for w in self._W],
                "b": [b.copy() for b in self._b]}

    def set_weights(self, weights: dict) -> None:
        self._W = [w.copy() for w in weights["W"]]
        self._b = [b.copy() for b in weights["b"]]


# ---------------------------------------------------------------------------
# LSTMPredictor — Single-layer LSTM + linear dense head
# ---------------------------------------------------------------------------

class LSTMPredictor:
    """
    Sequence-to-scalar regressor using a single LSTM layer plus a linear
    output head.

    Trained with full Backpropagation Through Time (BPTT) over a fixed-length
    input window.  Suitable for predicting the *next* spread value given the
    last ``seq_len`` feature vectors.

    Parameters
    ----------
    input_size  : Number of features per timestep.
    hidden_size : LSTM hidden state dimension.
    seq_len     : Training/inference sequence length (window).
    lr          : Adam learning rate.
    seed        : Reproducible initialisation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        seq_len: int = 20,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.seq_len     = seq_len
        self._h = np.zeros(hidden_size)
        self._c = np.zeros(hidden_size)

        rng = np.random.default_rng(seed)
        xh = input_size + hidden_size  # concatenated input size

        # LSTM weight matrices — one per gate (forget, input, gate, output)
        # Shape: (hidden_size, xh)
        scale = np.sqrt(2.0 / xh)
        self.Wf = rng.standard_normal((hidden_size, xh)) * scale
        self.Wi = rng.standard_normal((hidden_size, xh)) * scale
        self.Wg = rng.standard_normal((hidden_size, xh)) * scale
        self.Wo = rng.standard_normal((hidden_size, xh)) * scale

        # Initialise forget gate bias to 1 (standard trick for longer memory)
        self.bf = np.ones(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bg = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)

        # Dense output head: hidden_size → 1
        self.Wd = rng.standard_normal((1, hidden_size)) * np.sqrt(2.0 / hidden_size)
        self.bd = np.zeros(1)

        # Adam states for every parameter
        _mk = lambda: _AdamState(lr=lr)
        self._aWf, self._aWi, self._aWg, self._aWo = _mk(), _mk(), _mk(), _mk()
        self._abf, self._abi, self._abg, self._abo = _mk(), _mk(), _mk(), _mk()
        self._aWd, self._abd = _mk(), _mk()

    # ── Single LSTM step ────────────────────────────────────────────────────

    def _lstm_step(
        self,
        x: np.ndarray,
        h: np.ndarray,
        c: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """One LSTM timestep.  Returns (h_new, c_new, gate_cache)."""
        xh = np.concatenate([x, h])  # (input_size + hidden_size,)
        f  = _sigmoid(self.Wf @ xh + self.bf)
        i  = _sigmoid(self.Wi @ xh + self.bi)
        g  = _tanh(self.Wg @ xh + self.bg)
        o  = _sigmoid(self.Wo @ xh + self.bo)
        c_new = f * c + i * g
        h_new = o * _tanh(c_new)
        return h_new, c_new, (f, i, g, o, xh, c.copy())

    # ── Forward pass over sequence ──────────────────────────────────────────

    def forward_sequence(
        self,
        X: np.ndarray,
    ) -> Tuple[float, list]:
        """
        Forward pass over a sequence X of shape (T, input_size).

        Returns (y_pred, step_cache) where y_pred is the scalar prediction
        from the final timestep's hidden state and step_cache stores all
        per-step information needed for BPTT.
        """
        h, c = np.zeros(self.hidden_size), np.zeros(self.hidden_size)
        step_cache = []
        for x_t in X:
            h, c, gate_cache = self._lstm_step(x_t, h, c)
            step_cache.append((h.copy(), c.copy(), gate_cache))
        y_pred = float((self.Wd @ h + self.bd)[0])
        return y_pred, step_cache, h, c

    def predict(self, X: np.ndarray) -> float:
        """Inference: feed sequence through LSTM, return predicted scalar."""
        h, c = np.zeros(self.hidden_size), np.zeros(self.hidden_size)
        for x_t in X:
            h, c, _ = self._lstm_step(x_t, h, c)
        return float((self.Wd @ h + self.bd)[0])

    # ── BPTT training step ──────────────────────────────────────────────────

    def train_step(
        self,
        X: np.ndarray,
        y: float,
        grad_clip: float = 5.0,
    ) -> float:
        """
        One BPTT update: forward X (T, input_size) → predict y → MSE loss
        → backprop → Adam update of all parameters.

        Returns the scalar training loss.
        """
        T = len(X)
        y_pred, step_cache, h_final, _ = self.forward_sequence(X)

        # MSE loss: L = (y_pred - y)^2
        loss = (y_pred - y) ** 2

        # ── Output layer backward ─────────────────────────────────────────
        dL_dyhat = 2.0 * (y_pred - y)              # scalar
        dW_d = dL_dyhat * h_final.reshape(1, -1)
        db_d = np.array([dL_dyhat])
        dL_dh = (self.Wd * dL_dyhat).squeeze()     # (hidden_size,)

        # ── Accumulate LSTM gradients (BPTT) ──────────────────────────────
        dW_f = np.zeros_like(self.Wf)
        dW_i = np.zeros_like(self.Wi)
        dW_g = np.zeros_like(self.Wg)
        dW_o = np.zeros_like(self.Wo)
        db_f = np.zeros_like(self.bf)
        db_i = np.zeros_like(self.bi)
        db_g = np.zeros_like(self.bg)
        db_o = np.zeros_like(self.bo)

        dL_dc = np.zeros(self.hidden_size)

        for t in reversed(range(T)):
            h_t, c_t, (f, i, g, o, xh, c_prev) = step_cache[t]

            # h_t = o * tanh(c_t)
            tanh_ct = _tanh(c_t)
            dL_do_pre   = dL_dh * tanh_ct
            dL_dc      += dL_dh * o * (1.0 - tanh_ct ** 2)

            # c_t = f*c_prev + i*g
            dL_df_pre  = dL_dc * c_prev
            dL_di_pre  = dL_dc * g
            dL_dg_pre  = dL_dc * i
            dL_dc_prev  = dL_dc * f

            # Pre-activation gradients (through gate activations)
            df_raw = dL_df_pre * f * (1.0 - f)   # sigmoid'
            di_raw = dL_di_pre * i * (1.0 - i)   # sigmoid'
            dg_raw = dL_dg_pre * (1.0 - g ** 2)  # tanh'
            do_raw = dL_do_pre * o * (1.0 - o)   # sigmoid'

            # Accumulate weight gradients: dL/dW_x = outer(delta, xh)
            dW_f += np.outer(df_raw, xh)
            dW_i += np.outer(di_raw, xh)
            dW_g += np.outer(dg_raw, xh)
            dW_o += np.outer(do_raw, xh)
            db_f += df_raw
            db_i += di_raw
            db_g += dg_raw
            db_o += do_raw

            # Gradient to previous hidden state (last hidden_size cols of xh)
            dxh = (self.Wf.T @ df_raw + self.Wi.T @ di_raw
                   + self.Wg.T @ dg_raw + self.Wo.T @ do_raw)
            dL_dh  = dxh[self.input_size:]   # grad wrt h_{t-1}
            dL_dc  = dL_dc_prev

        # Clip all gradients
        for g_arr in [dW_f, dW_i, dW_g, dW_o, dW_d,
                      db_f, db_i, db_g, db_o, db_d]:
            np.clip(g_arr, -grad_clip, grad_clip, out=g_arr)

        # Adam updates
        self.Wf = self._aWf.step(self.Wf, dW_f)
        self.Wi = self._aWi.step(self.Wi, dW_i)
        self.Wg = self._aWg.step(self.Wg, dW_g)
        self.Wo = self._aWo.step(self.Wo, dW_o)
        self.bf = self._abf.step(self.bf, db_f)
        self.bi = self._abi.step(self.bi, db_i)
        self.bg = self._abg.step(self.bg, db_g)
        self.bo = self._abo.step(self.bo, db_o)
        self.Wd = self._aWd.step(self.Wd, dW_d)
        self.bd = self._abd.step(self.bd, db_d)

        return float(loss)

    # ── Serialisation ────────────────────────────────────────────────────────

    def get_weights(self) -> dict:
        return {k: getattr(self, k).copy()
                for k in ("Wf", "Wi", "Wg", "Wo", "bf", "bi", "bg", "bo", "Wd", "bd")}

    def set_weights(self, weights: dict) -> None:
        for k, v in weights.items():
            setattr(self, k, v.copy())
