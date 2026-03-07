"""
dqn_agent.py — Deep Q-Network (DQN) arbitrage trading agent
=============================================================

Learns an optimal entry/exit policy from simulated market experience using
a Deep Q-Network — a neural network that approximates Q(state, action).

Architecture
------------
State space  (8 features):
  0  spread_pct         — current best inter-exchange spread (%)
  1  spread_ma10        — 10-step moving average of spread (%)
  2  mid_change_pct     — % change in average mid-price since last step
  3  volume_imbalance   — normalised volume difference between exchanges
  4  rsi                — RSI(14) of average mid-price  (0–100, scaled ÷100)
  5  position           — 0.0 = flat, 1.0 = in trade
  6  steps_held         — timesteps since trade opened (normalised ÷ 100)
  7  unrealised_pnl_pct — current trade's unrealised P&L as % of size (0 if flat)

Action space (3 discrete actions):
  0  HOLD   — do nothing
  1  ENTER  — open a cross-exchange arbitrage position
  2  EXIT   — close current position

Reward
------
  ENTER when flat:     r = −fee_pct  (pay the entry fee)
  EXIT  when in trade: r = realised_pnl_pct − fee_pct
  HOLD  in trade:      r = −0.001 (small holding-cost to discourage idle holds)
  HOLD  when flat:     r = 0.0
  Invalid action:      r = −0.5 (strong penalty)

Training
--------
Uses experience replay (random mini-batch from a ring buffer) and a target
network updated every ``target_update_freq`` steps.  Epsilon-greedy
exploration decays from ``eps_start`` to ``eps_end`` over ``eps_decay`` steps.

References
----------
Mnih et al. (2015) Human-level control through deep reinforcement learning.
Nature 518(7540): 529–533.  https://www.nature.com/articles/nature14236
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from nn_arb.neural_net import FeedForwardNet

# ── Constants ──────────────────────────────────────────────────────────────

STATE_DIM  = 8
N_ACTIONS  = 3   # HOLD=0, ENTER=1, EXIT=2

ACTION_HOLD  = 0
ACTION_ENTER = 1
ACTION_EXIT  = 2


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network trading agent.

    Parameters
    ----------
    lr                  : Adam learning rate.
    gamma               : Discount factor for future rewards.
    eps_start / eps_end : Epsilon-greedy exploration range.
    eps_decay           : Steps over which epsilon decays from start to end.
    batch_size          : Mini-batch size for each training step.
    target_update_freq  : Copy online → target network every N training steps.
    buffer_capacity     : Maximum transitions stored in the replay buffer.
    hidden_sizes        : Hidden layer widths in the Q-network.
    seed                : Reproducibility.
    """

    def __init__(
        self,
        lr:                 float = 1e-3,
        gamma:              float = 0.99,
        eps_start:          float = 1.0,
        eps_end:            float = 0.05,
        eps_decay:          int   = 2_000,
        batch_size:         int   = 64,
        target_update_freq: int   = 200,
        buffer_capacity:    int   = 10_000,
        hidden_sizes:       List[int] = None,
        seed:               int   = 42,
    ) -> None:
        hidden_sizes = hidden_sizes or [64, 64]
        layer_sizes = [STATE_DIM] + hidden_sizes + [N_ACTIONS]

        # Online network: trained every step
        self.online = FeedForwardNet(layer_sizes, lr=lr, seed=seed)
        # Target network: updated periodically, used to compute TD targets
        self.target = FeedForwardNet(layer_sizes, lr=lr, seed=seed + 1)
        self._sync_target()

        self.gamma              = gamma
        self.eps_start          = eps_start
        self.eps_end            = eps_end
        self.eps_decay          = eps_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq

        self._buffer  = ReplayBuffer(capacity=buffer_capacity)
        self._steps   = 0          # total training steps taken
        self._epsilon = eps_start

        random.seed(seed)
        np.random.seed(seed)

    # ── Epsilon schedule ────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def _decay_epsilon(self) -> None:
        frac = min(1.0, self._steps / max(1, self.eps_decay))
        self._epsilon = self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    # ── Action selection ────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state   : 1-D state array of length STATE_DIM.
        explore : If False, always act greedily (inference mode).
        """
        if explore and random.random() < self._epsilon:
            return random.randint(0, N_ACTIONS - 1)
        q = self.online.predict(state)
        return int(np.argmax(q))

    # ── Store experience ────────────────────────────────────────────────────

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool = False,
    ) -> None:
        """Add one transition to the replay buffer."""
        self._buffer.push(Transition(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
        ))

    # ── Training step ───────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient update on the online net.

        Returns the mean TD loss (or None if the buffer is too small).
        """
        if len(self._buffer) < self.batch_size:
            return None

        batch     = self._buffer.sample(self.batch_size)
        total_loss = 0.0

        for tr in batch:
            # TD target: r + γ * max_a' Q_target(s', a')
            q_next = self.target.predict(tr.next_state)
            td_target = tr.reward + (0.0 if tr.done else self.gamma * float(np.max(q_next)))

            # Online Q-values and loss gradient
            q_online, cache = self.online.forward(tr.state)
            td_error = q_online[tr.action] - td_target

            # dL/dq_online: only the chosen action's component is non-zero
            dL_dq = np.zeros(N_ACTIONS)
            dL_dq[tr.action] = 2.0 * td_error   # MSE gradient

            self.online.backward_and_update(cache, dL_dq)
            total_loss += td_error ** 2

        self._steps += 1
        self._decay_epsilon()

        # Periodically copy online → target
        if self._steps % self.target_update_freq == 0:
            self._sync_target()

        return total_loss / self.batch_size

    # ── Utilities ────────────────────────────────────────────────────────────

    def _sync_target(self) -> None:
        """Copy online network weights to target network."""
        self.target.set_weights(self.online.get_weights())

    def save(self, path: str) -> None:
        """Save online-network weights as a .npz file."""
        weights = self.online.get_weights()
        arrays = {}
        for l_idx, (W, b) in enumerate(zip(weights["W"], weights["b"])):
            arrays[f"W_{l_idx}"] = W
            arrays[f"b_{l_idx}"] = b
        np.savez(path, **arrays)

    def load(self, path: str) -> None:
        """Load online-network weights from a .npz file and sync target."""
        data = np.load(path)
        n_layers = len(self.online._W)
        weights = {
            "W": [data[f"W_{l}"] for l in range(n_layers)],
            "b": [data[f"b_{l}"] for l in range(n_layers)],
        }
        self.online.set_weights(weights)
        self._sync_target()


# ---------------------------------------------------------------------------
# Reward calculator (stateless helper)
# ---------------------------------------------------------------------------

def compute_reward(
    action:           int,
    position:         int,   # 0 = flat, 1 = in trade
    realised_pnl_pct: float = 0.0,
    fee_pct:          float = 0.2,   # 0.2% round-trip fee
) -> Tuple[float, str]:
    """
    Compute reward and a short human-readable explanation.

    Returns (reward, reason_string).
    """
    if action == ACTION_ENTER:
        if position == 1:
            return -0.5, "invalid_enter_already_in"
        return -fee_pct / 100.0, "enter_paid_fee"

    if action == ACTION_EXIT:
        if position == 0:
            return -0.5, "invalid_exit_not_in_trade"
        net = realised_pnl_pct - fee_pct
        return net / 100.0, f"exit_pnl={net:.4f}%"

    # HOLD
    if position == 1:
        return -0.001, "hold_in_trade_cost"
    return 0.0, "hold_flat"
