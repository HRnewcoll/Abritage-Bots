"""
Tests for nn_arb — Neural Network Arbitrage Bots
=================================================
All tests use synthetic data.  No network calls, no API keys required.

Covers:
  - neural_net.py : FeedForwardNet (forward shape, backprop reduces loss,
                    weight save/load), LSTMPredictor (step shape, BPTT
                    reduces loss, weight save/load), Adam optimiser.
  - dqn_agent.py  : ReplayBuffer, DQNAgent (action shape, training step,
                    target-net sync, compute_reward).
  - lstm_predictor.py : SpreadLSTMPredictor (observe, train, predict API).

Run with:
    cd python
    python -m pytest tests/test_nn_arb.py -v
"""

from __future__ import annotations

import sys
import os
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_arb.neural_net import FeedForwardNet, LSTMPredictor, _relu, _sigmoid, _AdamState
from nn_arb.dqn_agent import (
    DQNAgent, ReplayBuffer, Transition, compute_reward,
    ACTION_HOLD, ACTION_ENTER, ACTION_EXIT, N_ACTIONS, STATE_DIM,
)
from nn_arb.lstm_predictor import SpreadLSTMPredictor
from simulator.market_data import SimOrderBook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_books(n: int = 3, price: float = 40_000.0, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    books = []
    for i in range(n):
        drift = rng.uniform(-0.005, 0.005)
        mid   = price * (1.0 + drift)
        half  = mid * 0.0005
        books.append(SimOrderBook(
            exchange=f"ex_{i}",
            symbol="BTC/USDT",
            timestamp=1_700_000_000_000,
            bid=round(mid - half, 4),
            ask=round(mid + half, 4),
            bid_qty=float(rng.uniform(0.5, 5.0)),
            ask_qty=float(rng.uniform(0.5, 5.0)),
        ))
    return books


def _rand_state(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(STATE_DIM).astype(float)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class TestActivations:
    def test_relu_positive(self):
        x = np.array([-2.0, 0.0, 3.0])
        r = _relu(x)
        assert r[0] == 0.0
        assert r[1] == 0.0
        assert r[2] == pytest.approx(3.0)

    def test_sigmoid_range(self):
        x = np.linspace(-10, 10, 100)
        s = _sigmoid(x)
        assert np.all(s > 0.0) and np.all(s < 1.0)

    def test_sigmoid_zero(self):
        assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_sigmoid_large_neg(self):
        """Should not produce NaN or overflow for very negative inputs."""
        s = _sigmoid(np.array([-500.0]))
        assert np.isfinite(s[0])


# ---------------------------------------------------------------------------
# Adam optimiser
# ---------------------------------------------------------------------------

class TestAdam:
    def test_update_reduces_param(self):
        adam = _AdamState(lr=0.01)
        p = np.array([1.0, 2.0])
        g = np.array([1.0, 1.0])
        p_new = adam.step(p, g)
        # After one step in direction of gradient, params should decrease
        assert np.all(p_new < p)

    def test_zero_grad_leaves_param_near_unchanged(self):
        adam = _AdamState(lr=0.01)
        p = np.array([1.0, 2.0])
        g = np.zeros(2)
        p_new = adam.step(p, g)
        # With zero gradient the numerator is 0; result equals p
        assert np.allclose(p_new, p, atol=1e-6)

    def test_step_counter_increments(self):
        adam = _AdamState()
        adam.step(np.ones(3), np.ones(3))
        assert adam.t == 1
        adam.step(np.ones(3), np.ones(3))
        assert adam.t == 2


# ---------------------------------------------------------------------------
# FeedForwardNet
# ---------------------------------------------------------------------------

class TestFeedForwardNet:
    def test_output_shape_scalar(self):
        net = FeedForwardNet([4, 8, 1])
        x = np.ones(4)
        out, _ = net.forward(x)
        assert out.shape == (1,)

    def test_output_shape_vector(self):
        net = FeedForwardNet([8, 64, 64, 3])
        x = np.ones(8)
        out, _ = net.forward(x)
        assert out.shape == (3,)

    def test_predict_matches_forward(self):
        net = FeedForwardNet([5, 16, 2], seed=7)
        x = np.array([1.0, -1.0, 0.5, 0.0, 2.0])
        out_fwd, _ = net.forward(x)
        out_pred   = net.predict(x)
        assert np.allclose(out_fwd, out_pred)

    def test_backward_reduces_loss(self):
        """After 50 gradient steps, MSE loss should decrease."""
        rng = np.random.default_rng(0)
        net = FeedForwardNet([4, 16, 1], lr=5e-3, seed=0)
        x = rng.standard_normal(4)
        y = np.array([1.0])

        losses = []
        for _ in range(50):
            out, cache = net.forward(x)
            loss = float(np.mean((out - y) ** 2))
            losses.append(loss)
            dL = 2.0 * (out - y)
            net.backward_and_update(cache, dL)

        assert losses[-1] < losses[0], "Loss should decrease over 50 steps"

    def test_get_set_weights(self):
        net1 = FeedForwardNet([3, 8, 2], seed=1)
        net2 = FeedForwardNet([3, 8, 2], seed=99)
        x = np.ones(3)
        before = net2.predict(x).copy()
        net2.set_weights(net1.get_weights())
        after = net2.predict(x)
        assert np.allclose(net1.predict(x), after)
        assert not np.allclose(before, after)

    def test_no_nan_in_output(self):
        net = FeedForwardNet([8, 64, 64, 3], seed=42)
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.standard_normal(8)
            out, _ = net.forward(x)
            assert np.all(np.isfinite(out)), "NaN/Inf in network output"

    def test_invalid_layer_sizes(self):
        with pytest.raises(ValueError):
            FeedForwardNet([5])


# ---------------------------------------------------------------------------
# LSTMPredictor
# ---------------------------------------------------------------------------

class TestLSTMPredictor:
    def test_forward_returns_scalar(self):
        lstm = LSTMPredictor(input_size=3, hidden_size=8, seq_len=5)
        X = np.random.default_rng(0).standard_normal((5, 3))
        y_pred, cache, h, c = lstm.forward_sequence(X)
        assert isinstance(y_pred, float)

    def test_predict_returns_scalar(self):
        lstm = LSTMPredictor(input_size=3, hidden_size=8, seq_len=5)
        X = np.random.default_rng(1).standard_normal((5, 3))
        pred = lstm.predict(X)
        assert isinstance(pred, float)
        assert np.isfinite(pred)

    def test_train_step_returns_float(self):
        lstm = LSTMPredictor(input_size=3, hidden_size=8, seq_len=5)
        X = np.random.default_rng(2).standard_normal((5, 3))
        loss = lstm.train_step(X, y=1.0)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_training_reduces_loss(self):
        """Loss should decrease after many training steps on the same sample."""
        rng = np.random.default_rng(7)
        lstm = LSTMPredictor(input_size=4, hidden_size=16, seq_len=10, lr=1e-2, seed=0)
        X = rng.standard_normal((10, 4))
        y = 0.5

        losses = [lstm.train_step(X, y) for _ in range(100)]
        assert losses[-1] < losses[0], f"Loss should drop: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_no_nan_after_training(self):
        lstm = LSTMPredictor(input_size=3, hidden_size=8, seq_len=5, seed=5)
        rng = np.random.default_rng(0)
        for _ in range(30):
            X = rng.standard_normal((5, 3))
            lstm.train_step(X, y=rng.uniform(-1, 1))
        pred = lstm.predict(rng.standard_normal((5, 3)))
        assert np.isfinite(pred), f"Prediction became NaN/Inf after training"

    def test_get_set_weights(self):
        lstm1 = LSTMPredictor(input_size=2, hidden_size=4, seq_len=3, seed=10)
        lstm2 = LSTMPredictor(input_size=2, hidden_size=4, seq_len=3, seed=99)
        X = np.ones((3, 2))
        pred_before = lstm2.predict(X)
        lstm2.set_weights(lstm1.get_weights())
        pred_after = lstm2.predict(X)
        assert np.isclose(lstm1.predict(X), pred_after)
        assert not np.isclose(pred_before, pred_after)

    def test_weight_keys(self):
        lstm = LSTMPredictor(input_size=2, hidden_size=4, seq_len=3)
        keys = set(lstm.get_weights().keys())
        assert {"Wf", "Wi", "Wg", "Wo", "bf", "bi", "bg", "bo", "Wd", "bd"} <= keys

    def test_save_load(self):
        lstm1 = LSTMPredictor(input_size=3, hidden_size=8, seq_len=5, seed=0)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 3))
        for _ in range(5):
            lstm1.train_step(X, 0.3)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lstm.npz")
            np.savez(path, **lstm1.get_weights())
            lstm2 = LSTMPredictor(input_size=3, hidden_size=8, seq_len=5, seed=99)
            lstm2.set_weights(dict(np.load(path)))
            assert np.isclose(lstm1.predict(X), lstm2.predict(X))


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def _transition(self, seed: int = 0) -> Transition:
        rng = np.random.default_rng(seed)
        return Transition(
            state=rng.standard_normal(STATE_DIM),
            action=0,
            reward=0.0,
            next_state=rng.standard_normal(STATE_DIM),
            done=False,
        )

    def test_len_increases(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(self._transition(i))
        assert len(buf) == 10

    def test_capacity_cap(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(20):
            buf.push(self._transition(i))
        assert len(buf) == 5

    def test_sample_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(self._transition(i))
        s = buf.sample(20)
        assert len(s) == 20

    def test_sample_smaller_than_buffer(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(5):
            buf.push(self._transition(i))
        s = buf.sample(20)
        assert len(s) == 5  # capped to buffer size


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_enter_flat_small_negative(self):
        r, reason = compute_reward(ACTION_ENTER, 0, fee_pct=0.2)
        assert r < 0.0
        assert "fee" in reason

    def test_exit_profitable(self):
        r, reason = compute_reward(ACTION_EXIT, 1, realised_pnl_pct=1.0, fee_pct=0.2)
        assert r > 0.0

    def test_exit_unprofitable(self):
        r, _ = compute_reward(ACTION_EXIT, 1, realised_pnl_pct=-1.0, fee_pct=0.2)
        assert r < 0.0

    def test_invalid_enter_when_in_trade(self):
        r, reason = compute_reward(ACTION_ENTER, 1)
        assert r == pytest.approx(-0.5)
        assert "invalid" in reason

    def test_invalid_exit_when_flat(self):
        r, reason = compute_reward(ACTION_EXIT, 0)
        assert r == pytest.approx(-0.5)
        assert "invalid" in reason

    def test_hold_flat_zero(self):
        r, _ = compute_reward(ACTION_HOLD, 0)
        assert r == pytest.approx(0.0)

    def test_hold_in_trade_small_negative(self):
        r, _ = compute_reward(ACTION_HOLD, 1)
        assert r < 0.0


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

class TestDQNAgent:
    def test_action_in_valid_range(self):
        agent = DQNAgent(seed=0)
        for i in range(20):
            s = _rand_state(i)
            a = agent.select_action(s)
            assert 0 <= a < N_ACTIONS

    def test_greedy_action_valid(self):
        agent = DQNAgent(seed=0)
        s = _rand_state()
        a = agent.select_action(s, explore=False)
        assert 0 <= a < N_ACTIONS

    def test_train_step_returns_none_when_empty(self):
        agent = DQNAgent(batch_size=32, seed=0)
        assert agent.train_step() is None

    def test_train_step_returns_float_when_filled(self):
        agent = DQNAgent(batch_size=4, seed=0)
        rng = np.random.default_rng(0)
        for i in range(8):
            s  = rng.standard_normal(STATE_DIM)
            ns = rng.standard_normal(STATE_DIM)
            agent.store(s, 0, 0.0, ns)
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_epsilon_decays(self):
        agent = DQNAgent(eps_start=1.0, eps_end=0.05, eps_decay=100, batch_size=4, seed=0)
        rng = np.random.default_rng(0)
        for i in range(200):
            s  = rng.standard_normal(STATE_DIM)
            ns = rng.standard_normal(STATE_DIM)
            agent.store(s, 0, 0.0, ns)
            agent.train_step()
        assert agent.epsilon < 1.0

    def test_target_syncs_on_construction(self):
        agent = DQNAgent(seed=42)
        # Both networks should produce same outputs right after construction
        s = _rand_state()
        q_online = agent.online.predict(s)
        q_target = agent.target.predict(s)
        assert np.allclose(q_online, q_target)

    def test_save_load(self):
        agent1 = DQNAgent(seed=0, hidden_sizes=[16, 16])
        rng = np.random.default_rng(0)
        for i in range(10):
            s  = rng.standard_normal(STATE_DIM)
            ns = rng.standard_normal(STATE_DIM)
            agent1.store(s, 0, 0.0, ns)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn")
            agent1.save(path)
            agent2 = DQNAgent(seed=99, hidden_sizes=[16, 16])
            agent2.load(path + ".npz")
            s = _rand_state(5)
            assert np.allclose(agent1.online.predict(s), agent2.online.predict(s))


# ---------------------------------------------------------------------------
# SpreadLSTMPredictor
# ---------------------------------------------------------------------------

class TestSpreadLSTMPredictor:
    def test_observe_returns_feature_vector(self):
        pred = SpreadLSTMPredictor(seq_len=5)
        books = _make_books(3)
        feat = pred.observe(books)
        assert feat is not None
        assert feat.shape == (6,)
        assert np.all(np.isfinite(feat))

    def test_observe_single_book_returns_none(self):
        pred = SpreadLSTMPredictor(seq_len=5)
        feat = pred.observe(_make_books(1))
        assert feat is None

    def test_cannot_predict_before_window(self):
        pred = SpreadLSTMPredictor(seq_len=10)
        books = _make_books(3)
        for _ in range(5):
            pred.observe(books)
        assert pred.predict() is None

    def test_can_predict_after_window(self):
        pred = SpreadLSTMPredictor(seq_len=5)
        books = _make_books(3)
        for _ in range(6):
            pred.observe(books)
        val = pred.predict()
        assert val is not None
        assert np.isfinite(val)

    def test_cannot_train_before_seq_len_plus_one(self):
        pred = SpreadLSTMPredictor(seq_len=10)
        books = _make_books(3)
        for _ in range(10):
            pred.observe(books)
        assert not pred.can_train()

    def test_can_train_after_window_filled(self):
        pred = SpreadLSTMPredictor(seq_len=5)
        books = _make_books(3)
        for _ in range(7):
            pred.observe(books)
        assert pred.can_train()

    def test_train_returns_finite_loss(self):
        pred = SpreadLSTMPredictor(seq_len=5)
        books = _make_books(3)
        for _ in range(7):
            pred.observe(books)
        loss = pred.train()
        assert np.isfinite(loss)

    def test_n_trained_increments(self):
        pred = SpreadLSTMPredictor(seq_len=5)
        books = _make_books(3)
        for _ in range(10):
            pred.observe(books)
            if pred.can_train():
                pred.train()
        assert pred.n_trained > 0

    def test_save_load(self):
        pred1 = SpreadLSTMPredictor(seq_len=5, seed=0)
        books = _make_books(3)
        for _ in range(10):
            pred1.observe(books)
            if pred1.can_train():
                pred1.train()

        val_before = pred1.predict()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lstm_spread")
            pred1.save(path)
            pred2 = SpreadLSTMPredictor(seq_len=5, seed=99)
            # Give pred2 the same window
            for _ in range(10):
                pred2.observe(books)
            pred2.load(path + ".npz")
            # After loading weights, their model parameters should match
            # (predictions may differ slightly due to different window state,
            #  but the underlying model should be the same)
            w1 = pred1._model.get_weights()
            w2 = pred2._model.get_weights()
            for key in w1:
                assert np.allclose(w1[key], w2[key]), f"Weight mismatch for {key}"
