"""
nn_arb — Neural Network Arbitrage Bots
========================================
Pure-NumPy neural network bots — no PyTorch/TensorFlow required.

Modules
-------
neural_net       Core building blocks: MLP, LSTM, Adam optimiser (all in NumPy).
lstm_predictor   LSTM spread predictor: predicts next inter-exchange spread.
dqn_agent        Deep Q-Network agent: learns optimal entry/exit policy via RL.
bot              Simulation runner / live-trading CLI (wraps both models).
"""
