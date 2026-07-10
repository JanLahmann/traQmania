"""A tiny fully-connected Q-network in pure numpy: tanh hidden layer, linear output.

Parameters live in ONE flat float64 vector; the layer matrices (W1, b1, W2, b2)
are views into it, so ``set_params`` is a single in-place copy and gradients are
returned in exactly the same flat layout.
"""

from __future__ import annotations

import numpy as np


class MLPQFunction:
    """Single-hidden-layer MLP implementing the :class:`~traqmania.agents.base.QFunction`
    contract, including analytic backprop for ``grad_selected``."""

    def __init__(
        self, n_features: int = 4, hidden: int = 8, n_actions: int = 4, seed: int = 0
    ) -> None:
        self.n_features = n_features
        self.hidden = hidden
        self.n_actions = n_actions

        f, h, a = n_features, hidden, n_actions
        self.n_params = f * h + h + h * a + a
        self._flat = np.zeros(self.n_params, dtype=np.float64)

        # Layer views into the flat vector (reshaped slices share memory).
        o = 0
        self.W1 = self._flat[o : o + f * h].reshape(f, h)
        o += f * h
        self.b1 = self._flat[o : o + h]
        o += h
        self.W2 = self._flat[o : o + h * a].reshape(h, a)
        o += h * a
        self.b2 = self._flat[o : o + a]

        rng = np.random.default_rng(seed)
        self.W1[:] = rng.normal(0.0, 1.0 / np.sqrt(f), size=(f, h))
        self.W2[:] = rng.normal(0.0, 1.0 / np.sqrt(h), size=(h, a))
        # Biases start at zero.

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """(B, F) -> (B, A)."""
        obs = np.asarray(obs, dtype=np.float64)
        hidden = np.tanh(obs @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2

    def grad_selected(
        self, obs: np.ndarray, action_idx: np.ndarray, upstream: np.ndarray
    ) -> np.ndarray:
        """Gradient of ``sum_b upstream[b] * Q[b, action_idx[b]]`` wrt the flat params."""
        obs = np.asarray(obs, dtype=np.float64)
        action_idx = np.asarray(action_idx, dtype=np.intp)
        upstream = np.asarray(upstream, dtype=np.float64)

        batch = obs.shape[0]
        hidden = np.tanh(obs @ self.W1 + self.b1)  # (B, H)

        d_q = np.zeros((batch, self.n_actions))  # dL/dQ
        d_q[np.arange(batch), action_idx] = upstream

        g_w2 = hidden.T @ d_q  # (H, A)
        g_b2 = d_q.sum(axis=0)  # (A,)
        d_hidden = d_q @ self.W2.T  # (B, H)
        d_pre = d_hidden * (1.0 - hidden**2)  # tanh'
        g_w1 = obs.T @ d_pre  # (F, H)
        g_b1 = d_pre.sum(axis=0)  # (H,)

        return np.concatenate([g_w1.ravel(), g_b1, g_w2.ravel(), g_b2])

    def get_params(self) -> np.ndarray:
        return self._flat.copy()

    def set_params(self, params: np.ndarray) -> None:
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"expected {self.n_params} params, got shape {params.shape}")
        self._flat[:] = params
