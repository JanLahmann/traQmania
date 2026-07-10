"""Shared agent interfaces: the discrete action set and the QFunction protocol.

Every Q-function backend (classical MLP, quantum circuit, ...) implements the
same numpy-facing contract so the training loop in ``agents.training`` can be
reused unchanged across backends.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

N_ACTIONS = 4

# (steer, throttle, brake): left/straight/right at full throttle, plus coast-brake.
ACTIONS = ((-1, 1, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1))


@runtime_checkable
class QFunction(Protocol):
    """Batched, differentiable state-action value function over flat parameters."""

    n_features: int
    n_actions: int

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q-values for a batch of observations: (B, F) -> (B, A)."""
        ...

    def grad_selected(
        self, obs: np.ndarray, action_idx: np.ndarray, upstream: np.ndarray
    ) -> np.ndarray:
        """Gradient of ``sum_b upstream[b] * Q[b, action_idx[b]]`` wrt the flat params.

        obs (B, F) float; action_idx (B,) int; upstream (B,) float. Returns (P,).
        """
        ...

    def get_params(self) -> np.ndarray:
        """Copy of the flat parameter vector, shape (P,)."""
        ...

    def set_params(self, params: np.ndarray) -> None:
        """Load a flat parameter vector, shape (P,)."""
        ...
