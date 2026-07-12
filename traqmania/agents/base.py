"""Shared agent interfaces: the discrete action set and the QFunction protocol.

Every Q-function backend (classical MLP, quantum circuit, ...) implements the
same numpy-facing contract so the training loop in ``agents.training`` can be
reused unchanged across backends.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

N_ACTIONS = 4

# (steer, throttle, brake): steer +1 increases theta = counterclockwise = a LEFT
# turn on screen, so this is right/straight/left at full throttle, plus coast-brake.
ACTIONS = ((-1, 1, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1))

# Scaled action sets: at n_actions > 4 the circuit reads Q_a = <Z_a> off the
# first n_actions qubits instead of the first four.  Every larger set keeps
# the smaller one as a PREFIX, so action indices (and existing 4-action
# weights) keep their meaning.  6 actions add trail braking — full steer
# WHILE braking; the 4-action set can only brake straight, so a hairpin
# entry must alternate brake/steer decisions and loses steering authority
# exactly where it matters.  8 actions add half-steer at full throttle for
# smoother lines with less speed scrub.  A pure coast action was considered
# and left out: with drag 0.35 coasting barely decelerates, so it is
# dominated by brake and throttle almost everywhere.
_FULL_ACTIONS = ACTIONS + ((-1, 0, 1), (1, 0, 1), (-0.5, 1, 0), (0.5, 1, 0))
_FULL_LABELS = ("Right", "Straight", "Left", "Brake",
                "Brake right", "Brake left", "Half right", "Half left")
ACTION_SIZES = (4, 6, 8)


def action_set(n_actions: int = N_ACTIONS) -> tuple[tuple[float, float, float], ...]:
    """The (steer, throttle, brake) table for an ``n_actions``-sized action set."""
    n_actions = int(n_actions)
    if n_actions not in ACTION_SIZES:
        raise ValueError(f"n_actions must be one of {ACTION_SIZES}, got {n_actions}")
    return _FULL_ACTIONS[:n_actions]


def action_labels(n_actions: int = N_ACTIONS) -> tuple[str, ...]:
    """Display label per action, matching :func:`action_set` order."""
    return _FULL_LABELS[: len(action_set(n_actions))]


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
