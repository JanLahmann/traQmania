"""Quantum Q-function on the numpy fast path, implementing the QFunction protocol.

Flat parameter vector layout (total P = L*n + L*n*2 + 2*A):

    lam   (L, n)     input-scaling params, init pi
    theta (L, n, 2)  variational angles (RY then RZ), init U(-0.1, 0.1)
    w     (A,)       output weights, init 1.0
    b     (A,)       output biases, init 0.0

Q_a = w[a] * <Z_a> + b[a]. numpy/stdlib only — no qiskit imports in this module.
"""

from __future__ import annotations

import numpy as np

from traqmania.agents.quantum import adjoint
from traqmania.agents.quantum.fastsim import FastStatevectorSim


class QuantumQFunction:
    """Batched, differentiable Q-function backed by ``FastStatevectorSim``."""

    def __init__(self, circuit_cfg: dict, seed: int | None = None):
        cfg = circuit_cfg.get("circuit", circuit_cfg)
        self.n_qubits = int(cfg.get("n_qubits", 4))
        self.n_layers = int(cfg.get("n_layers", 4))
        if seed is None:
            seed = int(cfg.get("seed", 7))
        self.seed = int(seed)

        self.n_features = self.n_qubits
        self.n_actions = self.n_qubits  # one Z_a readout per action

        self._sim = FastStatevectorSim(n_qubits=self.n_qubits, n_layers=self.n_layers)

        rng = np.random.default_rng(self.seed)
        self.lam = np.full((self.n_layers, self.n_qubits), np.pi, dtype=np.float64)
        self.theta = rng.uniform(-0.1, 0.1, size=(self.n_layers, self.n_qubits, 2))
        self.w = np.ones(self.n_actions, dtype=np.float64)
        self.b = np.zeros(self.n_actions, dtype=np.float64)

    @property
    def n_params(self) -> int:
        return self.lam.size + self.theta.size + self.w.size + self.b.size

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q-values for a batch of observations: (B, F) -> (B, A)."""
        return self.expectations(obs) * self.w + self.b

    def expectations(self, obs: np.ndarray) -> np.ndarray:
        """Raw readout expectations <Z_a> before the output head: (B, F) -> (B, A)."""
        obs = np.asarray(obs, dtype=np.float64)
        return self._sim.forward(obs, self.lam, self.theta)

    def grad_selected(
        self, obs: np.ndarray, action_idx: np.ndarray, upstream: np.ndarray
    ) -> np.ndarray:
        """Gradient of ``sum_b upstream[b] * Q[b, action_idx[b]]`` wrt the flat params."""
        obs = np.asarray(obs, dtype=np.float64)
        action_idx = np.asarray(action_idx, dtype=np.int64)
        upstream = np.asarray(upstream, dtype=np.float64)

        expectations, psi = self._sim.forward(obs, self.lam, self.theta, return_state=True)
        # Chain rule through the output head: dQ_sel/dE_sel = w[a_b].
        dlam, dtheta = adjoint.grad(
            obs, self.lam, self.theta, action_idx, upstream * self.w[action_idx], psi_final=psi
        )
        dw = np.zeros(self.n_actions, dtype=np.float64)
        db = np.zeros(self.n_actions, dtype=np.float64)
        e_selected = expectations[np.arange(obs.shape[0]), action_idx]
        np.add.at(dw, action_idx, upstream * e_selected)
        np.add.at(db, action_idx, upstream)
        return np.concatenate([dlam.ravel(), dtheta.ravel(), dw, db])

    def get_params(self) -> np.ndarray:
        """Copy of the flat parameter vector, shape (P,)."""
        return np.concatenate([self.lam.ravel(), self.theta.ravel(), self.w, self.b])

    def set_params(self, params: np.ndarray) -> None:
        """Load a flat parameter vector, shape (P,)."""
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"params must have shape ({self.n_params},), got {params.shape}")
        n_lam = self.lam.size
        n_theta = self.theta.size
        n_w = self.w.size
        self.lam = params[:n_lam].reshape(self.lam.shape).copy()
        self.theta = params[n_lam : n_lam + n_theta].reshape(self.theta.shape).copy()
        self.w = params[n_lam + n_theta : n_lam + n_theta + n_w].copy()
        self.b = params[n_lam + n_theta + n_w :].copy()
