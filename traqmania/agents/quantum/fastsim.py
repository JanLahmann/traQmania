"""Batched numpy statevector simulator for the canonical traQmania circuit.

Implements exactly the canonical circuit (see ``circuit.py`` for the Qiskit
twin): per block ``l`` an RY data-(re-)uploading sublayer with trainable input
scaling (angle ``lam[l, i] * s[i]``), an RY/RZ variational sublayer, and a CZ
ring; readout is ``E_a = <Z_a>``.

Conventions (identical to Qiskit): ``RY(phi) = exp(-i phi Y / 2)``,
``RZ(phi) = exp(-i phi Z / 2)``, and bit ``i`` of a statevector index is qubit
``i`` (little-endian).

numpy/stdlib only — no qiskit imports in this module.
"""

from __future__ import annotations

import numpy as np


def apply_ry(state: np.ndarray, qubit: int, angle, n_qubits: int) -> None:
    """Apply RY(angle) on ``qubit`` in place. state (B, 2^n); angle scalar or (B,)."""
    half = 0.5 * np.asarray(angle, dtype=np.float64)
    c = np.cos(half)
    sn = np.sin(half)
    if c.ndim == 1:  # per-sample angles broadcast over the reduced axes
        c = c[:, None, None]
        sn = sn[:, None, None]
    view = state.reshape(state.shape[0], 1 << (n_qubits - 1 - qubit), 2, 1 << qubit)
    a0 = view[:, :, 0, :]
    a1 = view[:, :, 1, :]
    new0 = c * a0 - sn * a1
    new1 = sn * a0 + c * a1
    view[:, :, 0, :] = new0
    view[:, :, 1, :] = new1


def apply_rz(state: np.ndarray, qubit: int, angle, n_qubits: int) -> None:
    """Apply RZ(angle) on ``qubit`` in place. state (B, 2^n); angle scalar or (B,)."""
    half = 0.5 * np.asarray(angle, dtype=np.float64)
    p0 = np.exp(-1j * half)
    p1 = np.exp(1j * half)
    if p0.ndim == 1:
        p0 = p0[:, None, None]
        p1 = p1[:, None, None]
    view = state.reshape(state.shape[0], 1 << (n_qubits - 1 - qubit), 2, 1 << qubit)
    view[:, :, 0, :] *= p0
    view[:, :, 1, :] *= p1


def cz_ring_diagonal(n_qubits: int) -> np.ndarray:
    """Diagonal (2^n,) of the CZ ring CZ(0,1) CZ(1,2) ... CZ(n-1,0), entries +-1."""
    idx = np.arange(1 << n_qubits)
    sign = np.ones(1 << n_qubits, dtype=np.float64)
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        both = ((idx >> i) & 1) * ((idx >> j) & 1)
        sign *= 1.0 - 2.0 * both
    return sign


def z_diagonals(n_qubits: int) -> np.ndarray:
    """(n, 2^n) diagonals of Z_a (Pauli Z on qubit a), entries +-1."""
    idx = np.arange(1 << n_qubits)
    diags = np.empty((n_qubits, 1 << n_qubits), dtype=np.float64)
    for a in range(n_qubits):
        diags[a] = 1.0 - 2.0 * ((idx >> a) & 1)
    return diags


class FastStatevectorSim:
    """Batched float64/complex128 statevector simulation of the canonical circuit."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 4):
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.dim = 1 << self.n_qubits
        self.ring_diagonal = cz_ring_diagonal(self.n_qubits)
        self.z_diags = z_diagonals(self.n_qubits)

    def forward(
        self,
        s: np.ndarray,
        lam: np.ndarray,
        theta: np.ndarray,
        return_state: bool = False,
    ):
        """Run the circuit for a batch of feature vectors.

        s (B, n) float in [0, 1]; lam (L, n); theta (L, n, 2).
        Returns E (B, n) with E[b, a] = <Z_a>, plus the final statevectors
        (B, 2^n) complex128 when ``return_state`` is True.
        """
        n, layers = self.n_qubits, self.n_layers
        s = np.asarray(s, dtype=np.float64)
        lam = np.asarray(lam, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != n:
            raise ValueError(f"s must have shape (B, {n}), got {s.shape}")
        if lam.shape != (layers, n):
            raise ValueError(f"lam must have shape ({layers}, {n}), got {lam.shape}")
        if theta.shape != (layers, n, 2):
            raise ValueError(f"theta must have shape ({layers}, {n}, 2), got {theta.shape}")

        batch = s.shape[0]
        psi = np.zeros((batch, self.dim), dtype=np.complex128)
        psi[:, 0] = 1.0
        for layer in range(layers):
            for i in range(n):  # data (re-)uploading with trainable scaling
                apply_ry(psi, i, lam[layer, i] * s[:, i], n)
            for i in range(n):  # variational sublayer
                apply_ry(psi, i, theta[layer, i, 0], n)
                apply_rz(psi, i, theta[layer, i, 1], n)
            psi *= self.ring_diagonal  # CZ ring (diagonal, commuting)

        probs = psi.real**2 + psi.imag**2
        expectations = probs @ self.z_diags.T
        if return_state:
            return expectations, psi
        return expectations
