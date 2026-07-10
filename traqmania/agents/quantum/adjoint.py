"""Reverse-mode adjoint differentiation of the canonical circuit (numpy only).

Derivation (per sample). With ``psi_k = U_k ... U_1 |0>``, an observable ``O``
and ``E = <psi_N| O |psi_N>``, every parameterized gate is
``U_k = exp(-i * angle_k * G_k / 2)`` with Pauli generator ``G_k`` (Y for RY,
Z for RZ), so ``dU_k/dangle_k = (-i/2) G_k U_k`` and

    dE/dangle_k = 2 Re <phi_k| (-i/2) G_k |psi_k> = Im <phi_k| G_k |psi_k>,

where ``phi_k = U_{k+1}^dag ... U_N^dag O psi_N``. The backward sweep keeps
``(psi, phi)`` synchronized: starting from ``psi = psi_N`` and
``phi = O psi_N``, each step evaluates the gradient of the current gate and
then un-applies it to both vectors.

For the data-uploading gates the circuit angle is ``lam[l, i] * s[b, i]``, so
``dE/dlam[l, i] = s[b, i] * dE/dangle`` (chain rule).

numpy/stdlib only — no qiskit imports in this module.
"""

from __future__ import annotations

import numpy as np

from traqmania.agents.quantum.fastsim import FastStatevectorSim, apply_ry, apply_rz


def _pauli_expectation_im(
    phi: np.ndarray, psi: np.ndarray, qubit: int, n_qubits: int, generator: str
) -> np.ndarray:
    """Im(<phi_b| G |psi_b>) per sample, G = Pauli Y or Z on ``qubit``. Returns (B,)."""
    shape = (phi.shape[0], 1 << (n_qubits - 1 - qubit), 2, 1 << qubit)
    p = phi.reshape(shape)
    q = psi.reshape(shape)
    if generator == "y":  # Y|0> = i|1>, Y|1> = -i|0>
        inner = np.sum(np.conj(p[:, :, 0, :]) * (-1j) * q[:, :, 1, :], axis=(1, 2))
        inner += np.sum(np.conj(p[:, :, 1, :]) * 1j * q[:, :, 0, :], axis=(1, 2))
    elif generator == "z":
        inner = np.sum(np.conj(p[:, :, 0, :]) * q[:, :, 0, :], axis=(1, 2))
        inner -= np.sum(np.conj(p[:, :, 1, :]) * q[:, :, 1, :], axis=(1, 2))
    else:
        raise ValueError(f"unknown generator {generator!r}")
    return inner.imag


def grad(
    s: np.ndarray,
    lam: np.ndarray,
    theta: np.ndarray,
    obs_idx: np.ndarray,
    upstream: np.ndarray,
    psi_final: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of ``sum_b upstream[b] * <Z_{obs_idx[b]}>_b`` wrt lam and theta.

    s (B, n) float; lam (L, n); theta (L, n, 2); obs_idx (B,) int in [0, n);
    upstream (B,) float. ``psi_final`` (B, 2^n) may pass cached final
    statevectors from ``FastStatevectorSim.forward(..., return_state=True)``
    to skip the internal forward pass. Returns ``(dlam (L, n), dtheta (L, n, 2))``.
    """
    s = np.asarray(s, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    obs_idx = np.asarray(obs_idx, dtype=np.int64)
    upstream = np.asarray(upstream, dtype=np.float64)
    layers, n = lam.shape

    sim = FastStatevectorSim(n_qubits=n, n_layers=layers)
    if psi_final is None:
        _, psi = sim.forward(s, lam, theta, return_state=True)
    else:
        psi = np.array(psi_final, dtype=np.complex128)  # copy: modified in place below

    phi = sim.z_diags[obs_idx] * psi  # O|psi_N> with O = Z_{obs_idx[b]} (diagonal)
    psi = psi.copy()

    dlam = np.zeros((layers, n), dtype=np.float64)
    dtheta = np.zeros((layers, n, 2), dtype=np.float64)

    for layer in reversed(range(layers)):
        # CZ ring: diagonal and self-inverse, not parameterized.
        psi *= sim.ring_diagonal
        phi *= sim.ring_diagonal
        # Variational sublayer, exact reverse order: per qubit RZ then RY.
        for i in reversed(range(n)):
            g = _pauli_expectation_im(phi, psi, i, n, "z")
            dtheta[layer, i, 1] = np.dot(upstream, g)
            apply_rz(psi, i, -theta[layer, i, 1], n)
            apply_rz(phi, i, -theta[layer, i, 1], n)

            g = _pauli_expectation_im(phi, psi, i, n, "y")
            dtheta[layer, i, 0] = np.dot(upstream, g)
            apply_ry(psi, i, -theta[layer, i, 0], n)
            apply_ry(phi, i, -theta[layer, i, 0], n)
        # Data-uploading sublayer: angle = lam[l, i] * s[b, i] (per-sample).
        for i in reversed(range(n)):
            g = _pauli_expectation_im(phi, psi, i, n, "y")
            dlam[layer, i] = np.dot(upstream, s[:, i] * g)
            angle = -lam[layer, i] * s[:, i]
            apply_ry(psi, i, angle, n)
            apply_ry(phi, i, angle, n)

    return dlam, dtheta
