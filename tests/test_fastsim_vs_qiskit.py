"""Parity tests: numpy fast path vs the Qiskit execution paths (EstimatorQNN/Aer).

Three seeded random parameter/observation sets are checked on each claim:
exact-forward parity, q_values parity, grad_selected parity, and a statistical
(5 sigma) check on the shot-based path.
"""

from __future__ import annotations

import numpy as np
import pytest

from traqmania.agents.base import QFunction
from traqmania.agents.quantum import QuantumQFunction, make_qfunction
from traqmania.agents.quantum.fastsim import FastStatevectorSim

N_QUBITS = 4
N_LAYERS = 4
CFG = {"n_qubits": N_QUBITS, "n_layers": N_LAYERS, "seed": 7}
N_SETS = 3
BATCH = 3


def _random_set(rng: np.random.Generator):
    """One random (s, lam, theta, w, b) draw."""
    s = rng.uniform(0.0, 1.0, size=(BATCH, N_QUBITS))
    lam = rng.uniform(0.0, 2.0 * np.pi, size=(N_LAYERS, N_QUBITS))
    theta = rng.uniform(-np.pi, np.pi, size=(N_LAYERS, N_QUBITS, 2))
    w = rng.uniform(-1.5, 1.5, size=N_QUBITS)
    b = rng.uniform(-0.5, 0.5, size=N_QUBITS)
    return s, lam, theta, w, b


def _flat(lam, theta, w, b):
    return np.concatenate([lam.ravel(), theta.ravel(), w, b])


@pytest.fixture(scope="module")
def qf_exact():
    return make_qfunction("aer_statevector", CFG)


def test_qiskit_qfunction_satisfies_protocol(qf_exact):
    assert isinstance(qf_exact, QFunction)
    assert qf_exact.n_features == N_QUBITS
    assert qf_exact.n_actions == N_QUBITS


def test_fastsim_forward_matches_qiskit_exact(qf_exact):
    """Raw expectation values <Z_a>: fastsim vs Aer exact path, <= 1e-9."""
    rng = np.random.default_rng(11)
    sim = FastStatevectorSim(N_QUBITS, N_LAYERS)
    for _ in range(N_SETS):
        s, lam, theta, _, _ = _random_set(rng)
        e_fast = sim.forward(s, lam, theta)
        # With w = 1 and b = 0 the Q-values ARE the expectation values.
        qf_exact.set_params(_flat(lam, theta, np.ones(N_QUBITS), np.zeros(N_QUBITS)))
        e_qiskit = qf_exact.q_values(s)
        assert np.max(np.abs(e_fast - e_qiskit)) <= 1e-9


def test_q_values_parity_fastsim_vs_aer_statevector(qf_exact):
    """QuantumQFunction.q_values vs QiskitQFunction(aer_statevector), <= 1e-7."""
    rng = np.random.default_rng(23)
    qf_fast = QuantumQFunction(CFG)
    for _ in range(N_SETS):
        s, lam, theta, w, b = _random_set(rng)
        params = _flat(lam, theta, w, b)
        qf_fast.set_params(params)
        qf_exact.set_params(params)
        q_fast = qf_fast.q_values(s)
        q_qiskit = qf_exact.q_values(s)
        assert q_fast.shape == (BATCH, N_QUBITS)
        assert np.max(np.abs(q_fast - q_qiskit)) <= 1e-7


def test_grad_selected_parity_fastsim_vs_aer_statevector(qf_exact):
    """Adjoint gradients vs param-shift gradients on the exact path, <= 1e-6."""
    rng = np.random.default_rng(37)
    qf_fast = QuantumQFunction(CFG)
    for _ in range(N_SETS):
        s, lam, theta, w, b = _random_set(rng)
        params = _flat(lam, theta, w, b)
        qf_fast.set_params(params)
        qf_exact.set_params(params)
        action_idx = rng.integers(0, N_QUBITS, size=BATCH)
        upstream = rng.normal(size=BATCH)
        g_fast = qf_fast.grad_selected(s, action_idx, upstream)
        g_qiskit = qf_exact.grad_selected(s, action_idx, upstream)
        assert g_fast.shape == g_qiskit.shape == (qf_fast.n_params,)
        assert np.max(np.abs(g_fast - g_qiskit)) <= 1e-6


def test_aer_shots_within_statistical_tolerance():
    """Shot-based q_values agree with exact within 5 sigma per entry."""
    shots = 1024
    qf_shots = make_qfunction("aer_shots", CFG, shots=shots)
    qf_fast = QuantumQFunction(CFG)  # same seed -> identical initial params
    rng = np.random.default_rng(53)
    s = rng.uniform(0.0, 1.0, size=(BATCH, N_QUBITS))

    q_exact = qf_fast.q_values(s)  # w = 1, b = 0 at init, so Q = <Z_a>
    q_shots = qf_shots.q_values(s)
    # Aer's EstimatorV2 samples N(expval, precision) with precision = 1/sqrt(shots),
    # so the per-entry standard deviation is exactly 1/sqrt(shots).
    sigma = 1.0 / np.sqrt(shots)
    assert np.all(np.abs(q_shots - q_exact) <= 5.0 * sigma)


def test_aer_noisy_runs_and_is_sane():
    """Noisy path works (with or without qiskit-ibm-runtime) and stays plausible."""
    qf_noisy = make_qfunction("aer_noisy", CFG, shots=1024)
    qf_fast = QuantumQFunction(CFG)
    rng = np.random.default_rng(71)
    s = rng.uniform(0.0, 1.0, size=(BATCH, N_QUBITS))

    q_noisy = qf_noisy.q_values(s)
    assert q_noisy.shape == (BATCH, N_QUBITS)
    assert np.all(np.isfinite(q_noisy))
    assert np.max(np.abs(q_noisy)) <= 1.0 + 1e-9  # w = 1, b = 0: still expectations
    # Depolarizing-ish noise shrinks expectations toward 0 but not arbitrarily far.
    assert np.max(np.abs(q_noisy - qf_fast.q_values(s))) < 0.5


def test_make_qfunction_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_qfunction("bogus", CFG)
