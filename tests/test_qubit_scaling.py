"""Qubit scaling: n_qubits is a config knob, the action head is pinned at 4.

Contract: n_qubits total features ((n_qubits - 1) rays + speed), readout always
Z_0..Z_3 on the first four qubits, flat layout [lam(L*n), theta(L*n*2), w(4),
b(4)] -> P = 3*L*n + 8 (56 at n=4, 80 at n=6). At n=4 behavior is bit-identical
to main: the bundled weights load and reproduce hardcoded Q-values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import traqmania
from traqmania.agents.quantum import make_qfunction
from traqmania.agents.quantum.circuit import circuit_spec
from traqmania.agents.quantum.fastsim import FastStatevectorSim
from traqmania.agents.quantum.qdqn import QuantumQFunction
from traqmania.config import load_config

N_LAYERS = 4
CFG4 = {"n_qubits": 4, "n_layers": N_LAYERS, "seed": 7}
CFG6 = {"n_qubits": 6, "n_layers": N_LAYERS, "seed": 7}
EPS = 1e-6

OVAL_WEIGHTS = Path(traqmania.__file__).resolve().parent / "weights" / "quantum_oval.npz"

# ------------------------------------------------------- parameter counts / spec


def test_param_count_56_at_4_qubits_80_at_6():
    qf4 = QuantumQFunction(CFG4)
    qf6 = QuantumQFunction(CFG6)
    assert qf4.n_params == 56
    assert qf6.n_params == 80
    for qf in (qf4, qf6):
        assert qf.n_actions == 4
        assert qf.w.shape == (4,)
        assert qf.b.shape == (4,)
    assert qf6.n_features == 6
    assert qf6.lam.shape == (N_LAYERS, 6)
    assert qf6.theta.shape == (N_LAYERS, 6, 2)


def test_param_roundtrip_at_6_qubits():
    qf = QuantumQFunction(CFG6)
    params = np.arange(80, dtype=np.float64)
    qf.set_params(params)
    np.testing.assert_array_equal(qf.get_params(), params)
    # Flat layout: [lam(24), theta(48), w(4), b(4)].
    np.testing.assert_array_equal(qf.lam.ravel(), params[:24])
    np.testing.assert_array_equal(qf.theta.ravel(), params[24:72])
    np.testing.assert_array_equal(qf.w, params[72:76])
    np.testing.assert_array_equal(qf.b, params[76:])


def test_circuit_spec_readout_pinned_at_four_actions():
    for n, total in ((4, 56), (6, 80)):
        spec = circuit_spec({"circuit": {"n_qubits": n, "n_layers": N_LAYERS}})
        assert spec["n_qubits"] == n
        assert spec["n_actions"] == 4
        assert spec["readout"] == ["Z_0", "Z_1", "Z_2", "Z_3"]
        assert spec["n_params"]["w"] == 4
        assert spec["n_params"]["b"] == 4
        assert spec["n_params"]["total"] == total


# ------------------------------------------------------------- 6-qubit parity


def test_fastsim_forward_matches_qiskit_exact_at_6_qubits():
    """Raw expectation values <Z_a>: fastsim vs Aer exact path at n=6, <= 1e-9."""
    rng = np.random.default_rng(11)
    sim = FastStatevectorSim(6, N_LAYERS)
    qf_exact = make_qfunction("aer_statevector", CFG6)
    assert qf_exact.n_actions == 4
    for _ in range(3):
        s = rng.uniform(0.0, 1.0, size=(3, 6))
        lam = rng.uniform(0.0, 2.0 * np.pi, size=(N_LAYERS, 6))
        theta = rng.uniform(-np.pi, np.pi, size=(N_LAYERS, 6, 2))
        e_fast = sim.forward(s, lam, theta)[:, :4]
        # With w = 1 and b = 0 the Q-values ARE the expectation values.
        qf_exact.set_params(
            np.concatenate([lam.ravel(), theta.ravel(), np.ones(4), np.zeros(4)])
        )
        e_qiskit = qf_exact.q_values(s)
        assert e_qiskit.shape == (3, 4)
        assert np.max(np.abs(e_fast - e_qiskit)) <= 1e-9


# ---------------------------------------------------------- 6-qubit gradients


def test_grad_selected_matches_finite_differences_at_6_qubits():
    """All 80 gradient components at n=6 vs central finite differences."""
    rng = np.random.default_rng(13)
    qf = QuantumQFunction(CFG6, seed=7)
    # Move off the symmetric init so every parameter block has generic gradients.
    params = qf.get_params()
    params[:72] += rng.uniform(-0.7, 0.7, size=72)  # lam, theta
    params[72:76] = rng.uniform(0.5, 1.5, size=4)  # w
    params[76:] = rng.uniform(-0.5, 0.5, size=4)  # b
    qf.set_params(params)
    batch = 8
    obs = rng.uniform(0.0, 1.0, size=(batch, 6))
    # Cover every action so all w/b components get a nonzero gradient.
    actions = np.concatenate([np.arange(4), rng.integers(0, 4, size=batch - 4)])
    upstream = rng.normal(size=batch)

    analytic = qf.grad_selected(obs, actions, upstream)
    assert analytic.shape == (80,)

    def loss(p: np.ndarray) -> float:
        qf.set_params(p)
        q = qf.q_values(obs)
        return float(np.sum(upstream * q[np.arange(batch), actions]))

    base = qf.get_params()
    numeric = np.zeros(80)
    for k in range(80):
        plus = base.copy()
        plus[k] += EPS
        minus = base.copy()
        minus[k] -= EPS
        numeric[k] = (loss(plus) - loss(minus)) / (2.0 * EPS)
    qf.set_params(base)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
    rel = np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric)
    assert rel < 1e-6


# ---------------------------------------------------------- 4-qubit regression

REGRESSION_OBS = np.array(
    [
        [0.10, 0.55, 0.90, 0.30],
        [0.75, 0.20, 0.40, 0.85],
        [0.33, 0.66, 0.05, 0.50],
    ]
)
# Q-values of the bundled oval weights on REGRESSION_OBS, captured on main
# BEFORE the n_qubits generalization — n=4 must stay bit-identical.
REGRESSION_Q = np.array(
    [
        [191.5449488526533, 185.5949295753721, 186.2008782112444, 193.42539604236663],
        [190.53311752642043, 36.37641532745649, 178.15811311295226, 172.1771555388296],
        [206.59007108817428, 206.3428989297937, 201.7028756510693, 195.93719008817874],
    ]
)


def test_bundled_oval_weights_regression_at_default_config():
    config = load_config()
    assert int(config["circuit"]["n_qubits"]) == 4  # default stays 4-qubit
    qf = QuantumQFunction(config["circuit"])
    params = np.load(OVAL_WEIGHTS)["params"]
    assert params.shape == (56,)
    qf.set_params(params)
    q = qf.q_values(REGRESSION_OBS)
    np.testing.assert_allclose(q, REGRESSION_Q, rtol=0.0, atol=1e-9)


# ------------------------------------------------------- fake-backend selection


def test_fake_backend_selection_is_qubit_aware():
    pytest.importorskip("qiskit_ibm_runtime")
    from traqmania import hardware

    backend = hardware.get_backend(use_fake=True, min_qubits=6)
    assert backend.num_qubits >= 6
    # The preferred fake (5-qubit manila) is skipped when it is too small.
    backend = hardware._fake_backend("fake_manila", min_qubits=6)
    assert backend.num_qubits >= 6


def test_hardware_qfunction_runs_at_6_qubits():
    pytest.importorskip("qiskit_ibm_runtime")
    from traqmania import hardware

    backend = hardware.get_backend(use_fake=True, min_qubits=6)
    hw = hardware.HardwareQFunction(CFG6, backend, shots=256)
    assert hw.n_params == 80
    assert hw.n_actions == 4
    obs = np.random.default_rng(3).uniform(0.0, 1.0, size=(2, 6))
    q = hw.q_values(obs)
    assert q.shape == (2, 4)
    assert np.all(np.isfinite(q))
