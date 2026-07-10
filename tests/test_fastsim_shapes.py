"""Shape, range, batching, and determinism tests for the numpy fast path."""

import numpy as np

from traqmania.agents.quantum.fastsim import FastStatevectorSim
from traqmania.agents.quantum.qdqn import QuantumQFunction

CFG = {"n_qubits": 4, "n_layers": 4, "seed": 7}


def _random_inputs(batch, rng):
    s = rng.uniform(0.0, 1.0, size=(batch, 4))
    lam = rng.uniform(0.5, 2.0 * np.pi, size=(4, 4))
    theta = rng.uniform(-np.pi, np.pi, size=(4, 4, 2))
    return s, lam, theta


def test_forward_shapes_and_dtype():
    sim = FastStatevectorSim(n_qubits=4, n_layers=4)
    s, lam, theta = _random_inputs(7, np.random.default_rng(0))
    expectations, psi = sim.forward(s, lam, theta, return_state=True)
    assert expectations.shape == (7, 4)
    assert expectations.dtype == np.float64
    assert psi.shape == (7, 16)
    assert psi.dtype == np.complex128
    # Statevectors stay normalized.
    np.testing.assert_allclose(np.sum(np.abs(psi) ** 2, axis=1), 1.0, atol=1e-12)


def test_expectations_bounded():
    sim = FastStatevectorSim()
    s, lam, theta = _random_inputs(32, np.random.default_rng(1))
    expectations = sim.forward(s, lam, theta)
    assert np.all(np.abs(expectations) <= 1.0 + 1e-12)


def test_batching_matches_single_sample_loop():
    sim = FastStatevectorSim()
    s, lam, theta = _random_inputs(9, np.random.default_rng(2))
    batched = sim.forward(s, lam, theta)
    singles = np.vstack([sim.forward(s[b : b + 1], lam, theta) for b in range(9)])
    np.testing.assert_allclose(batched, singles, atol=1e-14)


def test_qfunction_shapes_and_param_layout():
    qf = QuantumQFunction(CFG)
    assert qf.n_features == 4
    assert qf.n_actions == 4
    params = qf.get_params()
    assert params.shape == (56,)  # 16 lam + 32 theta + 4 w + 4 b
    np.testing.assert_allclose(params[:16], np.pi)  # lam init
    assert np.all(np.abs(params[16:48]) <= 0.1)  # theta init U(-0.1, 0.1)
    np.testing.assert_allclose(params[48:52], 1.0)  # w init
    np.testing.assert_allclose(params[52:56], 0.0)  # b init

    obs = np.random.default_rng(3).uniform(0.0, 1.0, size=(5, 4))
    q = qf.q_values(obs)
    assert q.shape == (5, 4)


def test_seed_determinism():
    obs = np.random.default_rng(4).uniform(0.0, 1.0, size=(6, 4))
    qf_a = QuantumQFunction(CFG, seed=7)
    qf_b = QuantumQFunction(CFG, seed=7)
    np.testing.assert_array_equal(qf_a.get_params(), qf_b.get_params())
    np.testing.assert_array_equal(qf_a.q_values(obs), qf_b.q_values(obs))

    qf_c = QuantumQFunction(CFG, seed=8)
    assert not np.array_equal(qf_a.get_params(), qf_c.get_params())


def test_set_params_round_trip():
    qf = QuantumQFunction(CFG)
    rng = np.random.default_rng(5)
    new_params = rng.normal(size=qf.get_params().shape)
    qf.set_params(new_params)
    np.testing.assert_array_equal(qf.get_params(), new_params)
