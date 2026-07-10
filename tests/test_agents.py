"""Unit tests for the classical MLP Q-function: shapes, param roundtrip, gradients."""

import numpy as np

from traqmania.agents.base import ACTIONS, N_ACTIONS
from traqmania.agents.classical import MLPQFunction


def test_action_table():
    assert N_ACTIONS == 4
    assert len(ACTIONS) == N_ACTIONS
    assert all(len(a) == 3 for a in ACTIONS)


def test_q_values_shape():
    qfunc = MLPQFunction(n_features=4, hidden=8, n_actions=4, seed=1)
    obs = np.random.default_rng(0).normal(size=(7, 4))
    q = qfunc.q_values(obs)
    assert q.shape == (7, 4)
    assert np.all(np.isfinite(q))


def test_get_set_params_roundtrip():
    qfunc = MLPQFunction(n_features=4, hidden=8, n_actions=4, seed=2)
    rng = np.random.default_rng(3)
    obs = rng.normal(size=(5, 4))

    original = qfunc.get_params()
    q_before = qfunc.q_values(obs)

    # get_params must be a copy: mutating it does not change the network.
    original_copy = original.copy()
    original[:] = 0.0
    np.testing.assert_array_equal(qfunc.q_values(obs), q_before)

    qfunc.set_params(rng.normal(size=original_copy.shape))
    assert not np.allclose(qfunc.q_values(obs), q_before)

    qfunc.set_params(original_copy)
    np.testing.assert_allclose(qfunc.q_values(obs), q_before, rtol=0, atol=0)


def test_grad_selected_matches_finite_differences():
    rng = np.random.default_rng(42)
    qfunc = MLPQFunction(n_features=4, hidden=8, n_actions=4, seed=7)
    batch = 6
    obs = rng.normal(size=(batch, 4))
    action_idx = rng.integers(0, 4, size=batch)
    upstream = rng.normal(size=batch)

    def loss(params):
        qfunc.set_params(params)
        q_sel = qfunc.q_values(obs)[np.arange(batch), action_idx]
        return float(np.sum(upstream * q_sel))

    params = qfunc.get_params()
    analytic = qfunc.grad_selected(obs, action_idx, upstream)

    h = 1e-6
    numeric = np.zeros_like(params)
    for i in range(params.shape[0]):
        plus = params.copy()
        minus = params.copy()
        plus[i] += h
        minus[i] -= h
        numeric[i] = (loss(plus) - loss(minus)) / (2.0 * h)
    qfunc.set_params(params)

    rel_err = np.linalg.norm(numeric - analytic) / max(np.linalg.norm(numeric), 1e-12)
    assert rel_err < 1e-5, f"finite-difference relative error {rel_err:.3e}"
