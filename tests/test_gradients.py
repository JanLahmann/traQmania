"""Adjoint gradients vs central finite differences (float64, eps=1e-6)."""

import numpy as np

from traqmania.agents.quantum import adjoint
from traqmania.agents.quantum.fastsim import FastStatevectorSim
from traqmania.agents.quantum.qdqn import QuantumQFunction

CFG = {"n_qubits": 4, "n_layers": 4, "seed": 7}
EPS = 1e-6


def _problem(seed=11, batch=6):
    rng = np.random.default_rng(seed)
    qf = QuantumQFunction(CFG, seed=7)
    # Move off the symmetric init so every parameter block has generic gradients.
    params = qf.get_params()
    params[:48] += rng.uniform(-0.7, 0.7, size=48)  # lam, theta
    params[48:52] = rng.uniform(0.5, 1.5, size=4)  # w
    params[52:] = rng.uniform(-0.5, 0.5, size=4)  # b
    qf.set_params(params)
    obs = rng.uniform(0.0, 1.0, size=(batch, 4))
    actions = rng.integers(0, 4, size=batch)
    upstream = rng.normal(size=batch)
    return qf, obs, actions, upstream


def _loss(qf, params, obs, actions, upstream):
    qf.set_params(params)
    q = qf.q_values(obs)
    return float(np.sum(upstream * q[np.arange(obs.shape[0]), actions]))


def _fd_gradient(qf, obs, actions, upstream):
    base = qf.get_params()
    num = np.zeros_like(base)
    for k in range(base.size):
        plus = base.copy()
        plus[k] += EPS
        minus = base.copy()
        minus[k] -= EPS
        num[k] = (
            _loss(qf, plus, obs, actions, upstream) - _loss(qf, minus, obs, actions, upstream)
        ) / (2.0 * EPS)
    qf.set_params(base)
    return num


def test_grad_selected_matches_finite_differences_full_vector():
    qf, obs, actions, upstream = _problem()
    analytic = qf.grad_selected(obs, actions, upstream)
    assert analytic.shape == (56,)
    numeric = _fd_gradient(qf, obs, actions, upstream)

    # Component-wise and overall relative error.
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
    rel = np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric)
    assert rel < 1e-6


def test_lam_gradient_specifically():
    """The input-scaling block (chain rule through angle = lam * s) vs FD."""
    qf, obs, actions, upstream = _problem(seed=23)
    analytic = qf.grad_selected(obs, actions, upstream)[:16]
    numeric = _fd_gradient(qf, obs, actions, upstream)[:16]
    assert np.linalg.norm(numeric) > 1e-3  # the check must have teeth
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_adjoint_grad_matches_fd_on_expectations():
    """adjoint.grad directly: d/dparams of sum_b upstream[b] * <Z_{a_b}>_b."""
    rng = np.random.default_rng(42)
    sim = FastStatevectorSim(4, 4)
    s = rng.uniform(0.0, 1.0, size=(5, 4))
    lam = np.pi + rng.uniform(-0.7, 0.7, size=(4, 4))
    theta = rng.uniform(-1.0, 1.0, size=(4, 4, 2))
    obs_idx = rng.integers(0, 4, size=5)
    upstream = rng.normal(size=5)

    dlam, dtheta = adjoint.grad(s, lam, theta, obs_idx, upstream)
    assert dlam.shape == (4, 4)
    assert dtheta.shape == (4, 4, 2)

    def value(lam_flat, theta_flat):
        e = sim.forward(s, lam_flat.reshape(lam.shape), theta_flat.reshape(theta.shape))
        return float(np.sum(upstream * e[np.arange(5), obs_idx]))

    packed = np.concatenate([lam.ravel(), theta.ravel()])
    analytic = np.concatenate([dlam.ravel(), dtheta.ravel()])
    for k in range(packed.size):
        plus = packed.copy()
        plus[k] += EPS
        minus = packed.copy()
        minus[k] -= EPS
        num = (value(plus[:16], plus[16:]) - value(minus[:16], minus[16:])) / (2 * EPS)
        np.testing.assert_allclose(analytic[k], num, rtol=1e-6, atol=1e-8)


def test_upstream_linearity():
    """grad_selected is linear in upstream (sanity of batch accumulation)."""
    qf, obs, actions, upstream = _problem(seed=5)
    g1 = qf.grad_selected(obs, actions, upstream)
    g2 = qf.grad_selected(obs, actions, 2.0 * upstream)
    np.testing.assert_allclose(g2, 2.0 * g1, rtol=1e-12, atol=1e-12)
