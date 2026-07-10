"""Hardware execution path, exercised entirely on LOCAL fake backends.

No network, no IBM account: ``get_backend(use_fake=True)`` returns a
``FakeBackendV2`` whose noise model runs on Aer, and runtime Sessions enter
local testing mode. Skipped wholesale when qiskit-ibm-runtime (the [hardware]
extra) is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")

from traqmania import hardware  # noqa: E402
from traqmania.agents.quantum.qdqn import QuantumQFunction  # noqa: E402
from traqmania.agents.training import spsa  # noqa: E402

CIRCUIT_CFG = {"n_qubits": 4, "n_layers": 4, "seed": 7}
OVAL_WEIGHTS = hardware.WEIGHTS_DIR / "quantum_oval.npz"


@pytest.fixture(scope="module")
def fake_backend():
    return hardware.get_backend(use_fake=True)


def test_get_backend_fake_has_five_plus_qubits(fake_backend):
    assert fake_backend.num_qubits >= 5


def test_hardware_qvalues_match_fastsim(fake_backend):
    """Fake-backend Q-values track the exact simulator on the same parameters.

    Tolerances are deliberately loose: the fake backend simulates a real
    device's noise model, which shrinks |<Z_a>| — we ask for argmax agreement
    on most observations and absolute agreement within 0.5. The Aer shot
    noise is seeded so the statistical margins cannot flake.
    """
    fast = QuantumQFunction(CIRCUIT_CFG, seed=7)
    hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend, shots=4096)
    hw.set_params(fast.get_params())
    hw._estimator.options.simulator.seed_simulator = 1234  # pin the shot noise

    obs = np.random.default_rng(3).uniform(0.0, 1.0, size=(8, 4))
    q_fast = fast.q_values(obs)
    q_hw = hw.q_values(obs)

    assert q_hw.shape == (8, 4)
    agreement = float(np.mean(np.argmax(q_hw, axis=1) == np.argmax(q_fast, axis=1)))
    assert agreement >= 0.6, f"argmax agreement {agreement:.2f} < 0.6\n{q_hw}\nvs\n{q_fast}"
    assert np.max(np.abs(q_hw - q_fast)) < 0.5


def test_hardware_qfunction_param_roundtrip_and_layout(fake_backend):
    hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend)
    assert hw.n_params == 56
    params = np.arange(56, dtype=np.float64)
    hw.set_params(params)
    np.testing.assert_allclose(hw.get_params(), params)
    # Same flat layout as the fastsim twin: [lam(16), theta(32), w(4), b(4)].
    np.testing.assert_allclose(hw.lam.ravel(), params[:16])
    np.testing.assert_allclose(hw.theta.ravel(), params[16:48])
    np.testing.assert_allclose(hw.w, params[48:52])
    np.testing.assert_allclose(hw.b, params[52:])


def test_grad_selected_is_not_implemented(fake_backend):
    hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend)
    obs = np.zeros((2, 4))
    with pytest.raises(NotImplementedError, match="spsa_sprint"):
        hw.grad_selected(obs, np.zeros(2, dtype=np.int64), np.ones(2))


def test_spsa_minimize_converges_on_quadratic():
    target = np.array([0.3, -0.2, 0.5, 0.1, -0.4])

    def f(x):
        return float(np.sum((x - target) ** 2))

    x0 = np.zeros(5)
    seen: list[int] = []
    result = spsa.minimize(f, x0, iterations=300, seed=1, callback=lambda k, _info: seen.append(k))

    assert len(result["loss_history"]) == 300
    assert seen == list(range(300))
    assert f(result["x"]) < 0.01 * f(x0)
    np.testing.assert_allclose(result["x"], target, atol=0.05)


def test_spsa_minimize_is_deterministic():
    def f(x):
        return float(np.sum(x**2))

    r1 = spsa.minimize(f, np.ones(4), iterations=20, seed=42)
    r2 = spsa.minimize(f, np.ones(4), iterations=20, seed=42)
    np.testing.assert_array_equal(r1["x"], r2["x"])
    assert r1["loss_history"] == r2["loss_history"]


def test_run_hardware_lap_smoke(fake_backend):
    decisions_seen: list[int] = []
    result = hardware.run_hardware_lap(
        "oval",
        OVAL_WEIGHTS,
        fake_backend,
        shots=512,
        max_decisions=40,
        on_decision=lambda i, _info: decisions_seen.append(i),
    )

    assert isinstance(result["lapped"], bool)
    assert 0 < result["decisions"] <= 40
    assert result["seconds_per_decision"] > 0.0
    assert len(decisions_seen) == result["decisions"]
    trajectory = result["trajectory"]
    assert len(trajectory) >= 1
    assert all(len(state) == 4 for state in trajectory)  # [x, y, theta, v]
    if result["lapped"]:
        assert result["best_lap_s"] > 0.0
    else:
        assert result["best_lap_s"] is None


def test_spsa_sprint_smoke(fake_backend):
    iters_seen: list[int] = []
    result = hardware.spsa_sprint(
        "oval",
        OVAL_WEIGHTS,
        fake_backend,
        iterations=3,
        shots=256,
        batch=8,
        on_iter=lambda k, _info: iters_seen.append(k),
    )

    assert len(result["loss_history"]) == 3
    assert iters_seen == [0, 1, 2]
    assert all(np.isfinite(loss) for loss in result["loss_history"])
    assert np.isfinite(result["return_before"])
    assert np.isfinite(result["return_after"])
    assert result["params"].shape == (56,)
