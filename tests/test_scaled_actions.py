"""Qubit-scaled action readout, multi-distance curvature features, pace-reward
shaping, and the 12-episode best-snapshot selection."""

import copy
import json

import numpy as np
import pytest

from traqmania.agents.base import ACTION_SIZES, ACTIONS, action_labels, action_set
from traqmania.agents.classical import MLPQFunction
from traqmania.agents.quantum.circuit import circuit_spec
from traqmania.agents.quantum.qdqn import QuantumQFunction
from traqmania.agents.training import DQNTrainer
from traqmania.config import load_config
from traqmania.env.racing_env import CarObserver, RacingEnv
from traqmania.env.track import Track
from traqmania.server import protocol as P
from traqmania.server.runtime import weights_actions


def _track(config, name="oval"):
    return Track.load(name, config["track"]["resample_spacing"])


# ------------------------------------------------------------- action tables


def test_action_sets_are_prefix_compatible():
    assert action_set(4) == ACTIONS
    for small, big in zip(ACTION_SIZES, ACTION_SIZES[1:]):
        assert action_set(big)[: small] == action_set(small)
        assert action_labels(big)[: small] == action_labels(small)
    assert len(action_set(6)) == 6 and len(action_labels(8)) == 8
    # 6 actions add trail braking: full steer with the brake on
    for steer, throttle, brake in action_set(6)[4:]:
        assert abs(steer) == 1 and throttle == 0 and brake == 1
    with pytest.raises(ValueError):
        action_set(5)


def test_env_action_table_follows_config():
    config = copy.deepcopy(load_config())
    env = RacingEnv(_track(config), config, n_envs=2, seed=0)
    assert env.n_actions == 4
    config["circuit"]["n_qubits"] = 6
    config["circuit"]["n_actions"] = 6
    config["observation"]["ray_angles_deg"] = [-60.0, -20.0, 0.0, 20.0, 60.0]
    env6 = RacingEnv(_track(config), config, n_envs=2, seed=0)
    assert env6.n_actions == 6
    env6.reset()
    obs, reward, done, info = env6.step(np.array([4, 5]))  # the trail-brake pair
    assert obs.shape == (2, 6) and reward.shape == (2,)


def test_quantum_qfunc_scaled_readout():
    cfg = {"n_qubits": 6, "n_layers": 4, "seed": 7, "n_actions": 6}
    qfunc = QuantumQFunction(cfg)
    assert qfunc.n_actions == 6 and qfunc.w.size == 6
    q = qfunc.q_values(np.random.default_rng(0).random((3, 6)))
    assert q.shape == (3, 6)
    # gradient covers the widened head — and matches finite differences there
    obs = np.random.default_rng(1).random((3, 6))
    idx = np.array([0, 4, 5])
    g = qfunc.grad_selected(obs, idx, np.ones(3))
    assert g.shape == (qfunc.n_params,)
    params = qfunc.get_params()
    eps = 1e-6
    for k in (0, 10, params.size - 7, params.size - 1):  # lam, theta, w5, b5 region
        bumped = params.copy()
        bumped[k] += eps
        qfunc.set_params(bumped)
        up = np.sum(qfunc.q_values(obs)[np.arange(3), idx])
        bumped[k] -= 2 * eps
        qfunc.set_params(bumped)
        down = np.sum(qfunc.q_values(obs)[np.arange(3), idx])
        qfunc.set_params(params)
        assert abs((up - down) / (2 * eps) - g[k]) < 1e-5
    # a 4-action head's params must NOT load into the 6-action layout
    qfunc4 = QuantumQFunction({"n_qubits": 6, "n_layers": 4, "seed": 7})
    with pytest.raises(ValueError):
        qfunc.set_params(qfunc4.get_params())
    with pytest.raises(ValueError):
        QuantumQFunction({"n_qubits": 4, "n_actions": 6})  # more actions than qubits


def test_scaled_readout_default_is_bit_identical():
    plain = QuantumQFunction({"n_qubits": 4, "n_layers": 4, "seed": 7})
    explicit = QuantumQFunction({"n_qubits": 4, "n_layers": 4, "seed": 7,
                                 "n_actions": 4})
    obs = np.random.default_rng(2).random((5, 4))
    np.testing.assert_array_equal(plain.q_values(obs), explicit.q_values(obs))


def test_circuit_spec_reports_actions():
    spec = circuit_spec({"circuit": {"n_qubits": 8, "n_actions": 6}})
    assert spec["n_actions"] == 6
    assert spec["readout"] == [f"Z_{a}" for a in range(6)]
    assert spec["action_labels"][:4] == ["Right", "Straight", "Left", "Brake"]
    assert spec["n_params"]["w"] == 6
    assert circuit_spec({"circuit": {"n_qubits": 4}})["action_labels"] == \
        ["Right", "Straight", "Left", "Brake"]


def test_weights_actions_reader(tmp_path):
    npz = tmp_path / "quantum_test.npz"
    np.savez(npz, params=np.zeros(3))
    meta = npz.with_suffix("").with_suffix(".meta.json")
    assert weights_actions(npz) is None  # no sidecar
    meta.write_text(json.dumps({"episodes": 1}), encoding="utf-8")
    assert weights_actions(npz) is None  # sidecar without actions
    meta.write_text(json.dumps({"actions": {"n_actions": 6}}), encoding="utf-8")
    assert weights_actions(npz) == 6


# --------------------------------------------------- multi-distance curvature


def test_multi_distance_curvature_features():
    config = copy.deepcopy(load_config())
    config["observation"]["ray_angles_deg"] = [-60.0, 0.0, 60.0]
    config["observation"]["features"] = [
        "rays", "speed", "curvature_ahead", "curvature_ahead:30", "curvature_ahead:50"]
    config["circuit"]["n_qubits"] = 7
    track = _track(config, "gp")
    observer = CarObserver(track, config)
    assert observer.feature_names[4:] == [
        "curvature ahead", "curvature ahead 30m", "curvature ahead 50m"]
    rng = np.random.default_rng(3)
    # random on-track poses: sample along the centerline
    s = rng.uniform(0, track.total_length, 16)
    xy = track.point_at(s) if hasattr(track, "point_at") else None
    if xy is None:  # fall back to env spawns stepped a few times
        env = RacingEnv(track, config, n_envs=16, seed=1)
        env.reset()
        for _ in range(10):
            env.step(rng.integers(4, size=16))
        state = env.state
    else:
        state = np.zeros((16, 4))
        state[:, :2] = xy
    obs = observer.observe(state)
    # max |kappa| over a longer window can only grow: 15m <= 30m <= 50m
    assert np.all(obs[:, 5] >= obs[:, 4] - 1e-12)
    assert np.all(obs[:, 6] >= obs[:, 5] - 1e-12)


def test_bad_curvature_suffix_rejected():
    config = copy.deepcopy(load_config())
    del config["circuit"]["n_qubits"]
    for bad in (["rays", "speed:30"], ["curvature_ahead:zero"], ["curvature_ahead:-5"]):
        config["observation"]["features"] = bad
        with pytest.raises(ValueError):
            CarObserver(_track(config), config)


# --------------------------------------------------------- pace reward shaping


def test_time_penalty_shapes_reward():
    config = copy.deepcopy(load_config())
    track = _track(config)
    base = RacingEnv(track, config, n_envs=3, seed=5)
    paced_cfg = copy.deepcopy(config)
    paced_cfg["reward"]["time_penalty"] = 0.5
    paced = RacingEnv(track, paced_cfg, n_envs=3, seed=5)
    base.reset(), paced.reset()
    rng = np.random.default_rng(4)
    for _ in range(20):
        a = rng.integers(4, size=3)
        _, r_base, d_base, _ = base.step(a)
        _, r_paced, d_paced, _ = paced.step(a)
        np.testing.assert_array_equal(d_base, d_paced)  # same dynamics
        np.testing.assert_allclose(r_paced, r_base - 0.5, atol=1e-12)


def test_training_pace_preset_bundled():
    pace = load_config()["training_pace"]
    assert pace["time_penalty"] > 0 and pace["epsilon_start"] <= 0.25


# ------------------------------------------------------ 12-episode snapshots


def test_eval_snapshot_uses_lapped_and_mean_lap():
    config = copy.deepcopy(load_config())
    track = _track(config)
    env = RacingEnv(track, config, n_envs=4, seed=0)
    qfunc = MLPQFunction(n_features=4, n_actions=4, seed=0)
    tcfg = dict(config["training"], eval_episodes=8)
    trainer = DQNTrainer(qfunc, env, tcfg, rng=np.random.default_rng(0),
                         env_factory=lambda: RacingEnv(track, config, n_envs=4, seed=99))
    assert trainer.eval_episodes == 8
    best = trainer._eval_snapshot(None, episode=0)
    assert best["eval_episodes"] == 8
    assert 0 <= best["lapped_episodes"] <= 8
    assert best["score"][0] == best["lapped_episodes"]
    assert set(best) >= {"params", "episode", "mean_lap", "laps", "best_lap"}


# --------------------------------------------------------- session adoption


def test_session_adopts_and_reverts_action_count(tmp_path, monkeypatch):
    import traqmania.server.runtime as runtime_mod
    import traqmania.server.session as session_mod

    weights = tmp_path / "weights"
    weights.mkdir()
    config = load_config("q6")
    # a 6-action, 6-qubit oval driver with a sidecar recording its action set
    qfunc = QuantumQFunction({**config["circuit"], "n_actions": 6})
    np.savez(weights / "quantum_oval_q6.npz", params=qfunc.get_params())
    (weights / "quantum_oval_q6.meta.json").write_text(
        json.dumps({"actions": {"n_actions": 6},
                    "observation": {
                        "ray_angles_deg": list(
                            config["observation"]["ray_angles_deg"]),
                        "features": ["rays", "speed"]}}),
        encoding="utf-8")
    # ... and a 4-action chicane driver at the same qubit count (no sidecar)
    qfunc4 = QuantumQFunction(config["circuit"])
    np.savez(weights / "quantum_chicane_q6.npz", params=qfunc4.get_params())
    monkeypatch.setattr(session_mod, "WEIGHTS_DIR", weights)
    monkeypatch.setattr(runtime_mod, "WEIGHTS_DIR", weights)

    session = session_mod.DemoSession(config, ghosts_dir=tmp_path)
    spec = session.welcome_payload()["circuit_spec"]
    assert spec["n_actions"] == 6 and len(spec["action_labels"]) == 6
    assert session.cars and session.cars[0].qfunc.n_actions == 6
    for _ in range(12):  # a decision happens and decodes through the 6-table
        session.tick()
    assert 0 <= session.cars[0].action < 6

    session.handle_message(P.SetTrack(track="chicane"))
    spec = session.welcome_payload()["circuit_spec"]
    assert spec["n_actions"] == 4  # reverts for the sidecar-less driver
    assert session.cars[0].qfunc.n_actions == 4
