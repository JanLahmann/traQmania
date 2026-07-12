"""Obs-aware weight resolution: CarObserver extraction, step_controls,
meta-sidecar observation recording/adoption, and the records plumbing."""

import copy
import json

import numpy as np

from traqmania.agents.base import ACTIONS
from traqmania.config import load_config
from traqmania.env.racing_env import CarObserver, RacingEnv
from traqmania.env.track import Track
from traqmania.records import discover_drivers, evaluate, load_records, save_record
from traqmania.server import protocol as P
from traqmania.server.runtime import WEIGHTS_DIR, weights_observation
from traqmania.server.session import DemoSession


def _feat_config():
    config = copy.deepcopy(load_config())
    config["observation"]["ray_angles_deg"] = [-60.0, 0.0, 60.0]
    config["observation"]["features"] = [
        "rays", "speed", "curvature_ahead", "corner_speed_ratio"]
    config["circuit"]["n_qubits"] = 6
    return config


def _track(config):
    return Track.load("oval", config["track"]["resample_spacing"])


def test_observer_matches_env_obs_default_and_features():
    for config in (load_config(), _feat_config()):
        track = _track(config)
        env = RacingEnv(track, config, n_envs=3, seed=7)
        obs = env.reset()
        assert obs.shape == (3, env.observer.n_features)
        np.testing.assert_array_equal(obs, env.observer.observe(env.state))
        # a standalone observer over the same config agrees bit-for-bit
        solo = CarObserver(track, config)
        np.testing.assert_array_equal(obs, solo.observe(env.state))


def test_observer_rays_slice_positions():
    config = _feat_config()
    observer = CarObserver(_track(config), config)
    assert observer.rays_slice == slice(0, 3)
    assert observer.feature_names[3] == "speed"
    config["observation"]["features"] = ["speed", "rays"]
    del config["circuit"]["n_qubits"]  # 4 scalars now; skip the qubit check
    observer = CarObserver(_track(config), config)
    assert observer.rays_slice == slice(1, 4)


def test_step_controls_matches_step():
    config = load_config()
    track = _track(config)
    env_a = RacingEnv(track, config, n_envs=4, seed=3)
    env_b = RacingEnv(track, config, n_envs=4, seed=3)
    env_a.reset(), env_b.reset()
    rng = np.random.default_rng(0)
    for _ in range(25):
        actions = rng.integers(len(ACTIONS), size=4)
        obs_a, r_a, d_a, _ = env_a.step(actions)
        obs_b, r_b, d_b, _ = env_b.step_controls(np.asarray(ACTIONS)[actions])
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(r_a, r_b)
        np.testing.assert_array_equal(d_a, d_b)


def test_weights_observation_reader(tmp_path):
    npz = tmp_path / "quantum_test.npz"
    np.savez(npz, params=np.zeros(3))
    assert weights_observation(npz) is None  # no sidecar
    meta = npz.with_suffix("").with_suffix(".meta.json")
    meta.write_text(json.dumps({"episodes": 1}), encoding="utf-8")
    assert weights_observation(npz) is None  # sidecar without observation
    meta.write_text(json.dumps({"observation": {"features": ["rays", "speed"]}}),
                    encoding="utf-8")
    assert weights_observation(npz) == {"features": ["rays", "speed"]}


def test_bundled_gp_q10_records_feature_observation():
    obs = weights_observation(WEIGHTS_DIR / "quantum_gp_q10.npz")
    assert obs is not None and "curvature_ahead" in obs["features"]
    assert len(obs["ray_angles_deg"]) == 5


def test_session_adopts_and_reverts_driver_observation(tmp_path):
    session = DemoSession(load_config("q10"), ghosts_dir=tmp_path)
    plain_labels = session._obs_labels()
    assert len(plain_labels) == 10 and "curvature ahead" not in plain_labels
    # gp's q10 weights were trained on engineered features: adopted on entry
    session.handle_message(P.SetTrack(track="gp"))
    feat_labels = session._obs_labels()
    assert "curvature ahead" in feat_labels and len(feat_labels) == 10
    assert session.cars and session.cars[0].kind == "quantum"
    obs = session._car_obs(session.cars[0])
    assert obs.shape == (1, 10) and len(session.cars[0].rays) == 5
    # oval's q10 weights are plain rays+speed: reverts to the profile obs
    session.handle_message(P.SetTrack(track="oval"))
    assert session._obs_labels() == plain_labels
    assert len(session.config["observation"]["ray_angles_deg"]) == 9


def test_records_discovery_and_merge(tmp_path):
    drivers = {d.id: d for d in discover_drivers()}
    assert "quantum_gp_q10" in drivers and "hero" in drivers
    assert drivers["quantum_gp_q10"].config["observation"]["features"][2] == "curvature_ahead"
    record = evaluate(drivers["mlp_oval"], "oval", episodes=2)
    assert record["lapped_episodes"] == 2 and record["best_s"] < 15.0
    out = tmp_path / "records.json"
    save_record(record, out)
    save_record({**record, "track": "chicane"}, out)
    stored = load_records(out)["records"]
    assert set(stored) == {"mlp_oval|oval", "mlp_oval|chicane"}
