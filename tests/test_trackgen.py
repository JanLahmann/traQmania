"""Tests for procedural track generation and the multi-track mixture env."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from traqmania.config import load_config
from traqmania.env import multi_track
from traqmania.env.multi_track import MultiTrackEnv
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track
from traqmania.env.trackgen import CHECKPOINTS, generate_track

# ACTIONS indices, same convention as test_racing_env.
STEER_NEG, STRAIGHT, STEER_POS, BRAKE = 0, 1, 2, 3

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_config()


# ------------------------------------------------------------- generate_track


def test_generate_track_deterministic():
    a = generate_track(seed=7)
    b = generate_track(seed=7)
    assert a.name == b.name == "random-7"
    np.testing.assert_array_equal(a.centerline, b.centerline)
    assert a.half_width == b.half_width

    other = generate_track(seed=8)
    assert a.centerline.shape != other.centerline.shape or not np.array_equal(
        a.centerline, other.centerline
    )
    # Same seed, different difficulty is a different track too.
    hard = generate_track(seed=7, difficulty=1.0)
    assert not (
        a.centerline.shape == hard.centerline.shape
        and np.array_equal(a.centerline, hard.centerline)
    )


def test_generate_track_name_override():
    track = generate_track(seed=3, name="my-track")
    assert track.name == "my-track"


@pytest.mark.parametrize("difficulty", [0.3, 0.6, 0.9])
def test_generate_track_30_seeds_pass_validation(difficulty):
    # Track's constructor runs the same validation as Track.load; construction
    # succeeding means every generated track passed it.
    for seed in range(30):
        track = generate_track(seed, difficulty=difficulty)
        assert track.generation_attempts <= 10, (
            f"seed {seed} difficulty {difficulty} needed {track.generation_attempts} attempts"
        )
        assert track.half_width >= 3.0
        assert list(track.checkpoints) == list(CHECKPOINTS)
        assert 100.0 < track.total_length < 1500.0


def test_harder_tracks_are_tighter_and_narrower():
    easy = generate_track(seed=11, difficulty=0.0)
    hard = generate_track(seed=11, difficulty=1.0)
    assert hard.half_width < easy.half_width
    assert hard.max_abs_curvature > easy.max_abs_curvature


# --------------------------------------------- driving on a generated track


def test_generated_track_off_track_detection(config):
    track = generate_track(seed=5, difficulty=0.5)
    env = RacingEnv(track, config, n_envs=2, seed=0)
    env.reset()
    done = info = None
    for _ in range(300):
        _, reward, done, info = env.step(np.full(2, STEER_POS))
        if np.any(done):
            break
    assert done is not None and np.any(done)
    assert np.all(info["off_track"][done])
    assert np.all(reward[done] < 0.0)
    # Auto-reset respawned the crashed cars at the start line with v = 0.
    x0, y0, _ = track.start_pose()
    for i in np.flatnonzero(done):
        assert env.state[i, 3] == 0.0
        assert np.hypot(env.state[i, 0] - x0, env.state[i, 1] - y0) <= track.half_width


def test_generated_track_scripted_lap(config):
    """A ray-balance controller (brake when the front ray is short) laps a
    moderately difficult generated track: lap bookkeeping fires end-to-end."""
    track = generate_track(seed=3, difficulty=0.3)
    env = RacingEnv(track, config, n_envs=1, seed=3)
    obs = env.reset()
    lap_time = np.nan
    for _ in range(config["reward"]["max_decisions"]):
        right, front, left, speed = obs[0]
        if front < 0.35 and speed > 0.45:
            action = BRAKE
        elif left > right + 0.03:
            action = STEER_POS
        elif right > left + 0.03:
            action = STEER_NEG
        else:
            action = STRAIGHT
        obs, _, _, info = env.step(np.array([action]))
        if not np.isnan(info["last_lap_time"][0]):
            lap_time = info["last_lap_time"][0]
            break
        assert not info["off_track"][0]
    assert not np.isnan(lap_time), "controller failed to lap the generated track"
    assert info["lap"][0] == 1
    episode_cap_s = config["reward"]["max_decisions"] * env.decision_dt
    assert track.total_length / config["physics"]["v_max"] < lap_time < episode_cap_s


# ---------------------------------------------------------------- MultiTrackEnv


def test_multi_track_env_shape_and_order(config):
    spacing = config["track"]["resample_spacing"]
    tracks = [Track.load("oval", spacing), Track.load("chicane", spacing)]
    env = MultiTrackEnv(tracks, config, n_envs=5, seed=0)
    assert env.n_envs == 5
    assert env.n_features == 4
    assert env.feature_names == RacingEnv(tracks[0], config, n_envs=1, seed=0).feature_names
    np.testing.assert_array_equal(env.track_index, [0, 1, 0, 1, 0])

    obs = env.reset()
    assert obs.shape == (5, 4)
    # Global env i spawns at the start line of track i % 2 (fixed order).
    snap = env.state_snapshot()
    assert snap["state"].shape == (5, 4)
    for i in range(5):
        x0, y0, _ = tracks[i % 2].start_pose()
        assert np.hypot(snap["state"][i, 0] - x0, snap["state"][i, 1] - y0) <= 3.0

    # Same construction is deterministic, including through step().
    env_b = MultiTrackEnv(tracks, config, n_envs=5, seed=0)
    np.testing.assert_array_equal(obs, env_b.reset())
    for _ in range(10):
        actions = np.array([STRAIGHT, STEER_POS, BRAKE, STRAIGHT, STEER_NEG])
        obs_a, rew_a, done_a, info_a = env.step(actions)
        obs_b, rew_b, done_b, info_b = env_b.step(actions)
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(rew_a, rew_b)
        np.testing.assert_array_equal(done_a, done_b)
        np.testing.assert_array_equal(info_a["progress"], info_b["progress"])
    assert obs_a.shape == (5, 4) and rew_a.shape == (5,) and done_a.shape == (5,)
    for key in ("progress", "lap", "last_lap_time", "off_track"):
        assert info_a[key].shape == (5,)


def test_multi_track_env_crash_routing(config):
    spacing = config["track"]["resample_spacing"]
    tracks = [Track.load("oval", spacing), Track.load("chicane", spacing)]
    env = MultiTrackEnv(tracks, config, n_envs=4, seed=1)
    env.reset()
    spawn = env.state_snapshot()["state"]
    # Env 0 steers hard into the wall; the others brake and never move.
    actions = np.array([STEER_POS, BRAKE, BRAKE, BRAKE])
    done = info = None
    for _ in range(300):
        _, _, done, info = env.step(actions)
        if done[0]:
            break
    assert done is not None and done[0], "env 0 never crashed"
    np.testing.assert_array_equal(done, [True, False, False, False])
    np.testing.assert_array_equal(info["off_track"], [True, False, False, False])
    assert info["progress"][0] > 0.0
    np.testing.assert_array_equal(info["progress"][1:], 0.0)
    # The braked envs (including env 2, sharing env 0's track) are untouched.
    after = env.state_snapshot()["state"]
    np.testing.assert_array_equal(after[1:], spawn[1:])
    # Env 0 respawned on ITS track and keeps running.
    x0, y0, _ = tracks[0].start_pose()
    assert np.hypot(after[0, 0] - x0, after[0, 1] - y0) <= tracks[0].half_width
    _, _, done, _ = env.step(np.full(4, BRAKE))
    assert not np.any(done)


def test_multi_track_env_n_features_mismatch_raises(config, monkeypatch):
    real_env = multi_track.RacingEnv

    class SkewedEnv(real_env):
        def __init__(self, track, config, n_envs, seed):
            super().__init__(track, config, n_envs=n_envs, seed=seed)
            if track.name == "chicane":
                self.n_features += 1

    monkeypatch.setattr(multi_track, "RacingEnv", SkewedEnv)
    spacing = config["track"]["resample_spacing"]
    tracks = [Track.load("oval", spacing), Track.load("chicane", spacing)]
    with pytest.raises(ValueError, match="n_features"):
        MultiTrackEnv(tracks, config, n_envs=2, seed=0)


def test_multi_track_env_rejects_empty_and_zero_envs(config):
    with pytest.raises(ValueError, match="track"):
        MultiTrackEnv([], config, n_envs=2, seed=0)
    track = Track.load("oval", config["track"]["resample_spacing"])
    with pytest.raises(ValueError, match="n_envs"):
        MultiTrackEnv([track], config, n_envs=0, seed=0)


def test_random_pool_deterministic(config):
    a = MultiTrackEnv.random_pool(config, n_envs=4, seed=5, pool_size=3, difficulty=0.4)
    b = MultiTrackEnv.random_pool(config, n_envs=4, seed=5, pool_size=3, difficulty=0.4)
    assert len(a.tracks) == 3
    assert [t.name for t in a.tracks] == [t.name for t in b.tracks]
    for ta, tb in zip(a.tracks, b.tracks, strict=True):
        np.testing.assert_array_equal(ta.centerline, tb.centerline)
    np.testing.assert_array_equal(a.reset(), b.reset())

    other = MultiTrackEnv.random_pool(config, n_envs=4, seed=6, pool_size=3, difficulty=0.4)
    assert [t.name for t in other.tracks] != [t.name for t in a.tracks]


# -------------------------------------------------------------- train_headless


@pytest.mark.parametrize("track_arg", ["multi", "random"])
def test_train_headless_multi_and_random(tmp_path, track_arg):
    from traqmania.train_headless import train

    history_path = tmp_path / "history.json"
    summary = train("mlp", track_arg, episodes=10, seed=1, profile=None,
                    out_dir=str(tmp_path), history_path=str(history_path))
    assert len(summary["episode_returns"]) >= 10
    assert (tmp_path / f"mlp_{track_arg}.npz").exists()
    meta = json.loads((tmp_path / f"mlp_{track_arg}.meta.json").read_text())
    assert meta["track"] == track_arg
    payload = json.loads(history_path.read_text())
    assert payload["track"] == track_arg


# --------------------------------------------------------------- import hygiene


@pytest.mark.parametrize("module", ["traqmania.env.trackgen", "traqmania.env.multi_track"])
def test_new_env_modules_are_qiskit_free(module):
    check = (
        f"import {module}, sys; "
        f"assert 'qiskit' not in sys.modules, 'importing {module} pulled in qiskit'"
    )
    result = subprocess.run([sys.executable, "-c", check], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
