"""Integration tests for the vectorized RacingEnv on the oval track."""

import copy

import numpy as np
import pytest

from traqmania.config import load_config
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track

# ACTIONS indices: 0 = steer -1 (clockwise), 1 = straight, 2 = steer +1 (counter-
# clockwise, toward the +60 deg ray), 3 = coast-brake; all but 3 at full throttle.
STEER_NEG, STRAIGHT, STEER_POS, BRAKE = 0, 1, 2, 3


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def oval(config):
    return Track.load("oval", config["track"]["resample_spacing"])


def make_env(oval, config, n_envs=4, seed=0):
    return RacingEnv(oval, config, n_envs=n_envs, seed=seed)


def test_obs_bounds_and_shape(oval, config):
    env = make_env(oval, config, n_envs=6, seed=1)
    rng = np.random.default_rng(0)
    obs = env.reset()
    assert obs.shape == (6, 4)
    for _ in range(100):
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        obs, reward, done, info = env.step(rng.integers(4, size=6))
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_driving_straight_gains_progress(oval, config):
    env = make_env(oval, config, n_envs=2, seed=7)
    env.reset()
    total_reward = np.zeros(2)
    for _ in range(20):
        _, reward, done, info = env.step(np.full(2, STRAIGHT))
        total_reward += reward
        assert not np.any(done)
    assert np.all(info["progress"] > 0.0)
    assert np.all(total_reward > 0.0)
    assert np.all(info["lap"] == 0)
    assert np.all(np.isnan(info["last_lap_time"]))


def test_steering_into_wall_terminates_with_penalty(oval, config):
    env = make_env(oval, config, n_envs=3, seed=2)
    env.reset()
    for _ in range(200):
        _, reward, done, info = env.step(np.full(3, STEER_POS))
        if np.any(done):
            break
    assert np.any(done)
    assert np.all(info["off_track"][done])
    # Off-track penalty (10) dominates any single-decision progress (< 2.5).
    assert np.all(reward[done] < 0.0)


def test_scripted_controller_completes_a_lap(oval, config):
    env = make_env(oval, config, n_envs=1, seed=3)
    obs = env.reset()
    lap_time = np.nan
    for _ in range(config["reward"]["max_decisions"]):
        right, left = obs[:, 0], obs[:, 2]  # rays at -60 and +60 degrees
        actions = np.where(
            left > right + 0.03, STEER_POS, np.where(right > left + 0.03, STEER_NEG, STRAIGHT)
        )
        obs, _, done, info = env.step(actions)
        if not np.isnan(info["last_lap_time"][0]):
            lap_time = info["last_lap_time"][0]
            break
        assert not info["off_track"][0]
    assert not np.isnan(lap_time), "controller failed to complete a lap within max_decisions"
    assert info["lap"][0] == 1
    # Sanity: lap time between flat-out minimum and the episode cap.
    episode_cap_s = config["reward"]["max_decisions"] * env.decision_dt
    assert oval.total_length / config["physics"]["v_max"] < lap_time < episode_cap_s


def test_timeout_done_without_penalty(oval, config):
    cfg = copy.deepcopy(config)
    cfg["reward"]["max_decisions"] = 5
    env = RacingEnv(oval, cfg, n_envs=2, seed=0)
    env.reset()
    for step in range(5):
        _, reward, done, info = env.step(np.full(2, BRAKE))  # cars never move
        assert np.all(done) == (step == 4)
    assert np.all(done)
    assert not np.any(info["off_track"])
    assert np.all(reward == 0.0)  # no progress, no penalty


def test_auto_reset_spawns_at_start(oval, config):
    env = make_env(oval, config, n_envs=3, seed=5)
    env.reset()
    obs = None
    for _ in range(200):
        obs, _, done, _ = env.step(np.full(3, STEER_NEG))
        if np.any(done):
            break
    assert np.any(done)
    # The obs returned alongside done is already the fresh spawn: speed reset
    # to zero and the car placed back at the start line (within jitter).
    x0, y0, _ = oval.start_pose()
    for i in np.flatnonzero(done):
        assert obs[i, 3] == 0.0
        assert np.hypot(env.state[i, 0] - x0, env.state[i, 1] - y0) <= oval.half_width
    # The reset envs keep running normally afterwards.
    obs, reward, done, _ = env.step(np.full(3, STRAIGHT))
    assert not np.any(done)


def test_determinism_same_seed(oval, config):
    env_a = make_env(oval, config, n_envs=4, seed=11)
    env_b = make_env(oval, config, n_envs=4, seed=11)
    obs_a, obs_b = env_a.reset(), env_b.reset()
    np.testing.assert_array_equal(obs_a, obs_b)
    rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
    for _ in range(50):
        obs_a, rew_a, done_a, info_a = env_a.step(rng_a.integers(4, size=4))
        obs_b, rew_b, done_b, info_b = env_b.step(rng_b.integers(4, size=4))
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(rew_a, rew_b)
        np.testing.assert_array_equal(done_a, done_b)
        np.testing.assert_array_equal(info_a["progress"], info_b["progress"])
