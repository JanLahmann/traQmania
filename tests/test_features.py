"""[observation] features: the default ["rays", "speed"] stays bit-identical
to the historical observation, each engineered kind (curvature_ahead,
lateral_offset, heading_error, corner_speed_ratio) is normalized to [0, 1] and
semantically sane on the oval, and the total feature count is validated
against [circuit] n_qubits."""

import copy
import math

import numpy as np
import pytest

from traqmania.config import load_config
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track

ENGINEERED = ["curvature_ahead", "lateral_offset", "heading_error", "corner_speed_ratio"]

# Oval centerline indices (ds ~ 1.5): the bottom straight runs s ~ 0..90, the
# first corner (|kappa| ~ track max) starts around s ~ 90.
MID_STRAIGHT = 20   # s ~ 30, corner far beyond the 15 m lookahead
APPROACH = 52       # s ~ 78, corner entry inside the lookahead window
IN_CORNER = 64      # s ~ 96, lookahead window fully inside the corner


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def oval(config):
    return Track.load("oval", config["track"]["resample_spacing"])


def feature_env(oval, config, features, n_qubits, n_envs=2, seed=0):
    cfg = copy.deepcopy(config)
    cfg["observation"]["features"] = features
    cfg["circuit"]["n_qubits"] = n_qubits
    return RacingEnv(oval, cfg, n_envs=n_envs, seed=seed)


def state_at(track, indices, lateral=0.0, heading_offset=0.0, v=0.0):
    """(len(indices), 4) car states placed relative to centerline points:
    ``lateral`` along the left normal, heading = tangent + ``heading_offset``."""
    idx = np.asarray(indices, dtype=np.intp)
    state = np.zeros((len(idx), 4))
    state[:, :2] = track.centerline[idx] + np.reshape(lateral, (-1, 1)) * track.normals[idx]
    state[:, 2] = np.arctan2(track.tangents[idx, 1], track.tangents[idx, 0]) + heading_offset
    state[:, 3] = v
    return state


def obs_at(env, indices, **kwargs):
    env.state = state_at(env.track, indices, **kwargs)
    return env._obs()


# ------------------------------------------------- default = historical obs

# Observations of RacingEnv(oval, load_config(), n_envs=3, seed=123) captured
# on main BEFORE the features generalization: at reset and after holding
# BASELINE_ACTIONS (one row of np.full(3, a) per entry).  The default config
# must reproduce them bit-identically.
BASELINE_ACTIONS = [1, 1, 2, 0, 3, 1, 1, 2, 1, 1]
BASELINE_RESET = np.array([
    [0.28567871264882555, 1.0, 0.25293494075766737, 0.0],
    [0.2033049266708596, 1.0, 0.3406125309988243, 0.0],
    [0.2348693520198997, 1.0, 0.30323969932433464, 0.0],
])
BASELINE_AFTER = np.array([
    [0.3151179032426825, 1.0, 0.23146381587323053, 0.25956627750041883],
    [0.22440470345266988, 1.0, 0.3120841158129513, 0.25956627750041883],
    [0.2732786378383157, 1.0, 0.2740512451705508, 0.25956627750041883],
])


def hand_rays_speed(env):
    """The pre-change observation recomputed by hand from env.state."""
    n_rays = len(env.ray_angles)
    origins = np.repeat(env.state[:, :2], n_rays, axis=0)
    angles = (env.state[:, 2][:, None] + env.ray_angles[None, :]).ravel()
    dist = env.track.raycast(origins, angles, env.ray_max_dist)
    rays = np.clip(dist.reshape(env.n_envs, n_rays) / env.ray_max_dist, 0.0, 1.0)
    v = np.clip(env.state[:, 3] / env.car.v_max, 0.0, 1.0)
    return np.concatenate([rays, v[:, None]], axis=1)


def test_default_obs_bit_identical_to_pre_change_capture(oval, config):
    assert config["observation"]["features"] == ["rays", "speed"]
    env = RacingEnv(oval, config, n_envs=3, seed=123)
    obs = env.reset()
    np.testing.assert_array_equal(obs, BASELINE_RESET)
    for a in BASELINE_ACTIONS:
        obs, _, _, _ = env.step(np.full(3, a))
    np.testing.assert_array_equal(obs, BASELINE_AFTER)


def test_default_obs_bit_identical_to_hand_computed_rays_speed(oval, config):
    env = RacingEnv(oval, config, n_envs=4, seed=5)
    obs = env.reset()
    np.testing.assert_array_equal(obs, hand_rays_speed(env))
    rng = np.random.default_rng(2)
    for _ in range(25):
        obs, _, _, _ = env.step(rng.integers(4, size=4))
        np.testing.assert_array_equal(obs, hand_rays_speed(env))


def test_missing_features_key_defaults_to_rays_speed(oval, config):
    cfg = copy.deepcopy(config)
    del cfg["observation"]["features"]
    env_default = RacingEnv(oval, cfg, n_envs=3, seed=123)
    env_explicit = RacingEnv(oval, config, n_envs=3, seed=123)
    assert env_default.features == ["rays", "speed"]
    np.testing.assert_array_equal(env_default.reset(), env_explicit.reset())
    for a in BASELINE_ACTIONS:
        obs_d, _, _, _ = env_default.step(np.full(3, a))
        obs_e, _, _, _ = env_explicit.step(np.full(3, a))
        np.testing.assert_array_equal(obs_d, obs_e)


def test_default_feature_names_and_count(oval, config):
    env = RacingEnv(oval, config, n_envs=1, seed=0)
    assert env.n_features == 4
    assert env.feature_names == ["ray -60°", "ray 0°", "ray +60°", "speed"]


# --------------------------------------------------------- engineered kinds


def test_engineered_features_shape_range_and_names(oval, config):
    env = feature_env(oval, config, ENGINEERED, n_qubits=4, n_envs=3, seed=9)
    assert env.n_features == 4
    assert env.feature_names == [
        "curvature ahead", "lateral offset", "heading error", "corner speed",
    ]
    obs = env.reset()
    assert obs.shape == (3, 4)
    rng = np.random.default_rng(1)
    for _ in range(50):
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        obs, _, _, _ = env.step(rng.integers(4, size=3))
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_curvature_ahead_rises_approaching_the_corner(oval, config):
    env = feature_env(oval, config, ENGINEERED, n_qubits=4, n_envs=3)
    obs = obs_at(env, [MID_STRAIGHT, APPROACH, IN_CORNER])
    far, approach, corner = obs[:, 0]
    assert far == 0.0  # the corner is beyond the lookahead window
    assert 0.0 < approach < corner  # corner entry enters the window
    assert corner > 0.8  # window holds ~the track's max |kappa| -> ~1


def test_lateral_offset_signed_and_centered(oval, config):
    env = feature_env(oval, config, ENGINEERED, n_qubits=4, n_envs=3)
    half = oval.half_width
    obs = obs_at(env, [MID_STRAIGHT] * 3, lateral=[-0.5 * half, 0.0, 0.5 * half])
    np.testing.assert_allclose(obs[:, 1], [0.25, 0.5, 0.75], atol=1e-6)


def test_heading_error_zero_when_aligned_and_wraps(oval, config):
    env = feature_env(oval, config, ENGINEERED, n_qubits=4, n_envs=1)
    aligned = obs_at(env, [MID_STRAIGHT])
    np.testing.assert_allclose(aligned[0, 2], 0.5, atol=1e-9)
    # wrapped: a full extra turn is still aligned
    full_turn = obs_at(env, [MID_STRAIGHT], heading_offset=2.0 * math.pi)
    np.testing.assert_allclose(full_turn[0, 2], 0.5, atol=1e-9)
    # signed: +0.5 rad left of the tangent maps above 0.5, -0.5 rad below
    left = obs_at(env, [MID_STRAIGHT], heading_offset=0.5)
    right = obs_at(env, [MID_STRAIGHT], heading_offset=-0.5)
    np.testing.assert_allclose(left[0, 2], 0.5 + 0.5 / (2.0 * math.pi), atol=1e-9)
    np.testing.assert_allclose(right[0, 2], 0.5 - 0.5 / (2.0 * math.pi), atol=1e-9)


def test_corner_speed_ratio_grows_with_speed(oval, config):
    env = feature_env(oval, config, ENGINEERED, n_qubits=4, n_envs=1)
    for idx in (MID_STRAIGHT, IN_CORNER):
        ratios = [obs_at(env, [idx], v=v)[0, 3] for v in (5.0, 10.0, 20.0)]
        assert ratios[0] < ratios[1] < ratios[2]


def test_corner_speed_ratio_matches_hand_computed_v_safe(oval, config):
    env = feature_env(oval, config, ENGINEERED, n_qubits=4, n_envs=1)
    k_steer = config["physics"]["k_steer"]
    v_turn = config["physics"]["v_turn"]
    lookahead = config["observation"]["lookahead_m"]
    v = 12.0
    for idx in (MID_STRAIGHT, APPROACH, IN_CORNER):
        env.state = state_at(oval, [idx], v=v)
        s_vals, _ = oval.project(env.state[:, :2])
        kappa = float(oval.curvature_ahead(s_vals, lookahead)[0])
        radius = 1.0 / max(kappa, 1e-6)
        v_safe = math.sqrt(max(0.0, 2.0 * k_steer * v_turn * radius - v_turn**2))
        expected = min(v / max(v_safe, 1e-6), 2.0) / 2.0
        np.testing.assert_allclose(env._obs()[0, 3], expected, rtol=1e-12)
    # sanity of the formula itself with the default physics numbers:
    # R = 25 -> v_safe = sqrt(2 * 2.6 * 9 * 25 - 81) = sqrt(1089) = 33
    if k_steer == 2.6 and v_turn == 9.0:
        assert math.sqrt(2.0 * k_steer * v_turn * 25.0 - v_turn**2) == 33.0


# ------------------------------------------------------- config validation


def test_feature_count_must_match_n_qubits(oval, config):
    cfg = copy.deepcopy(config)
    cfg["observation"]["features"] = ["rays", "speed", "curvature_ahead"]  # 5 scalars
    with pytest.raises(ValueError, match=r"5 scalars .* n_qubits = 4"):
        RacingEnv(oval, cfg, n_envs=1, seed=0)


def test_unknown_feature_kind_rejected(oval, config):
    cfg = copy.deepcopy(config)
    cfg["observation"]["features"] = ["rays", "warp_drive"]
    with pytest.raises(ValueError, match="unknown kind"):
        RacingEnv(oval, cfg, n_envs=1, seed=0)


def test_six_feature_mixed_config_builds_and_steps(oval, config):
    features = ["rays", "speed", "curvature_ahead", "corner_speed_ratio"]
    env = feature_env(oval, config, features, n_qubits=6, n_envs=2, seed=3)
    assert env.n_features == 6
    assert env.feature_names == [
        "ray -60°", "ray 0°", "ray +60°", "speed", "curvature ahead", "corner speed",
    ]
    obs = env.reset()
    assert obs.shape == (2, 6)
    # rays + speed prefix is exactly the historical observation
    np.testing.assert_array_equal(obs[:, :4], hand_rays_speed(env))
    rng = np.random.default_rng(4)
    for _ in range(30):
        obs, reward, done, _ = env.step(rng.integers(4, size=2))
        assert obs.shape == (2, 6)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert np.all(np.isfinite(reward))
