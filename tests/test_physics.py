"""Tests for the batched car physics model."""

import numpy as np
import pytest

from traqmania.config import load_config
from traqmania.env.car import CarPhysics


@pytest.fixture
def physics_cfg():
    return dict(load_config()["physics"])


def _state(v, theta=0.0, n=1):
    s = np.zeros((n, 4))
    s[:, 2] = theta
    s[:, 3] = v
    return s


def _drive(car, state, steer, throttle, brake, n_steps):
    b = len(state)
    steer = np.full(b, steer, dtype=float)
    throttle = np.full(b, throttle, dtype=float)
    brake = np.full(b, brake, dtype=float)
    for _ in range(n_steps):
        state = car.step(state, steer, throttle, brake)
    return state


def test_speed_never_exceeds_vmax(physics_cfg):
    car = CarPhysics(physics_cfg)
    rng = np.random.default_rng(0)
    state = _state(v=physics_cfg["v_max"], n=16)
    for _ in range(600):
        steer = rng.choice([-1.0, 0.0, 1.0], size=16)
        throttle = rng.choice([0.0, 1.0], size=16)
        brake = rng.choice([0.0, 1.0], size=16)
        state = car.step(state, steer, throttle, brake)
        assert np.all(state[:, 3] <= physics_cfg["v_max"] + 1e-12)
        assert np.all(state[:, 3] >= 0.0)


def test_drag_only_decay(physics_cfg):
    car = CarPhysics(physics_cfg)
    v0, n = 15.0, 300
    state = _drive(car, _state(v=v0), steer=0, throttle=0, brake=0, n_steps=n)
    v = state[0, 3]
    assert v < v0  # decays
    # Discrete drag map: v_{k+1} = v_k * (1 - drag * dt); close to exp decay.
    expected = v0 * np.exp(-physics_cfg["drag"] * n * physics_cfg["dt"])
    assert v == pytest.approx(expected, rel=0.02)


def test_braking_stronger_than_drag(physics_cfg):
    car = CarPhysics(physics_cfg)
    v0, n = 15.0, 30
    coasting = _drive(car, _state(v=v0), steer=0, throttle=0, brake=0, n_steps=n)
    braking = _drive(car, _state(v=v0), steer=0, throttle=0, brake=1, n_steps=n)
    assert braking[0, 3] < coasting[0, 3] - 1.0


def test_steering_rate_peaks_near_v_turn(physics_cfg):
    cfg = dict(physics_cfg, drag=0.0)  # keep speed constant during the step
    car = CarPhysics(cfg)
    v_turn = cfg["v_turn"]

    def dtheta(v):
        state = _drive(car, _state(v=v), steer=1, throttle=0, brake=0, n_steps=1)
        return state[0, 2]

    peak = dtheta(v_turn)
    assert peak > dtheta(0.5 * v_turn)
    assert peak > dtheta(2.0 * v_turn)
    assert dtheta(0.0) == pytest.approx(0.0, abs=1e-12)
    # At the peak the steering rate equals k_steer.
    assert peak == pytest.approx(cfg["k_steer"] * cfg["dt"], rel=1e-9)


def test_straight_line_distance_integration(physics_cfg):
    cfg = dict(physics_cfg, drag=0.0)  # constant speed, no inputs
    car = CarPhysics(cfg)
    v0, n = 10.0, 600
    state = _drive(car, _state(v=v0), steer=0, throttle=0, brake=0, n_steps=n)
    expected = v0 * n * cfg["dt"]
    assert state[0, 0] == pytest.approx(expected, rel=0.01)
    assert state[0, 1] == pytest.approx(0.0, abs=1e-9)
