"""The "pro" driver: the biggest classical agent we train, same recipe as all
the others.

A wide MLP Q-network trained with the standard double-DQN loop (no special
algorithm — just more parameters than the baselines) on the multi-track
mixture, sensing through a rich egocentric observation: 9 lidar rays, speed
and the four track-aware scalars. This module only provides inference: the
observation builder (matching ``RacingEnv._obs`` exactly for the pro training
profile) and a session-facing controller that argmaxes the trained Q-network
over the same 4 discrete actions every agent uses.
"""

from __future__ import annotations

import numpy as np

from traqmania.agents.base import ACTIONS

# must match the promlp training profile: 9 rays over [-60, +60] deg + speed
# + curvature_ahead + lateral_offset + heading_error + corner_speed_ratio
RAY_ANGLES = np.radians(np.linspace(-60.0, 60.0, 9))
RAY_MAX_DIST = 30.0
LOOKAHEAD_M = 15.0
N_FEATURES = 14


def observe(track, physics: dict, state: np.ndarray) -> np.ndarray:
    """(14,) observation for one car state [x, y, theta, v], identical to
    ``RacingEnv._obs`` under the pro training profile."""
    x, y, theta, v = (float(s) for s in state)
    origins = np.tile([[x, y]], (len(RAY_ANGLES), 1))
    dist = track.raycast(origins, theta + RAY_ANGLES, RAY_MAX_DIST)
    rays = np.clip(dist / RAY_MAX_DIST, 0.0, 1.0)
    speed = np.clip(v / physics["v_max"], 0.0, 1.0)
    s_vals, lateral = track.project([[x, y]])
    kappa = float(track.curvature_ahead(s_vals, LOOKAHEAD_M)[0])
    curv = np.clip(kappa / max(track.max_abs_curvature, 1e-9), 0.0, 1.0)
    lat = np.clip((float(lateral[0]) / track.half_width + 1.0) / 2.0, 0.0, 1.0)
    err = (theta - float(track.tangent_angle(s_vals)[0]) + np.pi) % (2.0 * np.pi) - np.pi
    head = np.clip((err / np.pi + 1.0) / 2.0, 0.0, 1.0)
    radius = 1.0 / max(kappa, 1e-6)
    k, vt = physics["k_steer"], physics["v_turn"]
    v_safe = np.sqrt(max(0.0, 2.0 * k * vt * radius - vt * vt))
    ratio = np.clip(v / max(v_safe, 1e-6), 0.0, 2.0) / 2.0
    return np.concatenate([rays, [speed, curv, lat, head, ratio]])


class ProController:
    """Session-facing controller for the trained pro Q-network: builds the
    rich observation from the car state and picks the greedy action — the
    same 4 discrete actions every trained agent drives with."""

    def __init__(self, track, physics: dict, qfunc) -> None:
        self.track = track
        self.physics = dict(physics)
        self.qfunc = qfunc

    def __call__(self, state: np.ndarray) -> tuple[float, float, float]:
        obs = observe(self.track, self.physics, state)[None, :]
        action = int(np.argmax(self.qfunc.q_values(obs)[0]))
        return tuple(float(c) for c in ACTIONS[action])
