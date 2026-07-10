"""Batched car physics for traQmania.

Kinematic model with a single scalar speed per car, linear drag, and a
speed-dependent steering authority that peaks at ``v_turn`` (slow cars barely
turn, very fast cars understeer).  All operations are vectorized over a batch
axis so any number of cars advance in one numpy call.
"""

from __future__ import annotations

import numpy as np


class CarPhysics:
    """Advance a batch of car states by one physics substep at a time.

    Config keys (see ``[physics]`` in ``traqmania/config/default.toml``):
    ``dt``, ``accel``, ``brake``, ``drag``, ``v_max``, ``v_turn``,
    ``k_steer``, ``substeps_per_decision``.
    """

    def __init__(self, physics_cfg: dict):
        self.dt = float(physics_cfg["dt"])
        self.accel = float(physics_cfg["accel"])
        self.brake = float(physics_cfg["brake"])
        self.drag = float(physics_cfg["drag"])
        self.v_max = float(physics_cfg["v_max"])
        self.v_turn = float(physics_cfg["v_turn"])
        self.k_steer = float(physics_cfg["k_steer"])
        self.substeps_per_decision = int(physics_cfg["substeps_per_decision"])

    def steer_falloff(self, v: np.ndarray) -> np.ndarray:
        """Steering authority in [0, 1] as a function of speed.

        ``v / (1 + (v / v_turn)**2)`` normalized so its maximum over v >= 0
        (attained at ``v == v_turn``) equals 1.  Zero at v = 0.
        """
        v = np.asarray(v, dtype=np.float64)
        return 2.0 * v * self.v_turn / (self.v_turn**2 + v**2)

    def step(self, state, steer, throttle, brake):
        """Advance ONE substep of ``dt``.

        state (B,4) float64 [x, y, theta, v]; steer (B,) in {-1,0,1};
        throttle, brake (B,) in {0,1}.  Returns a new (B,4) array (the input
        is not mutated).  Speed is clipped to [0, v_max].  Semi-implicit
        Euler: speed is updated first, then heading and position use the
        updated speed.  dtheta = steer * k_steer * steer_falloff(v) * dt.
        """
        state = np.asarray(state, dtype=np.float64)
        steer = np.asarray(steer, dtype=np.float64)
        throttle = np.asarray(throttle, dtype=np.float64)
        brake = np.asarray(brake, dtype=np.float64)

        x, y, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        dv = (throttle * self.accel - brake * self.brake - self.drag * v) * self.dt
        v_new = np.clip(v + dv, 0.0, self.v_max)
        theta_new = theta + steer * self.k_steer * self.steer_falloff(v_new) * self.dt
        x_new = x + v_new * np.cos(theta_new) * self.dt
        y_new = y + v_new * np.sin(theta_new) * self.dt
        return np.stack([x_new, y_new, theta_new, v_new], axis=1)
