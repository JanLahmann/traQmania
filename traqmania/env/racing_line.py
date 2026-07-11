"""Model-based "hero" driver: racing line + speed profile + tracking controller.

This is the expert-demo reference driver, deliberately NOT a learned agent:
it reads the track geometry directly, computes a curvature-minimizing racing
line with a physics-derived speed profile, and tracks it with continuous
steering/throttle/brake (the RL agents are limited to 4 bang-bang actions at
10 Hz — that action set, not intelligence, caps their lap times). Everything
derives from the same [physics] constants the car simulation uses.
"""

from __future__ import annotations

import numpy as np

RELAX_ITERATIONS = 3000
RELAX_ALPHA = 0.3
EDGE_MARGIN = 1.2  # keep the line this far inside the walls, world units


def _corner_radii(pts: np.ndarray) -> np.ndarray:
    """Per-point three-point circumradius of a closed polyline."""
    p0, p1, p2 = np.roll(pts, 1, axis=0), pts, np.roll(pts, -1, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    area2 = np.abs((p1 - p0)[:, 0] * (p2 - p0)[:, 1] - (p1 - p0)[:, 1] * (p2 - p0)[:, 0])
    with np.errstate(divide="ignore"):
        return np.where(area2 > 1e-12, a * b * c / (2.0 * area2), np.inf)


def racing_line(track) -> np.ndarray:
    """Curvature-minimizing line inside the track: iteratively relax each
    point toward its neighbours' midpoint, constrained to the racing surface
    (movement restricted to the local normal so arc spacing stays sane)."""
    center = track.centerline
    normals = track.normals
    limit = max(0.0, track.half_width - EDGE_MARGIN)
    offsets = np.zeros(len(center))
    for _ in range(RELAX_ITERATIONS):
        pts = center + normals * offsets[:, None]
        midpoints = 0.5 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))
        pull = np.sum((midpoints - pts) * normals, axis=1)
        offsets = np.clip(offsets + RELAX_ALPHA * pull, -limit, limit)
    return center + normals * offsets[:, None]


def speed_profile(pts: np.ndarray, physics: dict) -> np.ndarray:
    """Target speed at each line point: corner-limited (v_safe from the
    steering-authority falloff), then made brake- and accel-feasible by
    backward/forward passes around the closed loop."""
    k_steer, v_turn = physics["k_steer"], physics["v_turn"]
    v_max, brake, accel = physics["v_max"], physics["brake"], physics["accel"]
    radii = _corner_radii(pts)
    with np.errstate(invalid="ignore"):
        v_safe = np.sqrt(np.maximum(0.0, 2.0 * k_steer * v_turn * radii - v_turn**2))
    v = np.minimum(np.nan_to_num(v_safe, posinf=v_max), v_max)
    ds = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    n = len(pts)
    a_brake = brake * 0.9  # small margin under the car's true braking power
    a_accel = accel * 0.6  # drag steals accel at speed; stay conservative
    for _ in range(2):  # ring: two passes propagate constraints across the seam
        for i in range(n - 1, -1, -1):  # braking: can we slow down in time?
            v[i] = min(v[i], np.sqrt(v[(i + 1) % n] ** 2 + 2.0 * a_brake * ds[i]))
        for i in range(n):  # acceleration: can we actually be that fast?
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * a_accel * ds[i]))
    return v


class RacingLineController:
    """Continuous (steer, throttle, brake) tracking of the racing line via
    pure pursuit + the speed profile. Called at the 10 Hz decision rate with
    the car state ``[x, y, theta, v]``."""

    def __init__(self, track, physics: dict) -> None:
        self.physics = dict(physics)
        self.line = racing_line(track)
        self.v_target = speed_profile(self.line, self.physics)
        seg = np.linalg.norm(np.roll(self.line, -1, axis=0) - self.line, axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        self.total = float(seg.sum())

    def _index_ahead(self, i: int, dist: float) -> int:
        target = (self.arc[i] + dist) % self.total
        return int(np.searchsorted(self.arc, target) % len(self.line))

    def __call__(self, state: np.ndarray) -> tuple[float, float, float]:
        x, y, theta, v = (float(s) for s in state)
        p = self.physics
        i = int(np.argmin(np.sum((self.line - [x, y]) ** 2, axis=1)))

        # pure pursuit on a speed-scaled lookahead point
        j = self._index_ahead(i, float(np.clip(0.5 * v, 3.5, 11.0)))
        tx, ty = self.line[j]
        dist = max(float(np.hypot(tx - x, ty - y)), 1e-6)
        alpha = (np.arctan2(ty - y, tx - x) - theta + np.pi) % (2.0 * np.pi) - np.pi
        omega_desired = v * 2.0 * np.sin(alpha) / dist
        falloff = 2.0 * v * p["v_turn"] / (p["v_turn"] ** 2 + v * v) if v > 1e-6 else 1e-6
        steer = float(np.clip(omega_desired / (p["k_steer"] * falloff), -1.0, 1.0))

        # the profile is already brake-feasible: track it with a small lead
        v_target = float(self.v_target[self._index_ahead(i, max(v, 2.0) * 0.12)])
        throttle = 1.0 if v < v_target - 0.2 else 0.0
        brake = 1.0 if v > v_target + 0.4 else 0.0
        return (steer, throttle, brake)
