"""Model-based "hero" driver: racing line + speed profile + tracking controller.

This is the expert-demo reference driver, deliberately NOT a learned agent:
it reads the track geometry directly, computes a family of candidate racing
lines with physics-derived speed profiles, picks the best one by *simulating
itself* with the real car physics (closed loop, crash-free laps only), and
tracks it with continuous steering/throttle/brake — the RL agents are limited
to 4 bang-bang actions at 10 Hz. Everything derives from the same [physics]
constants the car simulation uses.
"""

from __future__ import annotations

import numpy as np

from traqmania.env.car import CarPhysics

RELAX_ITERATIONS = 3000
RELAX_ALPHA = 0.3
ROLLOUT_SECONDS = 60.0  # per-candidate simulation budget (standing + flying laps)


def _corner_radii(pts: np.ndarray) -> np.ndarray:
    """Per-point three-point circumradius of a closed polyline."""
    p0, p1, p2 = np.roll(pts, 1, axis=0), pts, np.roll(pts, -1, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    area2 = np.abs((p1 - p0)[:, 0] * (p2 - p0)[:, 1] - (p1 - p0)[:, 1] * (p2 - p0)[:, 0])
    with np.errstate(divide="ignore"):
        return np.where(area2 > 1e-12, a * b * c / (2.0 * area2), np.inf)


def _relax_shortest(center: np.ndarray, normals: np.ndarray, limit: float,
                    iterations: int = RELAX_ITERATIONS,
                    start: np.ndarray | None = None) -> np.ndarray:
    """Normal offsets of the shortest line in the corridor: iteratively pull
    each point toward its neighbours' midpoint (curve-shortening flow),
    movement restricted to the local normal so arc spacing stays sane."""
    offsets = np.zeros(len(center)) if start is None else start.copy()
    for _ in range(iterations):
        pts = center + normals * offsets[:, None]
        midpoints = 0.5 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))
        pull = np.sum((midpoints - pts) * normals, axis=1)
        offsets = np.clip(offsets + RELAX_ALPHA * pull, -limit, limit)
    return offsets


def _smooth_ring(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    pad = np.concatenate([x[-window:], x, x[:window]])
    return np.convolve(pad, kernel, mode="same")[window:-window]


def line_candidates(track, physics: dict) -> list[np.ndarray]:
    """Candidate racing lines inside the track.

    The shortest line (curve-shortening flow) is optimal wherever corners are
    fast enough to take flat out, but it hugs hairpin apexes at tiny radius —
    and this car's cornering speed *grows* with radius, so slow corners want
    to be driven wide. Candidates blend the shortest line toward (and
    slightly past) the centerline in slow-corner zones only, across a range
    of wall margins.
    """
    center = track.centerline
    normals = track.normals
    # corner radius above which v_safe(R) >= v_max: wider corners are free
    r_flat = (physics["v_max"] ** 2 + physics["v_turn"] ** 2) / (
        2.0 * physics["k_steer"] * physics["v_turn"])
    candidates = []
    for margin in (0.35, 0.55, 0.7, 0.85, 1.0, 1.3, 1.6):
        limit = max(0.0, track.half_width - margin)
        off_short = _relax_shortest(center, normals, limit)
        pts_short = center + normals * off_short[:, None]
        radii = np.minimum(_corner_radii(pts_short), 10.0 * r_flat)
        slow = np.clip((r_flat - _smooth_ring(radii, 9)) / r_flat, 0.0, 1.0)
        slow = _smooth_ring(slow, 15)
        for widen in (0.0, 0.3, 0.6, 0.9, 1.1, 1.3):
            off = np.clip(off_short * (1.0 - widen * slow), -limit, limit)
            off = _relax_shortest(center, normals, limit, iterations=40, start=off)
            candidates.append(center + normals * off[:, None])
    return candidates


def speed_profile(pts: np.ndarray, physics: dict, v_scale: float = 1.0) -> np.ndarray:
    """Target speed at each line point: corner-limited (v_safe from the
    steering-authority falloff), then made brake- and accel-feasible by
    backward/forward passes around the closed loop."""
    k_steer, v_turn = physics["k_steer"], physics["v_turn"]
    v_max, brake, accel = physics["v_max"], physics["brake"], physics["accel"]
    radii = _corner_radii(pts)
    with np.errstate(invalid="ignore"):
        v_safe = np.sqrt(np.maximum(0.0, 2.0 * k_steer * v_turn * radii - v_turn**2))
    # v_safe is the steady-state limit and conservative mid-corner; callers
    # may probe v_scale > 1 because the closed-loop rollout validates safety
    v = np.minimum(np.nan_to_num(v_safe * v_scale, posinf=v_max), v_max)
    ds = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    n = len(pts)
    a_brake = brake  # drag only adds braking power on top: this is a floor
    a_accel = accel * 0.6  # drag steals accel at speed; stay conservative
    for _ in range(2):  # ring: two passes propagate constraints across the seam
        for i in range(n - 1, -1, -1):  # braking: can we slow down in time?
            v[i] = min(v[i], np.sqrt(v[(i + 1) % n] ** 2 + 2.0 * a_brake * ds[i]))
        for i in range(n):  # acceleration: can we actually be that fast?
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * a_accel * ds[i]))
    return v


class _LineTracker:
    """Continuous (steer, throttle, brake) tracking of one line via pure
    pursuit + its speed profile, called at the 10 Hz decision rate."""

    def __init__(self, line: np.ndarray, physics: dict,
                 lookahead: tuple[float, float, float] = (0.5, 3.5, 11.0),
                 v_scale: float = 1.0) -> None:
        self.physics = physics
        self.lookahead = lookahead
        self.v_scale = v_scale
        self.line = line
        self.v_target = speed_profile(line, physics, v_scale)
        seg = np.linalg.norm(np.roll(line, -1, axis=0) - line, axis=1)
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
        gain, lo, hi = self.lookahead
        j = self._index_ahead(i, float(np.clip(gain * v, lo, hi)))
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


def _flying_lap(tracker: _LineTracker, track, physics: dict) -> float:
    """Best crash-free flying lap of a closed-loop simulation with the real
    car physics (10 Hz decisions, 60 Hz substeps), or inf on any off-track."""
    car = CarPhysics(physics)
    x, y, theta = track.start_pose()
    state = np.array([[x, y, theta, 0.0]])
    substeps = max(1, int(round((1.0 / 10.0) / physics["dt"])))
    total = track.total_length
    s_prev = float(track.project(state[:, :2])[0][0])
    progress, lap_start, t = 0.0, 0.0, 0.0
    laps: list[float] = []
    controls = (0.0, 0.0, 0.0)
    for step in range(int(ROLLOUT_SECONDS / physics["dt"])):
        if step % substeps == 0:
            controls = tracker(state[0])
        state = car.step(state, np.array([controls[0]]), np.array([controls[1]]),
                         np.array([controls[2]]))
        t += physics["dt"]
        if step % substeps == 0:
            if not bool(track.is_inside(state[:, :2])[0]):
                return float("inf")
            s = float(track.project(state[:, :2])[0][0])
            progress += (s - s_prev + total / 2.0) % total - total / 2.0
            s_prev = s
            if progress >= total:
                progress -= total
                laps.append(t - lap_start)
                lap_start = t
    return min(laps[1:]) if len(laps) > 1 else float("inf")  # flying laps only


class RacingLineController:
    """The hero driver: evaluates every candidate line by closed-loop
    simulation and drives the fastest crash-free one."""

    def __init__(self, track, physics: dict) -> None:
        self.physics = dict(physics)
        scored: list[tuple[float, _LineTracker]] = []
        for line in line_candidates(track, self.physics):
            for lookahead in ((0.5, 3.5, 11.0), (0.4, 3.0, 9.0), (0.62, 4.0, 13.0)):
                tracker = _LineTracker(line, self.physics, lookahead)
                lap = _flying_lap(tracker, track, self.physics)
                if np.isfinite(lap):
                    scored.append((lap, tracker))
        scored.sort(key=lambda pair: pair[0])
        best = scored[0][1] if scored else None
        best_lap = scored[0][0] if scored else np.inf
        # stage 2: the profile's v_safe is conservative — probe overspeed on
        # the top finishers (different lines tolerate it differently); the
        # rollout proves each probe crash-free before it can win
        for _base_lap, base in scored[:3]:
            for lookahead in (base.lookahead, (0.7, 4.0, 14.0), (0.85, 4.0, 16.0)):
                for v_scale in (1.0, 1.05, 1.1, 1.15, 1.2, 1.25):
                    if v_scale == 1.0 and lookahead == base.lookahead:
                        continue  # that is the stage-1 result itself
                    tracker = _LineTracker(base.line, self.physics, lookahead, v_scale)
                    lap = _flying_lap(tracker, track, self.physics)
                    if lap < best_lap:
                        best, best_lap = tracker, lap
        if best is None:  # nothing lapped cleanly: safest, widest fallback
            center, normals = track.centerline, track.normals
            limit = max(0.0, track.half_width - 1.6)
            off = _relax_shortest(center, normals, limit)
            best = _LineTracker(center + normals * (0.3 * off)[:, None], self.physics)
        self._tracker = best
        self.best_lap_simulated = best_lap  # honest: inf when the fallback drives
        self.line = best.line

    def __call__(self, state: np.ndarray) -> tuple[float, float, float]:
        return self._tracker(state)
