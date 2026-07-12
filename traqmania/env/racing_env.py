"""Vectorized racing environment: batched cars on a closed-loop track.

Glues :class:`~traqmania.env.car.CarPhysics` and :class:`~traqmania.env.track.Track`
into a gym-style vector env.  One ``step()`` holds a discrete action (see
``traqmania.agents.base.ACTIONS``) for ``substeps_per_decision`` physics
substeps, then observes the configured ``[observation] features`` (default:
normalized lidar rays plus normalized speed) and rewards signed arc-length
progress with checkpoint/lap bonuses and an off-track penalty.  Everything is
vectorized over the ``n_envs`` batch axis; done sub-envs auto-reset to the
start pose with small seeded jitter.
"""

from __future__ import annotations

import math

import numpy as np

from traqmania.agents.base import ACTIONS
from traqmania.env.car import CarPhysics
from traqmania.env.track import Track

# Spawn jitter: lateral offset as a fraction of half_width, heading in radians.
LATERAL_JITTER_FRAC = 0.25
HEADING_JITTER_RAD = 0.06

# Display label per single-scalar feature kind; "rays" expands to one scalar
# (and one label) per configured ray angle.
FEATURE_LABELS = {
    "speed": "speed",
    "curvature_ahead": "curvature ahead",
    "lateral_offset": "lateral offset",
    "heading_error": "heading error",
    "corner_speed_ratio": "corner speed",
}
FEATURE_KINDS = ("rays", *FEATURE_LABELS)

# Kinds that need the car's centerline projection / the lookahead curvature.
_PROJECTED_KINDS = frozenset(FEATURE_LABELS) - {"speed"}
_CURVATURE_KINDS = frozenset({"curvature_ahead", "corner_speed_ratio"})


def _ray_label(deg: float) -> str:
    return "ray 0°" if deg == 0 else f"ray {deg:+g}°"


class CarObserver:
    """The ``[observation]`` feature pipeline, factored out of the env so the
    demo server's live cars observe EXACTLY what headless training observed
    (including the engineered feature kinds).

    Parses and validates the observation config once; :meth:`observe` turns a
    batch of ``(n, 4)`` ``[x, y, theta, v]`` car states into ``(n,
    n_features)`` observations.  :class:`RacingEnv` delegates its ``_obs``
    here; the server calls :meth:`observe` per live car.

    ``rays_slice`` is the position of the rays block inside the observation
    (or None when ``features`` omits ``"rays"``) — viewers use it to pull the
    per-ray distances back out for display.
    """

    def __init__(self, track: Track, config: dict, car: CarPhysics | None = None):
        self.track = track
        self.car = car if car is not None else CarPhysics(config["physics"])

        obs_cfg = config["observation"]
        self.ray_angles = np.deg2rad(np.asarray(obs_cfg["ray_angles_deg"], dtype=np.float64))
        self.ray_max_dist = float(obs_cfg["ray_max_dist"])
        self.lookahead_m = float(obs_cfg.get("lookahead_m", 15.0))
        self.features = [str(kind) for kind in obs_cfg.get("features", ["rays", "speed"])]
        unknown = [kind for kind in self.features if kind not in FEATURE_KINDS]
        if unknown:
            raise ValueError(
                f"[observation] features: unknown kind(s) {unknown}; "
                f"known kinds: {list(FEATURE_KINDS)}"
            )
        self.feature_names: list[str] = []
        self.rays_slice: slice | None = None
        for kind in self.features:
            if kind == "rays":
                self.rays_slice = slice(len(self.feature_names),
                                        len(self.feature_names) + len(self.ray_angles))
                self.feature_names += [_ray_label(d) for d in obs_cfg["ray_angles_deg"]]
            else:
                self.feature_names.append(FEATURE_LABELS[kind])
        self.n_features = len(self.feature_names)
        self._needs_projection = bool(_PROJECTED_KINDS.intersection(self.features))
        self._needs_curvature = bool(_CURVATURE_KINDS.intersection(self.features))

        circuit_cfg = config.get("circuit")
        if circuit_cfg is not None and "n_qubits" in circuit_cfg:
            n_qubits = int(circuit_cfg["n_qubits"])
            if self.n_features != n_qubits:
                raise ValueError(
                    f"[observation] features {self.features} produce "
                    f"{self.n_features} scalars ({self.feature_names}) but "
                    f"[circuit] n_qubits = {n_qubits}; the circuit encodes one "
                    "feature per qubit — adjust features/ray_angles_deg or n_qubits"
                )

    def observe(self, state: np.ndarray) -> np.ndarray:
        """(n, 4) states -> (n, n_features): the configured feature blocks,
        concatenated in ``features`` order (default: normalized lidar rays
        then normalized speed).  Every scalar is in [0, 1]:

        - rays: distance to the boundary / ray_max_dist per ray angle.
        - speed: v / v_max.
        - curvature_ahead: max centerline |kappa| over ``lookahead_m`` ahead
          of the car's projection / the track's max |kappa|.
        - lateral_offset: signed centerline offset d / half_width -> (d+1)/2.
        - heading_error: wrapped signed angle to the track tangent e / pi
          -> (e+1)/2 (0.5 = aligned).
        - corner_speed_ratio: v / v_safe(R) with R = 1/max(|kappa_ahead|,
          1e-6) and v_safe = sqrt(max(0, 2*k_steer*v_turn*R - v_turn^2)),
          clipped to [0, 2] then halved (0.5 = exactly at the safe speed).
        """
        state = np.asarray(state, dtype=np.float64)
        n = state.shape[0]
        theta, v = state[:, 2], state[:, 3]
        if self._needs_projection:
            s_vals, lateral = self.track.project(state[:, :2])
        if self._needs_curvature:
            kappa_ahead = self.track.curvature_ahead(s_vals, self.lookahead_m)

        blocks = []
        for kind in self.features:
            if kind == "rays":
                n_rays = len(self.ray_angles)
                origins = np.repeat(state[:, :2], n_rays, axis=0)
                angles = (theta[:, None] + self.ray_angles[None, :]).ravel()
                dist = self.track.raycast(origins, angles, self.ray_max_dist)
                blocks.append(
                    np.clip(dist.reshape(n, n_rays) / self.ray_max_dist, 0.0, 1.0)
                )
                continue
            if kind == "speed":
                feat = np.clip(v / self.car.v_max, 0.0, 1.0)
            elif kind == "curvature_ahead":
                feat = np.clip(kappa_ahead / max(self.track.max_abs_curvature, 1e-9), 0.0, 1.0)
            elif kind == "lateral_offset":
                feat = np.clip((lateral / self.track.half_width + 1.0) / 2.0, 0.0, 1.0)
            elif kind == "heading_error":
                err = (theta - self.track.tangent_angle(s_vals) + np.pi) % (2.0 * np.pi) - np.pi
                feat = np.clip((err / np.pi + 1.0) / 2.0, 0.0, 1.0)
            else:  # corner_speed_ratio; kinds were validated in __init__
                radius = 1.0 / np.maximum(kappa_ahead, 1e-6)
                v_safe = np.sqrt(
                    np.maximum(0.0, 2.0 * self.car.k_steer * self.car.v_turn * radius
                               - self.car.v_turn**2)
                )
                feat = np.clip(v / np.maximum(v_safe, 1e-6), 0.0, 2.0) / 2.0
            blocks.append(feat[:, None])
        return np.concatenate(blocks, axis=1)


class RacingEnv:
    """Batched racing env with auto-reset.

    Public attributes: ``state`` (n_envs, 4) float64 [x, y, theta, v],
    ``lap`` (n_envs,) completed-lap counts, ``last_lap_time`` (n_envs,)
    seconds (nan until a lap is finished), ``n_features`` (observation width),
    ``feature_names`` (display label per observation scalar).

    obs concatenates the ``[observation] features`` kinds in order, every
    scalar normalized to [0, 1].  The default ``["rays", "speed"]`` is
    [each ray / ray_max_dist clipped to [0, 1], v / v_max]; ray angles are
    relative to the car heading and come from ``ray_angles_deg``.  The
    engineered kinds ``curvature_ahead``, ``lateral_offset``,
    ``heading_error`` and ``corner_speed_ratio`` derive from the car's
    centerline projection (see ``_obs``).
    """

    def __init__(self, track: Track, config: dict, n_envs: int, seed: int):
        self.track = track
        self.n_envs = int(n_envs)
        self.car = CarPhysics(config["physics"])

        # The observer owns observation parsing/validation; the attributes
        # below stay as aliases for existing callers.
        self.observer = CarObserver(track, config, car=self.car)
        self.ray_angles = self.observer.ray_angles
        self.ray_max_dist = self.observer.ray_max_dist
        self.lookahead_m = self.observer.lookahead_m
        self.features = self.observer.features
        self.feature_names = self.observer.feature_names
        self.n_features = self.observer.n_features

        reward_cfg = config["reward"]
        self.progress_scale = float(reward_cfg["progress_scale"])
        self.offtrack_penalty = float(reward_cfg["offtrack_penalty"])
        self.lap_bonus = float(reward_cfg["lap_bonus"])
        self.checkpoint_bonus = float(reward_cfg["checkpoint_bonus"])
        self.max_decisions = int(reward_cfg["max_decisions"])

        self.decision_dt = self.car.dt * self.car.substeps_per_decision
        self._rng = np.random.default_rng(seed)

        actions = np.asarray(ACTIONS, dtype=np.float64)  # (A, 3)
        self._steer_tab = actions[:, 0]
        self._throttle_tab = actions[:, 1]
        self._brake_tab = actions[:, 2]

        self._cp_s = np.asarray(track.checkpoints, dtype=np.float64) * track.total_length

        n = self.n_envs
        self.state = np.zeros((n, 4))
        self.lap = np.zeros(n, dtype=np.int64)
        self.last_lap_time = np.full(n, np.nan)
        self._s = np.zeros(n)  # wrapped arc-length position of each car
        self._progress = np.zeros(n)  # unwrapped signed arc-length since spawn
        self._decisions = np.zeros(n, dtype=np.int64)
        self._lap_start = np.zeros(n, dtype=np.int64)  # decision count at lap start
        self._cp_hit = np.zeros((n, len(self._cp_s)), dtype=bool)

    # ---------------------------------------------------------------- spawning

    def _spawn(self, mask: np.ndarray) -> None:
        """Place fresh cars at the start pose (small seeded lateral/heading jitter)
        for every env where ``mask`` is True, and zero their per-episode trackers."""
        k = int(mask.sum())
        if k == 0:
            return
        x0, y0, h0 = self.track.start_pose()
        lateral = self._rng.uniform(-1.0, 1.0, k) * LATERAL_JITTER_FRAC * self.track.half_width
        heading = h0 + self._rng.uniform(-1.0, 1.0, k) * HEADING_JITTER_RAD
        nx, ny = -math.sin(h0), math.cos(h0)  # left normal at the start line

        fresh = np.zeros((k, 4))
        fresh[:, 0] = x0 + lateral * nx
        fresh[:, 1] = y0 + lateral * ny
        fresh[:, 2] = heading
        self.state[mask] = fresh

        s_vals, _ = self.track.project(fresh[:, :2])
        self._s[mask] = s_vals
        self._progress[mask] = 0.0
        self.lap[mask] = 0
        self.last_lap_time[mask] = np.nan
        self._decisions[mask] = 0
        self._lap_start[mask] = 0
        self._cp_hit[mask] = False

    # -------------------------------------------------------------------- api

    def reset(self) -> np.ndarray:
        """Respawn every car; returns obs (n_envs, n_features)."""
        self._spawn(np.ones(self.n_envs, dtype=bool))
        return self._obs()

    def step(self, actions: np.ndarray):
        """Hold each action for ``substeps_per_decision`` substeps.

        actions (n_envs,) int indices into ``ACTIONS``.  Returns
        ``(obs, reward, done, info)``; done sub-envs are auto-reset, so the
        returned obs is the fresh spawn while ``info`` reflects the state at
        the end of the decision (before the reset).  info keys: ``progress``,
        ``lap``, ``last_lap_time`` (seconds or nan), ``off_track``.
        """
        actions = np.asarray(actions, dtype=np.intp)
        return self._advance(self._steer_tab[actions], self._throttle_tab[actions],
                             self._brake_tab[actions])

    def step_controls(self, controls: np.ndarray):
        """Like :meth:`step` but holding raw ``(n_envs, 3)`` [steer, throttle,
        brake] controls — for reference controllers (hero/pro) that don't act
        through the discrete action set."""
        controls = np.asarray(controls, dtype=np.float64)
        return self._advance(controls[:, 0], controls[:, 1], controls[:, 2])

    def _advance(self, steer: np.ndarray, throttle: np.ndarray, brake: np.ndarray):
        state = self.state
        for _ in range(self.car.substeps_per_decision):
            state = self.car.step(state, steer, throttle, brake)
        self.state = state

        total = self.track.total_length
        s_new, lateral = self.track.project(state[:, :2])
        # Wraparound-aware signed arc-length gain, |delta_s| < total_length / 2.
        delta_s = (s_new - self._s + 0.5 * total) % total - 0.5 * total
        reward = self.progress_scale * delta_s

        # Checkpoint bonuses: first forward crossing of each fraction this lap.
        if self._cp_s.size:
            fwd = (self._cp_s[None, :] - self._s[:, None]) % total  # (n, C)
            crossed = (
                (delta_s[:, None] > 0.0)
                & (fwd > 1e-9)
                & (fwd <= delta_s[:, None])
                & ~self._cp_hit
            )
            reward += self.checkpoint_bonus * crossed.sum(axis=1)
            self._cp_hit |= crossed

        self._s = s_new
        self._progress = self._progress + delta_s
        self._decisions += 1

        # Lap completion: one full track length of net progress since spawn.
        laps_now = np.floor(self._progress / total).astype(np.int64)
        lap_done = laps_now > self.lap
        if np.any(lap_done):
            reward[lap_done] += self.lap_bonus
            elapsed = self._decisions[lap_done] - self._lap_start[lap_done]
            self.last_lap_time[lap_done] = elapsed * self.decision_dt
            self._lap_start[lap_done] = self._decisions[lap_done]
            self.lap[lap_done] = laps_now[lap_done]
            self._cp_hit[lap_done] = False

        off_track = np.abs(lateral) > self.track.half_width
        reward = reward - self.offtrack_penalty * off_track
        done = off_track | (self._decisions >= self.max_decisions)  # timeout: no penalty

        info = {
            "progress": self._progress.copy(),
            "lap": self.lap.copy(),
            "last_lap_time": self.last_lap_time.copy(),
            "off_track": off_track.copy(),
        }

        if np.any(done):
            self._spawn(done)
        return self._obs(), reward, done, info

    def state_snapshot(self) -> dict:
        """Copies of the live car arrays for external viewers (e.g. the demo server
        sampling a training env from another thread): ``state`` (n_envs, 4)
        [x, y, theta, v], ``lap``, ``last_lap_time`` (nan until a lap), ``progress``.
        """
        return {
            "state": self.state.copy(),
            "lap": self.lap.copy(),
            "last_lap_time": self.last_lap_time.copy(),
            "progress": self._progress.copy(),
        }

    # ------------------------------------------------------------ observations

    def _obs(self) -> np.ndarray:
        """(n_envs, n_features) via the shared :class:`CarObserver` — see
        :meth:`CarObserver.observe` for the feature definitions."""
        return self.observer.observe(self.state)
