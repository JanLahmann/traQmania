"""Vectorized racing environment: batched cars on a closed-loop track.

Glues :class:`~traqmania.env.car.CarPhysics` and :class:`~traqmania.env.track.Track`
into a gym-style vector env.  One ``step()`` holds a discrete action (see
``traqmania.agents.base.ACTIONS``) for ``substeps_per_decision`` physics
substeps, then observes three normalized lidar rays plus normalized speed and
rewards signed arc-length progress with checkpoint/lap bonuses and an
off-track penalty.  Everything is vectorized over the ``n_envs`` batch axis;
done sub-envs auto-reset to the start pose with small seeded jitter.
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


class RacingEnv:
    """Batched racing env with auto-reset.

    Public attributes: ``state`` (n_envs, 4) float64 [x, y, theta, v],
    ``lap`` (n_envs,) completed-lap counts, ``last_lap_time`` (n_envs,)
    seconds (nan until a lap is finished), ``n_features`` (observation width).

    obs = [ray(-60deg), ray(0deg), ray(+60deg) each / ray_max_dist clipped to
    [0, 1], v / v_max]; ray angles are relative to the car heading and come
    from the ``[observation]`` config section.
    """

    def __init__(self, track: Track, config: dict, n_envs: int, seed: int):
        self.track = track
        self.n_envs = int(n_envs)
        self.car = CarPhysics(config["physics"])

        obs_cfg = config["observation"]
        self.ray_angles = np.deg2rad(np.asarray(obs_cfg["ray_angles_deg"], dtype=np.float64))
        self.ray_max_dist = float(obs_cfg["ray_max_dist"])
        self.n_features = len(self.ray_angles) + 1

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
        steer = self._steer_tab[actions]
        throttle = self._throttle_tab[actions]
        brake = self._brake_tab[actions]

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
        """(n_envs, n_features): normalized lidar rays then normalized speed."""
        n_rays = len(self.ray_angles)
        origins = np.repeat(self.state[:, :2], n_rays, axis=0)
        angles = (self.state[:, 2][:, None] + self.ray_angles[None, :]).ravel()
        dist = self.track.raycast(origins, angles, self.ray_max_dist)
        rays = np.clip(dist.reshape(self.n_envs, n_rays) / self.ray_max_dist, 0.0, 1.0)
        v = np.clip(self.state[:, 3] / self.car.v_max, 0.0, 1.0)
        return np.concatenate([rays, v[:, None]], axis=1)
