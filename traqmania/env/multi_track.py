"""Multi-track vectorized environment: one batch of cars over several tracks.

:class:`MultiTrackEnv` distributes ``n_envs`` sub-envs round-robin over one
:class:`~traqmania.env.racing_env.RacingEnv` per track (global sub-env ``i``
drives track ``i % len(tracks)``) and concatenates obs/reward/done and the
info arrays back into that fixed global order, so the DQN trainer and its
monitor wrappers see the exact single-env interface (``reset``/``step`` plus
``n_envs``/``n_features``/``feature_names``) and train a single policy on a
mixture of tracks unchanged.
"""

from __future__ import annotations

import numpy as np

from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track
from traqmania.env.trackgen import generate_track


class MultiTrackEnv:
    """Round-robin mixture of per-track RacingEnvs behind the RacingEnv interface.

    Public attributes: ``tracks`` (list of Track), ``track_index`` (n_envs,)
    mapping each global sub-env to its track, and the RacingEnv-compatible
    ``n_envs``, ``n_features``, ``feature_names``, ``decision_dt``.  Tracks
    beyond ``n_envs`` in the round-robin get no sub-envs and stay idle.
    """

    def __init__(self, tracks: list[Track], config: dict, n_envs: int, seed: int):
        if not tracks:
            raise ValueError("MultiTrackEnv needs at least one track")
        self.tracks = list(tracks)
        self.n_envs = int(n_envs)
        if self.n_envs < 1:
            raise ValueError(f"MultiTrackEnv needs n_envs >= 1, got {n_envs}")

        n_tracks = len(self.tracks)
        self.track_index = np.arange(self.n_envs) % n_tracks
        counts = np.bincount(self.track_index, minlength=n_tracks)
        # Sub-envs in track order; the k-th car of track t is global env t + k*T.
        self._envs = [
            (t, RacingEnv(self.tracks[t], config, n_envs=int(counts[t]), seed=seed + t))
            for t in range(n_tracks)
            if counts[t] > 0
        ]
        # Gather map: global env i sits at offset[i % T] + i // T in the
        # concatenation of the sub-env outputs (track order).
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self._gather = offsets[self.track_index] + np.arange(self.n_envs) // n_tracks

        first = self._envs[0][1]
        for _t, env in self._envs[1:]:
            if env.n_features != first.n_features:
                raise ValueError(
                    f"MultiTrackEnv: tracks disagree on n_features "
                    f"({first.n_features} vs {env.n_features})"
                )
        self.n_features = first.n_features
        self.feature_names = list(first.feature_names)
        self.n_actions = first.n_actions
        self.decision_dt = first.decision_dt

    @classmethod
    def random_pool(cls, config: dict, n_envs: int, seed: int, pool_size: int = 16,
                    difficulty: float = 0.5) -> MultiTrackEnv:
        """MultiTrackEnv over a deterministic pool of generated tracks: the
        per-track generator seeds derive from ``seed``, so the same
        ``(seed, pool_size, difficulty)`` always yields the same pool."""
        rng = np.random.default_rng(seed)
        track_seeds = rng.integers(0, 2**31 - 1, size=int(pool_size))
        spacing = config["track"]["resample_spacing"]
        tracks = [
            generate_track(int(s), resample_spacing=spacing, difficulty=difficulty)
            for s in track_seeds
        ]
        return cls(tracks, config, n_envs, seed)

    # -------------------------------------------------------------------- api

    def reset(self) -> np.ndarray:
        """Respawn every car on its track; returns obs (n_envs, n_features)."""
        obs = np.concatenate([env.reset() for _t, env in self._envs], axis=0)
        return obs[self._gather]

    def step(self, actions: np.ndarray):
        """Route each global action to its track's sub-env and merge the
        results back into global env order; same contract as RacingEnv.step."""
        actions = np.asarray(actions)
        n_tracks = len(self.tracks)
        obs_parts, reward_parts, done_parts, info_parts = [], [], [], []
        for t, env in self._envs:
            obs_t, reward_t, done_t, info_t = env.step(actions[t::n_tracks])
            obs_parts.append(obs_t)
            reward_parts.append(reward_t)
            done_parts.append(done_t)
            info_parts.append(info_t)
        g = self._gather
        info = {
            key: np.concatenate([part[key] for part in info_parts])[g]
            for key in info_parts[0]
        }
        return (
            np.concatenate(obs_parts, axis=0)[g],
            np.concatenate(reward_parts)[g],
            np.concatenate(done_parts)[g],
            info,
        )

    def state_snapshot(self) -> dict:
        """Merged copies of the live car arrays in global env order (same keys
        as ``RacingEnv.state_snapshot``)."""
        parts = [env.state_snapshot() for _t, env in self._envs]
        return {
            key: np.concatenate([part[key] for part in parts], axis=0)[self._gather]
            for key in parts[0]
        }
