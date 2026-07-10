"""End-to-end smoke test: DQNTrainer + MLPQFunction solve a tiny vectorized corridor env.

The corridor env mirrors the racing env protocol (vectorized over n_envs, auto-reset
on done) without depending on it: the agent must learn to hold full throttle to
drive position from 0 to 1. Reward is per-step progress, so a solved episode
returns ~1.0. The training_cfg keys mirror [training] in config/default.toml.
"""

import numpy as np

from traqmania.agents.classical import MLPQFunction
from traqmania.agents.training import DQNTrainer

TRAINING_CFG = {
    "algo": "double_dqn",
    "episodes": 300,
    "replay_size": 5000,
    "batch_size": 32,
    "gamma": 0.98,
    "lr": 0.01,
    "target_sync_every": 100,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 100,
    "n_parallel_envs": 4,
    "seed": 0,
}


class CorridorEnv:
    """1-D corridor, vectorized over n_envs. obs = (pos, vel, 1 - pos, 0.5).

    Actions map to acceleration {-1, 0, +1, 0}; action 3 additionally brakes
    (halves velocity). reward = delta pos; done at pos >= 1 (success) or after
    60 steps; done sub-envs auto-reset.
    """

    DT = 0.1
    MAX_STEPS = 60
    ACCEL = np.array([-1.0, 0.0, 1.0, 0.0])

    def __init__(self, n_envs: int = 4):
        self.n_envs = n_envs
        self.pos = np.zeros(n_envs)
        self.vel = np.zeros(n_envs)
        self.steps = np.zeros(n_envs, dtype=int)

    def _obs(self) -> np.ndarray:
        return np.stack(
            [self.pos, self.vel, 1.0 - self.pos, np.full(self.n_envs, 0.5)], axis=1
        )

    def reset(self) -> np.ndarray:
        self.pos[:] = 0.0
        self.vel[:] = 0.0
        self.steps[:] = 0
        return self._obs()

    def step(self, actions: np.ndarray):
        vel = self.vel + self.ACCEL[actions] * self.DT
        vel = np.where(actions == 3, 0.5 * vel, vel)  # brake
        self.vel = np.clip(vel, -1.0, 1.0)
        new_pos = np.clip(self.pos + self.vel * self.DT, 0.0, 1.0)
        reward = new_pos - self.pos
        self.pos = new_pos
        self.steps += 1
        done = (self.pos >= 1.0) | (self.steps >= self.MAX_STEPS)

        # Auto-reset done sub-envs; done=1 masks bootstrapping so this obs is safe.
        self.pos[done] = 0.0
        self.vel[done] = 0.0
        self.steps[done] = 0
        return self._obs(), reward, done, {}


def test_dqn_learns_corridor():
    env = CorridorEnv(n_envs=TRAINING_CFG["n_parallel_envs"])
    qfunc = MLPQFunction(n_features=4, hidden=8, n_actions=4, seed=TRAINING_CFG["seed"])
    trainer = DQNTrainer(
        qfunc, env, TRAINING_CFG, rng=np.random.default_rng(TRAINING_CFG["seed"])
    )

    callback_calls = []
    history = trainer.train(callback=lambda ep, stats: callback_calls.append((ep, stats)))

    returns = np.asarray(history["episode_returns"])
    assert len(returns) >= TRAINING_CFG["episodes"]
    assert history["wall_time_s"] < 30.0
    assert len(history["losses"]) > 0

    mean_late = returns[-50:].mean()
    assert mean_late > 0.8, f"mean of last 50 episode returns {mean_late:.3f} <= 0.8"

    assert len(callback_calls) == len(returns)
    ep0, stats0 = callback_calls[0]
    assert ep0 == 0
    assert set(stats0) == {"returns", "epsilon", "loss"}
