"""Micro-benchmarks for traQmania hot paths: env stepping, Q-network forwards,
and DQN gradient updates.

Run as ``python -m traqmania.bench [--n-envs 8]``.  Prints a small table of
throughput numbers; the whole run finishes in a few seconds.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from traqmania.agents.base import N_ACTIONS
from traqmania.agents.classical.mlp import MLPQFunction
from traqmania.agents.training.dqn import DQNTrainer
from traqmania.config import load_config
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track


def bench_env_steps(env: RacingEnv, n_steps: int = 200, warmup: int = 20) -> float:
    """Random-action env stepping; returns env.step() calls per second."""
    rng = np.random.default_rng(0)
    env.reset()
    for _ in range(warmup):
        env.step(rng.integers(N_ACTIONS, size=env.n_envs))
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(rng.integers(N_ACTIONS, size=env.n_envs))
    return n_steps / (time.perf_counter() - t0)


def bench_mlp_forward(qfunc: MLPQFunction, batch: int = 32, n_calls: int = 3000) -> float:
    """q_values forward passes per second at the given batch size."""
    obs = np.random.default_rng(1).random((batch, qfunc.n_features))
    for _ in range(50):
        qfunc.q_values(obs)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        qfunc.q_values(obs)
    return n_calls / (time.perf_counter() - t0)


def bench_dqn_update(trainer: DQNTrainer, n_updates: int = 200) -> float:
    """Seconds per double-DQN gradient update on a pre-filled replay buffer."""
    rng = np.random.default_rng(2)
    n_fill = max(1000, trainer.batch_size)
    trainer.buffer.add(
        rng.random((n_fill, trainer.qfunc.n_features)),
        rng.integers(trainer.qfunc.n_actions, size=n_fill),
        rng.normal(size=n_fill),
        rng.random((n_fill, trainer.qfunc.n_features)),
        (rng.random(n_fill) < 0.05).astype(float),
    )
    for _ in range(20):
        trainer._update()
    t0 = time.perf_counter()
    for _ in range(n_updates):
        trainer._update()
    return (time.perf_counter() - t0) / n_updates


def main() -> None:
    parser = argparse.ArgumentParser(description="traQmania micro-benchmarks")
    parser.add_argument("--n-envs", type=int, default=8, help="parallel sub-envs (default 8)")
    args = parser.parse_args()

    config = load_config()
    track = Track.load(config["track"]["default"], config["track"]["resample_spacing"])
    env = RacingEnv(track, config, n_envs=args.n_envs, seed=0)
    qfunc = MLPQFunction(n_features=env.n_features, n_actions=N_ACTIONS, seed=0)
    trainer = DQNTrainer(qfunc, env, config["training"])

    steps_per_s = bench_env_steps(env)
    forwards_per_s = bench_mlp_forward(qfunc)
    s_per_update = bench_dqn_update(trainer)

    rows = [
        (f"env steps/s (random actions, n_envs={args.n_envs})", f"{steps_per_s:,.0f}"),
        ("  car decision-steps/s (steps/s * n_envs)", f"{steps_per_s * args.n_envs:,.0f}"),
        ("MLP q_values forwards/s (batch 32)", f"{forwards_per_s:,.0f}"),
        ("DQN seconds/update (batch 32)", f"{s_per_update * 1e3:.3f} ms"),
    ]
    width = max(len(name) for name, _ in rows)
    print(f"traQmania benchmark — track '{track.name}'")
    print("-" * (width + 14))
    for name, value in rows:
        print(f"{name:<{width}}  {value:>10}")


if __name__ == "__main__":
    main()
