"""Generate evolution-mode stage weights: train the quantum agent and snapshot
its parameters at fixed episode counts.

Run as ``python tools/make_stages.py [--track oval --seed 42]``.  Trains a
fresh :class:`QuantumQFunction` with the standard :class:`DQNTrainer` config
(``[training]`` merged with any per-track preset) and, via the trainer
callback, checkpoints ``get_params()`` copies at episodes [100, 250, 400, 800].
Each snapshot is saved to ``traqmania/weights/quantum_<track>_stage<i>.npz``
plus a ``.meta.json`` sidecar carrying the episode count that evolution mode
shows as the car label ("ep N").
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from traqmania.agents.quantum.qdqn import QuantumQFunction
from traqmania.agents.training import DQNTrainer
from traqmania.config import load_config
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track
from traqmania.server.runtime import WEIGHTS_DIR, resolve_training_cfg
from traqmania.train_headless import config_hash

STAGE_EPISODES = (100, 250, 400, 800)


def make_stages(track_name: str = "oval", seed: int = 42) -> list:
    """Train quantum on ``track_name`` and save the 4 stage snapshots; returns paths."""
    config = load_config()
    tcfg = resolve_training_cfg(config, track_name)
    tcfg["seed"] = seed

    track = Track.load(track_name, config["track"]["resample_spacing"])
    env = RacingEnv(track, config, n_envs=int(tcfg["n_parallel_envs"]), seed=seed)
    qfunc = QuantumQFunction(config["circuit"], seed=seed)
    trainer = DQNTrainer(qfunc, env, tcfg, rng=np.random.default_rng(seed))

    snapshots: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()

    def callback(episode: int, stats: dict) -> None:
        done = episode + 1
        if done % 50 == 0:
            print(f"episode {done:>4}  eps={stats['epsilon']:.2f}  "
                  f"wall={time.perf_counter() - t0:6.1f}s", flush=True)
        if done in STAGE_EPISODES and done not in snapshots:
            snapshots[done] = qfunc.get_params().copy()
            print(f"snapshot at episode {done}", flush=True)

    total = max(STAGE_EPISODES)
    print(f"training quantum track={track_name} episodes={total} seed={seed}", flush=True)
    trainer.train(episodes=total, callback=callback)

    paths = []
    for i, episodes in enumerate(STAGE_EPISODES, start=1):
        params = snapshots.get(episodes, qfunc.get_params().copy())
        npz_path = WEIGHTS_DIR / f"quantum_{track_name}_stage{i}.npz"
        np.savez(npz_path, params=params)
        meta = {
            "agent": "quantum",
            "track": track_name,
            "config_hash": config_hash(config),
            "episodes": episodes,
            "stage": i,
            "date": "DATE",
        }
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"saved {npz_path}", flush=True)
        paths.append(npz_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="generate evolution-mode stage weights")
    parser.add_argument("--track", default="oval", help="track name (default oval)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = parser.parse_args()
    make_stages(args.track, args.seed)


if __name__ == "__main__":
    main()
