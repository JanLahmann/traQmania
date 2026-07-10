"""Headless training entry point for traQmania baselines.

Run as ``python -m traqmania.train_headless --agent mlp --track oval
[--episodes N --seed S --profile P]``.  Builds the vectorized racing env,
a Q-function, and the double-DQN trainer, trains for the requested number of
sub-env episodes, prints a per-20-episode mean-return trace, reports the first
episode (and wall-clock second) at which a full clean lap occurred, and saves
the learned weights to ``traqmania/weights/<agent>_<track>.npz`` plus a JSON
metadata sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from traqmania.agents.base import N_ACTIONS
from traqmania.agents.classical import MLPQFunction
from traqmania.agents.training import DQNTrainer
from traqmania.config import load_config
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
REPORT_EVERY = 20  # episodes per mean-return line


class CleanLapMonitor:
    """Env wrapper that records the first CLEAN lap seen during training.

    A clean lap means a sub-env reached ``lap >= 1`` while still on track;
    because off-track immediately ends an episode, any completed lap was
    driven entirely on the racing surface.  Records the 1-based index of the
    episode it happened in (the episode then in progress) and the wall-clock
    seconds since ``reset()``.
    """

    def __init__(self, env: RacingEnv) -> None:
        self.env = env
        self.episodes_done = 0
        self.first_clean_episode: int | None = None
        self.first_clean_wall_s: float | None = None
        self.best_lap_s: float = float("nan")
        self._t0 = time.perf_counter()

    def reset(self) -> np.ndarray:
        self._t0 = time.perf_counter()
        return self.env.reset()

    def step(self, actions: np.ndarray):
        obs, reward, done, info = self.env.step(actions)
        if self.first_clean_episode is None:
            clean = (info["lap"] >= 1) & ~info["off_track"]
            if np.any(clean):
                self.first_clean_episode = self.episodes_done + 1
                self.first_clean_wall_s = time.perf_counter() - self._t0
        laps = info["last_lap_time"]
        if np.any(~np.isnan(laps)):
            self.best_lap_s = np.nanmin([self.best_lap_s, np.nanmin(laps)])
        self.episodes_done += int(np.sum(done))
        return obs, reward, done, info


def build_qfunc(agent: str, n_features: int, seed: int, config: dict):
    """Q-function factory: classical MLP baseline or the quantum circuit."""
    if agent == "mlp":
        return MLPQFunction(n_features=n_features, n_actions=N_ACTIONS, seed=seed)
    if agent == "quantum":
        # numpy-only fast path (fastsim + adjoint); keeps headless training qiskit-free
        from traqmania.agents.quantum.qdqn import QuantumQFunction

        return QuantumQFunction(config["circuit"], seed=seed)
    raise ValueError(f"unknown agent '{agent}' (expected 'mlp' or 'quantum')")


def config_hash(config: dict) -> str:
    """Short stable hash of the fully-resolved config dict."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def save_weights(qfunc, agent: str, track_name: str, config: dict, episodes: int,
                 out_dir: Path | None = None) -> Path:
    """Write ``<agent>_<track>.npz`` (params) + ``.meta.json`` sidecar; returns npz path."""
    out_dir = Path(out_dir) if out_dir is not None else WEIGHTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{agent}_{track_name}.npz"
    np.savez(npz_path, params=qfunc.get_params())
    meta = {
        "agent": agent,
        "track": track_name,
        "config_hash": config_hash(config),
        "episodes": episodes,
        "date": "DATE",
    }
    meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return npz_path


def train(agent: str, track_name: str, episodes: int | None, seed: int | None,
          profile: str | None, out_dir: str | None = None, init: str | None = None,
          history_path: str | None = None) -> dict:
    """Build env/agent/trainer from config, train, print progress, save weights.

    Returns a summary dict: episode returns, wall time, and first-clean-lap
    episode/second (None if no clean lap happened).
    """
    config = load_config(profile=profile)
    training_cfg = dict(config["training"])
    if seed is not None:
        training_cfg["seed"] = seed
    seed = int(training_cfg["seed"])
    episodes = int(episodes) if episodes is not None else int(training_cfg["episodes"])

    track = Track.load(track_name, config["track"]["resample_spacing"])
    env = RacingEnv(track, config, n_envs=training_cfg["n_parallel_envs"], seed=seed)
    monitor = CleanLapMonitor(env)
    qfunc = build_qfunc(agent, env.n_features, seed, config)
    if init is not None:
        qfunc.set_params(np.load(init)["params"])
        print(f"warm-started from {init}")
    trainer = DQNTrainer(qfunc, monitor, training_cfg, rng=np.random.default_rng(seed))

    print(f"training agent={agent} track={track.name} episodes={episodes} seed={seed}")
    t0 = time.perf_counter()
    returns: list[float] = []

    def callback(episode: int, stats: dict) -> None:
        returns.append(stats["returns"])
        if (episode + 1) % REPORT_EVERY == 0:
            mean = float(np.mean(returns[-REPORT_EVERY:]))
            wall = time.perf_counter() - t0
            print(
                f"episode {episode + 1:>4}  mean return (last {REPORT_EVERY}): "
                f"{mean:>8.1f}  eps={stats['epsilon']:.2f}  wall={wall:6.1f}s"
            )

    history = trainer.train(episodes=episodes, callback=callback)

    print(f"trained {len(history['episode_returns'])} episodes "
          f"in {history['wall_time_s']:.1f}s wall clock")
    if monitor.first_clean_episode is not None:
        print(f"first clean lap: episode {monitor.first_clean_episode} "
              f"at {monitor.first_clean_wall_s:.1f}s wall clock")
    else:
        print("first clean lap: NEVER (no clean lap this run)")

    npz_path = save_weights(qfunc, agent, track.name, config, episodes,
                            out_dir=Path(out_dir) if out_dir else None)
    print(f"weights saved to {npz_path}")
    if not np.isnan(monitor.best_lap_s):
        print(f"best lap: {monitor.best_lap_s:.2f}s")

    summary = {
        "episode_returns": history["episode_returns"],
        "wall_time_s": history["wall_time_s"],
        "first_clean_episode": monitor.first_clean_episode,
        "first_clean_wall_s": monitor.first_clean_wall_s,
        "best_lap_s": None if np.isnan(monitor.best_lap_s) else float(monitor.best_lap_s),
    }
    if history_path:
        payload = dict(summary, agent=agent, track=track.name, seed=seed, episodes=episodes)
        Path(history_path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="traQmania headless training")
    parser.add_argument("--agent", default="mlp", choices=["mlp", "quantum"],
                        help="Q-function backend")
    parser.add_argument("--track", default="oval", help="track name (oval | chicane | gp)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="sub-env episodes (default: [training].episodes)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: [training].seed)")
    parser.add_argument("--profile", default=None, help="config profile overlay (e.g. pi5)")
    parser.add_argument("--out", default=None, help="weights output dir (default: bundled)")
    parser.add_argument("--init", default=None, help="warm-start from a weights .npz")
    parser.add_argument("--history", default=None,
                        help="write returns/lap summary JSON to this path")
    args = parser.parse_args()
    train(args.agent, args.track, args.episodes, args.seed, args.profile,
          out_dir=args.out, init=args.init, history_path=args.history)


if __name__ == "__main__":
    main()
