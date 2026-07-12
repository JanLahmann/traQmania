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
from traqmania.env.multi_track import MultiTrackEnv
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
REPORT_EVERY = 20  # episodes per mean-return line
MULTI_TRACK_NAMES = ("oval", "chicane", "gp", "combo")  # the --track multi mixture


class CleanLapMonitor:
    """Env wrapper that records the first CLEAN lap seen during training.

    A clean lap means a sub-env reached ``lap >= 1`` while still on track;
    because off-track immediately ends an episode, any completed lap was
    driven entirely on the racing surface.  Records the 1-based index of the
    episode it happened in (the episode then in progress) and the wall-clock
    seconds since ``reset()``.
    """

    def __init__(self, env: RacingEnv | MultiTrackEnv) -> None:
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


def build_qfunc(agent: str, n_features: int, seed: int, config: dict,
                n_actions: int = N_ACTIONS):
    """Q-function factory: classical MLP baseline or the quantum circuit."""
    if agent == "mlp":
        hidden = int(config.get("mlp", {}).get("hidden", 8))
        return MLPQFunction(n_features=n_features, hidden=hidden,
                            n_actions=n_actions, seed=seed)
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
    """Write ``<agent>_<track>.npz`` (params) + ``.meta.json`` sidecar; returns npz path.

    Non-default circuit sizes get a ``_q<n>`` filename tag (both agents, so a
    q6/q8/q10 training run never clobbers the bundled 4-feature weights).
    """
    out_dir = Path(out_dir) if out_dir is not None else WEIGHTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n_qubits = int(config.get("circuit", {}).get("n_qubits", 4))
    qtag = "" if n_qubits == 4 else f"_q{n_qubits}"
    npz_path = out_dir / f"{agent}_{track_name}{qtag}.npz"
    np.savez(npz_path, params=qfunc.get_params())
    obs_cfg = config["observation"]
    meta = {
        "agent": agent,
        "track": track_name,
        "config_hash": config_hash(config),
        "episodes": episodes,
        # what the driver was trained to see; loaders (see
        # runtime.weights_observation) overlay this on the profile obs
        "observation": {
            "ray_angles_deg": [float(a) for a in obs_cfg["ray_angles_deg"]],
            "features": [str(k) for k in obs_cfg.get("features", ["rays", "speed"])],
            **({"lookahead_m": float(obs_cfg["lookahead_m"])}
               if "lookahead_m" in obs_cfg else {}),
        },
        # how many discrete actions the driver was trained to pick between;
        # loaders (see runtime.weights_actions) adopt this per driver so a
        # 6/8-action policy drives with the action table it learned
        "actions": {"n_actions": int(qfunc.n_actions)},
        "date": "DATE",
    }
    meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return npz_path


def train(agent: str, track_name: str, episodes: int | None, seed: int | None,
          profile: str | None, out_dir: str | None = None, init: str | None = None,
          history_path: str | None = None, actions: int | None = None,
          pace: bool = False) -> dict:
    """Build env/agent/trainer from config, train, print progress, save weights.

    ``actions`` overrides ``[circuit] n_actions`` (the 6/8-action scaled
    readout).  ``pace`` merges the ``[training_pace]`` fine-tune recipe onto
    the training config — low epsilon plus a per-decision time penalty, so the
    objective becomes lap time rather than reliable progress; meant to be
    combined with ``--init`` on an already-lapping snapshot.

    Returns a summary dict: episode returns, wall time, and first-clean-lap
    episode/second (None if no clean lap happened).
    """
    config = load_config(profile=profile)
    if actions is not None:
        config.setdefault("circuit", {})["n_actions"] = int(actions)
    training_cfg = dict(config["training"])
    if pace:
        pace_cfg = dict(config.get("training_pace", {}))
        config["reward"]["time_penalty"] = float(pace_cfg.pop("time_penalty", 0.5))
        training_cfg.update(pace_cfg)
        if init is None:
            print("WARNING: --pace without --init fine-tunes random weights; "
                  "expected use is on an already-lapping snapshot")
    if seed is not None:
        training_cfg["seed"] = seed
    seed = int(training_cfg["seed"])
    episodes = int(episodes) if episodes is not None else int(training_cfg["episodes"])

    spacing = config["track"]["resample_spacing"]
    n_parallel = training_cfg["n_parallel_envs"]
    if track_name in ("multi", "random"):
        # Mixture training: one policy over several tracks (the universal
        # candidates); weights are saved under the literal name multi/random.
        if track_name == "multi":
            tracks = [Track.load(name, spacing) for name in MULTI_TRACK_NAMES]
            env = MultiTrackEnv(tracks, config, n_envs=n_parallel, seed=seed)
        else:
            env = MultiTrackEnv.random_pool(config, n_envs=n_parallel, seed=seed)
            tracks = env.tracks
        save_name = track_name

        def env_factory(tracks=tracks, config=config, seed=seed) -> MultiTrackEnv:
            return MultiTrackEnv(tracks, config, n_envs=4, seed=seed + 10_000)
    else:
        track = Track.load(track_name, spacing)
        env = RacingEnv(track, config, n_envs=n_parallel, seed=seed)
        save_name = track.name

        def env_factory(track=track, config=config, seed=seed) -> RacingEnv:
            return RacingEnv(track, config, n_envs=4, seed=seed + 10_000)

    monitor = CleanLapMonitor(env)
    qfunc = build_qfunc(agent, env.n_features, seed, config, n_actions=env.n_actions)
    if qfunc.n_actions != env.n_actions:
        raise ValueError(f"agent has {qfunc.n_actions} actions but the env's "
                         f"action table has {env.n_actions}")
    if init is not None:
        qfunc.set_params(np.load(init)["params"])
        print(f"warm-started from {init}")

    trainer = DQNTrainer(qfunc, monitor, training_cfg, rng=np.random.default_rng(seed),
                         env_factory=env_factory)

    print(f"training agent={agent} track={save_name} episodes={episodes} seed={seed}")
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
    if "best_eval" in history:
        be = history["best_eval"]
        mean = f"{be['mean_lap']:.1f}s" if be["mean_lap"] is not None else "none"
        lap = f"{be['best_lap']:.1f}s" if be["best_lap"] is not None else "none"
        print(f"saved snapshot: episode {be['episode']} (greedy eval: "
              f"{be['lapped_episodes']}/{be['eval_episodes']} episodes lapped, "
              f"mean lap {mean}, best {lap})")

    npz_path = save_weights(qfunc, agent, save_name, config, episodes,
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
        payload = dict(summary, agent=agent, track=save_name, seed=seed, episodes=episodes)
        Path(history_path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="traQmania headless training")
    parser.add_argument("--agent", default="mlp", choices=["mlp", "quantum"],
                        help="Q-function backend")
    parser.add_argument("--track", default="oval",
                        help="track name (oval | chicane | gp), 'multi' (oval+chicane+gp "
                             "mixture) or 'random' (pool of generated tracks from --seed)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="sub-env episodes (default: [training].episodes)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: [training].seed)")
    parser.add_argument("--profile", default=None, help="config profile overlay (e.g. pi5)")
    parser.add_argument("--out", default=None, help="weights output dir (default: bundled)")
    parser.add_argument("--init", default=None, help="warm-start from a weights .npz")
    parser.add_argument("--history", default=None,
                        help="write returns/lap summary JSON to this path")
    parser.add_argument("--actions", type=int, default=None, choices=[4, 6, 8],
                        help="action-set size ([circuit] n_actions override): "
                             "6 adds trail braking, 8 adds half-steer")
    parser.add_argument("--pace", action="store_true",
                        help="pace fine-tune: merge [training_pace] (low epsilon "
                             "+ per-decision time penalty); combine with --init")
    args = parser.parse_args()
    train(args.agent, args.track, args.episodes, args.seed, args.profile,
          out_dir=args.out, init=args.init, history_path=args.history,
          actions=args.actions, pace=args.pace)


if __name__ == "__main__":
    main()
