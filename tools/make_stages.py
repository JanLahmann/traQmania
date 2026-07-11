"""Generate HONEST evolution-mode stage weights.

Run as ``python tools/make_stages.py [--track oval --seed 42]``.  Trains a
fresh :class:`QuantumQFunction` with the standard :class:`DQNTrainer` config
(``[training]`` merged with any per-track preset) for 800 episodes, snapshots
``get_params()`` copies every 50 episodes, then GREEDY-EVALS every snapshot
(4 episodes on a fresh env, same scoring as ``DQNTrainer._greedy_eval``:
lexicographic ``(laps, -best_lap)``).  Four checkpoints with STRICTLY
improving eval score spanning early -> late are selected (first snapshot that
completes a lap, two intermediates, the best) and saved to
``traqmania/weights/quantum_<track>_stage<i>.npz`` plus a ``.meta.json``
sidecar carrying ``episodes`` (shown as the "ep N" car label in evolution
mode), ``eval_laps`` and ``eval_best_lap``.

Rationale: raw parameters at fixed episode counts are NOT monotonically
better — DQN policy churn made an ep-400 snapshot beat ep-800 on screen.
Eval-based selection guarantees the evolution-mode story (later stage =
better driver) is actually true.
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

N_STAGES = 4
SNAPSHOT_EVERY = 50
TOTAL_EPISODES = 800
EVAL_EPISODES = 4


def greedy_eval(
    params: np.ndarray,
    track: Track,
    config: dict,
    seed: int,
    eval_episodes: int = EVAL_EPISODES,
    max_steps: int = 5000,
) -> tuple[int, float, float]:
    """Greedy (epsilon = 0) eval of a parameter snapshot on a fresh env.

    Returns ``(laps_completed, best_lap_seconds, mean_step_return)`` over
    ``eval_episodes`` finished episodes; ``best_lap`` is ``+inf`` when no lap
    completed.  Laps and lap time mirror ``DQNTrainer._greedy_eval`` so scores
    are comparable; the mean per-step reward is a progress signal that still
    orders snapshots on tracks where greedy laps stay rare (see
    ``select_stages``)."""
    env = RacingEnv(track, config, n_envs=4, seed=seed + 10_000)
    qfunc = QuantumQFunction(config["circuit"], seed=seed)
    qfunc.set_params(params)

    obs = env.reset()
    finished = 0
    laps = 0
    best_lap = float("inf")
    reward_sum = 0.0
    steps = 0
    for _ in range(max_steps):
        actions = np.argmax(qfunc.q_values(obs), axis=1)
        obs, reward, done, info = env.step(actions)
        reward_sum += float(np.sum(reward))
        steps += env.n_envs
        lap_times = np.asarray(info["last_lap_time"], dtype=np.float64)
        if np.any(~np.isnan(lap_times)):
            best_lap = min(best_lap, float(np.nanmin(lap_times)))
        laps += int(np.sum(np.asarray(info["lap"])[np.flatnonzero(done)]))
        finished += int(np.sum(done))
        if finished >= eval_episodes:
            break
    return laps, best_lap, reward_sum / max(steps, 1)


def select_stages(scores: list[tuple]) -> list[int]:
    """Pick ``N_STAGES`` snapshot indices with strictly improving eval score.

    ``scores[i]`` is the lexicographic score ``(laps, -best_lap, return)`` of
    the i-th snapshot (episode order).  The longest strictly-increasing chain
    ending at the best-scoring snapshot is computed (O(n^2) DP); the returned
    four indices are the chain start (preferring the first member that
    completed a lap), the chain end (the best snapshot) and two intermediates
    spread evenly between them.
    """
    n = len(scores)
    best_idx = max(range(n), key=lambda i: (scores[i], -i))  # earliest best

    length = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if scores[j] < scores[i] and length[j] + 1 > length[i]:
                length[i] = length[j] + 1
                prev[i] = j
    chain: list[int] = []
    k = best_idx
    while k != -1:
        chain.append(k)
        k = prev[k]
    chain.reverse()

    if len(chain) < N_STAGES:
        raise RuntimeError(
            f"only {len(chain)} strictly improving snapshots found "
            f"(need {N_STAGES}); scores: {scores}"
        )
    # Prefer starting at the first chain member that completes a lap, as long
    # as at least N_STAGES chain members remain after it.
    start = next(
        (k for k, idx in enumerate(chain)
         if scores[idx][0] > 0 and len(chain) - k >= N_STAGES),
        0,
    )
    sub = chain[start:]
    if len(sub) < N_STAGES:
        sub = chain[-N_STAGES:]
    return [sub[round(t * (len(sub) - 1) / (N_STAGES - 1))] for t in range(N_STAGES)]


def make_stages(track_name: str = "oval", seed: int = 42,
                episodes: int = TOTAL_EPISODES, profile: str | None = None,
                init: str | None = None) -> list:
    """Train quantum on ``track_name``, eval-select 4 stages; returns saved paths.

    Also saves the last snapshot BEFORE the first lap-completing one as the
    track's ``_warmstart`` checkpoint (the warm live-training demo needs a
    pre-first-lap starting point), so one run per track regenerates the whole
    evolution + warm-start weight family. ``profile`` (e.g. ``q6``) trains at
    that circuit size and names files by the usual ``_q{n}`` rule."""
    config = load_config(profile=profile)
    tcfg = resolve_training_cfg(config, track_name)
    tcfg["seed"] = seed

    track = Track.load(track_name, config["track"]["resample_spacing"])
    env = RacingEnv(track, config, n_envs=int(tcfg["n_parallel_envs"]), seed=seed)
    qfunc = QuantumQFunction(config["circuit"], seed=seed)
    if init is not None:
        # warm start (tracks where fresh training never laps greedily — the
        # stage story then shows the fine-tune progression, honestly labelled
        # by ITS episode numbers)
        qfunc.set_params(np.load(init)["params"])
        print(f"warm-started from {init}", flush=True)
    trainer = DQNTrainer(qfunc, env, tcfg, rng=np.random.default_rng(seed))

    snapshots: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()

    def callback(episode: int, stats: dict) -> None:
        done = episode + 1
        if done % SNAPSHOT_EVERY == 0 and done not in snapshots:
            snapshots[done] = qfunc.get_params().copy()
            print(f"episode {done:>4}  eps={stats['epsilon']:.2f}  "
                  f"wall={time.perf_counter() - t0:6.1f}s  (snapshot)", flush=True)

    print(f"training quantum track={track_name} episodes={episodes} seed={seed}"
          f"{f' profile={profile}' if profile else ''}", flush=True)
    trainer.train(episodes=episodes, callback=callback)

    episodes_sorted = sorted(snapshots)
    scores: list[tuple[int, float]] = []
    print(f"\ngreedy-evaluating {len(episodes_sorted)} snapshots "
          f"({EVAL_EPISODES} episodes each)", flush=True)
    for episodes in episodes_sorted:
        laps, best_lap, mean_ret = greedy_eval(snapshots[episodes], track, config, seed)
        # laps and lap time dominate; the per-step-return tiebreak keeps
        # snapshots strictly ordered on tracks where greedy laps stay rare
        scores.append((laps, -best_lap, round(mean_ret, 4)))
        lap_txt = f"{best_lap:6.2f}s" if np.isfinite(best_lap) else "   --  "
        print(f"ep {episodes:>4}  laps={laps}  best_lap={lap_txt}  "
              f"return/step={mean_ret:7.4f}", flush=True)

    n_qubits = int(config["circuit"]["n_qubits"])
    qtag = "" if n_qubits == 4 else f"_q{n_qubits}"

    # warm-start checkpoint FIRST (before stage selection, which can fail):
    # the last snapshot with zero greedy laps before the first lap-completing
    # one — the "about to get it" starting point
    first_lapping = next((i for i, score in enumerate(scores) if score[0] > 0), None)
    if first_lapping is not None and first_lapping > 0:
        warm_idx = first_lapping - 1
        warm_ep = episodes_sorted[warm_idx]
        warm_path = WEIGHTS_DIR / f"quantum_{track_name}_warmstart{qtag}.npz"
        np.savez(warm_path, params=snapshots[warm_ep])
        warm_meta = {
            "agent": "quantum",
            "track": track_name,
            "config_hash": config_hash(config),
            "episodes": warm_ep,
            "provenance": "pre-first-lap checkpoint (make_stages warmstart export)",
            "date": time.strftime("%Y-%m-%d"),
        }
        warm_path.with_suffix("").with_suffix(".meta.json").write_text(
            json.dumps(warm_meta, indent=2) + "\n", encoding="utf-8")
        print(f"saved {warm_path}  (ep {warm_ep}, pre-first-lap)", flush=True)
    else:
        print("WARNING: no pre-first-lap snapshot (lapped in the first "
              "snapshot or never) — warmstart NOT saved", flush=True)

    picked = select_stages(scores)
    print(f"\nselected stages: {[episodes_sorted[i] for i in picked]}", flush=True)

    paths = []
    for stage, idx in enumerate(picked, start=1):
        episodes = episodes_sorted[idx]
        laps, neg_lap = scores[idx][0], scores[idx][1]
        best_lap = -neg_lap
        npz_path = WEIGHTS_DIR / f"quantum_{track_name}_stage{stage}{qtag}.npz"
        np.savez(npz_path, params=snapshots[episodes])
        meta = {
            "agent": "quantum",
            "track": track_name,
            "config_hash": config_hash(config),
            "episodes": episodes,
            "stage": stage,
            "eval_episodes": EVAL_EPISODES,
            "eval_laps": laps,
            "eval_best_lap": None if np.isinf(best_lap) else round(float(best_lap), 3),
            "date": time.strftime("%Y-%m-%d"),
        }
        meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"saved {npz_path}  (ep {episodes}, laps={laps})", flush=True)
        paths.append(npz_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="generate evolution-mode stage weights")
    parser.add_argument("--track", default="oval", help="track name (default oval)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES,
                        help="training episodes (gp/combo want their preset lengths)")
    parser.add_argument("--profile", default=None,
                        help="config profile overlay (e.g. q6) — files get the _q{n} tag")
    parser.add_argument("--init", default=None,
                        help="warm-start the stage training from a weights .npz")
    args = parser.parse_args()
    make_stages(args.track, args.seed, args.episodes, args.profile, args.init)


if __name__ == "__main__":
    main()
