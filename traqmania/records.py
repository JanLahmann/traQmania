"""Driver × track lap records: evaluate every bundled driver on every bundled
track — quantum specialists at every qubit count, the universal driver, the
classical MLP baselines, and the hero/pro reference controllers — and persist
the results for comparison.

Run as ``python -m traqmania.records [--episodes 12] [--drivers a,b]
[--tracks x,y] [--markdown]``.  Each (driver, track) cell is a greedy
evaluation over ``--episodes`` standing-start episodes (an episode ends at
the first crash or the decision budget); results merge into
``data/records.json`` keyed by (driver, track), so partial runs accumulate
and re-runs overwrite their own cells.  ``--markdown`` renders the stored
records as per-track comparison tables without evaluating anything.

Quantum drivers are evaluated under the observation recorded in their weights
``.meta.json`` (overlaid on the packaged q<n> profile) — the same rule the
demo server applies via ``runtime.weights_observation``.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from traqmania.agents.base import N_ACTIONS
from traqmania.agents.classical import MLPQFunction
from traqmania.config import load_config
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track
from traqmania.server.runtime import WEIGHTS_DIR, available_tracks, weights_observation

RECORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "records.json"
QUBIT_COUNTS = (4, 6, 8, 10)
QUANTUM_NAMES = ("oval", "chicane", "gp", "combo", "universal")
SCHEMA = 1


@dataclass
class Driver:
    """One evaluatable driver: a weights file (quantum/mlp) or a reference
    controller (hero/pro), plus the fully-resolved config it drives under."""

    id: str
    kind: str  # quantum | mlp | hero | pro
    home: str  # training track ("universal", or "-" for hero/pro)
    n_qubits: int | None
    path: Path | None
    config: dict = field(repr=False, default_factory=dict)


def _quantum_config(n_qubits: int, path: Path) -> dict:
    """Packaged q<n> profile with the weights' recorded observation overlaid."""
    config = load_config() if n_qubits == 4 else load_config(f"q{n_qubits}")
    obs = weights_observation(path)
    if obs:
        config = copy.deepcopy(config)
        config["observation"].update(obs)
    return config


def discover_drivers() -> list[Driver]:
    """Every driver the bundle can field, across all qubit counts."""
    drivers = []
    for n in QUBIT_COUNTS:
        qtag = "" if n == 4 else f"_q{n}"
        for name in QUANTUM_NAMES:
            path = WEIGHTS_DIR / f"quantum_{name}{qtag}.npz"
            if path.is_file():
                drivers.append(Driver(f"quantum_{name}{qtag}", "quantum", name, n,
                                      path, _quantum_config(n, path)))
    for name in QUANTUM_NAMES:
        path = WEIGHTS_DIR / f"mlp_{name}.npz"
        if path.is_file():
            drivers.append(Driver(f"mlp_{name}", "mlp", name, None, path, load_config()))
    drivers.append(Driver("hero", "hero", "-", None, None, load_config()))
    if (WEIGHTS_DIR / "mlp_pro.npz").is_file():
        drivers.append(Driver("pro", "pro", "-", None, WEIGHTS_DIR / "mlp_pro.npz",
                              load_config()))
    return drivers


def _load_qfunc(driver: Driver):
    params = np.load(driver.path)["params"]
    if driver.kind == "quantum":
        from traqmania.agents.quantum.qdqn import QuantumQFunction

        qfunc = QuantumQFunction(driver.config["circuit"])
    else:
        n_features = 4  # bundled mlp baselines use the default rays+speed obs
        hidden = (params.size - N_ACTIONS) // (n_features + 1 + N_ACTIONS)
        qfunc = MLPQFunction(n_features=n_features, hidden=hidden, n_actions=N_ACTIONS)
    qfunc.set_params(params)
    return qfunc


def _make_controller(driver: Driver, track: Track):
    if driver.kind == "hero":
        from traqmania.env.racing_line import RacingLineController

        return RacingLineController(track, driver.config["physics"])
    from traqmania.env.pro import N_FEATURES, ProController

    params = np.load(driver.path)["params"]
    hidden = (params.size - N_ACTIONS) // (N_FEATURES + 1 + N_ACTIONS)
    qfunc = MLPQFunction(n_features=N_FEATURES, hidden=hidden, n_actions=N_ACTIONS)
    qfunc.set_params(params)
    return ProController(track, driver.config["physics"], qfunc)


def _finish(laps: np.ndarray, done_mask: np.ndarray, lap_times: list[float],
            episodes: int) -> dict:
    lt = np.asarray(lap_times, dtype=np.float64)
    return {
        "episodes": episodes,
        "lapped_episodes": int(np.sum(laps >= 1)),
        "crashed_before_lap": int(np.sum(done_mask & (laps == 0))),
        "laps": int(lt.size),
        "best_s": round(float(lt.min()), 2) if lt.size else None,
        "mean_s": round(float(lt.mean()), 2) if lt.size else None,
        "p90_s": round(float(np.percentile(lt, 90)), 2) if lt.size else None,
    }


def _eval_batched(driver: Driver, track: Track, episodes: int, seed: int) -> dict:
    """Greedy qfunc rollout, all episodes as parallel sub-envs (frozen at
    their first done, so auto-reset never starts a second episode)."""
    env = RacingEnv(track, driver.config, n_envs=episodes, seed=seed)
    qfunc = _load_qfunc(driver)
    obs = env.reset()
    done_mask = np.zeros(episodes, dtype=bool)
    laps = np.zeros(episodes, dtype=int)
    prev_lap = np.zeros(episodes, dtype=int)
    lap_times: list[float] = []
    for _ in range(env.max_decisions + 1):
        obs, _, done, info = env.step(np.argmax(qfunc.q_values(obs), axis=1))
        cur_lap = np.asarray(info["lap"], dtype=int)
        lt = np.asarray(info["last_lap_time"], dtype=np.float64)
        event = (cur_lap > prev_lap) & ~done_mask
        lap_times.extend(lt[event].tolist())
        laps[event] += 1
        done_arr = np.asarray(done, dtype=bool)
        done_mask |= done_arr
        prev_lap = np.where(done_arr, 0, cur_lap)
        if done_mask.all():
            break
    return _finish(laps, done_mask, lap_times, episodes)


def _eval_controller(driver: Driver, track: Track, episodes: int, seed: int) -> dict:
    """Sequential single-env rollouts (a fresh controller per episode, in case
    a controller keeps internal state)."""
    laps = np.zeros(episodes, dtype=int)
    done_mask = np.zeros(episodes, dtype=bool)
    lap_times: list[float] = []
    for ep in range(episodes):
        env = RacingEnv(track, driver.config, n_envs=1, seed=seed + ep)
        controller = _make_controller(driver, track)
        env.reset()
        prev = 0
        for _ in range(env.max_decisions + 1):
            controls = np.asarray(controller(env.state[0]), dtype=np.float64)[None, :]
            _, _, done, info = env.step_controls(controls)
            cur = int(info["lap"][0])
            if cur > prev:
                lap_times.append(float(info["last_lap_time"][0]))
                laps[ep] += 1
            prev = cur
            if bool(done[0]):
                done_mask[ep] = True
                break
    return _finish(laps, done_mask, lap_times, episodes)


def evaluate(driver: Driver, track_name: str, episodes: int, seed: int = 20_000) -> dict:
    track = Track.load(track_name, driver.config["track"]["resample_spacing"])
    evaluator = _eval_batched if driver.path is not None and driver.kind in ("quantum", "mlp") \
        else _eval_controller
    result = evaluator(driver, track, episodes, seed)
    return {
        "driver": driver.id,
        "kind": driver.kind,
        "home_track": driver.home,
        "n_qubits": driver.n_qubits,
        "track": track_name,
        "date": time.strftime("%Y-%m-%d"),
        **result,
    }


# ------------------------------------------------------------------ storage


def load_records(path: Path = RECORDS_PATH) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": SCHEMA, "records": {}}
    data.setdefault("records", {})
    return data


def save_record(record: dict, path: Path = RECORDS_PATH) -> None:
    """Merge one (driver, track) cell into the records file."""
    data = load_records(path)
    data["schema"] = SCHEMA
    data["records"][f"{record['driver']}|{record['track']}"] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(path: Path = RECORDS_PATH) -> str:
    """Per-track comparison tables, fastest best lap first."""
    records = list(load_records(path)["records"].values())
    if not records:
        return "no records yet — run `python -m traqmania.records`"
    lines = []
    for track in sorted({r["track"] for r in records}):
        rows = sorted((r for r in records if r["track"] == track),
                      key=lambda r: (r["best_s"] is None, r["best_s"]))
        lines += [f"### {track}", "",
                  "| driver | kind | qubits | best | mean | laps | episodes lapped |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            fmt = lambda v: f"{v:.2f} s" if v is not None else "—"
            lines.append(
                f"| {r['driver']} | {r['kind']} | {r['n_qubits'] or '—'} "
                f"| {fmt(r['best_s'])} | {fmt(r['mean_s'])} | {r['laps']} "
                f"| {r['lapped_episodes']}/{r['episodes']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m traqmania.records",
        description="Evaluate bundled drivers on bundled tracks; persist lap records.")
    parser.add_argument("--episodes", type=int, default=12, help="episodes per cell")
    parser.add_argument("--drivers", default=None,
                        help="comma-separated driver ids (default: all bundled)")
    parser.add_argument("--tracks", default=None,
                        help="comma-separated track names (default: all bundled)")
    parser.add_argument("--out", default=None, help=f"records file (default {RECORDS_PATH})")
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--markdown", action="store_true",
                        help="render the stored records and exit (no evaluation)")
    args = parser.parse_args(argv)
    path = Path(args.out) if args.out else RECORDS_PATH

    if args.markdown:
        print(render_markdown(path))
        return

    drivers = discover_drivers()
    if args.drivers:
        wanted = {d.strip() for d in args.drivers.split(",")}
        unknown = wanted - {d.id for d in drivers}
        if unknown:
            parser.error(f"unknown drivers {sorted(unknown)} "
                         f"(bundled: {[d.id for d in drivers]})")
        drivers = [d for d in drivers if d.id in wanted]
    tracks = [t.strip() for t in args.tracks.split(",")] if args.tracks else available_tracks()

    for driver in drivers:
        for track_name in tracks:
            t0 = time.perf_counter()
            record = evaluate(driver, track_name, args.episodes, args.seed)
            save_record(record, path)
            best = f"{record['best_s']:.2f} s" if record["best_s"] is not None else "no laps"
            print(f"{driver.id:22s} on {track_name:10s} "
                  f"{record['lapped_episodes']:2d}/{record['episodes']} episodes lap, "
                  f"best {best:10s} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"records -> {path}")


if __name__ == "__main__":
    main()
