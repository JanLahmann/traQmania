"""Server-side glue: bundled-agent loading, TrackPayload building, and training
config resolution (per-track presets + warm-start recipes from default.toml)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from traqmania.agents.base import N_ACTIONS
from traqmania.agents.classical import MLPQFunction
from traqmania.agents.quantum.qdqn import QuantumQFunction
from traqmania.config import load_config
from traqmania.env.track import TRACKS_DIR, Track

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
GHOSTS_DIR = Path(__file__).resolve().parent.parent / "data" / "ghosts"
N_EVOLUTION_STAGES = 4


TRACK_ORDER = ("oval", "chicane", "gp", "combo")  # simple -> complex, UI order


def available_tracks() -> list[str]:
    """Bundled track names, simple-to-complex (unknown extras sorted last)."""
    found = {p.stem for p in TRACKS_DIR.glob("*.json")}
    return [n for n in TRACK_ORDER if n in found] + sorted(found - set(TRACK_ORDER))


def load_track(config: dict, name: str) -> Track:
    return Track.load(name, config["track"]["resample_spacing"])


def load_agent(kind: str, track_name: str, warm: bool = False, config: dict | None = None):
    """Build a Q-function of ``kind`` ('quantum' | 'mlp') with bundled weights loaded.

    ``warm=True`` (quantum only) loads ``quantum_<track>_warmstart.npz`` instead of
    the fully-trained weights.  Quantum weight filenames gain a ``_q<n>`` tag when
    ``config`` sets ``circuit.n_qubits`` to anything other than the default 4.
    """
    if config is None:
        config = load_config()
    if kind == "mlp":
        qfunc: Any = MLPQFunction(n_features=4, n_actions=N_ACTIONS)
        path = WEIGHTS_DIR / f"mlp_{track_name}.npz"
    elif kind == "quantum":
        qfunc = QuantumQFunction(config["circuit"])
        # n-qubit filename rule: no tag at the default 4 qubits, _q<n> otherwise.
        n_qubits = int(config.get("circuit", {}).get("n_qubits", 4))
        qtag = "" if n_qubits == 4 else f"_q{n_qubits}"
        suffix = "_warmstart" if warm else ""
        path = WEIGHTS_DIR / f"quantum_{track_name}{suffix}{qtag}.npz"
    else:
        raise ValueError(f"unknown agent kind '{kind}' (expected 'quantum' or 'mlp')")

    qfunc.set_params(np.load(path)["params"])
    return qfunc


def _weights_label(path: Path, fallback: str) -> str:
    """Evolution-car label from a weights file's .meta.json ('ep N'), or ``fallback``."""
    meta_path = path.with_suffix("").with_suffix(".meta.json")
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            episodes = json.load(f)["episodes"]
        return f"ep {int(episodes)}"
    except (OSError, ValueError, KeyError, TypeError):
        return fallback


def evolution_stage_specs(track_name: str) -> list[tuple[str, Path]]:
    """(label, weights_path) for the 4 evolution-mode cars on ``track_name``.

    Prefers the bundled ``quantum_<track>_stage{1..4}.npz`` snapshots; when none
    exist, falls back to [warmstart, final] duplicated to 4 cars.
    """
    specs = [
        (_weights_label(path, f"stage {i}"), path)
        for i in range(1, N_EVOLUTION_STAGES + 1)
        if (path := WEIGHTS_DIR / f"quantum_{track_name}_stage{i}.npz").is_file()
    ]
    if specs:
        best = WEIGHTS_DIR / f"quantum_{track_name}.npz"
        if best.is_file():
            # the last car runs the shipped best-snapshot driver rather than
            # the last training snapshot (final-episode params drift)
            specs[-1] = ("best", best)
        return specs
    warm = WEIGHTS_DIR / f"quantum_{track_name}_warmstart.npz"
    final = WEIGHTS_DIR / f"quantum_{track_name}.npz"

    def pair_label(kind: str, path: Path) -> str:
        ep = _weights_label(path, "")
        return f"{kind} ({ep})" if ep else kind

    return [(pair_label("warm-start", warm), warm), (pair_label("best", final), final)]


def ghost_path(track_name: str, ghosts_dir: Path | None = None) -> Path:
    return (ghosts_dir if ghosts_dir is not None else GHOSTS_DIR) / f"{track_name}.json"


def load_ghost(track_name: str, ghosts_dir: Path | None = None) -> dict | None:
    """Best-lap ghost record for ``track_name``: {lap_time, kind, points} or None.

    Returns None when no ghost is stored or the file fails basic validation.
    """
    path = ghost_path(track_name, ghosts_dir)
    try:
        with path.open("r", encoding="utf-8") as f:
            ghost = json.load(f)
        lap_time = float(ghost["lap_time"])
        points = ghost["points"]
        if lap_time <= 0.0 or not isinstance(points, list) or len(points) < 2:
            return None
        return {
            "lap_time": lap_time,
            "kind": str(ghost.get("kind", "quantum")),
            "driver": str(ghost["driver"]) if ghost.get("driver") else None,
            "points": [[float(p[0]), float(p[1]), float(p[2])] for p in points],
        }
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def save_ghost(track_name: str, ghost: dict, ghosts_dir: Path | None = None) -> Path:
    """Persist a best-lap ghost record as ``<ghosts_dir>/<track>.json``."""
    path = ghost_path(track_name, ghosts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"track": track_name, **ghost}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _track_theme(name: str) -> dict:
    """Theme dict from the bundled track JSON (Track itself does not keep it)."""
    path = TRACKS_DIR / f"{name}.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return dict(json.load(f).get("theme", {}))
    return {}


def track_payload(track: Track) -> dict:
    """TrackPayload dict per the WS protocol (boundaries recomputed as
    centerline +/- normal * half_width, matching Track's internal ones)."""
    x, y, theta = track.start_pose()
    left = track.centerline + track.normals * track.half_width
    right = track.centerline - track.normals * track.half_width
    return {
        "name": track.name,
        "half_width": float(track.half_width),
        "total_length": float(track.total_length),
        "checkpoints": [float(c) for c in track.checkpoints],
        "theme": _track_theme(track.name),
        "start": {"x": x, "y": y, "theta": theta},
        "centerline": track.centerline.tolist(),
        "left": left.tolist(),
        "right": right.tolist(),
    }


def resolve_training_cfg(config: dict, track: str, warm: bool = False) -> dict:
    """[training] merged with [training_presets.<track>] and, when ``warm``,
    [training_warm] (+ [training_warm_gp] on top for gp)."""
    cfg = dict(config["training"])
    cfg.update(config.get("training_presets", {}).get(track, {}))
    if warm:
        cfg.update(config.get("training_warm", {}))
        if track == "gp":
            cfg.update(config.get("training_warm_gp", {}))
    return cfg
