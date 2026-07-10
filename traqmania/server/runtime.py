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


def available_tracks() -> list[str]:
    """Bundled track names, sorted (e.g. ['chicane', 'gp', 'oval'])."""
    return sorted(p.stem for p in TRACKS_DIR.glob("*.json"))


def load_track(config: dict, name: str) -> Track:
    return Track.load(name, config["track"]["resample_spacing"])


def load_agent(kind: str, track_name: str, warm: bool = False, config: dict | None = None):
    """Build a Q-function of ``kind`` ('quantum' | 'mlp') with bundled weights loaded.

    ``warm=True`` (quantum only) loads ``quantum_<track>_warmstart.npz`` instead of
    the fully-trained weights.
    """
    if config is None:
        config = load_config()
    if kind == "mlp":
        qfunc: Any = MLPQFunction(n_features=4, n_actions=N_ACTIONS)
    elif kind == "quantum":
        qfunc = QuantumQFunction(config["circuit"])
    else:
        raise ValueError(f"unknown agent kind '{kind}' (expected 'quantum' or 'mlp')")

    suffix = "_warmstart" if (warm and kind == "quantum") else ""
    path = WEIGHTS_DIR / f"{kind}_{track_name}{suffix}.npz"
    qfunc.set_params(np.load(path)["params"])
    return qfunc


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
