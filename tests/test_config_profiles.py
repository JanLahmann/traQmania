"""q6/q8/q10 profile overlays: observation geometry, env feature counts,
weight-filename resolution, and graceful session degradation when the q{n}
weights are not bundled (they are trained later)."""

import numpy as np
import pytest

from traqmania.config import load_config
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track
from traqmania.server import protocol as P
from traqmania.server import session as session_mod
from traqmania.server.runtime import WEIGHTS_DIR
from traqmania.server.session import DemoSession, quantum_weights_path

# (profile, n_qubits, ray_angles_deg): n_qubits - 1 rays evenly spaced over
# [-60, +60] degrees plus normalized speed = n_qubits features, one per qubit.
PROFILES = [
    ("q6", 6, [-60.0, -30.0, 0.0, 30.0, 60.0]),
    ("q8", 8, [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]),
    ("q10", 10, [-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0]),
]


def make_env(config, n_envs=2, seed=0):
    track = Track.load("oval", config["track"]["resample_spacing"])
    return RacingEnv(track, config, n_envs=n_envs, seed=seed)


@pytest.mark.parametrize(("profile", "n_qubits", "rays"), PROFILES)
def test_qn_profile_overlay_and_env_features(profile, n_qubits, rays):
    config = load_config(profile)
    assert config["circuit"]["n_qubits"] == n_qubits
    assert config["observation"]["ray_angles_deg"] == rays
    assert len(rays) == n_qubits - 1
    assert np.allclose(np.diff(rays), 120.0 / (n_qubits - 2))  # evenly spaced
    # everything else inherits from default.toml
    assert config["circuit"]["n_layers"] == 4
    assert config["physics"]["v_max"] == 25.0

    env = make_env(config)
    assert env.n_features == n_qubits
    assert env.reset().shape == (2, n_qubits)


def test_default_config_unchanged():
    config = load_config()
    assert config["circuit"]["n_qubits"] == 4
    assert config["observation"]["ray_angles_deg"] == [-60.0, 0.0, 60.0]
    env = make_env(config)
    assert env.n_features == 4
    assert env.reset().shape == (2, 4)


def test_quantum_weights_path_rule():
    assert quantum_weights_path("oval", 4) == WEIGHTS_DIR / "quantum_oval.npz"
    assert quantum_weights_path("oval", 6) == WEIGHTS_DIR / "quantum_oval_q6.npz"
    assert (quantum_weights_path("oval", 4, "_warmstart")
            == WEIGHTS_DIR / "quantum_oval_warmstart.npz")
    assert (quantum_weights_path("oval", 6, "_warmstart")
            == WEIGHTS_DIR / "quantum_oval_warmstart_q6.npz")
    assert quantum_weights_path("gp", 8, "_stage2") == WEIGHTS_DIR / "quantum_gp_stage2_q8.npz"


def test_session_q6_missing_weights_degrades_gracefully(tmp_path, monkeypatch):
    # Resolve weights against an empty dir so the test never depends on
    # whether quantum_oval_q6.npz has been trained and bundled yet.
    monkeypatch.setattr(session_mod, "WEIGHTS_DIR", tmp_path / "weights")
    session = DemoSession(load_config("q6"), ghosts_dir=tmp_path)

    # construction survives the missing weights: empty attract mode + error msg
    assert session.mode == "attract"
    assert session.cars == []
    errors = [m for m in session.drain_outbox() if m["type"] == "error"]
    assert errors and "quantum_oval_q6.npz" in errors[0]["message"]
    P.parse_server(errors[0])  # existing wire shape; protocol not extended

    # mode switches that need weights are rejected; the session stays put
    for switch in (P.SetMode(mode="evolution"), P.SetMode(mode="race"),
                   P.SetMode(mode="hardware"), P.SetMode(mode="attract")):
        session.handle_message(switch)
        assert session.mode == "attract"
        assert session.cars == []
        errors = [m for m in session.drain_outbox() if m["type"] == "error"]
        assert len(errors) == 1, f"expected one rejection error for {switch}"
        assert f"'{switch.mode}'" in errors[0]["message"]
        P.parse_server(errors[0])

    # race start is rejected the same way
    session.handle_message(P.Race(action="start", opponent="quantum"))
    assert session.mode == "attract"
    errors = [m for m in session.drain_outbox() if m["type"] == "error"]
    assert errors and "quantum_oval_q6.npz" in errors[0]["message"]

    # the degraded session still ticks and broadcasts (car-less) states
    for _ in range(6):
        session.tick()
    states = [m for m in session.drain_outbox() if m["type"] == "state"]
    assert states and states[-1]["cars"] == []
    P.parse_server(states[-1])


def test_session_q6_bundled_weights_attract_drives(tmp_path):
    """Attract mode with the bundled quantum_oval_q6.npz: the 6-qubit car drives
    and the quantum introspection messages keep the pinned 4-action shape."""
    q6_weights = WEIGHTS_DIR / "quantum_oval_q6.npz"
    assert q6_weights.is_file(), "bundled q6 oval weights missing"
    config = load_config("q6")
    session = DemoSession(config, ghosts_dir=tmp_path)
    assert session.mode == "attract"
    assert [c.kind for c in session.cars] == ["quantum"]
    car = session.cars[0]
    assert car.qfunc.n_features == 6
    assert car.qfunc.n_actions == 4
    assert car.qfunc.n_params == 80

    for _ in range(120):  # 2 s of sim time = 20 agent decisions
        session.tick()
    msgs = session.drain_outbox()
    assert not [m for m in msgs if m["type"] == "error"]
    states = [m for m in msgs if m["type"] == "state"]
    assert states
    P.parse_server(states[-1])
    xs = [c["x"] for m in states for c in m["cars"]]
    ys = [c["y"] for m in states for c in m["cars"]]
    assert (max(xs) - min(xs)) + (max(ys) - min(ys)) > 1.0  # the car moves

    quantum = [m for m in msgs if m["type"] == "quantum"]
    assert quantum
    for q in quantum:
        P.parse_server(q)
        assert len(q["expectations"]) == 4  # readout pinned to Z_0..Z_3
        assert len(q["q_values"]) == 4
        assert 0 <= q["action"] < 4
        assert all(-1.0 <= e <= 1.0 for e in q["expectations"])
    # a 6-feature observation feeds the circuit: 5 rays broadcast per car
    rays = next(c["rays"] for m in reversed(states) for c in m["cars"] if "rays" in c)
    assert len(rays) == 5
