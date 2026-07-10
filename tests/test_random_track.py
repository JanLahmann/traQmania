"""Random-track support: set_track{seed} protocol parsing, generated-track
sessions (attract/race/train), the universal->gp weight fallback labeling,
ghost-persistence skipping, and graceful evolution/hardware rejections.

Session tests need the procedural generator and skip cleanly until
``traqmania.env.trackgen`` lands (the protocol tests always run).
"""

import re

import numpy as np
import pytest

from traqmania.config import load_config
from traqmania.server import protocol as P
from traqmania.server.session import DemoSession, random_track_weights


def make_config(**sections):
    """load_config() with per-section dict overrides (as in test_session)."""
    config = load_config()
    for name, overrides in sections.items():
        merged = dict(config[name])
        merged.update(overrides)
        config[name] = merged
    return config


def make_session(tmp_path, config=None):
    session = DemoSession(config if config is not None else load_config(), ghosts_dir=tmp_path)
    session.drain_outbox()
    return session


def by_type(msgs, type_tag):
    return [m for m in msgs if m["type"] == type_tag]


def set_random_track(session, seed):
    """Send set_track{random, seed}; return the broadcast track payload."""
    session.handle_message(P.SetTrack(track="random", seed=seed))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    tracks = by_type(msgs, "track")
    assert len(tracks) == 1
    return tracks[0]["track"]


# ------------------------------------------------------------------- protocol


def test_set_track_seed_round_trip():
    msg = P.SetTrack(track="random", seed=42)
    wire = P.serialize(msg)
    assert wire == {"type": "set_track", "track": "random", "seed": 42}
    assert P.parse_client(wire) == msg
    # plain set_track stays unchanged on the wire and parses to seed=None
    wire = P.serialize(P.SetTrack(track="oval"))
    assert wire == {"type": "set_track", "track": "oval"}
    assert P.parse_client(wire) == P.SetTrack(track="oval", seed=None)
    # an explicit null seed is tolerated (matches the other optional fields)
    assert P.parse_client({"type": "set_track", "track": "gp", "seed": None}).seed is None


@pytest.mark.parametrize("data", [
    {"type": "set_track", "track": "random", "seed": "42"},   # wrong type
    {"type": "set_track", "track": "random", "seed": True},   # bool is not an int
    {"type": "set_track", "track": "random", "seed": 4.2},    # float is not an int
    {"type": "set_track", "track": "random", "seed": -1},     # out of range
    {"type": "set_track", "seed": 42},                        # track stays required
    {"type": "set_track", "track": "random", "extra": 1},     # unknown field
], ids=lambda d: repr(d)[:60])
def test_set_track_seed_rejects_garbage(data):
    with pytest.raises(P.ProtocolError):
        P.parse_client(data)


# ------------------------------------------------------- random-track session


def test_random_track_attract_drives_with_honest_fallback(tmp_path):
    pytest.importorskip("traqmania.env.trackgen")
    session = make_session(tmp_path)
    payload = set_random_track(session, seed=7)
    assert payload["name"] == "random #7"
    assert len(payload["centerline"]) >= 8
    assert session.welcome_payload()["track"]["name"] == "random #7"

    # no quantum_universal.npz is bundled -> the chain picks the gp specialist
    path, label = random_track_weights(session.n_qubits)
    assert (path.name, label) == ("quantum_gp.npz", "gp-trained generalist")
    assert session._quantum_weights_path() == path
    assert session.mode == "attract"
    assert [c.kind for c in session.cars] == ["quantum"]
    assert np.array_equal(session.cars[0].qfunc.get_params(), np.load(path)["params"])
    assert session.cars[0].label == "driver: gp-trained generalist"

    for _ in range(120):  # 2 s of sim time = 20 agent decisions
        session.tick()
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    states = by_type(msgs, "state")
    assert states
    P.parse_server(states[-1])  # validates the payload shape on the wire
    cars = [c for m in states for c in m["cars"] if c["id"] == "quantum"]
    assert all(c["label"] == "driver: gp-trained generalist" for c in cars)
    xs, ys = [c["x"] for c in cars], [c["y"] for c in cars]
    assert (max(xs) - min(xs)) + (max(ys) - min(ys)) > 0.5  # the fallback drives


def test_random_track_weights_prefer_universal(tmp_path, monkeypatch):
    import traqmania.server.session as session_mod

    monkeypatch.setattr(session_mod, "WEIGHTS_DIR", tmp_path)
    # no universal weights bundled -> the gp specialist, labelled honestly
    path, label = random_track_weights(4)
    assert (path.name, label) == ("quantum_gp.npz", "gp-trained generalist")
    # quantum_universal.npz takes precedence once it exists
    (tmp_path / "quantum_universal.npz").touch()
    path, label = random_track_weights(4)
    assert (path.name, label) == ("quantum_universal.npz", "universal")
    # the n-qubit filename tag applies to the whole chain
    path, label = random_track_weights(6)
    assert (path.name, label) == ("quantum_gp_q6.npz", "gp-trained generalist")
    (tmp_path / "quantum_universal_q6.npz").touch()
    assert random_track_weights(6) == (tmp_path / "quantum_universal_q6.npz", "universal")
    # suffixes (warm starts) follow the same rule
    assert random_track_weights(4, "_warmstart")[0].name == "quantum_gp_warmstart.npz"
    (tmp_path / "quantum_universal_warmstart.npz").touch()
    assert random_track_weights(4, "_warmstart")[0].name == "quantum_universal_warmstart.npz"


def test_random_track_reproducible_by_seed(tmp_path):
    pytest.importorskip("traqmania.env.trackgen")
    session = make_session(tmp_path)
    first = set_random_track(session, seed=123)
    session.handle_message(P.SetTrack(track="oval"))
    session.drain_outbox()
    second = set_random_track(session, seed=123)
    assert first["name"] == second["name"] == "random #123"
    assert first["centerline"] == second["centerline"]
    assert first["half_width"] == second["half_width"]
    assert first["checkpoints"] == second["checkpoints"]


def test_random_track_without_seed_rolls_fresh(tmp_path):
    pytest.importorskip("traqmania.env.trackgen")
    session = make_session(tmp_path)
    session.handle_message(P.SetTrack(track="random"))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    name = by_type(msgs, "track")[0]["track"]["name"]
    assert re.fullmatch(r"random #\d+", name)
    assert session.track_name == name


def test_human_race_on_random_track(tmp_path):
    pytest.importorskip("traqmania.env.trackgen")
    session = make_session(tmp_path)
    set_random_track(session, seed=21)
    # the UI restarts races naming the current track ("random #21"): kept as-is
    session.handle_message(P.Race(action="start", opponent="quantum",
                                  track=session.track_name))
    assert session.mode == "race"
    assert {c.kind for c in session.cars} == {"human", "quantum"}
    session.handle_message(P.Input(keys=P.KEY_THROTTLE))
    for _ in range(12):
        session.tick()
    human = next(c for c in session.cars if c.kind == "human")
    assert human.state[3] > 0.0  # throttle accelerated the car
    assert not by_type(session.drain_outbox(), "error")


# --------------------------------------------------------------------- ghosts


def fake_lap_traj(car, n_points=40):
    return [(float(car.state[0]) + 0.5 * i, float(car.state[1]), float(car.state[2]))
            for i in range(n_points)]


def test_no_ghost_persisted_on_random_track(tmp_path):
    pytest.importorskip("traqmania.env.trackgen")
    session = make_session(tmp_path)
    set_random_track(session, seed=11)
    for _ in range(120):
        session.tick()
    car = session.cars[0]
    car.traj = fake_lap_traj(car)  # a clean lap ends: the record path is a no-op
    session._maybe_record_ghost(car, 12.5)
    assert session._ghost is None
    assert list(tmp_path.iterdir()) == []  # never writes ghosts_dir/random*.json

    # control: the same clean lap back on a bundled track persists as usual
    session.handle_message(P.SetTrack(track="oval"))
    session.drain_outbox()
    car = session.cars[0]
    car.traj = fake_lap_traj(car)
    session._maybe_record_ghost(car, 12.5)
    assert (tmp_path / "oval.json").is_file()
    assert list(tmp_path.iterdir()) == [tmp_path / "oval.json"]


# ------------------------------------------------------------ training + modes


def test_train_starts_on_random_track(tmp_path):
    pytest.importorskip("traqmania.env.trackgen")
    config = make_config(
        reward={"max_decisions": 50},
        training={"n_parallel_envs": 4, "replay_size": 2000, "batch_size": 16},
    )
    session = make_session(tmp_path, config)
    set_random_track(session, seed=5)
    session.handle_message(P.Train(action="start", agent="mlp", episodes=8))
    try:
        assert session.mode == "train"
        job = session.jobs["mlp"]
        assert job.env.track is session.track  # trains live on the generated track
        for _ in range(6):
            session.tick()
        assert not by_type(session.drain_outbox(), "error")
    finally:
        session.shutdown()
    assert job.error is None


def test_evolution_and_hardware_rejected_on_random_track(tmp_path):
    pytest.importorskip("traqmania.env.trackgen")
    session = make_session(tmp_path)
    set_random_track(session, seed=3)
    for mode in ("evolution", "hardware"):
        session.handle_message(P.SetMode(mode=mode))
        assert session.mode == "attract"  # rejected; session keeps ticking
        errors = by_type(session.drain_outbox(), "error")
        assert len(errors) == 1 and f"'{mode}'" in errors[0]["message"]
    # a direct hardware command is refused the same way, as a status message
    session.handle_message(P.HardwareMsg(action="lap", backend="fake"))
    statuses = by_type(session.drain_outbox(), "hardware_status")
    assert statuses and statuses[-1]["phase"] == "error"
    assert session.mode == "attract"
    # race against the mlp reuses the missing-weights rejection
    session.handle_message(P.Race(action="start", opponent="mlp"))
    assert session.mode == "attract"
    errors = by_type(session.drain_outbox(), "error")
    assert errors and "mlp" in errors[0]["message"]


def test_set_track_back_to_oval_restores_normal(tmp_path):
    pytest.importorskip("traqmania.env.trackgen")
    session = make_session(tmp_path)
    set_random_track(session, seed=9)
    session.handle_message(P.SetTrack(track="oval"))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    assert by_type(msgs, "track")[0]["track"]["name"] == "oval"
    assert session.track_is_random is False
    assert session.mode == "attract"
    assert session._quantum_weights_path().name == "quantum_oval.npz"
    car = session.cars[0]
    assert car.label is None  # no fallback labeling on bundled tracks
    assert np.array_equal(car.qfunc.get_params(),
                          np.load(session._quantum_weights_path())["params"])
