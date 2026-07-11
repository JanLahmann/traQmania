"""SetDriver: choosing which training's quantum weights drive the agent."""

import pytest

from traqmania.config import load_config
from traqmania.server import protocol as P
from traqmania.server.session import DemoSession


@pytest.fixture()
def session(tmp_path):
    s = DemoSession(load_config(), ghosts_dir=tmp_path)
    s.drain_outbox()
    return s


def by_type(msgs, tag):
    return [m for m in msgs if m["type"] == tag]


def test_set_driver_protocol_round_trip():
    msg = P.SetDriver(driver="gp")
    wire = P.serialize(msg)
    assert wire == {"type": "set_driver", "driver": "gp"}
    assert P.parse_client(wire) == msg
    with pytest.raises(P.ProtocolError):
        P.parse_client({"type": "set_driver"})
    with pytest.raises(P.ProtocolError):
        P.parse_client({"type": "set_driver", "driver": ""})
    with pytest.raises(P.ProtocolError):
        P.parse_client({"type": "set_driver", "driver": "gp", "extra": 1})


def test_welcome_lists_bundled_drivers(session):
    welcome = session.welcome_payload()
    assert welcome["driver"] == "auto"
    # at 4 qubits the oval/chicane/gp specialists are all bundled
    assert welcome["drivers"][0] == "auto"
    for name in ("oval", "chicane", "gp"):
        assert name in welcome["drivers"]


def test_driver_override_swaps_weights_and_labels(session):
    # default oval track, gp-trained driver: the cross-track transfer demo
    session.handle_message(P.SetDriver(driver="gp"))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    assert by_type(msgs, "welcome")[0]["driver"] == "gp"
    assert session._quantum_weights_path().name == "quantum_gp.npz"
    car = session.cars[0]
    assert car.kind == "quantum"
    assert car.label == "driver: gp-trained"

    # back to auto: the track specialist drives, label drops
    session.handle_message(P.SetDriver(driver="auto"))
    session.drain_outbox()
    assert session._quantum_weights_path().name == "quantum_oval.npz"
    assert session.cars[0].label is None


def test_hero_driver_laps_without_weights(session):
    # the model-based racing-line controller: no weights, honest label,
    # and it actually gets around the track
    assert "hero" in session.available_drivers()
    session.handle_message(P.SetDriver(driver="hero"))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    car = session.cars[0]
    assert car.kind == "hero"
    assert car.qfunc is None and car.controller is not None
    assert "not learned" in car.label
    for _ in range(60 * 20):  # 20 s of sim time: enough for a ~14 s oval lap
        session.tick()
    assert car.lap >= 1
    assert not [m for m in session.drain_outbox()
                if m.get("type") == "event" and m.get("kind") == "crash"]

    # race mode is untouched by the hero pick: the opponent stays quantum,
    # driven by the track specialist
    session.handle_message(P.Race(action="start", opponent="quantum"))
    assert not by_type(session.drain_outbox(), "error")
    assert {c.kind for c in session.cars} == {"human", "quantum"}
    assert session._quantum_weights_path().name == "quantum_oval.npz"


def test_hero_driver_on_generated_track(session):
    pytest.importorskip("traqmania.env.trackgen")
    session.handle_message(P.SetDriver(driver="hero"))
    session.handle_message(P.SetTrack(track="random", seed=99))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    assert session.cars[0].kind == "hero"
    for _ in range(60 * 5):
        session.tick()
    assert session.cars[0].state[3] > 1.0  # it drives


def test_unknown_driver_is_rejected(session):
    session.handle_message(P.SetDriver(driver="monaco"))
    msgs = session.drain_outbox()
    assert by_type(msgs, "error")
    assert session.driver == "auto"


def test_qubit_switch_resets_unavailable_driver(session):
    session.handle_message(P.SetDriver(driver="gp"))
    session.drain_outbox()
    # no quantum_gp_q6.npz is bundled -> the pick cannot survive the switch
    session.handle_message(P.Qubits(n=6))
    msgs = session.drain_outbox()
    welcome = by_type(msgs, "welcome")[0]
    assert welcome["driver"] == "auto"
    assert "gp" not in welcome["drivers"]
    assert "oval" in welcome["drivers"]  # quantum_oval_q6.npz ships
