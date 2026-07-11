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
