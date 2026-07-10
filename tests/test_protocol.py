"""Round-trip and validation tests for the WS protocol (server/protocol.py)."""

import pytest

from traqmania.server import protocol as P

CLIENT_MSGS = [
    P.Hello(),
    P.Input(keys=0),
    P.Input(keys=P.KEYS_ALL),
    P.Input(keys=0, steer=-0.5, throttle=1.0, brake=0.0),
    P.Input(keys=0, steer=0.25),
    P.SetMode(mode="attract"),
    P.SetMode(mode="train"),
    P.SetMode(mode="evolution"),
    P.SetMode(mode="hardware"),
    P.SetTrack(track="gp"),
    P.Train(action="start", agent="both", track="gp", warm=True, episodes=100),
    P.Train(action="start", agent="quantum"),
    P.Train(action="stop", agent="mlp"),
    P.Race(action="start", opponent="quantum", track="oval"),
    P.Race(action="reset", opponent="mlp"),
    P.HardwareMsg(action="lap", backend="fake", shots=256, max_decisions=10),
    P.HardwareMsg(action="lap", backend="real"),
    P.HardwareMsg(action="sprint", backend="fake", iterations=5, shots=128),
    P.HardwareMsg(action="abort", backend="fake"),
]

SERVER_MSGS = [
    P.Welcome(mode="attract", track={"name": "oval"}, tracks=["chicane", "gp", "oval"],
              circuit_spec={"n_qubits": 4}, ui={"kiosk": False}),
    P.TrackMsg(track={"name": "gp", "half_width": 7.0}),
    P.State(t=1.25, mode="race", cars=[
        P.CarState(id="human", kind="human", x=1.0, y=2.0, theta=0.1, v=3.0,
                   lap=0, progress=12.5, last_lap_time=None, off_track=False),
        P.CarState(id="quantum", kind="quantum", x=0.0, y=-2.0, theta=3.1, v=9.0,
                   lap=2, progress=410.0, last_lap_time=21.3, off_track=True,
                   rays=[0.1, 0.5, 1.0]),
    ]),
    P.Quantum(car_id="quantum", expectations=[0.1, -0.2, 0.3, -0.4],
              q_values=[1.0, 2.0, 3.0, 4.0], action=3),
    P.State(t=2.0, mode="evolution", cars=[
        P.CarState(id="stage1", kind="quantum", x=1.0, y=2.0, theta=0.1, v=3.0,
                   lap=0, progress=12.5, last_lap_time=None, off_track=False,
                   label="ep 100"),
        P.CarState(id="ghost", kind="quantum", x=0.0, y=-2.0, theta=3.1, v=9.0,
                   lap=0, progress=0.0, last_lap_time=19.8, off_track=False,
                   label="best 19.8s", ghost=True),
    ]),
    P.Telemetry(agent="mlp", episode=17, mean_return=42.0, epsilon=0.3,
                loss=0.01, returns_tail=[1.0, 2.0]),
    P.Telemetry(agent="quantum", episode=0, mean_return=0.0, epsilon=1.0,
                loss=None, returns_tail=[]),
    P.Telemetry(agent="quantum", episode=99, mean_return=10.0, epsilon=0.1,
                loss=0.5, returns_tail=[3.0], best_lap_s=18.7,
                lap_times=[[42, 21.4], [77, 18.7]]),
    P.Telemetry(agent="mlp", episode=3, mean_return=1.0, epsilon=0.9,
                loss=None, returns_tail=[1.0], best_lap_s=None, lap_times=[]),
    P.Event(kind="lap", car_id="human", lap_time=19.9),
    P.Event(kind="clean_lap", car_id="quantum", lap_time=18.2),
    P.Event(kind="crash", car_id="mlp"),
    P.Event(kind="training_done", agent="quantum"),
    P.Event(kind="new_best_lap", car_id="human", lap_time=17.5),
    P.Event(kind="new_best_lap", agent="quantum", lap_time=18.1),
    P.Error(message="boom"),
    P.HardwareStatus(phase="idle"),
    P.HardwareStatus(phase="connecting", message="connecting to IBM Quantum"),
    P.HardwareStatus(phase="transpiling", backend_name="fake_manila"),
    P.HardwareStatus(phase="running", backend_name="ibm_torino", decision=12,
                     seconds_per_decision=3.4),
    P.HardwareStatus(phase="running", iteration=7, loss=0.42),
    P.HardwareStatus(phase="done", lap_time=17.2, seconds_per_decision=2.1),
    P.HardwareStatus(phase="done", eval_return_before=51.0, eval_return_after=63.5),
    P.HardwareStatus(phase="replay", message="replaying hardware lap"),
    P.HardwareStatus(phase="error", message="no IBM Quantum account"),
    P.State(t=3.0, mode="hardware", cars=[
        P.CarState(id="hardware", kind="quantum", x=1.0, y=2.0, theta=0.1, v=3.0,
                   lap=0, progress=0.0, last_lap_time=17.2, off_track=False,
                   label="hardware lap", ghost=True),
    ]),
]


@pytest.mark.parametrize("msg", CLIENT_MSGS, ids=lambda m: repr(m))
def test_client_round_trip(msg):
    wire = P.serialize(msg)
    assert wire["type"] == msg.TYPE
    assert P.parse_client(wire) == msg


@pytest.mark.parametrize("msg", SERVER_MSGS, ids=lambda m: repr(m))
def test_server_round_trip(msg):
    wire = P.serialize(msg)
    assert wire["type"] == msg.TYPE
    assert P.parse_server(wire) == msg


def test_optional_none_fields_omitted_from_wire():
    wire = P.serialize(P.Event(kind="crash", car_id="human"))
    assert "lap_time" not in wire and "agent" not in wire
    wire = P.serialize(P.Train(action="start", agent="mlp"))
    assert "track" not in wire and "episodes" not in wire
    # Nullable-but-required fields stay as explicit nulls.
    wire = P.serialize(P.Telemetry(agent="mlp", episode=0, mean_return=0.0,
                                   epsilon=1.0, loss=None, returns_tail=[]))
    assert wire["loss"] is None
    # rays/label/ghost omitted per-car when absent.
    car = P.CarState(id="a", kind="mlp", x=0, y=0, theta=0, v=0, lap=0,
                     progress=0, last_lap_time=None, off_track=False)
    wire = P.serialize(P.State(t=0.0, mode="attract", cars=[car]))
    assert "rays" not in wire["cars"][0]
    assert "label" not in wire["cars"][0] and "ghost" not in wire["cars"][0]
    assert wire["cars"][0]["last_lap_time"] is None
    # telemetry best_lap_s / lap_times omitted when absent (pre-M10 shape).
    wire = P.serialize(P.Telemetry(agent="mlp", episode=0, mean_return=0.0,
                                   epsilon=1.0, loss=None, returns_tail=[]))
    assert "best_lap_s" not in wire and "lap_times" not in wire


def test_input_analog_fields_clamped_and_omitted():
    # out-of-range analog values are clamped, not rejected
    msg = P.parse_client({"type": "input", "keys": 0,
                          "steer": -3.5, "throttle": 2.0, "brake": -0.25})
    assert (msg.steer, msg.throttle, msg.brake) == (-1.0, 1.0, 0.0)
    # keys-only input carries no analog fields on the wire
    wire = P.serialize(P.Input(keys=P.KEY_THROTTLE))
    assert set(wire) == {"type", "keys"}
    # partial analog: absent axes stay None (server defaults them to 0.0)
    msg = P.parse_client({"type": "input", "keys": 0, "steer": 0.5})
    assert msg.steer == 0.5 and msg.throttle is None and msg.brake is None


def test_hardware_msg_and_status_optional_fields_omitted():
    wire = P.serialize(P.HardwareMsg(action="abort", backend="fake"))
    assert set(wire) == {"type", "action", "backend"}
    wire = P.serialize(P.HardwareStatus(phase="idle"))
    assert set(wire) == {"type", "phase"}
    wire = P.serialize(P.HardwareStatus(phase="running", decision=3,
                                        seconds_per_decision=2.5))
    assert set(wire) == {"type", "phase", "decision", "seconds_per_decision"}
    with pytest.raises(P.ProtocolError):
        P.parse_server({"type": "hardware_status", "phase": "warming_up"})
    with pytest.raises(P.ProtocolError):
        P.parse_server({"type": "hardware_status"})


def test_telemetry_lap_times_validation():
    base = {"type": "telemetry", "agent": "mlp", "episode": 0, "mean_return": 0.0,
            "epsilon": 1.0, "loss": None, "returns_tail": []}
    # explicit null best_lap_s is accepted
    msg = P.parse_server({**base, "best_lap_s": None, "lap_times": [[3, 21.5]]})
    assert msg.best_lap_s is None and msg.lap_times == [[3, 21.5]]
    for bad in ("nope", [[1.5, 2.0]], [[1, 2.0, 3.0]], [[1]], [21.5], [[-1, 2.0]]):
        with pytest.raises(P.ProtocolError):
            P.parse_server({**base, "lap_times": bad})


GARBAGE = [
    "not a dict",
    None,
    42,
    {},                                                # missing type
    {"type": "warp_drive"},                            # unknown type
    {"type": "hello", "extra": 1},                     # unknown field
    {"type": "input"},                                 # missing keys
    {"type": "input", "keys": "1"},                    # wrong type
    {"type": "input", "keys": True},                   # bool is not an int
    {"type": "input", "keys": -1},                     # out of range
    {"type": "input", "keys": 16},                     # out of range
    {"type": "set_mode", "mode": "zen"},               # bad enum
    {"type": "set_mode"},                              # missing mode
    {"type": "set_track", "track": ""},                # empty string
    {"type": "train", "action": "start"},              # missing agent
    {"type": "train", "action": "pause", "agent": "mlp"},
    {"type": "train", "action": "start", "agent": "gpt"},
    {"type": "train", "action": "start", "agent": "mlp", "episodes": 0},
    {"type": "train", "action": "start", "agent": "mlp", "warm": "yes"},
    {"type": "train", "action": "start", "agent": "mlp", "unknown": 1},
    {"type": "race", "action": "start", "opponent": "both"},  # 'both' invalid here
    {"type": "race", "action": "go", "opponent": "mlp"},
    {"type": "race", "action": "start"},               # missing opponent
    {"type": "input", "keys": 0, "steer": "left"},     # analog must be numeric
    {"type": "input", "keys": 0, "throttle": True},    # bool is not a number
    {"type": "input", "steer": 0.5},                   # keys stays required
    {"type": "hardware", "action": "lap"},             # missing backend
    {"type": "hardware", "action": "warp", "backend": "fake"},
    {"type": "hardware", "action": "lap", "backend": "aer"},
    {"type": "hardware", "action": "sprint", "backend": "fake", "iterations": 0},
    {"type": "hardware", "action": "lap", "backend": "fake", "shots": "many"},
    {"type": "hardware", "action": "lap", "backend": "fake", "max_decisions": 0},
    {"type": "hardware", "action": "lap", "backend": "fake", "unknown": 1},
]


@pytest.mark.parametrize("data", GARBAGE, ids=lambda d: repr(d)[:60])
def test_parse_client_rejects_garbage(data):
    with pytest.raises(P.ProtocolError):
        P.parse_client(data)


def test_parse_server_rejects_unknown_type():
    with pytest.raises(P.ProtocolError):
        P.parse_server({"type": "input", "keys": 1})  # client type, not server
    with pytest.raises(P.ProtocolError):
        P.parse_server([])
