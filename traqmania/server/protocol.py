"""Typed WebSocket protocol for the traQmania demo server.

Every message is a JSON object with a ``"type"`` field.  Client -> server
messages are strictly validated by :func:`parse_client` (unknown types, unknown
fields, wrong value types/enums all raise :class:`ProtocolError`); server ->
client messages round-trip through :func:`serialize` / :func:`parse_server`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

MODES = ("attract", "train", "race", "evolution")
TRAIN_AGENTS = ("quantum", "mlp", "both")
OPPONENTS = ("quantum", "mlp")
TRAIN_ACTIONS = ("start", "stop")
RACE_ACTIONS = ("start", "reset")
CAR_KINDS = ("human", "quantum", "mlp")
EVENT_KINDS = ("lap", "crash", "clean_lap", "training_done", "new_best_lap")

# input.keys bitmask
KEY_THROTTLE, KEY_BRAKE, KEY_LEFT, KEY_RIGHT = 1, 2, 4, 8
KEYS_ALL = KEY_THROTTLE | KEY_BRAKE | KEY_LEFT | KEY_RIGHT


class ProtocolError(ValueError):
    """An incoming message failed validation."""


# ------------------------------------------------------------- client -> server


@dataclass(frozen=True)
class Hello:
    TYPE: ClassVar[str] = "hello"


@dataclass(frozen=True)
class Input:
    keys: int
    TYPE: ClassVar[str] = "input"


@dataclass(frozen=True)
class SetMode:
    mode: str
    TYPE: ClassVar[str] = "set_mode"


@dataclass(frozen=True)
class SetTrack:
    track: str
    TYPE: ClassVar[str] = "set_track"


@dataclass(frozen=True)
class Train:
    action: str
    agent: str
    track: str | None = None
    warm: bool = False
    episodes: int | None = None
    TYPE: ClassVar[str] = "train"


@dataclass(frozen=True)
class Race:
    action: str
    opponent: str
    track: str | None = None
    TYPE: ClassVar[str] = "race"


# ------------------------------------------------------------- server -> client


@dataclass(frozen=True)
class CarState:
    id: str
    kind: str
    x: float
    y: float
    theta: float
    v: float
    lap: int
    progress: float
    last_lap_time: float | None
    off_track: bool
    rays: list | None = None
    label: str | None = None  # e.g. "ep 150" (evolution) or "best 19.8s" (ghost)
    ghost: bool | None = None  # True for the best-lap replay car


@dataclass(frozen=True)
class Welcome:
    mode: str
    track: dict
    tracks: list
    circuit_spec: dict
    ui: dict
    TYPE: ClassVar[str] = "welcome"


@dataclass(frozen=True)
class TrackMsg:
    track: dict
    TYPE: ClassVar[str] = "track"


@dataclass(frozen=True)
class State:
    t: float
    mode: str
    cars: list  # of CarState
    TYPE: ClassVar[str] = "state"


@dataclass(frozen=True)
class Quantum:
    car_id: str
    expectations: list
    q_values: list
    action: int
    TYPE: ClassVar[str] = "quantum"


@dataclass(frozen=True)
class Telemetry:
    agent: str
    episode: int
    mean_return: float
    epsilon: float
    loss: float | None
    returns_tail: list
    best_lap_s: float | None = None
    lap_times: list | None = None  # of [episode:int, lap_s:float], last <= 50
    TYPE: ClassVar[str] = "telemetry"


@dataclass(frozen=True)
class Event:
    kind: str
    car_id: str | None = None
    lap_time: float | None = None
    agent: str | None = None
    TYPE: ClassVar[str] = "event"


@dataclass(frozen=True)
class Error:
    message: str
    TYPE: ClassVar[str] = "error"


# ------------------------------------------------------------------- serialize

# Optional fields dropped from the wire when None (nullable-but-required fields
# like telemetry.loss and car.last_lap_time stay as explicit nulls).
_OMIT_IF_NONE: dict[str, set[str]] = {
    Train.TYPE: {"track", "episodes"},
    Race.TYPE: {"track"},
    Event.TYPE: {"car_id", "lap_time", "agent"},
    Telemetry.TYPE: {"best_lap_s", "lap_times"},
    CarState.__name__: {"rays", "label", "ghost"},
}


def serialize(msg: Any) -> dict:
    """Dataclass message -> JSON-ready dict with a ``"type"`` tag."""
    type_tag = getattr(msg, "TYPE", None)
    if type_tag is None:
        raise TypeError(f"not a protocol message: {msg!r}")
    payload = asdict(msg)
    omit = _OMIT_IF_NONE.get(type_tag, set())
    payload = {k: v for k, v in payload.items() if not (v is None and k in omit)}
    if type_tag == State.TYPE:
        car_omit = _OMIT_IF_NONE[CarState.__name__]
        payload["cars"] = [
            {k: v for k, v in car.items() if not (v is None and k in car_omit)}
            for car in payload["cars"]
        ]
    return {"type": type_tag, **payload}


# ------------------------------------------------------------------ validators


def _req(data: dict, key: str) -> Any:
    if key not in data:
        raise ProtocolError(f"missing field '{key}'")
    return data[key]


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"'{name}' must be a non-empty string")
    return value


def _enum(value: Any, name: str, options: tuple) -> str:
    value = _str(value, name)
    if value not in options:
        raise ProtocolError(f"'{name}' must be one of {options}, got {value!r}")
    return value


def _int(value: Any, name: str, lo: int | None = None, hi: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"'{name}' must be an integer")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ProtocolError(f"'{name}' out of range [{lo}, {hi}]: {value}")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"'{name}' must be a boolean")
    return value


def _float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError(f"'{name}' must be a number")
    return float(value)


def _num_list(value: Any, name: str, length: int | None = None) -> list:
    if not isinstance(value, list):
        raise ProtocolError(f"'{name}' must be a list")
    if length is not None and len(value) != length:
        raise ProtocolError(f"'{name}' must have length {length}, got {len(value)}")
    return [_float(v, name) for v in value]


def _lap_times(value: Any, name: str) -> list:
    """List of [episode:int, lap_s:float] pairs."""
    if not isinstance(value, list):
        raise ProtocolError(f"'{name}' must be a list")
    out = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ProtocolError(f"'{name}' entries must be [episode, lap_s] pairs")
        out.append([_int(item[0], f"{name}[].episode", 0), _float(item[1], f"{name}[].lap_s")])
    return out


def _check_extra(data: dict, allowed: set[str]) -> None:
    extra = set(data) - allowed - {"type"}
    if extra:
        raise ProtocolError(f"unexpected fields: {sorted(extra)}")


# --------------------------------------------------------------- parse: client


def _parse_hello(d: dict) -> Hello:
    _check_extra(d, set())
    return Hello()


def _parse_input(d: dict) -> Input:
    _check_extra(d, {"keys"})
    return Input(keys=_int(_req(d, "keys"), "keys", 0, KEYS_ALL))


def _parse_set_mode(d: dict) -> SetMode:
    _check_extra(d, {"mode"})
    return SetMode(mode=_enum(_req(d, "mode"), "mode", MODES))


def _parse_set_track(d: dict) -> SetTrack:
    _check_extra(d, {"track"})
    return SetTrack(track=_str(_req(d, "track"), "track"))


def _parse_train(d: dict) -> Train:
    _check_extra(d, {"action", "agent", "track", "warm", "episodes"})
    return Train(
        action=_enum(_req(d, "action"), "action", TRAIN_ACTIONS),
        agent=_enum(_req(d, "agent"), "agent", TRAIN_AGENTS),
        track=_str(d["track"], "track") if d.get("track") is not None else None,
        warm=_bool(d["warm"], "warm") if d.get("warm") is not None else False,
        episodes=_int(d["episodes"], "episodes", 1) if d.get("episodes") is not None else None,
    )


def _parse_race(d: dict) -> Race:
    _check_extra(d, {"action", "opponent", "track"})
    return Race(
        action=_enum(_req(d, "action"), "action", RACE_ACTIONS),
        opponent=_enum(_req(d, "opponent"), "opponent", OPPONENTS),
        track=_str(d["track"], "track") if d.get("track") is not None else None,
    )


_CLIENT_PARSERS = {
    Hello.TYPE: _parse_hello,
    Input.TYPE: _parse_input,
    SetMode.TYPE: _parse_set_mode,
    SetTrack.TYPE: _parse_set_track,
    Train.TYPE: _parse_train,
    Race.TYPE: _parse_race,
}


def parse_client(data: Any) -> Hello | Input | SetMode | SetTrack | Train | Race:
    """Strictly parse a client -> server dict; raises ProtocolError on anything off."""
    if not isinstance(data, dict):
        raise ProtocolError("message must be a JSON object")
    type_tag = data.get("type")
    parser = _CLIENT_PARSERS.get(type_tag)
    if parser is None:
        raise ProtocolError(f"unknown message type: {type_tag!r}")
    return parser(data)


# --------------------------------------------------------------- parse: server


def _opt_float(d: dict, key: str) -> float | None:
    return _float(d[key], key) if d.get(key) is not None else None


def _parse_car(d: Any) -> CarState:
    if not isinstance(d, dict):
        raise ProtocolError("car must be an object")
    return CarState(
        id=_str(_req(d, "id"), "id"),
        kind=_enum(_req(d, "kind"), "kind", CAR_KINDS),
        x=_float(_req(d, "x"), "x"),
        y=_float(_req(d, "y"), "y"),
        theta=_float(_req(d, "theta"), "theta"),
        v=_float(_req(d, "v"), "v"),
        lap=_int(_req(d, "lap"), "lap"),
        progress=_float(_req(d, "progress"), "progress"),
        last_lap_time=_opt_float(d, "last_lap_time"),
        off_track=_bool(_req(d, "off_track"), "off_track"),
        rays=_num_list(d["rays"], "rays") if d.get("rays") is not None else None,
        label=_str(d["label"], "label") if d.get("label") is not None else None,
        ghost=_bool(d["ghost"], "ghost") if d.get("ghost") is not None else None,
    )


def _parse_welcome(d: dict) -> Welcome:
    return Welcome(
        mode=_enum(_req(d, "mode"), "mode", MODES),
        track=dict(_req(d, "track")),
        tracks=[_str(t, "tracks[]") for t in _req(d, "tracks")],
        circuit_spec=dict(_req(d, "circuit_spec")),
        ui=dict(_req(d, "ui")),
    )


def _parse_track_msg(d: dict) -> TrackMsg:
    return TrackMsg(track=dict(_req(d, "track")))


def _parse_state(d: dict) -> State:
    return State(
        t=_float(_req(d, "t"), "t"),
        mode=_enum(_req(d, "mode"), "mode", MODES),
        cars=[_parse_car(c) for c in _req(d, "cars")],
    )


def _parse_quantum(d: dict) -> Quantum:
    return Quantum(
        car_id=_str(_req(d, "car_id"), "car_id"),
        expectations=_num_list(_req(d, "expectations"), "expectations"),
        q_values=_num_list(_req(d, "q_values"), "q_values"),
        action=_int(_req(d, "action"), "action", 0),
    )


def _parse_telemetry(d: dict) -> Telemetry:
    return Telemetry(
        agent=_enum(_req(d, "agent"), "agent", OPPONENTS),
        episode=_int(_req(d, "episode"), "episode", 0),
        mean_return=_float(_req(d, "mean_return"), "mean_return"),
        epsilon=_float(_req(d, "epsilon"), "epsilon"),
        loss=_opt_float(d, "loss"),
        returns_tail=_num_list(_req(d, "returns_tail"), "returns_tail"),
        best_lap_s=_opt_float(d, "best_lap_s"),
        lap_times=_lap_times(d["lap_times"], "lap_times")
        if d.get("lap_times") is not None else None,
    )


def _parse_event(d: dict) -> Event:
    return Event(
        kind=_enum(_req(d, "kind"), "kind", EVENT_KINDS),
        car_id=_str(d["car_id"], "car_id") if d.get("car_id") is not None else None,
        lap_time=_opt_float(d, "lap_time"),
        agent=_enum(d["agent"], "agent", OPPONENTS) if d.get("agent") is not None else None,
    )


def _parse_error(d: dict) -> Error:
    return Error(message=_str(_req(d, "message"), "message"))


_SERVER_PARSERS = {
    Welcome.TYPE: _parse_welcome,
    TrackMsg.TYPE: _parse_track_msg,
    State.TYPE: _parse_state,
    Quantum.TYPE: _parse_quantum,
    Telemetry.TYPE: _parse_telemetry,
    Event.TYPE: _parse_event,
    Error.TYPE: _parse_error,
}


def parse_server(data: Any) -> Welcome | TrackMsg | State | Quantum | Telemetry | Event | Error:
    """Parse a server -> client dict (used by tests and client tooling)."""
    if not isinstance(data, dict):
        raise ProtocolError("message must be a JSON object")
    type_tag = data.get("type")
    parser = _SERVER_PARSERS.get(type_tag)
    if parser is None:
        raise ProtocolError(f"unknown message type: {type_tag!r}")
    return parser(data)
