"""Typed WebSocket protocol for the traQmania demo server.

Every message is a JSON object with a ``"type"`` field.  Client -> server
messages are strictly validated by :func:`parse_client` (unknown types, unknown
fields, wrong value types/enums all raise :class:`ProtocolError`); server ->
client messages round-trip through :func:`serialize` / :func:`parse_server`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

MODES = ("attract", "train", "race", "evolution", "hardware")
TRACK_LENGTHS = ("short", "medium", "long")  # generated-track size presets
TRAIN_AGENTS = ("quantum", "mlp", "both")
OPPONENTS = ("quantum", "mlp")
TRAIN_ACTIONS = ("start", "stop")
RACE_ACTIONS = ("start", "reset")
CAR_KINDS = ("human", "quantum", "mlp", "hero", "pro")  # +expert reference drivers
EVENT_KINDS = ("lap", "crash", "clean_lap", "training_done", "new_best_lap")
HARDWARE_ACTIONS = ("lap", "sprint", "abort")
HARDWARE_BACKENDS = ("fake", "real")
HARDWARE_PHASES = ("idle", "connecting", "transpiling", "running", "replay", "done", "error")

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
    """Human controls. ``keys`` is always required; when ANY of the optional
    analog axes is present the server drives from them instead of the bitmask
    (missing axes default to 0.0 — clients send ``keys: 0`` alongside)."""

    keys: int
    steer: float | None = None  # clamped to [-1, 1] at parse time
    throttle: float | None = None  # clamped to [0, 1]
    brake: float | None = None  # clamped to [0, 1]
    TYPE: ClassVar[str] = "input"


@dataclass(frozen=True)
class SetMode:
    mode: str
    TYPE: ClassVar[str] = "set_mode"


@dataclass(frozen=True)
class SetTrack:
    """``seed`` and ``length`` are only meaningful with ``track == "random"``:
    ``seed`` makes the generated track reproducible (omitted -> the server
    rolls a fresh one) and ``length`` picks a size preset (short / medium /
    long; omitted -> medium)."""

    track: str
    seed: int | None = None
    length: str | None = None
    TYPE: ClassVar[str] = "set_track"


@dataclass(frozen=True)
class SetName:
    """The racer's display name for the leaderboard (empty clears it —
    anonymous laps are not recorded)."""

    name: str
    TYPE: ClassVar[str] = "set_name"


@dataclass(frozen=True)
class DrawTrack:
    """A hand-drawn centerline (list of [x, y] world coordinates, in stroke
    order): the server rescales, smooths and validates it into a drivable
    closed track, or answers with an error explaining what to fix."""

    points: list
    TYPE: ClassVar[str] = "draw_track"


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


@dataclass(frozen=True)
class Qubits:
    """Live circuit-size switch: rebuild the session on the packaged q{n}
    profile (n=4 -> plain default config) and reset to attract mode."""

    n: int
    TYPE: ClassVar[str] = "qubits"


@dataclass(frozen=True)
class SetDriver:
    """Pick which trained quantum weights drive the agent car: "auto" (the
    current track's specialist / honest random-track fallback) or a bundled
    training name like "oval" / "chicane" / "gp" / "universal". The server
    validates against ``welcome.drivers`` (what is actually bundled at the
    active qubit count)."""

    driver: str
    TYPE: ClassVar[str] = "set_driver"


@dataclass(frozen=True)
class HardwareMsg:
    """Run the quantum policy on IBM hardware (or its local fake twin).

    ``iterations`` only applies to ``action="sprint"`` (SPSA iterations).
    ``max_decisions`` only applies to ``action="lap"``: caps the rollout at N
    backend decisions (test-friendly; default is the env episode cap).
    """

    action: str  # "lap" | "sprint" | "abort"
    backend: str  # "fake" | "real"
    iterations: int | None = None
    shots: int | None = None
    max_decisions: int | None = None
    TYPE: ClassVar[str] = "hardware"


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
    obs_labels: list | None = None  # display names of the observation features
    driver: str = "auto"  # active quantum driver selection (see SetDriver)
    drivers: tuple = ("auto",)  # driver choices bundled at this qubit count
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
class HardwareStatus:
    """Progress of a hardware lap/sprint job (all optionals omitted when None)."""

    phase: str  # one of HARDWARE_PHASES
    backend_name: str | None = None
    message: str | None = None
    decision: int | None = None  # lap: decisions completed so far
    seconds_per_decision: float | None = None
    iteration: int | None = None  # sprint: SPSA iterations completed so far
    loss: float | None = None
    eval_return_before: float | None = None
    eval_return_after: float | None = None
    lap_time: float | None = None
    TYPE: ClassVar[str] = "hardware_status"


@dataclass(frozen=True)
class Leaderboard:
    """Per-track leaderboard: ``entries`` are ranked named human laps
    (``{name, lap_s, date}``, fastest first); ``references`` are the AI
    drivers' best laps (``{kind, driver, lap_s}``), shown for comparison but
    never ranked."""

    track: str
    entries: list
    references: list
    TYPE: ClassVar[str] = "leaderboard"


@dataclass(frozen=True)
class Control:
    """Per-client driver-lock status: whether YOU hold the wheel, whether
    anyone does, how many other clients are connected, plus the turn queue —
    how many stand in line, this client's 1-based place in it (null when not
    queued), and the seconds until the current turn expires (null when no
    countdown runs, i.e. nobody is waiting)."""

    driving: bool
    locked: bool
    watchers: int
    waiting: int = 0
    queue_pos: int | None = None
    turn_ends_in_s: int | None = None
    TYPE: ClassVar[str] = "control"


@dataclass(frozen=True)
class Error:
    message: str
    TYPE: ClassVar[str] = "error"


# ------------------------------------------------------------------- serialize

# Optional fields dropped from the wire when None (nullable-but-required fields
# like telemetry.loss and car.last_lap_time stay as explicit nulls).
_OMIT_IF_NONE: dict[str, set[str]] = {
    Input.TYPE: {"steer", "throttle", "brake"},
    SetTrack.TYPE: {"seed", "length"},
    Welcome.TYPE: {"obs_labels"},
    Train.TYPE: {"track", "episodes"},
    Race.TYPE: {"track"},
    HardwareMsg.TYPE: {"iterations", "shots", "max_decisions"},
    Event.TYPE: {"car_id", "lap_time", "agent"},
    Telemetry.TYPE: {"best_lap_s", "lap_times"},
    HardwareStatus.TYPE: {"backend_name", "message", "decision", "seconds_per_decision",
                          "iteration", "loss", "eval_return_before", "eval_return_after",
                          "lap_time"},
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


def _clamped_float(d: dict, key: str, lo: float, hi: float) -> float | None:
    """Optional analog axis: validated as a number, then clamped to [lo, hi]."""
    if d.get(key) is None:
        return None
    return min(hi, max(lo, _float(d[key], key)))


def _parse_input(d: dict) -> Input:
    _check_extra(d, {"keys", "steer", "throttle", "brake"})
    return Input(
        keys=_int(_req(d, "keys"), "keys", 0, KEYS_ALL),
        steer=_clamped_float(d, "steer", -1.0, 1.0),
        throttle=_clamped_float(d, "throttle", 0.0, 1.0),
        brake=_clamped_float(d, "brake", 0.0, 1.0),
    )


def _parse_set_mode(d: dict) -> SetMode:
    _check_extra(d, {"mode"})
    return SetMode(mode=_enum(_req(d, "mode"), "mode", MODES))


def _parse_set_track(d: dict) -> SetTrack:
    _check_extra(d, {"track", "seed", "length"})
    return SetTrack(
        track=_str(_req(d, "track"), "track"),
        seed=_int(d["seed"], "seed", 0) if d.get("seed") is not None else None,
        length=_enum(d["length"], "length", TRACK_LENGTHS)
        if d.get("length") is not None else None,
    )


def _parse_set_name(d: dict) -> SetName:
    _check_extra(d, {"name"})
    name = _req(d, "name")
    if not isinstance(name, str) or len(name) > 24:
        raise ProtocolError("'name' must be a string of at most 24 characters")
    return SetName(name=name.strip())


def _parse_draw_track(d: dict) -> DrawTrack:
    _check_extra(d, {"points"})
    raw = _req(d, "points")
    if not isinstance(raw, list) or len(raw) < 8:
        raise ProtocolError("'points' must be a list of at least 8 [x, y] pairs")
    if len(raw) > 5000:
        raise ProtocolError(f"'points' too long ({len(raw)} > 5000)")
    return DrawTrack(points=[_num_list(p, "points[]", 2) for p in raw])


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


def _parse_qubits(d: dict) -> Qubits:
    _check_extra(d, {"n"})
    return Qubits(n=_int(_req(d, "n"), "n", 1))


def _parse_set_driver(d: dict) -> SetDriver:
    _check_extra(d, {"driver"})
    return SetDriver(driver=_str(_req(d, "driver"), "driver"))


def _parse_hardware(d: dict) -> HardwareMsg:
    _check_extra(d, {"action", "backend", "iterations", "shots", "max_decisions"})
    return HardwareMsg(
        action=_enum(_req(d, "action"), "action", HARDWARE_ACTIONS),
        backend=_enum(_req(d, "backend"), "backend", HARDWARE_BACKENDS),
        iterations=_int(d["iterations"], "iterations", 1)
        if d.get("iterations") is not None else None,
        shots=_int(d["shots"], "shots", 1) if d.get("shots") is not None else None,
        max_decisions=_int(d["max_decisions"], "max_decisions", 1)
        if d.get("max_decisions") is not None else None,
    )


_CLIENT_PARSERS = {
    Hello.TYPE: _parse_hello,
    Input.TYPE: _parse_input,
    SetMode.TYPE: _parse_set_mode,
    SetTrack.TYPE: _parse_set_track,
    SetName.TYPE: _parse_set_name,
    DrawTrack.TYPE: _parse_draw_track,
    Train.TYPE: _parse_train,
    Race.TYPE: _parse_race,
    Qubits.TYPE: _parse_qubits,
    SetDriver.TYPE: _parse_set_driver,
    HardwareMsg.TYPE: _parse_hardware,
}


def parse_client(
    data: Any,
) -> (Hello | Input | SetMode | SetTrack | SetName | DrawTrack | Train | Race
      | Qubits | SetDriver | HardwareMsg):
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
        obs_labels=[_str(s, "obs_labels[]") for s in d["obs_labels"]]
        if d.get("obs_labels") is not None else None,
        driver=_str(d["driver"], "driver") if d.get("driver") is not None else "auto",
        drivers=tuple(_str(x, "drivers[]") for x in d["drivers"])
        if d.get("drivers") is not None else ("auto",),
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


def _parse_leaderboard(d: dict) -> Leaderboard:
    entries = _req(d, "entries")
    references = _req(d, "references")
    if not isinstance(entries, list) or not isinstance(references, list):
        raise ProtocolError("'entries' and 'references' must be lists")
    for e in entries:
        _str(_req(e, "name"), "entries[].name")
        _float(_req(e, "lap_s"), "entries[].lap_s")
    for r in references:
        _str(_req(r, "kind"), "references[].kind")
        _float(_req(r, "lap_s"), "references[].lap_s")
    return Leaderboard(track=_str(_req(d, "track"), "track"),
                       entries=entries, references=references)


def _parse_control(d: dict) -> Control:
    return Control(
        driving=_bool(_req(d, "driving"), "driving"),
        locked=_bool(_req(d, "locked"), "locked"),
        watchers=_int(_req(d, "watchers"), "watchers", 0),
        waiting=_int(d["waiting"], "waiting", 0) if d.get("waiting") is not None else 0,
        queue_pos=_opt_int(d, "queue_pos", 1),
        turn_ends_in_s=_opt_int(d, "turn_ends_in_s", 0),
    )


def _parse_error(d: dict) -> Error:
    return Error(message=_str(_req(d, "message"), "message"))


def _opt_int(d: dict, key: str, lo: int | None = None) -> int | None:
    return _int(d[key], key, lo) if d.get(key) is not None else None


def _parse_hardware_status(d: dict) -> HardwareStatus:
    return HardwareStatus(
        phase=_enum(_req(d, "phase"), "phase", HARDWARE_PHASES),
        backend_name=_str(d["backend_name"], "backend_name")
        if d.get("backend_name") is not None else None,
        message=_str(d["message"], "message") if d.get("message") is not None else None,
        decision=_opt_int(d, "decision", 0),
        seconds_per_decision=_opt_float(d, "seconds_per_decision"),
        iteration=_opt_int(d, "iteration", 0),
        loss=_opt_float(d, "loss"),
        eval_return_before=_opt_float(d, "eval_return_before"),
        eval_return_after=_opt_float(d, "eval_return_after"),
        lap_time=_opt_float(d, "lap_time"),
    )


_SERVER_PARSERS = {
    Welcome.TYPE: _parse_welcome,
    TrackMsg.TYPE: _parse_track_msg,
    State.TYPE: _parse_state,
    Quantum.TYPE: _parse_quantum,
    Telemetry.TYPE: _parse_telemetry,
    Event.TYPE: _parse_event,
    HardwareStatus.TYPE: _parse_hardware_status,
    Control.TYPE: _parse_control,
    Leaderboard.TYPE: _parse_leaderboard,
    Error.TYPE: _parse_error,
}


def parse_server(
    data: Any,
) -> (Welcome | TrackMsg | State | Quantum | Telemetry | Event | HardwareStatus
      | Control | Leaderboard | Error):
    """Parse a server -> client dict (used by tests and client tooling)."""
    if not isinstance(data, dict):
        raise ProtocolError("message must be a JSON object")
    type_tag = data.get("type")
    parser = _SERVER_PARSERS.get(type_tag)
    if parser is None:
        raise ProtocolError(f"unknown message type: {type_tag!r}")
    return parser(data)
