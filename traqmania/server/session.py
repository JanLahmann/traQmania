"""Demo session state machine: attract / train / race, tickable without asyncio.

The session advances in 60 Hz physics substeps via :meth:`DemoSession.tick`;
outgoing protocol messages accumulate in an outbox that the asyncio loop
(:meth:`DemoSession.run`) drains into the websocket hub.  Tests call ``tick()``
directly, so nothing here requires a running event loop.

Cars in attract/race are stepped manually with :class:`CarPhysics` +
:class:`Track` (not through :class:`RacingEnv`): manual stepping gives true
60 Hz car states for broadcasting, lets the human car survive off-track
excursions (freeze 1 s, then respawn), and lets agent actions be held between
10 Hz decisions exactly like training.  Training mode runs :class:`DQNTrainer`
in background threads and samples its live envs via
``RacingEnv.state_snapshot()``.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from traqmania.agents.base import ACTIONS, N_ACTIONS
from traqmania.agents.classical import MLPQFunction
from traqmania.agents.quantum.circuit import circuit_spec
from traqmania.agents.quantum.qdqn import QuantumQFunction
from traqmania.agents.training import DQNTrainer
from traqmania.config import load_config
from traqmania.env.car import CarPhysics
from traqmania.env.racing_env import RacingEnv
from traqmania.server import protocol
from traqmania.server.runtime import (
    LEADERBOARD_MAX_ENTRIES,
    N_EVOLUTION_STAGES,
    WEIGHTS_DIR,
    available_tracks,
    best_stage_label,
    evolution_stage_specs,
    load_agent,
    load_ghost,
    load_leaderboard,
    load_track,
    resolve_training_cfg,
    save_ghost,
    save_leaderboard,
    track_payload,
)

HUMAN_RESPAWN_DELAY_S = 1.0
QUANTUM_MSG_MIN_INTERVAL_S = 0.099  # <= 10 Hz
LAP_TIMES_KEEP = 50  # telemetry keeps the last N (episode, lap_s) pairs
GHOST_TRAJ_MAX_POINTS = 20_000  # ~33 min at 10 Hz; laps beyond this are not recorded
HARDWARE_STATUS_MIN_INTERVAL_S = 0.2  # running-phase hardware_status throttled to <= 5 Hz


def keys_to_controls(keys: int) -> tuple[float, float, float]:
    """input.keys bitmask -> (steer, throttle, brake).

    Bits: 1 throttle, 2 brake, 4 left, 8 right. Car steer +1 increases theta
    (counterclockwise = a LEFT turn on screen), so the left key maps to +1.
    Left+right cancel; brake overrides throttle.
    """
    steer = 0.0
    if keys & protocol.KEY_LEFT:
        steer += 1.0
    if keys & protocol.KEY_RIGHT:
        steer -= 1.0
    brake = 1.0 if keys & protocol.KEY_BRAKE else 0.0
    throttle = 1.0 if (keys & protocol.KEY_THROTTLE) and not brake else 0.0
    return steer, throttle, brake


def quantum_weights_path(track_name: str, n_qubits: int, suffix: str = "") -> Path:
    """Bundled quantum weights path honoring the n-qubit filename rule:
    ``quantum_<track><suffix>.npz`` at the default 4 qubits and
    ``quantum_<track><suffix>_q<n>.npz`` otherwise (``suffix`` carries the
    ``_warmstart`` / ``_stage<i>`` variants, which follow the same rule)."""
    qtag = "" if int(n_qubits) == 4 else f"_q{int(n_qubits)}"
    return WEIGHTS_DIR / f"quantum_{track_name}{suffix}{qtag}.npz"


RANDOM_TRACK = "random"  # set_track name that triggers procedural generation


def random_track_weights(n_qubits: int, suffix: str = "") -> tuple[Path, str]:
    """(weights path, honest driver label) for a generated random track:
    the trained ``quantum_universal*`` weights when bundled, else the gp
    specialist (measured to lap all three bundled tracks zero-shot)."""
    universal = quantum_weights_path("universal", n_qubits, suffix)
    if universal.is_file():
        return universal, "universal"
    return quantum_weights_path("gp", n_qubits, suffix), "gp-trained generalist"


@dataclass
class _Car:
    """One simulated car in attract/race mode (human or agent-driven)."""

    id: str
    kind: str  # "human" | "quantum" | "mlp"
    state: np.ndarray  # (4,) [x, y, theta, v]
    qfunc: Any = None  # None for the human car
    s: float = 0.0
    progress: float = 0.0
    lap: int = 0
    lap_start_t: float = 0.0
    last_lap_time: float | None = None
    off_track: bool = False
    lap_dirty: bool = False  # went off track during the current lap
    action: int = 0
    controls: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rays: list | None = None
    respawn_at: float | None = None  # sim time to respawn a frozen (crashed) human
    label: str | None = None  # shown next to the car (e.g. "ep 250" in evolution mode)
    controller: Any = None  # continuous-control policy (the hero racing-line driver)
    traj: list = field(default_factory=list)  # (x, y, theta) at decision rate, this lap
    traj_full: bool = True  # False once the trajectory overflowed and was dropped


@dataclass
class TrainingJob:
    """One background training thread plus the shared telemetry it feeds."""

    agent: str
    env: RacingEnv
    trainer: DQNTrainer
    stop_event: threading.Event
    epsilon: float
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    returns: list = field(default_factory=list)
    episode: int = 0
    loss: float | None = None
    history: dict | None = None
    error: str | None = None
    announced: bool = False  # training_done already emitted
    lap_times: list = field(default_factory=list)  # [episode, lap_s] pairs, last <= 50
    best_lap_s: float | None = None
    best_announced: float | None = None  # last best_lap_s emitted as new_best_lap


@dataclass
class HardwareJob:
    """One background hardware-execution thread (lap rollout or SPSA sprint).

    The worker appends ready-to-send ``hardware_status`` payload dicts to
    ``queue`` under ``lock``; the session tick drains them into the outbox.
    """

    kind: str  # "lap" | "sprint"
    stop_event: threading.Event
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    queue: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    aborted: bool = False  # abort requested (via message or mode switch)
    abandoned: bool = False  # mode switched away: reap silently
    last_running_emit: float = -math.inf  # time.monotonic() of last running status

    def push(self, phase: str, **fields: Any) -> None:
        """Thread-safe enqueue of a hardware_status payload (from the worker)."""
        payload = {"type": "hardware_status", "phase": phase, **fields}
        with self.lock:
            self.queue.append(payload)

    def push_running(self, **fields: Any) -> None:
        """Enqueue a running-phase status, throttled to <= 5 Hz."""
        now = time.monotonic()
        with self.lock:
            if now - self.last_running_emit < HARDWARE_STATUS_MIN_INTERVAL_S:
                return
            self.last_running_emit = now
            self.queue.append({"type": "hardware_status", "phase": "running", **fields})

    def drain(self) -> list[dict]:
        with self.lock:
            out = self.queue
            self.queue = []
        return out


class _TrainingLapMonitor:
    """Env wrapper fed to :class:`DQNTrainer` that records completed lap times.

    Runs on the training thread; appends (episode, lap_s) pairs and the best
    lap into the shared :class:`TrainingJob` fields under ``job.lock`` so the
    session tick can read them for telemetry / new_best_lap events.
    """

    def __init__(self, env: RacingEnv, job: TrainingJob | None = None) -> None:
        self.env = env
        self.job = job  # assigned right after the TrainingJob is built
        self.episodes_done = 0
        self._prev_lap = np.zeros(env.n_envs, dtype=np.int64)

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._prev_lap[:] = 0
        return obs

    def step(self, actions: np.ndarray):
        obs, reward, done, info = self.env.step(actions)
        lap = np.asarray(info["lap"])
        lap_done = lap > self._prev_lap
        if np.any(lap_done):
            times = np.asarray(info["last_lap_time"])[lap_done]
            job = self.job
            with job.lock:
                for lap_s in times:
                    if math.isnan(lap_s):
                        continue
                    job.lap_times.append([self.episodes_done, float(lap_s)])
                    if job.best_lap_s is None or lap_s < job.best_lap_s:
                        job.best_lap_s = float(lap_s)
                del job.lap_times[:-LAP_TIMES_KEEP]
        self._prev_lap = np.where(done, 0, lap)
        self.episodes_done += int(np.sum(done))
        return obs, reward, done, info


class DemoSession:
    """Single shared demo session (one kiosk): mode state machine + tick loop."""

    def __init__(self, config: dict, ghosts_dir: Path | None = None,
                 leaderboard_dir: Path | None = None):
        self._ghosts_dir = ghosts_dir  # None -> bundled traqmania/data/ghosts
        if leaderboard_dir is None and ghosts_dir is not None:
            # a test/host that isolates ghosts wants leaderboards isolated too
            leaderboard_dir = Path(ghosts_dir) / "leaderboard"
        self._leaderboard_dir = leaderboard_dir  # None -> data/leaderboard
        self._apply_config(config)

        self.t = 0.0
        self._substep = 0
        self._outbox: list[dict] = []
        self._keys = 0
        self._analog: tuple[float, float, float] | None = None
        self._agent_cache: dict[tuple[str, str], Any] = {}
        self._last_quantum_emit: dict[str, float] = {}
        self.hw_job: HardwareJob | None = None
        self._hw_replay: dict | None = None  # {points, lap_time, start_substep}

        self.mode = "attract"
        self.driver = "auto"  # SetDriver override: which training drives the agent
        self.track_name = str(config["track"]["default"])
        self.track_is_random = False
        self._random_seed_rng = np.random.default_rng()  # seeds for seedless rerolls
        self._drawn_count = 0  # numbers the "drawn #N" track names
        self.track = load_track(config, self.track_name)
        self._ghost: dict | None = load_ghost(self.track_name, self._ghosts_dir)
        self.racer_name = ""  # leaderboard display name; empty = don't record
        self._board = load_leaderboard(self.track_name, self._leaderboard_dir)
        self.cars: list[_Car] = []
        self.jobs: dict[str, TrainingJob] = {}
        self._enter_attract()

    def _apply_config(self, config: dict) -> None:
        """Set every config-derived attribute (shared by ``__init__`` and the
        live qubit switch, which swaps ``config`` wholesale)."""
        self.config = config
        physics = config["physics"]
        self.dt = float(physics["dt"])
        self.substeps_per_decision = int(physics["substeps_per_decision"])
        self.car_physics = CarPhysics(physics)

        obs_cfg = config["observation"]
        self.ray_angles = np.deg2rad(np.asarray(obs_cfg["ray_angles_deg"], dtype=np.float64))
        self.ray_max_dist = float(obs_cfg["ray_max_dist"])
        self.n_qubits = int(config.get("circuit", {}).get("n_qubits", 4))

        server_cfg = config["server"]
        hz = 1.0 / self.dt
        self.broadcast_every = max(1, round(hz / float(server_cfg["broadcast_hz"])))
        self.telemetry_every = max(1, round(hz / float(server_cfg["telemetry_hz"])))

    # -------------------------------------------------------------- messaging

    def welcome_payload(self) -> dict:
        return {
            "type": "welcome",
            "mode": self.mode,
            "track": track_payload(self.track),
            "tracks": available_tracks(),
            "circuit_spec": circuit_spec(self.config),
            "ui": dict(self.config["ui"]),
            "obs_labels": self._obs_labels(),
            "driver": self.driver,
            "drivers": self.available_drivers(),
        }

    def available_drivers(self) -> list[str]:
        """Driver choices for SetDriver: "auto" plus every training whose
        quantum weights are bundled at the active qubit count, plus "hero"
        (the model-based racing-line controller — no weights, any track;
        the UI only offers it in expert mode)."""
        return ["auto"] + [
            name for name in ("oval", "chicane", "gp", "universal")
            if quantum_weights_path(name, self.n_qubits).is_file()
        ] + ["hero"] + (
            ["pro"] if (WEIGHTS_DIR / "mlp_pro.npz").is_file() else []
        )

    def _obs_labels(self) -> list[str]:
        """Display names of the observation features feeding the circuit,
        from ``env.feature_names`` when the env provides it (older envs get
        the default rays-then-speed labels)."""
        env = RacingEnv(self.track, self.config, n_envs=1, seed=0)
        names = getattr(env, "feature_names", None)
        if names is None:
            angles = self.config["observation"]["ray_angles_deg"]
            names = [f"ray {a:+.0f}°" if a else "ray 0°" for a in angles] + ["speed"]
        return [str(name) for name in names]

    def drain_outbox(self) -> list[dict]:
        out = self._outbox
        self._outbox = []
        return out

    def _error(self, message: str) -> None:
        self._outbox.append({"type": "error", "message": message})

    def _event(self, kind: str, **fields: Any) -> None:
        self._outbox.append({"type": "event", "kind": kind, **fields})

    def set_input(self, keys: int, analog: tuple[float, float, float] | None = None) -> None:
        """Last-writer-wins human input: bitmask plus optional analog override.

        ``analog`` is ``(steer, throttle, brake)``; when not None it drives the
        human car instead of the keys bitmask (until a later keys-only input).
        Client steer is stick-sign (+1 = stick right); car steer +1 turns left
        (theta increases), so the sign flips here.
        """
        self._keys = int(keys)
        self._analog = (-analog[0], analog[1], analog[2]) if analog is not None else None

    def handle_message(self, msg: Any) -> None:
        """Apply a parsed client message to the session state."""
        if isinstance(msg, protocol.Hello):
            return  # welcome is sent per-connection by the ws layer
        if isinstance(msg, protocol.Input):
            analog = None
            if msg.steer is not None or msg.throttle is not None or msg.brake is not None:
                analog = (msg.steer or 0.0, msg.throttle or 0.0, msg.brake or 0.0)
            self.set_input(msg.keys, analog)
        elif isinstance(msg, protocol.SetMode):
            self._set_mode(msg.mode)
        elif isinstance(msg, protocol.SetTrack):
            self._set_track(msg.track, msg.seed, msg.length)
        elif isinstance(msg, protocol.SetName):
            self.racer_name = msg.name  # already stripped/length-capped
        elif isinstance(msg, protocol.DrawTrack):
            self._handle_draw_track(msg)
        elif isinstance(msg, protocol.Train):
            self._handle_train(msg)
        elif isinstance(msg, protocol.Race):
            self._handle_race(msg)
        elif isinstance(msg, protocol.Qubits):
            self._handle_qubits(msg)
        elif isinstance(msg, protocol.SetDriver):
            self._handle_driver(msg)
        elif isinstance(msg, protocol.HardwareMsg):
            self._handle_hardware(msg)
        else:
            self._error(f"unhandled message: {msg!r}")

    # -------------------------------------------------------- weight resolution

    def _quantum_weights_path(self, suffix: str = "") -> Path:
        if suffix == "" and self.driver not in ("auto", "hero", "pro"):
            # explicit driver pick: that training's weights, whatever the track
            # ("hero" has no weights — quantum cars fall through to the track
            # specialist while the hero pick only affects the attract car)
            return quantum_weights_path(self.driver, self.n_qubits)
        if self.track_is_random:  # no per-track specialist exists: fall back
            return random_track_weights(self.n_qubits, suffix)[0]
        return quantum_weights_path(self.track_name, self.n_qubits, suffix)

    def _load_quantum_qfunc(self, path: Path) -> QuantumQFunction:
        qfunc = QuantumQFunction(self.config["circuit"])
        qfunc.set_params(np.load(path)["params"])
        return qfunc

    def _agent_unavailable(self, kind: str) -> str | None:
        """Why the bundled ``kind`` weights cannot drive the current track (or None)."""
        if kind == "quantum":
            path = self._quantum_weights_path()
            if path.is_file():
                return None
            trained = [name for name in ("oval", "chicane", "gp", "combo", "universal")
                       if quantum_weights_path(name, self.n_qubits).is_file()]
            hint = (f"bundled at {self.n_qubits} qubits: {', '.join(trained)}"
                    if trained else
                    f"no training is bundled at {self.n_qubits} qubits yet")
            return (f"missing weights '{path.name}' ({hint}) — switch track or "
                    "qubit count, or train one in the Train tab")
        if self.n_qubits != 4:  # bundled mlp weights expect the 4-feature observation
            return f"bundled mlp weights use 4 features, config has {self.n_qubits}"
        path = WEIGHTS_DIR / f"mlp_{self.track_name}.npz"
        return None if path.is_file() else f"missing weights '{path.name}'"

    def _evolution_stage_specs(self) -> list[tuple[str, Path]]:
        """:func:`evolution_stage_specs` honoring the n-qubit weight filename rule.

        Returns [] when no suitable q{n} weights exist (evolution unavailable)."""
        if self.n_qubits == 4:
            return evolution_stage_specs(self.track_name)
        specs = [
            (f"stage {i}", path)
            for i in range(1, N_EVOLUTION_STAGES + 1)
            if (path := self._quantum_weights_path(f"_stage{i}")).is_file()
        ]
        if specs:
            best = quantum_weights_path(self.track_name, self.n_qubits)
            if best.is_file():
                specs[-1] = (best_stage_label(best), best)  # end on the shipped driver
            return specs
        pair = [("warm-start", self._quantum_weights_path("_warmstart")),
                (best_stage_label(quantum_weights_path(self.track_name, self.n_qubits)),
                 quantum_weights_path(self.track_name, self.n_qubits))]
        return pair if all(path.is_file() for _, path in pair) else []

    def _weights_unavailable(self, mode: str, opponent: str = "quantum") -> str | None:
        """Why bundled weights block switching to ``mode`` (None when they don't).

        train mode needs none (a missing warm start falls back to a cold start)."""
        if mode == "hardware" and self.track_is_random:
            # the hardware runner loads tracks by bundled name (Track.load)
            return "hardware execution needs a bundled track"
        if mode == "attract" and self.driver in ("hero", "pro"):
            return None  # reference drivers work on any track
        if mode in ("attract", "hardware"):
            return self._agent_unavailable("quantum")
        if mode == "race":
            return self._agent_unavailable(opponent)
        if mode == "evolution":
            specs = self._evolution_stage_specs()
            if not specs or not all(path.is_file() for _, path in specs):
                return (f"no evolution stage weights for track '{self.track_name}' "
                        f"at {self.n_qubits} qubits")
        return None

    def _try_make_agent_car(self, kind: str) -> _Car | None:
        """:meth:`_make_agent_car`, degrading to an error + None on missing weights."""
        reason = self._agent_unavailable(kind)
        if reason is None:
            return self._make_agent_car(kind)
        self._error(f"no '{kind}' agent for track '{self.track_name}': {reason}")
        return None

    # ------------------------------------------------------------ mode switches

    def _training_alive(self) -> bool:
        return any(j.thread is not None and j.thread.is_alive() for j in self.jobs.values())

    def _set_mode(self, mode: str) -> None:
        reason = self._weights_unavailable(mode)
        if reason is not None:  # reject the switch; stay in the previous mode
            self._error(f"cannot switch to '{mode}' mode: {reason}")
            return
        if mode != "train":
            self.stop_training()
        if mode != "hardware":
            self._abandon_hardware()
        self.mode = mode
        if mode == "attract":
            self._enter_attract()
        elif mode == "race":
            self._enter_race("quantum")
        elif mode == "evolution":
            self._enter_evolution()
        elif mode == "hardware":
            self._enter_hardware()
        else:  # train: cars appear once a train{start} arrives
            self.cars = []

    def _set_track(self, name: str, seed: int | None = None,
                   length: str | None = None) -> bool:
        if self._training_alive():
            self._error("cannot change track while training is running")
            return False
        if self._hardware_alive():
            self._error("cannot change track while a hardware job is running")
            return False
        if self.track_is_random and name == self.track_name:
            pass  # race/train restarts name the current generated track: keep it
        elif name == RANDOM_TRACK:
            track = self._generate_random_track(seed, length)
            if track is None:
                return False
            self.track = track
            self.track_name = track.name
            self.track_is_random = True
            self._ghost = None  # ghosts are never stored for random tracks
        else:
            if name not in available_tracks():
                self._error(f"unknown track '{name}'")
                return False
            self.track = load_track(self.config, name)
            self.track_name = name
            self.track_is_random = False
            self._ghost = load_ghost(name, self._ghosts_dir)
        self._reload_board()
        self._outbox.append({"type": "track", "track": track_payload(self.track)})
        self._outbox.append(self.leaderboard_payload())
        if self.mode == "attract":
            self._enter_attract()
        elif self.mode == "race":
            opponent = next((c.kind for c in self.cars if c.kind != "human"), "quantum")
            self._enter_race(opponent)
        elif self.mode == "evolution":
            self._enter_evolution()
        elif self.mode == "hardware":
            self._enter_hardware()
        return True

    def _generate_random_track(self, seed: int | None, length: str | None = None):
        """Fresh procedural track named ``random #<seed>`` (a seedless request
        rolls one; a non-default length is tagged into the name so the label
        stays reproducible), or None with an error queued when generation
        fails."""
        try:
            from traqmania.env.trackgen import generate_track
        except ImportError:
            self._error("random tracks are not available in this build")
            return None
        if seed is None:
            seed = int(self._random_seed_rng.integers(0, 1_000_000))
        length = length or "medium"
        suffix = "" if length == "medium" else f" ({length})"
        try:
            return generate_track(seed,
                                  resample_spacing=self.config["track"]["resample_spacing"],
                                  # 0.65: gp-and-tighter corners — the universal
                                  # driver still laps 10/10 here, and milder
                                  # settings read as boring under the v2 physics
                                  difficulty=0.65,
                                  name=f"random #{seed}{suffix}",
                                  length=length)
        except Exception as exc:
            self._error(f"random track generation failed (seed {seed}): {exc}")
            return None

    def _handle_draw_track(self, msg: protocol.DrawTrack) -> None:
        """Turn a hand-drawn centerline into the current track.

        Drawn tracks behave like generated ones: no stored ghosts, and the
        quantum car falls back to the universal driver (no specialist exists).
        Validation failures come back as a user-facing error explaining what
        to fix — the natural "adjust" flow is to draw again."""
        if self._training_alive():
            self._error("cannot change track while training is running")
            return
        if self._hardware_alive():
            self._error("cannot change track while a hardware job is running")
            return
        try:
            from traqmania.env.trackgen import track_from_drawing
        except ImportError:
            self._error("drawn tracks are not available in this build")
            return
        try:
            track = track_from_drawing(
                msg.points, self.config["track"]["resample_spacing"],
                name=f"drawn #{self._drawn_count + 1}")
        except ValueError as exc:
            self._error(f"could not build the drawn track: {exc}")
            return
        self._drawn_count += 1
        self.track = track
        self.track_name = track.name
        self.track_is_random = True  # same fallbacks as generated tracks
        self._ghost = None
        self._reload_board()
        self._outbox.append({"type": "track", "track": track_payload(self.track)})
        self._outbox.append(self.leaderboard_payload())
        if self.mode == "attract":
            self._enter_attract()
        elif self.mode == "race":
            opponent = next((c.kind for c in self.cars if c.kind != "human"), "quantum")
            self._enter_race(opponent)
        elif self.mode == "evolution":
            self._enter_evolution()
        elif self.mode == "hardware":
            self._enter_hardware()

    def _handle_driver(self, msg: protocol.SetDriver) -> None:
        """Pick which training's quantum weights drive the agent car; rebuilds
        the attract car immediately, race/evolution pick it up on restart."""
        if msg.driver not in self.available_drivers():
            self._error(f"unknown driver '{msg.driver}' at {self.n_qubits} qubits "
                        f"(available: {', '.join(self.available_drivers())})")
            return
        self.driver = msg.driver
        if self.mode == "attract":
            self._enter_attract()
        self._outbox.append(self.welcome_payload())

    def _handle_qubits(self, msg: protocol.Qubits) -> None:
        """Live circuit-size switch: overlay the packaged q{n} profile (plain
        default config at n=4), rebuild track/agents/spec state in place, reset
        to attract mode, and re-broadcast the welcome payload."""
        if self._training_alive():
            self._error("cannot change qubit count while training is running")
            return
        if self._hardware_alive():
            self._error("cannot change qubit count while a hardware job is running")
            return
        try:
            config = load_config() if msg.n == 4 else load_config(f"q{msg.n}")
        except FileNotFoundError:
            self._error(f"unknown qubit count {msg.n} (no packaged q{msg.n} profile)")
            return
        self._apply_config(config)
        if not self.track_is_random:  # a generated track keeps its geometry
            self.track = load_track(config, self.track_name)
            self._ghost = load_ghost(self.track_name, self._ghosts_dir)
        self._agent_cache.clear()  # cached qfuncs were built for the old circuit
        self._last_quantum_emit.clear()
        self._abandon_hardware()
        if self.driver not in self.available_drivers():
            self.driver = "auto"  # picked training isn't bundled at this size
        self.mode = "attract"
        self._enter_attract()  # graceful car-less fallback when q{n} is untrained
        self._outbox.append(self.welcome_payload())

    def _enter_attract(self) -> None:
        if self.driver in ("hero", "pro"):
            self.cars = [self._make_hero_car(self.driver)]
            return
        car = self._try_make_agent_car("quantum")
        self.cars = [car] if car is not None else []

    def _make_hero_car(self, which: str = "hero") -> _Car:
        """The expert-demo reference cars, honestly labelled: "hero" is the
        model-based racing-line controller (not learned), "pro" the biggest
        classical MLP we train — same double-DQN recipe as every other agent,
        just more parameters and a richer observation."""
        key = (which, self.track_name, "")
        if key not in self._agent_cache:
            if which == "hero":
                from traqmania.env.racing_line import RacingLineController

                self._agent_cache[key] = RacingLineController(
                    self.track, self.config["physics"])
            else:
                from traqmania.env.pro import N_FEATURES, ProController

                params = np.load(WEIGHTS_DIR / "mlp_pro.npz")["params"]
                hidden = (params.size - N_ACTIONS) // (N_FEATURES + 1 + N_ACTIONS)
                qfunc = MLPQFunction(n_features=N_FEATURES, hidden=hidden,
                                     n_actions=N_ACTIONS)
                qfunc.set_params(params)
                self._agent_cache[key] = ProController(
                    self.track, self.config["physics"], qfunc)
        label = ("driver: racing line (model-based, not learned)" if which == "hero"
                 else "driver: pro (classical DQN, big MLP)")
        x, y, theta = self.track.start_pose()
        car = _Car(id=which, kind=which, state=np.array([x, y, theta, 0.0]),
                   controller=self._agent_cache[key], label=label)
        self._respawn(car)
        return car

    def _enter_race(self, opponent: str) -> None:
        x, y, theta = self.track.start_pose()
        human = _Car(id="human", kind="human", state=np.array([x, y, theta, 0.0]))
        self._respawn(human)
        agent = self._try_make_agent_car(opponent)
        self.cars = [human] + ([agent] if agent is not None else [])

    def _make_agent_car(self, kind: str) -> _Car:
        driver = self.driver if kind == "quantum" else ""
        key = (kind, self.track_name, driver)
        if key not in self._agent_cache:
            if kind == "quantum":
                self._agent_cache[key] = self._load_quantum_qfunc(self._quantum_weights_path())
            else:
                self._agent_cache[key] = load_agent(kind, self.track_name, config=self.config)
        x, y, theta = self.track.start_pose()
        label = None
        if kind == "quantum" and self.driver != "auto":  # honest override labeling
            label = f"driver: {self.driver}-trained"
        elif self.track_is_random and kind == "quantum":  # honest fallback labeling
            label = f"driver: {random_track_weights(self.n_qubits)[1]}"
        car = _Car(id=kind, kind=kind, state=np.array([x, y, theta, 0.0]),
                   qfunc=self._agent_cache[key], label=label)
        self._respawn(car)
        return car

    def _enter_evolution(self) -> None:
        """4 quantum cars driving different training-stage weights, labelled 'ep N'."""
        reason = self._weights_unavailable("evolution")
        if reason is not None:
            self._error(f"cannot enter evolution mode: {reason}")
            self.cars = []
            return
        x, y, theta = self.track.start_pose()
        cars = []
        for i, (label, path) in enumerate(self._evolution_stage_specs(), start=1):
            key = ("stage", path.name)
            if key not in self._agent_cache:
                qfunc = QuantumQFunction(self.config["circuit"])
                qfunc.set_params(np.load(path)["params"])
                self._agent_cache[key] = qfunc
            car = _Car(id=f"stage{i}", kind="quantum", state=np.array([x, y, theta, 0.0]),
                       qfunc=self._agent_cache[key], label=label)
            self._respawn(car)
            cars.append(car)
        self.cars = cars

    def _enter_hardware(self) -> None:
        """Current track with one idle fastsim quantum car (drives during replay)."""
        self._hw_replay = None
        car = self._try_make_agent_car("quantum")
        self.cars = [car] if car is not None else []
        self._outbox.append({"type": "hardware_status", "phase": "idle"})

    def _handle_race(self, msg: protocol.Race) -> None:
        if msg.track is not None and not self._set_track(msg.track):
            return
        if msg.action == "start":
            reason = self._weights_unavailable("race", msg.opponent)
            if reason is not None:  # reject; stay in the previous mode
                self._error(f"cannot start race: {reason}")
                return
            self.stop_training()
            self.mode = "race"
            self._enter_race(msg.opponent)
        elif self.mode == "race":  # reset
            for car in self.cars:
                self._respawn(car)
        else:
            self._error("race reset only applies in race mode")

    # ---------------------------------------------------------------- training

    def _handle_train(self, msg: protocol.Train) -> None:
        if msg.action == "stop":
            self.stop_training()
            return
        if self._training_alive():
            self._error("training is already running")
            return
        if msg.track is not None and not self._set_track(msg.track):
            return
        self.mode = "train"
        self.cars = []
        self.jobs = {}
        agents = ("quantum", "mlp") if msg.agent == "both" else (msg.agent,)
        for offset, agent in enumerate(agents):
            self._start_job(agent, warm=msg.warm, episodes=msg.episodes, seed_offset=offset)

    def _start_job(self, agent: str, warm: bool, episodes: int | None, seed_offset: int) -> None:
        warm_path: Path | None = None
        if warm and agent == "quantum":
            warm_path = self._quantum_weights_path("_warmstart")
            if not warm_path.is_file():  # graceful fallback: cold start instead
                self._error(f"warm-start weights '{warm_path.name}' not found; "
                            "training quantum from scratch")
                warm_path, warm = None, False
        tcfg = resolve_training_cfg(self.config, self.track_name, warm)
        episodes = int(episodes) if episodes is not None else int(tcfg["episodes"])
        seed = int(tcfg.get("seed", 0)) + seed_offset

        env = RacingEnv(self.track, self.config, n_envs=int(tcfg["n_parallel_envs"]), seed=seed)
        if warm_path is not None:
            qfunc: Any = self._load_quantum_qfunc(warm_path)
        elif agent == "quantum":
            qfunc = QuantumQFunction(self.config["circuit"], seed=seed)
        else:
            qfunc = MLPQFunction(n_features=env.n_features, n_actions=N_ACTIONS, seed=seed)

        track, config = self.track, self.config  # bind now: self.track may change later

        def env_factory(track=track, config=config, seed=seed) -> RacingEnv:
            return RacingEnv(track, config, n_envs=4, seed=seed + 10_000)

        stop_event = threading.Event()
        monitor = _TrainingLapMonitor(env)  # records lap times into the job (set below)
        trainer = DQNTrainer(qfunc, monitor, tcfg, rng=np.random.default_rng(seed),
                             env_factory=env_factory, stop_event=stop_event)
        job = TrainingJob(agent=agent, env=env, trainer=trainer, stop_event=stop_event,
                          epsilon=float(tcfg["epsilon_start"]))
        monitor.job = job

        def callback(episode: int, stats: dict, job: TrainingJob = job) -> None:
            with job.lock:
                job.returns.append(float(stats["returns"]))
                job.episode = int(episode)
                job.epsilon = float(stats["epsilon"])
                job.loss = stats["loss"]

        def run(job: TrainingJob = job, episodes: int = episodes) -> None:
            try:
                job.history = job.trainer.train(episodes=episodes, callback=callback)
            except Exception as exc:  # surfaced as an error msg by the reaper
                job.error = f"{type(exc).__name__}: {exc}"

        job.thread = threading.Thread(target=run, name=f"traqmania-train-{agent}", daemon=True)
        self.jobs[agent] = job
        job.thread.start()

    def stop_training(self) -> None:
        for job in self.jobs.values():
            job.stop_event.set()

    def shutdown(self) -> None:
        """Stop training/hardware threads and wait briefly for them to exit."""
        self.stop_training()
        if self.hw_job is not None:
            self.hw_job.stop_event.set()
        for job in self.jobs.values():
            if job.thread is not None:
                job.thread.join(timeout=10.0)
        if self.hw_job is not None and self.hw_job.thread is not None:
            self.hw_job.thread.join(timeout=10.0)

    def _emit_telemetry(self, job: TrainingJob, final: bool = False) -> None:
        with job.lock:
            tail = [float(r) for r in job.returns[-100:]]
            episode, epsilon, loss = job.episode, job.epsilon, job.loss
            best_lap_s = job.best_lap_s
            lap_times = [[int(e), float(t)] for e, t in job.lap_times[-LAP_TIMES_KEEP:]]
        if not tail and not final:
            return
        if loss is not None and (not isinstance(loss, int | float) or math.isnan(loss)):
            loss = None
        self._outbox.append({
            "type": "telemetry",
            "agent": job.agent,
            "episode": int(episode),
            "mean_return": float(np.mean(tail)) if tail else 0.0,
            "epsilon": float(epsilon),
            "loss": None if loss is None else float(loss),
            "returns_tail": tail,
            "best_lap_s": None if best_lap_s is None else float(best_lap_s),
            "lap_times": lap_times,
        })

    def _tick_train(self) -> None:
        if self._substep % self.telemetry_every == 0:
            for job in self.jobs.values():
                with job.lock:
                    best = job.best_lap_s
                if best is not None and (job.best_announced is None or best < job.best_announced):
                    job.best_announced = best
                    self._event("new_best_lap", agent=job.agent, lap_time=float(best))
                if not job.announced:
                    self._emit_telemetry(job)
        for job in self.jobs.values():
            if job.announced or job.thread is None or job.thread.is_alive():
                continue
            self._emit_telemetry(job, final=True)
            if job.error is not None:
                self._error(f"training '{job.agent}' failed: {job.error}")
            self._event("training_done", agent=job.agent)
            job.announced = True

    # ---------------------------------------------------------------- hardware

    def _hardware_alive(self) -> bool:
        job = self.hw_job
        return job is not None and job.thread is not None and job.thread.is_alive()

    def _abandon_hardware(self) -> None:
        """Mode switched away: request cancellation, reap silently."""
        self._hw_replay = None
        if self.hw_job is not None:
            self.hw_job.stop_event.set()
            self.hw_job.aborted = True
            self.hw_job.abandoned = True

    def _handle_hardware(self, msg: protocol.HardwareMsg) -> None:
        if msg.action == "abort":
            if self._hardware_alive():
                self.hw_job.stop_event.set()
                self.hw_job.aborted = True
            else:
                self._outbox.append({"type": "hardware_status", "phase": "idle",
                                     "message": "no hardware job running"})
            return
        if self._hardware_alive():
            self._outbox.append({"type": "hardware_status", "phase": "error",
                                 "message": "a hardware job is already running"})
            return
        if self.hw_job is not None:  # finished but not yet reaped: flush its statuses
            self._tick_hardware()
        reason = self._weights_unavailable("hardware")
        if reason is not None:  # reject; stay in the previous mode
            self._outbox.append({"type": "hardware_status", "phase": "error",
                                 "message": f"cannot run hardware {msg.action}: {reason}"})
            return
        try:
            # Must happen on THIS long-lived thread, before any worker thread
            # touches qiskit — see hardware.ensure_qiskit_imported for the
            # segfault this prevents.
            from traqmania import hardware as hardware_mod
            hardware_mod.ensure_qiskit_imported()
        except Exception as exc:  # e.g. qiskit-ibm-runtime not installed
            self._outbox.append({"type": "hardware_status", "phase": "error",
                                 "message": f"{type(exc).__name__}: {exc}"})
            return
        self.stop_training()
        if self.mode != "hardware":
            self.mode = "hardware"
            self._enter_hardware()
        else:  # reset any previous replay; car waits at the start line again
            self._hw_replay = None
            self.cars = [self._make_agent_car("quantum")]

        hw_cfg = self.config.get("hardware", {})
        shots = int(msg.shots) if msg.shots is not None else int(hw_cfg.get("shots", 1024))
        iterations = (int(msg.iterations) if msg.iterations is not None
                      else int(hw_cfg.get("spsa_iterations", 30)))
        use_fake = msg.backend == "fake"
        track_name = self.track_name
        weights_path = self._quantum_weights_path()
        max_decisions = msg.max_decisions
        config = self.config  # bind now: the worker must see THIS session's circuit
        min_qubits = max(5, self.n_qubits)

        job = HardwareJob(kind=msg.action, stop_event=threading.Event())

        def run(job: HardwareJob = job) -> None:
            from traqmania import hardware

            try:
                job.push("connecting",
                         message="local fake backend" if use_fake
                         else "connecting to IBM Quantum (least busy backend)")
                backend = hardware.get_backend(use_fake=use_fake, min_qubits=min_qubits)
                backend_name = str(getattr(backend, "name", backend))
                job.push("transpiling", backend_name=backend_name,
                         message=f"transpiling circuit for {backend_name}")
                if job.kind == "lap":
                    def on_decision(i: int, info: dict) -> None:
                        job.push_running(backend_name=backend_name, decision=i + 1,
                                         seconds_per_decision=float(info["seconds"]))

                    job.result = hardware.run_hardware_lap(
                        track_name, weights_path, backend, shots=shots,
                        max_decisions=max_decisions, on_decision=on_decision,
                        stop_event=job.stop_event, config=config)
                else:  # sprint
                    def on_iter(k: int, info: dict) -> None:
                        job.push("running", backend_name=backend_name,
                                 iteration=k + 1, loss=float(info["loss"]))

                    job.result = hardware.spsa_sprint(
                        track_name, weights_path, backend, iterations=iterations,
                        shots=shots, on_iter=on_iter, stop_event=job.stop_event,
                        config=config)
            except Exception as exc:  # surfaced as hardware_status error by the reaper
                job.error = f"{type(exc).__name__}: {exc}"

        job.thread = threading.Thread(target=run, name=f"traqmania-hw-{msg.action}",
                                      daemon=True)
        self.hw_job = job
        job.thread.start()

    def _tick_hardware(self) -> None:
        """Drain worker status messages; finalize the job once its thread exits."""
        job = self.hw_job
        if job is None:
            return
        payloads = job.drain()
        if not job.abandoned:
            self._outbox.extend(payloads)
        if job.thread is not None and job.thread.is_alive():
            return
        self.hw_job = None
        if job.abandoned:
            return
        if job.error is not None:
            self._outbox.append({"type": "hardware_status", "phase": "error",
                                 "message": job.error})
        elif job.aborted or (job.result or {}).get("aborted"):
            self._outbox.append({"type": "hardware_status", "phase": "idle",
                                 "message": f"hardware {job.kind} aborted"})
        elif job.kind == "lap":
            self._finish_hardware_lap(job.result)
        else:
            self._outbox.append({
                "type": "hardware_status",
                "phase": "done",
                "eval_return_before": float(job.result["return_before"]),
                "eval_return_after": float(job.result["return_after"]),
            })

    def _finish_hardware_lap(self, result: dict) -> None:
        """Emit done, then replay the recorded trajectory as a pinned ghost car
        looping alongside a live fastsim car driving the same weights."""
        done: dict[str, Any] = {
            "type": "hardware_status",
            "phase": "done",
            "seconds_per_decision": float(result["seconds_per_decision"]),
        }
        if result["lapped"]:
            done["lap_time"] = float(result["best_lap_s"])
        self._outbox.append(done)
        points = [[float(p[0]), float(p[1]), float(p[2])] for p in result["trajectory"]]
        if self.mode == "hardware" and len(points) >= 2:
            self._hw_replay = {
                "points": points,
                "lap_time": float(result["best_lap_s"]) if result["lapped"] else None,
                "start_substep": self._substep,
            }
            self.cars = [self._make_agent_car("quantum")]  # fastsim car, same weights
            self._outbox.append({"type": "hardware_status", "phase": "replay",
                                 "message": "replaying hardware lap"})

    def _hw_replay_payload(self) -> dict:
        """Hardware-lap replay car, looping its 10 Hz trajectory lerped to 60 Hz."""
        replay = self._hw_replay
        points = replay["points"]
        n = len(points)
        phase = (self._substep - replay["start_substep"]) / self.substeps_per_decision
        frac = phase - math.floor(phase)
        i0 = int(math.floor(phase)) % n
        p0, p1 = points[i0], points[(i0 + 1) % n]
        dtheta = (p1[2] - p0[2] + math.pi) % (2.0 * math.pi) - math.pi  # shortest arc
        decision_dt = self.dt * self.substeps_per_decision
        return {
            "id": "hardware",
            "kind": "quantum",
            "x": p0[0] + (p1[0] - p0[0]) * frac,
            "y": p0[1] + (p1[1] - p0[1]) * frac,
            "theta": p0[2] + dtheta * frac,
            "v": math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / decision_dt,
            "lap": 0,
            "progress": 0.0,
            "last_lap_time": replay["lap_time"],
            "off_track": False,
            "ghost": True,
            "label": "hardware lap",
        }

    # -------------------------------------------------------------- simulation

    def tick(self) -> None:
        """Advance one 60 Hz substep; append outgoing messages to the outbox."""
        self._substep += 1
        self.t = self._substep * self.dt
        if self.hw_job is not None:
            self._tick_hardware()
        if self.mode == "train":
            self._tick_train()
        elif self.mode == "hardware" and self._hw_replay is None:
            pass  # idle car waits at the start line until a replay begins
        else:
            if (self._substep - 1) % self.substeps_per_decision == 0:
                self._decide()
            self._step_cars()
        if self._substep % self.broadcast_every == 0:
            self._emit_state()

    def _car_obs(self, car: _Car) -> np.ndarray:
        """(1, F) observation matching RacingEnv._obs for a single car."""
        x, y, theta, v = car.state
        origins = np.tile(car.state[:2], (len(self.ray_angles), 1))
        dist = self.track.raycast(origins, theta + self.ray_angles, self.ray_max_dist)
        rays = np.clip(dist / self.ray_max_dist, 0.0, 1.0)
        speed = np.clip(v / self.car_physics.v_max, 0.0, 1.0)
        car.rays = [float(r) for r in rays]
        return np.concatenate([rays, [speed]])[None, :]

    def _decide(self) -> None:
        """10 Hz agent decisions: greedy action per agent car, quantum introspection."""
        for car in self.cars:
            if car.respawn_at is not None:
                continue
            self._record_traj(car)
            if car.controller is not None:
                car.controls = car.controller(car.state)
                continue
            if car.qfunc is None:
                continue
            obs = self._car_obs(car)
            q = car.qfunc.q_values(obs)[0]
            car.action = int(np.argmax(q))
            car.controls = tuple(float(c) for c in ACTIONS[car.action])
            if car.kind == "quantum" and hasattr(car.qfunc, "expectations"):
                last = self._last_quantum_emit.get(car.id, -math.inf)
                if self.t - last >= QUANTUM_MSG_MIN_INTERVAL_S:
                    self._last_quantum_emit[car.id] = self.t
                    # the gauges show the WHOLE register (the first 4 qubits
                    # are the action readout the output head consumes)
                    if hasattr(car.qfunc, "all_expectations"):
                        expectations = car.qfunc.all_expectations(obs)[0]
                    else:
                        expectations = car.qfunc.expectations(obs)[0]
                    self._outbox.append({
                        "type": "quantum",
                        "car_id": car.id,
                        "expectations": [float(e) for e in expectations],
                        "q_values": [float(v) for v in q],
                        "action": car.action,
                    })

    def _step_cars(self) -> None:
        """One physics substep for every unfrozen car, then progress/lap/crash logic."""
        for car in self.cars:  # thaw crashed humans whose freeze expired
            if car.respawn_at is not None and self.t >= car.respawn_at:
                self._respawn(car)
        active = [car for car in self.cars if car.respawn_at is None]
        if not active:
            return

        for car in active:
            if car.kind == "human":
                car.controls = (self._analog if self._analog is not None
                                else keys_to_controls(self._keys))
        states = np.stack([car.state for car in active])
        controls = np.array([car.controls for car in active])
        new_states = self.car_physics.step(states, controls[:, 0], controls[:, 1], controls[:, 2])

        total = self.track.total_length
        s_new, lateral = self.track.project(new_states[:, :2])
        for i, car in enumerate(active):
            car.state = new_states[i]
            delta = (s_new[i] - car.s + 0.5 * total) % total - 0.5 * total
            car.s = float(s_new[i])
            car.progress += float(delta)

            laps_now = int(math.floor(car.progress / total))
            if laps_now > car.lap:
                lap_time = self.t - car.lap_start_t
                car.last_lap_time = lap_time
                car.lap = laps_now
                car.lap_start_t = self.t
                self._event("lap", car_id=car.id, lap_time=lap_time)
                if not car.lap_dirty:
                    self._event("clean_lap", car_id=car.id, lap_time=lap_time)
                    self._maybe_record_ghost(car, lap_time)
                    self._record_leaderboard(car, lap_time)
                car.lap_dirty = False
                car.traj.clear()
                car.traj_full = True

            car.off_track = bool(abs(lateral[i]) > self.track.half_width)
            if car.off_track:
                car.lap_dirty = True
                self._event("crash", car_id=car.id)
                if car.kind == "human":  # freeze, then respawn (no episode end)
                    car.respawn_at = self.t + HUMAN_RESPAWN_DELAY_S
                else:
                    self._respawn(car)

    def _respawn(self, car: _Car) -> None:
        x, y, theta = self.track.start_pose()
        car.state = np.array([x, y, theta, 0.0])
        s_vals, _ = self.track.project(car.state[None, :2])
        car.s = float(s_vals[0])
        car.progress = 0.0
        car.lap = 0
        car.lap_start_t = self.t
        car.last_lap_time = None
        car.off_track = False
        car.lap_dirty = False
        car.respawn_at = None
        car.action = 0
        car.controls = (0.0, 0.0, 0.0)
        car.rays = None
        car.traj.clear()
        car.traj_full = True

    # ------------------------------------------------------------- ghost laps

    def _record_traj(self, car: _Car) -> None:
        """Append the car's pose to its per-lap trajectory (decision rate)."""
        if len(car.traj) >= GHOST_TRAJ_MAX_POINTS:  # runaway lap: stop recording it
            car.traj.clear()
            car.traj_full = False
            return
        car.traj.append((float(car.state[0]), float(car.state[1]), float(car.state[2])))

    def _driver_description(self, car: _Car) -> str:
        """Honest one-liner of what drove a lap (stored in ghost records)."""
        if car.kind == "human":
            return "human"
        if car.kind == "hero":
            return "racing line (model-based)"
        if car.kind == "pro":
            return "pro (classical DQN, big MLP)"
        if car.label and car.label.startswith("driver: "):
            return car.label[len("driver: "):]
        if car.kind == "quantum":
            return f"{self.track_name}-trained"
        return f"{car.kind} ({self.track_name}-trained)"

    # ------------------------------------------------------------ leaderboard

    def _reload_board(self) -> None:
        """Track changed: bundled tracks load their stored board; ephemeral
        (random/drawn) tracks start a fresh session-lifetime board."""
        if self.track_is_random:
            self._board = {"entries": [], "references": {}}
        else:
            self._board = load_leaderboard(self.track_name, self._leaderboard_dir)

    def leaderboard_payload(self) -> dict:
        """The board as a wire message: ranked human entries plus the AI
        drivers' reference laps (never ranked)."""
        references = [
            {"kind": kind, "driver": ref["driver"], "lap_s": ref["lap_s"]}
            for kind, ref in sorted(self._board["references"].items(),
                                    key=lambda kv: kv[1]["lap_s"])
        ]
        return {"type": "leaderboard", "track": self.track_name,
                "entries": list(self._board["entries"]), "references": references}

    def _record_leaderboard(self, car: _Car, lap_time: float) -> None:
        """File a clean lap: named human race laps rank; agent laps only
        update their kind's reference row."""
        lap_time = float(lap_time)
        changed = False
        if car.kind == "human":
            if self.mode != "race" or not self.racer_name:
                return  # unnamed (or non-race) human laps are not recorded
            entries = self._board["entries"]
            entries.append({"name": self.racer_name, "lap_s": round(lap_time, 3),
                            "date": time.strftime("%Y-%m-%d")})
            entries.sort(key=lambda e: e["lap_s"])
            del entries[LEADERBOARD_MAX_ENTRIES:]
            changed = any(e["lap_s"] == round(lap_time, 3) and e["name"] == self.racer_name
                          for e in entries)
        else:
            ref = self._board["references"].get(car.kind)
            if ref is None or lap_time < ref["lap_s"]:
                self._board["references"][car.kind] = {
                    "driver": self._driver_description(car),
                    "lap_s": round(lap_time, 3),
                }
                changed = True
        if not changed:
            return
        if not self.track_is_random:
            save_leaderboard(self.track_name, self._board, self._leaderboard_dir)
        self._outbox.append(self.leaderboard_payload())

    def _maybe_record_ghost(self, car: _Car, lap_time: float) -> None:
        """Persist a clean lap as the track's best-lap ghost when it beats the record."""
        if self.track_is_random:  # ephemeral tracks: never write ghosts_dir/random*.json
            return
        if car.kind in ("hero", "pro"):  # reference drivers never set records
            return
        if not car.traj_full or len(car.traj) < 2:
            return
        if self._ghost is not None and lap_time >= self._ghost["lap_time"]:
            return
        x, y, theta = (float(c) for c in car.state[:3])
        ghost = {
            "lap_time": float(lap_time),
            "kind": car.kind,
            "driver": self._driver_description(car),
            "points": [list(p) for p in car.traj] + [[x, y, theta]],
        }
        save_ghost(self.track_name, ghost, self._ghosts_dir)
        self._ghost = ghost
        self._event("new_best_lap", car_id=car.id, lap_time=float(lap_time))

    def _ghost_payload(self) -> dict:
        """Best-lap ghost replay car, looping its 10 Hz trajectory lerped to 60 Hz."""
        ghost = self._ghost
        points = ghost["points"]
        n = len(points)
        phase = self._substep / self.substeps_per_decision  # trajectory index at 60 Hz
        frac = phase - math.floor(phase)
        i0 = int(math.floor(phase)) % n
        p0, p1 = points[i0], points[(i0 + 1) % n]
        dtheta = (p1[2] - p0[2] + math.pi) % (2.0 * math.pi) - math.pi  # shortest arc
        decision_dt = self.dt * self.substeps_per_decision
        return {
            "id": "ghost",
            "kind": ghost["kind"],
            "x": p0[0] + (p1[0] - p0[0]) * frac,
            "y": p0[1] + (p1[1] - p0[1]) * frac,
            "theta": p0[2] + dtheta * frac,
            "v": math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / decision_dt,
            "lap": 0,
            "progress": 0.0,
            "last_lap_time": ghost["lap_time"],
            "off_track": False,
            "ghost": True,
            "label": f"best {ghost['lap_time']:.1f}s"
            + (f" · {ghost['driver']}" if ghost.get("driver") else ""),
        }

    # ------------------------------------------------------------ broadcasting

    def _emit_state(self) -> None:
        if self.mode == "train":
            cars = self._train_car_payloads()
        else:
            cars = [self._car_payload(car) for car in self.cars]
            if self._ghost is not None and self.mode in ("attract", "race"):
                cars.append(self._ghost_payload())
            if self._hw_replay is not None and self.mode == "hardware":
                cars.append(self._hw_replay_payload())
        self._outbox.append({"type": "state", "t": self.t, "mode": self.mode, "cars": cars})

    def _car_payload(self, car: _Car) -> dict:
        x, y, theta, v = (float(c) for c in car.state)
        payload = {
            "id": car.id,
            "kind": car.kind,
            "x": x, "y": y, "theta": theta, "v": v,
            "lap": int(car.lap),
            "progress": float(car.progress),
            "last_lap_time": car.last_lap_time,
            "off_track": bool(car.off_track),
        }
        if car.rays is not None:
            payload["rays"] = car.rays
        if car.label is not None:
            payload["label"] = car.label
        return payload

    def _train_car_payloads(self) -> list[dict]:
        cars: list[dict] = []
        for job in self.jobs.values():
            snap = job.env.state_snapshot()
            for i, row in enumerate(snap["state"]):
                lap_time = float(snap["last_lap_time"][i])
                cars.append({
                    "id": f"{job.agent}-{i}",
                    "kind": job.agent,
                    "x": float(row[0]), "y": float(row[1]),
                    "theta": float(row[2]), "v": float(row[3]),
                    "lap": int(snap["lap"][i]),
                    "progress": float(snap["progress"][i]),
                    "last_lap_time": None if math.isnan(lap_time) else lap_time,
                    "off_track": False,
                })
        return cars

    # ------------------------------------------------------------- async loop

    async def run(self, hub: Any) -> None:
        """60 Hz tick loop with drift correction; broadcasts via ``hub.broadcast``."""
        period = self.dt
        next_t = time.monotonic()
        try:
            while True:
                self.tick()
                for msg in self.drain_outbox():
                    await hub.broadcast(msg)
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -0.5:  # fell far behind: resync instead of spiraling
                    next_t = time.monotonic()
        finally:
            self.shutdown()
