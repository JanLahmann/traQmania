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
from typing import Any

import numpy as np

from traqmania.agents.base import ACTIONS, N_ACTIONS
from traqmania.agents.classical import MLPQFunction
from traqmania.agents.quantum.circuit import circuit_spec
from traqmania.agents.quantum.qdqn import QuantumQFunction
from traqmania.agents.training import DQNTrainer
from traqmania.env.car import CarPhysics
from traqmania.env.racing_env import RacingEnv
from traqmania.server import protocol
from traqmania.server.runtime import (
    available_tracks,
    load_agent,
    load_track,
    resolve_training_cfg,
    track_payload,
)

HUMAN_RESPAWN_DELAY_S = 1.0
QUANTUM_MSG_MIN_INTERVAL_S = 0.099  # <= 10 Hz


def keys_to_controls(keys: int) -> tuple[float, float, float]:
    """input.keys bitmask -> (steer, throttle, brake).

    Bits: 1 throttle, 2 brake, 4 left (steer -1), 8 right (steer +1).
    Left+right cancel; brake overrides throttle.
    """
    steer = 0.0
    if keys & protocol.KEY_LEFT:
        steer -= 1.0
    if keys & protocol.KEY_RIGHT:
        steer += 1.0
    brake = 1.0 if keys & protocol.KEY_BRAKE else 0.0
    throttle = 1.0 if (keys & protocol.KEY_THROTTLE) and not brake else 0.0
    return steer, throttle, brake


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


class DemoSession:
    """Single shared demo session (one kiosk): mode state machine + tick loop."""

    def __init__(self, config: dict):
        self.config = config
        physics = config["physics"]
        self.dt = float(physics["dt"])
        self.substeps_per_decision = int(physics["substeps_per_decision"])
        self.car_physics = CarPhysics(physics)

        obs_cfg = config["observation"]
        self.ray_angles = np.deg2rad(np.asarray(obs_cfg["ray_angles_deg"], dtype=np.float64))
        self.ray_max_dist = float(obs_cfg["ray_max_dist"])

        server_cfg = config["server"]
        hz = 1.0 / self.dt
        self.broadcast_every = max(1, round(hz / float(server_cfg["broadcast_hz"])))
        self.telemetry_every = max(1, round(hz / float(server_cfg["telemetry_hz"])))

        self.t = 0.0
        self._substep = 0
        self._outbox: list[dict] = []
        self._keys = 0
        self._agent_cache: dict[tuple[str, str], Any] = {}
        self._last_quantum_emit: dict[str, float] = {}

        self.mode = "attract"
        self.track_name = str(config["track"]["default"])
        self.track = load_track(config, self.track_name)
        self.cars: list[_Car] = []
        self.jobs: dict[str, TrainingJob] = {}
        self._enter_attract()

    # -------------------------------------------------------------- messaging

    def welcome_payload(self) -> dict:
        return {
            "type": "welcome",
            "mode": self.mode,
            "track": track_payload(self.track),
            "tracks": available_tracks(),
            "circuit_spec": circuit_spec(self.config),
            "ui": dict(self.config["ui"]),
        }

    def drain_outbox(self) -> list[dict]:
        out = self._outbox
        self._outbox = []
        return out

    def _error(self, message: str) -> None:
        self._outbox.append({"type": "error", "message": message})

    def _event(self, kind: str, **fields: Any) -> None:
        self._outbox.append({"type": "event", "kind": kind, **fields})

    def set_input(self, keys: int) -> None:
        """Last-writer-wins human input bitmask."""
        self._keys = int(keys)

    def handle_message(self, msg: Any) -> None:
        """Apply a parsed client message to the session state."""
        if isinstance(msg, protocol.Hello):
            return  # welcome is sent per-connection by the ws layer
        if isinstance(msg, protocol.Input):
            self.set_input(msg.keys)
        elif isinstance(msg, protocol.SetMode):
            self._set_mode(msg.mode)
        elif isinstance(msg, protocol.SetTrack):
            self._set_track(msg.track)
        elif isinstance(msg, protocol.Train):
            self._handle_train(msg)
        elif isinstance(msg, protocol.Race):
            self._handle_race(msg)
        else:
            self._error(f"unhandled message: {msg!r}")

    # ------------------------------------------------------------ mode switches

    def _training_alive(self) -> bool:
        return any(j.thread is not None and j.thread.is_alive() for j in self.jobs.values())

    def _set_mode(self, mode: str) -> None:
        if mode != "train":
            self.stop_training()
        self.mode = mode
        if mode == "attract":
            self._enter_attract()
        elif mode == "race":
            self._enter_race("quantum")
        else:  # train: cars appear once a train{start} arrives
            self.cars = []

    def _set_track(self, name: str) -> bool:
        if self._training_alive():
            self._error("cannot change track while training is running")
            return False
        if name not in available_tracks():
            self._error(f"unknown track '{name}'")
            return False
        self.track = load_track(self.config, name)
        self.track_name = name
        self._outbox.append({"type": "track", "track": track_payload(self.track)})
        if self.mode == "attract":
            self._enter_attract()
        elif self.mode == "race":
            opponent = next((c.kind for c in self.cars if c.kind != "human"), "quantum")
            self._enter_race(opponent)
        return True

    def _enter_attract(self) -> None:
        self.cars = [self._make_agent_car("quantum")]

    def _enter_race(self, opponent: str) -> None:
        x, y, theta = self.track.start_pose()
        human = _Car(id="human", kind="human", state=np.array([x, y, theta, 0.0]))
        self._respawn(human)
        self.cars = [human, self._make_agent_car(opponent)]

    def _make_agent_car(self, kind: str) -> _Car:
        key = (kind, self.track_name)
        if key not in self._agent_cache:
            self._agent_cache[key] = load_agent(kind, self.track_name, config=self.config)
        x, y, theta = self.track.start_pose()
        car = _Car(id=kind, kind=kind, state=np.array([x, y, theta, 0.0]),
                   qfunc=self._agent_cache[key])
        self._respawn(car)
        return car

    def _handle_race(self, msg: protocol.Race) -> None:
        if msg.track is not None and not self._set_track(msg.track):
            return
        if msg.action == "start":
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
        tcfg = resolve_training_cfg(self.config, self.track_name, warm)
        episodes = int(episodes) if episodes is not None else int(tcfg["episodes"])
        seed = int(tcfg.get("seed", 0)) + seed_offset

        env = RacingEnv(self.track, self.config, n_envs=int(tcfg["n_parallel_envs"]), seed=seed)
        if warm and agent == "quantum":
            qfunc: Any = load_agent("quantum", self.track_name, warm=True, config=self.config)
        elif agent == "quantum":
            qfunc = QuantumQFunction(self.config["circuit"], seed=seed)
        else:
            qfunc = MLPQFunction(n_features=env.n_features, n_actions=N_ACTIONS, seed=seed)

        track, config = self.track, self.config  # bind now: self.track may change later

        def env_factory(track=track, config=config, seed=seed) -> RacingEnv:
            return RacingEnv(track, config, n_envs=4, seed=seed + 10_000)

        stop_event = threading.Event()
        trainer = DQNTrainer(qfunc, env, tcfg, rng=np.random.default_rng(seed),
                             env_factory=env_factory, stop_event=stop_event)
        job = TrainingJob(agent=agent, env=env, trainer=trainer, stop_event=stop_event,
                          epsilon=float(tcfg["epsilon_start"]))

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
        """Stop training threads and wait briefly for them to exit."""
        self.stop_training()
        for job in self.jobs.values():
            if job.thread is not None:
                job.thread.join(timeout=10.0)

    def _emit_telemetry(self, job: TrainingJob, final: bool = False) -> None:
        with job.lock:
            tail = [float(r) for r in job.returns[-100:]]
            episode, epsilon, loss = job.episode, job.epsilon, job.loss
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
        })

    def _tick_train(self) -> None:
        if self._substep % self.telemetry_every == 0:
            for job in self.jobs.values():
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

    # -------------------------------------------------------------- simulation

    def tick(self) -> None:
        """Advance one 60 Hz substep; append outgoing messages to the outbox."""
        self._substep += 1
        self.t = self._substep * self.dt
        if self.mode == "train":
            self._tick_train()
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
            if car.qfunc is None or car.respawn_at is not None:
                continue
            obs = self._car_obs(car)
            q = car.qfunc.q_values(obs)[0]
            car.action = int(np.argmax(q))
            car.controls = tuple(float(c) for c in ACTIONS[car.action])
            if car.kind == "quantum" and hasattr(car.qfunc, "expectations"):
                last = self._last_quantum_emit.get(car.id, -math.inf)
                if self.t - last >= QUANTUM_MSG_MIN_INTERVAL_S:
                    self._last_quantum_emit[car.id] = self.t
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
                car.controls = keys_to_controls(self._keys)
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
                car.lap_dirty = False

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

    # ------------------------------------------------------------ broadcasting

    def _emit_state(self) -> None:
        if self.mode == "train":
            cars = self._train_car_payloads()
        else:
            cars = [self._car_payload(car) for car in self.cars]
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
