"""End-to-end WebSocket drive of the traQmania demo server.

Run manually against a live server::

    .venv/bin/python -m traqmania --port 8123 &
    .venv/bin/python tests/e2e_ws_drive.py --port 8123

or let the script manage its own server process::

    .venv/bin/python tests/e2e_ws_drive.py --spawn

Exercises the full pinned protocol: welcome payload, attract mode (moving
quantum car + quantum introspection messages), the live qubit switch
(4 -> 6 drives, 8 degrades car-less, back to 4), evolution mode (labelled
training-stage cars), set_track, MLP training to completion, warm-started
quantum training with lap telemetry (lap_times / best_lap_s / new_best_lap),
a human-vs-quantum race with keyboard input, analog (gamepad-style) input and
reset, hardware mode (fake-backend lap with replay ghost + SPSA sprint),
the best-lap ghost car in attract mode, and a seeded random track (generated
payload, honest fallback driver, reproducibility, no ghost persistence).
Skipped under plain ``pytest`` (it needs a running server and takes minutes);
CI ignores it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

DEFAULT_PORT = 8123
BROADCAST_HZ = 20.0  # matches [server].broadcast_hz in default.toml
GHOSTS_DIR = Path(__file__).resolve().parent.parent / "traqmania" / "data" / "ghosts"


@pytest.mark.skip(reason="e2e: needs a live server; run `python tests/e2e_ws_drive.py`")
def test_e2e_ws_drive() -> None:  # pragma: no cover - manual e2e entry point
    raise AssertionError("run this module directly, not under pytest")


class Client:
    """Thin JSON-over-websocket client with predicate-based waiting."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.log: list[dict] = []

    async def send(self, **msg) -> None:
        await self.ws.send(json.dumps(msg))

    async def recv(self, timeout: float = 10.0, allow_errors: bool = False) -> dict:
        raw = await asyncio.wait_for(self.ws.recv(), timeout)
        msg = json.loads(raw)
        if msg.get("type") == "error" and not allow_errors:
            raise AssertionError(f"server error: {msg.get('message')}")
        self.log.append(msg)
        return msg

    async def wait_for(self, predicate, timeout: float = 30.0, desc: str = "message",
                       allow_errors: bool = False) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"timed out after {timeout}s waiting for {desc}")
            msg = await self.recv(timeout=remaining, allow_errors=allow_errors)
            if predicate(msg):
                return msg

    async def collect(self, duration: float) -> list[dict]:
        """Drain every message that arrives within ``duration`` seconds."""
        out: list[dict] = []
        deadline = time.monotonic() + duration
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return out
            try:
                out.append(await self.recv(timeout=remaining))
            except TimeoutError:
                return out


def check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(f"FAIL: {what}")
    print(f"  ok: {what}")


def car_by_kind(state: dict, kind: str) -> dict | None:
    """First live (non-ghost) car of ``kind`` in a state message."""
    return next(
        (c for c in state["cars"] if c["kind"] == kind and not c.get("ghost")), None)


def ghost_car(state: dict) -> dict | None:
    return next((c for c in state["cars"] if c.get("ghost") is True), None)


# ------------------------------------------------------------------ scenario


def verify_http(base: str) -> None:
    print("[http] health / index / js")
    with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
        check(json.loads(r.read())["status"] == "ok", "GET /health -> status ok")
    with urllib.request.urlopen(f"{base}/", timeout=5) as r:
        body = r.read().decode()
        check("<canvas" in body or "<!doctype" in body.lower(), "GET / serves index.html")
    with urllib.request.urlopen(f"{base}/js/main.js", timeout=5) as r:
        check(len(r.read()) > 100, "GET /js/main.js serves the module")


async def verify_welcome(c: Client) -> None:
    print("[ws] welcome")
    msg = await c.recv(timeout=10)
    check(msg["type"] == "welcome", "first message is welcome")
    track = msg["track"]
    for key in ("name", "half_width", "total_length", "checkpoints", "theme",
                "start", "centerline", "left", "right"):
        check(key in track, f"welcome.track has '{key}'")
    check(len(track["left"]) == len(track["right"]), "left/right rings same length")
    check(len(track["left"]) > 10, f"boundary rings non-trivial ({len(track['left'])} pts)")
    spec = msg["circuit_spec"]
    check(bool(spec.get("gates")), f"circuit_spec has gates ({len(spec['gates'])})")
    check(isinstance(msg["tracks"], list) and len(msg["tracks"]) >= 3, "welcome lists tracks")
    check(isinstance(msg["ui"], dict), "welcome carries ui config")


async def verify_attract(c: Client) -> None:
    print("[ws] attract mode: moving quantum car + quantum messages")
    await c.send(type="set_mode", mode="attract")
    duration = 3.0
    msgs = await c.collect(duration)
    states = [m for m in msgs if m["type"] == "state" and m["mode"] == "attract"]
    quantum = [m for m in msgs if m["type"] == "quantum"]
    min_states = BROADCAST_HZ * 0.5 * duration
    check(len(states) >= min_states,
          f"state rate: {len(states)} msgs in {duration}s (need >= {min_states:.0f})")
    positions = []
    for s in states:
        car = car_by_kind(s, "quantum")
        check(car is not None, "attract state has a quantum car") if s is states[0] else None
        if car is not None:
            positions.append((car["x"], car["y"]))
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
    check(spread > 1.0, f"quantum car moves (x+y spread {spread:.2f})")
    check(len(quantum) >= 2, f"quantum introspection messages arrive ({len(quantum)})")
    for q in quantum:
        check(len(q["expectations"]) == 4, "quantum.expectations has 4 entries") \
            if q is quantum[0] else None
        assert len(q["expectations"]) == 4
        assert all(-1.0 <= e <= 1.0 for e in q["expectations"]), q["expectations"]
        assert len(q["q_values"]) == 4 and 0 <= q["action"] < 4
    print(f"  ok: all {len(quantum)} quantum msgs have 4 expectations in [-1, 1]")


async def _attract_quantum_cars(c: Client, duration: float = 3.0) -> list[dict]:
    """Live (non-ghost) quantum cars from ``duration`` seconds of attract states."""
    states = [m for m in await c.collect(duration)
              if m["type"] == "state" and m["mode"] == "attract"]
    check(len(states) >= BROADCAST_HZ * 0.5 * duration,
          f"attract states streaming ({len(states)} in {duration}s)")
    return [q for q in (car_by_kind(s, "quantum") for s in states) if q is not None]


def _check_drives(cars: list[dict], n_rays: int, what: str) -> None:
    check(bool(cars), f"{what}: quantum car present in attract states")
    spread = (max(q["x"] for q in cars) - min(q["x"] for q in cars)) \
        + (max(q["y"] for q in cars) - min(q["y"] for q in cars))
    check(spread > 1.0, f"{what}: quantum car drives (x+y spread {spread:.2f})")
    rays = next(q["rays"] for q in reversed(cars) if q.get("rays") is not None)
    check(len(rays) == n_rays, f"{what}: state rays carry {n_rays} entries ({len(rays)})")


async def verify_qubit_switch(c: Client) -> None:
    print("[ws] live qubit switch: 4 -> 6 (drives) -> 8 (untrained, degraded) -> 4")
    await c.send(type="qubits", n=6)
    welcome = await c.wait_for(lambda m: m["type"] == "welcome",
                               timeout=60, desc="welcome after qubits=6")
    spec = welcome["circuit_spec"]
    check(spec["n_qubits"] == 6, "q6 welcome.circuit_spec.n_qubits == 6")
    check(spec["n_params"]["total"] == 80,
          f"q6 circuit has 80 params ({spec['n_params']['total']})")
    labels = welcome.get("obs_labels")
    check(isinstance(labels, list) and len(labels) == 6 and labels[-1] == "speed",
          f"q6 welcome.obs_labels lists 6 features ending in speed ({labels})")
    check(welcome["mode"] == "attract", "qubit switch resets to attract mode")
    _check_drives(await _attract_quantum_cars(c), n_rays=5, what="q6")

    # 8 qubits has no bundled oval weights: switch succeeds but degrades to a
    # car-less attract mode, with the existing weight-missing error alongside.
    await c.send(type="qubits", n=8)
    errors: list[dict] = []

    def until_welcome(m: dict) -> bool:
        if m["type"] == "error":
            errors.append(m)
        return m["type"] == "welcome"

    welcome = await c.wait_for(until_welcome, timeout=60, allow_errors=True,
                               desc="welcome after qubits=8")
    check(welcome["circuit_spec"]["n_qubits"] == 8, "q8 welcome.circuit_spec.n_qubits == 8")
    check(len(welcome.get("obs_labels") or []) == 8, "q8 welcome.obs_labels lists 8 features")
    check(any("_q8.npz" in e.get("message", "") for e in errors),
          f"missing q8 weights reported via the existing error path ({len(errors)} error(s))")
    states = [m for m in await c.collect(1.5) if m["type"] == "state"]
    check(len(states) >= 5, f"degraded q8 attract keeps broadcasting ({len(states)} states)")
    check(all(car_by_kind(s, "quantum") is None for s in states),
          "no live quantum car at the untrained qubit count")

    await c.send(type="qubits", n=4)
    welcome = await c.wait_for(lambda m: m["type"] == "welcome",
                               timeout=60, desc="welcome after qubits=4")
    spec = welcome["circuit_spec"]
    check(spec["n_qubits"] == 4, "back to 4: welcome.circuit_spec.n_qubits == 4")
    check(spec["n_params"]["total"] == 56,
          f"back to 4: circuit has 56 params ({spec['n_params']['total']})")
    check(welcome.get("obs_labels") == ["ray -60°", "ray 0°", "ray +60°", "speed"],
          f"back to 4: default obs_labels restored ({welcome.get('obs_labels')})")
    _check_drives(await _attract_quantum_cars(c), n_rays=3, what="back to 4")
    print("  ok: live qubit switch 4 -> 6 -> 8 -> 4 complete")


async def verify_evolution(c: Client) -> None:
    print("[ws] evolution mode: >=3 labelled training-stage cars")
    await c.send(type="set_mode", mode="evolution")
    first = await c.wait_for(
        lambda m: m["type"] == "state" and m["mode"] == "evolution",
        timeout=30, desc="evolution state")
    labelled = [car for car in first["cars"]
                if isinstance(car.get("label"), str) and not car.get("ghost")]
    check(len(labelled) >= 3,
          f"evolution state has >=3 labelled cars ({len(labelled)}: "
          f"{[car['label'] for car in labelled]})")
    check(all(car["kind"] == "quantum" for car in labelled), "stage cars are quantum")
    duration = 2.0
    states = [m for m in await c.collect(duration)
              if m["type"] == "state" and m["mode"] == "evolution"]
    check(len(states) >= BROADCAST_HZ * 0.5 * duration,
          f"evolution states streaming ({len(states)} in {duration}s)")
    xs = [car["x"] for s in states for car in s["cars"] if not car.get("ghost")]
    ys = [car["y"] for s in states for car in s["cars"] if not car.get("ghost")]
    spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
    check(spread > 1.0, f"evolution cars move (x+y spread {spread:.2f})")


async def verify_set_track(c: Client, name: str) -> None:
    print(f"[ws] set_track {name}")
    await c.send(type="set_track", track=name)
    msg = await c.wait_for(lambda m: m["type"] == "track", timeout=30, desc="track msg")
    check(msg["track"]["name"] == name, f"track msg carries '{name}' payload")
    check(len(msg["track"]["centerline"]) > 10, "track payload has centerline")


async def verify_train_mlp(c: Client, episodes: int = 10) -> None:
    print(f"[ws] train mlp for {episodes} episodes to completion")
    await c.send(type="train", action="start", agent="mlp", episodes=episodes)
    telemetry: list[dict] = []

    def keep(m: dict) -> bool:
        if m["type"] == "telemetry" and m["agent"] == "mlp":
            telemetry.append(m)
        return m["type"] == "event" and m["kind"] == "training_done" and m.get("agent") == "mlp"

    await c.wait_for(keep, timeout=300, desc="mlp training_done event")
    check(len(telemetry) >= 1, f"mlp telemetry arrived ({len(telemetry)} msgs)")
    last = telemetry[-1]
    check(last["episode"] >= episodes - 1, f"telemetry reached episode {last['episode']}")
    check(isinstance(last["mean_return"], (int, float)), "telemetry mean_return is numeric")
    check(0.0 <= last["epsilon"] <= 1.0, f"epsilon in [0,1] ({last['epsilon']:.3f})")
    check(isinstance(last["returns_tail"], list) and len(last["returns_tail"]) <= 100,
          f"returns_tail bounded ({len(last['returns_tail'])} entries)")
    print("  ok: training_done received")


async def verify_train_quantum_warm(c: Client, episodes: int = 60) -> None:
    print(f"[ws] warm quantum training on oval ({episodes} episodes, stopped early): "
          "lap telemetry + new_best_lap")
    await c.send(type="train", action="start", agent="quantum",
                 track="oval", warm=True, episodes=episodes)
    msg = await c.wait_for(lambda m: m["type"] == "track", timeout=30, desc="track msg (oval)")
    check(msg["track"]["name"] == "oval", "train start switched track to oval")
    telemetry: list[dict] = []
    best_events: list[dict] = []

    def got_laps(m: dict) -> bool:
        if m["type"] == "telemetry" and m["agent"] == "quantum":
            telemetry.append(m)
        if m["type"] == "event" and m["kind"] == "new_best_lap" and m.get("agent") == "quantum":
            best_events.append(m)
        if not telemetry or not best_events:
            return False
        last = telemetry[-1]
        return bool(last.get("lap_times")) and last.get("best_lap_s") is not None

    await c.wait_for(got_laps, timeout=420,
                     desc="quantum telemetry with lap_times/best_lap_s + new_best_lap event")
    last = telemetry[-1]
    check(telemetry[-1]["episode"] >= 1, f"quantum trained episodes ({telemetry[-1]['episode']})")
    check(all(isinstance(t["returns_tail"], list) for t in telemetry), "returns_tail present")
    laps = last["lap_times"]
    check(isinstance(laps, list) and len(laps) <= 50, f"lap_times bounded ({len(laps)} entries)")
    check(all(isinstance(p, list) and len(p) == 2
              and isinstance(p[0], int) and isinstance(p[1], (int, float)) and p[1] > 0
              for p in laps), "lap_times entries are [episode:int, lap_s:float>0] pairs")
    best = last["best_lap_s"]
    check(isinstance(best, (int, float)) and 0 < best <= min(t for _, t in laps) + 1e-9,
          f"best_lap_s ({best:.2f}s) <= every recent lap time")
    check(all(e["lap_time"] > 0 for e in best_events),
          f"new_best_lap event(s) carry positive lap_time ({len(best_events)} seen)")
    print(f"  ok: {len(laps)} lap_times, best_lap_s {best:.2f}s, "
          f"{len(best_events)} new_best_lap event(s)")
    await c.send(type="train", action="stop", agent="quantum")
    await c.wait_for(
        lambda m: m["type"] == "event" and m["kind"] == "training_done"
        and m.get("agent") == "quantum",
        timeout=300, desc="quantum training_done after stop")
    print("  ok: warm quantum training ran and stopped cleanly")


async def verify_race(c: Client) -> None:
    print("[ws] race vs quantum: human input moves the car, then reset")
    await c.send(type="race", action="start", opponent="quantum")
    first = await c.wait_for(
        lambda m: m["type"] == "state" and m["mode"] == "race"
        and car_by_kind(m, "human") is not None,
        timeout=30, desc="race state with human car")
    human0 = car_by_kind(first, "human")
    await c.send(type="input", keys=1 | 4)  # throttle + left
    msgs = await c.collect(2.0)
    states = [m for m in msgs if m["type"] == "state" and m["mode"] == "race"]
    check(len(states) >= 10, f"race states streaming ({len(states)} in 2s)")
    humans = [car_by_kind(s, "human") for s in states]
    humans = [h for h in humans if h is not None]
    moved = max(abs(h["x"] - human0["x"]) + abs(h["y"] - human0["y"]) for h in humans)
    check(moved > 0.5, f"human car moved under throttle+left input ({moved:.2f} units)")
    check(any(h["v"] > 0.1 for h in humans), "human car gained speed")
    check(all(car_by_kind(s, "quantum") is not None for s in states),
          "quantum opponent present in every race state")

    await c.send(type="input", keys=0)
    await c.send(type="race", action="reset", opponent="quantum")
    # drain for a second so buffered pre-reset states flush; judge the last one
    post = [m for m in await c.collect(1.0) if m["type"] == "state" and m["mode"] == "race"]
    check(len(post) >= 5, f"states keep streaming after reset ({len(post)})")
    human = car_by_kind(post[-1], "human")
    check(human is not None and human["lap"] == 0 and abs(human["progress"]) < 5.0,
          f"race reset put human back at start (progress {human['progress']:.2f})")
    print("  ok: race flow complete")


async def verify_race_analog(c: Client) -> None:
    print("[ws] race vs quantum: analog input (steer 1.0, throttle 1.0, keys 0) moves the car")
    await c.send(type="race", action="start", opponent="quantum")
    first = await c.wait_for(
        lambda m: m["type"] == "state" and m["mode"] == "race"
        and car_by_kind(m, "human") is not None,
        timeout=30, desc="race state with human car")
    human0 = car_by_kind(first, "human")
    await c.send(type="input", keys=0, steer=1.0, throttle=1.0)
    states = [m for m in await c.collect(2.0)
              if m["type"] == "state" and m["mode"] == "race"]
    check(len(states) >= 10, f"race states streaming ({len(states)} in 2s)")
    humans = [h for h in (car_by_kind(s, "human") for s in states) if h is not None]
    moved = max(abs(h["x"] - human0["x"]) + abs(h["y"] - human0["y"]) for h in humans)
    check(moved > 0.5, f"human car moved under analog steer+throttle ({moved:.2f} units)")
    check(any(h["v"] > 0.1 for h in humans), "human car gained speed from analog throttle")
    turned = max(abs(h["theta"] - human0["theta"]) for h in humans)
    check(turned > 0.05, f"human car heading changed under analog steer ({turned:.2f} rad)")
    await c.send(type="input", keys=0)  # release: back to (all-zero) keyboard controls
    print("  ok: analog input drives the human car")


class HardwareWatch:
    """Accumulates hardware_status messages while waiting for a target phase."""

    def __init__(self) -> None:
        self.statuses: list[dict] = []

    def until(self, *phases: str):
        def predicate(m: dict) -> bool:
            if m["type"] != "hardware_status":
                return False
            self.statuses.append(m)
            if m["phase"] == "error" and "error" not in phases:
                raise AssertionError(f"hardware_status error: {m.get('message')}")
            return m["phase"] in phases
        return predicate

    def phases(self) -> list[str]:
        return [s["phase"] for s in self.statuses]


async def verify_hardware_lap(c: Client) -> None:
    print("[ws] hardware mode: fake-backend lap (25 decisions) with replay ghost")
    await c.send(type="set_mode", mode="hardware")
    idle = await c.wait_for(
        lambda m: m["type"] == "hardware_status" and m["phase"] == "idle",
        timeout=30, desc="hardware_status idle after set_mode hardware")
    check(idle["phase"] == "idle", "entering hardware mode reports phase idle")
    state = await c.wait_for(
        lambda m: m["type"] == "state" and m["mode"] == "hardware",
        timeout=30, desc="hardware-mode state")
    check(car_by_kind(state, "quantum") is not None, "idle fastsim quantum car present")

    await c.send(type="hardware", action="lap", backend="fake", shots=128, max_decisions=25)
    watch = HardwareWatch()
    # generous timeout: the first fake-backend transpile alone can take minutes
    await c.wait_for(watch.until("done"), timeout=600,
                     desc="hardware lap phases reaching done")
    phases = watch.phases()
    for expected in ("connecting", "transpiling", "running", "done"):
        check(expected in phases, f"hardware lap reached phase '{expected}'")
    order = [phases.index(p) for p in ("connecting", "transpiling", "running", "done")]
    check(order == sorted(order), f"lap phases arrive in order ({phases})")
    running = [s for s in watch.statuses if s["phase"] == "running"]
    check(running[0].get("decision", 0) >= 1, "running status carries decision counter")
    check(running[0].get("seconds_per_decision", 0) > 0, "running status carries s/decision")
    check(bool(running[0].get("backend_name")), "running status carries backend_name")

    replay = await c.wait_for(
        lambda m: m["type"] == "state" and m["mode"] == "hardware"
        and any(car["id"] == "hardware" for car in m["cars"]),
        timeout=60, desc="state with hardware replay car")
    hw = next(car for car in replay["cars"] if car["id"] == "hardware")
    check(hw.get("ghost") is True and hw["kind"] == "quantum"
          and hw.get("label") == "hardware lap",
          "replay car is {id:hardware, kind:quantum, ghost:true, label:'hardware lap'}")
    check(car_by_kind(replay, "quantum") is not None,
          "fastsim comparison car drives alongside the hardware ghost")
    print("  ok: hardware lap ran on the fake backend and replays in the race canvas")


async def verify_hardware_sprint(c: Client) -> None:
    print("[ws] hardware mode: SPSA sprint (2 iterations) on the fake backend")
    await c.send(type="hardware", action="sprint", backend="fake",
                 iterations=2, shots=128)
    watch = HardwareWatch()
    await c.wait_for(watch.until("done"), timeout=600,
                     desc="hardware sprint phases reaching done")
    running = [s for s in watch.statuses if s["phase"] == "running"]
    check([s.get("iteration") for s in running] == [1, 2],
          f"sprint reports iterations 1..2 ({[s.get('iteration') for s in running]})")
    check(all("loss" in s for s in running), "sprint running statuses carry loss")
    done = next(s for s in watch.statuses if s["phase"] == "done")
    check(isinstance(done.get("eval_return_before"), (int, float)),
          f"done carries eval_return_before ({done.get('eval_return_before')})")
    check(isinstance(done.get("eval_return_after"), (int, float)),
          f"done carries eval_return_after ({done.get('eval_return_after')})")
    print(f"  ok: sprint done, eval return {done['eval_return_before']:.1f} -> "
          f"{done['eval_return_after']:.1f}")


def _inject_ghost_file(track: dict) -> None:
    """Write a synthetic best-lap ghost for a track from its centerline."""
    pts = track["centerline"]
    points = []
    for i, (x, y) in enumerate(pts):
        nx, ny = pts[(i + 1) % len(pts)]
        points.append([float(x), float(y), math.atan2(ny - y, nx - x)])
    payload = {"track": track["name"], "lap_time": 0.1 * len(points),
               "kind": "quantum", "points": points}
    GHOSTS_DIR.mkdir(parents=True, exist_ok=True)
    (GHOSTS_DIR / f"{track['name']}.json").write_text(json.dumps(payload) + "\n",
                                                      encoding="utf-8")


async def verify_attract_ghost(c: Client) -> None:
    print("[ws] attract on oval: best-lap ghost car in state")
    ghost_file = GHOSTS_DIR / "oval.json"
    await c.send(type="set_track", track="oval")
    track = (await c.wait_for(
        lambda m: m["type"] == "track" and m["track"]["name"] == "oval",
        timeout=30, desc="track msg (oval)"))["track"]
    await c.send(type="set_mode", mode="attract")

    def has_ghost(m: dict) -> bool:
        return m["type"] == "state" and m["mode"] == "attract" and ghost_car(m) is not None

    if not ghost_file.is_file():
        print("  (no stored ghost yet; letting attract lap to record one)")
        try:
            await c.wait_for(
                lambda m: has_ghost(m)
                or (m["type"] == "event" and m["kind"] == "new_best_lap"),
                timeout=120, desc="attract clean lap recording a ghost")
        except AssertionError:
            print("  (attract did not record a ghost in time; injecting a synthetic one)")
            _inject_ghost_file(track)
    check(ghost_file.is_file(), "ghost file traqmania/data/ghosts/oval.json exists")
    # force a reload from disk so the persisted record (not just memory) is verified
    await c.send(type="set_track", track="oval")
    await c.wait_for(lambda m: m["type"] == "track" and m["track"]["name"] == "oval",
                     timeout=30, desc="track msg (oval reload)")
    state = await c.wait_for(has_ghost, timeout=30, desc="attract state with ghost car")
    ghost = ghost_car(state)
    check(ghost["id"] == "ghost" and ghost["kind"] in ("quantum", "mlp"),
          f"ghost car has id 'ghost' and agent kind '{ghost['kind']}'")
    check(isinstance(ghost.get("label"), str) and ghost["label"].startswith("best "),
          f"ghost car labelled with best lap ({ghost.get('label')!r})")
    check(ghost["last_lap_time"] > 0, f"ghost carries lap time ({ghost['last_lap_time']:.2f}s)")
    check(car_by_kind(state, "quantum") is not None, "live quantum car still present")
    print("  ok: ghost replay car streams in attract mode")


async def verify_random_track(c: Client) -> None:
    print("[ws] random track: seeded generation, attract drive, reproducibility")
    seed = 4242
    await c.send(type="set_track", track="random", seed=seed)
    track = (await c.wait_for(lambda m: m["type"] == "track", timeout=30,
                              desc="random track msg"))["track"]
    check(track["name"] == f"random #{seed}", f"track payload named 'random #{seed}'")
    check(len(track["centerline"]) > 10, "random track payload has centerline")
    cars = await _attract_quantum_cars(c)
    _check_drives(cars, n_rays=3, what="random track")
    labels = {q.get("label") for q in cars}
    check(labels == {"driver: gp-trained generalist"},
          f"attract car carries the honest fallback-driver label ({labels})")
    # the same seed reproduces the same track after switching away and back
    await c.send(type="set_track", track="oval")
    await c.wait_for(lambda m: m["type"] == "track" and m["track"]["name"] == "oval",
                     timeout=30, desc="track msg (oval)")
    await c.send(type="set_track", track="random", seed=seed)
    again = (await c.wait_for(
        lambda m: m["type"] == "track" and m["track"]["name"] == f"random #{seed}",
        timeout=30, desc="reproduced random track"))["track"]
    check(again["centerline"] == track["centerline"], "same seed reproduces the centerline")
    check(not list(GHOSTS_DIR.glob("random*.json")),
          "no ghost file was written for the random track")
    await c.send(type="set_track", track="oval")
    msg = await c.wait_for(lambda m: m["type"] == "track" and m["track"]["name"] == "oval",
                           timeout=30, desc="track msg (oval restore)")
    check(len(msg["track"]["centerline"]) > 10, "set_track oval restores the bundled track")
    print("  ok: random track scenario complete")


async def drive(port: int) -> None:
    from websockets.asyncio.client import connect

    base = f"http://127.0.0.1:{port}"
    verify_http(base)
    async with connect(f"ws://127.0.0.1:{port}/ws", max_size=None,
                       open_timeout=10, close_timeout=5) as ws:
        c = Client(ws)
        await verify_welcome(c)
        await verify_attract(c)
        await verify_qubit_switch(c)
        await verify_evolution(c)
        await verify_set_track(c, "chicane")
        await verify_train_mlp(c, episodes=10)
        await verify_train_quantum_warm(c, episodes=60)
        await verify_race(c)
        await verify_race_analog(c)
        await verify_hardware_lap(c)
        await verify_hardware_sprint(c)
        await verify_attract_ghost(c)
        await verify_random_track(c)
        await c.send(type="set_mode", mode="attract")  # leave the kiosk in attract
    print("\nALL E2E CHECKS PASSED")


# --------------------------------------------------------------------- runner


def _snapshot_ghosts() -> dict[str, str] | None:
    """Contents of the bundled ghosts dir, or None when it does not exist."""
    if not GHOSTS_DIR.is_dir():
        return None
    return {p.name: p.read_text(encoding="utf-8") for p in GHOSTS_DIR.glob("*.json")}


def _restore_ghosts(snapshot: dict[str, str] | None) -> None:
    """Put the ghosts dir back exactly as snapshotted (spawned-server runs only)."""
    if not GHOSTS_DIR.is_dir():
        return
    for p in GHOSTS_DIR.glob("*.json"):
        if snapshot is None or p.name not in snapshot:
            p.unlink()
    if snapshot is None:
        with contextlib.suppress(OSError):
            GHOSTS_DIR.rmdir()
        return
    for name, text in snapshot.items():
        (GHOSTS_DIR / name).write_text(text, encoding="utf-8")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"server on port {port} never became healthy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"server port (default {DEFAULT_PORT})")
    parser.add_argument("--spawn", action="store_true",
                        help="start `python -m traqmania` on a free port for the test run")
    args = parser.parse_args(argv)

    proc = None
    ghosts_snapshot: dict[str, str] | None = None
    port = args.port
    if args.spawn:
        ghosts_snapshot = _snapshot_ghosts()  # restored afterwards: keep the repo clean
        port = _free_port()
        proc = subprocess.Popen([sys.executable, "-m", "traqmania", "--port", str(port)])
    try:
        if args.spawn:
            _wait_health(port)
        asyncio.run(drive(port))
        return 0
    except AssertionError as exc:
        print(f"\nE2E FAILURE: {exc}", file=sys.stderr)
        return 1
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            _restore_ghosts(ghosts_snapshot)


if __name__ == "__main__":
    sys.exit(main())
