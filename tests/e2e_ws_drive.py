"""End-to-end WebSocket drive of the traQmania demo server.

Run manually against a live server::

    .venv/bin/python -m traqmania --port 8123 &
    .venv/bin/python tests/e2e_ws_drive.py --port 8123

or let the script manage its own server process::

    .venv/bin/python tests/e2e_ws_drive.py --spawn

Exercises the full pinned protocol: welcome payload, attract mode (moving
quantum car + quantum introspection messages), evolution mode (labelled
training-stage cars), set_track, MLP training to completion, warm-started
quantum training with lap telemetry (lap_times / best_lap_s / new_best_lap),
a human-vs-quantum race with keyboard input and reset, and the best-lap
ghost car in attract mode.  Skipped under plain ``pytest`` (it needs a
running server and takes minutes); CI ignores it.
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

    async def recv(self, timeout: float = 10.0) -> dict:
        raw = await asyncio.wait_for(self.ws.recv(), timeout)
        msg = json.loads(raw)
        if msg.get("type") == "error":
            raise AssertionError(f"server error: {msg.get('message')}")
        self.log.append(msg)
        return msg

    async def wait_for(self, predicate, timeout: float = 30.0, desc: str = "message") -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"timed out after {timeout}s waiting for {desc}")
            msg = await self.recv(timeout=remaining)
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


async def drive(port: int) -> None:
    from websockets.asyncio.client import connect

    base = f"http://127.0.0.1:{port}"
    verify_http(base)
    async with connect(f"ws://127.0.0.1:{port}/ws", max_size=None,
                       open_timeout=10, close_timeout=5) as ws:
        c = Client(ws)
        await verify_welcome(c)
        await verify_attract(c)
        await verify_evolution(c)
        await verify_set_track(c, "chicane")
        await verify_train_mlp(c, episodes=10)
        await verify_train_quantum_warm(c, episodes=60)
        await verify_race(c)
        await verify_attract_ghost(c)
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
