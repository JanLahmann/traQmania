"""WebSocket hub: connection registry, broadcast fan-out, and the /ws handler.

One shared :class:`DemoSession` serves every client. Watching is a broadcast,
but CONTROL is exclusive and turn-based: the first client to send a control
message takes the wheel; further clients that try to interact join a FIFO
line. A solo driver keeps the wheel indefinitely (a kiosk never notices any
of this); while someone is waiting, a turn lasts at most ``driver_turn_s``
seconds, and the wheel also frees after ``driver_idle_s`` of inactivity or on
disconnect — a 1 Hz ticker (started by the app lifespan) performs handovers
and pushes countdown updates. Each client gets a ``welcome`` plus its
``control`` status on join; malformed messages get an ``error`` reply on
their own socket without disturbing others.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from traqmania.server.protocol import Hello, ProtocolError, parse_client

DRIVER_IDLE_RELEASE_S = 90.0  # driver inactivity before the wheel frees up
DRIVER_TURN_S = 120.0         # max turn length while someone is waiting


class DriverLock:
    """Exclusive, turn-based control of the shared session.

    ``try_control`` grants the wheel when it is free (or held by the caller,
    refreshing the idle timer); anyone else is appended to the waiting line.
    ``tick`` performs handovers: when the driver's turn expires (only counted
    while the line is non-empty) or the driver goes idle, the wheel passes to
    the next in line (or frees up)."""

    def __init__(self, idle_s: float = DRIVER_IDLE_RELEASE_S,
                 turn_s: float = DRIVER_TURN_S) -> None:
        self.idle_s = float(idle_s)
        self.turn_s = float(turn_s)
        self._driver: WebSocket | None = None
        self._last_control = 0.0
        self._turn_start = 0.0
        self._queue: list[WebSocket] = []

    # ------------------------------------------------------------------ state

    @property
    def locked(self) -> bool:
        return self._driver is not None

    def driving(self, websocket: WebSocket) -> bool:
        return self._driver is websocket

    def queue_pos(self, websocket: WebSocket) -> int | None:
        """1-based position in the waiting line, None when not queued."""
        try:
            return self._queue.index(websocket) + 1
        except ValueError:
            return None

    @property
    def waiting(self) -> int:
        return len(self._queue)

    def turn_ends_in(self, now: float | None = None) -> int | None:
        """Whole seconds until the current turn expires — None when no
        countdown is running (no driver, or nobody waiting)."""
        if self._driver is None or not self._queue:
            return None
        now = time.monotonic() if now is None else now
        return max(0, math.ceil(self._turn_start + self.turn_s - now))

    # ---------------------------------------------------------------- control

    def _grant(self, websocket: WebSocket, now: float) -> None:
        self._driver = websocket
        self._last_control = now
        self._turn_start = now
        if websocket in self._queue:
            self._queue.remove(websocket)

    def try_control(self, websocket: WebSocket, now: float | None = None) -> bool:
        """True when ``websocket`` may control the session right now.

        The current driver keeps (and refreshes) the wheel. A free wheel goes
        to the caller. Otherwise the caller joins the waiting line (once) —
        handovers happen in ``tick``, in line order, not to whoever asks."""
        now = time.monotonic() if now is None else now
        if self._driver is None or self._driver is websocket:
            if self._driver is None:
                self._grant(websocket, now)
            self._last_control = now
            return True
        if not self._queue and now - self._last_control > self.idle_s:
            self._grant(websocket, now)  # idle wheel, nobody in line: take it
            return True
        if websocket not in self._queue:
            self._queue.append(websocket)
        return False

    def tick(self, now: float | None = None) -> bool:
        """Advance turn/idle expiry; True when the driver changed."""
        now = time.monotonic() if now is None else now
        if self._driver is None:
            if not self._queue:
                return False
            self._grant(self._queue[0], now)
            return True
        turn_over = self._queue and now - self._turn_start > self.turn_s
        idled = now - self._last_control > self.idle_s
        if not (turn_over or idled):
            return False
        self._driver = None
        if self._queue:
            self._grant(self._queue[0], now)
        return True

    def remove(self, websocket: WebSocket, now: float | None = None) -> bool:
        """Forget a departed client; True when the driver changed (the wheel
        was released or handed to the next in line)."""
        if websocket in self._queue:
            self._queue.remove(websocket)
        if self._driver is not websocket:
            return False
        now = time.monotonic() if now is None else now
        self._driver = None
        if self._queue:
            self._grant(self._queue[0], now)
        return True

    # kept for callers/tests that speak in terms of plain release
    def release(self, websocket: WebSocket) -> bool:
        if self._driver is websocket:
            self._driver = None
            return True
        return False


class Hub:
    """Set of live websocket connections with best-effort broadcast."""

    def __init__(self, driver_idle_s: float = DRIVER_IDLE_RELEASE_S,
                 driver_turn_s: float = DRIVER_TURN_S) -> None:
        self._clients: set[WebSocket] = set()
        self.lock = DriverLock(driver_idle_s, driver_turn_s)
        self._control_sent: dict[WebSocket, tuple] = {}  # dedup per socket

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)
        self._control_sent.pop(websocket, None)

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    def control_payload(self, websocket: WebSocket) -> dict:
        """The ``control`` status message as seen by ``websocket``."""
        return {
            "type": "control",
            "driving": self.lock.driving(websocket),
            "locked": self.lock.locked,
            "watchers": max(self.n_clients - 1, 0),
            "waiting": self.lock.waiting,
            "queue_pos": self.lock.queue_pos(websocket),
            "turn_ends_in_s": self.lock.turn_ends_in(),
        }

    async def send_control_state(self, websocket: WebSocket) -> None:
        """Push ``websocket`` its control status, skipping no-op repeats (a
        spectator's polled gamepad would otherwise echo one per input)."""
        payload = self.control_payload(websocket)
        key = tuple(payload[k] for k in
                    ("driving", "locked", "watchers", "waiting",
                     "queue_pos", "turn_ends_in_s"))
        if self._control_sent.get(websocket) == key:
            return
        self._control_sent[websocket] = key
        try:
            await websocket.send_json(payload)
        except Exception:  # closed/broken socket: forget it
            self.disconnect(websocket)

    async def send_control_states(self) -> None:
        """Push each client its (personalized) control status."""
        for websocket in list(self._clients):
            await self.send_control_state(websocket)

    async def broadcast(self, payload: dict) -> None:
        """Send ``payload`` to every client; drop clients whose socket errors."""
        dead = []
        for websocket in list(self._clients):
            try:
                await websocket.send_json(payload)
            except Exception:  # closed/broken socket: forget it
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)


async def control_ticker(hub: Hub, session: Any, interval: float = 1.0) -> None:
    """Turn/idle expiry loop: hands the wheel to the next in line and keeps
    everyone's countdowns fresh (deduplicated, so quiet when nothing runs)."""
    while True:
        await asyncio.sleep(interval)
        if hub.lock.tick():
            session.set_input(0)  # new (or no) driver: drop held keys
        await hub.send_control_states()


async def handle_socket(websocket: WebSocket, hub: Hub, session: Any) -> None:
    """Per-connection receive loop: welcome, then parse and route client messages."""
    await hub.connect(websocket)
    try:
        await websocket.send_json(session.welcome_payload())
        await websocket.send_json(session.leaderboard_payload())
        await hub.send_control_states()  # join changes everyone's watcher count
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "invalid JSON"})
                continue
            try:
                msg = parse_client(data)
            except ProtocolError as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue
            if isinstance(msg, Hello):
                await websocket.send_json(session.welcome_payload())
                await hub.send_control_state(websocket)
                continue
            was_driving = hub.lock.driving(websocket)
            if hub.lock.try_control(websocket):
                session.handle_message(msg)
                if not was_driving:  # took the wheel
                    await hub.send_control_states()
            else:
                # spectator: the action is dropped; it joined the line (or
                # already stands in it) — tell everyone whose state changed
                await hub.send_control_states()
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(websocket)
        if hub.lock.remove(websocket):
            session.set_input(0)  # the driver left: stop holding their keys
        await hub.send_control_states()
        if hub.n_clients == 0:
            session.set_input(0)  # nobody holding the keys anymore
