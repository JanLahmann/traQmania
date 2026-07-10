"""WebSocket hub: connection registry, broadcast fan-out, and the /ws handler.

One shared :class:`DemoSession` serves every client; input is last-writer-wins
(a single human seat).  Each client gets a ``welcome`` on join; malformed
messages get an ``error`` reply on their own socket without disturbing others.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from traqmania.server.protocol import Hello, ProtocolError, parse_client


class Hub:
    """Set of live websocket connections with best-effort broadcast."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    @property
    def n_clients(self) -> int:
        return len(self._clients)

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


async def handle_socket(websocket: WebSocket, hub: Hub, session: Any) -> None:
    """Per-connection receive loop: welcome, then parse and route client messages."""
    await hub.connect(websocket)
    try:
        await websocket.send_json(session.welcome_payload())
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
            else:
                session.handle_message(msg)
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(websocket)
        if hub.n_clients == 0:
            session.set_input(0)  # nobody holding the keys anymore
