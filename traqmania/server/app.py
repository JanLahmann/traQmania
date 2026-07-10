"""FastAPI application factory: /health, the /ws demo socket, and the static
frontend (mounted LAST so /ws and /health win routing)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from traqmania.server.session import DemoSession
from traqmania.server.ws import Hub, handle_socket

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


def create_app(config: dict[str, Any]) -> FastAPI:
    session = DemoSession(config)
    hub = Hub()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(session.run(hub), name="traqmania-session")
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="traQmania", lifespan=lifespan)
    app.state.config = config
    app.state.session = session
    app.state.hub = hub

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await handle_socket(websocket, hub, session)

    if WEB_DIR.is_dir():  # keep LAST: the catch-all static mount must not shadow /ws
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
