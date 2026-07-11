"""FastAPI application factory: /health, the /ws demo socket, and the static
frontend (mounted LAST so /ws and /health win routing)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles

from traqmania.server.session import DemoSession
from traqmania.server.ws import Hub, handle_socket

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
REPO_ROOT = WEB_DIR.parent.parent  # only meaningful in a source checkout

# Documentation surfaced in the web UI (Explain -> Full documentation), in
# display order. These live at the repo root, not in the package, so the
# feature quietly disappears on a bare pip install without the sources.
_DOC_SOURCES = (
    ("SCIENCE", "docs/SCIENCE.md"),
    ("README", "README.md"),
    ("EXHIBITION", "docs/EXHIBITION.md"),
    ("ARCHITECTURE", "docs/ARCHITECTURE.md"),
    ("TM2020-CONCEPT", "docs/TM2020-CONCEPT.md"),
)


def _doc_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def discover_docs() -> dict[str, Path]:
    """Doc id -> existing markdown path (empty outside a source checkout)."""
    found = {}
    for doc_id, rel in _DOC_SOURCES:
        path = REPO_ROOT / rel
        if path.is_file():
            found[doc_id] = path
    return found


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

    @app.get("/api/docs")
    def docs_index() -> dict[str, Any]:
        docs = discover_docs()
        return {"docs": [{"id": doc_id, "title": _doc_title(path)}
                         for doc_id, path in docs.items()]}

    @app.get("/api/docs/{doc_id}")
    def doc_content(doc_id: str) -> dict[str, str]:
        path = discover_docs().get(doc_id)
        if path is None:
            raise HTTPException(status_code=404, detail=f"unknown doc '{doc_id}'")
        return {"id": doc_id, "title": _doc_title(path),
                "markdown": path.read_text(encoding="utf-8")}

    docs_assets = REPO_ROOT / "docs"
    if docs_assets.is_dir():  # images referenced by the markdown (hero GIF)
        app.mount("/docs-assets", StaticFiles(directory=docs_assets), name="docs-assets")

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await handle_socket(websocket, hub, session)

    if WEB_DIR.is_dir():  # keep LAST: the catch-all static mount must not shadow /ws
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
