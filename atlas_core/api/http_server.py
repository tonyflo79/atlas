"""Atlas HTTP server — FastAPI on localhost:9879.

Mirrors the MCP tool surface as REST endpoints for non-MCP clients (the
web dashboard, programmatic curl access, integration tests).

Phase 2 W6 ships the minimal surface: health, tool listing, tool dispatch,
and a /verify-chain shortcut. Phase 2 W7 layers in WebSocket streaming for
the live-Ripple visualization the launch demo needs.

Spec: 05 - Atlas Architecture & Schema § 2
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import stat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI

    from atlas_core.api.mcp_server import AtlasMCPServer


log = logging.getLogger(__name__)


DEFAULT_HTTP_PORT: int = 9879
"""Atlas HTTP port — vault-search uses 9878, Atlas takes 9879."""


API_TOKEN_ENV: str = "ATLAS_API_TOKEN"
"""Env override for the HTTP bearer token. Takes precedence over the file."""

API_TOKEN_FILENAME: str = "api_token"
"""Per-install bearer token, stored under the Atlas data dir (mode 0600)."""


def load_or_create_api_token(data_dir: Path) -> str:
    """Resolve the HTTP API bearer token, fail-closed by provisioning one.

    Resolution order (matches the repo's ``ATLAS_*`` env + ``~/.atlas``
    convention):

      1. ``ATLAS_API_TOKEN`` env var, if set and non-empty.
      2. ``<data_dir>/api_token``, if it already exists.
      3. Otherwise mint a fresh URL-safe token, write it ``0600``, return it.

    Trusted local clients read the same file (or share the env var); a
    cross-origin web page cannot guess the value, so mutation and
    data-returning endpoints are safe even under wildcard CORS.
    """
    env_token = os.environ.get(API_TOKEN_ENV, "").strip()
    if env_token:
        return env_token

    token_path = data_dir / API_TOKEN_FILENAME
    if token_path.exists():
        existing = token_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing

    token = secrets.token_urlsafe(32)
    data_dir.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token + "\n", encoding="utf-8")
    token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 — owner-only
    log.info("Provisioned Atlas HTTP API token at %s", token_path)
    return token


# Defined at module scope (not inside create_http_app) so FastAPI can resolve
# the annotation. `from __future__ import annotations` turns annotations into
# strings, and FastAPI resolves them against module globals.
from pydantic import BaseModel as _BaseModel  # noqa: E402  (intentional placement; see comment above)
from pydantic import Field as _Field  # noqa: E402


class DispatchBody(_BaseModel):
    params: dict[str, Any] = _Field(default_factory=dict)


def create_http_app(
    *, mcp_server: AtlasMCPServer, auth_token: str | None = None,
) -> FastAPI:
    """Build the FastAPI app wrapping the MCP server.

    Endpoints:
      GET  /health          — liveness probe (unauthenticated)
      GET  /tools           — list registered MCP tools (authenticated)
      POST /tools/{name}    — dispatch a tool with JSON body params (authenticated)
      GET  /verify-chain    — shortcut for ledger.verify_chain (authenticated)
      GET  /events          — live SSE stream (unauthenticated, read-only)

    When ``auth_token`` is provided, ``/tools`` and ``/verify-chain`` require an
    ``Authorization: Bearer <token>`` header. The launchd/uvicorn entry point
    (``api_app``) always supplies one via :func:`load_or_create_api_token`, so a
    real deployment is fail-closed; a page a user visits cannot forge the header
    and therefore cannot dispatch mutations (``memory.forget``,
    ``adjudication.resolve``, ``sharing.grant`` …) or read personal memory back,
    even though CORS stays permissive for the read-only ``/events`` stream. When
    ``auth_token`` is ``None`` the surface is open — reserved for in-process test
    clients and explicit local opt-out.
    """
    from fastapi import Depends, FastAPI, Header, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware

    # Annotated with ``str | None`` (not a fastapi type) so FastAPI can resolve
    # the hint under ``from __future__ import annotations``; ``Header`` is a
    # runtime default, evaluated here where the import is in scope.
    async def require_auth(
        authorization: str | None = Header(default=None),
    ) -> None:
        """Reject requests lacking a valid bearer token. No-op when unset."""
        if auth_token is None:
            return
        scheme, _, presented = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            # bytes, not str: compare_digest raises TypeError on non-ASCII
            # str input, which would turn a garbage header into a 500.
            presented.encode("utf-8"), auth_token.encode("utf-8"),
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    app = FastAPI(
        title="Atlas API",
        version="0.1.0a1",
        description=(
            "Open-source local-first cognitive memory with AGM-compliant "
            "belief revision and automatic downstream reassessment."
        ),
    )

    # Permissive CORS — the Obsidian plugin and the local viewer pages
    # (served from :8765 or any other port) need to subscribe to /events
    # from a different origin; without CORS, browsers block the
    # EventSource handshake and the page shows "error — is the API server
    # running?" even when the API is healthy. See site/live-real.html for
    # the consumer. Wildcard CORS is safe for the privileged surface only
    # because /tools and /verify-chain require a bearer token a
    # cross-origin page cannot obtain (see require_auth); with
    # allow_credentials=False there are no ambient cookies to ride either.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "atlas",
            "version": "0.1.0a1",
        }

    @app.get("/tools", dependencies=[Depends(require_auth)])
    async def list_tools() -> dict[str, Any]:
        return {"tools": mcp_server.list_tools()}

    @app.post("/tools/{tool_name}", dependencies=[Depends(require_auth)])
    async def dispatch_tool(tool_name: str, body: DispatchBody) -> dict[str, Any]:
        result = await mcp_server.dispatch(tool_name, body.params)
        if not result.ok:
            raise HTTPException(status_code=400, detail=result.error)
        return {"ok": True, "result": result.result}

    @app.get("/verify-chain", dependencies=[Depends(require_auth)])
    async def verify_chain() -> dict[str, Any]:
        result = await mcp_server.dispatch("ledger.verify_chain", {})
        if not result.ok:
            raise HTTPException(status_code=500, detail=result.error)
        return result.result

    @app.get("/events")
    async def events_stream():
        """Server-Sent Events stream for live Atlas activity. The
        Obsidian plugin and live-Ripple visualization both subscribe.
        Format: text/event-stream with one `data: {json}` per event."""
        from fastapi.responses import StreamingResponse

        from atlas_core.api.events import GLOBAL_BROADCASTER

        async def event_generator():
            queue = GLOBAL_BROADCASTER.subscribe()
            try:
                while True:
                    event = await queue.get()
                    yield event.to_sse_line()
            finally:
                GLOBAL_BROADCASTER.unsubscribe(queue)

        return StreamingResponse(
            event_generator(), media_type="text/event-stream",
        )

    @app.get("/events/stats")
    async def events_stats() -> dict[str, Any]:
        """Liveness check for the event broadcaster — useful for
        debugging when the Obsidian plugin shows no events."""
        from atlas_core.api.events import GLOBAL_BROADCASTER
        return {
            "subscribers": GLOBAL_BROADCASTER.n_subscribers,
            "buffered_events": GLOBAL_BROADCASTER.n_buffered,
        }

    return app
