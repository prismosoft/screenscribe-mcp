"""ASGI entry point: OAuth + Streamable HTTP MCP."""

from __future__ import annotations

import contextlib
import os

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from screenscribe.oauth import OAuth
from screenscribe.server import DEFAULT_MODEL, mcp


oauth = OAuth.from_env()


class McpOAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/mcp"):
            auth = request.headers.get("authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else None
            if not oauth.token_valid(token):
                return JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers=oauth.challenge_headers(invalid=bool(token)),
                )
        return await call_next(request)


async def health(_request: Request) -> Response:
    oauth.store.cleanup()
    return JSONResponse({
        "ok": True,
        "service": "video-analyzer-mcp",
        "transport": "streamable-http",
        "mcp_url": oauth.resource,
        "oauth": True,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "gemini_model": os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        "thinking_level": os.getenv("GEMINI_THINKING_LEVEL", "medium"),
        "media_resolution": os.getenv("GEMINI_MEDIA_RESOLUTION", "low"),
    })


mcp_app = mcp.streamable_http_app()

@contextlib.asynccontextmanager
async def lifespan(_app: Starlette):
    async with mcp.session_manager.run():
        yield

routes = [Route("/health", health, methods=["GET"]), *oauth.routes(), Mount("/", app=mcp_app)]
app = Starlette(routes=routes, lifespan=lifespan)
app.add_middleware(McpOAuthMiddleware)


def run() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    run()
