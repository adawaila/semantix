"""Optional API key authentication middleware.

If the ``SEMANTIX_API_KEY`` environment variable is set, every request must
supply the key via one of:

    Authorization: Bearer <key>
    X-API-Key: <key>

Requests to ``/health``, ``/docs``, ``/openapi.json``, and ``/redoc`` are
always allowed so the server stays monitorable and self-documented without
credentials.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_EXEMPT = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXEMPT:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        key_header = request.headers.get("X-API-Key", "")

        if auth.startswith("Bearer "):
            provided = auth[len("Bearer "):]
        else:
            provided = key_header

        if provided != self.api_key:
            return JSONResponse(
                {"detail": "Invalid or missing API key"},
                status_code=401,
            )

        return await call_next(request)
