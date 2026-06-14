"""
aua/auth_middleware.py — Bearer token authentication middleware.

Activated when security.auth_enabled: true in aua_config.yaml.

Flow:
  1. Check if the path is a public endpoint (health, docs) — pass through.
  2. Read Authorization: Bearer <token> header.
  3. Call TokenManager.verify() — raises TokenError on bad/expired/revoked token.
  4. Check the token has the scope required for this endpoint (ENDPOINT_SCOPES).
  5. Pass through on success; return 401 / 403 on failure.

When auth_enabled=false (default), this middleware is NOT added to the app,
so there is zero overhead on every request in development mode.
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from aua.auth import ENDPOINT_SCOPES, PUBLIC_ENDPOINTS, TokenError, get_token_manager

log = logging.getLogger(__name__)


class AUAAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforces bearer token authentication on all non-public endpoints.

    Token format: Authorization: Bearer aua_tk_<b64payload>.<hmac>

    Public endpoints (no token required):
        GET /health/live, GET /health/ready, GET /health/startup,
        GET /version, GET /docs, GET /openapi.json

    Configuration:
        security:
          auth_enabled: true
          token_secret_env: AUA_TOKEN_SECRET   # env var holding the secret
          token_expiry_days: 30

    Generate a token:
        aua token create --scope aua:query --expires 30d

    Use a token:
        curl -H "Authorization: Bearer aua_tk_..." http://localhost:8000/query ...
    """

    def __init__(self, app: Any, config: Any = None) -> None:
        super().__init__(app)
        self._config = config

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        method = request.method
        path = request.url.path

        # ── Public endpoints — no auth required ────────────────────────────
        endpoint_key = f"{method} {path}"
        if endpoint_key in PUBLIC_ENDPOINTS:
            return await call_next(request)

        # Strip trailing slashes and query params for matching
        clean_path = path.rstrip("/")
        clean_key = f"{method} {clean_path}"
        if clean_key in PUBLIC_ENDPOINTS:
            return await call_next(request)

        # ── Extract token ──────────────────────────────────────────────────
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "missing_token",
                    "message": (
                        "Authorization: Bearer <token> header required. "
                        "Generate a token with: aua token create --scope aua:query"
                    ),
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_str = auth_header[len("Bearer ") :]

        # ── Verify token ───────────────────────────────────────────────────
        try:
            manager = get_token_manager()
            token = manager.verify(token_str)
        except TokenError as e:
            return JSONResponse(
                status_code=401,
                content={"error": "invalid_token", "message": str(e)},
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            log.error("Token verification error: %s", e)
            return JSONResponse(
                status_code=401,
                content={"error": "token_error", "message": "Token verification failed"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # ── Check scope ────────────────────────────────────────────────────
        required_scope = ENDPOINT_SCOPES.get(endpoint_key) or ENDPOINT_SCOPES.get(clean_key)
        if required_scope and not token.has_scope(required_scope):
            return JSONResponse(
                status_code=403,
                content={
                    "error": "insufficient_scope",
                    "message": (
                        f"Token scope '{required_scope}' required for {endpoint_key}. "
                        f"Token has scopes: {token.scopes}"
                    ),
                },
            )

        # ── Pass through — attach token info to request state ─────────────
        request.state.aua_token = token
        request.state.aua_token_id = token.token_id
        request.state.aua_scopes = token.scopes

        return await call_next(request)
