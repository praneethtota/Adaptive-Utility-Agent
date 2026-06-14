"""
aua/rate_limit.py — Per-scope rate limiting with 429 + Retry-After.

Configuration:
    rate_limits:
      aua:query:
        requests_per_minute: 60
      aua:admin:
        requests_per_minute: 10
      default:
        requests_per_minute: 120

Behavior: reject with 429 + Retry-After header when limit exceeded.
Tracks per (client_ip, scope) using a sliding window.

Usage (as FastAPI middleware):
    from aua.rate_limit import RateLimitMiddleware
    app.add_middleware(RateLimitMiddleware, config=cfg)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    In-process sliding window rate limiter.

    Tracks request timestamps per (client_id, scope) key.
    Thread-safe enough for single-process async use.
    """

    def __init__(self, requests_per_minute: int = 120) -> None:
        self.rpm = requests_per_minute
        self.window_s = 60.0
        self._windows: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str) -> tuple[bool, float]:
        """
        Check whether a request is allowed.

        Returns:
            (allowed, retry_after_seconds)
            retry_after_seconds is 0 when allowed.
        """
        now = time.time()
        window = self._windows[key]

        # Evict timestamps outside the window
        cutoff = now - self.window_s
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.rpm:
            # Retry after oldest request exits the window
            retry_after = self.window_s - (now - window[0])
            return False, max(0.1, retry_after)

        window.append(now)
        return True, 0.0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware that enforces per-scope rate limits.

    Exempt paths: /health/*, /version, /docs, /openapi.json, /metrics

    Config format (from aua_config.yaml):
        rate_limits:
          aua:query:
            requests_per_minute: 60
          default:
            requests_per_minute: 120
    """

    EXEMPT_PREFIXES = ("/health/", "/version", "/docs", "/openapi.json", "/metrics", "/redoc")

    def __init__(self, app: Any, config: Any | None = None) -> None:
        super().__init__(app)
        self._limiters: dict[str, SlidingWindowRateLimiter] = {}
        self._default_rpm = 120
        self._load_config(config)

    def _load_config(self, config: Any) -> None:
        if config is None:
            self._limiters["default"] = SlidingWindowRateLimiter(self._default_rpm)
            return

        rl_cfg = getattr(config, "rate_limits", None)
        if rl_cfg is None:
            self._limiters["default"] = SlidingWindowRateLimiter(self._default_rpm)
            return

        if isinstance(rl_cfg, dict):
            items = rl_cfg.items()
        else:
            items = vars(rl_cfg).items() if hasattr(rl_cfg, "__dict__") else []

        for scope, limits in items:
            rpm = (
                limits.get("requests_per_minute", self._default_rpm)
                if isinstance(limits, dict)
                else self._default_rpm
            )
            self._limiters[scope] = SlidingWindowRateLimiter(rpm)
            log.info("Rate limit: %s → %d rpm", scope, rpm)

        if "default" not in self._limiters:
            self._limiters["default"] = SlidingWindowRateLimiter(self._default_rpm)

    def _get_limiter(self, scope: str) -> SlidingWindowRateLimiter:
        return self._limiters.get(scope, self._limiters["default"])

    def _get_client_id(self, request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # Skip exempt paths
        path = request.url.path
        if any(path.startswith(p) for p in self.EXEMPT_PREFIXES):
            return await call_next(request)

        client_id = self._get_client_id(request)
        # #44: prefix key with tenant ID when X-Tenant-ID is present so
        # per-tenant traffic is rate-limited independently of other tenants
        tenant_id = request.headers.get("x-tenant-id", "")
        # Derive scope from path (simplified — full auth integration in v1.0)
        scope = _path_to_scope(path)
        key = f"{tenant_id}:{client_id}:{scope}" if tenant_id else f"{client_id}:{scope}"

        limiter = self._get_limiter(scope)
        allowed, retry_after = limiter.is_allowed(key)

        if not allowed:
            log.warning(
                "Rate limited: client=%s scope=%s retry_after=%.1fs", client_id, scope, retry_after
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "AUA_RATE_LIMITED",
                    "message": f"Rate limit exceeded. Retry after {retry_after:.1f}s.",
                    "retry_after_seconds": round(retry_after, 1),
                    "scope": scope,
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        return await call_next(request)


def _path_to_scope(path: str) -> str:
    """Map a URL path to its primary scope (simplified)."""
    mapping = {
        "/query": "aua:query",
        "/query/stream": "aua:stream",
        "/query/batch": "aua:batch",
        "/status": "aua:status",
        "/config": "aua:config:read",
        "/corrections": "aua:corrections:read",
        "/deploy": "aua:deploy",
        "/extensions": "aua:extensions:read",
        "/metrics": "aua:status",
    }
    for prefix, scope in mapping.items():
        if path.startswith(prefix):
            return scope
    return "default"
