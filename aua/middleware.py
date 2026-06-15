"""
aua/middleware.py — Request/response middleware pipeline.

Middleware runs before and after the query pipeline. Each middleware
receives the output of the previous one. Execution is ordered by YAML
registration order.

Short-circuit: raise an exception in before_query() to abort the request
and return an error response to the client without calling specialists.

Built-in middleware (examples / shipped defaults):
    PIIRedactionMiddleware  — redacts SSN, credit card, email patterns
    AuditMiddleware         — logs every request/response to the audit log
    TenantPolicyMiddleware  — applies per-tenant routing and field restrictions

YAML registration:
    middleware:
      - import_path: plugins.middleware:PIIRedactionMiddleware
        config:
          patterns:
            - "\\\\d{3}-\\\\d{2}-\\\\d{4}"   # SSN
      - import_path: aua.middleware:AuditMiddleware  # built-in

Usage:
    from aua.middleware import MiddlewarePipeline
    pipeline = MiddlewarePipeline()
    pipeline.add(PIIRedactionMiddleware())
    request = await pipeline.before_query(request)
    ...
    response = await pipeline.after_response(response)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

log = logging.getLogger(__name__)


class MiddlewarePipeline:
    """
    Ordered list of middleware components (#52: extended pipeline).

    Supports four extension points:
      before_query(request)           — runs before field classification
      after_response(response)        — runs after specialist + arbiter (reverse order)
      on_chunk(chunk, metadata)       — intercepts each SSE token chunk during streaming
      before_batch(job)               — runs before a batch job starts processing
      after_batch(job, results)       — runs after all batch items complete
      on_error(exc, request)          — runs when the query pipeline raises an exception

    Each hook is optional. If a middleware class does not implement a hook,
    that hook is silently skipped for that middleware.
    """

    def __init__(self) -> None:
        self._stack: list[Any] = []

    def add(self, mw: Any) -> None:
        self._stack.append(mw)
        log.info("Added middleware: %s", type(mw).__name__)

    async def before_query(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run all before_query() methods in stack order."""
        for mw in self._stack:
            if hasattr(mw, "before_query"):
                try:
                    result = await mw.before_query(request)
                    if isinstance(result, dict):
                        request = result
                except Exception:
                    log.error("Middleware %s.before_query failed", type(mw).__name__, exc_info=True)
                    raise
        return request

    async def after_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Run all after_response() methods in reverse stack order."""
        for mw in reversed(self._stack):
            if hasattr(mw, "after_response"):
                try:
                    result = await mw.after_response(response)
                    if isinstance(result, dict):
                        response = result
                except Exception:
                    log.error(
                        "Middleware %s.after_response failed", type(mw).__name__, exc_info=True
                    )
        return response

    async def on_chunk(self, chunk: str, metadata: dict[str, Any]) -> str:
        """
        (#52) Run all on_chunk() methods in stack order during SSE streaming.

        Each middleware receives the current chunk (possibly modified by a prior
        middleware) and the stream metadata dict. Return the chunk unchanged to
        pass it through. Return an empty string to suppress the chunk.

        metadata keys: session_id, trace_id, domain, routing_mode, chunk_index
        """
        for mw in self._stack:
            if hasattr(mw, "on_chunk"):
                try:
                    result = mw.on_chunk(chunk, metadata)
                    if hasattr(result, "__await__"):
                        result = await result
                    if isinstance(result, str):
                        chunk = result
                except Exception:  # noqa: BLE001
                    log.error(
                        "Middleware %s.on_chunk failed (chunk passed through)",
                        type(mw).__name__,
                        exc_info=True,
                    )
        return chunk

    async def before_batch(self, job: dict[str, Any]) -> dict[str, Any]:
        """
        (#52) Run all before_batch() methods in stack order before batch processing.

        job dict keys: job_id, n_queries, priority, queries (list), submitted_at
        """
        for mw in self._stack:
            if hasattr(mw, "before_batch"):
                try:
                    result = mw.before_batch(job)
                    if hasattr(result, "__await__"):
                        result = await result
                    if isinstance(result, dict):
                        job = result
                except Exception:  # noqa: BLE001
                    log.error("Middleware %s.before_batch failed", type(mw).__name__, exc_info=True)
        return job

    async def after_batch(
        self, job: dict[str, Any], results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        (#52) Run all after_batch() methods in reverse stack order after batch completion.

        results: list of per-query result dicts (response, u_score, latency_ms, error)
        """
        for mw in reversed(self._stack):
            if hasattr(mw, "after_batch"):
                try:
                    result = mw.after_batch(job, results)
                    if hasattr(result, "__await__"):
                        result = await result
                    if isinstance(result, list):
                        results = result
                except Exception:  # noqa: BLE001
                    log.error("Middleware %s.after_batch failed", type(mw).__name__, exc_info=True)
        return results

    async def on_error(self, exc: Exception, request: dict[str, Any]) -> dict[str, Any] | None:
        """
        (#52) Run all on_error() methods when the query pipeline raises an exception.

        Called in reverse stack order (innermost middleware gets first chance to handle).
        Return a fallback response dict to recover gracefully, or None to let the
        exception propagate.

        request: the before_query request dict that was in flight when the error occurred
        """
        for mw in reversed(self._stack):
            if hasattr(mw, "on_error"):
                try:
                    result = mw.on_error(exc, request)
                    if hasattr(result, "__await__"):
                        result = await result
                    if isinstance(result, dict):
                        return result  # first middleware that returns a dict wins
                except Exception:  # noqa: BLE001
                    log.error(
                        "Middleware %s.on_error raised another exception",
                        type(mw).__name__,
                        exc_info=True,
                    )
        return None  # let the exception propagate

    def registered(self) -> list[str]:
        return [type(mw).__name__ for mw in self._stack]


# ── Built-in middleware ───────────────────────────────────────────────────────


class PIIRedactionMiddleware:
    """
    Redacts personally identifiable information from queries before they
    reach specialist models.

    Default patterns: SSN, credit card numbers, email addresses.
    Custom patterns can be added via config.

    YAML:
        middleware:
          - import_path: aua.middleware:PIIRedactionMiddleware
            config:
              patterns:
                - "\\\\d{3}-\\\\d{2}-\\\\d{4}"
    """

    DEFAULT_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    ]

    def __init__(self, patterns: list[str] | None = None, replacement: str = "[REDACTED]") -> None:
        all_patterns = self.DEFAULT_PATTERNS + (patterns or [])
        self._regexes = [re.compile(p) for p in all_patterns]
        self._replacement = replacement

    async def before_query(self, request: dict[str, Any]) -> dict[str, Any]:
        query = request.get("query", "")
        for rx in self._regexes:
            query = rx.sub(self._replacement, query)
        return {**request, "query": query}

    async def after_response(self, response: dict[str, Any]) -> dict[str, Any]:
        return response  # PII not expected in responses


class AuditMiddleware:
    """
    Logs every request and response to the AUA audit log (state store).

    Does not modify request or response — pure side-effect middleware.

    YAML:
        middleware:
          - import_path: aua.middleware:AuditMiddleware
    """

    def __init__(self) -> None:
        self._request_times: dict[str, float] = {}

    async def before_query(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id", "unknown")
        self._request_times[session_id] = time.time()
        log.info(
            "audit.query_start session=%s query_len=%d",
            session_id,
            len(request.get("query", "")),
        )
        return request

    async def after_response(self, response: dict[str, Any]) -> dict[str, Any]:
        session_id = response.get("session_id", "unknown")
        start = self._request_times.pop(session_id, time.time())
        elapsed_ms = (time.time() - start) * 1000
        log.info(
            "audit.query_complete session=%s u_score=%.3f latency_ms=%.1f",
            session_id,
            response.get("u_score", 0),
            elapsed_ms,
        )
        return response


class TenantPolicyMiddleware:
    """
    Per-tenant isolation — field restrictions, rate limits, model bindings (#44).

    Reads the X-Tenant-ID request header, enforces policy, and sets the
    tenant context (via aua.tenancy) so state writes are namespaced.

    YAML:
        middleware:
          - import_path: aua.middleware:TenantPolicyMiddleware
            config:
              reject_unknown: true   # 403 for unknown tenants (default: false)
              tenants:
                tenant-a:
                  allowed_fields: [software_engineering, mathematics]
                  rate_limit_rpm: 60        # requests per minute
                  model_binding: swe        # force all queries to this specialist
                tenant-b:
                  allowed_fields: [law, software_engineering]
                  rate_limit_rpm: 120

    Enforcement:
      - allowed_fields: queries routed outside these fields get a 403-equivalent
        error via _tenant_allowed_fields (checked by the router)
      - model_binding: _tenant_model_binding is set on the request; the router
        uses it as a force_domain override
      - rate_limit_rpm: enforced via a per-tenant sliding window (one per tenant
        ID); 429 raised before the query reaches the router
      - reject_unknown: if true, requests with an unrecognised X-Tenant-ID get
        a 403 error rather than falling through as anonymous
    """

    def __init__(
        self,
        tenants: dict[str, dict[str, Any]] | None = None,
        reject_unknown: bool = False,
    ) -> None:
        from aua.tenancy import parse_tenant_policies

        self._policies = parse_tenant_policies(tenants or {})
        self._reject_unknown = reject_unknown
        # Per-tenant sliding-window rate limiters (created on demand)
        self._limiters: dict[str, Any] = {}

    def _get_limiter(self, tenant_id: str, rpm: int):
        if tenant_id not in self._limiters:
            from aua.rate_limit import SlidingWindowRateLimiter

            self._limiters[tenant_id] = SlidingWindowRateLimiter(rpm)
        return self._limiters[tenant_id]

    async def before_query(self, request: dict[str, Any]) -> dict[str, Any]:
        tenant_id = request.get("headers", {}).get("x-tenant-id")

        # Unknown tenant handling
        if tenant_id and tenant_id not in self._policies:
            if self._reject_unknown:
                raise PermissionError(
                    f"Unknown tenant '{tenant_id}'. "
                    "Contact your administrator to register this tenant ID."
                )
            # Unknown tenant but reject_unknown=false → pass through as anonymous
            tenant_id = None

        if tenant_id is None:
            return request  # Anonymous — no tenant policy

        # Set context variable so state writes are namespaced
        from aua.tenancy import set_tenant_id

        set_tenant_id(tenant_id)

        policy = self._policies[tenant_id]
        request = {**request, "_tenant_id": tenant_id}

        # Field allowlist
        if policy.allowed_fields:
            request = {**request, "_tenant_allowed_fields": policy.allowed_fields}

        # Model binding (force_domain equivalent)
        if policy.model_binding:
            request = {**request, "_tenant_model_binding": policy.model_binding}

        # Per-tenant rate limiting
        if policy.rate_limit_rpm:
            limiter = self._get_limiter(tenant_id, policy.rate_limit_rpm)
            allowed, retry_after = limiter.is_allowed(tenant_id)
            if not allowed:
                raise PermissionError(
                    f"Rate limit exceeded for tenant '{tenant_id}'. "
                    f"Retry after {retry_after:.1f}s."
                )

        return request

    async def after_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # Clear tenant context after response — keeps contextvars clean
        from aua.tenancy import set_tenant_id

        set_tenant_id(None)
        return response


# ── Global pipeline ───────────────────────────────────────────────────────────

_pipeline: MiddlewarePipeline | None = None


def get_middleware_pipeline() -> MiddlewarePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MiddlewarePipeline()
    return _pipeline


def reset_middleware_pipeline() -> MiddlewarePipeline:
    global _pipeline
    _pipeline = MiddlewarePipeline()
    return _pipeline
