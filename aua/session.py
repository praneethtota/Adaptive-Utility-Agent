"""
aua/session.py — Session context and ID propagation.

Every query in AUA gets three IDs that propagate through every component:

    session_id   — identifies a logical conversation/session (client-supplied
                   or auto-generated). Persistent across multiple queries in
                   a conversation.

    trace_id     — identifies a distributed trace. One per request. Used by
                   OTEL, audit log, and structured logs. W3C trace context format.

    request_id   — identifies a single HTTP request. One per request. Never
                   reused. Returned in X-Request-ID response header.

IDs are generated as UUID4 strings when not supplied by the client.

Usage:
    from aua.session import SessionContext, new_session_context, get_current

    # In router — create on each request:
    ctx = new_session_context(
        session_id=request.session_id,  # None → auto-generate
        trace_id=request.headers.get("traceparent"),
    )

    # In any component — read current context:
    ctx = get_current()
    log.info("Processing query", extra=ctx.log_fields())
"""

from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionContext:
    """
    Carries all IDs for a single query through every component.

    Stored in a contextvar so it's available to any code running
    in the same async task without explicit passing.
    """

    session_id: str
    trace_id: str
    request_id: str
    created_at: float = field(default_factory=time.time)

    # Optional enrichment fields (set by router after classification)
    domain: str | None = None
    routing_mode: str | None = None
    token_id: str | None = None

    def log_fields(self) -> dict[str, Any]:
        """Return a dict suitable for structured log extra= fields."""
        fields: dict[str, Any] = {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
        }
        if self.domain:
            fields["domain"] = self.domain
        if self.routing_mode:
            fields["routing_mode"] = self.routing_mode
        if self.token_id:
            fields["token_id"] = self.token_id
        return fields

    def as_headers(self) -> dict[str, str]:
        """Return HTTP headers to propagate context to downstream services."""
        return {
            "X-Session-ID": self.session_id,
            "X-Trace-ID": self.trace_id,
            "X-Request-ID": self.request_id,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return all fields as a plain dict (for API responses)."""
        return {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
        }


# ── Context var ───────────────────────────────────────────────────────────────

_ctx: contextvars.ContextVar[SessionContext | None] = contextvars.ContextVar(
    "aua_session_context", default=None
)


def new_session_context(
    session_id: str | None = None,
    trace_id: str | None = None,
    request_id: str | None = None,
) -> SessionContext:
    """
    Create a new SessionContext and set it as the current context.

    Args:
        session_id:  client-supplied session ID, or None to auto-generate
        trace_id:    W3C traceparent or existing trace ID, or None to auto-generate
        request_id:  request ID, or None to auto-generate

    Returns:
        The new SessionContext (also set as the thread-local current context).
    """
    ctx = SessionContext(
        session_id=session_id or str(uuid.uuid4()),
        trace_id=trace_id or _new_trace_id(),
        request_id=request_id or str(uuid.uuid4()),
    )
    _ctx.set(ctx)
    return ctx


def get_current() -> SessionContext:
    """
    Return the current SessionContext.

    Raises RuntimeError if called outside a request context.
    Use get_current_or_none() if outside is acceptable.
    """
    ctx = _ctx.get()
    if ctx is None:
        raise RuntimeError(
            "No SessionContext in current context. "
            "Call new_session_context() at the start of each request."
        )
    return ctx


def get_current_or_none() -> SessionContext | None:
    """Return the current SessionContext, or None if not in a request context."""
    return _ctx.get()


def set_current(ctx: SessionContext) -> None:
    """Set the current SessionContext explicitly (useful in tests)."""
    _ctx.set(ctx)


def _new_trace_id() -> str:
    """Generate a W3C-compatible trace ID (32 hex chars)."""
    return uuid.uuid4().hex + uuid.uuid4().hex[:16]
