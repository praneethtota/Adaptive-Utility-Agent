"""
aua/circuit_breaker.py — Per-specialist circuit breaker (#37, #38).

Prevents cascade failures when a specialist endpoint goes down. Instead of
letting every query time out for the full specialist_timeout seconds, the
circuit breaker opens after a failure threshold and immediately returns a
failure response for subsequent requests until a probe succeeds.

State machine:
    CLOSED  → normal operation; all calls pass through
              failure_count incremented on each failure
              transitions to OPEN when failure_count >= failure_threshold
              within failure_window_s seconds

    OPEN    → endpoint is down; calls are rejected immediately
              transitions to HALF_OPEN after recovery_timeout_s seconds

    HALF_OPEN → probe mode; one call is allowed through
                success → CLOSED (reset failure_count)
                failure → OPEN (reset recovery timer)

Failure conditions counted:
    - httpx.ConnectError / ConnectTimeout / ReadTimeout
    - HTTP 502, 503, 504, 429 responses
    - Any exception from _call() that is also retryable (after retries exhausted)

Non-failure conditions (do NOT trip the circuit):
    - HTTP 400, 401, 403, 404, 422, 500 — these are caller or specialist bugs,
      not endpoint availability issues. Tripping on 500 would hide bugs.

YAML config:
    router:
      circuit_breaker:
        enabled: true              # false = disable entirely (default: true)
        failure_threshold: 5       # failures within window before opening (default: 5)
        failure_window_s: 60.0     # sliding window for counting failures (default: 60)
        recovery_timeout_s: 30.0   # time in OPEN before HALF_OPEN probe (default: 30)
        success_threshold: 2       # consecutive successes in HALF_OPEN to close (default: 2)

Degraded-mode flag (#38):
    When a circuit is OPEN and a query would have gone to that specialist,
    the router routes to the arbiter fallback and sets degraded_mode=True
    on the RouterResponse. The response includes a degraded_specialists
    list so callers can detect partial availability.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from enum import Enum
from typing import Any

import httpx

# CircuitBreakerConfig lives in aua.config; re-exported here.
from aua.config import CircuitBreakerConfig as CircuitBreakerConfig  # noqa: F401

log = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Exceptions / status codes that count as circuit failures
_FAILURE_CODES: frozenset[int] = frozenset({429, 502, 503, 504})


def _is_circuit_failure(exc: Exception) -> bool:
    """Return True when the exception represents an endpoint availability failure."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
        return True
    try:
        from fastapi import HTTPException

        if isinstance(exc, HTTPException) and exc.status_code in _FAILURE_CODES:
            return True
    except ImportError:
        pass
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _FAILURE_CODES:
        return True
    return False


class CircuitBreaker:
    """
    Per-specialist circuit breaker.

    Thread-safe for asyncio (single-threaded event loop). Not safe for
    concurrent access from multiple OS threads.
    """

    def __init__(self, specialist_name: str, config: CircuitBreakerConfig) -> None:
        self._name = specialist_name
        self._cfg = config
        self._state = CircuitState.CLOSED
        self._failure_times: deque[float] = deque()  # timestamps of recent failures
        self._open_since: float = 0.0
        self._half_open_successes: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        self._maybe_transition_to_half_open()
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    def allows_call(self) -> bool:
        """
        Return True if the circuit should let this call through.

        CLOSED  → True (all calls pass)
        OPEN    → False (all calls blocked)
        HALF_OPEN → True for the first probe call; subsequent calls blocked
        """
        if not self._cfg.enabled:
            return True
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.OPEN:
            log.debug(
                "Circuit OPEN for %s — call blocked (recovery in %.0fs)",
                self._name,
                max(0.0, self._open_since + self._cfg.recovery_timeout_s - time.monotonic()),
            )
            return False
        # HALF_OPEN: allow exactly one probe
        return True

    def record_success(self) -> None:
        """Call after a successful specialist response."""
        if not self._cfg.enabled:
            return
        state = self.state
        if state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._cfg.success_threshold:
                self._close()
        elif state == CircuitState.CLOSED:
            # Prune old failure timestamps on success to prevent stale counts
            self._prune_failures()

    def record_failure(self, exc: Exception) -> None:
        """Call after a failed specialist call. Increments counter and may open circuit."""
        if not self._cfg.enabled:
            return
        if not _is_circuit_failure(exc):
            return  # only count availability failures, not logic errors

        now = time.monotonic()
        self._failure_times.append(now)
        self._prune_failures()

        state = self.state
        if state == CircuitState.HALF_OPEN:
            # Probe failed — reopen
            self._open()
        elif state == CircuitState.CLOSED:
            if len(self._failure_times) >= self._cfg.failure_threshold:
                log.warning(
                    "Circuit OPENING for %s — %d failures in %.0fs",
                    self._name,
                    len(self._failure_times),
                    self._cfg.failure_window_s,
                )
                self._open()

    def status_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable status snapshot for /status and /health endpoints."""
        return {
            "specialist": self._name,
            "state": self.state.value,
            "failure_count": len(self._failure_times),
            "failure_threshold": self._cfg.failure_threshold,
            "open_since": self._open_since if self._state != CircuitState.CLOSED else None,
            "recovery_timeout_s": self._cfg.recovery_timeout_s,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._open_since = time.monotonic()
        self._half_open_successes = 0

    def _close(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_times.clear()
        self._half_open_successes = 0
        self._open_since = 0.0
        log.info("Circuit CLOSED for %s — endpoint recovered", self._name)

    def _maybe_transition_to_half_open(self) -> None:
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._open_since >= self._cfg.recovery_timeout_s
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_successes = 0
            log.info(
                "Circuit HALF_OPEN for %s — sending probe request",
                self._name,
            )

    def _prune_failures(self) -> None:
        """Remove failure timestamps outside the sliding window."""
        cutoff = time.monotonic() - self._cfg.failure_window_s
        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()


class CircuitBreakerRegistry:
    """
    Holds one CircuitBreaker per specialist name.

    Created once at Router startup and shared across all requests.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, specialist_name: str) -> CircuitBreaker:
        if specialist_name not in self._breakers:
            self._breakers[specialist_name] = CircuitBreaker(specialist_name, self._config)
        return self._breakers[specialist_name]

    def all_status(self) -> list[dict[str, Any]]:
        return [cb.status_dict() for cb in self._breakers.values()]

    def open_specialists(self) -> list[str]:
        return [name for name, cb in self._breakers.items() if cb.is_open]
