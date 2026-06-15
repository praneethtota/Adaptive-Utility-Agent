"""
aua/retry.py — Transport-level retry with exponential backoff (#39).

This is distinct from aua/policy.py's assertion-level retry (which re-calls
a specialist when a policy assertion fails). This module handles transient
HTTP failures: 503s, connection errors, timeouts. It wraps _call() at the
transport layer before the response reaches scoring.

Algorithm:
    attempt 1: immediate
    attempt 2: base_delay_ms * jitter
    attempt 3: base_delay_ms * 2 * jitter
    attempt n: min(base_delay_ms * 2^(n-2), max_delay_ms) * jitter

Jitter: uniform random in [0.75, 1.25] of the computed delay.
This prevents thundering-herd when multiple specialists fail simultaneously.

Retryable conditions:
    - httpx.ConnectError         (endpoint not reachable)
    - httpx.ConnectTimeout       (connect phase timed out)
    - httpx.ReadTimeout          (response timed out)
    - HTTP 429 Too Many Requests (rate limited — back off and retry)
    - HTTP 503 Service Unavailable
    - HTTP 502 Bad Gateway
    - HTTP 504 Gateway Timeout

NOT retried:
    - HTTP 400 Bad Request       (bad payload — won't get better)
    - HTTP 401/403               (auth — won't get better)
    - HTTP 404                   (endpoint doesn't exist)
    - HTTP 422 Unprocessable     (validation error in our request)
    - HTTP 500 Internal Error    (specialist bug — unlikely to recover)

YAML config:
    router:
      retry:
        max_retries: 3          # 0 = disabled (default: 3)
        base_delay_ms: 200      # first retry delay in ms (default: 200)
        max_delay_ms: 5000      # cap on computed delay (default: 5000)
        jitter: true            # add ±25% random jitter (default: true)
        retryable_status_codes: [429, 502, 503, 504]  # default
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

from aua.config import RetryConfig as RetryConfig  # noqa: F401 — re-export

log = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that are worth retrying
DEFAULT_RETRYABLE_CODES: frozenset[int] = frozenset({429, 502, 503, 504})


# RetryConfig is the canonical class (defined in aua.config).
# Re-exported here for callers that import from aua.retry.


def _is_retryable_error(exc: Exception, retryable_codes: frozenset[int]) -> bool:
    """Return True when the exception represents a transient, retryable failure."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
        return True
    # FastAPI HTTPException with a retryable status code
    try:
        from fastapi import HTTPException

        if isinstance(exc, HTTPException) and exc.status_code in retryable_codes:
            return True
    except ImportError:
        pass
    # httpx response errors
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in retryable_codes:
        return True
    return False


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    retry_config: RetryConfig,
    specialist_name: str = "specialist",
    **kwargs: Any,
) -> T:
    """
    Call fn(*args, **kwargs) with retry on transient failures.

    Args:
        fn:               async callable to wrap
        *args:            positional args passed to fn
        retry_config:     RetryConfig controlling retry behaviour
        specialist_name:  name for logging (e.g. "swe")
        **kwargs:         keyword args passed to fn

    Returns:
        Result of fn(*args, **kwargs) on success.

    Raises:
        The last exception if all attempts fail.
    """
    retryable_codes = frozenset(retry_config.retryable_status_codes)
    max_attempts = max(1, retry_config.max_retries + 1)  # +1 for the initial attempt

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        delay = retry_config.delay_for_attempt(attempt)
        if delay > 0:
            log.info(
                "Retry %d/%d for %s — waiting %.2fs",
                attempt - 1,
                retry_config.max_retries,
                specialist_name,
                delay,
            )
            await asyncio.sleep(delay)

        try:
            return await fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_retryable_error(exc, retryable_codes):
                log.debug(
                    "%s: non-retryable error on attempt %d: %s",
                    specialist_name,
                    attempt,
                    exc,
                )
                raise
            if attempt < max_attempts:
                log.warning(
                    "%s: transient error on attempt %d/%d: %s",
                    specialist_name,
                    attempt,
                    max_attempts,
                    exc,
                )
            else:
                log.error(
                    "%s: all %d attempt(s) failed. Last error: %s",
                    specialist_name,
                    max_attempts,
                    exc,
                )

    # Should not be reached, but satisfy the type checker
    assert last_exc is not None
    raise last_exc
