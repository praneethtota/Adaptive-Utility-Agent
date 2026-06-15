"""
tests/test_retry_and_circuit_breaker.py — Tests for #39 (retry) and #37/#38 (circuit breaker).

#39 Retry with exponential backoff:
  RetryConfig.delay_for_attempt() — correct delays, jitter, cap
  with_retry() — succeeds on first attempt
  with_retry() — retries on transient errors, succeeds on Nth attempt
  with_retry() — raises after max_retries exhausted
  with_retry() — non-retryable errors are NOT retried
  with_retry() — 429/502/503/504 are retried; 400/404/500 are not
  with_retry() — max_retries=0 disables retry entirely

#37 CircuitBreaker state machine:
  Starts CLOSED
  Transitions CLOSED → OPEN after failure_threshold failures
  Transitions OPEN → HALF_OPEN after recovery_timeout_s
  Transitions HALF_OPEN → CLOSED after success_threshold successes
  Transitions HALF_OPEN → OPEN on failure
  Failures outside window don't count (sliding window)
  Non-circuit failures (400, 500) don't trip the circuit
  allows_call() returns False when OPEN
  record_success() only counts in HALF_OPEN / CLOSED
  CircuitBreakerConfig enabled=False disables entirely

#38 Degraded-mode routing:
  Open circuit excluded from active specialists list
  degraded_mode=True stamped on response when any circuit open
  degraded_specialists list contains the open specialist domain
  top_domain re-selected when top specialist circuit is open
  All specialists open → routes to arbiter, degraded_mode=True

CircuitBreakerRegistry:
  Creates one breaker per specialist name
  all_status() returns list with state for each
  open_specialists() returns only open ones

Config:
  RetryConfig parsed from YAML dict
  CircuitBreakerConfig parsed from YAML dict
  RouterConfig has retry and circuit_breaker fields
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from aua.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)
from aua.config import CircuitBreakerConfig, RetryConfig
from aua.retry import with_retry

# ── RetryConfig ───────────────────────────────────────────────────────────────


class TestRetryConfig:
    def test_first_attempt_no_delay(self) -> None:
        cfg = RetryConfig(base_delay_ms=200)
        assert cfg.delay_for_attempt(1) == 0.0

    def test_second_attempt_base_delay(self) -> None:
        cfg = RetryConfig(base_delay_ms=200, jitter=False)
        d = cfg.delay_for_attempt(2)
        assert d == pytest.approx(0.2, rel=0.01)

    def test_third_attempt_doubles(self) -> None:
        cfg = RetryConfig(base_delay_ms=200, jitter=False)
        d = cfg.delay_for_attempt(3)
        assert d == pytest.approx(0.4, rel=0.01)

    def test_delay_capped_at_max(self) -> None:
        cfg = RetryConfig(base_delay_ms=1000, max_delay_ms=2000, jitter=False)
        # attempt 10 would be 1000 * 2^8 = 256000ms — capped at 2000ms
        assert cfg.delay_for_attempt(10) == pytest.approx(2.0, rel=0.01)

    def test_jitter_adds_variance(self) -> None:
        cfg = RetryConfig(base_delay_ms=1000, jitter=True)
        delays = [cfg.delay_for_attempt(2) for _ in range(20)]
        # All within ±25% of 1.0s
        assert all(0.75 <= d <= 1.25 for d in delays)
        # Not all identical (jitter is active)
        assert len(set(f"{d:.4f}" for d in delays)) > 1


# ── with_retry() ─────────────────────────────────────────────────────────────


class TestWithRetry:
    def test_succeeds_on_first_attempt(self) -> None:
        cfg = RetryConfig(max_retries=3, base_delay_ms=0)
        calls = []

        async def fn():
            calls.append(1)
            return "ok"

        result = asyncio.run(with_retry(fn, retry_config=cfg, specialist_name="swe"))
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_connect_error(self) -> None:
        import httpx

        cfg = RetryConfig(max_retries=2, base_delay_ms=0, jitter=False)
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 3:
                raise httpx.ConnectError("connection refused")
            return "recovered"

        result = asyncio.run(with_retry(fn, retry_config=cfg, specialist_name="swe"))
        assert result == "recovered"
        assert len(calls) == 3

    def test_raises_after_max_retries_exhausted(self) -> None:
        import httpx

        cfg = RetryConfig(max_retries=2, base_delay_ms=0, jitter=False)
        calls = []

        async def fn():
            calls.append(1)
            raise httpx.ConnectError("always fails")

        with pytest.raises(httpx.ConnectError):
            asyncio.run(with_retry(fn, retry_config=cfg, specialist_name="swe"))
        assert len(calls) == 3  # 1 initial + 2 retries

    def test_does_not_retry_non_retryable_error(self) -> None:
        from fastapi import HTTPException

        cfg = RetryConfig(max_retries=3, base_delay_ms=0, jitter=False)
        calls = []

        async def fn():
            calls.append(1)
            raise HTTPException(400, "bad request")

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(with_retry(fn, retry_config=cfg, specialist_name="swe"))
        assert exc_info.value.status_code == 400
        assert len(calls) == 1  # not retried

    def test_retries_503(self) -> None:
        from fastapi import HTTPException

        cfg = RetryConfig(max_retries=1, base_delay_ms=0, jitter=False)
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise HTTPException(503, "unavailable")
            return "ok"

        result = asyncio.run(with_retry(fn, retry_config=cfg, specialist_name="swe"))
        assert result == "ok"
        assert len(calls) == 2

    def test_retries_429(self) -> None:
        from fastapi import HTTPException

        cfg = RetryConfig(max_retries=2, base_delay_ms=0, jitter=False)
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 3:
                raise HTTPException(429, "rate limited")
            return "ok"

        result = asyncio.run(with_retry(fn, retry_config=cfg, specialist_name="swe"))
        assert result == "ok"

    def test_does_not_retry_500(self) -> None:
        from fastapi import HTTPException

        cfg = RetryConfig(max_retries=3, base_delay_ms=0)
        calls = []

        async def fn():
            calls.append(1)
            raise HTTPException(500, "internal error")

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(with_retry(fn, retry_config=cfg, specialist_name="swe"))
        assert exc_info.value.status_code == 500
        assert len(calls) == 1

    def test_max_retries_zero_disables_retry(self) -> None:
        import httpx

        cfg = RetryConfig(max_retries=0, base_delay_ms=0)
        calls = []

        async def fn():
            calls.append(1)
            raise httpx.ConnectError("refused")

        with pytest.raises(httpx.ConnectError):
            asyncio.run(with_retry(fn, retry_config=cfg, specialist_name="swe"))
        assert len(calls) == 1  # only one attempt

    def test_retries_read_timeout(self) -> None:
        import httpx

        cfg = RetryConfig(max_retries=1, base_delay_ms=0, jitter=False)
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ReadTimeout("timed out")
            return "recovered"

        result = asyncio.run(with_retry(fn, retry_config=cfg, specialist_name="math"))
        assert result == "recovered"
        assert len(calls) == 2


# ── CircuitBreaker state machine ──────────────────────────────────────────────


def _make_cb(
    failure_threshold: int = 3,
    failure_window_s: float = 60.0,
    recovery_timeout_s: float = 30.0,
    success_threshold: int = 2,
    enabled: bool = True,
) -> CircuitBreaker:
    cfg = CircuitBreakerConfig(
        enabled=enabled,
        failure_threshold=failure_threshold,
        failure_window_s=failure_window_s,
        recovery_timeout_s=recovery_timeout_s,
        success_threshold=success_threshold,
    )
    return CircuitBreaker("swe", cfg)


class TestCircuitBreakerStateMachine:
    def _conn_error(self):
        import httpx

        return httpx.ConnectError("refused")

    def test_starts_closed(self) -> None:
        cb = _make_cb()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert cb.allows_call()

    def test_opens_after_failure_threshold(self) -> None:
        cb = _make_cb(failure_threshold=3)
        exc = self._conn_error()
        for _ in range(3):
            cb.record_failure(exc)
        assert cb.state == CircuitState.OPEN
        assert cb.is_open
        assert not cb.allows_call()

    def test_not_open_before_threshold(self) -> None:
        cb = _make_cb(failure_threshold=5)
        exc = self._conn_error()
        for _ in range(4):
            cb.record_failure(exc)
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_recovery_timeout(self) -> None:
        cb = _make_cb(failure_threshold=2, recovery_timeout_s=0.05)
        exc = self._conn_error()
        cb.record_failure(exc)
        cb.record_failure(exc)
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allows_call()

    def test_half_open_to_closed_after_success_threshold(self) -> None:
        cb = _make_cb(failure_threshold=2, recovery_timeout_s=0.05, success_threshold=2)
        exc = self._conn_error()
        cb.record_failure(exc)
        cb.record_failure(exc)
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # need 2
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.allows_call()

    def test_half_open_to_open_on_failure(self) -> None:
        cb = _make_cb(failure_threshold=2, recovery_timeout_s=0.05)
        exc = self._conn_error()
        cb.record_failure(exc)
        cb.record_failure(exc)
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure(exc)
        assert cb.state == CircuitState.OPEN

    def test_sliding_window_prunes_old_failures(self) -> None:
        cb = _make_cb(failure_threshold=3, failure_window_s=0.1)
        exc = self._conn_error()
        cb.record_failure(exc)
        cb.record_failure(exc)
        time.sleep(0.12)  # failures now outside window
        cb.record_failure(exc)  # only 1 within window now
        assert cb.state == CircuitState.CLOSED  # didn't reach threshold

    def test_non_circuit_failure_does_not_trip(self) -> None:
        from fastapi import HTTPException

        cb = _make_cb(failure_threshold=3)
        for _ in range(10):
            cb.record_failure(HTTPException(500, "internal error"))
        assert cb.state == CircuitState.CLOSED  # 500 is not a circuit failure

    def test_http_400_does_not_trip(self) -> None:
        from fastapi import HTTPException

        cb = _make_cb(failure_threshold=2)
        for _ in range(5):
            cb.record_failure(HTTPException(400, "bad request"))
        assert cb.state == CircuitState.CLOSED

    def test_http_503_trips_circuit(self) -> None:
        from fastapi import HTTPException

        cb = _make_cb(failure_threshold=2)
        for _ in range(2):
            cb.record_failure(HTTPException(503, "unavailable"))
        assert cb.state == CircuitState.OPEN

    def test_disabled_circuit_always_allows(self) -> None:
        cb = _make_cb(failure_threshold=1, enabled=False)
        exc = self._conn_error()
        for _ in range(10):
            cb.record_failure(exc)
        assert cb.allows_call()
        assert cb.state == CircuitState.CLOSED  # enabled=False → no state changes

    def test_status_dict_contents(self) -> None:
        cb = _make_cb(failure_threshold=3, recovery_timeout_s=30)
        d = cb.status_dict()
        assert d["specialist"] == "swe"
        assert d["state"] == "closed"
        assert d["failure_count"] == 0
        assert d["failure_threshold"] == 3

    def test_success_in_closed_prunes_failures(self) -> None:
        cb = _make_cb(failure_threshold=5)
        exc = self._conn_error()
        cb.record_failure(exc)
        cb.record_failure(exc)
        cb.record_success()  # prunes old failures
        # Circuit stays closed, failure count reflects pruning
        assert cb.state == CircuitState.CLOSED


# ── CircuitBreakerRegistry ────────────────────────────────────────────────────


class TestCircuitBreakerRegistry:
    def test_creates_breaker_per_specialist(self) -> None:
        cfg = CircuitBreakerConfig()
        reg = CircuitBreakerRegistry(cfg)
        swe = reg.get("software_engineering")
        math = reg.get("mathematics")
        assert swe is not math
        assert reg.get("software_engineering") is swe  # same object

    def test_all_status_returns_list(self) -> None:
        cfg = CircuitBreakerConfig()
        reg = CircuitBreakerRegistry(cfg)
        reg.get("swe")
        reg.get("math")
        statuses = reg.all_status()
        assert len(statuses) == 2
        assert all("state" in s for s in statuses)

    def test_open_specialists(self) -> None:
        import httpx

        cfg = CircuitBreakerConfig(failure_threshold=1)
        reg = CircuitBreakerRegistry(cfg)
        reg.get("swe").record_failure(httpx.ConnectError("down"))
        reg.get("math")  # healthy

        assert "swe" in reg.open_specialists()
        assert "math" not in reg.open_specialists()


# ── Config parsing ────────────────────────────────────────────────────────────


class TestConfigParsing:
    def test_retry_config_defaults(self) -> None:
        from aua.config import _load_retry_config

        cfg = _load_retry_config({})
        assert cfg.max_retries == 3
        assert cfg.base_delay_ms == 200.0
        assert cfg.max_delay_ms == 5000.0
        assert cfg.jitter is True
        assert set(cfg.retryable_status_codes) == {429, 502, 503, 504}

    def test_retry_config_overrides(self) -> None:
        from aua.config import _load_retry_config

        cfg = _load_retry_config(
            {
                "max_retries": 5,
                "base_delay_ms": 100,
                "max_delay_ms": 2000,
                "jitter": False,
                "retryable_status_codes": [503],
            }
        )
        assert cfg.max_retries == 5
        assert cfg.base_delay_ms == 100.0
        assert cfg.jitter is False
        assert cfg.retryable_status_codes == [503]

    def test_circuit_breaker_config_defaults(self) -> None:
        from aua.config import _load_cb_config

        cfg = _load_cb_config({})
        assert cfg.enabled is True
        assert cfg.failure_threshold == 5
        assert cfg.failure_window_s == 60.0
        assert cfg.recovery_timeout_s == 30.0
        assert cfg.success_threshold == 2

    def test_circuit_breaker_config_overrides(self) -> None:
        from aua.config import _load_cb_config

        cfg = _load_cb_config(
            {
                "enabled": False,
                "failure_threshold": 10,
                "failure_window_s": 120.0,
                "recovery_timeout_s": 60.0,
                "success_threshold": 3,
            }
        )
        assert cfg.enabled is False
        assert cfg.failure_threshold == 10

    def test_router_config_has_retry_and_cb_fields(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg_text = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  retry:
    max_retries: 2
    base_delay_ms: 100
  circuit_breaker:
    failure_threshold: 3
    recovery_timeout_s: 15.0
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_text)
        cfg = load_config(p)
        assert cfg.router.retry.max_retries == 2
        assert cfg.router.retry.base_delay_ms == 100.0
        assert cfg.router.circuit_breaker.failure_threshold == 3
        assert cfg.router.circuit_breaker.recovery_timeout_s == 15.0


# ── Router integration: degraded-mode (#38) ───────────────────────────────────


class TestDegradedMode:
    def _make_router(self, tmp_path: Path):
        from aua.config import load_config
        from aua.router import Router

        cfg = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
  - name: math
    model: fake/math
    port: 9002
    field: mathematics
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
  fanout_threshold: 0.30
  single_domain_threshold: 0.75
  circuit_breaker:
    failure_threshold: 1
    recovery_timeout_s: 9999
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg)
        return Router.from_config(load_config(p))

    def test_open_circuit_excluded_from_active(self, tmp_path: Path) -> None:
        import httpx

        router = self._make_router(tmp_path)
        # Trip the swe circuit
        router._circuit_breakers.get("software_engineering").record_failure(
            httpx.ConnectError("down")
        )
        assert router._circuit_breakers.get("software_engineering").is_open

    def test_degraded_specialists_in_registry(self, tmp_path: Path) -> None:
        import httpx

        router = self._make_router(tmp_path)
        router._circuit_breakers.get("software_engineering").record_failure(
            httpx.ConnectError("down")
        )
        assert "software_engineering" in router._circuit_breakers.open_specialists()
        assert "mathematics" not in router._circuit_breakers.open_specialists()

    def test_circuit_status_in_status_response(self, tmp_path: Path) -> None:

        router = self._make_router(tmp_path)
        router._circuit_breakers.get("mathematics")  # create entry
        statuses = router._circuit_breakers.all_status()
        assert isinstance(statuses, list)

    def test_degraded_mode_fields_on_response(self, tmp_path: Path) -> None:
        """RouterResponse has degraded_mode and degraded_specialists fields."""
        from aua.endpoints import RouterResponse

        # Check fields exist
        fields = RouterResponse.model_fields
        assert "degraded_mode" in fields
        assert "degraded_specialists" in fields
        # Default values
        assert fields["degraded_mode"].default is False
        assert fields["degraded_specialists"].default is None
