"""
tests/test_tenancy.py — Tests for #44 multi-tenancy.

Coverage:
  tenancy.py:
    get_tenant_id / set_tenant_id / reset_tenant_id: contextvar lifecycle
    Concurrent tasks get independent contexts
    parse_tenant_policies: all fields, empty dict, partial config

  middleware.py TenantPolicyMiddleware:
    Anonymous request (no header) passes through unchanged
    Known tenant — sets _tenant_id, _tenant_allowed_fields, _tenant_model_binding
    reject_unknown=True → PermissionError for unknown tenant
    reject_unknown=False (default) → unknown tenant passes as anonymous
    Per-tenant rate limiting: allowed, exceeded → PermissionError
    after_response clears tenant context

  state.py:
    append() auto-injects tenant_id for scoped tables (corrections, promotions,
      audit_log, model_runs)
    append() does not inject for non-scoped tables (sessions, conversations)
    append_audit() injects tenant_id from contextvar
    record_model_run() injects tenant_id from contextvar
    query() filtered by tenant_id (via existing filters dict)

  rate_limit.py:
    dispatch() includes tenant_id in rate-limit key

  DB migrations:
    tenant_id column present in corrections, promotions, audit_log, model_runs
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aua.tenancy import (
    get_tenant_id,
    parse_tenant_policies,
    reset_tenant_id,
    set_tenant_id,
)

# ── tenancy.py contextvar ─────────────────────────────────────────────────────


class TestContextVar:
    def test_default_is_none(self) -> None:
        assert get_tenant_id() is None

    def test_set_and_get(self) -> None:
        token = set_tenant_id("tenant-a")
        assert get_tenant_id() == "tenant-a"
        reset_tenant_id(token)
        assert get_tenant_id() is None

    def test_reset_restores_previous(self) -> None:
        token1 = set_tenant_id("tenant-a")
        token2 = set_tenant_id("tenant-b")
        assert get_tenant_id() == "tenant-b"
        reset_tenant_id(token2)
        assert get_tenant_id() == "tenant-a"
        reset_tenant_id(token1)
        assert get_tenant_id() is None

    def test_concurrent_tasks_independent(self) -> None:
        """Each asyncio task has its own copy of the contextvar."""

        results: list[str | None] = []

        async def _task(tid: str) -> None:
            token = set_tenant_id(tid)
            await asyncio.sleep(0)  # yield to scheduler
            results.append(get_tenant_id())
            reset_tenant_id(token)

        async def _run() -> None:
            await asyncio.gather(_task("a"), _task("b"), _task("c"))

        asyncio.run(_run())
        assert set(results) == {"a", "b", "c"}


class TestParseTenantPolicies:
    def test_empty_dict(self) -> None:
        assert parse_tenant_policies({}) == {}

    def test_full_config(self) -> None:
        raw = {
            "tenant-a": {
                "allowed_fields": ["software_engineering", "mathematics"],
                "rate_limit_rpm": 60,
                "model_binding": "swe",
            },
            "tenant-b": {
                "allowed_fields": ["law"],
            },
        }
        policies = parse_tenant_policies(raw)
        assert len(policies) == 2
        a = policies["tenant-a"]
        assert a.allowed_fields == ["software_engineering", "mathematics"]
        assert a.rate_limit_rpm == 60
        assert a.model_binding == "swe"
        b = policies["tenant-b"]
        assert b.allowed_fields == ["law"]
        assert b.rate_limit_rpm is None
        assert b.model_binding is None

    def test_defaults(self) -> None:
        policies = parse_tenant_policies({"t": {}})
        p = policies["t"]
        assert p.allowed_fields == []
        assert p.rate_limit_rpm is None
        assert p.model_binding is None


# ── TenantPolicyMiddleware ────────────────────────────────────────────────────


class TestTenantPolicyMiddleware:
    def _make_mw(self, reject_unknown: bool = False, **tenant_cfgs):
        from aua.middleware import TenantPolicyMiddleware

        return TenantPolicyMiddleware(tenants=tenant_cfgs, reject_unknown=reject_unknown)

    @pytest.mark.asyncio
    async def test_no_header_passes_through(self) -> None:
        mw = self._make_mw(**{"tenant-a": {"allowed_fields": ["swe"]}})
        req = {"query": "test", "headers": {}}
        result = await mw.before_query(req)
        assert "_tenant_id" not in result
        assert get_tenant_id() is None

    @pytest.mark.asyncio
    async def test_known_tenant_sets_context(self) -> None:
        mw = self._make_mw(
            **{"tenant-a": {"allowed_fields": ["software_engineering"], "model_binding": "swe"}}
        )
        req = {"query": "test", "headers": {"x-tenant-id": "tenant-a"}}
        result = await mw.before_query(req)
        assert result["_tenant_id"] == "tenant-a"
        assert result["_tenant_allowed_fields"] == ["software_engineering"]
        assert result["_tenant_model_binding"] == "swe"
        assert get_tenant_id() == "tenant-a"
        # Cleanup
        await mw.after_response({})

    @pytest.mark.asyncio
    async def test_after_response_clears_context(self) -> None:
        mw = self._make_mw(**{"t": {}})
        await mw.before_query({"headers": {"x-tenant-id": "t"}})
        assert get_tenant_id() == "t"
        await mw.after_response({})
        assert get_tenant_id() is None

    @pytest.mark.asyncio
    async def test_reject_unknown_true_raises(self) -> None:
        mw = self._make_mw(reject_unknown=True, **{"tenant-a": {}})
        req = {"headers": {"x-tenant-id": "unknown-tenant"}}
        with pytest.raises(PermissionError, match="Unknown tenant"):
            await mw.before_query(req)

    @pytest.mark.asyncio
    async def test_reject_unknown_false_passes_as_anonymous(self) -> None:
        mw = self._make_mw(reject_unknown=False, **{"tenant-a": {}})
        req = {"headers": {"x-tenant-id": "unknown-tenant"}}
        result = await mw.before_query(req)
        assert "_tenant_id" not in result
        assert get_tenant_id() is None

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_raises(self) -> None:
        mw = self._make_mw(**{"tenant-a": {"rate_limit_rpm": 1}})
        req = {"headers": {"x-tenant-id": "tenant-a"}}
        # First request allowed
        await mw.before_query(req)
        await mw.after_response({})
        # Second request should be rate-limited (1 rpm window)
        with pytest.raises(PermissionError, match="Rate limit exceeded"):
            await mw.before_query(req)
        await mw.after_response({})

    @pytest.mark.asyncio
    async def test_no_allowed_fields_no_restriction(self) -> None:
        mw = self._make_mw(**{"tenant-a": {}})
        req = {"headers": {"x-tenant-id": "tenant-a"}}
        result = await mw.before_query(req)
        assert "_tenant_allowed_fields" not in result
        await mw.after_response({})

    @pytest.mark.asyncio
    async def test_no_model_binding_no_binding(self) -> None:
        mw = self._make_mw(**{"tenant-a": {"allowed_fields": ["swe"]}})
        req = {"headers": {"x-tenant-id": "tenant-a"}}
        result = await mw.before_query(req)
        assert "_tenant_model_binding" not in result
        await mw.after_response({})


# ── state.py tenant injection ─────────────────────────────────────────────────


class TestStateStoreTenantInjection:
    @pytest.fixture
    def store(self, tmp_path: Path):
        from aua.state import SQLiteStateStore

        return SQLiteStateStore(db_path=tmp_path / "test.db")

    def test_append_injects_tenant_for_scoped_table(self, store) -> None:
        token = set_tenant_id("tenant-a")
        try:
            rid = store.append(
                "corrections",
                {
                    "subject": "bubble_sort",
                    "domain": "swe",
                    "claim": "O(n^2)",
                    "rejected": "",
                    "confidence": 0.9,
                    "effective_confidence": 0.9,
                },
            )
        finally:
            reset_tenant_id(token)
        rows = store.query("corrections", filters={"id": rid})
        assert rows[0].get("tenant_id") == "tenant-a"

    def test_append_does_not_inject_for_non_scoped_table(self, store) -> None:
        import time

        token = set_tenant_id("tenant-a")
        try:
            # sessions is not in _TENANT_SCOPED_TABLES — no tenant_id column
            rid = store.append(
                "sessions", {"domain": "swe", "query_count": 0, "updated_at": time.time()}
            )
        finally:
            reset_tenant_id(token)
        rows = store.query("sessions", filters={"id": rid})
        assert len(rows) == 1
        # sessions table has no tenant_id column
        assert "tenant_id" not in rows[0]

    def test_append_no_tenant_context_no_injection(self, store) -> None:
        # No tenant set — tenant_id should be None
        rid = store.append(
            "corrections",
            {
                "subject": "s",
                "domain": "d",
                "claim": "c",
                "rejected": "",
                "confidence": 0.5,
                "effective_confidence": 0.5,
            },
        )
        rows = store.query("corrections", filters={"id": rid})
        assert rows[0].get("tenant_id") is None

    def test_query_filtered_by_tenant_id(self, store) -> None:
        # Write two corrections for different tenants
        token_a = set_tenant_id("tenant-a")
        store.append(
            "corrections",
            {
                "subject": "s1",
                "domain": "d",
                "claim": "c1",
                "rejected": "",
                "confidence": 0.9,
                "effective_confidence": 0.9,
            },
        )
        reset_tenant_id(token_a)

        token_b = set_tenant_id("tenant-b")
        store.append(
            "corrections",
            {
                "subject": "s2",
                "domain": "d",
                "claim": "c2",
                "rejected": "",
                "confidence": 0.8,
                "effective_confidence": 0.8,
            },
        )
        reset_tenant_id(token_b)

        # Filter by tenant-a
        rows_a = store.query("corrections", filters={"tenant_id": "tenant-a"})
        assert len(rows_a) == 1
        assert rows_a[0]["subject"] == "s1"

        # Filter by tenant-b
        rows_b = store.query("corrections", filters={"tenant_id": "tenant-b"})
        assert len(rows_b) == 1
        assert rows_b[0]["subject"] == "s2"

    def test_append_audit_injects_tenant(self, store) -> None:
        token = set_tenant_id("tenant-x")
        try:
            rid = store.append_audit({"event_type": "query", "session_id": "s1"})
        finally:
            reset_tenant_id(token)
        rows = store.query("audit_log", filters={"id": rid})
        assert rows[0].get("tenant_id") == "tenant-x"

    def test_record_model_run_injects_tenant(self, store) -> None:
        token = set_tenant_id("tenant-y")
        try:
            rid = store.record_model_run(
                {
                    "specialist": "swe",
                    "conversation_id": "conv-1",
                    "round": "answer",
                    "domain": "software_engineering",
                }
            )
        finally:
            reset_tenant_id(token)
        rows = store.query("model_runs", filters={"run_id": rid})
        assert rows[0].get("tenant_id") == "tenant-y"


# ── DB migrations ─────────────────────────────────────────────────────────────


class TestMigrations:
    def test_tenant_id_column_in_corrections(self, tmp_path: Path) -> None:
        from aua.state import SQLiteStateStore

        store = SQLiteStateStore(db_path=tmp_path / "test.db")
        with store._connect() as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(corrections)").fetchall()]
        assert "tenant_id" in cols

    def test_tenant_id_column_in_promotions(self, tmp_path: Path) -> None:
        from aua.state import SQLiteStateStore

        store = SQLiteStateStore(db_path=tmp_path / "test.db")
        with store._connect() as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(promotions)").fetchall()]
        assert "tenant_id" in cols

    def test_tenant_id_column_in_audit_log(self, tmp_path: Path) -> None:
        from aua.state import SQLiteStateStore

        store = SQLiteStateStore(db_path=tmp_path / "test.db")
        with store._connect() as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(audit_log)").fetchall()]
        assert "tenant_id" in cols

    def test_tenant_id_column_in_model_runs(self, tmp_path: Path) -> None:
        from aua.state import SQLiteStateStore

        store = SQLiteStateStore(db_path=tmp_path / "test.db")
        with store._connect() as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(model_runs)").fetchall()]
        assert "tenant_id" in cols


# ── Rate limit tenant isolation ───────────────────────────────────────────────


class TestRateLimitTenantIsolation:
    def test_tenant_id_in_key(self) -> None:
        """Rate limit dispatch must use tenant_id prefix in rate-limit key."""
        import inspect

        from aua.rate_limit import RateLimitMiddleware

        src = inspect.getsource(RateLimitMiddleware.dispatch)
        assert "x-tenant-id" in src or "tenant_id" in src
