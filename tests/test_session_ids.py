"""
tests/test_session_ids.py — #15: end-to-end session/trace/request ID tests.

Every query gets session_id, trace_id, request_id: client-supplied IDs are
honored, UUIDs are generated otherwise, all three are returned in every API
response (headers + body), and the context propagates to downstream
specialist calls and the audit log.
"""

from __future__ import annotations

import re
import uuid

import pytest
from fastapi.testclient import TestClient

from aua.router import Router

TRACE_RE = re.compile(r"^[0-9a-f]{48}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@pytest.fixture
def isolated_router(minimal_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Router.from_config(minimal_config)


@pytest.fixture
def client(isolated_router):
    return TestClient(isolated_router.app, raise_server_exceptions=True)


def test_every_response_carries_id_headers(client):
    """#15: IDs returned on EVERY API response, not just /query."""
    for path in ("/health/live", "/version", "/analytics", "/domain-tree"):
        r = client.get(path)
        assert UUID_RE.match(r.headers["X-Session-ID"]), path
        assert TRACE_RE.match(r.headers["X-Trace-ID"]), path
        assert UUID_RE.match(r.headers["X-Request-ID"]), path


def test_client_supplied_header_ids_are_honored(client):
    sid, rid = "my-session-42", str(uuid.uuid4())
    tid = uuid.uuid4().hex + uuid.uuid4().hex[:16]
    r = client.get(
        "/health/live",
        headers={"X-Session-ID": sid, "X-Trace-ID": tid, "X-Request-ID": rid},
    )
    assert r.headers["X-Session-ID"] == sid
    assert r.headers["X-Trace-ID"] == tid
    assert r.headers["X-Request-ID"] == rid


def test_request_ids_never_reused(client):
    ids = {client.get("/health/live").headers["X-Request-ID"] for _ in range(5)}
    assert len(ids) == 5


def test_query_echoes_generated_session_and_trace(client, fake_swe_server):
    """No session_id supplied → generated UUID echoed in body and header."""
    r = client.post("/query", json={"query": "Explain binary search"})
    body = r.json()
    assert UUID_RE.match(body["session_id"])
    assert body["session_id"] == r.headers["X-Session-ID"]
    assert TRACE_RE.match(body["trace_id"])
    assert body["trace_id"] == r.headers["X-Trace-ID"]
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_query_body_session_id_wins(client, fake_swe_server):
    """Body session_id overrides the header and is echoed back."""
    r = client.post(
        "/query",
        json={"query": "Explain binary search", "session_id": "body-session"},
        headers={"X-Session-ID": "header-session"},
    )
    assert r.json()["session_id"] == "body-session"


def test_library_api_generates_context(isolated_router, fake_swe_server):
    """Router.query() without HTTP still gets session/trace/request IDs."""
    import asyncio

    resp = asyncio.run(isolated_router.query("Explain binary search"))
    assert resp.session_id == "default"  # library default preserved
    assert TRACE_RE.match(resp.trace_id)
    assert UUID_RE.match(resp.request_id)


def test_ids_propagate_to_audit_log(client, isolated_router, fake_swe_server):
    sid = "audit-session-1"
    r = client.post("/query", json={"query": "Explain binary search", "session_id": sid})
    tid = r.json()["trace_id"]
    events = isolated_router._state_store.query(
        "audit_log", filters={"event_type": "query"}, limit=5
    )
    assert events, "query audit event written"
    assert events[0]["session_id"] == sid
    assert events[0]["trace_id"] == tid


def test_ids_propagate_to_specialist_calls(client, isolated_router, monkeypatch, fake_swe_server):
    """#15: downstream specialist HTTP calls carry the three ID headers."""
    captured: dict = {}
    import httpx

    orig_post = httpx.AsyncClient.post

    async def spy_post(self, url, *args, **kwargs):
        captured.update(kwargs.get("headers") or {})
        return await orig_post(self, url, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", spy_post)
    client.post(
        "/query",
        json={"query": "Explain binary search", "session_id": "prop-test"},
    )
    assert captured.get("X-Session-ID") == "prop-test"
    assert TRACE_RE.match(captured.get("X-Trace-ID", ""))
    assert UUID_RE.match(captured.get("X-Request-ID", ""))
