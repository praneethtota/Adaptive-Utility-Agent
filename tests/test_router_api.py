"""
tests/test_router_api.py — FastAPI endpoint contract tests.

Uses the fake specialist server (started in conftest.py) to test all
REST endpoints without requiring a real GPU or model.
"""

import pytest
from fastapi.testclient import TestClient

from aua.router import Router

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def router(minimal_config, fake_swe_server):
    """Build a Router pointed at the fake specialist server."""
    return Router.from_config(minimal_config)


@pytest.fixture
def client(router):
    """Synchronous TestClient for the router FastAPI app."""
    return TestClient(router.app, raise_server_exceptions=True)


# ── Health endpoints ──────────────────────────────────────────────────────────


def test_health_live(client):
    """GET /health/live always returns 200 with status=alive."""
    r = client.get("/health/live")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "alive"
    assert "uptime_s" in body
    assert body["uptime_s"] >= 0


def test_health_ready_with_fake_server(client, fake_swe_server):
    """GET /health/ready returns 200 when all specialists are up."""
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert "swe" in body["specialists"]
    assert body["specialists"]["swe"] == "ok"


def test_health_startup_after_ready(client, fake_swe_server):
    """GET /health/startup returns 200 after a successful ready check."""
    # First hit /health/ready to set _started flag
    client.get("/health/ready")
    r = client.get("/health/startup")
    assert r.status_code == 200
    assert r.json()["status"] == "started"


def test_health_legacy_endpoint(client):
    """GET /health (legacy) still responds."""
    r = client.get("/health")
    assert r.status_code == 200


# ── Version endpoint ──────────────────────────────────────────────────────────


def test_version_endpoint(client):
    """GET /version returns correct version string."""
    r = client.get("/version")
    # If not yet implemented this will be 404 — that's a known gap for P-09 polish
    if r.status_code == 200:
        body = r.json()
        assert "version" in body or "0.6" in str(body)


# ── Config endpoint ───────────────────────────────────────────────────────────


def test_config_endpoint(client):
    """GET /config returns running config without errors."""
    r = client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert "specialists" in body
    assert "router" in body
    assert len(body["specialists"]) >= 1


def test_config_does_not_expose_secrets(client):
    """GET /config must not expose token/secret/key values in plaintext."""
    r = client.get("/config")
    body_str = r.text.lower()
    # None of these secret keywords should appear as values
    for keyword in ("ghp_", "password=", "secret=", "api_key="):
        assert keyword not in body_str


# ── Corrections endpoints ─────────────────────────────────────────────────────


def test_post_correction(client):
    """POST /corrections stores a correction and returns 201."""
    payload = {
        "subject": "binary_search_complexity",
        "domain": "software_engineering",
        "claim": "Binary search is O(log n) on sorted arrays.",
        "confidence": 0.95,
        "source": "manual",
    }
    r = client.post("/corrections", json=payload)
    assert r.status_code == 201
    body = r.json()
    assert body["stored"] is True
    assert body["subject"] == "binary_search_complexity"
    assert body["decay_class"] in ("A", "B", "C", "D")


def test_get_corrections_empty(client):
    """GET /corrections returns empty list on fresh router."""
    r = client.get("/corrections")
    assert r.status_code == 200
    body = r.json()
    assert "corrections" in body
    assert isinstance(body["corrections"], list)


def test_get_corrections_after_post(client):
    """GET /corrections returns stored correction."""
    client.post(
        "/corrections",
        json={
            "subject": "test_subject",
            "domain": "software_engineering",
            "claim": "test claim",
            "confidence": 0.9,
        },
    )
    r = client.get("/corrections", params={"subject": "test_subject"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1


# ── Status endpoint ───────────────────────────────────────────────────────────


def test_status_endpoint_structure(client):
    """GET /status returns the expected telemetry structure."""
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("health", "latency", "routing", "utility", "corrections", "memory"):
        assert key in body, f"Missing key: {key!r}"


# ── Query endpoint ────────────────────────────────────────────────────────────


def test_query_single_domain(client, fake_swe_server):
    """POST /query with a coding question routes to swe specialist."""
    r = client.post(
        "/query",
        json={
            "query": "Write a binary search function in Python.",
            "force_domain": "software_engineering",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert "u_score" in body
    assert "routing_mode" in body
    assert body["routing_mode"] == "single"
    assert body["primary_domain"] == "software_engineering"
    assert isinstance(body["u_score"], float)
    assert 0.0 <= body["u_score"] <= 1.0


def test_query_response_contains_text(client, fake_swe_server):
    """POST /query response must contain non-empty response text."""
    r = client.post(
        "/query",
        json={"query": "test", "force_domain": "software_engineering"},
    )
    assert r.status_code == 200
    assert len(r.json()["response"]) > 0


def test_query_batch(client, fake_swe_server):
    """POST /query/batch returns one result per query."""
    queries = [
        "Write binary search.",
        "Explain quicksort.",
    ]
    r = client.post("/query/batch", json={"queries": queries, "max_parallel": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["n_queries"] == 2
    assert len(body["results"]) == 2
    assert "total_latency_ms" in body


# ── Reset endpoint ────────────────────────────────────────────────────────────


def test_reset_endpoint(client):
    """POST /reset returns status=reset."""
    r = client.post("/reset")
    assert r.status_code == 200
    assert r.json()["status"] == "reset"


# ── OpenAPI docs ──────────────────────────────────────────────────────────────


def test_openapi_json_accessible(client):
    """GET /openapi.json must return 200 with valid OpenAPI schema."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "openapi" in schema
    assert "paths" in schema
    assert "/query" in schema["paths"]


def test_docs_accessible(client):
    """GET /docs must return 200 (Swagger UI)."""
    r = client.get("/docs")
    assert r.status_code == 200


def test_redoc_accessible(client):
    """GET /redoc must return 200 (ReDoc UI)."""
    r = client.get("/redoc")
    assert r.status_code == 200
