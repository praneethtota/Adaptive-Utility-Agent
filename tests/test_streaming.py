"""
tests/test_streaming.py — POST /query/stream SSE streaming tests.

Tests the full SSE event sequence: start → chunk(s) → done | error.
"""

import json

import pytest
from fastapi.testclient import TestClient

from aua.router import Router


@pytest.fixture
def router(minimal_config, fake_swe_server):
    return Router.from_config(minimal_config)


@pytest.fixture
def client(router):
    return TestClient(router.app, raise_server_exceptions=True)


def _parse_sse(response_text: str) -> list[dict]:
    """Parse SSE stream text into list of event dicts."""
    events = []
    for line in response_text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    return events


def test_stream_returns_200(client, fake_swe_server):
    """POST /query/stream must return 200 with text/event-stream content type."""
    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "Write binary search.", "force_domain": "software_engineering"},
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")


def test_stream_emits_start_event(client, fake_swe_server):
    """First SSE event must be type=start."""
    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "test", "force_domain": "software_engineering"},
    ) as r:
        text = r.read().decode()

    events = _parse_sse(text)
    assert len(events) >= 1
    assert events[0]["type"] == "start"
    assert "routing_mode" in events[0]
    assert "primary_domain" in events[0]


def test_stream_emits_chunk_events(client, fake_swe_server):
    """Stream must contain at least one chunk event."""
    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "test", "force_domain": "software_engineering"},
    ) as r:
        text = r.read().decode()

    events = _parse_sse(text)
    chunks = [e for e in events if e.get("type") == "chunk"]
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "text" in chunk
        assert "index" in chunk
        assert isinstance(chunk["index"], int)


def test_stream_emits_done_event(client, fake_swe_server):
    """Last event must be type=done with full metadata."""
    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "test", "force_domain": "software_engineering"},
    ) as r:
        text = r.read().decode()

    events = _parse_sse(text)
    done_events = [e for e in events if e.get("type") == "done"]
    assert len(done_events) == 1

    done = done_events[0]
    assert "full_response" in done
    assert "u_score" in done
    assert "confidence" in done
    assert "latency_ms" in done
    assert "routing_mode" in done
    assert isinstance(done["u_score"], float)
    assert 0.0 <= done["u_score"] <= 1.0


def test_stream_event_order(client, fake_swe_server):
    """Events must arrive in order: start → chunk(s) → done."""
    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "test", "force_domain": "software_engineering"},
    ) as r:
        text = r.read().decode()

    events = _parse_sse(text)
    types = [e.get("type") for e in events]

    assert types[0] == "start"
    assert types[-1] == "done"
    # All middle events should be chunk
    middle = types[1:-1]
    assert all(t == "chunk" in ("chunk", "start", "done") for t in middle)


def test_stream_chunks_concatenate_to_response(client, fake_swe_server):
    """Concatenated chunk texts must equal full_response in done event."""
    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "test", "force_domain": "software_engineering"},
    ) as r:
        text = r.read().decode()

    events = _parse_sse(text)
    chunks = [e for e in events if e.get("type") == "chunk"]
    done = next(e for e in events if e.get("type") == "done")

    # Chunks concatenated should equal full_response
    concatenated = "".join(c["text"] for c in chunks)
    assert concatenated == done["full_response"]


def test_stream_sse_headers(client, fake_swe_server):
    """Stream response must include correct SSE headers."""
    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "test", "force_domain": "software_engineering"},
    ) as r:
        headers = r.headers
    assert headers.get("cache-control") == "no-cache"
    assert headers.get("x-accel-buffering") == "no"
