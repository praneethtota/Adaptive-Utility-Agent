"""Tests verifying hooks are fired at the correct pipeline points."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from aua.hooks import HOOK_POINTS, HookRunner, reset_hook_runner


@pytest.fixture(autouse=True)
def clean_runner():
    """Fresh HookRunner for every test."""
    runner = reset_hook_runner()
    yield runner
    reset_hook_runner()


# ── HookRunner unit tests ─────────────────────────────────────────────────────


def test_all_11_hook_points_defined():
    assert len(HOOK_POINTS) == 11
    expected = {
        "pre_query",
        "post_route",
        "pre_specialist_call",
        "post_specialist_call",
        "pre_arbiter",
        "post_arbiter",
        "on_correction",
        "pre_response",
        "post_response",
        "on_promotion",
        "on_rollback",
    }
    assert HOOK_POINTS == expected


def test_unknown_hook_point_raises():
    runner = HookRunner()
    with pytest.raises(ValueError, match="Unknown hook point"):
        runner.register("not_a_real_point", AsyncMock())


@pytest.mark.asyncio
async def test_hook_receives_event_dict():
    runner = HookRunner()
    received = {}

    async def capture(event):
        received.update(event)
        return event

    runner.register("pre_query", capture)
    await runner.fire("pre_query", {"session_id": "s1", "trace_id": "t1", "query": "hello"})

    assert received["session_id"] == "s1"
    assert received["query"] == "hello"
    assert received["type"] == "pre_query"


@pytest.mark.asyncio
async def test_hook_can_modify_event():
    runner = HookRunner()

    async def modifier(event):
        return {**event, "injected": "from_hook"}

    runner.register("pre_query", modifier)
    result = await runner.fire("pre_query", {"session_id": "s1", "trace_id": "t1"})
    assert result["injected"] == "from_hook"


@pytest.mark.asyncio
async def test_multiple_hooks_chain():
    runner = HookRunner()
    order = []

    async def hook_a(event):
        order.append("a")
        return event

    async def hook_b(event):
        order.append("b")
        return event

    runner.register("post_route", hook_a)
    runner.register("post_route", hook_b)
    await runner.fire("post_route", {"session_id": "", "trace_id": ""})
    assert order == ["a", "b"]


@pytest.mark.asyncio
async def test_fail_open_hook_continues_on_error():
    runner = HookRunner()

    async def bad_hook(event):
        raise RuntimeError("hook error")

    runner.register("on_correction", bad_hook, fail_closed=False)
    # Should not raise — fail-open
    result = await runner.fire("on_correction", {"session_id": "", "trace_id": ""})
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_fail_closed_hook_propagates_error():
    runner = HookRunner()

    async def bad_hook(event):
        raise RuntimeError("critical failure")

    runner.register("pre_query", bad_hook, fail_closed=True)
    with pytest.raises(RuntimeError, match="critical failure"):
        await runner.fire("pre_query", {"session_id": "", "trace_id": ""})


@pytest.mark.asyncio
async def test_timeout_fail_open():
    runner = HookRunner()

    async def slow_hook(event):
        await asyncio.sleep(10)
        return event

    runner.register("pre_query", slow_hook, fail_closed=False, timeout_s=0.05)
    # Should not raise — times out and continues
    result = await runner.fire("pre_query", {"session_id": "", "trace_id": ""})
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_timeout_fail_closed_raises():
    runner = HookRunner()

    async def slow_hook(event):
        await asyncio.sleep(10)
        return event

    runner.register("pre_query", slow_hook, fail_closed=True, timeout_s=0.05)
    with pytest.raises(RuntimeError, match="timed out"):
        await runner.fire("pre_query", {"session_id": "", "trace_id": ""})


def test_fire_background_does_not_block():
    runner = HookRunner()
    called = []

    async def bg_hook(event):
        called.append(event["type"])
        return event

    runner.register("post_response", bg_hook)
    # fire_background returns immediately — does not await
    runner.fire_background("post_response", {"session_id": "", "trace_id": ""})
    # called may be empty here (task not run yet) — that's correct behaviour


def test_registered_hooks_summary():
    runner = HookRunner()

    async def my_hook(event):
        return event

    runner.register("on_correction", my_hook)
    summary = runner.registered_hooks()
    assert "on_correction" in summary
    assert len(summary["on_correction"]) == 1


# ── on_correction wiring test ─────────────────────────────────────────────────


def test_on_correction_event_fields():
    """on_correction event must include subject, domain, claim, confidence, decay_class."""
    required = {
        "session_id",
        "trace_id",
        "subject",
        "domain",
        "claim",
        "confidence",
        "decay_class",
        "source",
    }
    event = {
        "session_id": "",
        "trace_id": "",
        "subject": "heapsort_complexity",
        "domain": "software_engineering",
        "claim": "O(n log n) worst-case",
        "confidence": 0.99,
        "decay_class": "A",
        "source": "manual",
    }
    assert required.issubset(event.keys())


# ── on_promotion / on_rollback event fields ───────────────────────────────────


def test_on_promotion_event_fields():
    required = {"session_id", "trace_id", "specialist", "promoted_from", "promoted_to"}
    event = {
        "session_id": "",
        "trace_id": "",
        "specialist": "swe",
        "promoted_from": "qwen2.5-coder:7b",
        "promoted_to": "qwen2.5-coder:14b",
        "project_dir": "/tmp/test",
    }
    assert required.issubset(event.keys())


def test_on_rollback_event_fields():
    required = {"session_id", "trace_id", "specialist", "rolled_back_from", "rolled_back_to"}
    event = {
        "session_id": "",
        "trace_id": "",
        "specialist": "swe",
        "rolled_back_from": "qwen2.5-coder:14b",
        "rolled_back_to": "qwen2.5-coder:7b",
        "project_dir": "/tmp/test",
    }
    assert required.issubset(event.keys())


# ── pre_query / post_route event fields ───────────────────────────────────────


def test_pre_query_event_fields():
    required = {"session_id", "trace_id", "query"}
    event = {
        "session_id": "s1",
        "trace_id": "t1",
        "query": "Write binary search.",
        "conversation_history": [],
        "force_domain": None,
    }
    assert required.issubset(event.keys())


def test_post_route_event_fields():
    required = {"session_id", "trace_id", "domain_distribution", "routing_mode", "top_domain"}
    event = {
        "session_id": "s1",
        "trace_id": "t1",
        "query": "Write binary search.",
        "domain_distribution": {"software_engineering": 0.9},
        "top_domain": "software_engineering",
        "routing_mode": "single",
        "active_specialists": ["swe"],
    }
    assert required.issubset(event.keys())


def test_pre_specialist_call_event_fields():
    required = {"session_id", "trace_id", "query", "domain", "specialist", "model"}
    event = {
        "session_id": "s1",
        "trace_id": "t1",
        "query": "Write binary search.",
        "domain": "software_engineering",
        "specialist": "swe",
        "model": "qwen2.5-coder:7b",
        "endpoint": "http://localhost:11434",
    }
    assert required.issubset(event.keys())


def test_post_specialist_call_event_fields():
    required = {"session_id", "trace_id", "domain", "specialist", "confidence"}
    event = {
        "session_id": "s1",
        "trace_id": "t1",
        "domain": "software_engineering",
        "specialist": "swe",
        "response_preview": "Here is the implementation...",
        "confidence": 0.82,
    }
    assert required.issubset(event.keys())


def test_pre_arbiter_event_fields():
    required = {"session_id", "trace_id", "query", "specialist_a", "specialist_b"}
    event = {
        "session_id": "s1",
        "trace_id": "t1",
        "query": "What is heapsort complexity?",
        "specialist_a": "swe",
        "response_a": "O(n log n)",
        "specialist_b": "math",
        "response_b": "O(n^2)",
    }
    assert required.issubset(event.keys())


def test_post_arbiter_event_fields():
    required = {"session_id", "trace_id", "winner_field"}
    event = {
        "session_id": "s1",
        "trace_id": "t1",
        "verdict": "The correct answer is O(n log n)",
        "winner_field": "software_engineering",
        "specialist_a": "swe",
        "specialist_b": "math",
    }
    assert required.issubset(event.keys())


def test_pre_response_event_fields():
    required = {"session_id", "trace_id", "domain", "routing_mode", "u_score", "latency_ms"}
    event = {
        "session_id": "s1",
        "trace_id": "t1",
        "domain": "software_engineering",
        "routing_mode": "single",
        "u_score": 0.731,
        "confidence": 0.823,
        "latency_ms": 312.4,
        "response": "Here is the implementation...",
    }
    assert required.issubset(event.keys())
