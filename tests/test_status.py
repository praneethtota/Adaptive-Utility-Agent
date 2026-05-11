"""
tests/test_status.py — aua/status.py unit tests.

Tests the dashboard rendering logic without needing a live router.
"""

import pytest

from aua.status import _fmt_uptime, _mini_bar, _render

# ── Helper function tests ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "0s"),
        (45, "45s"),
        (90, "1m 30s"),
        (3600, "1h 0m"),
        (3723, "1h 2m"),
        (7200, "2h 0m"),
    ],
)
def test_fmt_uptime(seconds, expected):
    assert _fmt_uptime(seconds) == expected


def test_mini_bar_full():
    """100% fraction should produce all filled blocks."""
    bar = _mini_bar(1.0, width=10)
    assert "█" * 10 == bar.plain


def test_mini_bar_empty():
    """0% fraction should produce all empty blocks."""
    bar = _mini_bar(0.0, width=10)
    assert "░" * 10 == bar.plain


def test_mini_bar_half():
    """50% fraction should produce half filled."""
    bar = _mini_bar(0.5, width=10)
    plain = bar.plain
    assert plain.count("█") == 5
    assert plain.count("░") == 5


def test_mini_bar_width():
    """Total width is always equal to the specified width."""
    for width in [10, 20, 30]:
        bar = _mini_bar(0.75, width=width)
        assert len(bar.plain) == width


# ── Render tests ──────────────────────────────────────────────────────────────

MOCK_STATUS = {
    "version": "0.6.0a0",
    "backend": "vllm",
    "uptime_s": 600,
    "health": {"swe": "ok", "math": "unreachable", "arbiter": "ok"},
    "latency": {"router": {"p50_ms": 310, "p95_ms": 540, "last_ms": 290, "samples": 10}},
    "memory": {"gpu0": "19200 / 24564 MiB"},
    "utility": {
        "software_engineering": {
            "mean_u": 0.633,
            "last_u": 0.641,
            "confidence": 0.72,
            "queries": 45,
        }
    },
    "routing": {
        "total_queries": 10,
        "by_mode": {"single": 8, "fanout": 1, "arbiter": 1},
    },
    "corrections": {
        "total_contradictions": 2,
        "dpo_pairs": 1,
        "assertions_stored": 3,
        "contradiction_rate": 0.2,
    },
    "arbiter_verdicts": {"case_1": 1, "case_2": 0, "case_3": 0, "case_4": 1},
}


def test_render_returns_panel():
    """_render returns a renderable Rich panel."""
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    result = _render(MOCK_STATUS, "http://localhost:8000")
    console.print(result)
    output = buf.getvalue()
    assert len(output) > 0


def test_render_shows_up_down():
    """Rendered output shows UP for healthy and DOWN for unreachable specialists."""
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    console.print(_render(MOCK_STATUS, "http://localhost:8000"))
    out = buf.getvalue()
    assert "UP" in out
    assert "DOWN" in out


def test_render_shows_utility_score():
    """Rendered output includes the mean utility score."""
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    console.print(_render(MOCK_STATUS, "http://localhost:8000"))
    out = buf.getvalue()
    assert "0.633" in out


def test_render_shows_memory():
    """Rendered output includes memory info."""
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    console.print(_render(MOCK_STATUS, "http://localhost:8000"))
    out = buf.getvalue()
    assert "19200" in out


def test_render_none_shows_error_panel():
    """_render(None, ...) shows an error panel when router is unreachable."""
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False)
    console.print(_render(None, "http://localhost:8000"))
    out = buf.getvalue()
    assert "not reachable" in out.lower() or "unreachable" in out.lower() or "✗" in out
