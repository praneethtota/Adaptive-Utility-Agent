"""
aua/status.py — Live terminal dashboard for aua status.

Polls GET /status on the running router every --interval seconds and
renders a Rich Live display covering:
  - Specialist health, latency, request counts
  - Memory: VRAM used/total (NVIDIA/AMD), unified memory (Apple Silicon), RAM (CPU/Ollama)
  - Routing mode breakdown (single / fanout / arbiter fallback)
  - Utility scores per domain (mean U, last U, query count)
  - Contradiction and DPO pair counts
  - Arbiter verdict distribution with mini bar charts

Usage:
    from aua.status import run_status
    run_status(router_url="http://localhost:8000", interval=2)

CLI:
    aua status
    aua status --config /path/to/aua_config.yaml
    aua status --interval 5
"""

from __future__ import annotations

import time

import httpx
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

STATUS_POLL_TIMEOUT = 3.0
_UP_STYLE = "bold green"
_DOWN_STYLE = "bold red"
_DIM = "dim"
_ACCENT = "bold cyan"


# ── Public API ────────────────────────────────────────────────────────────────


def run_status(
    router_url: str = "http://localhost:8000",
    interval: int = 2,
    once: bool = False,
    as_json: bool = False,
) -> None:
    """
    Poll the router's /status endpoint and render a live dashboard.

    Args:
        router_url: base URL of the AUA router
        interval:   refresh interval in seconds (ignored when once=True)
        once:       fetch once and exit (no loop)
        as_json:    print raw JSON and exit (implies once=True)
    """
    import json as _json

    if as_json or once:
        data = _fetch(router_url)
        if as_json:
            print(_json.dumps(data or {}, indent=2))
        else:
            console.print(_render(data, router_url))
        return

    console.print(f"[dim]Connecting to router at [cyan]{router_url}[/cyan]…[/dim]")

    with Live(
        _render(None, router_url),
        refresh_per_second=1,
        screen=False,
        console=console,
    ) as live:
        try:
            while True:
                data = _fetch(router_url)
                live.update(_render(data, router_url))
                time.sleep(interval)
        except KeyboardInterrupt:
            pass

    console.print("[dim]aua status stopped.[/dim]")


# ── Fetch ─────────────────────────────────────────────────────────────────────


def _fetch(router_url: str) -> dict | None:
    """Fetch /status; return None if unreachable."""
    try:
        with httpx.Client(timeout=STATUS_POLL_TIMEOUT) as client:
            r = client.get(f"{router_url}/status")
            if r.status_code == 200:
                return r.json()
            return None
    except Exception:
        return None


# ── Render ────────────────────────────────────────────────────────────────────


def _render(data: dict | None, router_url: str):
    """Build the full dashboard from a /status response (or an error panel)."""
    ts = time.strftime("%H:%M:%S")

    if data is None:
        return Panel(
            Text.assemble(
                (f"  Router at {router_url} is not reachable.\n", "red"),
                ("  Is aua serve running?\n\n", _DIM),
                ("  aua serve --dry-run   ", "white"),
                ("# preview startup commands\n", _DIM),
                ("  aua serve             ", "white"),
                ("# start the framework\n", _DIM),
            ),
            title=f"[bold red]✗ aua status[/bold red]  [dim]{ts}[/dim]",
            border_style="red",
            padding=(0, 1),
        )

    uptime = _fmt_uptime(data.get("uptime_s", 0))
    backend = data.get("backend", "?")
    version = data.get("version", "?")

    sections = [
        _specialists_table(data),
        _routing_table(data),
        _utility_table(data),
        _corrections_table(data),
        _arbiter_table(data),
    ]

    # Stack all sections, filtered to non-None
    from rich.console import Group as RichGroup

    body = RichGroup(*[s for s in sections if s is not None])

    return Panel(
        body,
        title=(
            f"[bold]aua status[/bold]  v{version}  "
            f"[dim]{backend} · uptime {uptime} · {ts}[/dim]"
        ),
        subtitle="[dim]Ctrl+C to stop[/dim]",
        border_style="blue",
        padding=(0, 1),
    )


# ── Section builders ──────────────────────────────────────────────────────────


def _specialists_table(data: dict):
    health = data.get("health", {})
    latency = data.get("latency", {})
    memory = data.get("memory", {})

    t = Table(
        title="Specialists",
        box=box.SIMPLE_HEAD,
        title_style="bold",
        title_justify="left",
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
    )
    t.add_column("Name", style=_ACCENT, min_width=10)
    t.add_column("Status", min_width=8)
    t.add_column("p50 ms", justify="right", min_width=8)
    t.add_column("p95 ms", justify="right", min_width=8)
    t.add_column("Requests", justify="right", min_width=9)
    t.add_column("Memory", min_width=22)

    for name, status in health.items():
        ok = status == "ok"
        status_str = Text("● UP", style=_UP_STYLE) if ok else Text("○ DOWN", style=_DOWN_STYLE)
        lat = latency.get(name, {})
        p50 = f"{lat['p50_ms']:.0f}" if lat.get("p50_ms") is not None else "—"
        p95 = f"{lat['p95_ms']:.0f}" if lat.get("p95_ms") is not None else "—"
        samples = str(lat.get("samples", 0))

        # Memory: try gpu0 first, then system, then first available key
        vram_str = memory.get("gpu0") or memory.get("system") or next(iter(memory.values()), "—")

        t.add_row(name, status_str, p50, p95, samples, Text(str(vram_str), style=_DIM))

    return t


def _routing_table(data: dict) -> Table:
    routing = data.get("routing", {})
    total = routing.get("total_queries", 0)
    modes = routing.get("by_mode", {})

    t = Table(
        title="Routing",
        box=box.SIMPLE_HEAD,
        title_style="bold",
        title_justify="left",
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
    )
    t.add_column("Mode", min_width=20)
    t.add_column("Count", justify="right", min_width=7)
    t.add_column("Share", justify="right", min_width=7)
    t.add_column("", min_width=22)

    for mode, label in [
        ("single", "Single domain"),
        ("fanout", "Fan-out (cross-domain)"),
        ("arbiter", "Arbiter fallback"),
    ]:
        count = modes.get(mode, 0)
        pct = count / total * 100 if total > 0 else 0.0
        bar = _mini_bar(pct / 100, width=20)
        t.add_row(label, str(count), f"{pct:.1f}%", bar)

    t.add_row(
        Text("Total queries", style="bold"),
        Text(str(total), style="bold"),
        "",
        "",
    )
    return t


def _utility_table(data: dict) -> Table | None:
    utility = data.get("utility", {})
    if not utility:
        return None

    t = Table(
        title="Utility Scores",
        box=box.SIMPLE_HEAD,
        title_style="bold",
        title_justify="left",
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
    )
    t.add_column("Domain", min_width=22)
    t.add_column("Mean U", justify="right", min_width=8)
    t.add_column("Last U", justify="right", min_width=8)
    t.add_column("Confidence", justify="right", min_width=11)
    t.add_column("Queries", justify="right", min_width=8)

    for domain, stats in utility.items():
        mean_u = f"{stats['mean_u']:.4f}" if stats.get("mean_u") is not None else "—"
        last_u = f"{stats['last_u']:.4f}" if stats.get("last_u") is not None else "—"
        conf = f"{stats['confidence']:.4f}" if stats.get("confidence") is not None else "—"
        q = str(stats.get("queries", 0))
        t.add_row(domain, mean_u, last_u, conf, q)

    return t


def _corrections_table(data: dict) -> Table:
    corr = data.get("corrections", {})

    t = Table(
        title="Corrections",
        box=box.SIMPLE_HEAD,
        title_style="bold",
        title_justify="left",
        show_header=False,
        padding=(0, 1),
    )
    t.add_column("Metric", min_width=28)
    t.add_column("Value", justify="right", min_width=8)

    contra = corr.get("total_contradictions", 0)
    rate = corr.get("contradiction_rate", 0.0)

    t.add_row("Contradictions detected", str(contra))
    t.add_row("DPO pairs accumulated", str(corr.get("dpo_pairs", 0)))
    t.add_row("Assertions stored", str(corr.get("assertions_stored", 0)))
    t.add_row(
        "Contradiction rate",
        Text(
            f"{rate*100:.1f}%", style="red" if rate > 0.3 else "yellow" if rate > 0.1 else "green"
        ),
    )
    return t


def _arbiter_table(data: dict) -> Table | None:
    verdicts = data.get("arbiter_verdicts", {})
    total = sum(verdicts.values())
    if total == 0:
        return None

    t = Table(
        title="Arbiter Verdicts",
        box=box.SIMPLE_HEAD,
        title_style="bold",
        title_justify="left",
        show_header=True,
        header_style="bold dim",
        padding=(0, 1),
    )
    t.add_column("Case", min_width=24)
    t.add_column("Count", justify="right", min_width=6)
    t.add_column("Share", justify="right", min_width=7)
    t.add_column("", min_width=20)

    labels = {
        "case_1": "Case 1 — A correct, B wrong",
        "case_2": "Case 2 — B correct, A wrong",
        "case_3": "Case 3 — both wrong",
        "case_4": "Case 4 — inconclusive (escalate)",
    }
    styles = {"case_1": "green", "case_2": "green", "case_3": "yellow", "case_4": "red"}
    for key, label in labels.items():
        count = verdicts.get(key, 0)
        pct = count / total * 100 if total > 0 else 0.0
        bar = _mini_bar(pct / 100, width=20)
        t.add_row(
            Text(label, style=styles[key]),
            str(count),
            f"{pct:.0f}%",
            bar,
        )
    t.add_row(Text("Total", style="bold"), Text(str(total), style="bold"), "", "")
    return t


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mini_bar(fraction: float, width: int = 20) -> Text:
    """Render a mini horizontal bar: ██████░░░░░░░░  (filled / empty)."""
    filled = max(0, min(width, round(fraction * width)))
    bar = "█" * filled + "░" * (width - filled)
    pct = fraction * 100
    style = "green" if pct >= 66 else "yellow" if pct >= 33 else "red"
    return Text(bar, style=style)


def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m {s%60}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m}m"
