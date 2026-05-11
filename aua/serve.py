"""
aua/serve.py — Core logic for `aua serve`.

Starts all specialist servers sequentially (one at a time, waiting for each
to become healthy before starting the next), then starts the FastAPI router.

Sequential startup is required: parallel vLLM startup causes CUDA graph
profiling conflicts on single-GPU setups. Measured on RTX 4090: sequential
with --enforce-eager is reliable; parallel without it fails.

Usage (programmatic):
    from aua.serve import serve
    serve(config, dry_run=False, startup_timeout=120)

Usage (CLI — preferred):
    aua serve
    aua serve --config /path/to/aua_config.yaml
    aua serve --dry-run
    aua serve --no-router
    aua serve --router-only
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aua.config import ArbiterConfig, AUAConfig, SpecialistConfig

console = Console()

# How long to poll a server before giving up (seconds)
DEFAULT_STARTUP_TIMEOUT = 120
# Delay between poll attempts
POLL_INTERVAL = 3.0
# Extra sleep after a server reports healthy before starting the next one
# (vLLM needs a moment to finish model loading after /v1/models responds)
POST_HEALTH_SLEEP = 5.0


# ── Public API ────────────────────────────────────────────────────────────────


def serve(
    config: AUAConfig,
    dry_run: bool = False,
    no_router: bool = False,
    router_only: bool = False,
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
) -> None:
    """
    Start all specialists and the router from config.

    Args:
        config:          loaded AUAConfig
        dry_run:         print commands without executing
        no_router:       start specialists only, skip FastAPI router
        router_only:     skip specialists, only start FastAPI router
        startup_timeout: seconds to wait for each specialist to become healthy
    """
    _print_banner(config, dry_run, no_router, router_only)

    processes: list[subprocess.Popen] = []

    # Register cleanup on Ctrl+C and SIGTERM
    def _shutdown(sig=None, frame=None):
        console.print("\n[yellow]⚡ Shutting down...[/yellow]")
        for p in processes:
            if p.poll() is None:
                p.terminate()
        # Give them 5s to exit gracefully, then kill
        time.sleep(2)
        for p in processes:
            if p.poll() is None:
                p.kill()
        console.print("[yellow]All processes stopped.[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start specialists ──────────────────────────────────────────────────
    if not router_only:
        if config.backend == "ollama":
            p = _start_ollama(config, dry_run, startup_timeout)
            if p:
                processes.append(p)
        else:
            for spec in config.specialists:
                p = _start_specialist(spec, dry_run, startup_timeout)
                if p:
                    processes.append(p)

            # Arbiter
            p = _start_arbiter(config.arbiter, dry_run, startup_timeout)
            if p:
                processes.append(p)

    # ── Start router ───────────────────────────────────────────────────────
    if not no_router:
        _start_router(config, dry_run, processes)

    # ── dry_run exits here ─────────────────────────────────────────────────
    if dry_run:
        console.print("\n[dim]Dry run complete. No processes started.[/dim]")
        return

    # ── Wait for all child processes (router runs in-process via uvicorn) ──
    # If we reach here without starting uvicorn (--no-router), just wait.
    if no_router:
        console.print(
            "\n[green]All specialists running.[/green] " "[dim]Press Ctrl+C to stop.[/dim]"
        )
        try:
            while True:
                # Check any specialist died unexpectedly
                for p in processes:
                    if p.poll() is not None:
                        console.print(
                            f"\n[red]A specialist process exited unexpectedly "
                            f"(pid={p.pid}, code={p.returncode}).[/red]"
                        )
                        _shutdown()
                time.sleep(5)
        except KeyboardInterrupt:
            _shutdown()


# ── Specialist startup ────────────────────────────────────────────────────────


def _start_specialist(
    spec: SpecialistConfig,
    dry_run: bool,
    timeout: int,
) -> subprocess.Popen | None:
    cmd = spec.vllm_command()

    hw_detail = (
        f"GPU {spec.gpu} ({spec.gpu_memory_utilization*100:.0f}% VRAM)"
        if spec.backend == "vllm"
        else "Ollama"
    )
    console.print(
        f"\n[bold]Starting specialist:[/bold] [cyan]{spec.name}[/cyan]  "
        f"[dim]{spec.field} · port {spec.port} · {hw_detail}[/dim]"
    )
    console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")

    if dry_run:
        return None

    p = subprocess.Popen(
        cmd,
        env=_build_env(spec.gpu, spec.backend),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _wait_healthy(spec.name, spec.models_url, p, timeout)
    return p


def _start_arbiter(
    arb: ArbiterConfig,
    dry_run: bool,
    timeout: int,
) -> subprocess.Popen | None:
    cmd = arb.vllm_command()

    hw_detail_arb = (
        f"GPU {arb.gpu} ({arb.gpu_memory_utilization*100:.0f}% VRAM)"
        if arb.backend == "vllm"
        else "Ollama"
    )
    console.print(
        f"\n[bold]Starting arbiter:[/bold] [magenta]arbiter[/magenta]  "
        f"[dim]port {arb.port} · {hw_detail_arb}[/dim]"
    )
    console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")

    if dry_run:
        return None

    p = subprocess.Popen(
        cmd,
        env=_build_env(arb.gpu, arb.backend),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _wait_healthy("arbiter", arb.models_url, p, timeout)
    return p


def _wait_healthy(
    name: str,
    url: str,
    proc: subprocess.Popen,
    timeout: int,
) -> None:
    """Poll url until HTTP 200 or timeout. Raises SystemExit on failure."""
    deadline = time.time() + timeout
    attempt = 0

    with console.status(
        f"[dim]Waiting for [cyan]{name}[/cyan] to become healthy " f"(timeout {timeout}s)...[/dim]",
        spinner="dots",
    ) as status:
        while time.time() < deadline:
            # Check if process died while we were waiting
            if proc.poll() is not None:
                stderr = ""
                try:
                    if proc.stderr is not None:
                        stderr = proc.stderr.read().decode(errors="replace")[-500:]
                except Exception:
                    pass
                console.print(
                    f"\n[red]✗ {name} exited unexpectedly "
                    f"(code={proc.returncode}).[/red]\n"
                    f"[dim]{stderr}[/dim]"
                )
                sys.exit(1)

            try:
                with httpx.Client(timeout=3.0) as client:
                    r = client.get(url)
                    if r.status_code == 200:
                        elapsed = timeout - (deadline - time.time())
                        console.print(
                            f"  [green]✓ {name} healthy[/green] " f"[dim]({elapsed:.0f}s)[/dim]"
                        )
                        time.sleep(POST_HEALTH_SLEEP)
                        return
            except Exception:
                pass

            attempt += 1
            elapsed_s = attempt * POLL_INTERVAL
            status.update(
                f"[dim]Waiting for [cyan]{name}[/cyan] " f"({elapsed_s:.0f}s / {timeout}s)...[/dim]"
            )
            time.sleep(POLL_INTERVAL)

    console.print(
        f"\n[red]✗ {name} did not become healthy within {timeout}s.[/red]\n"
        f"[dim]Check that the model path is correct and the hardware has enough memory.[/dim]"
    )
    proc.terminate()
    sys.exit(1)


# ── Ollama startup ────────────────────────────────────────────────────────────


def _start_ollama(
    config: AUAConfig,
    dry_run: bool,
    timeout: int,
) -> subprocess.Popen | None:
    """
    Ensure Ollama is running and all required models are pulled.

    Steps:
        1. Check `ollama` binary is in PATH
        2. Start `ollama serve` if not already reachable
        3. Pull each model that is not yet present
    """
    ollama_url = f"http://localhost:{config.arbiter.port}/api/tags"

    console.print(f"\n[bold]Backend: Ollama[/bold]  [dim]port {config.arbiter.port}[/dim]")

    # ── Check binary ──────────────────────────────────────────────────────
    if not dry_run and shutil.which("ollama") is None:
        console.print(
            "\n[red]✗ 'ollama' not found in PATH.[/red]\n"
            "[dim]Install with: brew install ollama[/dim]\n"
            "[dim]Then run: ollama serve[/dim]"
        )
        sys.exit(1)
    else:
        console.print("  [dim]$ ollama serve  (if not already running)[/dim]")

    if dry_run:
        all_models = [s.model for s in config.specialists] + [config.arbiter.model]
        for m in all_models:
            console.print(f"  [dim]$ ollama pull {m}[/dim]")
        return None

    # ── Start ollama serve if not reachable ───────────────────────────────
    proc = None
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(ollama_url)
            if r.status_code == 200:
                console.print("  [green]✓ Ollama already running[/green]")
    except Exception:
        console.print("  Starting ollama serve...")
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_healthy("ollama", ollama_url, proc, timeout)

    # ── Pull models ───────────────────────────────────────────────────────
    all_models = list(dict.fromkeys([s.model for s in config.specialists] + [config.arbiter.model]))
    for model in all_models:
        _ollama_pull(model)

    return proc


def _ollama_pull(model: str) -> None:
    """Pull an Ollama model if not already present."""
    console.print(f"  Checking model [cyan]{model}[/cyan]...")
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )
    if model in result.stdout:
        console.print(f"  [green]✓ {model} already pulled[/green]")
        return

    console.print(f"  Pulling [cyan]{model}[/cyan] (this may take a while)...")
    pull = subprocess.run(
        ["ollama", "pull", model],
        capture_output=False,  # show progress to terminal
    )
    if pull.returncode != 0:
        console.print(f"  [red]✗ Failed to pull {model}[/red]")
        sys.exit(1)
    console.print(f"  [green]✓ {model} ready[/green]")


# ── Router startup ────────────────────────────────────────────────────────────


def _start_router(
    config: AUAConfig,
    dry_run: bool,
    specialist_procs: list[subprocess.Popen],
) -> None:
    """Start the FastAPI router with uvicorn (runs in the current process)."""
    import uvicorn

    from aua.router import Router

    host = config.router.host
    port = config.router.port

    console.print(f"\n[bold]Starting router[/bold]  " f"[dim]http://{host}:{port}[/dim]")

    if dry_run:
        console.print(f"  [dim]$ uvicorn aua.router:app --host {host} --port {port}[/dim]")
        return

    # Print the ready panel
    _print_ready(config)

    router = Router.from_config(config)

    # uvicorn.run blocks until Ctrl+C
    uvicorn.run(
        router.app,
        host=host,
        port=port,
        log_level=config.logging.level.lower(),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_env(gpu_index: int, backend: str = "vllm") -> dict:
    """Build environment, setting CUDA_VISIBLE_DEVICES for vLLM/ROCm only."""
    env = os.environ.copy()
    if backend == "vllm":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # Ollama / CPU: don't set CUDA_VISIBLE_DEVICES — Ollama manages devices itself
    return env


def _print_banner(
    config: AUAConfig,
    dry_run: bool,
    no_router: bool,
    router_only: bool,
) -> None:
    mode_tags = []
    if dry_run:
        mode_tags.append("[yellow]DRY RUN[/yellow]")
    if no_router:
        mode_tags.append("[dim]--no-router[/dim]")
    if router_only:
        mode_tags.append("[dim]--router-only[/dim]")
    mode_str = "  " + "  ".join(mode_tags) if mode_tags else ""

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Component", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Port", justify="right")
    table.add_column("GPU", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Field")

    for s in config.specialists:
        table.add_row(
            s.name,
            s.model.split("/")[-1],
            str(s.port),
            str(s.gpu),
            f"{s.gpu_memory_utilization*100:.0f}%" if s.backend == "vllm" else "—",
            s.field,
        )
    table.add_row(
        "arbiter",
        config.arbiter.model.split("/")[-1],
        str(config.arbiter.port),
        str(config.arbiter.gpu),
        (
            f"{config.arbiter.gpu_memory_utilization*100:.0f}%"
            if config.arbiter.backend == "vllm"
            else "—"
        ),
        "general",
    )
    table.add_row(
        "[bold green]router[/bold green]",
        "[dim]FastAPI + uvicorn[/dim]",
        str(config.router.port),
        "—",
        "—",
        "—",
    )

    console.print(
        Panel(
            table,
            title=f"[bold]aua serve[/bold]  v{config.version}{mode_str}",
            subtitle=(
                "[dim]Sequential startup · --enforce-eager · Ctrl+C to stop[/dim]"
                if config.backend == "vllm"
                else "[dim]Ollama backend · Ctrl+C to stop[/dim]"
            ),
            border_style="blue",
            padding=(0, 1),
        )
    )


def _print_ready(config: AUAConfig) -> None:
    host = config.router.host
    display_host = "localhost" if host == "0.0.0.0" else host
    port = config.router.port

    lines = Text()
    lines.append(f"  POST  http://{display_host}:{port}/query\n", style="green")
    lines.append(f"  GET   http://{display_host}:{port}/health\n", style="dim")
    lines.append(f"  GET   http://{display_host}:{port}/stats\n", style="dim")
    lines.append(f"  GET   http://{display_host}:{port}/docs\n", style="dim")

    console.print(
        Panel(
            lines,
            title="[bold green]✓ All specialists ready — router starting[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )
