"""
aua/serve.py — Core logic for `aua serve`.

Starts all specialist servers sequentially (one at a time, waiting for each
to become healthy before starting the next), then starts the FastAPI router.

Sequential startup is required: parallel vLLM startup causes CUDA graph
profiling conflicts on single-GPU setups. Measured on RTX 4090: sequential
with --enforce-eager is reliable; parallel without it fails.

Foreground-only: aua serve runs in the foreground and blocks until Ctrl+C
or SIGTERM. Use a process supervisor (systemd, supervisor, screen) for
background/daemon operation.

Usage (programmatic):
    from aua.serve import serve
    serve(config, dry_run=False, startup_timeout=120)

Usage (CLI — preferred):
    aua serve
    aua serve --config /path/to/aua_config.yaml
    aua serve --dry-run
    aua serve --no-router
    aua serve --router-only
    aua serve --reuse-running   # skip port-conflict check
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import httpx
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aua.config import ArbiterConfig, AUAConfig, RuntimeConfig, SpecialistConfig

console = Console()

# How long to poll a server before giving up (seconds)
DEFAULT_STARTUP_TIMEOUT = 120
# Delay between poll attempts
POLL_INTERVAL = 3.0
# Extra sleep after a server reports healthy before starting the next one
# (vLLM needs a moment to finish model loading after /v1/models responds)
POST_HEALTH_SLEEP = 5.0
# Grace period for SIGTERM before escalating to SIGKILL (seconds)
SHUTDOWN_GRACE_SECONDS = 15


# ── Public API ────────────────────────────────────────────────────────────────


def serve(
    config: AUAConfig,
    dry_run: bool = False,
    no_router: bool = False,
    router_only: bool = False,
    startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
    reuse_running: bool = False,
    config_path: str | None = None,
    no_download: bool = False,
) -> None:
    """
    Start all specialists and the router from config.

    Runs in the foreground — blocks until Ctrl+C or SIGTERM.
    For daemon/background operation, wrap with systemd or supervisord.

    Args:
        config:          loaded AUAConfig
        dry_run:         print commands without executing; always exits 0
        no_router:       start specialists only, skip FastAPI router
        router_only:     skip specialists, only start FastAPI router
        startup_timeout: seconds to wait for each specialist to become healthy
        reuse_running:   skip port-conflict check (use when services are already up)
    """
    _print_banner(config, dry_run, no_router, router_only)

    # ── Ensure runtime directories ─────────────────────────────────────────
    if not dry_run:
        config.runtime.ensure()

    # ── Port-conflict check ────────────────────────────────────────────────
    if not dry_run and not reuse_running and not router_only:
        _check_ports(config)

    processes: list[subprocess.Popen] = []

    # ── Register graceful shutdown ─────────────────────────────────────────
    def _shutdown(sig=None, frame=None) -> None:
        console.print("\n[yellow]⚡ Shutting down...[/yellow]")
        # Phase 1: SIGTERM — ask nicely
        for p in processes:
            if p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass
        # Phase 2: wait up to SHUTDOWN_GRACE_SECONDS
        deadline = time.time() + SHUTDOWN_GRACE_SECONDS
        while time.time() < deadline:
            if all(p.poll() is not None for p in processes):
                break
            time.sleep(0.5)
        # Phase 3: SIGKILL — force remaining processes
        for p in processes:
            if p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass
        # Remove stale PID files
        if not dry_run:
            _remove_pid_files(config)
        console.print("[yellow]All processes stopped.[/yellow]")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start specialists ──────────────────────────────────────────────────
    if not router_only:
        if config.backend == "ollama":
            p = _start_ollama(config, dry_run, startup_timeout, no_download=no_download)
            if p:
                processes.append(p)
        else:
            for spec in config.specialists:
                p = _start_specialist(
                    spec, dry_run, startup_timeout, config.runtime, no_download=no_download
                )
                if p:
                    processes.append(p)

            # Arbiter
            p = _start_arbiter(
                config.arbiter, dry_run, startup_timeout, config.runtime, no_download=no_download
            )
            if p:
                processes.append(p)

    # ── Start router ───────────────────────────────────────────────────────
    if not no_router:
        _start_router(config, dry_run, processes, config_path=config_path)

    # ── dry_run exits here ─────────────────────────────────────────────────
    if dry_run:
        console.print("\n[dim]Dry run complete. No processes started.[/dim]")
        return  # exit 0

    # ── Wait for all child processes (router runs in-process via uvicorn) ──
    # If we reach here without starting uvicorn (--no-router), just wait.
    if no_router:
        console.print(
            "\n[green]All specialists running.[/green] " "[dim]Press Ctrl+C to stop.[/dim]"
        )
        try:
            while True:
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


# ── Port-conflict detection ───────────────────────────────────────────────────


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if `port` is already bound on `host`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError):
            return False


def _check_ports(config: AUAConfig) -> None:
    """
    Verify all required ports are free before starting any process.
    Exits with a clear error if a conflict is detected.

    Skip with --reuse-running if services are already running (e.g. after a crash).
    """
    conflicts = []
    # For Ollama backend, specialists share port 11434 with the Ollama server.
    # That port is intentionally already in use — skip specialist port checks.
    if config.backend == "ollama":
        services = [("router", config.router.port)]
    else:
        services = [(s.name, s.port) for s in config.specialists]
        services.append(("arbiter", config.arbiter.port))
        services.append(("router", config.router.port))

    for name, port in services:
        if _port_in_use(port):
            conflicts.append((name, port))

    if conflicts:
        lines = "\n".join(f"  {name}: port {port} already in use" for name, port in conflicts)
        console.print(
            f"\n[red]✗ Port conflict detected:[/red]\n{lines}\n\n"
            "[dim]Options:\n"
            "  --reuse-running   skip this check (if services are already up)\n"
            "  Change ports in aua_config.yaml\n"
            "  Kill the conflicting processes[/dim]"
        )
        sys.exit(1)


# ── PID file helpers ──────────────────────────────────────────────────────────


def _write_pid_file(name: str, pid: int, runtime: RuntimeConfig) -> None:
    """Write PID to .aua/pids/{name}.pid"""
    try:
        pid_path = runtime.pids / f"{name}.pid"
        pid_path.write_text(str(pid))
    except OSError:
        pass  # non-fatal — PID files are best-effort


def _remove_pid_files(config: AUAConfig) -> None:
    """Remove all PID files for this config's services."""
    names = [s.name for s in config.specialists] + ["arbiter", "router"]
    for name in names:
        try:
            pid_path = config.runtime.pids / f"{name}.pid"
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass


# ── Log file helpers ──────────────────────────────────────────────────────────


def _open_log(name: str, runtime: RuntimeConfig):
    """Open a log file for service `name` under .aua/logs/. Returns file handle."""
    try:
        log_path = runtime.logs / f"{name}.log"
        return open(log_path, "ab")  # append binary — preserves prior runs
    except OSError:
        return subprocess.DEVNULL


# ── Specialist startup ────────────────────────────────────────────────────────


def _start_specialist(
    spec: SpecialistConfig,
    dry_run: bool,
    timeout: int,
    runtime: RuntimeConfig,
    no_download: bool = False,
) -> subprocess.Popen | None:
    cmd = spec.vllm_command()

    # #46/#57: resolve version-pinned / MLflow model refs, then download
    if not dry_run and spec.backend == "vllm":
        from aua.model_registry import parse_model_ref, resolve_model_ref

        ref = parse_model_ref(spec.model)
        if ref.revision is not None or (ref.mlflow_uri is not None):
            # Has a pinned revision or MLflow URI — resolve to local path
            try:
                resolved = resolve_model_ref(
                    spec.model, mlflow_tracking_uri=spec.mlflow_tracking_uri
                )
                if resolved != spec.model:
                    console.print(f"  [dim]Resolved [cyan]{spec.model}[/cyan] → {resolved}[/dim]")
                    spec = dataclasses.replace(spec, model=resolved)
            except Exception as _reg_err:
                console.print(f"  [red]✗ Model registry error:[/red] {_reg_err}")
                sys.exit(1)
        if not no_download:
            _hf_download(spec.model)

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

    log_fh = _open_log(spec.name, runtime)
    p = subprocess.Popen(
        cmd,
        env=_build_env(spec.gpu, spec.backend, gpu_ids=spec.gpu_ids),
        stdout=log_fh,
        stderr=log_fh,
    )
    _write_pid_file(spec.name, p.pid, runtime)
    console.print(f"  [dim]pid={p.pid}  log={runtime.logs / (spec.name + '.log')}[/dim]")
    _wait_healthy(spec.name, spec.models_url, p, timeout)
    return p


def _start_arbiter(
    arb: ArbiterConfig,
    dry_run: bool,
    timeout: int,
    runtime: RuntimeConfig,
    no_download: bool = False,
) -> subprocess.Popen | None:
    cmd = arb.vllm_command()

    # #46/#57: resolve version-pinned / MLflow model refs, then download
    if not dry_run and arb.backend == "vllm":
        from aua.model_registry import parse_model_ref, resolve_model_ref

        ref = parse_model_ref(arb.model)
        if ref.revision is not None or (ref.mlflow_uri is not None):
            try:
                resolved = resolve_model_ref(arb.model, mlflow_tracking_uri=arb.mlflow_tracking_uri)
                if resolved != arb.model:
                    console.print(f"  [dim]Resolved [cyan]{arb.model}[/cyan] → {resolved}[/dim]")
                    arb = dataclasses.replace(arb, model=resolved)
            except Exception as _reg_err:
                console.print(f"  [red]✗ Model registry error:[/red] {_reg_err}")
                sys.exit(1)
        if not no_download:
            _hf_download(arb.model)

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

    log_fh = _open_log("arbiter", runtime)
    p = subprocess.Popen(
        cmd,
        env=_build_env(arb.gpu, arb.backend, gpu_ids=arb.gpu_ids),
        stdout=log_fh,
        stderr=log_fh,
    )
    _write_pid_file("arbiter", p.pid, runtime)
    console.print(f"  [dim]pid={p.pid}  log={runtime.logs / 'arbiter.log'}[/dim]")
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
            if proc.poll() is not None:
                console.print(
                    f"\n[red]✗ {name} exited unexpectedly " f"(code={proc.returncode}).[/red]"
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
    no_download: bool = False,
) -> subprocess.Popen | None:
    """
    Ensure Ollama is running and all required models are pulled.

    Steps:
        1. Check `ollama` binary is in PATH
        2. Start `ollama serve` if not already reachable
        3. Pull each model that is not yet present
    """
    ollama_url = f"http://{config.arbiter.host}:{config.arbiter.port}/api/tags"

    console.print(f"\n[bold]Backend: Ollama[/bold]  [dim]port {config.arbiter.port}[/dim]")

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

    if not no_download:
        all_models = list(
            dict.fromkeys([s.model for s in config.specialists] + [config.arbiter.model])
        )
        for model in all_models:
            _ollama_pull(model)

    return proc


def _ollama_model_present(model: str, base_url: str = "http://127.0.0.1:11434") -> bool:
    """Return True if the model tag is already present in Ollama."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        tags = r.json().get("models", [])
        return any(m.get("name", "") == model or m.get("name", "").startswith(model) for m in tags)
    except Exception:
        return False


def _ollama_pull(model: str, base_url: str = "http://127.0.0.1:11434") -> None:
    """Pull an Ollama model if not already present, with Rich progress display."""
    if _ollama_model_present(model, base_url):
        console.print(f"  [green]✓ {model} already present[/green]")
        return

    console.print(f"  Pulling [cyan]{model}[/cyan]...")
    pull = subprocess.run(["ollama", "pull", model])
    if pull.returncode != 0:
        console.print(f"  [red]✗ Failed to pull {model}[/red]")
        sys.exit(1)
    console.print(f"  [green]✓ {model} ready[/green]")


def _hf_download(model_repo: str) -> None:
    """
    Download a HuggingFace model with progress display (#57).

    Steps:
      1. Check HF_TOKEN env var (required for gated models, optional otherwise)
      2. Check local cache — skip download if already present
      3. Pre-flight disk space check (~5 GB minimum as a heuristic)
      4. snapshot_download() with Rich progress
    """
    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
    except ImportError:
        console.print(
            "  [yellow]⚠ huggingface_hub not installed — skipping download.[/yellow]\n"
            "  [dim]Install with: pip install huggingface_hub[/dim]"
        )
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Check cache first (free — no network required)
    cached = try_to_load_from_cache(model_repo, filename="config.json")
    if cached and cached != "not_cached_path_but_exists":
        console.print(f"  [green]✓ {model_repo} already in HF cache[/green]")
        return

    # Disk space heuristic — warn if < 10 GB free
    import shutil as _shutil

    free_gb = _shutil.disk_usage("/").free / 1024**3
    if free_gb < 10:
        console.print(
            f"  [yellow]⚠ Low disk space ({free_gb:.1f} GB free). "
            "Download may fail if the model is large.[/yellow]"
        )

    if not token:
        console.print(
            f"  [dim]HF_TOKEN not set — downloading {model_repo} as anonymous user.\n"
            "  Gated models (Llama, Gemma) require: export HF_TOKEN=hf_...[/dim]"
        )

    console.print(f"  Downloading [cyan]{model_repo}[/cyan] from HuggingFace Hub...")
    try:
        with console.status(
            f"  [dim]Downloading {model_repo} (this may take several minutes)...[/dim]",
            spinner="dots",
        ):
            snapshot_download(
                repo_id=model_repo,
                token=token,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
        console.print(f"  [green]✓ {model_repo} downloaded[/green]")
    except Exception as e:
        if "401" in str(e) or "403" in str(e):
            console.print(
                f"  [red]✗ Access denied for {model_repo}.[/red]\n"
                "  [dim]This is a gated model. Set HF_TOKEN=hf_... and accept the "
                f"model's terms at https://huggingface.co/{model_repo}[/dim]"
            )
        else:
            console.print(f"  [red]✗ Download failed for {model_repo}: {e}[/red]")
        sys.exit(1)


# ── Router startup ────────────────────────────────────────────────────────────


def _start_router(
    config: AUAConfig,
    dry_run: bool,
    specialist_procs: list[subprocess.Popen],
    config_path: str | None = None,
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

    _print_ready(config)
    _write_pid_file("router", os.getpid(), config.runtime)

    router = Router.from_config(config, config_path=config_path)
    uvicorn.run(
        router.app,
        host=host,
        port=port,
        log_level=config.logging.level.lower(),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_env(
    gpu_index: int,
    backend: str = "vllm",
    gpu_ids: list[int] | None = None,
) -> dict:
    """
    Build environment, setting CUDA_VISIBLE_DEVICES for vLLM/ROCm only (#66).

    Single-GPU (default):
        CUDA_VISIBLE_DEVICES=<gpu_index>
        vLLM sees one GPU remapped to index 0.

    Multi-GPU tensor/pipeline parallel:
        CUDA_VISIBLE_DEVICES=0,1,2,3  (the gpu_ids list, comma-joined)
        vLLM sees N GPUs remapped to indices 0..N-1.
        The --tensor-parallel-size N flag in vllm_command() tells vLLM to
        use all N visible GPUs for intra-op tensor parallelism via NCCL.
    """
    env = os.environ.copy()
    if backend == "vllm":
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
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
        gpu_col = ",".join(str(g) for g in s.gpu_ids) if s.gpu_ids else str(s.gpu)
        tp_note = f" TP×{s.tensor_parallel_size}" if s.tensor_parallel_size > 1 else ""
        pp_note = f" PP×{s.pipeline_parallel_size}" if s.pipeline_parallel_size > 1 else ""
        table.add_row(
            s.name,
            s.model.split("/")[-1],
            str(s.port),
            gpu_col + tp_note + pp_note,
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
    lines.append(f"  POST  http://{display_host}:{port}/query/stream\n", style="green")
    lines.append(f"  GET   http://{display_host}:{port}/health/live\n", style="dim")
    lines.append(f"  GET   http://{display_host}:{port}/status\n", style="dim")
    lines.append(f"  GET   http://{display_host}:{port}/docs\n", style="dim")
    pids_path = config.runtime.pids
    lines.append(f"\n  PIDs: {pids_path}/\n", style="dim")
    lines.append(f"  Logs: {config.runtime.logs}/\n", style="dim")

    console.print(
        Panel(
            lines,
            title="[bold green]✓ All specialists ready — router starting[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )
