"""
aua/cli.py — Command-line interface.

Commands:
    aua serve     #04 ✓  start all specialists + router
    aua init      #05 ✓  scaffold a new project
    aua status    #06 —  stub
    aua doctor    #07 —  stub
    aua rollback  #08 —  stub
"""

import sys
from pathlib import Path

import click
from rich.console import Console

from aua.version import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="aua")
def main():
    """Adaptive Utility Agents — deployable specialist framework.

    \b
    Quickstart:
        aua init          scaffold a new project
        aua serve         start all specialists + router
        aua doctor        check readiness before serving
        aua status        live terminal dashboard
    """


# ── aua serve ─────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--config", "-c", default="aua_config.yaml", show_default=True, help="Path to aua_config.yaml."
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Print startup commands without executing them."
)
@click.option(
    "--no-router",
    is_flag=True,
    default=False,
    help="Start specialists only — skip the FastAPI router.",
)
@click.option(
    "--router-only",
    is_flag=True,
    default=False,
    help="Start the FastAPI router only (assume specialists already running).",
)
@click.option(
    "--startup-timeout",
    default=120,
    show_default=True,
    type=int,
    help="Seconds to wait for each specialist to become healthy.",
)
@click.option(
    "--tier",
    "-t",
    default=None,
    type=click.Choice(
        ["macbook", "single-4090", "quad-4090", "a100-cluster", "rtx4090", "a100"],
        case_sensitive=False,
    ),
    help="Use a built-in hardware-tier template (rtx4090/a100 are aliases).",
)
def serve(config, dry_run, no_router, router_only, startup_timeout, tier):
    """Start all specialists + router from aua_config.yaml.

    \b
    Examples:
        aua serve
        aua serve --tier macbook
        aua serve --tier rtx4090
        aua serve --config my_config.yaml
        aua serve --dry-run
        aua serve --no-router
        aua serve --router-only
    """
    from aua.config import load_config, load_tier
    from aua.serve import serve as _serve

    try:
        cfg = load_tier(tier) if tier else load_config(config)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    _serve(
        config=cfg,
        dry_run=dry_run,
        no_router=no_router,
        router_only=router_only,
        startup_timeout=startup_timeout,
    )


# ── aua init ──────────────────────────────────────────────────────────────────


@main.command()
@click.argument("project_dir", default=".")
@click.option(
    "--tier",
    "-t",
    default="single-4090",
    show_default=True,
    type=click.Choice(
        ["macbook", "single-4090", "quad-4090", "a100-cluster", "rtx4090", "a100"],
        case_sensitive=False,
    ),
    help="Hardware tier template to scaffold (rtx4090/a100 are backward-compatible aliases).",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Overwrite existing aua_config.yaml if present.",
)
def init(project_dir, tier, force):
    """Scaffold a new AUA project directory.

    \b
    Creates:
        PROJECT_DIR/
        ├── aua_config.yaml   pre-filled for the chosen hardware tier
        ├── models/           put your AWQ / Ollama models here
        ├── dpo_pairs/        accumulated automatically by the correction loop
        ├── results/          experiment outputs and baselines
        ├── logs/             runtime logs
        └── .gitignore        ignores models/ and results/

    \b
    Examples:
        aua init
        aua init ./my-project
        aua init --tier macbook
        aua init --tier a100
        aua init --force
    """
    import shutil

    from aua.config import TIER_ALIASES

    target = Path(project_dir).resolve()

    if not target.exists():
        target.mkdir(parents=True)
        console.print(f"[green]✓[/green] Created directory: [cyan]{target}[/cyan]")
    else:
        console.print(f"  Using existing directory: [cyan]{target}[/cyan]")

    config_path = target / "aua_config.yaml"
    if config_path.exists() and not force:
        console.print(
            "[yellow]⚠[/yellow]  [cyan]aua_config.yaml[/cyan] already exists. "
            "Use [bold]--force[/bold] to overwrite."
        )
    else:
        canonical = TIER_ALIASES.get(tier, tier)
        tier_src = Path(__file__).parent / "tiers" / f"{canonical}.yaml"
        shutil.copy(tier_src, config_path)
        action = "Overwrote" if force else "Created"
        console.print(
            f"[green]✓[/green] {action} [cyan]aua_config.yaml[/cyan]  " f"[dim](tier: {tier})[/dim]"
        )

    for name, note in [
        ("models", "put downloaded AWQ / Ollama models here"),
        ("dpo_pairs", "accumulated automatically — do not edit manually"),
        ("results", "experiment outputs, baselines, and promotion logs"),
        ("logs", "runtime logs from aua serve"),
    ]:
        d = target / name
        if not d.exists():
            d.mkdir()
            console.print(f"[green]✓[/green] Created [cyan]{name}/[/cyan]  [dim]{note}[/dim]")
        else:
            console.print(f"  [dim]{name}/ already exists[/dim]")

    gitignore_path = target / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(
            "# AUA project\nmodels/\nresults/\nlogs/\n*.log\n__pycache__/\n*.pyc\n.DS_Store\n"
        )
        console.print("[green]✓[/green] Created [cyan].gitignore[/cyan]")

    from rich.panel import Panel
    from rich.text import Text

    canonical_tier = TIER_ALIASES.get(tier, tier) if tier else tier
    if canonical_tier == "macbook":
        step2 = "brew install ollama  # if not already installed"
        step3 = "aua serve --tier macbook"
    else:
        step2 = "# download models into models/ (see aua_config.yaml)"
        step3 = "aua serve"

    cd_str = f"cd {project_dir}  &&  " if project_dir != "." else ""
    next_steps = Text()
    next_steps.append(f"  1.  {cd_str}aua doctor\n", style="white")
    next_steps.append(f"  2.  {step2}\n", style="white")
    next_steps.append(f"  3.  {step3}\n", style="bold green")

    console.print(
        Panel(
            next_steps,
            title=f"[bold green]✓ Project scaffolded[/bold green]  [dim]{tier} tier[/dim]",
            border_style="green",
            padding=(0, 1),
        )
    )


# ── aua doctor ────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--config", "-c", default="aua_config.yaml", show_default=True, help="Path to aua_config.yaml."
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit results as JSON.")
@click.option("--strict", is_flag=True, default=False, help="Treat warnings as failures (exit 2).")
def doctor(config, as_json, strict):
    """Check the entire setup before running aua serve.

    \b
    Check groups (in order):
        1. Config       file found · YAML syntax · schema valid
        2. Dependencies required packages · backend binary (vllm / ollama)
        3. Hardware     CUDA · VRAM projection · ports free
        4. Models       local paths exist · HuggingFace IDs flagged
        5. Specialists  live ping (warns, not fails, if not yet started)

    Each check outputs PASS / FAIL / WARN with a fix instruction.
    Returns exit code 1 if any check fails.

    \b
    Examples:
        aua doctor
        aua doctor --config /path/to/aua_config.yaml
    """
    from aua.doctor import run_doctor

    n_failures = run_doctor(config, as_json=as_json, strict=strict)
    if strict and n_failures > 0:
        sys.exit(2)
    elif n_failures > 0:
        sys.exit(1)


# ── aua status ────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--config",
    "-c",
    default="aua_config.yaml",
    show_default=True,
    help="Path to aua_config.yaml (used to read router port).",
)
@click.option(
    "--interval", default=2, show_default=True, type=int, help="Refresh interval in seconds."
)
@click.option("--url", default=None, help="Router URL override (default: read from config).")
@click.option("--once", is_flag=True, default=False, help="Run once and exit (no auto-refresh).")
@click.option("--refresh", default=None, type=int, help="Alias for --interval.")
@click.option(
    "--json", "as_json", is_flag=True, default=False, help="Emit status as JSON and exit."
)
def status(config, interval, url, once, refresh, as_json):
    """Live terminal dashboard — specialist health, U scores, and more.

    \b
    Displays (auto-refreshes every --interval seconds):
        - Specialists  up/down · p50/p95 latency · requests · VRAM
        - Routing      single / fan-out / arbiter fallback breakdown
        - Utility      mean U · last U · confidence per domain
        - Corrections  contradictions · DPO pairs · assertions stored
        - Arbiter      verdict distribution (case 1–4) with bar charts

    \b
    Examples:
        aua status
        aua status --interval 5
        aua status --url http://my-server:8000
    """
    from aua.status import run_status

    if url:
        router_url = url
    else:
        try:
            from aua.config import load_config

            cfg = load_config(config)
            host = "localhost" if cfg.router.host == "0.0.0.0" else cfg.router.host
            router_url = f"http://{host}:{cfg.router.port}"
        except Exception:
            router_url = "http://localhost:8000"

    effective_interval = refresh if refresh is not None else interval
    run_status(
        router_url=router_url, interval=effective_interval, once=once or as_json, as_json=as_json
    )


# ── aua rollback ──────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--config", "-c", default="aua_config.yaml", show_default=True, help="Path to aua_config.yaml."
)
@click.option(
    "--specialist", "-s", default=None, help="Name of the specialist to roll back (e.g. swe, math)."
)
@click.option(
    "--all",
    "all_specialists",
    is_flag=True,
    default=False,
    help="Roll back all specialists that have promotion history.",
)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.option(
    "--no-restart",
    "no_restart",
    is_flag=True,
    default=False,
    help="Update aua_config.yaml only — do not restart the server.",
)
def rollback(config, specialist, all_specialists, yes, no_restart):
    """Revert specialist(s) to their previous BLUE model.

    \b
    Reads results/aua_promotions.json, finds the last non-reverted promotion
    for the target specialist, reverts aua_config.yaml to the BLUE model,
    restarts the server, and marks the promotion as reverted.

    \b
    Steps:
        1. Load promotions log (results/aua_promotions.json)
        2. Confirm rollback plan with user (unless --yes)
        3. Update aua_config.yaml  model: → BLUE model path
        4. Kill running vLLM process on the specialist port
        5. Restart with BLUE model and wait for health
        6. Mark promotion as reverted in the log

    \b
    Examples:
        aua rollback --specialist swe
        aua rollback --specialist swe --yes
        aua rollback --all
        aua rollback --specialist swe --no-restart   # config only
    """
    from aua.rollback import run_rollback

    if not specialist and not all_specialists:
        console.print(
            "[red]✗  Specify a specialist:[/red]\n"
            "    aua rollback --specialist swe\n"
            "    aua rollback --all"
        )
        sys.exit(1)

    result = run_rollback(
        config_path=config,
        specialist=specialist,
        all_specialists=all_specialists,
        yes=yes,
        restart=not no_restart,
    )
    if result != 0:
        sys.exit(1)


# ── aua config ────────────────────────────────────────────────────────────────


@main.group()
def config():
    """Manage AUA config — reload, validate, expand."""
    pass


@config.command("reload")
@click.option(
    "--config",
    "-c",
    default="aua_config.yaml",
    show_default=True,
    help="Path to aua_config.yaml.",
)
@click.option(
    "--pid",
    default=None,
    type=int,
    help="PID of running aua serve process to signal (auto-detected from .aua/pids/router.pid).",
)
def config_reload(config, pid):
    """Hot-reload aua_config.yaml into a running aua serve process.

    \b
    Hot-reloadable (no restart):
        routing thresholds, promotion thresholds, logging level, cors_origins

    \b
    Requires restart:
        model, port, gpu, backend changes

    \b
    Examples:
        aua config reload
        aua config reload --pid 12345
    """
    from aua.hot_reload import HotReloader

    # Auto-detect router PID if not provided
    if pid is None:
        pid_file = Path(".aua/pids/router.pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
            except ValueError:
                pass

    # Validate config first
    reloader = HotReloader(config)
    result = reloader.reload()

    if result.errors:
        console.print("[red]✗ Config validation failed:[/red]")
        for err in result.errors:
            console.print(f"  {err}")
        sys.exit(1)

    if result.hot_reloaded:
        console.print("[green]✓ Hot-reloadable changes detected:[/green]")
        for field in result.hot_reloaded:
            console.print(f"  · {field}")

    if result.restart_required:
        console.print("[yellow]⚠ Restart required for:[/yellow]")
        for field in result.restart_required:
            console.print(f"  · {field}")

    # Send SIGHUP to running process
    if pid:
        try:
            import os as _os
            import signal as _signal

            _os.kill(pid, _signal.SIGHUP)
            console.print(f"[green]✓ SIGHUP sent to pid {pid}[/green]")
        except ProcessLookupError:
            console.print(f"[red]✗ No process with pid {pid}[/red]")
            sys.exit(1)
        except PermissionError:
            console.print(f"[red]✗ Permission denied sending SIGHUP to pid {pid}[/red]")
            sys.exit(1)
    else:
        console.print("[dim]No running process found — config validated only.[/dim]")
        console.print("[dim]To reload a running server: aua config reload --pid <PID>[/dim]")


@config.command("validate")
@click.option(
    "--config",
    "-c",
    default="aua_config.yaml",
    show_default=True,
    help="Path to aua_config.yaml.",
)
def config_validate(config):
    """Validate aua_config.yaml without starting anything.

    \b
    Examples:
        aua config validate
        aua config validate --config /path/to/aua_config.yaml
    """
    try:
        from aua.config import load_config

        cfg = load_config(config)
        console.print(
            f"[green]✓[/green] Config valid  "
            f"[dim]{len(cfg.specialists)} specialist(s) · backend={cfg.backend}[/dim]"
        )
    except FileNotFoundError as e:
        console.print(f"[red]✗ File not found:[/red] {e}")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]✗ Validation error:[/red] {e}")
        sys.exit(1)
