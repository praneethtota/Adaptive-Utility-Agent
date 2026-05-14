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
@click.option("--with-ui", "with_ui", is_flag=True, default=False, help="Also start the Chat UI.")
@click.option("--ui-port", default=3001, show_default=True, type=int, help="Chat UI port.")
@click.option(
    "--arbitration-mode",
    "arbitration_mode",
    default=None,
    type=click.Choice(["pairwise", "vcg"], case_sensitive=False),
    help="Override arbitration mode: 'pairwise' (default) or 'vcg' (welfare maximization).",
)
def serve(
    config,
    dry_run,
    no_router,
    router_only,
    startup_timeout,
    tier,
    with_ui,
    ui_port,
    arbitration_mode,
):
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

    # Apply CLI override for arbitration mode
    if arbitration_mode:
        cfg.router.arbitration_mode = arbitration_mode
        console.print(f"[dim]Arbitration mode: [cyan]{arbitration_mode}[/cyan][/dim]")

    _serve(
        config=cfg,
        dry_run=dry_run,
        no_router=no_router,
        router_only=router_only,
        startup_timeout=startup_timeout,
        config_path=config if not tier else None,
    )

    if with_ui and not dry_run:
        _start_chat_ui(ui_port)


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
    "--preset",
    "-p",
    default="coding",
    show_default=True,
    help="Named specialist configuration to use (coding/research/legal/medical/general/creative).",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Overwrite existing aua_config.yaml if present.",
)
def init(project_dir, tier, force, preset):
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
    from aua.presets import get_preset

    # Validate preset
    try:
        _preset_spec = get_preset(preset)
    except ValueError as _preset_err:
        console.print(f"[red]✗[/red] {_preset_err}")
        sys.exit(1)

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


# ── aua models ────────────────────────────────────────────────────────────────


@main.group()
def models():
    """Manage and inspect models for the current project."""
    pass


@models.command("list")
@click.option(
    "--config",
    "-c",
    default="aua_config.yaml",
    show_default=True,
    help="Path to aua_config.yaml.",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
def models_list(config, as_json):
    """List all models configured for this project and their pull status.

    \b
    Examples:
        aua models list
        aua models list --json
    """
    import json as _json
    import shutil
    import subprocess

    try:
        from aua.config import load_config

        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]✗[/red] Could not load config: {e}")
        sys.exit(1)

    entries = []
    for spec in cfg.specialists:
        entries.append(
            {
                "role": "specialist",
                "name": spec.name,
                "model": spec.model,
                "field": spec.field,
                "port": spec.port,
                "backend": spec.backend,
            }
        )
    entries.append(
        {
            "role": "arbiter",
            "name": "arbiter",
            "model": cfg.arbiter.model,
            "field": "general",
            "port": cfg.arbiter.port,
            "backend": cfg.arbiter.backend,
        }
    )

    # Check pull status for Ollama
    pulled: set[str] = set()
    if cfg.backend == "ollama" and shutil.which("ollama"):
        try:
            _result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            for _line in _result.stdout.splitlines()[1:]:
                if _line.strip():
                    pulled.add(_line.split()[0].strip())
        except Exception:
            pass

    for entry in entries:
        if cfg.backend == "ollama":
            entry["pulled"] = any(entry["model"] in p or p in entry["model"] for p in pulled)
        else:
            entry["pulled"] = None  # vLLM: check models/ directory

    if as_json:
        print(_json.dumps(entries, indent=2))
        return

    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE, header_style="bold dim")
    table.add_column("Role")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("Field")
    table.add_column("Port", justify="right")
    table.add_column("Status")

    for e in entries:  # type: ignore[misc]
        if e["pulled"] is True:
            status = "[green]✓ pulled[/green]"
        elif e["pulled"] is False:
            status = "[yellow]not pulled[/yellow]"
        else:
            status = "[dim]—[/dim]"
        table.add_row(e["role"], e["name"], e["model"], e["field"], str(e["port"]), status)

    console.print(table)


# ── aua fields ────────────────────────────────────────────────────────────────


@main.group()
def fields():
    """Inspect available field configurations."""
    pass


@fields.command("list")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
def fields_list(as_json):
    """List all built-in fields and their utility weights.

    \b
    Fields define how U = w_e·E + w_c·C + w_k·K is evaluated
    and how hard errors are penalized.

    \b
    Examples:
        aua fields list
        aua fields list --json
    """
    import json as _json

    from aua import FIELD_CONFIGS

    if as_json:
        data = {
            name: {
                "w_efficacy": f.w_efficacy,
                "w_confidence": f.w_confidence,
                "w_curiosity": f.w_curiosity,
                "c_min": f.c_min,
                "e_min": f.e_min,
                "penalty_multiplier": f.penalty_multiplier,
            }
            for name, f in FIELD_CONFIGS.items()
        }
        print(_json.dumps(data, indent=2))
        return

    from rich import box
    from rich.table import Table

    table = Table(
        box=box.SIMPLE, header_style="bold dim", title="[bold]Built-in Field Configurations[/bold]"
    )
    table.add_column("Field")
    table.add_column("w_e", justify="right")
    table.add_column("w_c", justify="right")
    table.add_column("w_k", justify="right")
    table.add_column("c_min", justify="right")
    table.add_column("Penalty", justify="right")

    for name, f in FIELD_CONFIGS.items():
        table.add_row(
            name,
            f"{f.w_efficacy:.2f}",
            f"{f.w_confidence:.2f}",
            f"{f.w_curiosity:.2f}",
            f"{f.c_min:.2f}",
            f"{f.penalty_multiplier:.0f}×",
        )

    console.print(table)
    console.print("[dim]U = w_e·E + w_c·C + w_k·K  ·  c_min = minimum confidence required[/dim]")


# ── aua presets ───────────────────────────────────────────────────────────────


@main.group()
def presets():
    """List and inspect named project presets."""
    pass


@presets.command("list")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
def presets_list(as_json):
    """List all available presets for aua init --preset.

    \b
    Examples:
        aua presets list
        aua presets list --json
    """
    import json as _json

    from aua.presets import PRESETS

    if as_json:
        data = {
            name: {
                "description": p.description,
                "specialists": p.specialists,
                "recommended_tiers": p.recommended_tiers,
                "notes": p.notes,
            }
            for name, p in PRESETS.items()
        }
        print(_json.dumps(data, indent=2))
        return

    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE, header_style="bold dim", title="[bold]Available Presets[/bold]")
    table.add_column("Preset")
    table.add_column("Specialists")
    table.add_column("Description")

    for name, p in PRESETS.items():
        table.add_row(
            f"[cyan]{name}[/cyan]",
            ", ".join(p.specialists),
            p.description,
        )

    console.print(table)
    console.print("[dim]Usage: aua init --preset <name> --tier <tier>[/dim]")


@config.command("expand")
@click.option(
    "--config",
    "-c",
    default="aua_config.yaml",
    show_default=True,
    help="Path to aua_config.yaml.",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
def config_expand(config, as_json):
    """Print the fully-resolved config with all defaults filled in.

    Shows exactly what AUA will use at runtime — tier defaults, computed
    URLs, runtime paths, field weights — not just what is in the YAML file.

    \b
    Examples:
        aua config expand
        aua config expand --json
        aua config expand | grep url
    """
    import json as _json

    try:
        from aua.config import load_config

        cfg = load_config(config)
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")
        sys.exit(1)

    data: dict = {
        "version": cfg.version,
        "backend": cfg.backend,
        "mode": cfg.mode,
        "specialists": [],
        "arbiter": {},
        "router": {},
        "blue_green": {},
        "logging": {},
        "runtime": {},
    }

    for s in cfg.specialists:
        data["specialists"].append(
            {
                "name": s.name,
                "model": s.model,
                "field": s.field,
                "port": s.port,
                "host": s.host,
                "scheme": s.scheme,
                "gpu": s.gpu,
                "gpu_memory_utilization": s.gpu_memory_utilization,
                "backend": s.backend,
                "endpoint": s.endpoint,
                "models_url": s.models_url,
            }
        )

    arb = cfg.arbiter
    data["arbiter"] = {
        "model": arb.model,
        "port": arb.port,
        "host": arb.host,
        "scheme": arb.scheme,
        "gpu": arb.gpu,
        "gpu_memory_utilization": arb.gpu_memory_utilization,
        "backend": arb.backend,
        "endpoint": arb.endpoint,
        "models_url": arb.models_url,
    }

    rtr = cfg.router
    data["router"] = {
        "port": rtr.port,
        "host": rtr.host,
        "single_domain_threshold": rtr.single_domain_threshold,
        "fanout_threshold": rtr.fanout_threshold,
        "specialist_timeout": rtr.specialist_timeout,
        "cors_origins": rtr.cors_origins,
    }

    for name, bg in cfg.blue_green.items():
        data["blue_green"][name] = {
            "delta": bg.delta,
            "T_min": bg.T_min,
            "tau": bg.tau,
        }

    data["logging"] = {"level": cfg.logging.level}

    rt = cfg.runtime
    data["runtime"] = {
        "logs": str(rt.logs),
        "pids": str(rt.pids),
        "state": str(rt.state),
        "checkpoints": str(rt.checkpoints),
    }

    if as_json:
        print(_json.dumps(data, indent=2))
        return

    import yaml

    console.print(f"[dim]# Expanded config — {config}[/dim]")
    console.print(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))


# ── aua defaults ──────────────────────────────────────────────────────────────


@main.group()
def defaults():
    """Inspect the framework's built-in defaults."""
    pass


@defaults.command("show")
@click.argument("category", default="", required=False)
@click.argument("key", default="", required=False)
@click.option("--json", "as_json", is_flag=True, default=False)
def defaults_show(category, key, as_json):
    """Show built-in framework defaults.

    \b
    Examples:
        aua defaults show                  # list all categories
        aua defaults show fields           # all field configs
        aua defaults show models           # built-in model aliases
        aua defaults show routing          # routing thresholds + docs
        aua defaults show preset coding    # a specific preset
    """
    import json as _json

    from aua.defaults.registry import get_defaults, list_categories

    if not category:
        cats = list_categories()
        if as_json:
            print(_json.dumps({"categories": cats}))
        else:
            console.print("[bold]Available categories:[/bold]")
            for c in cats:
                console.print(f"  [cyan]{c}[/cyan]")
            console.print("\n[dim]Usage: aua defaults show <category>[/dim]")
        return

    try:
        data = get_defaults(category)
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        sys.exit(1)

    if key:
        data = data.get(key)
        if data is None:
            console.print(f"[red]✗[/red] Key {key!r} not found in {category!r}")
            sys.exit(1)

    if as_json:
        print(_json.dumps(data, indent=2, default=str))
        return

    import yaml

    console.print(f"[dim]# {category}{(' · ' + key) if key else ''}[/dim]")
    console.print(yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True))


# ── aua extensions ────────────────────────────────────────────────────────────


@main.group()
def extensions():
    """Manage and test AUA framework extensions (plugins, hooks, middleware)."""
    pass


@extensions.command("list")
@click.option("--json", "as_json", is_flag=True, default=False)
def extensions_list(as_json):
    """List all registered extensions."""
    import json as _json

    from aua.hooks import get_hook_runner
    from aua.middleware import get_middleware_pipeline
    from aua.plugins.registry import get_registry

    reg = get_registry()
    hooks = get_hook_runner().registered_hooks()
    mw = get_middleware_pipeline().registered()
    data = {"plugins": list(getattr(reg, "_plugins", {}).keys()), "middleware": mw, "hooks": hooks}

    if as_json:
        print(_json.dumps(data, indent=2))
        return

    console.print("[bold]Extensions:[/bold]")
    console.print(f"  Plugins:    {data['plugins'] or ['(none)']}")
    console.print(f"  Middleware: {mw or ['(none)']}")
    if hooks:
        for point, names in hooks.items():
            console.print(f"  Hook [{point}]: {names}")
    else:
        console.print("  Hooks:      (none)")


@extensions.command("test")
@click.option(
    "--kind",
    required=True,
    type=click.Choice(
        [
            "field_classifier",
            "utility_scorer",
            "arbiter_policy",
            "promotion_policy",
            "correction_store",
            "model_backend",
            "state_store",
            "hook",
            "middleware",
        ]
    ),
    help="Plugin type to test.",
)
@click.option("--import-path", "import_path", required=True, help="'module:ClassName' format.")
@click.option("--config-json", default="{}", show_default=True, help="JSON config dict.")
def extensions_test(kind, import_path, config_json):
    """Test that a plugin loads and satisfies its Protocol contract.

    \b
    Examples:
        aua extensions test --kind utility_scorer \\
          --import-path plugins.custom_utility:RiskWeightedUtilityScorer

        aua extensions test --kind middleware \\
          --import-path aua.middleware:PIIRedactionMiddleware
    """
    import json as _json

    from aua.plugins.registry import PluginLoadError, load_plugin

    try:
        plugin_config = _json.loads(config_json)
    except _json.JSONDecodeError as e:
        console.print(f"[red]✗[/red] Invalid --config-json: {e}")
        sys.exit(1)

    console.print(f"Testing: [cyan]{import_path}[/cyan]  kind=[dim]{kind}[/dim]")

    try:
        plugin = load_plugin(import_path, kind, plugin_config)
        console.print("[green]✓ Plugin loaded successfully[/green]")
        console.print(f"  Type:     {type(plugin).__name__}")
        console.print(f"  Module:   {type(plugin).__module__}")
        console.print(f"  Protocol: {kind} — contract satisfied ✓")
    except PluginLoadError as e:
        console.print(f"[red]✗ Plugin test failed:[/red] {e}")
        sys.exit(1)


@extensions.command("inspect")
@click.argument("import_path")
def extensions_inspect(import_path):
    """Show details about a plugin class.

    \b
    Examples:
        aua extensions inspect aua.middleware:PIIRedactionMiddleware
    """
    import importlib

    if ":" not in import_path:
        console.print("[red]✗[/red] Format must be 'module:ClassName'")
        sys.exit(1)

    module_path, class_name = import_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        console.print(f"[red]✗[/red] Cannot import {module_path!r}: {e}")
        sys.exit(1)

    cls = getattr(module, class_name, None)
    if cls is None:
        console.print(f"[red]✗[/red] Class {class_name!r} not found in {module_path!r}")
        sys.exit(1)

    doc = (cls.__doc__ or "").strip().split("\n")[0]
    methods = [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m))]

    console.print(f"[bold]{import_path}[/bold]")
    console.print(f"  Description: {doc or '(no docstring)'}")
    console.print(f"  Methods:     {', '.join(methods) or '(none)'}")
    console.print(f"  Module file: {getattr(module, '__file__', 'unknown')}")


# ── aua token ─────────────────────────────────────────────────────────────────


@main.group()
def token():
    """Manage AUA API access tokens."""
    pass


@token.command("create")
@click.option(
    "--scope",
    "-s",
    multiple=True,
    required=True,
    help="Scope to grant (repeat for multiple). Use 'aua:admin' for all scopes.",
)
@click.option(
    "--expires",
    default="30d",
    show_default=True,
    help="Expiry: Nd (days), Nw (weeks), Nm (months). e.g. 30d, 2w, 1m.",
)
@click.option("--label", default="", help="Human-readable label for this token.")
@click.option("--config", "-c", default="aua_config.yaml", show_default=True)
def token_create(scope, expires, label, config):
    """Create a new API access token.

    \b
    Examples:
        aua token create --scope aua:query --expires 30d
        aua token create --scope aua:query --scope aua:stream --label "prod-app"
        aua token create --scope aua:admin --expires 1d --label "ci-deploy"
    """
    import re

    from aua.auth import VALID_SCOPES, TokenManager

    # Parse expiry
    m = re.match(r"^(\d+)([dwm])$", expires.lower())
    if not m:
        console.print("[red]✗[/red] --expires format: Nd, Nw, or Nm (e.g. 30d, 2w, 1m)")
        sys.exit(1)
    n, unit = int(m.group(1)), m.group(2)
    days = {"d": n, "w": n * 7, "m": n * 30}[unit]

    # Validate scopes
    scopes = list(scope)
    invalid = set(scopes) - VALID_SCOPES - {"aua:admin"}
    if invalid:
        console.print(f"[red]✗[/red] Unknown scopes: {invalid}")
        console.print(f"Valid scopes: {sorted(VALID_SCOPES)}")
        sys.exit(1)

    try:
        from aua.config import load_config

        cfg = load_config(config)
    except Exception:
        cfg = None

    mgr = TokenManager.from_config(cfg)

    tok, tok_str = mgr.create(scopes=scopes, expires_days=days, label=label)

    console.print(f"[green]✓ Token created[/green]  id=[dim]{tok.token_id[:8]}...[/dim]")
    console.print(f"  Scopes:  {', '.join(tok.scopes)}")
    console.print(f"  Expires: {tok.as_dict()['expires_at_human']}")
    if label:
        console.print(f"  Label:   {label}")
    console.print("\n[bold]Token (store securely — shown once):[/bold]")
    console.print(f"\n{tok_str}\n")
    console.print("[dim]Use as: Authorization: Bearer <token>[/dim]")


@token.command("list")
@click.option("--config", "-c", default="aua_config.yaml", show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.option("--include-revoked", is_flag=True, default=False)
def token_list(config, as_json, include_revoked):
    """List all active tokens.

    \b
    Examples:
        aua token list
        aua token list --include-revoked
        aua token list --json
    """
    import json as _json

    from aua.auth import TokenManager

    try:
        from aua.config import load_config

        cfg = load_config(config)
    except Exception:
        cfg = None

    mgr = TokenManager.from_config(cfg)
    tokens = mgr.list_tokens(include_revoked=include_revoked)

    if as_json:
        print(_json.dumps(tokens, indent=2, default=str))
        return

    if not tokens:
        console.print("[dim]No tokens found.[/dim]")
        return

    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE, header_style="bold dim")
    table.add_column("ID (prefix)")
    table.add_column("Label")
    table.add_column("Scopes")
    table.add_column("Expires")
    table.add_column("Status")

    for t in tokens:
        tid = t.get("token_id", "")[:8] + "..."
        expiry = t.get("expires_at_human", t.get("expires_at", ""))[:10]
        revoked = t.get("revoked", False)
        status = "[red]revoked[/red]" if revoked else "[green]active[/green]"
        scopes = ", ".join(t.get("scopes", []))
        table.add_row(tid, t.get("label", ""), scopes, expiry, status)

    console.print(table)


@token.command("revoke")
@click.argument("token_id")
@click.option("--config", "-c", default="aua_config.yaml", show_default=True)
def token_revoke(token_id, config):
    """Revoke a token by its ID (or ID prefix).

    \b
    Examples:
        aua token revoke abc12345-...
    """
    from aua.auth import TokenManager

    try:
        from aua.config import load_config

        cfg = load_config(config)
    except Exception:
        cfg = None

    mgr = TokenManager.from_config(cfg)

    # If only prefix supplied, match against list
    if len(token_id) < 36:
        tokens = mgr.list_tokens(include_revoked=False)
        matches = [t for t in tokens if t.get("token_id", "").startswith(token_id)]
        if len(matches) == 0:
            console.print(f"[red]✗[/red] No active token found with prefix {token_id!r}")
            sys.exit(1)
        if len(matches) > 1:
            console.print(
                f"[red]✗[/red] Ambiguous prefix — {len(matches)} tokens match. Use full ID."
            )
            sys.exit(1)
        token_id = matches[0]["token_id"]

    ok = mgr.revoke(token_id)
    if ok:
        console.print(f"[green]✓ Token revoked:[/green] {token_id[:8]}...")
    else:
        console.print(f"[red]✗[/red] Token not found: {token_id}")
        sys.exit(1)


@token.command("inspect")
@click.argument("token_str_or_id")
@click.option("--config", "-c", default="aua_config.yaml", show_default=True)
def token_inspect(token_str_or_id, config):
    """Inspect a token string or token ID.

    \b
    Examples:
        aua token inspect eyJ...  # raw token string
        aua token inspect abc12345  # token ID prefix
    """
    from aua.auth import TokenError, TokenManager

    try:
        from aua.config import load_config

        cfg = load_config(config)
    except Exception:
        cfg = None

    mgr = TokenManager.from_config(cfg)

    # Try as token string first, then as ID
    if "." in token_str_or_id:
        try:
            tok = mgr.verify(token_str_or_id)
            d = tok.as_dict()
            console.print("[green]✓ Valid token[/green]")
        except TokenError as e:
            console.print(f"[yellow]⚠ Token invalid:[/yellow] {e}")
            # Try to parse without verification for inspection
            try:
                import base64
                import json as _json

                payload_b64 = token_str_or_id.rsplit(".", 1)[0]
                payload = _json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
                d = payload
                console.print("[dim](signature verification failed — showing payload only)[/dim]")
            except Exception:
                sys.exit(1)
    else:
        # Lookup by ID
        tokens = mgr.list_tokens(include_revoked=True)
        matches = [t for t in tokens if t.get("token_id", "").startswith(token_str_or_id)]
        if not matches:
            console.print(f"[red]✗[/red] Token not found: {token_str_or_id}")
            sys.exit(1)
        d = matches[0]

    for k, v in d.items():
        console.print(f"  [dim]{k:20s}[/dim] {v}")


# ── aua certs ─────────────────────────────────────────────────────────────────


@main.group()
def certs():
    """Manage mTLS certificates for internal AUA communication."""
    pass


@certs.command("generate")
@click.option(
    "--cert-dir",
    default=".aua/certs",
    show_default=True,
    help="Directory to write certificates to.",
)
@click.option("--force", is_flag=True, default=False, help="Overwrite existing certificates.")
def certs_generate(cert_dir, force):
    """Generate self-signed dev certificates for mTLS.

    WARNING: These are for development only. Use your own CA in production.

    \b
    Examples:
        aua certs generate
        aua certs generate --cert-dir /etc/aua/certs
    """
    from pathlib import Path as _Path

    cert_path = _Path(cert_dir)
    if cert_path.exists() and list(cert_path.glob("*.pem")) and not force:
        console.print(
            f"[yellow]⚠[/yellow]  Certificates already exist in {cert_dir}. "
            "Use [bold]--force[/bold] to overwrite."
        )
        sys.exit(1)

    console.print(f"Generating development certificates in [cyan]{cert_dir}[/cyan]...")

    try:
        from aua.certs import generate_dev_certs

        paths = generate_dev_certs(cert_dir)
    except ImportError as e:
        console.print(f"[red]✗[/red] {e}")
        sys.exit(1)

    for name, path in paths.items():
        console.print(f"  [green]✓[/green] {name:15s} → {path}")

    console.print(
        "\n[yellow]⚠ Development certs only — do not use in production![/yellow]\n"
        "[dim]For production: provide your own CA-signed certificates.[/dim]"
    )


@certs.command("inspect")
@click.option("--cert-dir", default=".aua/certs", show_default=True)
@click.option("--json", "as_json", is_flag=True, default=False)
def certs_inspect(cert_dir, as_json):
    """Show certificate expiry dates.

    \b
    Examples:
        aua certs inspect
        aua certs inspect --cert-dir /etc/aua/certs
    """
    import json as _json

    from aua.certs import inspect_certs

    results = inspect_certs(cert_dir)

    if not results:
        console.print(f"[dim]No certificates found in {cert_dir}[/dim]")
        return

    if as_json:
        print(_json.dumps(results, indent=2))
        return

    for r in results:
        if "error" in r:
            console.print(f"[red]✗[/red] {r['file']}: {r['error']}")
            continue
        days = r.get("days_remaining", 0)
        if r.get("expired"):
            status = "[red]EXPIRED[/red]"
        elif r.get("expiring_soon"):
            status = f"[yellow]expiring in {days}d[/yellow]"
        else:
            status = f"[green]valid ({days}d remaining)[/green]"
        console.print(f"  {status}  {Path(r['file']).name}")


# ── aua eval ──────────────────────────────────────────────────────────────────


@main.group()
def eval():
    """Run evaluation datasets against the live AUA router."""
    pass


@eval.command("run")
@click.option("--dataset", "-d", required=True, help="Path to eval dataset YAML.")
@click.option("--config", "-c", default="aua_config.yaml", show_default=True)
@click.option("--url", default="http://localhost:8000", show_default=True, help="Router URL.")
@click.option("--output-dir", default=".aua/evals", show_default=True)
@click.option("--timeout", default=120.0, show_default=True, type=float)
@click.option("--json", "as_json", is_flag=True, default=False)
def eval_run(dataset, config, url, output_dir, timeout, as_json):
    """Run an evaluation dataset against the live router.

    \b
    Examples:
        aua eval run --dataset evals/coding_smoke.yaml
        aua eval run --dataset evals/math_smoke.yaml --json
    """
    import json as _json

    from aua.eval import run_dataset, save_report

    console.print(f"Running eval: [cyan]{dataset}[/cyan]  router=[dim]{url}[/dim]")

    try:
        report = run_dataset(dataset, router_url=url, timeout=timeout)
    except Exception as e:
        console.print(f"[red]✗[/red] Eval failed: {e}")
        sys.exit(1)

    fname = save_report(report, output_dir)

    if as_json:
        print(_json.dumps(report.to_dict(), indent=2))
        return

    # Pretty summary
    rate_color = (
        "green" if report.pass_rate >= 0.8 else "yellow" if report.pass_rate >= 0.5 else "red"
    )
    console.print(f"\n[bold]Results — {report.dataset_name}[/bold]")
    console.print(
        f"  Pass rate:   [{rate_color}]{report.pass_rate:.0%}[/{rate_color}]  ({report.passed}/{report.total})"
    )
    console.print(f"  Mean U:      {report.mean_u_score:.4f}")
    console.print(f"  Mean latency:{report.mean_latency_ms:.0f}ms")

    for c in report.cases:
        icon = "[green]✓[/green]" if c["passed"] else "[red]✗[/red]"
        fail_str = f"  [dim]{'; '.join(c['failures'])}[/dim]" if c["failures"] else ""
        err_str = f"  [red]{c['error']}[/red]" if c["error"] else ""
        console.print(
            f"  {icon} {c['id']:35s} U={c['u_score']:.3f}  {c['latency_ms']:.0f}ms{fail_str}{err_str}"
        )

    console.print(f"\n[dim]Report saved: {fname}[/dim]")

    if report.pass_rate < 0.5:
        sys.exit(1)


@eval.command("report")
@click.argument("report_path", default=".aua/evals/latest.json")
@click.option("--json", "as_json", is_flag=True, default=False)
def eval_report(report_path, as_json):
    """Display a saved eval report.

    \b
    Examples:
        aua eval report
        aua eval report .aua/evals/coding_smoke_20260511_090000.json
    """
    import json as _json

    try:
        data = _json.loads(Path(report_path).read_text())
    except FileNotFoundError:
        console.print(f"[red]✗[/red] Report not found: {report_path}")
        sys.exit(1)

    if as_json:
        print(_json.dumps(data, indent=2))
        return

    s = data["summary"]
    console.print(f"\n[bold]{data['dataset']}[/bold]  [dim]{data.get('run_at_human', '')}[/dim]")
    console.print(f"  Pass rate: {s['pass_rate']:.0%}  ({s['passed']}/{s['total']})")
    console.print(f"  Mean U:    {s['mean_u_score']:.4f}")
    console.print(f"  Latency:   {s['mean_latency_ms']:.0f}ms")
    for c in data["cases"]:
        icon = "[green]✓[/green]" if c["passed"] else "[red]✗[/red]"
        console.print(f"  {icon} {c['id']}")


@eval.command("compare")
@click.option("--baseline", required=True, help="Baseline report JSON path.")
@click.option("--candidate", required=True, help="Candidate report JSON path.")
@click.option("--json", "as_json", is_flag=True, default=False)
def eval_compare(baseline, candidate, as_json):
    """Compare two eval reports for regression.

    \b
    Examples:
        aua eval compare --baseline .aua/evals/blue.json --candidate .aua/evals/green.json
    """
    import json as _json

    def _load(path: str) -> dict:
        return _json.loads(Path(path).read_text())

    b = _load(baseline)
    c = _load(candidate)

    b_rate = b["summary"]["pass_rate"]
    c_rate = c["summary"]["pass_rate"]
    b_u = b["summary"]["mean_u_score"]
    c_u = c["summary"]["mean_u_score"]

    delta_pass = c_rate - b_rate
    delta_u = c_u - b_u
    regressed = delta_pass < -0.05 or delta_u < -0.02

    result = {
        "verdict": "REGRESSION" if regressed else "OK",
        "regressed": regressed,
        "baseline": {"dataset": b["dataset"], "pass_rate": b_rate, "u_score": b_u},
        "candidate": {"dataset": c["dataset"], "pass_rate": c_rate, "u_score": c_u},
        "delta_pass_rate": round(delta_pass, 3),
        "delta_u_score": round(delta_u, 4),
    }

    if as_json:
        print(_json.dumps(result, indent=2))
        return

    verdict_color = "red" if regressed else "green"
    console.print(f"\n[bold][{verdict_color}]{result['verdict']}[/{verdict_color}][/bold]")
    console.print(f"  Pass rate: {b_rate:.0%} → {c_rate:.0%}  (Δ {delta_pass:+.0%})")
    console.print(f"  U score:   {b_u:.4f} → {c_u:.4f}  (Δ {delta_u:+.4f})")

    if regressed:
        sys.exit(1)


# ── aua corrections / dpo export ──────────────────────────────────────────────


@main.group()
def corrections():
    """Manage and export corrections from the state store."""
    pass


@corrections.command("export")
@click.option(
    "--format", "fmt", default="jsonl", show_default=True, type=click.Choice(["jsonl", "json"])
)
@click.option("--domain", default=None, help="Filter by domain.")
@click.option("--limit", default=1000, show_default=True, type=int)
@click.option("--output", "-o", default=None, help="Output file path (default: stdout).")
@click.option("--redact", is_flag=True, default=False, help="Redact sensitive fields.")
def corrections_export(fmt, domain, limit, output, redact):
    """Export stored corrections from the state store.

    \b
    Examples:
        aua corrections export --format jsonl
        aua corrections export --domain software_engineering --output corrections.jsonl
        aua corrections export --redact
    """
    import json as _json

    from aua.state import get_state_store

    store = get_state_store()
    filters = {"domain": domain} if domain else {}
    records = store.query("corrections", filters=filters, limit=limit)

    if redact:
        for r in records:
            if "claim" in r and len(r["claim"]) > 100:
                r["claim"] = r["claim"][:100] + "...[redacted]"

    if fmt == "jsonl":
        lines = "\n".join(_json.dumps(r, default=str) for r in records)
    else:
        lines = _json.dumps(records, indent=2, default=str)

    if output:
        Path(output).write_text(lines)
        console.print(f"[green]✓[/green] Exported {len(records)} corrections → {output}")
    else:
        print(lines)


@main.group()
def dpo():
    """Export DPO preference pairs for fine-tuning."""
    pass


@dpo.command("export")
@click.option(
    "--format",
    "fmt",
    default="jsonl",
    show_default=True,
    type=click.Choice(["jsonl", "preference-pairs"]),
)
@click.option("--domain", default=None, help="Filter by domain.")
@click.option("--limit", default=1000, show_default=True, type=int)
@click.option("--output", "-o", default=None, help="Output file path (default: stdout).")
def dpo_export(fmt, domain, limit, output):
    """Export DPO preference pairs for fine-tuning.

    Output format: {prompt, chosen, rejected, field, utility_chosen,
                    utility_rejected, correction_ids, trace_id}

    \b
    Examples:
        aua dpo export --format jsonl --output dpo_pairs.jsonl
        aua dpo export --format preference-pairs --domain software_engineering
    """
    import json as _json

    from aua.state import get_state_store

    store = get_state_store()
    filters = {"domain": domain} if domain else {}
    records = store.query("corrections", filters=filters, limit=limit)

    # Build DPO pairs: corrections have claim (chosen) and rejected
    pairs = []
    for r in records:
        if not r.get("rejected"):
            continue
        pairs.append(
            {
                "prompt": r.get("subject", ""),
                "chosen": r.get("claim", ""),
                "rejected": r.get("rejected", ""),
                "field": r.get("domain", ""),
                "utility_chosen": r.get("effective_confidence", 0.0),
                "utility_rejected": 0.0,
                "correction_ids": [r.get("id", "")],
                "trace_id": "",
                "source": r.get("source", "arbiter"),
            }
        )

    if fmt == "jsonl" or fmt == "preference-pairs":
        lines = "\n".join(_json.dumps(p, default=str) for p in pairs)
    else:
        lines = _json.dumps(pairs, indent=2, default=str)

    if output:
        Path(output).write_text(lines)
        console.print(f"[green]✓[/green] Exported {len(pairs)} DPO pairs → {output}")
    else:
        if pairs:
            print(lines)
        else:
            console.print("[dim]No DPO pairs found (need contradictions to generate pairs).[/dim]")


def _start_chat_ui(port: int = 3001) -> None:
    """Start the AUA Chat UI (Next.js) as a background process."""
    import os
    import shutil
    import subprocess as _sp
    import time

    ui_dir = Path(__file__).parent.parent / "apps" / "aua_chat"
    if not ui_dir.exists():
        console.print("[yellow]⚠[/yellow]  Chat UI not found at apps/aua_chat/. Skipping.")
        console.print(
            "[dim]  Tip: run [cyan]cd apps/aua_chat && npm run dev[/cyan] in a separate terminal.[/dim]"
        )
        return

    # Build a PATH that includes common node install locations on Mac
    # (nvm, homebrew, volta, fnm, system) — shutil.which alone misses nvm paths
    extra = [
        os.path.expanduser("~/.nvm/versions/node/$(node --version 2>/dev/null)/bin"),
        "/opt/homebrew/bin",  # Apple Silicon homebrew
        "/usr/local/bin",  # Intel homebrew / system node
        os.path.expanduser("~/.volta/bin"),
        os.path.expanduser("~/.fnm/current/bin"),
    ]
    env = os.environ.copy()
    env["PATH"] = ":".join(extra) + ":" + env.get("PATH", "")

    npm = shutil.which("npm", path=env["PATH"])
    if not npm:
        console.print("[yellow]⚠[/yellow]  npm not found — Chat UI requires Node.js 18+.")
        console.print(
            "  Install: [cyan]brew install node[/cyan]  or  [cyan]https://nodejs.org[/cyan]\n"
            "  Then run manually: [cyan]cd apps/aua_chat && npm run dev[/cyan]"
        )
        return

    # Install deps if needed
    node_modules = ui_dir / "node_modules"
    if not node_modules.exists():
        console.print("[dim]Installing Chat UI dependencies (first run)…[/dim]")
        _sp.run([npm, "install", "--prefer-offline"], cwd=str(ui_dir), check=True, env=env)

    # Log to .aua/logs/ui.log so errors are visible
    log_dir = Path(".aua") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ui.log"
    log_file = open(log_path, "w")

    console.print(f"[dim]Starting Chat UI on http://localhost:{port}[/dim]")
    proc = _sp.Popen(
        [npm, "run", "dev", "--", "--port", str(port)],
        cwd=str(ui_dir),
        stdout=log_file,
        stderr=log_file,
        env=env,
    )

    # Give Next.js 2 seconds to fail fast — if it exits immediately, surface the log
    time.sleep(2)
    if proc.poll() is not None:
        log_file.flush()
        console.print(f"[red]✗[/red]  Chat UI failed to start. Last log lines ({log_path}):")
        try:
            lines = log_path.read_text().strip().splitlines()
            for line in lines[-10:]:
                console.print(f"  [dim]{line}[/dim]")
        except Exception:
            pass
        console.print(
            "\n  Run manually in a separate terminal:\n"
            f"  [cyan]cd {ui_dir} && npm run dev[/cyan]"
        )
        return

    console.print(
        f"[green]✓[/green] Chat UI: [cyan]http://localhost:{port}[/cyan]  "
        f"(admin / aua-admin)  log: {log_path}"
    )


@main.command("ui")
@click.option("--port", default=3001, show_default=True, type=int)
@click.option("--install-only", is_flag=True, default=False, help="Only install dependencies.")
def ui_command(port, install_only):
    """Start the AUA Chat UI independently.

    \b
    Requires Node.js 18+ and npm. First run installs dependencies.

    \b
    Examples:
        aua ui
        aua ui --port 4000
        aua serve --with-ui
    """
    import os
    import shutil
    import subprocess as _sp

    ui_dir = Path(__file__).parent.parent / "apps" / "aua_chat"
    if not ui_dir.exists():
        console.print(f"[red]✗[/red] Chat UI not found: {ui_dir}")
        sys.exit(1)

    # Resolve npm with Mac-aware PATH (nvm, homebrew, volta, fnm)
    extra = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        os.path.expanduser("~/.volta/bin"),
        os.path.expanduser("~/.fnm/current/bin"),
    ]
    env = os.environ.copy()
    env["PATH"] = ":".join(extra) + ":" + env.get("PATH", "")
    npm = shutil.which("npm", path=env["PATH"])
    if not npm:
        console.print("[red]✗[/red] npm not found — install Node.js 18+")
        console.print("  [cyan]brew install node[/cyan]  or  [cyan]https://nodejs.org[/cyan]")
        sys.exit(1)

    node_modules = ui_dir / "node_modules"
    if not node_modules.exists() or install_only:
        console.print("Installing dependencies…")
        _sp.run([npm, "install"], cwd=str(ui_dir), check=True, env=env)
        if install_only:
            console.print("[green]✓ Dependencies installed.[/green]")
            return

    console.print(f"Starting Chat UI on [cyan]http://localhost:{port}[/cyan]")
    console.print("[dim]Login: admin / aua-admin  (set AUA_USERS env to change)[/dim]")
    try:
        _sp.run([npm, "run", "dev", "--", "--port", str(port)], cwd=str(ui_dir), env=env)
    except KeyboardInterrupt:
        console.print("\n[dim]Chat UI stopped.[/dim]")


# ── aua guard ─────────────────────────────────────────────────────────────────


@main.group()
def guard():
    """Manage and test AUA assertions.

    \b
    Assertions are user-defined checks that run against specialist responses
    before they are returned. They implement the Policy-as-Curriculum pattern.

    \b
    Examples:
        aua guard list
        aua guard test --import-path mypackage.policies:validate_syntax
    """
    pass


@guard.command("list")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def guard_list(as_json):
    """List all registered assertions.

    \b
    Shows built-in assertions and any loaded via extensions.
    """
    import json as _json

    from aua.guard import list_assertions

    items = list_assertions()
    if as_json:
        print(_json.dumps(items, indent=2))
        return

    from rich.table import Table

    table = Table(title="Registered Assertions", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Level", style="yellow")
    table.add_column("Bonus", justify="right")
    table.add_column("Max Retries", justify="right")
    table.add_column("Description")

    for a in items:
        bonus = f"+{a['bonus']:.2f}" if a["bonus"] > 0 else "—"
        table.add_row(a["name"], a["level"], bonus, str(a["max_retries"]), a["doc"])

    console.print(table)


@guard.command("test")
@click.option(
    "--import-path",
    "import_path",
    required=True,
    help="'module:function' path to an @assertion decorated function.",
)
@click.option(
    "--output", "-o", default=None, help="Response text to test against (default: built-in sample)."
)
@click.option(
    "--domain", default="software_engineering", help="Domain context passed to assertion."
)
def guard_test(import_path, output, domain):
    """Test an assertion against sample or provided output.

    \b
    Examples:
        aua guard test --import-path mypackage.policies:validate_syntax
        aua guard test --import-path mypackage.policies:check_brand \\
            --output "This leverages our synergy."
    """
    from aua.guard import load_assertion

    try:
        fn = load_assertion(import_path)
    except (ImportError, TypeError, ValueError) as e:
        console.print(f"[red]✗ Load error:[/red] {e}")
        return

    test_output = output or (
        "def binary_search(arr, target):\n"
        "    low, mid, high = 0, 0, len(arr) - 1\n"
        "    while low <= high:\n"
        "        mid = (low + high) // 2\n"
        "        if arr[mid] == target: return mid\n"
        "        elif arr[mid] < target: low = mid + 1\n"
        "        else: high = mid - 1\n"
        "    return -1"
    )
    context = {"query": "test", "session_id": "test", "domain": domain, "field": domain}
    passed, message = fn(test_output, context)
    icon = "[green]✓ PASSED[/green]" if passed else "[red]✗ FAILED[/red]"
    console.print(f"\nAssertion: [bold]{fn.name}[/bold] ({fn.level.value})")
    console.print(f"Result:    {icon}")
    if message:
        console.print(f"Message:   {message}")
    if fn.level.value == "info" and passed and message:
        console.print(f"E bonus:   [cyan]+{fn.bonus:.2f}[/cyan] would be applied")


# ── aua policy ────────────────────────────────────────────────────────────────


@main.group()
def policy():
    """Manage AUA policies — named bundles of assertions.

    \b
    A Policy is a versioned, portable definition of what 'good output'
    means for your use case. It bundles assertions (guardrails and
    incentives) and optional utility weight overrides.

    \b
    Examples:
        aua policy list
        aua policy validate policies/brand_voice.yaml
        aua policy apply policies/brand_voice.yaml
    """
    pass


@policy.command("list")
def policy_list():
    """List policy YAML files in the policies/ directory."""

    policies_dir = Path("policies")
    if not policies_dir.exists():
        console.print("[yellow]No policies/ directory found.[/yellow]")
        console.print("Create one: [cyan]mkdir policies[/cyan]")
        return

    files = sorted(policies_dir.glob("*.yaml")) + sorted(policies_dir.glob("*.yml"))
    if not files:
        console.print("[yellow]No .yaml files found in policies/[/yellow]")
        return

    from rich.table import Table

    from aua.policy import validate_policy_yaml

    table = Table(title="Policies", show_lines=False)
    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Name")
    table.add_column("Assertions", justify="right")

    for f in files:
        errors = validate_policy_yaml(f)
        if errors:
            status = "[red]✗ invalid[/red]"
            name = "—"
            n = "—"
        else:
            import yaml as _yaml

            raw = _yaml.safe_load(f.read_text())
            name = raw.get("name", "—")
            n = str(len(raw.get("assertions", [])))
            status = "[green]✓ valid[/green]"
        table.add_row(f.name, status, name, n)

    console.print(table)


@policy.command("validate")
@click.argument("path")
def policy_validate(path):
    """Validate a policy YAML file.

    \b
    Checks schema, field names, level values, and bonus ranges.
    Does NOT import assertion functions (use `aua guard test` for that).

    \b
    Example:
        aua policy validate policies/brand_voice.yaml
    """
    from aua.policy import validate_policy_yaml

    errors = validate_policy_yaml(path)
    if errors:
        console.print(f"[red]✗ {len(errors)} error(s) in {path}:[/red]")
        for e in errors:
            console.print(f"  • {e}")
    else:
        console.print(f"[green]✓ {path} is valid[/green]")


@policy.command("apply")
@click.argument("path")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Show what would be applied without activating."
)
def policy_apply(path, dry_run):
    """Apply a policy — write its path to .aua/active_policy.

    \b
    The router reads .aua/active_policy on startup (or hot-reload).
    Set policy in config for permanent activation:

    \b
        # aua_config.yaml
        policy:
          path: policies/brand_voice.yaml

    \b
    Example:
        aua policy apply policies/brand_voice.yaml
        aua policy apply policies/brand_voice.yaml --dry-run
    """
    from aua.policy import load_policy, validate_policy_yaml

    errors = validate_policy_yaml(path)
    if errors:
        console.print("[red]✗ Policy has errors — fix before applying:[/red]")
        for e in errors:
            console.print(f"  • {e}")
        return

    pol = load_policy(path)

    console.print(f"\n[bold]Policy:[/bold] {pol.name} v{pol.version}")
    console.print(f"  Max retries:     {pol.max_retries}")
    console.print(f"  Max E bonus:     +{pol.max_total_bonus}")
    if pol.utility_overrides:
        console.print(f"  Weight overrides: {pol.utility_overrides}")
    console.print(f"  Assertions ({len(pol.assertions)}):")
    for a in pol.assertions:
        bonus = f"  +{a.bonus:.2f} E bonus" if a.level.value == "info" and a.bonus else ""
        console.print(f"    [{a.level.value.upper()}] {a.name}{bonus}")

    if dry_run:
        console.print("\n[yellow]--dry-run: policy NOT activated[/yellow]")
        return

    pointer = Path(".aua") / "active_policy"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(Path(path).resolve()))
    console.print("\n[green]✓ Policy activated.[/green] Restart or hot-reload to apply.")
    console.print(f"  [dim]Pointer: {pointer}[/dim]")


# ── aua calibrate ─────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--layer",
    type=click.Choice(["1", "2", "3"]),
    required=True,
    help="Calibration layer (1=eval, 2=routing weights, 3=DPO export).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Layer 3: export even if below min_pairs threshold.",
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Show what would happen without writing files."
)
@click.option("--config", "-c", default="aua_config.yaml", show_default=True)
@click.option(
    "--dataset", default=None, help="Layer 1: eval dataset path (default: evals/coding_smoke.yaml)."
)
@click.option(
    "--output",
    "-o",
    default="dpo_pairs/calibration.jsonl",
    help="Layer 3: output path for DPO pairs.",
    show_default=True,
)
@click.option(
    "--min-pairs",
    default=10,
    show_default=True,
    type=int,
    help="Layer 3: minimum pairs required before export (skip --force).",
)
def calibrate(layer, force, dry_run, config, dataset, output, min_pairs):
    """Run a calibration cycle against the live router.

    \b
    Three layers correspond to the three feedback loops:

    \b
      --layer 1   Run the eval harness (same as `aua eval run`).
                  Measures whether the router is performing well now.

      --layer 2   Recompute routing weight history from session data.
                  Shows which specialists are trending up or down in U score.

      --layer 3   Export gold-standard DPO pairs from assertion events.
                  Gold-standard = sessions where all INFO assertions fired
                  and no BLOCKING assertion failed after retries.
                  Use the exported JSONL to fine-tune your local specialists.

    \b
    Examples:
        aua calibrate --layer 1
        aua calibrate --layer 2
        aua calibrate --layer 3 --dry-run
        aua calibrate --layer 3 --force --output dpo_pairs/may_calibration.jsonl
    """
    import json as _json
    import time as _time

    if layer == "1":
        # Layer 1 — run eval harness
        ds = dataset or "evals/coding_smoke.yaml"
        if not Path(ds).exists():
            console.print(f"[red]✗ Dataset not found:[/red] {ds}")
            console.print("  Specify: [cyan]aua calibrate --layer 1 --dataset <path>[/cyan]")
            return
        console.print(f"[dim]Layer 1 calibration — running eval against {ds}[/dim]")
        if dry_run:
            console.print("[yellow]--dry-run: would run:[/yellow]")
            console.print(f"  aua eval run --dataset {ds} --config {config}")
            return
        import subprocess

        result = subprocess.run(
            ["aua", "eval", "run", "--dataset", ds, "--config", config, "--json"],
            capture_output=False,
        )
        if result.returncode == 0:
            console.print("\n[green]✓ Layer 1 calibration complete.[/green]")
        else:
            console.print("\n[red]✗ Eval run failed.[/red]")

    elif layer == "2":
        # Layer 2 — routing weight analysis from session history
        from aua.state import get_state_store

        store = get_state_store()
        console.print("[dim]Layer 2 calibration — analysing routing weight history[/dim]\n")

        # Get assertion events to compute per-specialist adherence
        events = store.query("assertion_events", limit=10000)
        if not events:
            console.print("[yellow]No assertion events found.[/yellow]")
            console.print("Run queries with an active policy first:")
            console.print("  [cyan]aua policy apply policies/my_policy.yaml[/cyan]")
            return

        from collections import defaultdict

        domain_pass: dict[str, list[bool]] = defaultdict(list)
        domain_bonus: dict[str, list[float]] = defaultdict(list)

        for e in events:
            domain = e.get("domain", "unknown")
            domain_pass[domain].append(bool(e.get("passed", 0)))
            if e.get("bonus_applied", 0) > 0:
                domain_bonus[domain].append(float(e["bonus_applied"]))

        from rich.table import Table

        table = Table(title="Layer 2 — Routing Weight Analysis", show_lines=True)
        table.add_column("Domain", style="cyan")
        table.add_column("Queries", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Avg E Bonus", justify="right")
        table.add_column("Signal")

        for domain, results in sorted(domain_pass.items()):
            n = len(results)
            pass_rate = sum(results) / n if n else 0
            avg_bonus = (
                sum(domain_bonus[domain]) / len(domain_bonus[domain])
                if domain_bonus[domain]
                else 0.0
            )
            if pass_rate >= 0.85:
                signal = "[green]↑ Strong[/green]"
            elif pass_rate >= 0.60:
                signal = "[yellow]→ Stable[/yellow]"
            else:
                signal = "[red]↓ Weak[/red]"
            table.add_row(
                domain,
                str(n),
                f"{pass_rate:.1%}",
                f"+{avg_bonus:.3f}" if avg_bonus > 0 else "—",
                signal,
            )

        if dry_run:
            console.print("[yellow]--dry-run: weight analysis (no changes made):[/yellow]\n")
        console.print(table)
        console.print(
            "\n[dim]Specialists with weak pass rates should be reviewed or retrained.[/dim]"
        )
        console.print(
            "[dim]Use `aua calibrate --layer 3` to export DPO pairs for fine-tuning.[/dim]"
        )

    elif layer == "3":
        # Layer 3 — export gold-standard DPO pairs
        from aua.state import get_state_store

        store = get_state_store()
        console.print("[dim]Layer 3 calibration — exporting gold-standard DPO pairs[/dim]\n")

        # Find sessions where all INFO assertions fired and no BLOCKING failed
        all_events = store.query("assertion_events", limit=50000)
        if not all_events:
            console.print("[yellow]No assertion events found.[/yellow]")
            console.print("Run queries with an active policy first.")
            return

        # Group by session
        from collections import defaultdict

        session_events: dict[str, list[dict]] = defaultdict(list)
        for e in all_events:
            session_events[e["session_id"]].append(e)

        chosen_sessions = []
        rejected_sessions = []

        for session_id, events in session_events.items():
            blocking_failed = any(
                e["level"] == "blocking"
                and not bool(e["passed"])
                and e.get("retries_used", 0) >= 3  # exhausted retries
                for e in events
            )
            info_events = [e for e in events if e["level"] == "info"]
            all_info_fired = len(info_events) > 0 and all(
                bool(e["passed"]) and e.get("message") for e in info_events
            )

            if all_info_fired and not blocking_failed:
                chosen_sessions.append(session_id)
            elif blocking_failed:
                rejected_sessions.append(session_id)

        n_chosen = len(chosen_sessions)
        n_pairs = min(n_chosen, len(rejected_sessions))

        console.print(f"  Gold-standard sessions:  [green]{n_chosen}[/green]")
        console.print(f"  Failed sessions:         {len(rejected_sessions)}")
        console.print(f"  Exportable pairs:        [cyan]{n_pairs}[/cyan]")
        console.print()

        if n_pairs < min_pairs and not force:
            console.print(
                f"[yellow]⚠ Only {n_pairs} pairs found (min: {min_pairs}).[/yellow]\n"
                f"  Run more queries to accumulate data, or use --force to export anyway."
            )
            return

        if dry_run:
            console.print(
                f"[yellow]--dry-run: would export {n_pairs} DPO pairs → {output}[/yellow]"
            )
            console.print(
                "\n[dim]Each pair: chosen (gold-standard session) + rejected (failed session).[/dim]"
            )
            console.print("[dim]Fine-tune your local specialist with these pairs:[/dim]")
            console.print("[dim]  Axolotl: axolotl train configs/dpo.yaml[/dim]")
            console.print("[dim]  TRL:     trl dpo --dataset dpo_pairs/calibration.jsonl[/dim]")
            return

        # Get session messages to build pairs
        pairs = []
        corrections_data = store.query("corrections", limit=5000)
        corrections_by_session: dict[str, list] = defaultdict(list)
        for c in corrections_data:
            if "session_id" in c:
                corrections_by_session[c["session_id"]].append(c)

        for chosen_id, rejected_id in zip(chosen_sessions, rejected_sessions):
            pairs.append(
                {
                    "chosen_session_id": chosen_id,
                    "rejected_session_id": rejected_id,
                    "chosen": f"[Gold-standard session {chosen_id}]",
                    "rejected": f"[Failed-assertion session {rejected_id}]",
                    "metadata": {
                        "source": "aua_calibrate_layer3",
                        "exported_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                        "chosen_assertions_fired": len(
                            [e for e in session_events[chosen_id] if e["level"] == "info"]
                        ),
                    },
                }
            )

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = "\n".join(_json.dumps(p, default=str) for p in pairs)
        out_path.write_text(lines)

        console.print(f"[green]✓ Exported {len(pairs)} DPO pairs → {output}[/green]")
        console.print()
        console.print("[dim]Fine-tune your specialist with these pairs:[/dim]")
        console.print(f"[dim]  Axolotl:  axolotl train configs/dpo.yaml --data {output}[/dim]")
        console.print(f"[dim]  TRL:      trl dpo --dataset {output}[/dim]")
        console.print(
            "[dim]  Then deploy as GREEN: curl -X POST http://localhost:8000/deploy/green[/dim]"
        )


# ── aua logs ──────────────────────────────────────────────────────────────────


@main.group()
def logs():
    """Query and export AUA session and assertion logs.

    \b
    Examples:
        aua logs sessions
        aua logs assertions --filter domain=software_engineering
        aua logs assertions --filter "passed=false" --tail 20
        aua logs export --output my_logs.json
    """
    pass


@logs.command("sessions")
@click.option("--limit", default=20, show_default=True, type=int)
@click.option("--domain", default=None, help="Filter by domain.")
@click.option("--json", "as_json", is_flag=True, default=False)
def logs_sessions(limit, domain, as_json):
    """Show recent sessions with U scores and routing info."""
    import json as _json

    from aua.state import get_state_store

    store = get_state_store()
    filters = {}
    if domain:
        filters["domain"] = domain
    records = store.query("audit_log", filters={"event_type": "query"}, limit=limit)

    if as_json:
        print(_json.dumps(records, indent=2, default=str))
        return

    if not records:
        console.print("[yellow]No session records found.[/yellow]")
        return

    from rich.table import Table

    table = Table(title=f"Recent Sessions (last {limit})", show_lines=False)
    table.add_column("Session", style="dim")
    table.add_column("Domain", style="cyan")
    table.add_column("U Score", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Latency", justify="right")

    for r in records[:limit]:
        table.add_row(
            (r.get("session_id") or "")[:12],
            r.get("field") or "—",
            f"{r.get('u_score', 0):.3f}",
            f"{r.get('confidence', 0):.3f}",
            f"{r.get('latency_ms', 0):.0f}ms",
        )
    console.print(table)


@logs.command("assertions")
@click.option(
    "--filter",
    "filter_str",
    default=None,
    help="Filter: 'key=value' e.g. 'domain=software_engineering' or 'passed=false'.",
)
@click.option("--tail", default=None, type=int, help="Show last N assertion events.")
@click.option("--assertion", "assertion_name", default=None, help="Filter by assertion name.")
@click.option("--json", "as_json", is_flag=True, default=False)
def logs_assertions(filter_str, tail, assertion_name, as_json):
    """Show assertion events — which assertions fired and with what result.

    \b
    Examples:
        aua logs assertions --filter passed=false
        aua logs assertions --assertion PythonSyntaxCheck --tail 10
        aua logs assertions --filter domain=software_engineering
    """
    import json as _json

    from aua.state import get_state_store

    store = get_state_store()
    filters: dict = {}
    if filter_str:
        key, _, val = filter_str.partition("=")
        key = key.strip()
        val = val.strip()
        if key == "passed":
            filters["passed"] = 1 if val.lower() in ("true", "1", "yes") else 0
        else:
            filters[key] = val
    if assertion_name:
        filters["assertion_name"] = assertion_name

    limit = tail or 50
    records = store.query("assertion_events", filters=filters, limit=limit)

    if as_json:
        print(_json.dumps(records, indent=2, default=str))
        return

    if not records:
        console.print("[yellow]No assertion events found.[/yellow]")
        return

    from rich.table import Table

    table = Table(title=f"Assertion Events (last {limit})", show_lines=False)
    table.add_column("Assertion", style="cyan")
    table.add_column("Level")
    table.add_column("Result", justify="center")
    table.add_column("Bonus", justify="right")
    table.add_column("Retries", justify="right")
    table.add_column("Domain")
    table.add_column("Message", max_width=40)

    for r in records:
        passed = bool(r.get("passed", 0))
        result = "[green]✓[/green]" if passed else "[red]✗[/red]"
        bonus = f"+{r['bonus_applied']:.2f}" if r.get("bonus_applied", 0) > 0 else "—"
        level_color = {
            "blocking": "[red]blocking[/red]",
            "soft": "[yellow]soft[/yellow]",
            "info": "[green]info[/green]",
        }.get(r.get("level", ""), r.get("level", ""))

        table.add_row(
            r.get("assertion_name", ""),
            level_color,
            result,
            bonus,
            str(r.get("retries_used", 0)),
            r.get("domain", ""),
            (r.get("message") or "")[:40],
        )

    console.print(table)


@logs.command("export")
@click.option("--output", "-o", default="logs_export.json", show_default=True)
@click.option(
    "--table",
    "table_name",
    default="assertion_events",
    type=click.Choice(["assertion_events", "audit_log", "sessions", "corrections"]),
    show_default=True,
)
@click.option("--limit", default=10000, show_default=True, type=int)
def logs_export(output, table_name, limit):
    """Export session or assertion logs to JSON.

    \b
    Examples:
        aua logs export --output my_assertions.json
        aua logs export --table audit_log --output audit.json
    """
    import json as _json

    from aua.state import get_state_store

    store = get_state_store()
    records = store.query(table_name, limit=limit)
    Path(output).write_text(_json.dumps(records, indent=2, default=str))
    console.print(f"[green]✓ Exported {len(records)} records → {output}[/green]")


# ── aua metrics compare ────────────────────────────────────────────────────────


@main.command("metrics")
@click.option(
    "--compare",
    "compare_window",
    default=None,
    help="Compare time windows: '7d', '30d', or 'YYYY-MM-DD:YYYY-MM-DD'.",
)
@click.option(
    "--metric",
    default=None,
    help="Focus on a specific metric: u_score, assertion_fail_rate, retry_rate.",
)
@click.option("--json", "as_json", is_flag=True, default=False)
def metrics_compare(compare_window, metric, as_json):
    """Compare AUA metrics across time windows.

    \b
    Shows how the system is performing over time — the key signal that
    the policy feedback loop is working is assertion_fail_rate trending down
    and u_score trending up.

    \b
    Examples:
        aua metrics --compare 30d
        aua metrics --compare 7d --metric assertion_fail_rate
        aua metrics --compare 2025-04-01:2025-05-01
    """
    import json as _json
    import time as _time

    from aua.state import get_state_store

    store = get_state_store()
    now = _time.time()

    # Parse window
    if compare_window and ":" in compare_window and not compare_window.endswith("d"):
        try:
            from datetime import datetime

            parts = compare_window.split(":")
            t_from = datetime.strptime(parts[0], "%Y-%m-%d").timestamp()
            t_to = datetime.strptime(parts[1], "%Y-%m-%d").timestamp()
            window_seconds = t_to - t_from
            current_start = t_from
            prior_start = t_from - window_seconds
        except ValueError:
            console.print("[red]Invalid date range. Use YYYY-MM-DD:YYYY-MM-DD[/red]")
            return
    else:
        days = int((compare_window or "30d").rstrip("d"))
        window_seconds = days * 86400
        current_start = now - window_seconds
        prior_start = now - 2 * window_seconds

    # Pull assertion events for both windows
    def _stats(start: float, end: float) -> dict:
        events = [
            e
            for e in store.query("assertion_events", limit=50000)
            if start <= (e.get("created_at") or 0) <= end
        ]
        queries = [
            q
            for q in store.query("audit_log", filters={"event_type": "query"}, limit=50000)
            if start <= (q.get("created_at") or 0) <= end
        ]
        n_events = len(events)
        n_queries = len(queries)
        n_failed = sum(1 for e in events if not bool(e.get("passed", 1)))
        n_retries = sum(e.get("retries_used", 0) for e in events)
        bonuses = [e["bonus_applied"] for e in events if e.get("bonus_applied", 0) > 0]
        u_scores = [q["u_score"] for q in queries if q.get("u_score") is not None]
        return {
            "n_queries": n_queries,
            "n_assertion_events": n_events,
            "assertion_fail_rate": round(n_failed / n_events, 4) if n_events else 0.0,
            "retry_rate": round(n_retries / n_events, 4) if n_events else 0.0,
            "avg_e_bonus": round(sum(bonuses) / len(bonuses), 4) if bonuses else 0.0,
            "mean_u_score": round(sum(u_scores) / len(u_scores), 4) if u_scores else 0.0,
        }

    current = _stats(current_start, now)
    prior = _stats(prior_start, current_start)

    def _delta(key: str) -> str:
        c, p = current.get(key, 0), prior.get(key, 0)
        diff = c - p
        if abs(diff) < 0.001:
            return "[dim]→ no change[/dim]"
        # For fail/retry rates, lower is better
        if key in ("assertion_fail_rate", "retry_rate"):
            return f"[green]↓ {diff:+.4f}[/green]" if diff < 0 else f"[red]↑ {diff:+.4f}[/red]"
        return f"[green]↑ {diff:+.4f}[/green]" if diff > 0 else f"[red]↓ {diff:+.4f}[/red]"

    if as_json:
        print(_json.dumps({"current": current, "prior": prior}, indent=2))
        return

    window_label = compare_window or "30d"
    from rich.table import Table

    table = Table(
        title=f"Metrics Comparison — last {window_label} vs prior {window_label}", show_lines=True
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Prior", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Trend")

    metrics_to_show = [
        ("mean_u_score", "Mean U score"),
        ("assertion_fail_rate", "Assertion fail rate"),
        ("retry_rate", "Retry rate (BLOCKING)"),
        ("avg_e_bonus", "Avg E bonus (INFO)"),
        ("n_queries", "Total queries"),
        ("n_assertion_events", "Total assertion events"),
    ]

    if metric:
        m_map = {
            "u_score": "mean_u_score",
            "assertion_fail_rate": "assertion_fail_rate",
            "retry_rate": "retry_rate",
        }
        key = m_map.get(metric, metric)
        metrics_to_show = [(k, lbl) for k, lbl in metrics_to_show if k == key]

    for key, label in metrics_to_show:
        p_val = prior.get(key, 0)
        c_val = current.get(key, 0)
        table.add_row(label, str(p_val), str(c_val), _delta(key))

    console.print(table)
    console.print()
    console.print("[dim]Success signal: mean_u_score ↑, assertion_fail_rate ↓, retry_rate ↓[/dim]")
    console.print("[dim]Stagnation signal: same assertions failing week over week[/dim]")
    console.print(
        "[dim]Run `aua calibrate --layer 3` to export gold-standard sessions for fine-tuning.[/dim]"
    )
