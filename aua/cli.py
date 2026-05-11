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
def serve(config, dry_run, no_router, router_only, startup_timeout, tier, with_ui, ui_port):
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
    import shutil
    import subprocess as _sp

    ui_dir = Path(__file__).parent.parent / "apps" / "aua_chat"
    if not ui_dir.exists():
        console.print("[yellow]⚠[/yellow]  Chat UI not found at apps/aua_chat/. Skipping.")
        return

    if not shutil.which("node"):
        console.print("[yellow]⚠[/yellow]  Node.js not found — Chat UI requires Node.js 18+.")
        return

    # Install deps if needed
    node_modules = ui_dir / "node_modules"
    if not node_modules.exists():
        console.print("[dim]Installing Chat UI dependencies (first run)…[/dim]")
        _sp.run(["npm", "install", "--prefer-offline"], cwd=str(ui_dir), check=True)

    console.print(f"[dim]Starting Chat UI on http://localhost:{port}[/dim]")
    _sp.Popen(
        ["npm", "run", "dev", "--", "--port", str(port)],
        cwd=str(ui_dir),
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )
    console.print(
        f"[green]✓[/green] Chat UI: [cyan]http://localhost:{port}[/cyan]  (admin / aua-admin)"
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
    import shutil
    import subprocess as _sp

    ui_dir = Path(__file__).parent.parent / "apps" / "aua_chat"
    if not ui_dir.exists():
        console.print(f"[red]✗[/red] Chat UI not found: {ui_dir}")
        sys.exit(1)
    if not shutil.which("node"):
        console.print("[red]✗[/red] Node.js not found — install Node.js 18+")
        sys.exit(1)

    node_modules = ui_dir / "node_modules"
    if not node_modules.exists() or install_only:
        console.print("Installing dependencies…")
        _sp.run(["npm", "install"], cwd=str(ui_dir), check=True)
        if install_only:
            console.print("[green]✓ Dependencies installed.[/green]")
            return

    console.print(f"Starting Chat UI on [cyan]http://localhost:{port}[/cyan]")
    console.print("[dim]Login: admin / aua-admin  (set AUA_USERS env to change)[/dim]")
    try:
        _sp.run(["npm", "run", "dev", "--", "--port", str(port)], cwd=str(ui_dir))
    except KeyboardInterrupt:
        console.print("\n[dim]Chat UI stopped.[/dim]")
