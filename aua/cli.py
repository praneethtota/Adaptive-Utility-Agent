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
    data = {"plugins": list(reg._plugins.keys()), "middleware": mw, "hooks": hooks}  # type: ignore[attr-defined]

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
