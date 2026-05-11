"""
aua/rollback.py — One-command revert to the previous BLUE model.

Rollback works from a promotions log at .aua/state/promotions.jsonl.
(Configurable via RuntimeConfig.)
Every blue-green promotion records a PromotionEvent there. Rollback
reads the log, finds the last non-reverted promotion for the target
specialist, reverts aua_config.yaml, and restarts the specialist server.

Usage (programmatic):
    from aua.rollback import run_rollback
    run_rollback("aua_config.yaml", specialist="swe")

CLI:
    aua rollback --specialist swe
    aua rollback --all
    aua rollback --specialist swe --yes   # skip confirmation
    aua rollback --no-restart             # update config only, don't restart server
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from filelock import FileLock
from rich.console import Console

console = Console()

PROMOTIONS_FILE = "results/aua_promotions.json"  # legacy path — kept for migration
STATE_FILE = ".aua/state/promotions.jsonl"  # canonical path (P-09+)
HEALTH_TIMEOUT = 120
POLL_INTERVAL = 3.0


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class PromotionEvent:
    """One entry in the promotions log."""

    id: str
    timestamp: str
    specialist: str
    from_model: str  # model before promotion (the BLUE model)
    to_model: str  # model after promotion (the GREEN model)
    event: str  # "promote" | "rollback"
    u_delta: float = 0.0  # U improvement that triggered promotion
    reverted: bool = False
    reverted_at: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# ── Promotions log ────────────────────────────────────────────────────────────


def _state_path(project_dir: str) -> Path:
    """Return the canonical .aua/state/promotions.jsonl path."""
    return Path(project_dir) / STATE_FILE


def _lock_path(project_dir: str) -> Path:
    return Path(project_dir) / ".aua/state/promotions.lock"


def load_promotions(project_dir: str = ".") -> list[PromotionEvent]:
    """Load the promotions log from .aua/state/promotions.jsonl.

    Also migrates legacy results/aua_promotions.json if present.
    File-locked for safe concurrent access.
    """
    state = _state_path(project_dir)
    legacy = Path(project_dir) / PROMOTIONS_FILE

    # Migrate legacy JSON → JSONL on first access
    if not state.exists() and legacy.exists():
        state.parent.mkdir(parents=True, exist_ok=True)
        try:
            old_events = json.loads(legacy.read_text())
            with state.open("w") as f:
                for e in old_events:
                    f.write(json.dumps(e) + "\n")
            console.print(f"  [dim]Migrated promotions log: {legacy} → {state}[/dim]")
        except Exception:
            pass

    if not state.exists():
        return []

    lock = FileLock(str(_lock_path(project_dir)))
    try:
        with lock.acquire(timeout=10):
            lines = state.read_text().splitlines()
        events = []
        for line in lines:
            line = line.strip()
            if line:
                events.append(PromotionEvent(**json.loads(line)))
        return events
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow]  Could not read promotions log: {e}")
        return []


def save_promotions(events: list[PromotionEvent], project_dir: str = ".") -> None:
    """Write the promotions log atomically to .aua/state/promotions.jsonl.

    Uses file locking and atomic replace (.tmp → final) to prevent corruption
    from concurrent rollback+deploy operations.
    """
    state = _state_path(project_dir)
    state.parent.mkdir(parents=True, exist_ok=True)

    tmp = state.with_suffix(".jsonl.tmp")
    lock = FileLock(str(_lock_path(project_dir)))

    try:
        with lock.acquire(timeout=10):
            # Write to .tmp first (atomic within same filesystem)
            tmp.write_text("\n".join(json.dumps(e.as_dict()) for e in events) + "\n")
            # Atomic rename — POSIX guarantees this is atomic
            os.replace(str(tmp), str(state))
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow]  Could not save promotions log: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def record_promotion(
    specialist: str,
    from_model: str,
    to_model: str,
    u_delta: float = 0.0,
    project_dir: str = ".",
) -> PromotionEvent:
    """
    Record a blue-green promotion. Called by the promotion checker after a
    successful promotion. Returns the new event.
    """
    events = load_promotions(project_dir)
    event = PromotionEvent(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        specialist=specialist,
        from_model=from_model,
        to_model=to_model,
        event="promote",
        u_delta=u_delta,
        reverted=False,
    )
    events.append(event)
    save_promotions(events, project_dir)
    return event


# ── Core rollback logic ───────────────────────────────────────────────────────


def run_rollback(
    config_path: str = "aua_config.yaml",
    specialist: str | None = None,
    all_specialists: bool = False,
    yes: bool = False,
    restart: bool = True,
    dry_run: bool = False,
) -> int:
    """
    Revert specialist(s) to their previous BLUE model.

    Args:
        config_path:     path to aua_config.yaml
        specialist:      name of specialist to roll back (e.g. "swe")
        all_specialists: if True, roll back all specialists with promotions
        yes:             skip confirmation prompt
        restart:         restart the server after config update (default True)

    Returns:
        0 on success, 1 on failure.
    """
    from aua.config import load_config

    # ── Load config ───────────────────────────────────────────────────────
    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]✗  Config error:[/red] {e}")
        return 1

    project_dir = str(Path(config_path).parent)

    # ── Determine targets ─────────────────────────────────────────────────
    if all_specialists:
        targets = [s.name for s in cfg.specialists]
    elif specialist:
        known = [s.name for s in cfg.specialists]
        if specialist not in known:
            console.print(
                f"[red]✗  Unknown specialist '[bold]{specialist}[/bold]'.[/red]  " f"Known: {known}"
            )
            return 1
        targets = [specialist]
    else:
        console.print("[red]✗  Specify --specialist NAME or --all.[/red]")
        return 1

    # ── Load promotions log ───────────────────────────────────────────────
    events = load_promotions(project_dir)

    # ── Process each target ───────────────────────────────────────────────
    n_errors = 0
    for target_name in targets:
        # Check if there's anything to roll back before attempting
        promotions = [
            e
            for e in events
            if e.specialist == target_name and e.event == "promote" and not e.reverted
        ]
        if not promotions:
            if all_specialists:
                # --all: silently skip specialists with no history
                console.print(f"  [dim]{target_name}: no un-reverted promotions — skipping[/dim]")
                continue
        result = _rollback_one(
            target_name, cfg, config_path, project_dir, events, yes, restart, dry_run=dry_run
        )
        if result != 0:
            n_errors += 1

    return 0 if n_errors == 0 else 1


def _rollback_one(
    name: str,
    cfg,
    config_path: str,
    project_dir: str,
    events: list[PromotionEvent],
    yes: bool,
    restart: bool,
    dry_run: bool = False,
) -> int:
    """Roll back a single specialist. Returns 0 on success, 1 on failure."""

    spec = next((s for s in cfg.specialists if s.name == name), None)
    if spec is None:
        console.print(f"[red]✗  Specialist '{name}' not found in config.[/red]")
        return 1

    # ── Find the last non-reverted promotion for this specialist ──────────
    promotions = [
        e for e in events if e.specialist == name and e.event == "promote" and not e.reverted
    ]

    if not promotions:
        _print_no_promotion(name, spec, project_dir)
        return 1

    last = promotions[-1]
    blue_model = last.from_model  # what we're rolling back TO
    green_model = last.to_model  # what's currently deployed

    # ── Confirm ───────────────────────────────────────────────────────────
    if not yes:
        console.print(f"\n[bold]Rollback plan for [cyan]{name}[/cyan]:[/bold]")
        console.print(f"  Current model : [red]{green_model}[/red]")
        console.print(f"  Restore to    : [green]{blue_model}[/green]")
        console.print(
            f"  Promoted at   : [dim]{last.timestamp}[/dim]  " f"U_delta={last.u_delta:+.4f}"
        )
        if not _confirm("Proceed with rollback?"):
            console.print("[dim]Rollback cancelled.[/dim]")
            return 0

    console.print(
        f"\n[bold]Rolling back [cyan]{name}[/cyan][/bold]  "
        f"[dim]{green_model} → {blue_model}[/dim]"
    )

    if dry_run:
        console.print("  [yellow]DRY RUN[/yellow] — no changes made")
        console.print(f"  Would update: {config_path}  ({name}.model = {blue_model})")
        console.print(f"  Would save rollback event to: {_state_path(project_dir)}")
        return 0

    # ── 1. Update aua_config.yaml ─────────────────────────────────────────
    _update_config_model(config_path, name, blue_model)
    console.print(
        f"  [green]✓[/green] aua_config.yaml updated  " f"[dim]({name}.model = {blue_model})[/dim]"
    )

    # ── 2. Restart server (vLLM only; Ollama is self-managing) ───────────
    if restart and cfg.backend == "vllm":
        _restart_specialist(spec, blue_model)
    elif cfg.backend == "ollama":
        console.print(
            "  [dim]Ollama backend — no restart needed (model change takes effect on next query)[/dim]"
        )

    # ── 3. Mark promotion as reverted in log ──────────────────────────────
    last.reverted = True
    last.reverted_at = datetime.now(timezone.utc).isoformat()

    # Record rollback event
    rollback_event = PromotionEvent(
        id=str(uuid.uuid4()),
        timestamp=last.reverted_at,
        specialist=name,
        from_model=green_model,
        to_model=blue_model,
        event="rollback",
        reverted=False,
    )
    events.append(rollback_event)
    save_promotions(events, project_dir)
    console.print("  [green]✓[/green] Promotions log updated")

    console.print(f"\n[bold green]✓ {name} rolled back to {blue_model}[/bold green]")
    return 0


def _restart_specialist(spec, blue_model: str) -> None:
    """Kill the running vLLM process on the specialist port and restart it."""
    from aua.serve import _wait_healthy

    port = spec.port
    console.print(f"  Stopping server on port [cyan]{port}[/cyan]...")

    # Find and kill the process
    pid = _find_pid_on_port(port)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait up to 15s for graceful shutdown
            for _ in range(15):
                time.sleep(1)
                try:
                    os.kill(pid, 0)  # check if still alive
                except ProcessLookupError:
                    break
            else:
                os.kill(pid, signal.SIGKILL)  # force kill
            console.print(f"  [green]✓[/green] Stopped (pid {pid})")
        except ProcessLookupError:
            console.print(f"  [dim]Process {pid} already stopped[/dim]")
        except PermissionError:
            console.print(
                f"  [yellow]⚠[/yellow]  Cannot kill pid {pid} — "
                f"permission denied. Stop the server manually and re-run."
            )
            return
    else:
        console.print(f"  [dim]No process found on port {port} — server not running[/dim]")
        console.print(f"  [dim]Config updated; run 'aua serve' to start with {blue_model}[/dim]")
        return

    # Rebuild the vllm command with the blue model
    import copy

    spec_copy = copy.copy(spec)
    spec_copy.model = blue_model
    cmd = spec_copy.vllm_command()

    console.print(f"  Starting [cyan]{spec.name}[/cyan] with BLUE model...")
    console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")

    env = os.environ.copy()
    if spec.backend == "vllm":
        env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)

    p = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _wait_healthy(spec.name, spec.models_url, p, timeout=HEALTH_TIMEOUT)
    console.print(f"  [green]✓[/green] {spec.name} healthy on port {port}")


# ── Config patching ───────────────────────────────────────────────────────────


def _update_config_model(config_path: str, specialist_name: str, new_model: str) -> None:
    """
    Update the model: field for one specialist in aua_config.yaml in-place.
    Uses PyYAML to preserve structure; writes back with consistent formatting.
    """
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text())

    for s in raw.get("specialists", []):
        if s.get("name") == specialist_name:
            s["model"] = new_model
            break

    # Atomic write: .tmp → replace (prevents partial writes on crash)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(
        "# aua_config.yaml — updated by aua rollback\n"
        + yaml.dump(raw, default_flow_style=False, sort_keys=False, allow_unicode=True)
    )
    os.replace(str(tmp), str(path))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_pid_on_port(port: int) -> int | None:
    """Return the PID listening on port, or None."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True, timeout=3
        )
        pids = result.stdout.strip().splitlines()
        return int(pids[0]) if pids else None
    except Exception:
        return None


def _confirm(prompt: str) -> bool:
    """Interactive yes/no prompt. Returns True if user confirms."""
    try:
        answer = input(f"  {prompt} [y/N] ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _print_no_promotion(name: str, spec, project_dir: str) -> None:
    """Print a helpful message when there's no promotion history."""
    log_path = Path(project_dir) / PROMOTIONS_FILE
    blue_json = Path(project_dir) / "results" / "blue_baseline.json"

    console.print(f"\n[yellow]⚠[/yellow]  No promotion history found for " f"[bold]{name}[/bold].")
    console.print()
    console.print(
        f"  Promotions log: [dim]{log_path}[/dim]  "
        f"({'exists' if log_path.exists() else 'not found'})"
    )

    if blue_json.exists():
        try:
            baseline = json.loads(blue_json.read_text())
            console.print(
                f"\n  BLUE baseline found at [dim]{blue_json}[/dim]\n"
                f"    accuracy={baseline.get('accuracy', '?')}  "
                f"mean_U={baseline.get('mean_u', '?')}  "
                f"timestamp={baseline.get('timestamp', '?')}"
            )
        except Exception:
            pass

    console.print(
        "\n  [dim]aua rollback only works after a successful promotion.\n"
        "  If GREEN was never promoted, there is nothing to roll back.\n"
        "  To manually revert, update model: in aua_config.yaml and run aua serve.[/dim]"
    )
