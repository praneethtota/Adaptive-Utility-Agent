"""
aua/doctor.py — Pre-flight diagnostics for aua doctor.

Runs five check groups in order and prints pass/fail per check with
fix instructions. Designed to be run before aua serve.

Check groups:
  1. Config     — file found, YAML valid, schema valid
  2. Dependencies — required packages installed, backend binary present
  3. Hardware   — CUDA/GPU available, VRAM sufficient, ports free
  4. Models     — local model paths exist (HuggingFace IDs flagged as warnings)
  5. Specialists — live ping (only if already running; warns if not)

Usage:
    from aua.doctor import run_doctor
    run_doctor("aua_config.yaml")

CLI:
    aua doctor
    aua doctor --config /path/to/aua_config.yaml
"""

from __future__ import annotations

import importlib
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console
from rich.text import Text

console = Console()

PASS = ("✓", "bold green")
FAIL = ("✗", "bold red")
WARN = ("⚠", "yellow")
SKIP = ("–", "dim")
INFO = ("·", "dim")

REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "httpx",
    "pydantic",
    "yaml",
    "click",
    "rich",
    "scipy",
]


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Check:
    group: str
    name: str
    status: str  # "pass" | "fail" | "warn" | "skip" | "info"
    detail: str
    fix: str | None = None

    @property
    def symbol(self):
        return {"pass": PASS, "fail": FAIL, "warn": WARN, "skip": SKIP, "info": INFO}.get(
            self.status, INFO
        )


# ── Public API ────────────────────────────────────────────────────────────────


def run_doctor(
    config_path: str = "aua_config.yaml",
    as_json: bool = False,
    strict: bool = False,
) -> int:
    """
    Run all diagnostic checks and print results.

    Args:
        config_path: path to aua_config.yaml
        as_json:     emit JSON instead of Rich terminal output
        strict:      treat WARN as FAIL (exit code 2 in CLI)

    Returns:
        Number of FAIL checks (0 = all good). In strict mode, WARNs also count.
    """
    import json as _json

    checks: list[Check] = []

    # ── Group 1: Config ───────────────────────────────────────────────────
    cfg = None
    checks += _check_config(config_path)
    try:
        from aua.config import load_config

        cfg = load_config(config_path)
    except Exception:
        pass

    # ── Group 2: Dependencies ─────────────────────────────────────────────
    checks += _check_dependencies(cfg)

    # ── Group 3: Hardware ─────────────────────────────────────────────────
    checks += _check_hardware(cfg)

    # ── Group 4: Models ───────────────────────────────────────────────────
    if cfg:
        checks += _check_models(cfg)

    # ── Group 5: Live specialists ─────────────────────────────────────────
    if cfg:
        checks += _check_specialists_live(cfg)

    # ── Output ────────────────────────────────────────────────────────────
    if as_json:
        result = {
            "config": config_path,
            "strict": strict,
            "n_pass": sum(1 for c in checks if c.status == "pass"),
            "n_fail": sum(1 for c in checks if c.status == "fail"),
            "n_warn": sum(1 for c in checks if c.status == "warn"),
            "n_skip": sum(1 for c in checks if c.status in ("skip", "info")),
            "checks": [
                {
                    "group": c.name,
                    "detail": c.detail,
                    "status": c.status,
                    "fix": c.fix or "",
                }
                for c in checks
            ],
        }
        print(_json.dumps(result, indent=2))
    else:
        _print_results(checks, config_path)

    n_fail = sum(1 for c in checks if c.status == "fail")
    if strict:
        n_fail += sum(1 for c in checks if c.status == "warn")
    return n_fail


# ── Check groups ──────────────────────────────────────────────────────────────


def _check_config(config_path: str) -> list[Check]:
    checks = []
    p = Path(config_path)

    # File exists
    if p.exists():
        checks.append(Check("Config", "Config file found", "pass", str(p.resolve())))
    else:
        checks.append(
            Check(
                "Config",
                "Config file found",
                "fail",
                f"{config_path} not found",
                fix="Run 'aua init' to scaffold a starter config, "
                "or copy a tier template:\n"
                "    aua init --tier rtx4090",
            )
        )
        return checks  # no point continuing config checks

    # Parse and validate
    try:
        import yaml

        raw = yaml.safe_load(p.read_text())
        if not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        checks.append(Check("Config", "YAML syntax valid", "pass", ""))
    except Exception as e:
        checks.append(
            Check(
                "Config",
                "YAML syntax valid",
                "fail",
                str(e),
                fix="Fix the YAML syntax error shown above",
            )
        )
        return checks

    # Schema validation via load_config
    try:
        from aua.config import load_config

        cfg = load_config(config_path)
        n_spec = len(cfg.specialists)
        checks.append(
            Check(
                "Config",
                "Schema valid",
                "pass",
                f"{n_spec} specialist(s) · 1 arbiter · backend={cfg.backend}",
            )
        )
    except ValueError as e:
        checks.append(
            Check("Config", "Schema valid", "fail", str(e), fix="Fix the config error shown above")
        )
    except Exception as e:
        checks.append(Check("Config", "Schema valid", "warn", str(e)))

    return checks


def _check_dependencies(cfg) -> list[Check]:
    checks = []
    backend = cfg.backend if cfg else "vllm"

    # Core Python packages
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            checks.append(Check("Dependencies", pkg, "pass", ver))
        except ImportError:
            fix = f"pip install {pkg}" if pkg != "yaml" else "pip install pyyaml"
            checks.append(Check("Dependencies", pkg, "fail", "not installed", fix=fix))

    # Backend binary
    if backend == "ollama":
        if shutil.which("ollama"):
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            ver = result.stdout.strip() or "installed"
            checks.append(Check("Dependencies", "ollama binary", "pass", ver))
        else:
            checks.append(
                Check(
                    "Dependencies",
                    "ollama binary",
                    "fail",
                    "not found in PATH",
                    fix="brew install ollama  (macOS)\n"
                    "    or: curl -fsSL https://ollama.com/install.sh | sh  (Linux)",
                )
            )
    else:  # vllm
        try:
            import vllm

            ver = getattr(vllm, "__version__", "?")
            checks.append(Check("Dependencies", "vllm", "pass", ver))
        except ImportError:
            checks.append(
                Check(
                    "Dependencies",
                    "vllm",
                    "fail",
                    "not installed",
                    fix="pip install vllm  (requires CUDA; install on Linux GPU host)",
                )
            )

    return checks


# ── Hardware detection ────────────────────────────────────────────────────────


@dataclass
class _HWInfo:
    kind: str  # "nvidia" | "amd_rocm" | "apple_silicon" | "cpu"
    devices: list  # list of {"index": int, "name": str, "vram_mib": int or None}
    system_ram_mib: int | None = None  # total system RAM (all platforms)


def _detect_hardware() -> _HWInfo:
    """
    Probe available hardware in a platform-agnostic way.

    Detection order:
      1. nvidia-smi   → NVIDIA GPU (Linux / Windows with CUDA drivers)
      2. rocm-smi     → AMD GPU with ROCm drivers (Linux)
      3. Apple Silicon → macOS arm64 (sysctl hw.memsize + system_profiler)
      4. CPU fallback  → no dedicated GPU found
    """
    import platform
    import shutil
    import subprocess

    system = platform.system()  # "Darwin" | "Linux" | "Windows"
    machine = platform.machine()  # "arm64" | "x86_64" | "AMD64"
    ram_mib = _system_ram_mib()

    # ── 1. NVIDIA ─────────────────────────────────────────────────────────
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                devices = []
                for line in r.stdout.strip().splitlines():
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(", ")]
                    devices.append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "vram_mib": int(parts[2]) if len(parts) > 2 else None,
                        }
                    )
                if devices:
                    return _HWInfo(kind="nvidia", devices=devices, system_ram_mib=ram_mib)
        except Exception:
            pass

    # ── 2. AMD ROCm ──────────────────────────────────────────────────────
    if shutil.which("rocm-smi"):
        try:
            r = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                devices = []
                for i, line in enumerate(r.stdout.strip().splitlines()):
                    # rocm-smi output: "GPU[N] : VRAM Total: XXXX MiB"
                    mib = None
                    import re as _re

                    m = _re.search(r"(\d+)\s+MiB", line)
                    if m:
                        mib = int(m.group(1))
                    devices.append({"index": i, "name": f"AMD GPU {i}", "vram_mib": mib})
                if devices:
                    return _HWInfo(kind="amd_rocm", devices=devices, system_ram_mib=ram_mib)
        except Exception:
            pass
    # Also check for ROCm without rocm-smi (newer ROCm uses amdgpu_info)
    import os as _os

    if _os.path.exists("/sys/class/kfd/kfd"):
        return _HWInfo(
            kind="amd_rocm",
            devices=[{"index": 0, "name": "AMD GPU (ROCm)", "vram_mib": None}],
            system_ram_mib=ram_mib,
        )

    # ── 3. Apple Silicon ─────────────────────────────────────────────────
    if system == "Darwin" and machine in ("arm64", "arm"):
        devices = []
        # Get GPU core count and unified memory from system_profiler
        try:
            r = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            import json as _json

            sp_data = _json.loads(r.stdout)
            for entry in sp_data.get("SPDisplaysDataType", []):
                name = entry.get("sppci_model", "Apple GPU")
                vram_str = entry.get("spdisplays_vram", "")
                vram_mib = None
                if vram_str:
                    import re as _re

                    m = _re.search(r"(\d+)\s*(GB|MB|MiB|GiB)", vram_str, _re.IGNORECASE)
                    if m:
                        v = int(m.group(1))
                        if "G" in m.group(2).upper():
                            v *= 1024
                        vram_mib = v
                devices.append({"index": 0, "name": name, "vram_mib": vram_mib or ram_mib})
        except Exception:
            # Fallback: just report the unified memory
            devices = [{"index": 0, "name": "Apple Silicon (unified memory)", "vram_mib": ram_mib}]
        return _HWInfo(kind="apple_silicon", devices=devices, system_ram_mib=ram_mib)

    # ── 4. CPU / no GPU ──────────────────────────────────────────────────
    return _HWInfo(kind="cpu", devices=[], system_ram_mib=ram_mib)


def _system_ram_mib() -> int | None:
    """Read total system RAM in MiB, platform-agnostic."""
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Darwin":
            r = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                return int(r.stdout.split(":")[1].strip()) // (1024 * 1024)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def _check_hardware(cfg) -> list[Check]:
    checks = []
    hw = _detect_hardware()

    # ── Report detected hardware ──────────────────────────────────────────
    kind_labels = {
        "nvidia": "NVIDIA GPU",
        "amd_rocm": "AMD GPU (ROCm)",
        "apple_silicon": "Apple Silicon (MPS / Metal)",
        "cpu": "CPU only",
    }
    kind_label = kind_labels.get(hw.kind, hw.kind)

    if hw.kind == "cpu":
        if cfg and cfg.backend == "vllm":
            checks.append(
                Check(
                    "Hardware",
                    "GPU detection",
                    "fail",
                    "No GPU detected — vLLM requires CUDA or ROCm",
                    fix="Switch to the Ollama backend for CPU / Apple Silicon:\n"
                    "    aua init --tier macbook\n"
                    "    aua serve --tier macbook",
                )
            )
        else:
            checks.append(
                Check(
                    "Hardware",
                    "GPU detection",
                    "warn",
                    "No GPU detected — Ollama will use CPU (slow for large models)",
                )
            )
    elif hw.kind == "apple_silicon":
        for dev in hw.devices:
            checks.append(
                Check(
                    "Hardware",
                    "Apple Silicon GPU",
                    "pass",
                    (
                        f"{dev['name']}  ·  {dev['vram_mib']} MiB unified memory"
                        if dev.get("vram_mib")
                        else dev["name"]
                    ),
                )
            )
        if cfg and cfg.backend == "vllm":
            checks.append(
                Check(
                    "Hardware",
                    "Backend compatibility",
                    "fail",
                    "vLLM does not support Apple Silicon",
                    fix="Use the Ollama backend instead:\n"
                    "    aua init --tier macbook\n"
                    "    aua serve --tier macbook",
                )
            )
        else:
            # Check MPS is available via torch if installed
            try:
                import torch

                mps_ok = torch.backends.mps.is_available()
                checks.append(
                    Check(
                        "Hardware",
                        "MPS (Metal Performance Shaders)",
                        "pass" if mps_ok else "warn",
                        "available" if mps_ok else "not available — Ollama will use CPU",
                    )
                )
            except ImportError:
                checks.append(
                    Check(
                        "Hardware",
                        "MPS (Metal Performance Shaders)",
                        "info",
                        "torch not installed — Ollama manages Metal acceleration directly",
                    )
                )
    else:
        # NVIDIA or AMD ROCm
        for dev in hw.devices:
            checks.append(
                Check(
                    "Hardware",
                    f"GPU {dev['index']} ({kind_label})",
                    "pass",
                    (
                        f"{dev['name']}  ·  {dev.get('vram_mib', '?')} MiB VRAM"
                        if dev.get("vram_mib")
                        else dev["name"]
                    ),
                )
            )

        # VRAM projection for vLLM
        if cfg and cfg.backend == "vllm" and hw.devices:
            vram_by_gpu = {dev["index"]: dev.get("vram_mib") for dev in hw.devices}
            util_by_gpu: dict = {}
            for s in list(cfg.specialists) + [cfg.arbiter]:
                g = getattr(s, "gpu", 0)
                u = getattr(s, "gpu_memory_utilization", 0.18)
                util_by_gpu[g] = util_by_gpu.get(g, 0.0) + u

            for gpu_idx, total_util in util_by_gpu.items():
                pct = total_util * 100
                avail_mib = vram_by_gpu.get(gpu_idx)
                proj_mib = int(avail_mib * total_util) if avail_mib else None
                detail = (
                    f"{pct:.0f}% projected ({proj_mib} / {avail_mib} MiB)"
                    if proj_mib
                    else f"{pct:.0f}% projected"
                )
                if total_util > 0.95:
                    checks.append(
                        Check(
                            "Hardware",
                            f"VRAM gpu{gpu_idx}",
                            "fail",
                            detail,
                            fix="Reduce gpu_memory_utilization in aua_config.yaml, or split across GPUs",
                        )
                    )
                elif total_util > 0.90:
                    checks.append(
                        Check(
                            "Hardware", f"VRAM gpu{gpu_idx}", "warn", detail + " — tight, may work"
                        )
                    )
                else:
                    checks.append(Check("Hardware", f"VRAM gpu{gpu_idx}", "pass", detail))

        # AMD ROCm: warn if backend is vllm — needs ROCm-specific vllm build
        if hw.kind == "amd_rocm" and cfg and cfg.backend == "vllm":
            checks.append(
                Check(
                    "Hardware",
                    "ROCm vLLM",
                    "warn",
                    "AMD GPU detected — ensure you have the ROCm vLLM build",
                    fix="pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm5.6",
                )
            )

    # ── RAM check for Ollama / CPU backends ───────────────────────────────
    if cfg and cfg.backend == "ollama" and hw.system_ram_mib:
        # Estimate RAM needed: ~8 GiB per 7B model (4-bit), ~2 GiB per 3B
        needed_mib = 0
        for s in cfg.specialists:
            if "3b" in s.model.lower() or "3B" in s.model:
                needed_mib += 2048
            elif "7b" in s.model.lower() or "7B" in s.model:
                needed_mib += 8192
            else:
                needed_mib += 6144  # conservative default
        # Add arbiter
        arb = cfg.arbiter
        if "3b" in arb.model.lower():
            needed_mib += 2048
        else:
            needed_mib += 6144

        avail_mib = hw.system_ram_mib
        if needed_mib > avail_mib:
            checks.append(
                Check(
                    "Hardware",
                    "RAM (Ollama)",
                    "warn",
                    f"~{needed_mib//1024} GiB needed, {avail_mib//1024} GiB available "
                    f"— models may be slow or fail to load",
                    fix="Reduce model size (use :3b variants) or add more RAM",
                )
            )
        else:
            checks.append(
                Check(
                    "Hardware",
                    "RAM (Ollama)",
                    "pass",
                    f"~{needed_mib//1024} GiB needed, {avail_mib//1024} GiB available",
                )
            )

    # ── Port availability ─────────────────────────────────────────────────
    if cfg:
        _add_port_checks(cfg, checks)

    return checks


def _add_port_checks(cfg, checks: list[Check]):
    """Check that all configured ports are free (not already bound)."""
    ports_to_check = (
        [(s.name, s.port) for s in cfg.specialists]
        + [("arbiter", cfg.arbiter.port)]
        + [("router", cfg.router.port)]
    )
    for name, port in ports_to_check:
        in_use, pid = _port_in_use(port)
        if in_use:
            checks.append(
                Check(
                    "Hardware",
                    f"Port {port} ({name})",
                    "warn",
                    f"already in use (pid {pid})" if pid else "already in use",
                    fix=f"Either a server is already running on port {port}, "
                    f"or another process is using it.\n"
                    f"    kill {pid}  (if you want to restart)  "
                    f"or change the port in aua_config.yaml",
                )
            )
        else:
            checks.append(Check("Hardware", f"Port {port} ({name})", "pass", "free"))


def _check_models(cfg) -> list[Check]:
    checks = []
    all_servers = list(cfg.specialists) + [cfg.arbiter]

    for s in all_servers:
        name = getattr(s, "name", "arbiter")
        model = s.model

        if cfg.backend == "ollama":
            # Check if model is pulled
            try:
                result = subprocess.run(
                    ["ollama", "list"], capture_output=True, text=True, timeout=5
                )
                if model in result.stdout:
                    checks.append(Check("Models", f"{name} ({model})", "pass", "pulled"))
                else:
                    checks.append(
                        Check(
                            "Models",
                            f"{name} ({model})",
                            "warn",
                            "not yet pulled — will be pulled on aua serve",
                            fix=f"ollama pull {model}",
                        )
                    )
            except FileNotFoundError:
                checks.append(
                    Check(
                        "Models",
                        f"{name} ({model})",
                        "skip",
                        "ollama not installed — skipping model check",
                    )
                )
            except Exception as e:
                checks.append(Check("Models", f"{name} ({model})", "warn", f"could not check: {e}"))
        else:
            # vLLM: check if it's a local path or a HuggingFace ID
            model_path = Path(model)
            if model.startswith("/") or model.startswith("./") or model.startswith("models/"):
                # Local path — must exist
                if model_path.exists():
                    size = _dir_size_str(model_path)
                    checks.append(Check("Models", f"{name}", "pass", f"{model}  ({size})"))
                else:
                    checks.append(
                        Check(
                            "Models",
                            f"{name}",
                            "fail",
                            f"{model} — path does not exist",
                            fix=f"Download the model or fix the path in aua_config.yaml.\n"
                            f"    vllm download {model}  (or use a HuggingFace ID)",
                        )
                    )
            else:
                # HuggingFace ID — will be downloaded on first run
                checks.append(
                    Check(
                        "Models",
                        f"{name} ({model})",
                        "warn",
                        "HuggingFace ID — will download on first aua serve run (~5–15 min)",
                        fix=f"Pre-download with:\n" f"    huggingface-cli download {model}",
                    )
                )

    return checks


def _check_specialists_live(cfg) -> list[Check]:
    """Ping each specialist. Warn (not fail) if not running — they may not be started yet."""
    checks = []
    all_servers = [(s.name, s.models_url) for s in cfg.specialists] + [
        ("arbiter", cfg.arbiter.models_url)
    ]
    for name, url in all_servers:
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(url)
                if r.status_code == 200:
                    checks.append(Check("Specialists", f"{name}", "pass", f"reachable at {url}"))
                else:
                    checks.append(
                        Check(
                            "Specialists",
                            f"{name}",
                            "warn",
                            f"HTTP {r.status_code} at {url}",
                            fix="Restart the specialist server",
                        )
                    )
        except Exception:
            checks.append(
                Check(
                    "Specialists",
                    f"{name}",
                    "warn",
                    f"not reachable at {url}",
                    fix="Start with:  aua serve\n"
                    "    (this is expected if you haven't started the framework yet)",
                )
            )
    return checks


# ── Output ────────────────────────────────────────────────────────────────────


def _print_results(checks: list[Check], config_path: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    n_pass = sum(1 for c in checks if c.status == "pass")
    n_fail = sum(1 for c in checks if c.status == "fail")
    n_warn = sum(1 for c in checks if c.status == "warn")

    console.print()
    console.print(f"[bold]aua doctor[/bold]  [dim]{config_path} · {ts}[/dim]")
    console.print("─" * 64)

    current_group = None
    for check in checks:
        if check.group != current_group:
            current_group = check.group
            console.print(f"\n[bold]{check.group}[/bold]")

        sym, style = check.symbol
        name_w = 32
        name = check.name[:name_w].ljust(name_w)
        detail = Text(check.detail, style="dim") if check.detail else Text("")

        console.print(f"  [{ style}]{sym}[/{style}]  {name}  ", detail, end="\n")
        if check.fix and check.status in ("fail", "warn"):
            for line in check.fix.splitlines():
                console.print(f"[dim]       {line}[/dim]")

    # Summary
    console.print()
    console.print("─" * 64)
    summary_parts = []
    if n_pass:
        summary_parts.append(f"[green]{n_pass} passed[/green]")
    if n_warn:
        summary_parts.append(f"[yellow]{n_warn} warning(s)[/yellow]")
    if n_fail:
        summary_parts.append(f"[red]{n_fail} failed[/red]")
    console.print("  " + " · ".join(summary_parts))
    console.print()

    if n_fail == 0 and n_warn == 0:
        console.print("  [bold green]✓ All checks passed — ready to run aua serve[/bold green]")
    elif n_fail == 0:
        console.print(
            "  [yellow]⚠ Warnings above are non-blocking — " "aua serve may still work[/yellow]"
        )
    else:
        console.print(f"  [red]✗ Fix {n_fail} issue(s) before running aua serve[/red]")
    console.print()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _port_in_use(port: int) -> tuple[bool, int | None]:
    """Return (in_use, pid_or_None)."""
    # Try to bind — if it fails, port is in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            return False, None
        except OSError:
            pass

    # Try to find the PID via lsof (best-effort, Unix only)
    pid = None
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True, timeout=3
        )
        pids = result.stdout.strip().splitlines()
        pid = int(pids[0]) if pids else None
    except Exception:
        pass
    return True, pid


def _dir_size_str(path: Path) -> str:
    """Human-readable size of a directory."""
    try:
        total: float = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        for unit in ["B", "KB", "MB", "GB"]:
            if total < 1024:
                return f"{total:.0f} {unit}"
            total /= 1024
        return f"{total:.1f} TB"
    except Exception:
        return "?"
