"""
aua/config.py — Single source of truth for all AUA configuration.

Two responsibilities:
  1. YAML loader  — reads aua_config.yaml and returns a validated AUAConfig
  2. Field configs — FieldConfig dataclass + FIELD_CONFIGS dict (unchanged
                     from the POC; these are the whitepaper field weights)

Usage:
    from aua.config import load_config
    config = load_config("aua_config.yaml")

    # Access specialist endpoints
    for s in config.specialists:
        print(s.name, s.endpoint)   # http://127.0.0.1:<port>/v1/chat/completions

    # Access field weights
    from aua.config import FIELD_CONFIGS
    cfg = FIELD_CONFIGS["software_engineering"]
    print(cfg.penalty_multiplier)   # 2.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ── FieldConfig ───────────────────────────────────────────────────────────────
# Unchanged from the POC — these are the whitepaper field weights and bounds.


@dataclass
class FieldConfig:
    name: str
    w_efficacy: float  # weight on efficacy term (E)
    w_confidence: float  # weight on confidence term (C)
    w_curiosity: float  # weight on curiosity term (K)
    c_min: float  # minimum confidence to act (below → abstain)
    e_min: float  # minimum efficacy to act
    penalty_multiplier: float  # contradiction penalty scale (surgery=10×, SWE=2×)

    def __post_init__(self) -> None:
        total = self.w_efficacy + self.w_confidence + self.w_curiosity
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0 for field '{self.name}' (got {total:.4f})")


FIELD_CONFIGS: dict[str, FieldConfig] = {
    "surgery": FieldConfig(
        name="surgery",
        w_efficacy=0.20,
        w_confidence=0.70,
        w_curiosity=0.10,
        c_min=0.95,
        e_min=0.90,
        penalty_multiplier=10.0,
    ),
    "aviation": FieldConfig(
        name="aviation",
        w_efficacy=0.20,
        w_confidence=0.70,
        w_curiosity=0.10,
        c_min=0.95,
        e_min=0.90,
        penalty_multiplier=10.0,
    ),
    "law": FieldConfig(
        name="law",
        w_efficacy=0.30,
        w_confidence=0.60,
        w_curiosity=0.10,
        c_min=0.85,
        e_min=0.80,
        penalty_multiplier=5.0,
    ),
    "structural_engineering": FieldConfig(
        name="structural_engineering",
        w_efficacy=0.40,
        w_confidence=0.50,
        w_curiosity=0.10,
        c_min=0.80,
        e_min=0.75,
        penalty_multiplier=4.0,
    ),
    "software_engineering": FieldConfig(
        name="software_engineering",
        w_efficacy=0.55,
        w_confidence=0.35,
        w_curiosity=0.10,
        c_min=0.70,
        e_min=0.65,
        penalty_multiplier=2.0,
    ),
    "mathematics": FieldConfig(
        name="mathematics",
        w_efficacy=0.50,
        w_confidence=0.40,
        w_curiosity=0.10,
        c_min=0.75,
        e_min=0.70,
        penalty_multiplier=3.0,
    ),
    "stem_research": FieldConfig(
        name="stem_research",
        w_efficacy=0.50,
        w_confidence=0.30,
        w_curiosity=0.20,
        c_min=0.65,
        e_min=0.60,
        penalty_multiplier=2.0,
    ),
    "education": FieldConfig(
        name="education",
        w_efficacy=0.50,
        w_confidence=0.30,
        w_curiosity=0.20,
        c_min=0.60,
        e_min=0.55,
        penalty_multiplier=1.5,
    ),
    "art": FieldConfig(
        name="art",
        w_efficacy=0.80,
        w_confidence=0.10,
        w_curiosity=0.10,
        c_min=0.10,
        e_min=0.20,
        penalty_multiplier=1.0,
    ),
    "creative_writing": FieldConfig(
        name="creative_writing",
        w_efficacy=0.80,
        w_confidence=0.05,
        w_curiosity=0.15,
        c_min=0.05,
        e_min=0.15,
        penalty_multiplier=1.0,
    ),
    "general": FieldConfig(
        name="general",
        w_efficacy=0.50,
        w_confidence=0.35,
        w_curiosity=0.15,
        c_min=0.50,
        e_min=0.50,
        penalty_multiplier=1.5,
    ),
}


def get_effective_config(field_distribution: dict[str, float]) -> FieldConfig:
    """
    Blend FieldConfigs by probability weight when domain is ambiguous.
    This makes the agent more conservative under ambiguity — the blended
    C_min is a weighted average pulled toward the most conservative field.

    Args:
        field_distribution: {field_name: probability}, must sum to 1.0

    Returns:
        A blended FieldConfig instance (name="blended").
    """
    total = sum(field_distribution.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"field_distribution must sum to 1.0 (got {total:.4f})")

    blended = FieldConfig.__new__(FieldConfig)
    blended.name = "blended"
    blended.w_efficacy = blended.w_confidence = blended.w_curiosity = 0.0
    blended.c_min = blended.e_min = blended.penalty_multiplier = 0.0

    for fname, prob in field_distribution.items():
        cfg = FIELD_CONFIGS.get(fname, FIELD_CONFIGS["general"])
        blended.w_efficacy += prob * cfg.w_efficacy
        blended.w_confidence += prob * cfg.w_confidence
        blended.w_curiosity += prob * cfg.w_curiosity
        blended.c_min += prob * cfg.c_min
        blended.e_min += prob * cfg.e_min
        blended.penalty_multiplier += prob * cfg.penalty_multiplier

    return blended


# ── Deployment config dataclasses ─────────────────────────────────────────────
# These are populated from aua_config.yaml.

# ── Allowed keys per config section (used by unknown-key validator) ───────────
_KNOWN_TOP_LEVEL: set[str] = {"aua", "specialists", "arbiter", "router", "blue_green", "logging"}
_KNOWN_AUA_KEYS: set[str] = {"version", "mode", "backend", "project_name"}
_KNOWN_SPECIALIST_KEYS: set[str] = {
    "name",
    "model",
    "port",
    "field",
    "backend",
    "gpu",
    "gpu_memory_utilization",
    "max_model_len",
    "quantization",
    "enforce_eager",
    "host",
    "scheme",
    "endpoint_override",
    "models_url_override",
}
_KNOWN_ARBITER_KEYS: set[str] = {
    "model",
    "port",
    "backend",
    "gpu",
    "gpu_memory_utilization",
    "max_model_len",
    "quantization",
    "enforce_eager",
    "host",
    "scheme",
    "endpoint_override",
    "models_url_override",
}
_KNOWN_ROUTER_KEYS: set[str] = {
    "port",
    "host",
    "single_domain_threshold",
    "fanout_threshold",
    "specialist_timeout",
    "cors_origins",
}
_KNOWN_BG_KEYS: set[str] = {"delta", "T_min", "tau"}
_KNOWN_LOG_KEYS: set[str] = {"level", "format"}


@dataclass
class SpecialistConfig:
    """One specialist server — vLLM or Ollama."""

    name: str  # e.g. "swe"
    model: str  # HF repo, local path, or Ollama tag
    port: int  # server port
    field: str  # maps to a key in FIELD_CONFIGS
    backend: str = "vllm"  # "vllm" | "ollama" (inherits from AUAConfig)
    gpu: int = 0  # CUDA device index (vLLM only)
    gpu_memory_utilization: float = 0.34  # vLLM only — must be in (0, 1]
    max_model_len: int = 2048  # vLLM only
    quantization: str | None = "awq"  # vLLM only: "awq" | "gptq" | None
    enforce_eager: bool = True  # vLLM only: prevents CUDA graph conflicts

    # P-05: host/scheme fields replace hardcoded localhost
    host: str = "127.0.0.1"  # bind/connect host for this specialist
    scheme: str = "http"  # "http" | "https"
    endpoint_override: str | None = None  # full URL override (ignores host/scheme/port)
    models_url_override: str | None = None  # full models URL override

    @property
    def endpoint(self) -> str:
        """Full chat completions URL for this specialist."""
        if self.endpoint_override:
            return self.endpoint_override
        path = "/v1/chat/completions"  # Ollama supports OpenAI-compat; /api/chat only for native streaming
        return f"{self.scheme}://{self.host}:{self.port}{path}"

    @property
    def models_url(self) -> str:
        """Full models/tags health-check URL."""
        if self.models_url_override:
            return self.models_url_override
        if self.backend == "ollama":
            return f"{self.scheme}://{self.host}:{self.port}/api/tags"
        return f"{self.scheme}://{self.host}:{self.port}/v1/models"

    @property
    def serve_model_name(self) -> str:
        """Model name to use in the request body.
        vLLM: the --served-model-name (same as self.name).
        Ollama: the model tag (e.g. qwen2.5-coder:7b).
        """
        if self.backend == "ollama":
            return self.model  # Ollama uses the tag directly
        return self.name  # vLLM uses --served-model-name

    @property
    def field_config(self) -> FieldConfig:
        return FIELD_CONFIGS.get(self.field, FIELD_CONFIGS["general"])

    def vllm_command(self) -> list[str]:
        """Return the vLLM startup command as an argv list."""
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--served-model-name",
            self.name,
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        if self.enforce_eager:
            cmd += ["--enforce-eager"]
        return cmd


@dataclass
class ArbiterConfig:
    """The arbiter (general/small) specialist."""

    model: str
    port: int
    backend: str = "vllm"  # inherits from AUAConfig
    gpu: int = 0
    gpu_memory_utilization: float = 0.18  # must be in (0, 1]
    max_model_len: int = 2048
    quantization: str | None = "awq"
    enforce_eager: bool = True

    # P-05: host/scheme fields replace hardcoded localhost
    host: str = "127.0.0.1"
    scheme: str = "http"
    endpoint_override: str | None = None
    models_url_override: str | None = None

    @property
    def endpoint(self) -> str:
        if self.endpoint_override:
            return self.endpoint_override
        path = "/v1/chat/completions"  # Ollama supports OpenAI-compat; /api/chat only for native streaming
        return f"{self.scheme}://{self.host}:{self.port}{path}"

    @property
    def models_url(self) -> str:
        if self.models_url_override:
            return self.models_url_override
        if self.backend == "ollama":
            return f"{self.scheme}://{self.host}:{self.port}/api/tags"
        return f"{self.scheme}://{self.host}:{self.port}/v1/models"

    @property
    def serve_model_name(self) -> str:
        if self.backend == "ollama":
            return self.model
        return "arbiter"

    def vllm_command(self) -> list[str]:
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--served-model-name",
            "arbiter",
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        if self.enforce_eager:
            cmd += ["--enforce-eager"]
        return cmd


@dataclass
class RouterConfig:
    """FastAPI router settings."""

    port: int = 8000
    single_domain_threshold: float = 0.75  # above → single specialist; must be in [0, 1]
    fanout_threshold: float = 0.30  # both above → fan out; must be in [0, 1]
    specialist_timeout: float = 60.0  # seconds per specialist call
    host: str = "0.0.0.0"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class RuntimeConfig:
    """
    Paths for runtime artifacts — all nested under a single base directory.

    Default layout:
        .aua/
          logs/         — rotating log files per service
          pids/         — PID files written by aua serve
          state/        — promotion log (promotions.jsonl) + rollback state
          checkpoints/  — model checkpoint symlinks for blue-green
    """

    base: Path = field(default_factory=lambda: Path(".aua"))

    @property
    def logs(self) -> Path:
        return self.base / "logs"

    @property
    def pids(self) -> Path:
        return self.base / "pids"

    @property
    def state(self) -> Path:
        return self.base / "state"

    @property
    def checkpoints(self) -> Path:
        return self.base / "checkpoints"

    def ensure(self) -> None:
        """Create all runtime directories (idempotent)."""
        for p in (self.logs, self.pids, self.state, self.checkpoints):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class BlueGreenFieldConfig:
    """Per-field blue-green promotion thresholds."""

    delta: float = 0.025  # minimum U improvement to promote
    T_min: int = 10  # minimum interactions in canary before shift decision
    tau: float = 0.20  # softmax temperature for traffic routing
    # tau(surgery)=0.05 (very conservative), tau(creative)=0.50 (aggressive)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class AUAConfig:
    """
    Top-level configuration loaded from aua_config.yaml.
    All deployment values live here — no hardcoded values anywhere else.
    """

    version: str
    mode: str  # "local" | "kubernetes" | "cluster"
    specialists: list[SpecialistConfig]
    arbiter: ArbiterConfig
    router: RouterConfig
    blue_green: dict[str, BlueGreenFieldConfig]  # keyed by specialist name
    backend: str = "vllm"  # "vllm" | "ollama"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Derived — built on load
    _specialist_by_name: dict[str, SpecialistConfig] = field(
        default_factory=dict, init=False, repr=False
    )
    _specialist_by_field: dict[str, SpecialistConfig] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        for s in self.specialists:
            self._specialist_by_name[s.name] = s
            self._specialist_by_field[s.field] = s

    def specialist(self, name: str) -> SpecialistConfig:
        """Look up a specialist by name, e.g. config.specialist("swe")."""
        if name not in self._specialist_by_name:
            raise KeyError(
                f"No specialist named '{name}'. " f"Available: {list(self._specialist_by_name)}"
            )
        return self._specialist_by_name[name]

    def specialist_for_field(self, field_name: str) -> SpecialistConfig | None:
        """Return the specialist that covers a given field, or None."""
        return self._specialist_by_field.get(field_name)

    def all_endpoints(self) -> dict[str, str]:
        """Return all endpoint URLs: {name: url} including the arbiter."""
        eps = {s.name: s.endpoint for s in self.specialists}
        eps["arbiter"] = self.arbiter.endpoint
        return eps

    def blue_green_for(self, specialist_name: str) -> BlueGreenFieldConfig:
        """Return blue-green thresholds for a specialist (falls back to defaults)."""
        return self.blue_green.get(specialist_name, BlueGreenFieldConfig())


# ── YAML loader ───────────────────────────────────────────────────────────────


def load_config(path: str | os.PathLike = "aua_config.yaml") -> AUAConfig:
    """
    Load and validate aua_config.yaml.

    Args:
        path: path to the YAML file (default: aua_config.yaml in cwd)

    Returns:
        Validated AUAConfig instance.

    Raises:
        FileNotFoundError: if the config file doesn't exist.
        ValueError: if required fields are missing, unknown keys are present,
                    ports are duplicated, or values are out of range.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {p.resolve()}\n"
            f"Run 'aua init' to scaffold a starter config, or copy the example:\n"
            f"  cp aua_config.yaml.example aua_config.yaml"
        )

    with p.open() as f:
        raw = yaml.safe_load(f)

    return _parse_config(raw, source=str(p))


def _parse_config(raw: dict, source: str = "<unknown>") -> AUAConfig:
    """Parse a raw YAML dict into a validated AUAConfig."""

    # ── Unknown top-level keys ─────────────────────────────────────────────
    _reject_unknown_keys(raw, _KNOWN_TOP_LEVEL, "top-level", source)

    # ── Top-level aua block ────────────────────────────────────────────────
    aua_block = raw.get("aua", {})
    _reject_unknown_keys(aua_block, _KNOWN_AUA_KEYS, "aua", source)
    version = str(aua_block.get("version", "0.5"))
    mode = str(aua_block.get("mode", "local"))
    backend = str(aua_block.get("backend", "vllm"))

    # ── Specialists ────────────────────────────────────────────────────────
    raw_specialists = raw.get("specialists", [])
    if not raw_specialists:
        raise ValueError(f"[{source}] 'specialists' list is required and must not be empty.")

    specialists: list[SpecialistConfig] = []
    for i, s in enumerate(raw_specialists):
        _reject_unknown_keys(s, _KNOWN_SPECIALIST_KEYS, f"specialists[{i}]", source)
        _require(s, ["name", "model", "port", "field"], context=f"specialists[{i}]", source=source)
        spec_backend = str(s.get("backend", backend))
        gpu_util = float(s.get("gpu_memory_utilization", 0.34))
        _validate_range(
            gpu_util, "gpu_memory_utilization", 0.0, 1.0, exclusive_lo=True, source=source
        )
        specialists.append(
            SpecialistConfig(
                name=s["name"],
                model=s["model"],
                port=int(s["port"]),
                field=s["field"],
                backend=spec_backend,
                gpu=int(s.get("gpu", 0)),
                gpu_memory_utilization=gpu_util,
                max_model_len=int(s.get("max_model_len", 2048)),
                quantization=s.get("quantization", "awq") or None,
                enforce_eager=bool(s.get("enforce_eager", True)),
                host=str(s.get("host", "127.0.0.1")),
                scheme=str(s.get("scheme", "http")),
                endpoint_override=s.get("endpoint_override") or None,
                models_url_override=s.get("models_url_override") or None,
            )
        )

    # ── Arbiter ────────────────────────────────────────────────────────────
    raw_arb = raw.get("arbiter", {})
    _reject_unknown_keys(raw_arb, _KNOWN_ARBITER_KEYS, "arbiter", source)
    _require(raw_arb, ["model", "port"], context="arbiter", source=source)
    arb_gpu_util = float(raw_arb.get("gpu_memory_utilization", 0.18))
    _validate_range(
        arb_gpu_util, "arbiter.gpu_memory_utilization", 0.0, 1.0, exclusive_lo=True, source=source
    )
    arbiter = ArbiterConfig(
        model=raw_arb["model"],
        port=int(raw_arb["port"]),
        backend=str(raw_arb.get("backend", backend)),
        gpu=int(raw_arb.get("gpu", 0)),
        gpu_memory_utilization=arb_gpu_util,
        max_model_len=int(raw_arb.get("max_model_len", 2048)),
        quantization=raw_arb.get("quantization", "awq") or None,
        enforce_eager=bool(raw_arb.get("enforce_eager", True)),
        host=str(raw_arb.get("host", "127.0.0.1")),
        scheme=str(raw_arb.get("scheme", "http")),
        endpoint_override=raw_arb.get("endpoint_override") or None,
        models_url_override=raw_arb.get("models_url_override") or None,
    )

    # ── Router ─────────────────────────────────────────────────────────────
    raw_router = raw.get("router", {})
    _reject_unknown_keys(raw_router, _KNOWN_ROUTER_KEYS, "router", source)
    sdt = float(raw_router.get("single_domain_threshold", 0.75))
    ft = float(raw_router.get("fanout_threshold", 0.30))
    _validate_range(sdt, "router.single_domain_threshold", 0.0, 1.0, source=source)
    _validate_range(ft, "router.fanout_threshold", 0.0, 1.0, source=source)
    cors = raw_router.get("cors_origins", ["*"])
    if isinstance(cors, str):
        cors = [cors]
    router = RouterConfig(
        port=int(raw_router.get("port", 8000)),
        single_domain_threshold=sdt,
        fanout_threshold=ft,
        specialist_timeout=float(raw_router.get("specialist_timeout", 60.0)),
        host=str(raw_router.get("host", "0.0.0.0")),
        cors_origins=list(cors),
    )

    # ── Blue-green per specialist ──────────────────────────────────────────
    raw_bg = raw.get("blue_green", {})
    blue_green: dict[str, BlueGreenFieldConfig] = {}
    for name, bg in raw_bg.items():
        _reject_unknown_keys(bg, _KNOWN_BG_KEYS, f"blue_green.{name}", source)
        blue_green[name] = BlueGreenFieldConfig(
            delta=float(bg.get("delta", 0.025)),
            T_min=int(bg.get("T_min", 10)),
            tau=float(bg.get("tau", 0.20)),
        )

    # ── Logging ────────────────────────────────────────────────────────────
    raw_log = raw.get("logging", {})
    _reject_unknown_keys(raw_log, _KNOWN_LOG_KEYS, "logging", source)
    logging_cfg = LoggingConfig(
        level=str(raw_log.get("level", "INFO")).upper(),
        format=str(raw_log.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")),
    )

    # ── Validate field names against FIELD_CONFIGS ─────────────────────────
    for s in specialists:
        if s.field not in FIELD_CONFIGS:
            raise ValueError(
                f"[{source}] Specialist '{s.name}' references unknown field "
                f"'{s.field}'. Valid fields: {sorted(FIELD_CONFIGS)}"
            )

    # ── Duplicate port check ───────────────────────────────────────────────
    _validate_no_duplicate_ports(specialists, arbiter, router, source)

    return AUAConfig(
        version=version,
        mode=mode,
        specialists=specialists,
        backend=backend,
        arbiter=arbiter,
        router=router,
        blue_green=blue_green,
        logging=logging_cfg,
    )


# ── Validators ────────────────────────────────────────────────────────────────


def _require(d: dict, keys: list[str], context: str, source: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"[{source}] '{context}' is missing required field(s): {missing}")


def _reject_unknown_keys(d: dict, known: set[str], context: str, source: str) -> None:
    """Raise ValueError if d contains any key not in known."""
    unknown = set(d) - known
    if unknown:
        raise ValueError(
            f"[{source}] Unknown key(s) in '{context}': {sorted(unknown)}. "
            f"Known keys: {sorted(known)}"
        )


def _validate_range(
    value: float,
    name: str,
    lo: float,
    hi: float,
    source: str,
    exclusive_lo: bool = False,
    exclusive_hi: bool = False,
) -> None:
    """Raise ValueError if value is outside [lo, hi] (or exclusive variant)."""
    lo_ok = value > lo if exclusive_lo else value >= lo
    hi_ok = value < hi if exclusive_hi else value <= hi
    if not (lo_ok and hi_ok):
        lo_bracket = "(" if exclusive_lo else "["
        hi_bracket = ")" if exclusive_hi else "]"
        raise ValueError(
            f"[{source}] '{name}' must be in {lo_bracket}{lo}, {hi}{hi_bracket}, got {value}"
        )


def _validate_no_duplicate_ports(
    specialists: list[SpecialistConfig],
    arbiter: ArbiterConfig,
    router: RouterConfig,
    source: str,
) -> None:
    """Raise ValueError if any two vLLM services share the same port.

    Ollama specialists intentionally share port 11434 (one Ollama process
    serves all models), so Ollama-backend entries are excluded from this check.
    The router port is always checked against every other service.
    """
    seen: dict[int, str] = {}

    # Only check vLLM specialists — Ollama ones share a single process/port
    vllm_specialists = [s for s in specialists if s.backend != "ollama"]
    all_services = [(s.name, s.port) for s in vllm_specialists]

    if arbiter.backend != "ollama":
        all_services.append(("arbiter", arbiter.port))

    # Router port is always unique regardless of backend
    all_services.append(("router", router.port))

    for name, port in all_services:
        if port in seen:
            raise ValueError(
                f"[{source}] Duplicate port {port} used by both "
                f"'{seen[port]}' and '{name}'. Each service must use a unique port."
            )
        seen[port] = name


# ── Tier loader ───────────────────────────────────────────────────────────────

# Canonical tier names — use these in new code.
AVAILABLE_TIERS: list[str] = ["macbook", "single-4090", "quad-4090", "a100-cluster"]

# Backward-compatible aliases — resolve to canonical names.
# Deprecated: will be removed in v1.0.
TIER_ALIASES: dict[str, str] = {
    "rtx4090": "single-4090",
    "a100": "a100-cluster",
}


def load_tier(tier: str) -> AUAConfig:
    """
    Load a bundled hardware-tier config template.

    Args:
        tier: canonical tier name ("macbook", "single-4090", "quad-4090",
              "a100-cluster") or backward-compatible alias ("rtx4090", "a100").

    Returns:
        AUAConfig from the bundled template.

    Raises:
        ValueError: if the tier name is not recognised.
    """
    canonical = TIER_ALIASES.get(tier, tier)
    if canonical not in AVAILABLE_TIERS:
        known = sorted(AVAILABLE_TIERS) + [f"{k} (alias)" for k in sorted(TIER_ALIASES)]
        raise ValueError(
            f"Unknown tier '{tier}'. Available: {known}\n"
            f"Use 'aua serve --config my.yaml' for a custom config."
        )
    tier_path = Path(__file__).parent / "tiers" / f"{canonical}.yaml"
    return load_config(tier_path)
