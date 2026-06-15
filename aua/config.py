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
from typing import Any

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
_KNOWN_TOP_LEVEL: set[str] = {
    "aua",
    "specialists",
    "arbiter",
    "router",
    "blue_green",
    "logging",
    "secrets",
    "state",
    "security",
    "plugins",
    "hooks",
    "middleware",
    "experiment_tracking",
}
_KNOWN_AUA_KEYS: set[str] = {"version", "mode", "backend", "project_name"}
_KNOWN_SPECIALIST_KEYS: set[str] = {
    "name",
    "model",
    "port",
    "field",
    "backend",
    "gpu",
    "gpu_ids",
    "gpu_memory_utilization",
    "max_model_len",
    "quantization",
    "enforce_eager",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "host",
    "scheme",
    "endpoint_override",
    "models_url_override",
    "mlflow_tracking_uri",
}
_KNOWN_ARBITER_KEYS: set[str] = {
    "model",
    "port",
    "backend",
    "gpu",
    "gpu_ids",
    "gpu_memory_utilization",
    "max_model_len",
    "quantization",
    "enforce_eager",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "host",
    "scheme",
    "endpoint_override",
    "models_url_override",
    "mlflow_tracking_uri",
}
_KNOWN_ROUTER_KEYS: set[str] = {
    "port",
    "host",
    "single_domain_threshold",
    "fanout_threshold",
    "specialist_timeout",
    "arbitration_mode",
    "tau",
    "retry",
    "circuit_breaker",
    "cors_origins",
}
_KNOWN_BG_KEYS: set[str] = {
    "delta",
    "T_min",
    "tau",
    "regression_dataset",
    "regression_block",
    "shadow_endpoint",
    "shadow_min_queries",
}
_KNOWN_LOG_KEYS: set[str] = {"level", "format"}
_KNOWN_SECRETS_KEYS: set[str] = {"provider", "region", "url", "token_env"}
_KNOWN_STATE_KEYS: set[str] = {"backend", "path"}
_KNOWN_SECURITY_KEYS: set[str] = {
    "cors_origins",
    "mtls",
    "encryption",
    "auth_enabled",
    "token_secret_env",
    "token_expiry_days",
}
_KNOWN_EXPERIMENT_KEYS: set[str] = {"enabled", "mlflow", "wandb"}
_KNOWN_PLUGIN_KINDS: set[str] = {
    "field_classifier",
    "utility_scorer",
    "arbiter_policy",
    "promotion_policy",
    "correction_store",
    "model_backend",
    "state_store",
    # #51: extended plugin types
    "contradiction_detector",
    "assertion_store",
    "routing_strategy",
    "scoring_component",
    # now wired (previously load-only)
    "arbiter_policy",
    "promotion_policy",
    "full_promotion_policy",
}
_KNOWN_PLUGIN_ENTRY_KEYS: set[str] = {"import_path", "config"}
_KNOWN_HOOK_ENTRY_KEYS: set[str] = {"hook_point", "import_path", "config", "fail_closed"}
# Exactly the 11 points the runtime fires — accepting a point that never
# fires would let a hook registration silently do nothing.
_VALID_HOOK_POINTS: set[str] = {
    "pre_query",
    "post_route",
    "pre_specialist_call",
    "post_specialist_call",
    "pre_arbiter",
    "post_arbiter",
    "pre_response",
    "post_response",
    "on_correction",
    "on_promotion",
    "on_rollback",
}


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

    # #66: tensor and pipeline parallelism (vLLM multi-GPU)
    tensor_parallel_size: int = 1  # number of GPUs for tensor parallelism (must be power of 2)
    pipeline_parallel_size: int = 1  # number of pipeline stages across nodes
    gpu_ids: list[int] | None = None  # explicit GPU indices; overrides gpu when set
    # e.g. [0,1,2,3] for 4-GPU tensor parallel

    # P-05: host/scheme fields replace hardcoded localhost
    host: str = "127.0.0.1"  # bind/connect host for this specialist
    scheme: str = "http"  # "http" | "https"
    endpoint_override: str | None = None  # full URL override (ignores host/scheme/port)
    models_url_override: str | None = None  # full models URL override
    mlflow_tracking_uri: str | None = None  # #46: MLflow tracking URI for models:/ URIs

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
        """Return the vLLM startup command as an argv list.

        Tensor parallel (#66): --tensor-parallel-size N passes N GPUs to vLLM.
        vLLM internally handles NCCL communication across those GPUs.
        CUDA_VISIBLE_DEVICES must expose exactly N GPU indices (handled by
        _build_env in serve.py using gpu_ids when set, or gpu otherwise).

        Pipeline parallel (#66): --pipeline-parallel-size M splits the model
        into M pipeline stages across nodes. Requires N×M total GPUs and
        a distributed vLLM setup (ray is launched automatically by vLLM).
        """
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
        if self.tensor_parallel_size > 1:
            cmd += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.pipeline_parallel_size > 1:
            cmd += ["--pipeline-parallel-size", str(self.pipeline_parallel_size)]
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

    # #66: tensor and pipeline parallelism (vLLM multi-GPU)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_ids: list[int] | None = None

    # P-05: host/scheme fields replace hardcoded localhost
    host: str = "127.0.0.1"
    scheme: str = "http"
    endpoint_override: str | None = None
    models_url_override: str | None = None
    mlflow_tracking_uri: str | None = None  # #46

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
        if self.tensor_parallel_size > 1:
            cmd += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.pipeline_parallel_size > 1:
            cmd += ["--pipeline-parallel-size", str(self.pipeline_parallel_size)]
        return cmd


@dataclass
class RetryConfig:
    """
    Transport-level retry configuration for specialist HTTP calls (#39).
    Set max_retries: 0 to disable.
    """

    max_retries: int = 3
    base_delay_ms: float = 200.0
    max_delay_ms: float = 5000.0
    jitter: bool = True
    retryable_status_codes: list[int] = field(default_factory=lambda: [429, 502, 503, 504])

    def delay_for_attempt(self, attempt: int) -> float:
        """
        Compute delay in seconds for attempt N (1-indexed).

        attempt=1: 0.0 (first call, no delay)
        attempt=2: base_delay * 2^0
        attempt=3: base_delay * 2^1  ...capped at max_delay_ms
        """
        import random as _random

        if attempt <= 1:
            return 0.0
        exp = attempt - 2
        delay_ms = min(self.base_delay_ms * (2**exp), self.max_delay_ms)
        if self.jitter:
            delay_ms *= _random.uniform(0.75, 1.25)
        return delay_ms / 1000.0


@dataclass
class CircuitBreakerConfig:
    """
    Per-specialist circuit breaker configuration (#37, #38).
    Set enabled: false to disable entirely.
    """

    enabled: bool = True
    failure_threshold: int = 5
    failure_window_s: float = 60.0
    recovery_timeout_s: float = 30.0
    success_threshold: int = 2


@dataclass
class RouterConfig:
    """FastAPI router settings."""

    port: int = 8000
    single_domain_threshold: float = 0.75  # above → single specialist; must be in [0, 1]
    fanout_threshold: float = 0.30  # both above → fan out; must be in [0, 1]
    specialist_timeout: float = 60.0  # seconds per specialist call
    host: str = "0.0.0.0"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    arbitration_mode: str = "pairwise"  # "pairwise" | "vcg" | "llm"
    tau: float = 1.0  # softmax temperature for routing; 1.0=off, <1=sharper, >1=softer
    retry: RetryConfig = field(default_factory=RetryConfig)  # #39
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)  # #37


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
    # #49: regression gate on promotion
    regression_dataset: str | None = None  # path to eval YAML; None = skip regression check
    regression_block: bool = True  # True = block promotion on regression; False = warn only
    # #48: shadow mode — GREEN receives traffic silently until threshold
    shadow_endpoint: str | None = None  # HTTP endpoint of running GREEN specialist
    shadow_min_queries: int = 50  # minimum shadow queries before promotion is considered


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class StateConfig:
    """State store selection. Documented in tutorial Part 2."""

    backend: str = "sqlite"  # "sqlite" | "files"
    path: str = ".aua/state/aua.db"


@dataclass
class SecurityConfig:
    """
    Security block. cors_origins here overrides router.cors_origins so the
    tutorial's `security:` examples work; mtls/encryption are validated and
    surfaced to the certs/encryption modules.
    """

    cors_origins: list[str] | None = None
    mtls: dict[str, Any] = field(default_factory=dict)
    encryption: dict[str, Any] = field(default_factory=dict)
    # #auth: bearer token auth
    auth_enabled: bool = False
    token_secret_env: str = "AUA_TOKEN_SECRET"
    token_expiry_days: int = 30


@dataclass
class PluginSpec:
    """One `plugins:` entry — F-09 extension import system."""

    import_path: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookSpec:
    """One `hooks:` entry — F-10 lifecycle hook registration."""

    hook_point: str
    import_path: str
    config: dict[str, Any] = field(default_factory=dict)
    fail_closed: bool = False


@dataclass
class MiddlewareSpec:
    """One `middleware:` entry — F-11 ordered pipeline."""

    import_path: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecretsConfig:
    """
    #19: secrets provider selection. Config references secret NAMES,
    never values; the provider resolves them at startup.

        secrets:
          provider: vault            # "env" (default) | "vault" | "aws" | "gcp"
          url: http://127.0.0.1:8200 # vault only
          token_env: VAULT_TOKEN     # vault only — env var holding the token
          region: us-east-1          # aws only
    """

    provider: str = "env"
    region: str = "us-east-1"
    url: str = "http://127.0.0.1:8200"
    token_env: str = "VAULT_TOKEN"


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
    secrets: SecretsConfig = field(default_factory=SecretsConfig)
    state: StateConfig = field(default_factory=StateConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    plugins: dict[str, PluginSpec] = field(default_factory=dict)
    hooks: list[HookSpec] = field(default_factory=list)
    middleware: list[MiddlewareSpec] = field(default_factory=list)
    experiment_tracking: Any = field(
        default_factory=lambda: __import__(
            "aua.experiment_tracker", fromlist=["ExperimentConfig"]
        ).ExperimentConfig()
    )  # ExperimentConfig — typed as Any to avoid circular import

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


def _load_retry_config(raw: dict) -> RetryConfig:
    return RetryConfig(
        max_retries=int(raw.get("max_retries", 3)),
        base_delay_ms=float(raw.get("base_delay_ms", 200.0)),
        max_delay_ms=float(raw.get("max_delay_ms", 5000.0)),
        jitter=bool(raw.get("jitter", True)),
        retryable_status_codes=list(raw.get("retryable_status_codes", [429, 502, 503, 504])),
    )


def _load_cb_config(raw: dict) -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        enabled=bool(raw.get("enabled", True)),
        failure_threshold=int(raw.get("failure_threshold", 5)),
        failure_window_s=float(raw.get("failure_window_s", 60.0)),
        recovery_timeout_s=float(raw.get("recovery_timeout_s", 30.0)),
        success_threshold=int(raw.get("success_threshold", 2)),
    )


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
        tp = int(s.get("tensor_parallel_size", 1))
        pp = int(s.get("pipeline_parallel_size", 1))
        if tp > 1 and (tp & (tp - 1)) != 0:
            raise ValueError(
                f"[{source}] specialists[{i}].tensor_parallel_size={tp} "
                "must be a power of 2 (NVLink/PCIe requirement)."
            )
        raw_gpu_ids = s.get("gpu_ids")
        gpu_ids = [int(g) for g in raw_gpu_ids] if raw_gpu_ids else None
        if gpu_ids and len(gpu_ids) != tp:
            raise ValueError(
                f"[{source}] specialists[{i}].gpu_ids has {len(gpu_ids)} entries "
                f"but tensor_parallel_size={tp}. They must match."
            )
        specialists.append(
            SpecialistConfig(
                name=s["name"],
                model=s["model"],
                port=int(s["port"]),
                field=s["field"],
                backend=spec_backend,
                gpu=int(s.get("gpu", 0)),
                gpu_ids=gpu_ids,
                gpu_memory_utilization=gpu_util,
                max_model_len=int(s.get("max_model_len", 2048)),
                quantization=s.get("quantization", "awq") or None,
                enforce_eager=bool(s.get("enforce_eager", True)),
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                host=str(s.get("host", "127.0.0.1")),
                scheme=str(s.get("scheme", "http")),
                endpoint_override=s.get("endpoint_override") or None,
                models_url_override=s.get("models_url_override") or None,
                mlflow_tracking_uri=s.get("mlflow_tracking_uri") or None,
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
    arb_tp = int(raw_arb.get("tensor_parallel_size", 1))
    arb_pp = int(raw_arb.get("pipeline_parallel_size", 1))
    if arb_tp > 1 and (arb_tp & (arb_tp - 1)) != 0:
        raise ValueError(f"[{source}] arbiter.tensor_parallel_size={arb_tp} must be a power of 2.")
    raw_arb_gpu_ids = raw_arb.get("gpu_ids")
    arb_gpu_ids = [int(g) for g in raw_arb_gpu_ids] if raw_arb_gpu_ids else None
    arbiter = ArbiterConfig(
        model=raw_arb["model"],
        port=int(raw_arb["port"]),
        backend=str(raw_arb.get("backend", backend)),
        gpu=int(raw_arb.get("gpu", 0)),
        gpu_ids=arb_gpu_ids,
        gpu_memory_utilization=arb_gpu_util,
        max_model_len=int(raw_arb.get("max_model_len", 2048)),
        quantization=raw_arb.get("quantization", "awq") or None,
        enforce_eager=bool(raw_arb.get("enforce_eager", True)),
        tensor_parallel_size=arb_tp,
        pipeline_parallel_size=arb_pp,
        host=str(raw_arb.get("host", "127.0.0.1")),
        scheme=str(raw_arb.get("scheme", "http")),
        endpoint_override=raw_arb.get("endpoint_override") or None,
        models_url_override=raw_arb.get("models_url_override") or None,
        mlflow_tracking_uri=raw_arb.get("mlflow_tracking_uri") or None,
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
        arbitration_mode=str(raw_router.get("arbitration_mode", "pairwise")),
        tau=float(raw_router.get("tau", 1.0)),
        retry=_load_retry_config(raw_router.get("retry") or {}),
        circuit_breaker=_load_cb_config(raw_router.get("circuit_breaker") or {}),
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
            regression_dataset=bg.get("regression_dataset") or None,
            regression_block=bool(bg.get("regression_block", True)),
            shadow_endpoint=bg.get("shadow_endpoint") or None,
            shadow_min_queries=int(bg.get("shadow_min_queries", 50)),
        )

    # ── Logging ────────────────────────────────────────────────────────────
    raw_log = raw.get("logging", {})
    _reject_unknown_keys(raw_log, _KNOWN_LOG_KEYS, "logging", source)
    logging_cfg = LoggingConfig(
        level=str(raw_log.get("level", "INFO")).upper(),
        format=str(raw_log.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")),
    )

    # ── Secrets (#19) ──────────────────────────────────────────────────────
    raw_secrets = raw.get("secrets", {})
    _reject_unknown_keys(raw_secrets, _KNOWN_SECRETS_KEYS, "secrets", source)
    _provider = str(raw_secrets.get("provider", "env"))
    if _provider not in ("env", "vault", "aws", "gcp"):
        raise ValueError(
            f"[{source}] secrets.provider must be one of "
            f"'env' | 'vault' | 'aws' | 'gcp', got {_provider!r}"
        )
    secrets_cfg = SecretsConfig(
        provider=_provider,
        region=str(raw_secrets.get("region", "us-east-1")),
        url=str(raw_secrets.get("url", "http://127.0.0.1:8200")),
        token_env=str(raw_secrets.get("token_env", "VAULT_TOKEN")),
    )

    # ── State (F-05) ───────────────────────────────────────────────────────
    raw_state = raw.get("state", {})
    _reject_unknown_keys(raw_state, _KNOWN_STATE_KEYS, "state", source)
    _state_backend = str(raw_state.get("backend", "sqlite"))
    if _state_backend not in ("sqlite", "files"):
        raise ValueError(
            f"[{source}] state.backend must be 'sqlite' or 'files', got {_state_backend!r}"
        )
    state_cfg = StateConfig(
        backend=_state_backend,
        path=str(raw_state.get("path", ".aua/state/aua.db")),
    )

    # ── Security ───────────────────────────────────────────────────────────
    raw_security = raw.get("security", {})
    _reject_unknown_keys(raw_security, _KNOWN_SECURITY_KEYS, "security", source)
    _sec_cors = raw_security.get("cors_origins")
    if _sec_cors is not None and not isinstance(_sec_cors, list):
        raise ValueError(f"[{source}] security.cors_origins must be a list of origins")
    security_cfg = SecurityConfig(
        cors_origins=[str(o) for o in _sec_cors] if _sec_cors is not None else None,
        mtls=dict(raw_security.get("mtls") or {}),
        encryption=dict(raw_security.get("encryption") or {}),
        auth_enabled=bool(raw_security.get("auth_enabled", False)),
        token_secret_env=str(raw_security.get("token_secret_env", "AUA_TOKEN_SECRET")),
        token_expiry_days=int(raw_security.get("token_expiry_days", 30)),
    )

    # ── Plugins (F-09) ─────────────────────────────────────────────────────
    raw_plugins = raw.get("plugins", {})
    if not isinstance(raw_plugins, dict):
        raise ValueError(f"[{source}] plugins must be a mapping of kind -> spec")
    _reject_unknown_keys(raw_plugins, _KNOWN_PLUGIN_KINDS, "plugins", source)
    plugins_cfg: dict[str, PluginSpec] = {}
    for _kind, _spec in raw_plugins.items():
        if not isinstance(_spec, dict):
            raise ValueError(f"[{source}] plugins.{_kind} must be a mapping")
        _reject_unknown_keys(_spec, _KNOWN_PLUGIN_ENTRY_KEYS, f"plugins.{_kind}", source)
        _ip = _spec.get("import_path", "")
        if not _ip or ":" not in _ip:
            raise ValueError(
                f"[{source}] plugins.{_kind}.import_path must look like "
                f"'package.module:ClassName', got {_ip!r}"
            )
        plugins_cfg[_kind] = PluginSpec(import_path=_ip, config=dict(_spec.get("config") or {}))

    # ── Hooks (F-10) ───────────────────────────────────────────────────────
    raw_hooks = raw.get("hooks", [])
    if not isinstance(raw_hooks, list):
        raise ValueError(f"[{source}] hooks must be a list of hook entries")
    hooks_cfg: list[HookSpec] = []
    for i, _h in enumerate(raw_hooks):
        if not isinstance(_h, dict):
            raise ValueError(f"[{source}] hooks[{i}] must be a mapping")
        _reject_unknown_keys(_h, _KNOWN_HOOK_ENTRY_KEYS, f"hooks[{i}]", source)
        _hp = _h.get("hook_point", "")
        if _hp not in _VALID_HOOK_POINTS:
            raise ValueError(
                f"[{source}] hooks[{i}].hook_point {_hp!r} is not a valid hook point. "
                f"Valid: {sorted(_VALID_HOOK_POINTS)}"
            )
        _ip = _h.get("import_path", "")
        if not _ip or ":" not in _ip:
            raise ValueError(
                f"[{source}] hooks[{i}].import_path must look like "
                f"'package.module:ClassName', got {_ip!r}"
            )
        hooks_cfg.append(
            HookSpec(
                hook_point=_hp,
                import_path=_ip,
                config=dict(_h.get("config") or {}),
                fail_closed=bool(_h.get("fail_closed", False)),
            )
        )

    # ── Middleware (F-11) ──────────────────────────────────────────────────
    raw_mw = raw.get("middleware", [])
    if not isinstance(raw_mw, list):
        raise ValueError(f"[{source}] middleware must be a list (ordered pipeline)")
    middleware_cfg: list[MiddlewareSpec] = []
    for i, _m in enumerate(raw_mw):
        if isinstance(_m, str):
            _ip, _mc = _m, {}
        elif isinstance(_m, dict):
            _reject_unknown_keys(_m, _KNOWN_PLUGIN_ENTRY_KEYS, f"middleware[{i}]", source)
            _ip, _mc = _m.get("import_path", ""), dict(_m.get("config") or {})
        else:
            raise ValueError(f"[{source}] middleware[{i}] must be a string or mapping")
        if not _ip or ":" not in _ip:
            raise ValueError(
                f"[{source}] middleware[{i}] import path must look like "
                f"'package.module:ClassName', got {_ip!r}"
            )
        middleware_cfg.append(MiddlewareSpec(import_path=_ip, config=_mc))

    # ── Experiment tracking (#47) ──────────────────────────────────────────
    raw_exp = raw.get("experiment_tracking", {}) or {}
    if raw_exp:
        _reject_unknown_keys(raw_exp, _KNOWN_EXPERIMENT_KEYS, "experiment_tracking", source)
    from aua.experiment_tracker import experiment_config_from_dict

    experiment_cfg = experiment_config_from_dict(raw_exp)

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
        secrets=secrets_cfg,
        state=state_cfg,
        security=security_cfg,
        plugins=plugins_cfg,
        hooks=hooks_cfg,
        middleware=middleware_cfg,
        experiment_tracking=experiment_cfg,
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
AVAILABLE_TIERS: list[str] = [
    "macbook",
    "gaming-pc",
    "single-4090",
    "quad-4090",
    "a100-cluster",
    "h100-cluster",
]

# Backward-compatible aliases — resolve to canonical names.
TIER_ALIASES: dict[str, str] = {
    "rtx4090": "single-4090",
    "a100": "a100-cluster",
    "h100": "h100-cluster",
    "gaming": "gaming-pc",
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
