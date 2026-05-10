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
        print(s.name, s.endpoint)

    # Access field weights
    from aua.config import FIELD_CONFIGS
    cfg = FIELD_CONFIGS["software_engineering"]
    print(cfg.penalty_multiplier)   # 2.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ── FieldConfig ───────────────────────────────────────────────────────────────
# Unchanged from the POC — these are the whitepaper field weights and bounds.

@dataclass
class FieldConfig:
    name: str
    w_efficacy: float       # weight on efficacy term (E)
    w_confidence: float     # weight on confidence term (C)
    w_curiosity: float      # weight on curiosity term (K)
    c_min: float            # minimum confidence to act (below → abstain)
    e_min: float            # minimum efficacy to act
    penalty_multiplier: float  # contradiction penalty scale (surgery=10×, SWE=2×)

    def __post_init__(self) -> None:
        total = self.w_efficacy + self.w_confidence + self.w_curiosity
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0 for field '{self.name}' (got {total:.4f})"
            )


FIELD_CONFIGS: Dict[str, FieldConfig] = {
    "surgery": FieldConfig(
        name="surgery",
        w_efficacy=0.20, w_confidence=0.70, w_curiosity=0.10,
        c_min=0.95, e_min=0.90, penalty_multiplier=10.0,
    ),
    "aviation": FieldConfig(
        name="aviation",
        w_efficacy=0.20, w_confidence=0.70, w_curiosity=0.10,
        c_min=0.95, e_min=0.90, penalty_multiplier=10.0,
    ),
    "law": FieldConfig(
        name="law",
        w_efficacy=0.30, w_confidence=0.60, w_curiosity=0.10,
        c_min=0.85, e_min=0.80, penalty_multiplier=5.0,
    ),
    "structural_engineering": FieldConfig(
        name="structural_engineering",
        w_efficacy=0.40, w_confidence=0.50, w_curiosity=0.10,
        c_min=0.80, e_min=0.75, penalty_multiplier=4.0,
    ),
    "software_engineering": FieldConfig(
        name="software_engineering",
        w_efficacy=0.55, w_confidence=0.35, w_curiosity=0.10,
        c_min=0.70, e_min=0.65, penalty_multiplier=2.0,
    ),
    "mathematics": FieldConfig(
        name="mathematics",
        w_efficacy=0.50, w_confidence=0.40, w_curiosity=0.10,
        c_min=0.75, e_min=0.70, penalty_multiplier=3.0,
    ),
    "stem_research": FieldConfig(
        name="stem_research",
        w_efficacy=0.50, w_confidence=0.30, w_curiosity=0.20,
        c_min=0.65, e_min=0.60, penalty_multiplier=2.0,
    ),
    "education": FieldConfig(
        name="education",
        w_efficacy=0.50, w_confidence=0.30, w_curiosity=0.20,
        c_min=0.60, e_min=0.55, penalty_multiplier=1.5,
    ),
    "art": FieldConfig(
        name="art",
        w_efficacy=0.80, w_confidence=0.10, w_curiosity=0.10,
        c_min=0.10, e_min=0.20, penalty_multiplier=1.0,
    ),
    "creative_writing": FieldConfig(
        name="creative_writing",
        w_efficacy=0.80, w_confidence=0.05, w_curiosity=0.15,
        c_min=0.05, e_min=0.15, penalty_multiplier=1.0,
    ),
    "general": FieldConfig(
        name="general",
        w_efficacy=0.50, w_confidence=0.35, w_curiosity=0.15,
        c_min=0.50, e_min=0.50, penalty_multiplier=1.5,
    ),
}


def get_effective_config(field_distribution: Dict[str, float]) -> FieldConfig:
    """
    Blend FieldConfigs by probability weight when domain is ambiguous.
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
        blended.w_efficacy         += prob * cfg.w_efficacy
        blended.w_confidence       += prob * cfg.w_confidence
        blended.w_curiosity        += prob * cfg.w_curiosity
        blended.c_min              += prob * cfg.c_min
        blended.e_min              += prob * cfg.e_min
        blended.penalty_multiplier += prob * cfg.penalty_multiplier

    return blended


# ── Deployment config dataclasses ─────────────────────────────────────────────

@dataclass
class SpecialistConfig:
    """One specialist server — vLLM or Ollama."""
    name: str
    model: str
    port: int
    field: str
    backend: str = "vllm"
    gpu: int = 0
    gpu_memory_utilization: float = 0.34
    max_model_len: int = 2048
    quantization: Optional[str] = "awq"
    enforce_eager: bool = True

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        if self.backend == "ollama":
            return f"http://localhost:{self.port}/api/tags"
        return f"http://localhost:{self.port}/v1/models"

    @property
    def serve_model_name(self) -> str:
        if self.backend == "ollama":
            return self.model
        return self.name

    @property
    def field_config(self) -> FieldConfig:
        return FIELD_CONFIGS.get(self.field, FIELD_CONFIGS["general"])

    def vllm_command(self) -> list[str]:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--served-model-name", self.name,
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        if self.enforce_eager:
            cmd += ["--enforce-eager"]
        return cmd


@dataclass
class ArbiterConfig:
    model: str
    port: int
    backend: str = "vllm"
    gpu: int = 0
    gpu_memory_utilization: float = 0.18
    max_model_len: int = 2048
    quantization: Optional[str] = "awq"
    enforce_eager: bool = True

    @property
    def endpoint(self) -> str:
        return f"http://localhost:{self.port}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        if self.backend == "ollama":
            return f"http://localhost:{self.port}/api/tags"
        return f"http://localhost:{self.port}/v1/models"

    @property
    def serve_model_name(self) -> str:
        if self.backend == "ollama":
            return self.model
        return "arbiter"

    def vllm_command(self) -> list[str]:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--served-model-name", "arbiter",
        ]
        if self.quantization:
            cmd += ["--quantization", self.quantization]
        if self.enforce_eager:
            cmd += ["--enforce-eager"]
        return cmd


@dataclass
class RouterConfig:
    port: int = 8000
    single_domain_threshold: float = 0.75
    fanout_threshold: float = 0.30
    specialist_timeout: float = 60.0
    host: str = "0.0.0.0"


@dataclass
class BlueGreenFieldConfig:
    delta: float = 0.025
    T_min: int = 10
    tau: float = 0.20


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class AUAConfig:
    version: str
    mode: str
    specialists: List[SpecialistConfig]
    arbiter: ArbiterConfig
    router: RouterConfig
    blue_green: Dict[str, BlueGreenFieldConfig]
    backend: str = "vllm"
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    _specialist_by_name: Dict[str, SpecialistConfig] = field(
        default_factory=dict, init=False, repr=False
    )
    _specialist_by_field: Dict[str, SpecialistConfig] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        for s in self.specialists:
            self._specialist_by_name[s.name]  = s
            self._specialist_by_field[s.field] = s

    def specialist(self, name: str) -> SpecialistConfig:
        if name not in self._specialist_by_name:
            raise KeyError(f"No specialist named '{name}'. "
                           f"Available: {list(self._specialist_by_name)}")
        return self._specialist_by_name[name]

    def specialist_for_field(self, field_name: str) -> Optional[SpecialistConfig]:
        return self._specialist_by_field.get(field_name)

    def all_endpoints(self) -> Dict[str, str]:
        eps = {s.name: s.endpoint for s in self.specialists}
        eps["arbiter"] = self.arbiter.endpoint
        return eps

    def blue_green_for(self, specialist_name: str) -> BlueGreenFieldConfig:
        return self.blue_green.get(specialist_name, BlueGreenFieldConfig())


# ── YAML loader ───────────────────────────────────────────────────────────────

def load_config(path: str | os.PathLike = "aua_config.yaml") -> AUAConfig:
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
    aua_block = raw.get("aua", {})
    version = str(aua_block.get("version", "0.5"))
    mode    = str(aua_block.get("mode", "local"))
    backend = str(aua_block.get("backend", "vllm"))

    raw_specialists = raw.get("specialists", [])
    if not raw_specialists:
        raise ValueError(f"[{source}] 'specialists' list is required and must not be empty.")

    specialists: List[SpecialistConfig] = []
    for i, s in enumerate(raw_specialists):
        _require(s, ["name", "model", "port", "field"],
                 context=f"specialists[{i}]", source=source)
        spec_backend = str(s.get("backend", backend))
        specialists.append(SpecialistConfig(
            name=s["name"], model=s["model"], port=int(s["port"]),
            field=s["field"], backend=spec_backend,
            gpu=int(s.get("gpu", 0)),
            gpu_memory_utilization=float(s.get("gpu_memory_utilization", 0.34)),
            max_model_len=int(s.get("max_model_len", 2048)),
            quantization=s.get("quantization", "awq") or None,
            enforce_eager=bool(s.get("enforce_eager", True)),
        ))

    raw_arb = raw.get("arbiter", {})
    _require(raw_arb, ["model", "port"], context="arbiter", source=source)
    arbiter = ArbiterConfig(
        model=raw_arb["model"], port=int(raw_arb["port"]),
        backend=str(raw_arb.get("backend", backend)),
        gpu=int(raw_arb.get("gpu", 0)),
        gpu_memory_utilization=float(raw_arb.get("gpu_memory_utilization", 0.18)),
        max_model_len=int(raw_arb.get("max_model_len", 2048)),
        quantization=raw_arb.get("quantization", "awq") or None,
        enforce_eager=bool(raw_arb.get("enforce_eager", True)),
    )

    raw_router = raw.get("router", {})
    router = RouterConfig(
        port=int(raw_router.get("port", 8000)),
        single_domain_threshold=float(raw_router.get("single_domain_threshold", 0.75)),
        fanout_threshold=float(raw_router.get("fanout_threshold", 0.30)),
        specialist_timeout=float(raw_router.get("specialist_timeout", 60.0)),
        host=str(raw_router.get("host", "0.0.0.0")),
    )

    raw_bg = raw.get("blue_green", {})
    blue_green: Dict[str, BlueGreenFieldConfig] = {}
    for name, bg in raw_bg.items():
        blue_green[name] = BlueGreenFieldConfig(
            delta=float(bg.get("delta", 0.025)),
            T_min=int(bg.get("T_min", 10)),
            tau=float(bg.get("tau", 0.20)),
        )

    raw_log = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=str(raw_log.get("level", "INFO")).upper(),
        format=str(raw_log.get("format",
                               "%(asctime)s [%(levelname)s] %(name)s: %(message)s")),
    )

    for s in specialists:
        if s.field not in FIELD_CONFIGS:
            raise ValueError(
                f"[{source}] Specialist '{s.name}' references unknown field "
                f"'{s.field}'. Valid fields: {sorted(FIELD_CONFIGS)}"
            )

    return AUAConfig(
        version=version, mode=mode, specialists=specialists,
        backend=backend, arbiter=arbiter, router=router,
        blue_green=blue_green, logging=logging_cfg,
    )


def _require(d: dict, keys: list[str], context: str, source: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(
            f"[{source}] '{context}' is missing required field(s): {missing}"
        )


# ── Tier loader ───────────────────────────────────────────────────────────────

AVAILABLE_TIERS = ["macbook", "rtx4090", "a100"]

def load_tier(tier: str) -> AUAConfig:
    if tier not in AVAILABLE_TIERS:
        raise ValueError(
            f"Unknown tier '{tier}'. Available: {AVAILABLE_TIERS}\n"
            f"Use 'aua serve --config my.yaml' for a custom config."
        )
    tier_path = Path(__file__).parent / "tiers" / f"{tier}.yaml"
    return load_config(tier_path)
