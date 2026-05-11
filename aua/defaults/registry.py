"""
aua/defaults/registry.py — Framework defaults registry.

All default values that ship with AUA are registered here. This makes
batteries-included defaults inspectable via `aua defaults show` without
digging through source code.

Categories:
    models      Built-in model aliases (e.g. qwen-coder-7b-awq)
    fields      Built-in field configs with weights and thresholds
    presets     Named specialist configurations
    routing     Default routing thresholds
    utility     Default utility function weights
    security    Default security settings
    prompts     Built-in prompt template names

Usage:
    from aua.defaults.registry import get_defaults
    defaults = get_defaults("fields")
"""

from __future__ import annotations

from typing import Any


def get_defaults(category: str) -> dict[str, Any]:
    """Return the defaults for a given category."""
    registry = {
        "models": _model_defaults(),
        "fields": _field_defaults(),
        "presets": _preset_defaults(),
        "routing": _routing_defaults(),
        "utility": _utility_defaults(),
        "security": _security_defaults(),
        "prompts": _prompt_defaults(),
    }
    if category not in registry:
        raise ValueError(f"Unknown category {category!r}. Available: {sorted(registry.keys())}")
    return registry[category]


def list_categories() -> list[str]:
    return ["models", "fields", "presets", "routing", "utility", "security", "prompts"]


def _model_defaults() -> dict[str, Any]:
    return {
        "qwen-coder-7b-awq": {
            "full_id": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
            "provider": "Qwen",
            "backend": "vllm",
            "quantization": "awq",
            "recommended_vram_gb": 8,
            "recommended_fields": ["software_engineering"],
        },
        "qwen-math-7b-awq": {
            "full_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "provider": "Qwen",
            "backend": "vllm",
            "quantization": "awq",
            "recommended_vram_gb": 8,
            "recommended_fields": ["mathematics", "general"],
        },
        "qwen-3b-awq": {
            "full_id": "Qwen/Qwen2.5-3B-Instruct-AWQ",
            "provider": "Qwen",
            "backend": "vllm",
            "quantization": "awq",
            "recommended_vram_gb": 4,
            "recommended_fields": ["general"],
            "typical_use": "arbiter",
        },
        "qwen-coder-7b-ollama": {
            "full_id": "qwen2.5-coder:7b",
            "provider": "Qwen",
            "backend": "ollama",
            "quantization": None,
            "recommended_fields": ["software_engineering"],
        },
        "qwen-7b-ollama": {
            "full_id": "qwen2.5:7b",
            "provider": "Qwen",
            "backend": "ollama",
            "quantization": None,
            "recommended_fields": ["mathematics", "general"],
        },
        "qwen-3b-ollama": {
            "full_id": "qwen2.5:3b",
            "provider": "Qwen",
            "backend": "ollama",
            "quantization": None,
            "recommended_fields": ["general"],
            "typical_use": "arbiter",
        },
    }


def _field_defaults() -> dict[str, Any]:
    from aua import FIELD_CONFIGS

    return {
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


def _preset_defaults() -> dict[str, Any]:
    from aua.presets import PRESETS

    return {
        name: {
            "description": p.description,
            "specialists": p.specialists,
            "recommended_tiers": p.recommended_tiers,
            "notes": p.notes,
        }
        for name, p in PRESETS.items()
    }


def _routing_defaults() -> dict[str, Any]:
    return {
        "single_domain_threshold": 0.75,
        "fanout_threshold": 0.30,
        "specialist_timeout": 60.0,
        "cors_origins": ["*"],
        "description": {
            "single_domain_threshold": (
                "Route to a single specialist when one field scores above this. "
                "Higher = more focused routing. Range: (0, 1]."
            ),
            "fanout_threshold": (
                "Fan out to multiple specialists when two+ fields both exceed this. "
                "Lower = more cross-domain queries. Range: [0, 1)."
            ),
            "specialist_timeout": "Seconds before a specialist call is timed out (AUA_SPECIALIST_TIMEOUT).",
        },
    }


def _utility_defaults() -> dict[str, Any]:
    return {
        "formula": "U = w_e * E + w_c * C + w_k * K",
        "components": {
            "E": "Efficacy — Mann-Whitney dominance probability over prior outputs. Range [0, 1].",
            "C": "Confidence — Kalman-filtered consistency score, penalized on contradiction. Range [0, 1].",
            "K": "Curiosity — UCB exploration bonus for uncertain domains. Capped at 0.50 of total U.",
        },
        "default_weights": {"w_efficacy": 0.50, "w_confidence": 0.35, "w_curiosity": 0.15},
        "kalman": {"process_noise": 1e-5, "measurement_noise": 1e-2},
    }


def _security_defaults() -> dict[str, Any]:
    return {
        "auth_enabled": False,
        "cors_origins": ["*"],
        "expose_docs": True,
        "expose_config": True,
        "log_queries": True,
        "log_responses": False,
        "redact_secrets_in_logs": True,
        "note": "auth_enabled=False is development-only. Never deploy publicly without auth.",
    }


def _prompt_defaults() -> dict[str, Any]:
    from aua.templates.registry import list_templates

    return {
        "classifier_template": "classifier_v1",
        "arbiter_template": "arbiter_balanced_v1",
        "abstention_template": "abstention_v1",
        "available_templates": list_templates(),
    }
