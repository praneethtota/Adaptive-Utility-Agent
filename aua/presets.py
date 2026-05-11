"""
aua/presets.py — Named starting configurations (presets) for aua init.

A preset is a opinionated combination of specialists and fields suited
to a specific use case. Presets are tier-agnostic: they define *what*
specialists to include, while the tier defines *how* to run them
(hardware, backend, VRAM allocation).

Available presets:
    coding      swe + math (default)
    research    swe + math + science
    legal       law + swe
    medical     surgery + medicine
    general     swe + math + law (broad coverage)
    creative    creative_writing + swe

Usage:
    aua init --preset coding --tier single-4090
    aua init --preset legal --tier macbook
    aua presets list
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PresetSpec:
    """A named starting configuration for aua init."""

    name: str
    description: str
    specialists: list[str]  # field names to include
    recommended_tiers: list[str]
    notes: str = ""


# ── Preset registry ────────────────────────────────────────────────────────────

PRESETS: dict[str, PresetSpec] = {
    "coding": PresetSpec(
        name="coding",
        description="Software engineering + mathematics. Best all-round starting point.",
        specialists=["software_engineering", "mathematics"],
        recommended_tiers=["macbook", "single-4090", "quad-4090", "a100-cluster"],
        notes="Default preset. Good for code generation, algorithm questions, and technical Q&A.",
    ),
    "research": PresetSpec(
        name="research",
        description="Software engineering + mathematics + science. For research and analysis tasks.",
        specialists=["software_engineering", "mathematics", "science"],
        recommended_tiers=["single-4090", "quad-4090", "a100-cluster"],
        notes="Adds a science specialist for empirical reasoning. Requires more VRAM than 'coding'.",
    ),
    "legal": PresetSpec(
        name="legal",
        description="Law + software engineering. For legal tech and contract analysis.",
        specialists=["law", "software_engineering"],
        recommended_tiers=["macbook", "single-4090", "quad-4090", "a100-cluster"],
        notes="High-stakes: law field uses c_min=0.85, penalty=5x. Review outputs carefully.",
    ),
    "medical": PresetSpec(
        name="medical",
        description="Surgery + medicine. Highest-stakes configuration — use with care.",
        specialists=["surgery", "medicine"],
        recommended_tiers=["single-4090", "quad-4090", "a100-cluster"],
        notes=(
            "CRITICAL: surgery field uses c_min=0.95, penalty=10x. "
            "Always have a qualified professional review outputs. "
            "Never use for actual clinical decisions."
        ),
    ),
    "general": PresetSpec(
        name="general",
        description="Software engineering + mathematics + law. Broad coverage.",
        specialists=["software_engineering", "mathematics", "law"],
        recommended_tiers=["quad-4090", "a100-cluster"],
        notes="Three specialists — needs more VRAM. Use quad-4090 or a100-cluster tier.",
    ),
    "creative": PresetSpec(
        name="creative",
        description="Creative writing + software engineering. For content and code generation.",
        specialists=["creative_writing", "software_engineering"],
        recommended_tiers=["macbook", "single-4090", "quad-4090", "a100-cluster"],
        notes="creative_writing field uses low c_min=0.05 and high w_efficacy=0.80.",
    ),
}

AVAILABLE_PRESETS: list[str] = sorted(PRESETS.keys())


def get_preset(name: str) -> PresetSpec:
    """Return a PresetSpec by name. Raises ValueError for unknown presets."""
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset {name!r}. Available: {known}")
    return PRESETS[name]
