"""
Field-specific weights, bounds, and penalty multipliers.
Derived from existing societal competence standards.
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class FieldConfig:
    name: str
    w_efficacy: float       # weight on efficacy term
    w_confidence: float     # weight on confidence term
    w_curiosity: float      # weight on curiosity term
    c_min: float            # minimum confidence to act
    e_min: float            # minimum efficacy to act
    penalty_multiplier: float  # contradiction penalty scale

    def __post_init__(self):
        assert abs(self.w_efficacy + self.w_confidence + self.w_curiosity - 1.0) < 1e-6, \
            f"Weights must sum to 1.0 for field {self.name}"


FIELD_CONFIGS: Dict[str, FieldConfig] = {
    "surgery": FieldConfig(
        name="surgery",
        w_efficacy=0.20, w_confidence=0.70, w_curiosity=0.10,
        c_min=0.95, e_min=0.90, penalty_multiplier=10.0
    ),
    "aviation": FieldConfig(
        name="aviation",
        w_efficacy=0.20, w_confidence=0.70, w_curiosity=0.10,
        c_min=0.95, e_min=0.90, penalty_multiplier=10.0
    ),
    "law": FieldConfig(
        name="law",
        w_efficacy=0.30, w_confidence=0.60, w_curiosity=0.10,
        c_min=0.85, e_min=0.80, penalty_multiplier=5.0
    ),
    "structural_engineering": FieldConfig(
        name="structural_engineering",
        w_efficacy=0.40, w_confidence=0.50, w_curiosity=0.10,
        c_min=0.80, e_min=0.75, penalty_multiplier=4.0
    ),
    "software_engineering": FieldConfig(
        name="software_engineering",
        w_efficacy=0.55, w_confidence=0.35, w_curiosity=0.10,
        c_min=0.70, e_min=0.65, penalty_multiplier=2.0
    ),
    "stem_research": FieldConfig(
        name="stem_research",
        w_efficacy=0.50, w_confidence=0.30, w_curiosity=0.20,
        c_min=0.65, e_min=0.60, penalty_multiplier=2.0
    ),
    "education": FieldConfig(
        name="education",
        w_efficacy=0.50, w_confidence=0.30, w_curiosity=0.20,
        c_min=0.60, e_min=0.55, penalty_multiplier=1.5
    ),
    "art": FieldConfig(
        name="art",
        w_efficacy=0.80, w_confidence=0.10, w_curiosity=0.10,
        c_min=0.10, e_min=0.20, penalty_multiplier=1.0
    ),
    "creative_writing": FieldConfig(
        name="creative_writing",
        w_efficacy=0.80, w_confidence=0.05, w_curiosity=0.15,
        c_min=0.05, e_min=0.15, penalty_multiplier=1.0
    ),
    "general": FieldConfig(
        name="general",
        w_efficacy=0.50, w_confidence=0.35, w_curiosity=0.15,
        c_min=0.50, e_min=0.50, penalty_multiplier=1.5
    ),
}


def get_effective_config(field_distribution: Dict[str, float]) -> FieldConfig:
    """
    When field is ambiguous, blend configs by probability weight.
    This makes the agent more conservative under ambiguity.

    Args:
        field_distribution: dict of {field_name: probability}

    Returns:
        Blended FieldConfig
    """
    assert abs(sum(field_distribution.values()) - 1.0) < 1e-6, \
        "Field distribution must sum to 1.0"

    blended = FieldConfig.__new__(FieldConfig)
    blended.name = "blended"
    blended.w_efficacy = 0.0
    blended.w_confidence = 0.0
    blended.w_curiosity = 0.0
    blended.c_min = 0.0
    blended.e_min = 0.0
    blended.penalty_multiplier = 0.0

    for field, prob in field_distribution.items():
        cfg = FIELD_CONFIGS.get(field, FIELD_CONFIGS["general"])
        blended.w_efficacy += prob * cfg.w_efficacy
        blended.w_confidence += prob * cfg.w_confidence
        blended.w_curiosity += prob * cfg.w_curiosity
        blended.c_min += prob * cfg.c_min
        blended.e_min += prob * cfg.e_min
        blended.penalty_multiplier += prob * cfg.penalty_multiplier

    return blended
