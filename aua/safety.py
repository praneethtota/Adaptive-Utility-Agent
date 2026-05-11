"""
aua/safety.py — Safety and abstention policy for high-risk fields.

Controls when AUA abstains from answering (low confidence or high-risk domain)
rather than returning a potentially incorrect response.

YAML configuration:
    safety:
      abstention_enabled: true
      high_risk_fields:
        - medicine
        - law
        - surgery
        - aviation
      require_arbiter_for_high_risk: true
      min_confidence_for_direct_answer: 0.90

Usage:
    from aua.safety import SafetyPolicy
    policy = SafetyPolicy.from_config(cfg)
    if policy.should_abstain(domain="surgery", confidence=0.72):
        return policy.abstention_response(domain="surgery", confidence=0.72)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aua import FIELD_CONFIGS


@dataclass
class SafetyConfig:
    abstention_enabled: bool = True
    high_risk_fields: list[str] = field(default_factory=lambda: ["medicine", "surgery", "law"])
    require_arbiter_for_high_risk: bool = True
    min_confidence_for_direct_answer: float = 0.90


class SafetyPolicy:
    """
    Governs when AUA abstains from answering.

    Abstention occurs when:
    1. confidence < field.c_min  (field-specific threshold)
    2. confidence < min_confidence_for_direct_answer AND field is high-risk
    3. abstention_enabled = True (global switch)
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self._cfg = config or SafetyConfig()

    @classmethod
    def from_config(cls, aua_config: Any) -> SafetyPolicy:
        safety_raw = getattr(aua_config, "safety", None)
        if safety_raw is None:
            return cls()
        cfg = SafetyConfig(
            abstention_enabled=getattr(safety_raw, "abstention_enabled", True),
            high_risk_fields=getattr(
                safety_raw, "high_risk_fields", ["medicine", "surgery", "law"]
            ),
            require_arbiter_for_high_risk=getattr(
                safety_raw, "require_arbiter_for_high_risk", True
            ),
            min_confidence_for_direct_answer=getattr(
                safety_raw, "min_confidence_for_direct_answer", 0.90
            ),
        )
        return cls(cfg)

    def should_abstain(self, domain: str, confidence: float) -> bool:
        """Return True if AUA should abstain from answering."""
        if not self._cfg.abstention_enabled:
            return False

        # Check field-specific c_min
        field_cfg = FIELD_CONFIGS.get(domain)
        if field_cfg and confidence < field_cfg.c_min:
            return True

        # Check high-risk field threshold
        if (
            domain in self._cfg.high_risk_fields
            and confidence < self._cfg.min_confidence_for_direct_answer
        ):
            return True

        return False

    def requires_arbiter(self, domain: str) -> bool:
        """Return True if this domain always requires arbiter verification."""
        return self._cfg.require_arbiter_for_high_risk and domain in self._cfg.high_risk_fields

    def abstention_response(self, domain: str, confidence: float) -> str:
        """Return the abstention message to show the user."""
        field_cfg = FIELD_CONFIGS.get(domain)
        c_min = field_cfg.c_min if field_cfg else self._cfg.min_confidence_for_direct_answer
        return (
            f"I don't have sufficient confidence to answer this question accurately "
            f"in the {domain} domain.\n\n"
            f"My confidence score ({confidence:.0%}) is below the minimum threshold "
            f"({c_min:.0%}) required for this field.\n\n"
            "Please consult a qualified professional for this question."
        )

    @property
    def config(self) -> SafetyConfig:
        return self._cfg
