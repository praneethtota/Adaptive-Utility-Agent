"""
Example custom utility scorer plugin.

Tests:
    aua extensions test \
      --kind utility_scorer \
      --import-path plugins.custom_utility:RiskWeightedUtilityScorer
"""

from __future__ import annotations

HIGH_RISK_FIELDS = frozenset(["surgery", "medicine", "law", "aviation"])


class RiskWeightedUtilityScorer:
    """
    Risk-weighted utility scorer.

    For high-risk fields (surgery, medicine, law, aviation), applies a
    penalty when confidence is low — making the system prefer abstention
    over a confidently-wrong answer.

    Config:
        risk_weight (float): penalty multiplier for low-confidence high-risk
                             responses. Default 0.7.
    """

    def __init__(self, risk_weight: float = 0.7) -> None:
        self.risk_weight = float(risk_weight)

    def score(
        self,
        response: str,
        field: str,
        prior_u: float,
        confidence: float,
        metadata: dict,
    ) -> float:
        """Compute risk-adjusted utility score."""
        base_u = prior_u * 0.5 + confidence * 0.5

        if field in HIGH_RISK_FIELDS:
            # Apply risk penalty: low confidence in high-risk domain → lower U
            risk_penalty = self.risk_weight * (1.0 - confidence)
            adjusted = base_u * (1.0 - risk_penalty)
            return max(0.0, min(1.0, adjusted))

        return max(0.0, min(1.0, base_u))
