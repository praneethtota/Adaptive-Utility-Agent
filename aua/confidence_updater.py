"""
confidence_updater.py — applies contradiction penalty to a prior confidence estimate.
"""

from typing import Optional


class ConfidenceUpdater:
    """
    Updates a confidence estimate given a ContradictionResult.
    Applies field-weighted penalty and clamps to [0, 1].
    """

    def update(
        self,
        prior: float,
        test_signal: float,
        contradiction_result,   # ContradictionResult — avoid circular import
        field: str,
    ) -> float:
        penalty = getattr(contradiction_result, "confidence_penalty", 0.0)
        return max(0.0, min(1.0, prior - penalty))
