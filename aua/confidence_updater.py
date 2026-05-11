"""
confidence_updater.py — applies contradiction penalty to a prior confidence estimate.
"""


class ConfidenceUpdater:
    """
    Updates a confidence estimate given a contradiction result.
    Applies field-weighted penalty and clamps to [0, 1].
    """

    def update(
        self,
        prior: float,
        test_signal: float,
        contradiction_result,  # ContradictionResult
        field: str,
    ) -> float:
        penalty = getattr(contradiction_result, "confidence_penalty", 0.0)
        updated = prior - penalty
        return max(0.0, min(1.0, updated))
