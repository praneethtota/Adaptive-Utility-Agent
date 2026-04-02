"""
Utility scorer: computes Efficacy, Confidence, Curiosity, and final U.
Maintains running scores per domain and logs history.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from config import FieldConfig


@dataclass
class TaskScore:
    task_id: str
    field: str
    efficacy: float
    confidence: float
    curiosity: float
    utility: float
    timestamp: str
    below_minimum: bool
    notes: str = ""


@dataclass
class DomainState:
    """Running state for a domain — confidence and efficacy evolve over time."""
    domain: str
    confidence: float = 0.5         # starts uncertain
    efficacy: float = 0.5           # starts at baseline
    interaction_count: int = 0
    contradiction_count: int = 0
    success_count: int = 0
    potential_ceiling: float = 0.9  # estimated max achievable efficacy


class UtilityScorer:
    """
    Computes and tracks the utility function:
    U = w_e * E + w_c * C + w_k * K
    subject to C >= C_min and E >= E_min
    """

    def __init__(self):
        self.domain_states: Dict[str, DomainState] = {}
        self.history: List[TaskScore] = []

    def score(
        self,
        task_id: str,
        field_config: FieldConfig,
        test_pass_rate: float,          # 0.0 to 1.0 — from test runner
        human_baseline_score: float,    # 0.0 to 1.0 — normalized human performance
        contradiction_penalty: float,   # from ContradictionDetector
        problem_novelty: float,         # 0.0 to 1.0 — how new is this problem type
    ) -> TaskScore:

        domain = field_config.name
        state = self._get_or_create_state(domain)

        # ── Efficacy ──────────────────────────────────────────────────────────
        # How much better than human baseline, normalized to [0, 1]
        # test_pass_rate is the agent quality signal
        # human_baseline_score is what a typical human would score
        efficacy = self._compute_efficacy(test_pass_rate, human_baseline_score)

        # ── Confidence ────────────────────────────────────────────────────────
        # Update running confidence based on this interaction
        confidence = self._update_confidence(
            state, test_pass_rate, contradiction_penalty, field_config
        )

        # ── Curiosity ─────────────────────────────────────────────────────────
        # Pull toward domains where we're weak but upside is high
        curiosity = self._compute_curiosity(state, problem_novelty)

        # ── Utility ───────────────────────────────────────────────────────────
        utility = (
            field_config.w_efficacy * efficacy +
            field_config.w_confidence * confidence +
            field_config.w_curiosity * curiosity
        )

        # Check minimum bounds
        below_min = confidence < field_config.c_min or efficacy < field_config.e_min
        notes = ""
        if below_min:
            notes = (
                f"BELOW MINIMUM: confidence={confidence:.2f} (min={field_config.c_min}), "
                f"efficacy={efficacy:.2f} (min={field_config.e_min}). "
                f"Agent should abstain or escalate."
            )

        # Update domain state
        state.interaction_count += 1
        if test_pass_rate >= 0.8:
            state.success_count += 1
        state.efficacy = efficacy  # track current efficacy

        task_score = TaskScore(
            task_id=task_id,
            field=domain,
            efficacy=round(efficacy, 4),
            confidence=round(confidence, 4),
            curiosity=round(curiosity, 4),
            utility=round(utility, 4),
            timestamp=datetime.utcnow().isoformat(),
            below_minimum=below_min,
            notes=notes
        )

        self.history.append(task_score)
        return task_score

    def get_domain_summary(self, domain: str) -> dict:
        """Return current state for a domain."""
        state = self.domain_states.get(domain)
        if not state:
            return {}
        return {
            "domain": domain,
            "confidence": round(state.confidence, 4),
            "efficacy": round(state.efficacy, 4),
            "interactions": state.interaction_count,
            "success_rate": round(state.success_count / max(state.interaction_count, 1), 4),
            "contradiction_rate": round(state.contradiction_count / max(state.interaction_count, 1), 4),
        }

    def get_utility_trend(self, domain: Optional[str] = None, last_n: int = 20) -> List[float]:
        """Return recent utility scores, optionally filtered by domain."""
        scores = [s for s in self.history if domain is None or s.field == domain]
        return [s.utility for s in scores[-last_n:]]

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_or_create_state(self, domain: str) -> DomainState:
        if domain not in self.domain_states:
            self.domain_states[domain] = DomainState(domain=domain)
        return self.domain_states[domain]

    def _compute_efficacy(self, agent_score: float, human_baseline: float) -> float:
        """
        Efficacy = how much better than baseline.
        If agent matches human: 0.5. If agent beats human: > 0.5. Below: < 0.5.
        Normalized to [0, 1].
        """
        if human_baseline == 0:
            return agent_score
        ratio = agent_score / human_baseline
        # Sigmoid-like normalization so ratio of 1.0 → 0.5, 2.0 → ~0.73
        return 1.0 - 1.0 / (1.0 + ratio)

    def _update_confidence(
        self,
        state: DomainState,
        test_pass_rate: float,
        contradiction_penalty: float,
        config: FieldConfig
    ) -> float:
        """
        Update running confidence using exponential moving average.
        Contradictions apply a multiplied penalty.
        """
        # Base signal: test pass rate is a confidence input
        raw_signal = test_pass_rate

        # Apply contradiction penalty (amplified by field multiplier)
        effective_penalty = contradiction_penalty * config.penalty_multiplier
        penalized = raw_signal * (1.0 - effective_penalty)

        # Track contradiction count
        if contradiction_penalty > 0:
            state.contradiction_count += 1

        # EMA update: new confidence is blend of prior and current signal
        alpha = 0.2  # learning rate — lower = more stable
        state.confidence = (1 - alpha) * state.confidence + alpha * penalized

        # Clamp to [0, 1]
        state.confidence = max(0.0, min(1.0, state.confidence))
        return state.confidence

    def _compute_curiosity(self, state: DomainState, novelty: float) -> float:
        """
        Curiosity = potential gain × (1 - current confidence)
        High when: domain is new AND upside is large
        Low when: domain is well-understood OR upside is small
        """
        potential_gain = state.potential_ceiling - state.confidence
        potential_gain = max(0.0, potential_gain)
        return potential_gain * novelty
