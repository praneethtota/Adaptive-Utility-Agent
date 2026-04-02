"""
Utility scorer: computes Efficacy, Confidence, Curiosity, and final U.
Maintains running scores per domain and logs history.

Changes from v0.1:
- Growing curiosity function: K scales with log(1 + interactions_without_novelty)
- 50% curiosity cap: K_effective = min(K_raw, (w_e·E + w_c·C) / w_k)
  This prevents utility gaming while keeping curiosity meaningful.
  K can never be the dominant term — capped at 50% of total U.
- Field-specific alpha for curiosity growth rate
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from config import FieldConfig


# Growth rate of curiosity pressure per field.
# High for exploratory fields, near-zero for high-stakes fields.
CURIOSITY_ALPHA: Dict[str, float] = {
    "surgery":              0.00,
    "aviation":             0.00,
    "law":                  0.02,
    "structural_engineering": 0.03,
    "software_engineering": 0.08,
    "stem_research":        0.12,
    "education":            0.08,
    "art":                  0.15,
    "creative_writing":     0.15,
    "general":              0.07,
    "blended":              0.07,
}


@dataclass
class TaskScore:
    task_id: str
    field: str
    efficacy: float
    confidence: float
    curiosity_raw: float        # before cap
    curiosity_effective: float  # after cap
    utility: float
    timestamp: str
    below_minimum: bool
    curiosity_capped: bool = False   # was the cap active?
    notes: str = ""


@dataclass
class DomainState:
    """Running state for a domain — all values evolve over time."""
    domain: str
    confidence: float = 0.5
    efficacy: float = 0.5
    interaction_count: int = 0
    contradiction_count: int = 0
    success_count: int = 0
    potential_ceiling: float = 0.9
    interactions_without_novelty: int = 0   # resets on novel problem


class UtilityScorer:
    """
    Computes and tracks the utility function:

        U = w_e·E + w_c·C + w_k·K_effective

    Where:
        K_raw       = potential_ceiling × (1 - C) × growth(t, field)
        K_effective = min(K_raw, (w_e·E + w_c·C) / w_k)   [50% cap]
        growth      = 1 + α(field) × log(1 + interactions_without_novelty)

    Subject to: C ≥ C_min(field), E ≥ E_min(field)
    """

    def __init__(self):
        self.domain_states: Dict[str, DomainState] = {}
        self.history: List[TaskScore] = []

    def score(
        self,
        task_id: str,
        field_config: FieldConfig,
        test_pass_rate: float,        # 0.0–1.0 from test runner
        human_baseline_score: float,  # 0.0–1.0 normalized human performance
        contradiction_penalty: float, # from ContradictionDetector
        problem_novelty: float,       # 0.0–1.0: how new is this problem type
    ) -> TaskScore:

        domain = field_config.name
        state = self._get_or_create_state(domain)

        # ── Efficacy ──────────────────────────────────────────────────────────
        efficacy = self._compute_efficacy(test_pass_rate, human_baseline_score)

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = self._update_confidence(
            state, test_pass_rate, contradiction_penalty, field_config
        )

        # ── Curiosity (with growth and cap) ───────────────────────────────────
        curiosity_raw, curiosity_effective, capped = self._compute_curiosity(
            state, field_config, problem_novelty, efficacy, confidence
        )

        # ── Update novelty counter ────────────────────────────────────────────
        # Novel problem → reset counter (exploration rewarded)
        # Familiar problem → increment counter (curiosity pressure grows)
        if problem_novelty >= 0.6:
            state.interactions_without_novelty = 0
        else:
            state.interactions_without_novelty += 1

        # ── Utility ───────────────────────────────────────────────────────────
        utility = (
            field_config.w_efficacy * efficacy +
            field_config.w_confidence * confidence +
            field_config.w_curiosity * curiosity_effective
        )

        # ── Minimum bounds check ──────────────────────────────────────────────
        below_min = confidence < field_config.c_min or efficacy < field_config.e_min
        notes = ""
        if below_min:
            notes = (
                f"BELOW MINIMUM: confidence={confidence:.2f} (min={field_config.c_min}), "
                f"efficacy={efficacy:.2f} (min={field_config.e_min}). "
                f"Agent should abstain or escalate."
            )
        if capped:
            cap_note = (
                f"Curiosity capped: raw={curiosity_raw:.3f} → "
                f"effective={curiosity_effective:.3f} (50% rule)"
            )
            notes = f"{notes}  |  {cap_note}".strip(" |")

        # ── Update domain state ───────────────────────────────────────────────
        state.interaction_count += 1
        if test_pass_rate >= 0.8:
            state.success_count += 1
        state.efficacy = efficacy

        task_score = TaskScore(
            task_id=task_id,
            field=domain,
            efficacy=round(efficacy, 4),
            confidence=round(confidence, 4),
            curiosity_raw=round(curiosity_raw, 4),
            curiosity_effective=round(curiosity_effective, 4),
            utility=round(utility, 4),
            timestamp=datetime.utcnow().isoformat(),
            below_minimum=below_min,
            curiosity_capped=capped,
            notes=notes
        )

        self.history.append(task_score)
        return task_score

    def get_domain_summary(self, domain: str) -> dict:
        state = self.domain_states.get(domain)
        if not state:
            return {}
        return {
            "domain": domain,
            "confidence": round(state.confidence, 4),
            "efficacy": round(state.efficacy, 4),
            "interactions": state.interaction_count,
            "interactions_without_novelty": state.interactions_without_novelty,
            "success_rate": round(state.success_count / max(state.interaction_count, 1), 4),
            "contradiction_rate": round(state.contradiction_count / max(state.interaction_count, 1), 4),
        }

    def get_utility_trend(self, domain: Optional[str] = None, last_n: int = 20) -> List[float]:
        scores = [s for s in self.history if domain is None or s.field == domain]
        return [s.utility for s in scores[-last_n:]]

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_or_create_state(self, domain: str) -> DomainState:
        if domain not in self.domain_states:
            self.domain_states[domain] = DomainState(domain=domain)
        return self.domain_states[domain]

    def _compute_efficacy(self, agent_score: float, human_baseline: float) -> float:
        """
        Sigmoid-normalized efficacy.
        agent == human  →  0.5
        agent > human   →  > 0.5
        agent < human   →  < 0.5
        """
        if human_baseline == 0:
            return agent_score
        ratio = agent_score / human_baseline
        return 1.0 - 1.0 / (1.0 + ratio)

    def _update_confidence(
        self,
        state: DomainState,
        test_pass_rate: float,
        contradiction_penalty: float,
        config: FieldConfig,
    ) -> float:
        """
        EMA confidence update. Contradiction penalty is amplified by field multiplier.
        """
        effective_penalty = contradiction_penalty * config.penalty_multiplier
        penalized_signal = test_pass_rate * (1.0 - effective_penalty)

        if contradiction_penalty > 0:
            state.contradiction_count += 1

        alpha = 0.2  # EMA learning rate
        state.confidence = (1 - alpha) * state.confidence + alpha * penalized_signal
        state.confidence = max(0.0, min(1.0, state.confidence))
        return state.confidence

    def _compute_curiosity(
        self,
        state: DomainState,
        config: FieldConfig,
        novelty: float,
        efficacy: float,
        confidence: float,
    ):
        """
        K_raw = potential_gain × novelty × growth(t, field)

        growth = 1 + α(field) × log(1 + interactions_without_novelty)
            → starts at 1.0, grows logarithmically the longer the agent
              stays in familiar territory, bounded by the 50% cap below.

        K_effective = min(K_raw, (w_e·E + w_c·C) / w_k)
            → curiosity can never exceed 50% of total utility.
            → self-scaling: when E and C are high the cap is loose;
              when the agent is weak the cap tightens automatically.

        Returns (K_raw, K_effective, was_capped).
        """
        # Growth multiplier — logarithmic, field-specific alpha
        alpha = CURIOSITY_ALPHA.get(config.name, 0.07)
        growth = 1.0 + alpha * math.log1p(state.interactions_without_novelty)

        # Base curiosity signal
        potential_gain = max(0.0, state.potential_ceiling - confidence)
        k_raw = potential_gain * novelty * growth

        # 50% cap: K_effective ≤ (w_e·E + w_c·C) / w_k
        # Derived from: w_k·K ≤ 0.5 × (w_e·E + w_c·C + w_k·K)
        # Rearranges to: K ≤ (w_e·E + w_c·C) / w_k
        if config.w_curiosity > 0:
            weighted_ec = config.w_efficacy * efficacy + config.w_confidence * confidence
            cap = weighted_ec / config.w_curiosity
        else:
            cap = 0.0

        k_effective = min(k_raw, cap)
        capped = k_effective < k_raw

        return k_raw, k_effective, capped
