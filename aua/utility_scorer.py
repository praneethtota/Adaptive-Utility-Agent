"""
Utility scorer: computes Efficacy, Confidence, Curiosity, and final U.

    U = w_e·E_ema  +  w_c·C  +  w_k·K_effective

Changes from v0.1 (simulation):
    - Efficacy uses EMA accumulation (domain-level state, not per-interaction)
    - Gap bonus dual cap: per-gap ≤ K_natural_max; collective ≤ 2/3 of K_budget
    - Dynamic difficulty routing: harder problems routed as domain confidence rises
    - DomainState tracks efficacy_ema separately from per-interaction raw efficacy
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from aua.config import FieldConfig


CURIOSITY_ALPHA: Dict[str, float] = {
    "surgery":                0.00,
    "aviation":               0.00,
    "law":                    0.02,
    "structural_engineering": 0.03,
    "software_engineering":   0.08,
    "stem_research":          0.12,
    "education":              0.08,
    "art":                    0.15,
    "creative_writing":       0.15,
    "general":                0.07,
    "blended":                0.07,
}

DIFFICULTY_THRESHOLDS = {
    "hard":   0.85,
    "medium": 0.70,
}


@dataclass
class TaskScore:
    task_id: str
    field: str
    efficacy: float
    efficacy_ema: float
    confidence: float
    curiosity_raw: float
    curiosity_effective: float
    gap_bonus: float
    utility: float
    timestamp: str
    below_minimum: bool
    curiosity_capped: bool = False
    recommended_difficulty: str = "easy"
    notes: str = ""


@dataclass
class DomainState:
    domain: str
    confidence: float = 0.5
    efficacy_ema: float = 0.5
    efficacy_raw_last: float = 0.5
    interaction_count: int = 0
    contradiction_count: int = 0
    success_count: int = 0
    potential_ceiling: float = 0.9
    interactions_without_novelty: int = 0


class UtilityScorer:
    """
    Computes and tracks U = w_e·E_ema + w_c·C + w_k·K_effective.
    Efficacy uses EMA; curiosity gap bonus requires an injected ArbiterAgent.
    """

    EFFICACY_ALPHA = 0.2
    CONFIDENCE_ALPHA = 0.2

    def __init__(self, arbiter=None):
        self.domain_states: Dict[str, DomainState] = {}
        self.history: List[TaskScore] = []
        self.arbiter = arbiter

    def score(
        self,
        task_id: str,
        field_config: FieldConfig,
        test_pass_rate: float,
        human_baseline_score: float,
        contradiction_penalty: float,
        problem_novelty: float,
        active_gap_subject: Optional[str] = None,
    ) -> TaskScore:
        domain = field_config.name
        state = self._get_or_create_state(domain)

        efficacy_raw = self._compute_efficacy_raw(test_pass_rate, human_baseline_score)
        efficacy_ema = self._update_efficacy_ema(state, efficacy_raw)
        confidence = self._update_confidence(state, test_pass_rate, contradiction_penalty, field_config)
        curiosity_raw, curiosity_base, capped = self._compute_curiosity(
            state, field_config, problem_novelty, efficacy_ema, confidence
        )

        gap_bonus = 0.0
        if self.arbiter and active_gap_subject:
            gap_bonus = self.arbiter.get_gap_bonus(
                subject=active_gap_subject,
                k_effective=curiosity_base,
                k_budget_total=curiosity_base,
            )
        curiosity_effective = curiosity_base + gap_bonus

        if problem_novelty >= 0.6:
            state.interactions_without_novelty = 0
        else:
            state.interactions_without_novelty += 1

        utility = (
            field_config.w_efficacy * efficacy_ema +
            field_config.w_confidence * confidence +
            field_config.w_curiosity * curiosity_effective
        )

        below_min = confidence < field_config.c_min or efficacy_ema < field_config.e_min
        notes = []
        if below_min:
            notes.append(
                f"BELOW MINIMUM: C={confidence:.2f}(min={field_config.c_min}), "
                f"E_ema={efficacy_ema:.2f}(min={field_config.e_min}). Abstain or escalate."
            )
        if capped:
            notes.append(f"Curiosity capped: raw={curiosity_raw:.3f} → base={curiosity_base:.3f}")
        if gap_bonus > 0:
            notes.append(f"Gap bonus: +{gap_bonus:.3f} on '{active_gap_subject}'")

        state.interaction_count += 1
        if test_pass_rate >= 0.8:
            state.success_count += 1

        task_score = TaskScore(
            task_id=task_id, field=domain,
            efficacy=round(efficacy_raw, 4), efficacy_ema=round(efficacy_ema, 4),
            confidence=round(confidence, 4),
            curiosity_raw=round(curiosity_raw, 4), curiosity_effective=round(curiosity_effective, 4),
            gap_bonus=round(gap_bonus, 4), utility=round(utility, 4),
            timestamp=datetime.utcnow().isoformat(),
            below_minimum=below_min, curiosity_capped=capped,
            recommended_difficulty=self._recommended_difficulty(confidence),
            notes=" | ".join(notes),
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
            "efficacy_ema": round(state.efficacy_ema, 4),
            "interactions": state.interaction_count,
            "interactions_without_novelty": state.interactions_without_novelty,
            "success_rate": round(state.success_count / max(state.interaction_count, 1), 4),
            "contradiction_rate": round(state.contradiction_count / max(state.interaction_count, 1), 4),
            "recommended_difficulty": self._recommended_difficulty(state.confidence),
        }

    def get_utility_trend(self, domain: Optional[str] = None, last_n: int = 20) -> List[float]:
        scores = [s for s in self.history if domain is None or s.field == domain]
        return [s.utility for s in scores[-last_n:]]

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_or_create_state(self, domain: str) -> DomainState:
        if domain not in self.domain_states:
            self.domain_states[domain] = DomainState(domain=domain)
        return self.domain_states[domain]

    def _compute_efficacy_raw(self, agent_score: float, human_baseline: float) -> float:
        if human_baseline == 0:
            return agent_score
        ratio = agent_score / human_baseline
        return 1.0 - 1.0 / (1.0 + ratio)

    def _update_efficacy_ema(self, state: DomainState, efficacy_raw: float) -> float:
        state.efficacy_raw_last = efficacy_raw
        state.efficacy_ema = (
            (1 - self.EFFICACY_ALPHA) * state.efficacy_ema +
            self.EFFICACY_ALPHA * efficacy_raw
        )
        state.efficacy_ema = max(0.0, min(1.0, state.efficacy_ema))
        return state.efficacy_ema

    def _update_confidence(
        self, state: DomainState, test_pass_rate: float,
        contradiction_penalty: float, config: FieldConfig,
    ) -> float:
        effective_penalty = contradiction_penalty * config.penalty_multiplier
        penalized_signal = test_pass_rate * (1.0 - effective_penalty)
        if contradiction_penalty > 0:
            state.contradiction_count += 1
        state.confidence = (
            (1 - self.CONFIDENCE_ALPHA) * state.confidence +
            self.CONFIDENCE_ALPHA * penalized_signal
        )
        state.confidence = max(0.0, min(1.0, state.confidence))
        return state.confidence

    def _compute_curiosity(
        self, state: DomainState, config: FieldConfig,
        novelty: float, efficacy: float, confidence: float,
    ):
        alpha = CURIOSITY_ALPHA.get(config.name, 0.07)
        growth = 1.0 + alpha * math.log1p(state.interactions_without_novelty)
        potential_gain = max(0.0, state.potential_ceiling - confidence)
        k_raw = potential_gain * novelty * growth
        if config.w_curiosity > 0:
            cap = (config.w_efficacy * efficacy + config.w_confidence * confidence) / config.w_curiosity
        else:
            cap = 0.0
        k_base = min(k_raw, cap)
        return k_raw, k_base, k_base < k_raw

    def _recommended_difficulty(self, confidence: float) -> str:
        if confidence >= DIFFICULTY_THRESHOLDS["hard"]:
            return "hard"
        if confidence >= DIFFICULTY_THRESHOLDS["medium"]:
            return "medium"
        return "easy"
