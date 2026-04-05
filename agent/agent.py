"""
UtilityAgent v0.4 — wires all components together.

Architecture (from whitepaper v0.4):
    - Field classifier with robustness mechanisms
    - Assertions store with field-specific decay (Class A-D)
    - Contradiction detector (logical, mathematical, cross-session)
    - Arbiter Agent (4 checks, 4 verdict cases, gap bonus, adaptive sampling)
    - Trust manager (credential bootstrapping, tit-for-tat, escalation gating)
    - Utility scorer (EMA efficacy, gap bonus dual cap, difficulty routing)
    - Personality manager (3-layer stability safeguards)

Not implemented (requires distributed infrastructure):
    - Distributed model graph (Phase 6)
    - Blue-green deployment (Phase 6)
    - External escalation live API calls (Phase 4 stub only)
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from config import FIELD_CONFIGS, get_effective_config
from field_classifier import FieldClassifier
from contradiction_detector import ContradictionDetector
from utility_scorer import UtilityScorer, TaskScore
from personality_manager import PersonalityManager
from assertions_store import AssertionsStore
from arbiter import ArbiterAgent, ArbiterVerdict, VerdictCase
from trust_manager import TrustManager, Credential


@dataclass
class AgentResponse:
    task_id: str
    field: str
    field_distribution: Dict[str, float]
    answer: str
    utility_score: TaskScore
    arbiter_verdict: Optional[ArbiterVerdict]
    personality_state: dict
    domain_summary: dict
    active_corrections: List[str]
    should_abstain: bool
    recommended_difficulty: str
    notes: str = ""


class UtilityAgent:
    """
    A wrapper around a frontier language model governed by the utility function
    U = w_e·E + w_c·C + w_k·K, with field-specific weights and bounds derived
    from societal licensing standards.

    The utility function is the active loss-weighting mechanism for calibration,
    not a monitoring metric. Contradiction corrections are weighted by
    field_penalty_multiplier in DPO training.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
        enable_arbiter: bool = True,
    ):
        self.api_key = api_key
        self.model = model

        # Core components
        self.field_classifier = FieldClassifier()
        self.assertions_store = AssertionsStore(confidence_threshold=0.5)
        self.arbiter = ArbiterAgent(
            assertions_store=self.assertions_store,
        ) if enable_arbiter else None
        self.trust_manager = TrustManager()
        self.utility_scorer = UtilityScorer(arbiter=self.arbiter)
        self.contradiction_detector = ContradictionDetector()
        self.personality_manager = PersonalityManager()

        # Session state
        self.active_corrections: List[str] = []   # injected into system prompt
        self.interaction_count: int = 0
        self.calibration_dpo_pairs: List[dict] = []

        # Personality evolution: run every N interactions
        self.personality_evolution_interval = 3

    def run(
        self,
        task_id: str,
        problem: str,
        solution: str,
        human_baseline_score: float = 0.7,
        claimed_complexity: Optional[str] = None,
        problem_novelty: float = 0.5,
        test_pass_rate: float = 0.8,
        competing_solution: Optional[str] = None,  # for Arbiter
        entity_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Score an interaction and update all internal state.

        In the full live system, the agent calls the LLM here.
        In the MVP/simulation, the solution is passed in directly.
        """
        self.interaction_count += 1

        # ── 1. Field classification ───────────────────────────────────────────
        field_dist = self.field_classifier.classify(problem)
        field_config = get_effective_config(field_dist)
        primary_field = max(field_dist, key=field_dist.get)

        # ── 2. Contradiction detection ────────────────────────────────────────
        contradiction_result = self.contradiction_detector.check(
            problem=problem,
            solution=solution,
            claimed_complexity=claimed_complexity,
        )

        # ── 3. Arbiter (if competing solution provided) ───────────────────────
        arbiter_verdict: Optional[ArbiterVerdict] = None
        active_gap_subject: Optional[str] = None

        if self.arbiter and competing_solution:
            arbiter_verdict = self.arbiter.arbitrate(
                subject=task_id,
                domain=primary_field,
                output_A=solution,
                output_B=competing_solution,
                field_penalty_multiplier=field_config.penalty_multiplier,
                claimed_complexity_A=claimed_complexity,
            )
            # Case 3 → gap bonus active
            if arbiter_verdict.case == VerdictCase.CASE_3:
                active_gap_subject = task_id

            # Build active correction from Arbiter verdict (internal)
            if arbiter_verdict.correct_A:
                self._add_correction(
                    f"[{primary_field}] Arbiter correction on '{task_id}': "
                    f"prior output contained an error. "
                    f"Use verified claim: {arbiter_verdict.verified_claim or 'see evidence'}"
                )

        # ── 4. Utility scoring ────────────────────────────────────────────────
        task_score = self.utility_scorer.score(
            task_id=task_id,
            field_config=field_config,
            test_pass_rate=test_pass_rate,
            human_baseline_score=human_baseline_score,
            contradiction_penalty=contradiction_result.confidence_penalty,
            problem_novelty=problem_novelty,
            active_gap_subject=active_gap_subject,
        )

        # ── 5. DPO pair generation (for calibration pipeline) ─────────────────
        if contradiction_result.contradictions:
            self._record_dpo_pair(
                task_id=task_id,
                field=primary_field,
                preferred=None,   # correct version not available yet
                rejected=solution,
                weight=field_config.penalty_multiplier,
                reason=[c.description for c in contradiction_result.contradictions],
            )
            # Add session-level behavioral correction
            for c in contradiction_result.contradictions:
                self._add_correction(
                    f"[{primary_field}] Detected {c.type} contradiction: {c.description}"
                )

        # ── 6. Abstain check ──────────────────────────────────────────────────
        should_abstain = task_score.below_minimum

        # ── 7. Personality evolution ──────────────────────────────────────────
        if self.interaction_count % self.personality_evolution_interval == 0:
            utility_trend = self.utility_scorer.get_utility_trend(primary_field)
            domain_summary = self.utility_scorer.get_domain_summary(primary_field)
            contradiction_rate = domain_summary.get("contradiction_rate", 0.0)
            self.personality_manager.evolve(
                utility_trend=utility_trend,
                contradiction_rate=contradiction_rate,
                field=primary_field,
            )

        # ── 8. Build response ─────────────────────────────────────────────────
        return AgentResponse(
            task_id=task_id,
            field=primary_field,
            field_distribution=field_dist,
            answer=solution,
            utility_score=task_score,
            arbiter_verdict=arbiter_verdict,
            personality_state=self.personality_manager.get_state(),
            domain_summary=self.utility_scorer.get_domain_summary(primary_field),
            active_corrections=list(self.active_corrections),
            should_abstain=should_abstain,
            recommended_difficulty=task_score.recommended_difficulty,
            notes=task_score.notes,
        )

    def get_system_prompt(self, field: str) -> str:
        """
        Build the system prompt for the LLM with:
            - Field context and confidence bounds
            - Active corrections from this session
            - Current personality trait weights
        """
        config = FIELD_CONFIGS.get(field, FIELD_CONFIGS["general"])
        personality = self.personality_manager.get_active_weights(field)
        corrections_block = ""
        if self.active_corrections:
            corrections_str = "\n".join(f"  - {c}" for c in self.active_corrections[-10:])
            corrections_block = f"\nACTIVE CORRECTIONS (verified):\n{corrections_str}\n"

        return f"""You are an AI assistant operating in the {field} domain.

Minimum confidence standard for this field: {config.c_min:.0%}
If your confidence is below this threshold, you must abstain and recommend
consultation with a qualified professional.
{corrections_block}
Personality calibration:
  Curiosity: {personality.get('curiosity', 0.5):.2f}
  Caution: {personality.get('caution', 0.5):.2f}
  Analytical rigor: {personality.get('analytical_rigor', 0.5):.2f}
  Assertiveness: {personality.get('assertiveness', 0.5):.2f}

Always: state your reasoning, acknowledge uncertainty, and never fabricate facts."""

    def status(self) -> dict:
        """Full system status for monitoring."""
        return {
            "interactions": self.interaction_count,
            "active_corrections": len(self.active_corrections),
            "dpo_pairs_accumulated": len(self.calibration_dpo_pairs),
            "assertions_store": self.assertions_store.summary(),
            "arbiter": self.arbiter.status() if self.arbiter else None,
            "personality": self.personality_manager.get_state(),
            "domain_summaries": {
                domain: self.utility_scorer.get_domain_summary(domain)
                for domain in self.utility_scorer.domain_states
            },
        }

    def export_dpo_pairs(self, min_weight: float = 1.0) -> List[dict]:
        """Return accumulated DPO pairs for calibration pipeline."""
        return [p for p in self.calibration_dpo_pairs if p["weight"] >= min_weight]

    # ── Private ───────────────────────────────────────────────────────────────

    def _add_correction(self, correction: str):
        """Add to active session corrections (injected into system prompt)."""
        if correction not in self.active_corrections:
            self.active_corrections.append(correction)
            # Keep last 20 corrections in working memory
            if len(self.active_corrections) > 20:
                self.active_corrections = self.active_corrections[-20:]

    def _record_dpo_pair(
        self,
        task_id: str,
        field: str,
        preferred: Optional[str],
        rejected: str,
        weight: float,
        reason: List[str],
    ):
        self.calibration_dpo_pairs.append({
            "task_id": task_id,
            "field": field,
            "preferred": preferred,
            "rejected": rejected,
            "weight": weight,
            "reason": reason,
        })
