"""
Arbiter Agent — structured contradiction resolution across conflicting outputs.

Runs four checks in order of cost:
    1. Logical     (w=0.30) — does output contradict its own premises?
    2. Mathematical (w=0.40) — are numerical/complexity claims provably wrong?
    3. Cross-session (w=0.20) — contradicts prior verified assertions?
    4. Empirical   (w=0.10) — contradicts external ground truth?

Verdict cases:
    Case 1: A correct, B wrong  → correct B, reinforce A
    Case 2: B correct, A wrong  → correct A, reinforce B
    Case 3: Both wrong          → correct both + curiosity gap bonus
    Case 4: Inconclusive        → flag for external escalation

Evidence chains are shared internally with affected submodels only.
Nothing is disclosed externally except the verified answer (or a minimal
hedge on Case 4).

Arbiter calibration: a sample of verdicts (~2-5%) is independently verified
against domain experts. If correction rate is high, sampling escalates
adaptively (up to 15% ceiling).
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from assertions_store import AssertionsStore, AssertionMatch


class VerdictCase(str, Enum):
    CASE_1 = "case_1"   # A correct, B wrong
    CASE_2 = "case_2"   # B correct, A wrong
    CASE_3 = "case_3"   # both wrong
    CASE_4 = "case_4"   # inconclusive — escalate


@dataclass
class CheckResult:
    """Result of a single arbitration check."""
    check_type: str     # "logical" | "mathematical" | "cross_session" | "empirical"
    converged: bool     # did this check produce a clear answer?
    winner: Optional[str] = None   # "A" | "B" | "neither" | None
    explanation: str = ""
    confidence: float = 0.0


@dataclass
class ArbiterVerdict:
    """Full verdict from the Arbiter Agent."""
    subject: str
    domain: str
    case: VerdictCase
    arbiter_confidence: float     # 0-1: how confident the Arbiter is
    checks_run: List[CheckResult] = field(default_factory=list)

    # Correction signals (internal — never disclosed externally)
    correct_A: bool = False       # should A receive a correction?
    correct_B: bool = False       # should B receive a correction?
    verified_claim: Optional[str] = None   # what the correct answer is
    evidence_summary: str = ""    # internal evidence chain summary

    # Gap bonus (Case 3 only)
    gap_bonus_active: bool = False
    gap_multiplier: float = 1.0

    # External flag
    needs_escalation: bool = False

    # Sampling flag
    selected_for_calibration_sample: bool = False

    @property
    def external_response(self) -> str:
        """
        What the user sees — minimum disclosure.
        Cases 1/2/3: the verified answer.
        Case 4: minimal hedge only.
        """
        if self.case == VerdictCase.CASE_4:
            return "I have limited confidence in this answer. Please verify with a domain expert."
        if self.verified_claim:
            return self.verified_claim
        return ""


# Arbiter confidence check weights (from whitepaper §9.5)
CHECK_WEIGHTS: Dict[str, float] = {
    "logical":      0.30,
    "mathematical": 0.40,
    "cross_session": 0.20,
    "empirical":    0.10,
}

# Minimum confidence to issue a verdict (below = Case 4)
VERDICT_CONFIDENCE_THRESHOLD = 0.85

# Base calibration sampling rate
BASE_SAMPLE_RATE = 0.03   # 3%

# Adaptive sampling thresholds and rates
ADAPTIVE_SAMPLE_RATES = [
    (2.0, 0.10),   # correction_rate > 2× baseline → 10%
    (1.5, 0.08),   # correction_rate > 1.5× baseline → 8%
    (1.0, BASE_SAMPLE_RATE),  # normal
]
SAMPLE_RATE_CEILING = 0.15   # hard cap — never exceeded


@dataclass
class GapRecord:
    """Tracks an active Case 3 knowledge gap."""
    subject: str
    domain: str
    gap_multiplier: float
    opened_at: float
    interactions_since: int = 0
    resolved: bool = False


class ArbiterAgent:
    """
    Resolves contradictions between two outputs A and B on subject S in domain D.

    Usage:
        arbiter = ArbiterAgent(assertions_store)
        verdict = arbiter.arbitrate(
            subject="bubble_sort_complexity",
            domain="software_engineering",
            output_A="Bubble sort is O(n) average case",
            output_B="Bubble sort is O(n^2) average case",
            field_penalty_multiplier=2.0,
        )
    """

    def __init__(
        self,
        assertions_store: AssertionsStore,
        field_penalty_multipliers: Optional[Dict[str, float]] = None,
    ):
        self.store = assertions_store
        self.field_penalty_multipliers = field_penalty_multipliers or {}

        # Calibration tracking
        self.total_verdicts = 0
        self.total_corrections_issued = 0
        self.calibration_samples: List[ArbiterVerdict] = []

        # Gap bonus tracking
        self.active_gaps: Dict[str, GapRecord] = {}  # subject → GapRecord

        # Correction rate baseline (used for adaptive sampling)
        self._baseline_correction_rate: float = 0.3   # initial estimate

    def arbitrate(
        self,
        subject: str,
        domain: str,
        output_A: str,
        output_B: str,
        field_penalty_multiplier: float = 1.0,
        claimed_complexity_A: Optional[str] = None,
        claimed_complexity_B: Optional[str] = None,
    ) -> ArbiterVerdict:
        """
        Run the full arbitration pipeline on two conflicting outputs.
        """
        checks: List[CheckResult] = []

        # ── Step 1: Run checks in order of cost ──────────────────────────────
        logical = self._check_logical(output_A, output_B)
        checks.append(logical)

        math_check = self._check_mathematical(
            output_A, output_B, claimed_complexity_A, claimed_complexity_B
        )
        checks.append(math_check)

        cross = self._check_cross_session(subject, domain, output_A, output_B)
        checks.append(cross)

        empirical = self._check_empirical(subject, domain, output_A, output_B)
        checks.append(empirical)

        # ── Step 2: Compute arbiter confidence ───────────────────────────────
        arbiter_conf = self._compute_confidence(checks)

        # ── Step 3: Determine verdict ─────────────────────────────────────────
        case, winner = self._determine_case(checks, arbiter_conf)

        # ── Step 4: Build verdict ─────────────────────────────────────────────
        verdict = ArbiterVerdict(
            subject=subject,
            domain=domain,
            case=case,
            arbiter_confidence=round(arbiter_conf, 4),
            checks_run=checks,
        )

        if case == VerdictCase.CASE_1:
            verdict.correct_B = True
            verdict.verified_claim = output_A
            verdict.evidence_summary = self._build_evidence_summary(checks, "A")

        elif case == VerdictCase.CASE_2:
            verdict.correct_A = True
            verdict.verified_claim = output_B
            verdict.evidence_summary = self._build_evidence_summary(checks, "B")

        elif case == VerdictCase.CASE_3:
            verdict.correct_A = True
            verdict.correct_B = True
            verdict.evidence_summary = self._build_evidence_summary(checks, "neither")
            gap_multiplier = self._compute_gap_multiplier(domain, field_penalty_multiplier)
            verdict.gap_bonus_active = True
            verdict.gap_multiplier = gap_multiplier
            self._open_gap(subject, domain, gap_multiplier)

        elif case == VerdictCase.CASE_4:
            verdict.needs_escalation = True

        # ── Step 5: Store verified claim in assertions store ──────────────────
        if verdict.verified_claim and case in (VerdictCase.CASE_1, VerdictCase.CASE_2):
            self.store.add(
                subject=subject,
                domain=domain,
                claim=verdict.verified_claim,
                confidence=arbiter_conf,
                source="arbiter",
                evidence_summary=verdict.evidence_summary,
            )

        # ── Step 6: Update calibration tracking ──────────────────────────────
        self.total_verdicts += 1
        if verdict.correct_A or verdict.correct_B:
            self.total_corrections_issued += 1

        verdict.selected_for_calibration_sample = self._should_sample()
        if verdict.selected_for_calibration_sample:
            self.calibration_samples.append(verdict)

        return verdict

    def get_gap_bonus(
        self,
        subject: str,
        k_effective: float,
        k_budget_total: float,
    ) -> float:
        """
        Compute gap curiosity bonus for subject S, with dual cap:
            Constraint 1: K_gap ≤ K_natural_max (per-gap ceiling)
            Constraint 2: collective gap budget ≤ 2/3 of K_budget_total

        Returns the bonus to add to K_effective, or 0 if no active gap.
        """
        gap = self.active_gaps.get(subject)
        if gap is None or gap.resolved:
            return 0.0

        # Per-gap cap: cannot exceed natural max curiosity
        k_natural_max = k_effective   # current natural curiosity is the ceiling
        k_gap_uncapped = k_effective * gap.gap_multiplier

        # Apply per-gap cap
        k_gap = min(k_gap_uncapped, k_natural_max)

        # Apply collective budget cap: all active gaps ≤ 2/3 of total budget
        # Allocate proportionally if multiple gaps exist
        total_gap_demand = sum(
            k_effective * g.gap_multiplier
            for g in self.active_gaps.values()
            if not g.resolved
        )
        budget_ceiling = (2.0 / 3.0) * k_budget_total

        if total_gap_demand > 0:
            this_gap_share = k_gap / total_gap_demand
            k_gap = min(k_gap, budget_ceiling * this_gap_share)

        gap.interactions_since += 1
        return max(0.0, k_gap)

    def check_gap_resolved(
        self,
        subject: str,
        confidence_A: float,
        confidence_B: float,
        c_min: float,
        t_field: int = 10,
    ) -> bool:
        """
        Check if a Case 3 gap has been resolved:
            Both submodels confidence(S) ≥ C_min AND
            no new contradictions for T(field) interactions.
        """
        gap = self.active_gaps.get(subject)
        if gap is None or gap.resolved:
            return True

        if (confidence_A >= c_min and confidence_B >= c_min and
                gap.interactions_since >= t_field):
            gap.resolved = True
            return True
        return False

    def correction_rate(self) -> float:
        """Current fraction of verdicts that issued at least one correction."""
        if self.total_verdicts == 0:
            return 0.0
        return self.total_corrections_issued / self.total_verdicts

    def adaptive_sample_rate(self) -> float:
        """
        Current sampling rate for calibration, based on correction rate
        relative to baseline. Hard ceiling of 15%.
        """
        if self.total_verdicts < 20:
            return BASE_SAMPLE_RATE  # not enough data yet

        rate_ratio = self.correction_rate() / max(self._baseline_correction_rate, 0.01)

        for threshold, sample_rate in sorted(ADAPTIVE_SAMPLE_RATES, reverse=True):
            if rate_ratio >= threshold:
                return min(sample_rate, SAMPLE_RATE_CEILING)

        return BASE_SAMPLE_RATE

    def status(self) -> dict:
        return {
            "total_verdicts": self.total_verdicts,
            "total_corrections": self.total_corrections_issued,
            "correction_rate": round(self.correction_rate(), 3),
            "adaptive_sample_rate": round(self.adaptive_sample_rate(), 3),
            "calibration_samples_collected": len(self.calibration_samples),
            "active_gaps": len([g for g in self.active_gaps.values() if not g.resolved]),
        }

    # ── Check implementations ─────────────────────────────────────────────────

    def _check_logical(self, output_A: str, output_B: str) -> CheckResult:
        """
        Check if either output contradicts its own stated premises.
        Basic heuristics for code generation domain.
        """
        a_self_contradicts = self._self_contradicts(output_A)
        b_self_contradicts = self._self_contradicts(output_B)

        if a_self_contradicts and not b_self_contradicts:
            return CheckResult("logical", True, "B", "A contradicts its own premises", 0.8)
        if b_self_contradicts and not a_self_contradicts:
            return CheckResult("logical", True, "A", "B contradicts its own premises", 0.8)
        if a_self_contradicts and b_self_contradicts:
            return CheckResult("logical", True, "neither", "Both contradict own premises", 0.7)
        return CheckResult("logical", False, None, "No self-contradictions detected", 0.0)

    def _check_mathematical(
        self,
        output_A: str,
        output_B: str,
        complexity_A: Optional[str],
        complexity_B: Optional[str],
    ) -> CheckResult:
        """
        Check if claimed complexity or mathematical results are provably wrong.
        Uses nested loop counting as a proxy for complexity analysis.
        """
        import ast, re

        def extract_code(text):
            m = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
            return m[0].strip() if m else None

        def count_nested_loops(code):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return 0
            max_d = [0]
            def walk(node, d):
                if isinstance(node, (ast.For, ast.While)):
                    d += 1
                    max_d[0] = max(max_d[0], d)
                for child in ast.iter_child_nodes(node):
                    walk(child, d)
            walk(tree, 0)
            return max_d[0]

        def complexity_mismatch(code, claimed):
            if not code or not claimed:
                return False
            claimed_l = claimed.lower()
            loops = count_nested_loops(code)
            if ("o(n)" in claimed_l or "o(1)" in claimed_l) and loops >= 2:
                return True
            if "o(n log n)" in claimed_l and loops >= 3:
                return True
            return False

        code_A = extract_code(output_A)
        code_B = extract_code(output_B)
        a_wrong = complexity_mismatch(code_A, complexity_A)
        b_wrong = complexity_mismatch(code_B, complexity_B)

        if a_wrong and not b_wrong:
            return CheckResult("mathematical", True, "B",
                f"A claims {complexity_A} but code structure contradicts it", 0.85)
        if b_wrong and not a_wrong:
            return CheckResult("mathematical", True, "A",
                f"B claims {complexity_B} but code structure contradicts it", 0.85)
        if a_wrong and b_wrong:
            return CheckResult("mathematical", True, "neither",
                "Both have complexity claim mismatches", 0.75)
        return CheckResult("mathematical", False, None, "No mathematical contradictions", 0.0)

    def _check_cross_session(
        self, subject: str, domain: str, output_A: str, output_B: str
    ) -> CheckResult:
        """
        Check both outputs against the assertions store.
        Prior verified facts take precedence.
        """
        matches = self.store.query(subject, domain)
        if not matches:
            return CheckResult("cross_session", False, None, "No prior assertions on this subject", 0.0)

        # Check each output against best matching stored assertion
        best_match = matches[0]
        stored_claim = best_match.assertion.claim.lower()
        eff_conf = best_match.effective_confidence

        a_contradicts = self.store._claims_contradict(stored_claim, output_A.lower())
        b_contradicts = self.store._claims_contradict(stored_claim, output_B.lower())

        explanation = f"Prior assertion (conf={eff_conf:.2f}): '{best_match.assertion.claim[:80]}'"

        if a_contradicts and not b_contradicts:
            return CheckResult("cross_session", True, "B", explanation, eff_conf)
        if b_contradicts and not a_contradicts:
            return CheckResult("cross_session", True, "A", explanation, eff_conf)
        if a_contradicts and b_contradicts:
            return CheckResult("cross_session", True, "neither",
                f"Both contradict stored assertion. " + explanation, eff_conf * 0.7)
        return CheckResult("cross_session", False, None, "No cross-session contradictions", 0.0)

    def _check_empirical(
        self, subject: str, domain: str, output_A: str, output_B: str
    ) -> CheckResult:
        """
        Empirical check against external ground truth.
        In the MVP this is a stub — the live system calls external APIs.
        Returns no verdict (not converged) to avoid false positives.
        """
        # TODO Phase 1: integrate external knowledge APIs per field
        # e.g. PubMed for medicine, arXiv for CS, SymPy for math
        return CheckResult(
            "empirical", False, None,
            "Empirical check not yet implemented (Phase 1 item)", 0.0
        )

    # ── Confidence and case logic ─────────────────────────────────────────────

    def _compute_confidence(self, checks: List[CheckResult]) -> float:
        """
        conf_arbiter = Σ(w_i × 1[check_i converges]) / Σ(w_i)
        """
        numerator = sum(
            CHECK_WEIGHTS[c.check_type] * (1.0 if c.converged else 0.0)
            for c in checks
        )
        denominator = sum(CHECK_WEIGHTS[c.check_type] for c in checks)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _determine_case(
        self,
        checks: List[CheckResult],
        arbiter_conf: float,
    ) -> Tuple[VerdictCase, Optional[str]]:
        """Determine verdict case from check results."""
        if arbiter_conf < VERDICT_CONFIDENCE_THRESHOLD:
            return VerdictCase.CASE_4, None

        converged = [c for c in checks if c.converged]
        if not converged:
            return VerdictCase.CASE_4, None

        # Majority vote among converged checks (weighted)
        vote_A = sum(CHECK_WEIGHTS[c.check_type] for c in converged if c.winner == "A")
        vote_B = sum(CHECK_WEIGHTS[c.check_type] for c in converged if c.winner == "B")
        vote_N = sum(CHECK_WEIGHTS[c.check_type] for c in converged if c.winner == "neither")

        if vote_N > vote_A and vote_N > vote_B:
            return VerdictCase.CASE_3, "neither"
        if vote_A > vote_B:
            return VerdictCase.CASE_1, "A"
        if vote_B > vote_A:
            return VerdictCase.CASE_2, "B"

        return VerdictCase.CASE_4, None

    # ── Gap bonus helpers ─────────────────────────────────────────────────────

    def _compute_gap_multiplier(self, domain: str, penalty_multiplier: float) -> float:
        """γ(f) = 1 + μ(f) / 10"""
        return 1.0 + penalty_multiplier / 10.0

    def _open_gap(self, subject: str, domain: str, gap_multiplier: float):
        if subject not in self.active_gaps:
            self.active_gaps[subject] = GapRecord(
                subject=subject,
                domain=domain,
                gap_multiplier=gap_multiplier,
                opened_at=time.time(),
            )
        else:
            # Re-open if previously resolved
            self.active_gaps[subject].resolved = False
            self.active_gaps[subject].interactions_since = 0

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _should_sample(self) -> bool:
        return random.random() < self.adaptive_sample_rate()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _self_contradicts(self, output: str) -> bool:
        """
        Heuristic self-contradiction detection.
        Look for patterns like 'X is Y' followed by 'X is not Y'.
        """
        lines = output.lower().split(".")
        claims = {}
        for line in lines:
            line = line.strip()
            if " is " in line:
                parts = line.split(" is ", 1)
                subject_key = parts[0].strip()[-30:]  # last 30 chars as key
                predicate = parts[1].strip()[:50]
                if subject_key in claims:
                    prev = claims[subject_key]
                    # Check for negation
                    if ("not " + prev) in predicate or ("not " + predicate) in prev:
                        return True
                else:
                    claims[subject_key] = predicate
        return False

    def _build_evidence_summary(self, checks: List[CheckResult], winner: str) -> str:
        """
        Build internal evidence chain summary.
        This is INTERNAL only — never disclosed externally.
        """
        parts = [f"Arbiter verdict: {winner} is correct (or neither)"]
        for c in checks:
            if c.converged:
                parts.append(f"  [{c.check_type}] winner={c.winner} conf={c.confidence:.2f}: {c.explanation}")
        return " | ".join(parts)
