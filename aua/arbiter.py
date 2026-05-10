"""
Arbiter Agent — structured contradiction resolution across conflicting outputs.

Runs four checks in order of cost:
    1. Logical      (w=0.30) — does output contradict its own premises?
    2. Mathematical (w=0.40) — are numerical/complexity claims provably wrong?
    3. Cross-session (w=0.20) — contradicts prior verified assertions?
    4. Empirical    (w=0.10) — contradicts external ground truth?

Verdict cases:
    Case 1: A correct, B wrong  → correct B, reinforce A
    Case 2: B correct, A wrong  → correct A, reinforce B
    Case 3: Both wrong          → correct both + curiosity gap bonus
    Case 4: Inconclusive        → flag for external escalation
"""

import ast
import math
import re
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from aua.assertions_store import AssertionsStore, AssertionMatch


class VerdictCase(str, Enum):
    CASE_1 = "case_1"
    CASE_2 = "case_2"
    CASE_3 = "case_3"
    CASE_4 = "case_4"


@dataclass
class CheckResult:
    check_type: str
    converged: bool
    winner: Optional[str] = None   # "A" | "B" | "neither" | None
    explanation: str = ""
    confidence: float = 0.0


@dataclass
class ArbiterVerdict:
    subject: str
    domain: str
    case: VerdictCase
    arbiter_confidence: float
    checks_run: List[CheckResult] = field(default_factory=list)
    correct_A: bool = False
    correct_B: bool = False
    verified_claim: Optional[str] = None
    evidence_summary: str = ""
    gap_bonus_active: bool = False
    gap_multiplier: float = 1.0
    needs_escalation: bool = False
    selected_for_calibration_sample: bool = False

    @property
    def external_response(self) -> str:
        if self.case == VerdictCase.CASE_4:
            return "I have limited confidence in this answer. Please verify with a domain expert."
        return self.verified_claim or ""


CHECK_WEIGHTS: Dict[str, float] = {
    "logical":       0.30,
    "mathematical":  0.40,
    "cross_session": 0.20,
    "empirical":     0.10,
}

VERDICT_CONFIDENCE_THRESHOLD = 0.85
BASE_SAMPLE_RATE = 0.03
SAMPLE_RATE_CEILING = 0.15


@dataclass
class GapRecord:
    subject: str
    domain: str
    gap_multiplier: float
    opened_at: float
    interactions_since: int = 0
    resolved: bool = False


class ArbiterAgent:
    """
    Resolves contradictions between two outputs A and B.

    Usage:
        store = AssertionsStore()
        arbiter = ArbiterAgent(store)
        verdict = arbiter.arbitrate(
            subject="bubble_sort_complexity",
            domain="software_engineering",
            output_A="Bubble sort is O(n) average case",
            output_B="Bubble sort is O(n²) average case",
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
        self.total_verdicts = 0
        self.total_corrections_issued = 0
        self.calibration_samples: List[ArbiterVerdict] = []
        self.active_gaps: Dict[str, GapRecord] = {}
        self._baseline_correction_rate: float = 0.3

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
        checks: List[CheckResult] = []
        checks.append(self._check_logical(output_A, output_B))
        checks.append(self._check_mathematical(
            output_A, output_B, claimed_complexity_A, claimed_complexity_B
        ))
        checks.append(self._check_cross_session(subject, domain, output_A, output_B))
        checks.append(self._check_empirical(subject, domain, output_A, output_B))

        arbiter_conf = self._compute_confidence(checks)
        case, winner = self._determine_case(checks, arbiter_conf)

        verdict = ArbiterVerdict(
            subject=subject, domain=domain, case=case,
            arbiter_confidence=round(arbiter_conf, 4), checks_run=checks,
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
            verdict.correct_A = verdict.correct_B = True
            verdict.evidence_summary = self._build_evidence_summary(checks, "neither")
            gm = self._compute_gap_multiplier(domain, field_penalty_multiplier)
            verdict.gap_bonus_active = True
            verdict.gap_multiplier = gm
            self._open_gap(subject, domain, gm)
        elif case == VerdictCase.CASE_4:
            verdict.needs_escalation = True

        if verdict.verified_claim and case in (VerdictCase.CASE_1, VerdictCase.CASE_2):
            self.store.add(
                subject=subject, domain=domain, claim=verdict.verified_claim,
                confidence=arbiter_conf, source="arbiter",
                evidence_summary=verdict.evidence_summary,
            )

        self.total_verdicts += 1
        if verdict.correct_A or verdict.correct_B:
            self.total_corrections_issued += 1
        verdict.selected_for_calibration_sample = self._should_sample()
        if verdict.selected_for_calibration_sample:
            self.calibration_samples.append(verdict)

        return verdict

    def get_gap_bonus(self, subject: str, k_effective: float, k_budget_total: float) -> float:
        gap = self.active_gaps.get(subject)
        if gap is None or gap.resolved:
            return 0.0
        k_gap = min(k_effective * gap.gap_multiplier, k_effective)
        total_demand = sum(
            k_effective * g.gap_multiplier
            for g in self.active_gaps.values() if not g.resolved
        )
        budget_ceiling = (2.0 / 3.0) * k_budget_total
        if total_demand > 0:
            k_gap = min(k_gap, budget_ceiling * (k_gap / total_demand))
        gap.interactions_since += 1
        return max(0.0, k_gap)

    def check_gap_resolved(
        self, subject: str, confidence_A: float, confidence_B: float,
        c_min: float, t_field: int = 10,
    ) -> bool:
        gap = self.active_gaps.get(subject)
        if gap is None or gap.resolved:
            return True
        if confidence_A >= c_min and confidence_B >= c_min and gap.interactions_since >= t_field:
            gap.resolved = True
            return True
        return False

    def correction_rate(self) -> float:
        if self.total_verdicts == 0:
            return 0.0
        return self.total_corrections_issued / self.total_verdicts

    def adaptive_sample_rate(self) -> float:
        if self.total_verdicts < 20:
            return BASE_SAMPLE_RATE
        rate_ratio = self.correction_rate() / max(self._baseline_correction_rate, 0.01)
        for threshold, rate in sorted([(2.0, 0.10), (1.5, 0.08), (1.0, BASE_SAMPLE_RATE)], reverse=True):
            if rate_ratio >= threshold:
                return min(rate, SAMPLE_RATE_CEILING)
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

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_logical(self, output_A: str, output_B: str) -> CheckResult:
        a_bad = self._self_contradicts(output_A)
        b_bad = self._self_contradicts(output_B)
        if a_bad and not b_bad:
            return CheckResult("logical", True, "B", "A contradicts its own premises", 0.8)
        if b_bad and not a_bad:
            return CheckResult("logical", True, "A", "B contradicts its own premises", 0.8)
        if a_bad and b_bad:
            return CheckResult("logical", True, "neither", "Both contradict own premises", 0.7)
        return CheckResult("logical", False, None, "No self-contradictions detected", 0.0)

    def _check_mathematical(
        self, output_A: str, output_B: str,
        complexity_A: Optional[str], complexity_B: Optional[str],
    ) -> CheckResult:
        def extract_code(text):
            m = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
            return m[0].strip() if m else None

        def count_loops(code):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                return 0
            max_d = [0]
            def walk(node, d):
                if isinstance(node, (ast.For, ast.While)):
                    d += 1; max_d[0] = max(max_d[0], d)
                for child in ast.iter_child_nodes(node):
                    walk(child, d)
            walk(tree, 0)
            return max_d[0]

        def mismatch(code, claimed):
            if not code or not claimed:
                return False
            cl = claimed.lower()
            loops = count_loops(code)
            if ("o(n)" in cl or "o(1)" in cl) and loops >= 2:
                return True
            if "o(n log n)" in cl and loops >= 3:
                return True
            return False

        a_wrong = mismatch(extract_code(output_A), complexity_A)
        b_wrong = mismatch(extract_code(output_B), complexity_B)

        if a_wrong and not b_wrong:
            return CheckResult("mathematical", True, "B", f"A claims {complexity_A} but code contradicts it", 0.85)
        if b_wrong and not a_wrong:
            return CheckResult("mathematical", True, "A", f"B claims {complexity_B} but code contradicts it", 0.85)
        if a_wrong and b_wrong:
            return CheckResult("mathematical", True, "neither", "Both have complexity mismatches", 0.75)
        return CheckResult("mathematical", False, None, "No mathematical contradictions", 0.0)

    def _check_cross_session(self, subject: str, domain: str, output_A: str, output_B: str) -> CheckResult:
        matches = self.store.query(subject, domain)
        if not matches:
            return CheckResult("cross_session", False, None, "No prior assertions on this subject", 0.0)
        best = matches[0]
        stored = best.assertion.claim.lower()
        eff = best.effective_confidence
        a_con = self.store._claims_contradict(stored, output_A.lower())
        b_con = self.store._claims_contradict(stored, output_B.lower())
        note = f"Prior assertion (conf={eff:.2f}): '{best.assertion.claim[:80]}'"
        if a_con and not b_con:
            return CheckResult("cross_session", True, "B", note, eff)
        if b_con and not a_con:
            return CheckResult("cross_session", True, "A", note, eff)
        if a_con and b_con:
            return CheckResult("cross_session", True, "neither", "Both contradict stored assertion. " + note, eff * 0.7)
        return CheckResult("cross_session", False, None, "No cross-session contradictions", 0.0)

    def _check_empirical(self, subject: str, domain: str, output_A: str, output_B: str) -> CheckResult:
        # Stub — Phase 1 item: integrate external APIs (PubMed, arXiv, SymPy)
        return CheckResult("empirical", False, None, "Empirical check not yet implemented", 0.0)

    # ── Confidence and case ───────────────────────────────────────────────────

    def _compute_confidence(self, checks: List[CheckResult]) -> float:
        num = sum(CHECK_WEIGHTS[c.check_type] * (1.0 if c.converged else 0.0) for c in checks)
        den = sum(CHECK_WEIGHTS[c.check_type] for c in checks)
        return num / den if den > 0 else 0.0

    def _determine_case(self, checks: List[CheckResult], conf: float) -> Tuple[VerdictCase, Optional[str]]:
        if conf < VERDICT_CONFIDENCE_THRESHOLD:
            return VerdictCase.CASE_4, None
        converged = [c for c in checks if c.converged]
        if not converged:
            return VerdictCase.CASE_4, None
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

    def _compute_gap_multiplier(self, domain: str, penalty_multiplier: float) -> float:
        return 1.0 + penalty_multiplier / 10.0

    def _open_gap(self, subject: str, domain: str, gap_multiplier: float):
        if subject not in self.active_gaps:
            self.active_gaps[subject] = GapRecord(
                subject=subject, domain=domain,
                gap_multiplier=gap_multiplier, opened_at=time.time(),
            )
        else:
            self.active_gaps[subject].resolved = False
            self.active_gaps[subject].interactions_since = 0

    def _should_sample(self) -> bool:
        return random.random() < self.adaptive_sample_rate()

    def _self_contradicts(self, output: str) -> bool:
        lines = output.lower().split(".")
        claims: Dict[str, str] = {}
        for line in lines:
            line = line.strip()
            if " is " in line:
                parts = line.split(" is ", 1)
                key = parts[0].strip()[-30:]
                predicate = parts[1].strip()[:50]
                if key in claims:
                    prev = claims[key]
                    if ("not " + prev) in predicate or ("not " + predicate) in prev:
                        return True
                else:
                    claims[key] = predicate
        return False

    def _build_evidence_summary(self, checks: List[CheckResult], winner: str) -> str:
        parts = [f"Arbiter verdict: {winner} is correct (or neither)"]
        for c in checks:
            if c.converged:
                parts.append(f"  [{c.check_type}] winner={c.winner} conf={c.confidence:.2f}: {c.explanation}")
        return " | ".join(parts)
