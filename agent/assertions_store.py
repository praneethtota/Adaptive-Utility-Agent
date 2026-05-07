"""
Assertions store — persistent cross-session fact store with field-specific
confidence decay.

Decay classes (from whitepaper §9.5):
    Class A — No decay: mathematically/logically proven facts
    Class B — Slow decay (τ = 10yr): mechanical engineering, classical physics
    Class C — Moderate decay (τ = 3yr): medicine, architecture, legal principles
    Class D — Fast decay (τ = 6mo): clinical guidelines, security practices, ML findings
"""

import math
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DecayClass(str, Enum):
    A = "A"  # no decay — proven mathematical/logical facts
    B = "B"  # slow decay — τ = 10 years
    C = "C"  # moderate decay — τ = 3 years
    D = "D"  # fast decay — τ = 6 months


# Half-life in seconds for each decay class
DECAY_TAU_SECONDS: Dict[DecayClass, Optional[float]] = {
    DecayClass.A: None,                    # immortal
    DecayClass.B: 10 * 365.25 * 86400,    # 10 years
    DecayClass.C: 3  * 365.25 * 86400,    # 3 years
    DecayClass.D: 0.5 * 365.25 * 86400,   # 6 months
}

# Default decay class per field
FIELD_DECAY_CLASS: Dict[str, DecayClass] = {
    "surgery":                DecayClass.C,   # anatomy stable; guidelines change
    "aviation":               DecayClass.C,
    "law":                    DecayClass.C,   # common law slow; regulations faster
    "structural_engineering": DecayClass.B,   # physics principles very stable
    "software_engineering":   DecayClass.D,   # best practices change fast
    "stem_research":          DecayClass.D,   # ML findings change fast
    "education":              DecayClass.C,
    "art":                    DecayClass.C,
    "creative_writing":       DecayClass.C,
    "general":                DecayClass.C,
    "mathematics":            DecayClass.A,   # proofs don't expire
    "pure_physics":           DecayClass.A,   # fundamental laws
    "blended":                DecayClass.C,
}

# Override rules — specific subjects that override field default
SUBJECT_DECAY_OVERRIDES: Dict[str, DecayClass] = {
    # Mathematical facts — always Class A
    "time_complexity": DecayClass.A,
    "space_complexity": DecayClass.A,
    "algorithm_correctness": DecayClass.A,
    "mathematical_proof": DecayClass.A,
    "logical_validity": DecayClass.A,
    # Fast-changing topics — always Class D
    "security_vulnerability": DecayClass.D,
    "cve": DecayClass.D,
    "clinical_guideline": DecayClass.D,
    "drug_dosage": DecayClass.D,
    "ml_benchmark": DecayClass.D,
    "api_version": DecayClass.D,
}


@dataclass
class Assertion:
    """A single verified fact stored in the assertions store."""
    subject: str              # what the assertion is about
    domain: str               # field it was verified in
    claim: str                # the actual factual claim
    confidence_at_write: float # C score when this was verified
    decay_class: DecayClass   # how fast it ages
    timestamp: float          # unix timestamp of verification
    source: str = "arbiter"   # "arbiter" | "expert" | "cross_session"
    evidence_summary: str = ""# internal only — never disclosed externally

    def effective_confidence(self, now: Optional[float] = None) -> float:
        """
        C_effective = C_at_write × decay(class, Δt)

        Class A: no decay → C_effective = C_at_write always
        Others: exponential decay with field-specific τ
        """
        if self.decay_class == DecayClass.A:
            return self.confidence_at_write

        tau = DECAY_TAU_SECONDS[self.decay_class]
        if tau is None:
            return self.confidence_at_write

        if now is None:
            now = time.time()

        delta_t = now - self.timestamp
        if delta_t < 0:
            delta_t = 0.0

        decay_factor = math.exp(-delta_t / tau)
        return self.confidence_at_write * decay_factor

    def is_trustworthy(self, threshold: float = 0.5, now: Optional[float] = None) -> bool:
        return self.effective_confidence(now) >= threshold


@dataclass
class AssertionMatch:
    """Returned when a query matches stored assertions."""
    assertion: Assertion
    effective_confidence: float
    is_trustworthy: bool


class AssertionsStore:
    """
    Persistent cross-session store for verified facts.

    Facts are stored with a decay class determined by field and subject type.
    On retrieval, effective confidence is computed based on age.
    Only assertions above a confidence threshold are returned as active evidence.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.assertions: List[Assertion] = []
        self.confidence_threshold = confidence_threshold

    def add(
        self,
        subject: str,
        domain: str,
        claim: str,
        confidence: float,
        source: str = "arbiter",
        evidence_summary: str = "",
        decay_class_override: Optional[DecayClass] = None,
    ) -> Assertion:
        """Store a verified fact with appropriate decay class."""
        decay_class = decay_class_override or self._assign_decay_class(subject, domain)

        assertion = Assertion(
            subject=subject,
            domain=domain,
            claim=claim,
            confidence_at_write=confidence,
            decay_class=decay_class,
            timestamp=time.time(),
            source=source,
            evidence_summary=evidence_summary,
        )
        # Replace any existing assertion for same subject+domain
        self.assertions = [
            a for a in self.assertions
            if not (a.subject == subject and a.domain == domain and a.claim[:50] == claim[:50])
        ]
        self.assertions.append(assertion)
        return assertion

    def query(
        self,
        subject: str,
        domain: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[AssertionMatch]:
        """
        Retrieve active assertions matching subject (and optionally domain).
        Returns only assertions whose effective confidence exceeds threshold.
        """
        threshold = min_confidence or self.confidence_threshold
        now = time.time()
        results = []

        for assertion in self.assertions:
            # Subject match (case-insensitive partial)
            if subject.lower() not in assertion.subject.lower():
                continue
            # Domain filter
            if domain and assertion.domain != domain:
                continue
            eff_conf = assertion.effective_confidence(now)
            if eff_conf >= threshold:
                results.append(AssertionMatch(
                    assertion=assertion,
                    effective_confidence=round(eff_conf, 4),
                    is_trustworthy=True,
                ))

        # Sort by effective confidence descending
        results.sort(key=lambda m: m.effective_confidence, reverse=True)
        return results

    def query_contradictions(
        self,
        subject: str,
        new_claim: str,
        domain: Optional[str] = None,
    ) -> List[Tuple[AssertionMatch, str]]:
        """
        Check if new_claim contradicts any stored assertion on subject.
        Returns list of (match, contradiction_description).
        """
        matches = self.query(subject, domain)
        contradictions = []

        for match in matches:
            stored_claim = match.assertion.claim.lower()
            new_lower = new_claim.lower()
            # Simple contradiction heuristics — extend with domain-specific logic
            if self._claims_contradict(stored_claim, new_lower):
                contradictions.append((
                    match,
                    f"New claim conflicts with stored assertion "
                    f"(confidence {match.effective_confidence:.2f}): "
                    f"'{match.assertion.claim[:100]}'"
                ))

        return contradictions

    def prune_stale(self, min_effective_confidence: float = 0.05) -> int:
        """Remove assertions whose effective confidence has fallen below floor."""
        now = time.time()
        before = len(self.assertions)
        self.assertions = [
            a for a in self.assertions
            if a.effective_confidence(now) >= min_effective_confidence
        ]
        return before - len(self.assertions)

    def summary(self) -> dict:
        now = time.time()
        by_class = {c.value: 0 for c in DecayClass}
        trustworthy = 0
        for a in self.assertions:
            by_class[a.decay_class.value] += 1
            if a.is_trustworthy(self.confidence_threshold, now):
                trustworthy += 1
        return {
            "total": len(self.assertions),
            "trustworthy": trustworthy,
            "by_decay_class": by_class,
        }

    def to_json(self) -> str:
        data = []
        for a in self.assertions:
            d = asdict(a)
            d["decay_class"] = a.decay_class.value
            data.append(d)
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str, **kwargs) -> "AssertionsStore":
        store = cls(**kwargs)
        for d in json.loads(json_str):
            d["decay_class"] = DecayClass(d["decay_class"])
            store.assertions.append(Assertion(**d))
        return store

    # ── Private ───────────────────────────────────────────────────────────────

    def _assign_decay_class(self, subject: str, domain: str) -> DecayClass:
        """Assign decay class — subject overrides take priority over field default."""
        subject_lower = subject.lower()
        for key, cls in SUBJECT_DECAY_OVERRIDES.items():
            if key in subject_lower:
                return cls
        return FIELD_DECAY_CLASS.get(domain, DecayClass.C)

    def _claims_contradict(self, stored: str, new: str) -> bool:
        """
        Heuristic contradiction detection between two claim strings.
        Extend this with domain-specific logic as needed.
        """
        # Negation patterns
        negation_pairs = [
            ("is ", "is not "), ("can ", "cannot "), ("does ", "does not "),
            ("always ", "never "), ("true", "false"), ("correct", "incorrect"),
            ("valid", "invalid"), ("o(n)", "o(n^2)"), ("o(1)", "o(n)"),
            ("o(n log n)", "o(n^2)"), ("o(n)", "o(n log n)"),
        ]
        for pos, neg in negation_pairs:
            if pos in stored and neg in new:
                return True
            if neg in stored and pos in new:
                return True
        return False
