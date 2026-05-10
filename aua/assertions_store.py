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
    A = "A"
    B = "B"
    C = "C"
    D = "D"


DECAY_TAU_SECONDS: Dict[DecayClass, Optional[float]] = {
    DecayClass.A: None,
    DecayClass.B: 10 * 365.25 * 86400,
    DecayClass.C: 3  * 365.25 * 86400,
    DecayClass.D: 0.5 * 365.25 * 86400,
}

FIELD_DECAY_CLASS: Dict[str, DecayClass] = {
    "surgery":                DecayClass.C,
    "aviation":               DecayClass.C,
    "law":                    DecayClass.C,
    "structural_engineering": DecayClass.B,
    "software_engineering":   DecayClass.D,
    "stem_research":          DecayClass.D,
    "education":              DecayClass.C,
    "art":                    DecayClass.C,
    "creative_writing":       DecayClass.C,
    "general":                DecayClass.C,
    "mathematics":            DecayClass.A,
    "pure_physics":           DecayClass.A,
    "blended":                DecayClass.C,
}

SUBJECT_DECAY_OVERRIDES: Dict[str, DecayClass] = {
    "time_complexity":        DecayClass.A,
    "space_complexity":       DecayClass.A,
    "algorithm_correctness":  DecayClass.A,
    "mathematical_proof":     DecayClass.A,
    "logical_validity":       DecayClass.A,
    "security_vulnerability": DecayClass.D,
    "cve":                    DecayClass.D,
    "clinical_guideline":     DecayClass.D,
    "drug_dosage":            DecayClass.D,
    "ml_benchmark":           DecayClass.D,
    "api_version":            DecayClass.D,
}


@dataclass
class Assertion:
    subject: str
    domain: str
    claim: str
    confidence_at_write: float
    decay_class: DecayClass
    timestamp: float
    source: str = "arbiter"
    evidence_summary: str = ""

    def effective_confidence(self, now: Optional[float] = None) -> float:
        if self.decay_class == DecayClass.A:
            return self.confidence_at_write
        tau = DECAY_TAU_SECONDS[self.decay_class]
        if tau is None:
            return self.confidence_at_write
        delta_t = max(0.0, (now or time.time()) - self.timestamp)
        return self.confidence_at_write * math.exp(-delta_t / tau)

    def is_trustworthy(self, threshold: float = 0.5, now: Optional[float] = None) -> bool:
        return self.effective_confidence(now) >= threshold


@dataclass
class AssertionMatch:
    assertion: Assertion
    effective_confidence: float
    is_trustworthy: bool


class AssertionsStore:
    """Persistent cross-session store for verified facts with decay-based confidence."""

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
        decay_class = decay_class_override or self._assign_decay_class(subject, domain)
        assertion = Assertion(
            subject=subject, domain=domain, claim=claim,
            confidence_at_write=confidence, decay_class=decay_class,
            timestamp=time.time(), source=source, evidence_summary=evidence_summary,
        )
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
        threshold = min_confidence or self.confidence_threshold
        now = time.time()
        results = []
        for assertion in self.assertions:
            if subject.lower() not in assertion.subject.lower():
                continue
            if domain and assertion.domain != domain:
                continue
            eff_conf = assertion.effective_confidence(now)
            if eff_conf >= threshold:
                results.append(AssertionMatch(
                    assertion=assertion,
                    effective_confidence=round(eff_conf, 4),
                    is_trustworthy=True,
                ))
        results.sort(key=lambda m: m.effective_confidence, reverse=True)
        return results

    def query_contradictions(
        self, subject: str, new_claim: str, domain: Optional[str] = None
    ) -> List[Tuple[AssertionMatch, str]]:
        matches = self.query(subject, domain)
        contradictions = []
        for match in matches:
            stored = match.assertion.claim.lower()
            if self._claims_contradict(stored, new_claim.lower()):
                contradictions.append((
                    match,
                    f"Conflicts with stored assertion (conf={match.effective_confidence:.2f}): "
                    f"'{match.assertion.claim[:100]}'",
                ))
        return contradictions

    def prune_stale(self, min_effective_confidence: float = 0.05) -> int:
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
        by_source: Dict[str, int] = {}
        for a in self.assertions:
            by_class[a.decay_class.value] += 1
            if a.is_trustworthy(self.confidence_threshold, now):
                trustworthy += 1
            by_source[a.source] = by_source.get(a.source, 0) + 1
        return {
            "total": len(self.assertions),
            "trustworthy": trustworthy,
            "by_decay_class": by_class,
            "by_source": by_source,
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
        subject_lower = subject.lower()
        for key, cls in SUBJECT_DECAY_OVERRIDES.items():
            if key in subject_lower:
                return cls
        return FIELD_DECAY_CLASS.get(domain, DecayClass.C)

    def _claims_contradict(self, stored: str, new: str) -> bool:
        pairs = [
            ("is ", "is not "), ("can ", "cannot "), ("does ", "does not "),
            ("always ", "never "), ("true", "false"), ("correct", "incorrect"),
            ("valid", "invalid"), ("o(n)", "o(n^2)"), ("o(1)", "o(n)"),
            ("o(n log n)", "o(n^2)"), ("o(n)", "o(n log n)"),
        ]
        for pos, neg in pairs:
            if pos in stored and neg in new:
                return True
            if neg in stored and pos in new:
                return True
        return False
