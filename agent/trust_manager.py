"""
Trust and reputation system.

Entity trust is two-dimensional:
    - domain_expertise: bootstrapped from verifiable credentials on day one
    - behavioral_trust: accumulates through interaction history (tit-for-tat)

External escalation requires BOTH to exceed field-specific thresholds.
Sybil resistance: escalation queries carry attribution, creating reputational
accountability that discourages false information from real experts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class CredentialLevel(float, Enum):
    """Domain expertise score bootstrapped from verifiable credentials."""
    NONE        = 0.00
    STUDENT     = 0.30
    PRACTITIONER = 0.50  # e.g. general practitioner, junior engineer
    SPECIALIST  = 0.70  # e.g. specialist physician, senior engineer
    EXPERT      = 0.85  # e.g. department head, principal engineer
    AUTHORITY   = 0.95  # e.g. professor, board-certified specialist, author


# Field-specific escalation trust thresholds (behavioral_trust floor)
ESCALATION_TRUST_THRESHOLD: Dict[str, float] = {
    "surgery":                0.90,
    "aviation":               0.90,
    "law":                    0.80,
    "structural_engineering": 0.80,
    "software_engineering":   0.70,
    "stem_research":          0.65,
    "education":              0.65,
    "art":                    0.60,
    "creative_writing":       0.60,
    "general":                0.65,
    "blended":                0.70,
}

# Domain expertise floor required for escalation (must be above median = 0.50)
EXPERTISE_ESCALATION_FLOOR = 0.50


@dataclass
class Credential:
    """Verifiable credential provided by an entity."""
    credential_type: str      # "degree", "certification", "experience", "publication"
    domain: str
    level: CredentialLevel
    institution: str = ""
    years: int = 0
    verified: bool = False    # whether we've verified the credential externally


@dataclass
class InteractionRecord:
    """Record of a single interaction for behavioral trust computation."""
    interaction_id: str
    input_accuracy: float    # 0-1: was their input factually correct?
    cooperative: bool        # did they behave cooperatively?
    defected: bool           # did they provide false/misleading input?
    timestamp: float = 0.0


@dataclass
class EntityTrust:
    """
    Trust profile for a single entity.

    domain_expertise:   bootstrapped from credentials, stable over time
    behavioral_trust:   earned through interaction history, lenient tit-for-tat
    """
    entity_id: str
    domain_expertise: float = 0.0        # 0-1, from credentials
    behavioral_trust: float = 0.5        # 0-1, starts cooperative neutral
    interactions: List[InteractionRecord] = field(default_factory=list)
    credentials: List[Credential] = field(default_factory=list)
    is_cooperative: bool = True          # current tit-for-tat state
    defection_count: int = 0
    forgiven_defections: int = 0
    notes: str = ""

    @property
    def interaction_count(self) -> int:
        return len(self.interactions)

    def is_eligible_for_escalation(self, field: str) -> bool:
        """
        Escalation requires BOTH:
            domain_expertise > median (0.50)
            behavioral_trust > field-specific threshold
        Neither dimension can compensate for the other.
        """
        trust_threshold = ESCALATION_TRUST_THRESHOLD.get(field, 0.70)
        return (
            self.domain_expertise > EXPERTISE_ESCALATION_FLOOR and
            self.behavioral_trust > trust_threshold
        )

    def escalation_weight(self) -> float:
        """
        Weight applied to this entity's input as empirical evidence.
        w_expert = trust × expertise
        """
        return self.behavioral_trust * self.domain_expertise


class TrustManager:
    """
    Manages entity trust scores across the system.

    Lenient tit-for-tat strategy:
        - Start cooperatively (trust = 0.5, not 0)
        - Mirror behavior: reward cooperation, penalize defection
        - Forgive occasional defection (up to 1 in 5 interactions)
        - Never reach 0 — trust floor at 0.1 to allow recovery
    """

    TRUST_FLOOR = 0.10
    TRUST_CEILING = 0.98
    FORGIVENESS_RATE = 0.20   # forgive if defection_rate < 20%
    COOPERATION_REWARD = 0.05
    DEFECTION_PENALTY = 0.15
    ACCURACY_WEIGHT = 0.6     # weight of input accuracy in behavioral trust
    COOPERATION_WEIGHT = 0.4  # weight of behavioral cooperation

    def __init__(self):
        self.entities: Dict[str, EntityTrust] = {}

    def register_entity(
        self,
        entity_id: str,
        credentials: Optional[List[Credential]] = None,
    ) -> EntityTrust:
        """
        Register a new entity. Domain expertise bootstrapped from credentials
        immediately — behavioral trust starts at cooperative neutral (0.5).
        """
        entity = EntityTrust(entity_id=entity_id)
        if credentials:
            entity.credentials = credentials
            entity.domain_expertise = self._compute_expertise_from_credentials(credentials)
        self.entities[entity_id] = entity
        return entity

    def get_entity(self, entity_id: str) -> Optional[EntityTrust]:
        return self.entities.get(entity_id)

    def get_or_create(self, entity_id: str) -> EntityTrust:
        if entity_id not in self.entities:
            self.entities[entity_id] = EntityTrust(entity_id=entity_id)
        return self.entities[entity_id]

    def record_interaction(
        self,
        entity_id: str,
        interaction_id: str,
        input_accuracy: float,
        cooperative: bool,
        defected: bool = False,
        timestamp: float = 0.0,
    ) -> EntityTrust:
        """
        Update behavioral trust from an interaction.
        Accuracy = factual correctness of input.
        Cooperative = did not attempt to game, mislead, or probe the system.
        """
        import time
        entity = self.get_or_create(entity_id)

        record = InteractionRecord(
            interaction_id=interaction_id,
            input_accuracy=input_accuracy,
            cooperative=cooperative,
            defected=defected,
            timestamp=timestamp or time.time(),
        )
        entity.interactions.append(record)

        if defected:
            entity.defection_count += 1
            entity.is_cooperative = False
            entity.behavioral_trust = max(
                self.TRUST_FLOOR,
                entity.behavioral_trust - self.DEFECTION_PENALTY
            )
        else:
            # Tit-for-tat: mirror cooperation with reward
            trust_delta = (
                self.ACCURACY_WEIGHT * input_accuracy * self.COOPERATION_REWARD +
                self.COOPERATION_WEIGHT * (1.0 if cooperative else 0.0) * self.COOPERATION_REWARD
            )
            entity.behavioral_trust = min(
                self.TRUST_CEILING,
                entity.behavioral_trust + trust_delta
            )

            # Forgiveness: if defection rate drops below threshold, restore
            # cooperative state
            if not entity.is_cooperative:
                total = entity.interaction_count
                defect_rate = entity.defection_count / max(total, 1)
                if defect_rate < self.FORGIVENESS_RATE:
                    entity.is_cooperative = True
                    entity.forgiven_defections += 1

        return entity

    def get_eligible_escalation_entities(
        self,
        field: str,
        top_n: int = 3,
    ) -> List[EntityTrust]:
        """
        Return entities eligible for external escalation in this field,
        sorted by escalation_weight (trust × expertise) descending.
        """
        eligible = [
            e for e in self.entities.values()
            if e.is_eligible_for_escalation(field)
        ]
        eligible.sort(key=lambda e: e.escalation_weight(), reverse=True)
        return eligible[:top_n]

    def trust_summary(self, entity_id: str) -> dict:
        entity = self.entities.get(entity_id)
        if not entity:
            return {}
        return {
            "entity_id": entity_id,
            "domain_expertise": round(entity.domain_expertise, 3),
            "behavioral_trust": round(entity.behavioral_trust, 3),
            "escalation_weight": round(entity.escalation_weight(), 3),
            "interactions": entity.interaction_count,
            "defection_count": entity.defection_count,
            "is_cooperative": entity.is_cooperative,
            "credentials": len(entity.credentials),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_expertise_from_credentials(
        self, credentials: List[Credential]
    ) -> float:
        """
        Bootstrap domain expertise from verified credentials.
        Takes the maximum level across all credentials (specialisms stack
        horizontally, not vertically — two practitioner certs don't
        equal one expert cert).
        Unverified credentials are halved.
        """
        if not credentials:
            return 0.0

        max_level = 0.0
        for cred in credentials:
            level = float(cred.level)
            if not cred.verified:
                level *= 0.5   # unverified halved
            if cred.years >= 5:
                level = min(self.TRUST_CEILING, level + 0.05)  # experience bonus
            max_level = max(max_level, level)

        return round(max_level, 3)
