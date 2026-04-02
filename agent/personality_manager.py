"""
Personality manager: maintains trait weights and adjusts them
based on accumulated utility history.

Runs as a periodic service (every N interactions or every few hours).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Trait:
    name: str
    score: float = 0.5          # current weight, 0 to 1
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)


INITIAL_TRAITS = {
    "curiosity": Trait(
        name="curiosity", score=0.6,
        advantages=["explores new domains", "asks clarifying questions", "finds novel approaches"],
        disadvantages=["may attempt tasks beyond confidence bounds", "less conservative"]
    ),
    "caution": Trait(
        name="caution", score=0.5,
        advantages=["avoids overconfident errors", "stays within confidence bounds"],
        disadvantages=["may abstain too often", "slower to improve efficacy"]
    ),
    "assertiveness": Trait(
        name="assertiveness", score=0.5,
        advantages=["gives direct answers", "drives toward high efficacy"],
        disadvantages=["may overclaim confidence", "less likely to hedge"]
    ),
    "analytical_rigor": Trait(
        name="analytical_rigor", score=0.6,
        advantages=["catches contradictions", "improves confidence calibration"],
        disadvantages=["slower responses", "verbose"]
    ),
    "creativity": Trait(
        name="creativity", score=0.4,
        advantages=["novel solutions", "higher efficacy ceiling in creative domains"],
        disadvantages=["less reproducible", "harder to verify"]
    ),
    "conciseness": Trait(
        name="conciseness", score=0.5,
        advantages=["efficient responses", "clearer outputs"],
        disadvantages=["may omit useful caveats", "less context"]
    ),
}

# Which traits are most relevant for each situation type
SITUATIONAL_RELEVANCE = {
    "high_stakes": {"caution": 0.9, "analytical_rigor": 0.8, "assertiveness": 0.2},
    "exploration":  {"curiosity": 0.9, "creativity": 0.7, "caution": 0.3},
    "technical":    {"analytical_rigor": 0.9, "conciseness": 0.6, "creativity": 0.4},
    "creative":     {"creativity": 0.9, "curiosity": 0.7, "analytical_rigor": 0.3},
    "uncertain":    {"caution": 0.8, "analytical_rigor": 0.7, "assertiveness": 0.2},
}


class PersonalityManager:
    """
    Manages personality trait weights and generates situational trait prompts.
    Runs an evolution step periodically based on utility history.
    """

    def __init__(self):
        self.traits: Dict[str, Trait] = {k: Trait(**vars(v)) for k, v in INITIAL_TRAITS.items()}
        self.evolution_history: List[dict] = []

    def get_active_weights(self, situation_type: str = "technical") -> Dict[str, float]:
        """
        Compute active trait weights for a given situation.
        Blends base trait scores with situational relevance.
        """
        relevance = SITUATIONAL_RELEVANCE.get(situation_type, {})
        weights = {}

        for name, trait in self.traits.items():
            situational_boost = relevance.get(name, 0.5)
            weights[name] = trait.score * situational_boost

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def build_personality_prompt(self, situation_type: str = "technical") -> str:
        """
        Generate a personality instruction string to inject into the system prompt.
        """
        weights = self.get_active_weights(situation_type)
        top_traits = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]

        lines = ["Your active personality traits for this task (in order of emphasis):"]
        for name, weight in top_traits:
            trait = self.traits[name]
            lines.append(
                f"  - {name.upper()} (weight={weight:.2f}): "
                f"advantages: {', '.join(trait.advantages[:2])}"
            )
        return "\n".join(lines)

    def evolve(self, utility_history: List[float], contradiction_rate: float, domain: str):
        """
        Periodic evolution step. Adjusts trait scores based on recent performance.
        Call every N interactions or on a schedule.

        Args:
            utility_history: recent U scores (last N interactions)
            contradiction_rate: fraction of recent interactions with contradictions
            domain: current primary domain
        """
        if len(utility_history) < 5:
            return  # not enough data yet

        recent_trend = utility_history[-5:]
        is_improving = recent_trend[-1] > recent_trend[0]
        avg_utility = sum(recent_trend) / len(recent_trend)

        changes = {}

        # If utility is declining and contradictions are high → boost caution and rigor
        if not is_improving and contradiction_rate > 0.2:
            changes["caution"] = +0.05
            changes["analytical_rigor"] = +0.05
            changes["assertiveness"] = -0.05

        # If utility is improving → can afford to boost curiosity and creativity
        if is_improving and avg_utility > 0.6:
            changes["curiosity"] = +0.03
            changes["creativity"] = +0.02

        # If very high contradiction rate → strong caution boost regardless
        if contradiction_rate > 0.4:
            changes["caution"] = changes.get("caution", 0) + 0.08
            changes["assertiveness"] = changes.get("assertiveness", 0) - 0.08

        # Apply changes with bounds [0.1, 0.9]
        for trait_name, delta in changes.items():
            if trait_name in self.traits:
                new_score = self.traits[trait_name].score + delta
                self.traits[trait_name].score = max(0.1, min(0.9, new_score))

        self.evolution_history.append({
            "domain": domain,
            "avg_utility": round(avg_utility, 4),
            "contradiction_rate": round(contradiction_rate, 4),
            "is_improving": is_improving,
            "changes": changes
        })

    def get_trait_summary(self) -> Dict[str, float]:
        return {name: round(t.score, 3) for name, t in self.traits.items()}
