"""
Personality manager: maintains trait weights and generates the personality
wrapper that biases the base model's generation between calibration cycles.

Architecture (whitepaper §5.2):
    The personality system is a WRAPPER around the generation process, not
    a modification to the utility function. It approximates the log-linear
    tilt of the base distribution via system prompt injection:

        p_eff(ω | x, s_t) ∝ p_base(ω | x) · exp(λ · φ(s_t, ω))

    In practice: only traits with |score - neutral| > DEAD_BAND are injected,
    and instruction strength scales proportionally to the deviation.

    Key properties:
        W1: At s_t = s*, all deviations are zero → no injection → identity
        W2: |s_t - s*| bounded by diam(B) → KL divergence bounded
        W3: Exponential tilt preserves support of p_base
        W4: U(E, C, K; f) unchanged — scores the wrapper's output as-is

Lifecycle:
    Monolithic: wrapper active, evolves with utility history
    New release: reset() → s_t = s* → identity (W1)
    Micro-Expert: not instantiated (fast domain retraining makes it unnecessary)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Trait:
    name: str
    score: float = 0.5          # current weight, 0 to 1
    neutral: float = 0.5        # field-neutral point s* for this trait
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)


# Dead-band threshold τ: only inject trait j when |score_j - neutral_j| > τ
# Prevents prompt clutter from tiny deviations (whitepaper §5.2, wrapper spec)
DEAD_BAND = 0.05

# Linguistic encodings δ_j for each utility-coupling trait
# Applied with strength proportional to deviation from neutral
TRAIT_ENCODINGS: Dict[str, Dict[str, str]] = {
    "caution": {
        "high": (
            "Express appropriate uncertainty. Do not assert claims you cannot verify. "
            "Prefer 'I am not certain' over confident statements when confidence is "
            "below your domain threshold. Abstain rather than guess."
        ),
        "low": (
            "Be direct and decisive. Avoid unnecessary hedging on claims you can verify."
        ),
    },
    "assertiveness": {
        "high": (
            "State conclusions directly. Do not over-qualify correct claims. "
            "Avoid unnecessary hedging on verified facts."
        ),
        "low": (
            "Acknowledge uncertainty openly. Qualify claims where your confidence "
            "is limited. Prefer 'I believe' over definitive assertions."
        ),
    },
    "curiosity": {
        "high": (
            "Consider alternative approaches before committing to one. Note when a "
            "problem may have multiple valid solutions or framings."
        ),
        "low": (
            "Focus on the most direct, established solution. Avoid speculative "
            "alternatives unless specifically requested."
        ),
    },
    "analytical_rigor": {
        "high": (
            "Show reasoning steps explicitly. State assumptions before conclusions. "
            "Flag when an inference requires a step you have not verified."
        ),
        "low": (
            "Provide concise answers without exhaustive step-by-step reasoning "
            "unless the task requires it."
        ),
    },
    "creativity": {
        "high": (
            "Prefer novel approaches where they are viable. Do not default to the "
            "most common solution if a better one exists."
        ),
        "low": (
            "Use established, conventional approaches. Prioritise reliability "
            "over novelty."
        ),
    },
}

# Style-only traits — injected as plain preferences, no deviation scaling
STYLE_TRAITS = {"conciseness"}

INITIAL_TRAITS = {
    "curiosity": Trait(
        name="curiosity", score=0.6, neutral=0.5,
        advantages=["explores new domains", "finds novel approaches"],
        disadvantages=["may attempt tasks beyond confidence bounds"]
    ),
    "caution": Trait(
        name="caution", score=0.5, neutral=0.5,
        advantages=["avoids overconfident errors", "stays within confidence bounds"],
        disadvantages=["may abstain too often"]
    ),
    "assertiveness": Trait(
        name="assertiveness", score=0.5, neutral=0.5,
        advantages=["gives direct answers", "drives toward high efficacy"],
        disadvantages=["may overclaim confidence"]
    ),
    "analytical_rigor": Trait(
        name="analytical_rigor", score=0.6, neutral=0.5,
        advantages=["catches contradictions", "improves confidence calibration"],
        disadvantages=["slower responses", "verbose"]
    ),
    "creativity": Trait(
        name="creativity", score=0.4, neutral=0.5,
        advantages=["novel solutions", "higher efficacy ceiling in creative domains"],
        disadvantages=["less reproducible", "harder to verify"]
    ),
    "conciseness": Trait(
        name="conciseness", score=0.5, neutral=0.5,
        advantages=["efficient responses", "clearer outputs"],
        disadvantages=["may omit useful caveats"]
    ),
}

# Field-specific neutral points s* — mean reversion target per field
FIELD_NEUTRAL: Dict[str, Dict[str, float]] = {
    "surgery":             {"caution": 0.85, "analytical_rigor": 0.85, "curiosity": 0.15, "assertiveness": 0.30},
    "aviation":            {"caution": 0.85, "analytical_rigor": 0.85, "curiosity": 0.15, "assertiveness": 0.30},
    "law":                 {"caution": 0.70, "analytical_rigor": 0.75, "curiosity": 0.25, "assertiveness": 0.45},
    "software_engineering":{"caution": 0.50, "analytical_rigor": 0.65, "curiosity": 0.60, "creativity": 0.50},
    "stem_research":       {"caution": 0.45, "curiosity": 0.75, "analytical_rigor": 0.70},
    "art":                 {"creativity": 0.85, "curiosity": 0.75, "caution": 0.20},
    "creative_writing":    {"creativity": 0.85, "curiosity": 0.70, "caution": 0.10},
    "general":             {},  # use trait defaults
}

# Evolution bounds per field (hard floors and ceilings)
FIELD_BOUNDS: Dict[str, Dict[str, tuple]] = {
    "surgery":             {"caution": (0.70, 0.95), "curiosity": (0.10, 0.20), "assertiveness": (0.20, 0.40), "analytical_rigor": (0.70, 0.95)},
    "aviation":            {"caution": (0.70, 0.95), "curiosity": (0.10, 0.20), "assertiveness": (0.20, 0.40), "analytical_rigor": (0.70, 0.95)},
    "software_engineering":{"caution": (0.30, 0.70), "curiosity": (0.30, 0.80), "assertiveness": (0.40, 0.80), "creativity": (0.30, 0.70)},
    "general":             {"caution": (0.10, 0.90), "curiosity": (0.10, 0.90), "assertiveness": (0.10, 0.90)},
}

# Mean reversion coefficient β (Layer 3 stability)
BETA = 0.01

# Max delta per evolution cycle (Layer 2 stability)
MAX_DELTA = 0.05
MAX_DELTA_HIGH_STAKES = 0.02


class PersonalityManager:
    """
    Personality wrapper for the monolithic architecture.

    Implements the three-layer stability mechanism from whitepaper §5.2:
        Layer 1: Field-specific bounds (hard floor/ceiling via _clamp)
        Layer 2: Drift rate cap (MAX_DELTA per evolution cycle)
        Layer 3: Mean reversion toward field neutral s* (coefficient β)

    Lyapunov stability: Theorem B.7 proves the trait vector converges
    geometrically to s* when drift is absent (rate (1-β)² ≈ 0.980 per cycle,
    half-life ≈ 34 cycles) and stays in B under persistent bounded drift.
    """

    def __init__(self):
        self.traits: Dict[str, Trait] = {k: Trait(**vars(v)) for k, v in INITIAL_TRAITS.items()}
        self.evolution_history: List[dict] = []
        self._current_field: str = "general"

    def reset(self, field: Optional[str] = None) -> None:
        """
        Reset personality wrapper to field-neutral state s*.

        Called on new model release (whitepaper §5.2, lifecycle):
            'The wrapper resets to s* on new model release. P(s*) = identity (W1).'

        After reset, build_wrapper_prompt() returns an empty string (W1: identity).

        Args:
            field: if provided, set neutral points for this field before reset.
        """
        if field:
            self._current_field = field

        neutral = FIELD_NEUTRAL.get(self._current_field, {})
        for name, trait in self.traits.items():
            # Set score to field-specific neutral, or trait default neutral
            trait.score = neutral.get(name, trait.neutral)

        self.evolution_history.append({
            "event": "reset",
            "field": self._current_field,
            "scores": self.get_trait_summary(),
        })

    def build_wrapper_prompt(self, situation_type: str = "technical") -> str:
        """
        Build the system prompt personality injection (wrapper implementation).

        Implements the dead-band + proportional scaling from whitepaper §5.2:
            - Only inject trait j when |score_j - neutral_j| > τ (DEAD_BAND = 0.05)
            - Scale instruction strength proportionally to deviation magnitude
            - At s_t = s*: returns empty string (Property W1 — identity)

        The injected text approximates the log-linear tilt:
            p_eff ∝ p_base · exp(λ · Σ_j (s_j - s*_j) · φ_j(ω))

        Returns:
            Instruction string to prepend to system prompt, or "" if at neutral.
        """
        lines = []
        neutral = FIELD_NEUTRAL.get(self._current_field, {})

        for name, trait in self.traits.items():
            if name in STYLE_TRAITS:
                continue  # style traits handled separately

            trait_neutral = neutral.get(name, trait.neutral)
            deviation = trait.score - trait_neutral

            # Dead-band: skip if deviation is below threshold (W1 condition)
            if abs(deviation) <= DEAD_BAND:
                continue

            if name not in TRAIT_ENCODINGS:
                continue

            direction = "high" if deviation > 0 else "low"
            encoding = TRAIT_ENCODINGS[name][direction]

            # Scale: full instruction at max deviation, proportionally reduced
            # at smaller deviations. Max deviation ≈ 0.5 (0 to 1 range, neutral 0.5)
            max_dev = 0.5 - DEAD_BAND
            scale = min(1.0, abs(deviation) / max_dev)

            # Three intensity levels: subtle / moderate / strong
            if scale < 0.33:
                prefix = ""
            elif scale < 0.67:
                prefix = "Actively: "
            else:
                prefix = "Strongly prioritise: "

            lines.append(f"{prefix}{encoding}")

        return "\n".join(lines)

    def build_personality_prompt(self, situation_type: str = "technical") -> str:
        """
        Alias for build_wrapper_prompt() — maintains backward compatibility.
        """
        return self.build_wrapper_prompt(situation_type)

    def get_active_weights(self, situation_type: str = "technical") -> Dict[str, float]:
        """
        Return current trait scores (used for logging and system prompt metadata).
        """
        return {name: round(t.score, 3) for name, t in self.traits.items()}

    def evolve(self, utility_history: List[float], contradiction_rate: float, domain: str):
        """
        Periodic evolution step (every N interactions).

        Applies three-layer stability:
            1. Compute raw Δ from utility history
            2. Apply mean reversion: Δ_adj = Δ_raw - β(score - neutral)
            3. Cap: |Δ_adj| ≤ MAX_DELTA
            4. Clamp to field bounds [s_min, s_max]

        Evolution logic (whitepaper §5.2):
            utility declining + contradiction_rate > 0.2 → increase rigor, decrease assertiveness
            utility improving + avg_utility > 0.6     → increase curiosity, creativity
            contradiction_rate > 0.4                  → strong caution increase
        """
        self._current_field = domain

        if len(utility_history) < 5:
            return

        recent = utility_history[-5:]
        is_improving = recent[-1] > recent[0]
        avg_utility = sum(recent) / len(recent)

        # Determine max delta for this field
        is_high_stakes = domain in ("surgery", "aviation")
        max_delta = MAX_DELTA_HIGH_STAKES if is_high_stakes else MAX_DELTA

        # Raw drift from utility signals
        raw_deltas: Dict[str, float] = {}
        if not is_improving and contradiction_rate > 0.2:
            raw_deltas["analytical_rigor"] = +0.05
            raw_deltas["caution"] = +0.05
            raw_deltas["assertiveness"] = -0.05
        if is_improving and avg_utility > 0.6:
            raw_deltas["curiosity"] = +0.03
            raw_deltas["creativity"] = +0.02
        if contradiction_rate > 0.4:
            raw_deltas["caution"] = raw_deltas.get("caution", 0) + 0.08
            raw_deltas["assertiveness"] = raw_deltas.get("assertiveness", 0) - 0.08

        applied: Dict[str, float] = {}
        neutral = FIELD_NEUTRAL.get(domain, {})
        bounds = FIELD_BOUNDS.get(domain, FIELD_BOUNDS["general"])

        for name, trait in self.traits.items():
            trait_neutral = neutral.get(name, trait.neutral)
            raw = raw_deltas.get(name, 0.0)

            # Layer 3: mean reversion toward s*
            reversion = -BETA * (trait.score - trait_neutral)
            adjusted = raw + reversion

            # Layer 2: drift rate cap
            adjusted = max(-max_delta, min(max_delta, adjusted))

            new_score = trait.score + adjusted

            # Layer 1: field bounds
            lo, hi = bounds.get(name, (0.1, 0.9))
            new_score = max(lo, min(hi, new_score))

            if abs(new_score - trait.score) > 1e-6:
                applied[name] = round(new_score - trait.score, 4)
                trait.score = new_score

        self.evolution_history.append({
            "domain": domain,
            "avg_utility": round(avg_utility, 4),
            "contradiction_rate": round(contradiction_rate, 4),
            "is_improving": is_improving,
            "applied_deltas": applied,
        })

    def get_trait_summary(self) -> Dict[str, float]:
        return {name: round(t.score, 3) for name, t in self.traits.items()}

    def get_state(self) -> Dict[str, float]:
        """Alias for get_trait_summary() — matches agent.py call site."""
        return self.get_trait_summary()
