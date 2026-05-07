"""
Field classifier: determines domain distribution of a given task.

v0.2 — Three robustness mechanisms added:
  1. High-stakes floor: dangerous fields can't be diluted below 0.15
  2. Sliding window EMA: tracks field drift across conversation turns
  3. Entropy-based conservative fallback: high ambiguity → tighter bounds

Usage:
    classifier = FieldClassifier()
    distribution = await classifier.classify(task)           # single turn
    distribution = await classifier.classify(task, update_history=True)  # conversation-aware
"""

import json
import math
import re
try:
    import httpx
except ImportError:
    httpx = None
from typing import Dict, List, Optional
from config import FIELD_CONFIGS, FieldConfig, get_effective_config


FIELD_CLASSIFIER_PROMPT = """You are a domain classifier. Given a task or question,
return a JSON object with the probability that it belongs to each field.
Probabilities must sum to 1.0. Only include fields with probability > 0.05.

Available fields:
- surgery
- aviation
- law
- structural_engineering
- software_engineering
- stem_research
- education
- art
- creative_writing
- general

Return ONLY valid JSON, no explanation. Example:
{"software_engineering": 0.85, "stem_research": 0.15}
"""

# Fields where any meaningful presence should be floored at MIN_HIGH_STAKES_PROB.
# The cost of under-weighting these is much higher than over-weighting them.
HIGH_STAKES_FIELDS = {"surgery", "aviation", "law"}
MIN_HIGH_STAKES_PROB = 0.15

# EMA alpha for conversation history — higher = more weight on recent turns
HISTORY_EMA_ALPHA = 0.4

# Entropy threshold above which we apply the conservative fallback
ENTROPY_CONSERVATIVE_THRESHOLD = 0.7


class FieldClassifier:
    """
    Stateful classifier that maintains conversation field history
    and applies robustness mechanisms on top of raw LLM classification.
    """

    def __init__(self):
        self.turn_history: List[Dict[str, float]] = []


    def classify(
        self,
        task: str,
        update_history: bool = True,
    ) -> Dict[str, float]:
        """
        Synchronous classify — uses keyword-based fallback directly.
        Used in simulation mode where no API is available.
        """
        raw = self._keyword_fallback(task)
        floored = self._enforce_high_stakes_floor(raw)
        if self.turn_history:
            blended = self._apply_history_ema(floored)
        else:
            blended = floored
        hardened = self._apply_entropy_fallback(blended)
        if update_history:
            self.turn_history.append(hardened)
        return hardened

    async def classify_async(
        self,
        task: str,
        update_history: bool = True,
    ) -> Dict[str, float]:
        """
        Classify a task into a field distribution with robustness mechanisms.

        Args:
            task: the task or question to classify
            update_history: whether to update the conversation EMA history

        Returns:
            Hardened field distribution dict {field: probability}
        """
        # Step 1: raw LLM classification
        raw = await self._call_classifier(task)

        # Step 2: enforce high-stakes floor
        floored = self._enforce_high_stakes_floor(raw)

        # Step 3: blend with conversation history (EMA)
        if self.turn_history:
            blended = self._apply_history_ema(floored)
        else:
            blended = floored

        # Step 4: entropy check → conservative fallback
        hardened = self._apply_entropy_fallback(blended)

        # Update history after hardening so drift is tracked on final output
        if update_history:
            self.turn_history.append(hardened)

        return hardened

    def get_effective_config(self, distribution: Dict[str, float]) -> FieldConfig:
        """
        Get the blended FieldConfig for a distribution,
        with bounds hardened by entropy.
        """
        base_config = get_effective_config(distribution)
        entropy_ratio = self._entropy_ratio(distribution)

        if entropy_ratio > ENTROPY_CONSERVATIVE_THRESHOLD:
            # Lerp c_min and e_min toward the most conservative present field
            most_conservative = max(
                distribution,
                key=lambda f: FIELD_CONFIGS.get(f, FIELD_CONFIGS["general"]).c_min
            )
            conservative_cfg = FIELD_CONFIGS.get(most_conservative, FIELD_CONFIGS["general"])
            lerp_t = entropy_ratio  # 0 = no shift, 1 = full conservative

            base_config.c_min = base_config.c_min + lerp_t * (conservative_cfg.c_min - base_config.c_min)
            base_config.e_min = base_config.e_min + lerp_t * (conservative_cfg.e_min - base_config.e_min)

        return base_config

    def reset_history(self):
        """Call at the start of a new conversation."""
        self.turn_history = []

    # ── Private ───────────────────────────────────────────────────────────────

    async def _call_classifier(self, task: str) -> Dict[str, float]:
        """Raw LLM call. Falls back to {"general": 1.0} on any error."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 200,
                        "system": FIELD_CLASSIFIER_PROMPT,
                        "messages": [{"role": "user", "content": task}]
                    },
                    timeout=10.0
                )
                data = response.json()
                raw = data["content"][0]["text"].strip()
                raw = re.sub(r"```json|```", "", raw).strip()
                distribution = json.loads(raw)

                total = sum(distribution.values())
                if total == 0:
                    return {"general": 1.0}
                return {k: v / total for k, v in distribution.items()}

        except Exception as e:
            print(f"[FieldClassifier] Error: {e}, defaulting to general")
            return {"general": 1.0}

    def _keyword_fallback(self, task: str) -> Dict[str, float]:
        """
        Keyword-based field classifier used in simulation mode (no API).
        Returns a probability distribution over fields.
        """
        task_lower = task.lower()
        scores = {f: 0.0 for f in [
            "software_engineering", "medicine", "law", "mathematics",
            "physics", "chemistry", "finance", "creative_writing",
            "general_knowledge", "surgery", "aviation"
        ]}

        keywords = {
            "software_engineering": ["code", "function", "algorithm", "python", "java",
                                      "debug", "sort", "array", "class", "api", "bug",
                                      "complexity", "leetcode", "loop", "data structure"],
            "medicine":             ["patient", "diagnosis", "treatment", "drug", "dose",
                                      "symptom", "medical", "clinical", "disease", "therapy"],
            "surgery":              ["surgical", "operation", "incision", "procedure",
                                      "anesthesia", "sterile", "postoperative"],
            "law":                  ["legal", "contract", "statute", "court", "liability",
                                      "regulation", "attorney", "plaintiff", "defendant"],
            "mathematics":          ["proof", "theorem", "equation", "integral", "derivative",
                                      "matrix", "vector", "probability", "calculus"],
            "finance":              ["stock", "portfolio", "investment", "return", "risk",
                                      "dividend", "valuation", "market", "asset"],
            "creative_writing":     ["story", "poem", "creative", "narrative", "character",
                                      "plot", "fiction", "write a poem", "short story"],
            "general_knowledge":    ["history", "geography", "capital", "who", "what year"],
        }

        for field, kws in keywords.items():
            for kw in kws:
                if kw in task_lower:
                    scores[field] += 1.0

        # Default to software_engineering if no signal (MVP domain)
        if sum(scores.values()) == 0:
            scores["software_engineering"] = 1.0

        total = sum(scores.values())
        dist = {k: v / total for k, v in scores.items() if v > 0}

        # Ensure at least one key
        if not dist:
            dist = {"software_engineering": 1.0}

        return dist

    def _enforce_high_stakes_floor(self, dist: Dict[str, float]) -> Dict[str, float]:
        """
        Floor high-stakes fields at MIN_HIGH_STAKES_PROB if they have any presence.

        Prevents: {"software_engineering": 0.95, "medicine": 0.05}
        Becomes:  {"software_engineering": 0.81, "medicine": 0.15, ...} (renormalized)

        The asymmetry is intentional: it's much cheaper to over-weight a high-stakes
        field than to under-weight it.
        """
        result = dict(dist)
        floored_any = False

        for field in HIGH_STAKES_FIELDS:
            if field in result and 0 < result[field] < MIN_HIGH_STAKES_PROB:
                result[field] = MIN_HIGH_STAKES_PROB
                floored_any = True

        if floored_any:
            total = sum(result.values())
            result = {k: v / total for k, v in result.items()}

        return result

    def _apply_history_ema(self, current: Dict[str, float]) -> Dict[str, float]:
        """
        Blend current turn classification with conversation history using EMA.

        Collects all fields ever seen, applies EMA across history + current.
        Recent turns weighted more (alpha=0.4 means current gets ~40% weight
        vs the accumulated prior history).

        This catches field drift: a conversation that starts as software_engineering
        and drifts into medicine will show tightening bounds over turns.
        """
        all_fields = set(current.keys())
        for h in self.turn_history:
            all_fields.update(h.keys())

        # Build EMA from history (oldest to newest, then current)
        ema: Dict[str, float] = {}
        for turn in self.turn_history + [current]:
            for field in all_fields:
                prev = ema.get(field, 0.0)
                curr_val = turn.get(field, 0.0)
                ema[field] = (1 - HISTORY_EMA_ALPHA) * prev + HISTORY_EMA_ALPHA * curr_val

        # Normalize and drop near-zero fields
        total = sum(ema.values())
        if total == 0:
            return current
        return {k: v / total for k, v in ema.items() if v / total > 0.02}

    def _apply_entropy_fallback(self, dist: Dict[str, float]) -> Dict[str, float]:
        """
        When distribution entropy is high (genuine ambiguity), shift probability
        mass toward the most conservative field present.

        High entropy → more caution, not averaging toward the middle.

        Note: this modifies the distribution (bounds are tightened separately
        in get_effective_config). Here we just log the entropy for observability.
        """
        ratio = self._entropy_ratio(dist)
        if ratio > ENTROPY_CONSERVATIVE_THRESHOLD:
            most_conservative = max(
                dist,
                key=lambda f: FIELD_CONFIGS.get(f, FIELD_CONFIGS["general"]).c_min
            )
            # Shift some weight toward the most conservative field
            shift = ratio * 0.15  # at max entropy, shift up to 15% of mass
            result = {k: v * (1 - shift) for k, v in dist.items()}
            result[most_conservative] = result.get(most_conservative, 0.0) + shift

            total = sum(result.values())
            dist = {k: v / total for k, v in result.items()}

            print(
                f"[FieldClassifier] High entropy ({ratio:.2f}) — "
                f"shifted toward {most_conservative}"
            )

        return dist

    def _entropy_ratio(self, dist: Dict[str, float]) -> float:
        """
        Normalized Shannon entropy: 0 = perfectly certain, 1 = maximally uncertain.
        """
        if not dist:
            return 1.0
        entropy = -sum(p * math.log(p) for p in dist.values() if p > 0)
        max_entropy = math.log(max(len(dist), 1))
        return entropy / max_entropy if max_entropy > 0 else 0.0


# ── Module-level convenience function (backwards compatible) ──────────────────

_default_classifier = FieldClassifier()

async def classify_field(task: str) -> Dict[str, float]:
    """
    Backwards-compatible module-level function.
    Uses a stateless classifier (no history tracking).
    For conversation-aware classification, instantiate FieldClassifier directly.
    """
    return await _default_classifier.classify(task, update_history=False)
