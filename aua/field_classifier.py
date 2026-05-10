"""
Field classifier: determines domain distribution of a given task.

v0.2 — Three robustness mechanisms added:
  1. High-stakes floor: dangerous fields can't be diluted below 0.15
  2. Sliding window EMA: tracks field drift across conversation turns
  3. Entropy-based conservative fallback: high ambiguity → tighter bounds

Usage:
    classifier = FieldClassifier()
    distribution = classifier.classify(task)           # single turn
    distribution = classifier.classify(task, update_history=True)  # conversation-aware
"""

import json
import math
import re
try:
    import httpx
except ImportError:
    httpx = None
from typing import Dict, List, Optional
from aua.config import FIELD_CONFIGS, FieldConfig, get_effective_config


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

HIGH_STAKES_FIELDS = {"surgery", "aviation", "law"}
MIN_HIGH_STAKES_PROB = 0.15
HISTORY_EMA_ALPHA = 0.4
ENTROPY_CONSERVATIVE_THRESHOLD = 0.7


class FieldClassifier:
    """
    Stateful classifier that maintains conversation field history
    and applies robustness mechanisms on top of raw LLM classification.
    """

    def __init__(self):
        self.turn_history: List[Dict[str, float]] = []

    def classify(self, task: str, update_history: bool = True) -> Dict[str, float]:
        """Synchronous classify — uses keyword-based fallback."""
        raw = self._keyword_fallback(task)
        floored = self._enforce_high_stakes_floor(raw)
        blended = self._apply_history_ema(floored) if self.turn_history else floored
        hardened = self._apply_entropy_fallback(blended)
        if update_history:
            self.turn_history.append(hardened)
        return hardened

    async def classify_async(self, task: str, update_history: bool = True) -> Dict[str, float]:
        """Async classify — calls LLM classifier, falls back to keyword on error."""
        raw = await self._call_classifier(task)
        floored = self._enforce_high_stakes_floor(raw)
        blended = self._apply_history_ema(floored) if self.turn_history else floored
        hardened = self._apply_entropy_fallback(blended)
        if update_history:
            self.turn_history.append(hardened)
        return hardened

    def get_effective_config(self, distribution: Dict[str, float]) -> FieldConfig:
        base_config = get_effective_config(distribution)
        entropy_ratio = self._entropy_ratio(distribution)
        if entropy_ratio > ENTROPY_CONSERVATIVE_THRESHOLD:
            most_conservative = max(
                distribution,
                key=lambda f: FIELD_CONFIGS.get(f, FIELD_CONFIGS["general"]).c_min
            )
            conservative_cfg = FIELD_CONFIGS.get(most_conservative, FIELD_CONFIGS["general"])
            lerp_t = entropy_ratio
            base_config.c_min = base_config.c_min + lerp_t * (conservative_cfg.c_min - base_config.c_min)
            base_config.e_min = base_config.e_min + lerp_t * (conservative_cfg.e_min - base_config.e_min)
        return base_config

    def reset_history(self):
        self.turn_history = []

    # ── Private ───────────────────────────────────────────────────────────────

    async def _call_classifier(self, task: str) -> Dict[str, float]:
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
        task_lower = task.lower()
        scores: Dict[str, float] = {
            "software_engineering": 0.0, "medicine": 0.0, "law": 0.0,
            "mathematics": 0.0, "physics": 0.0, "chemistry": 0.0,
            "finance": 0.0, "creative_writing": 0.0,
            "general_knowledge": 0.0, "surgery": 0.0, "aviation": 0.0,
        }
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
        if sum(scores.values()) == 0:
            scores["software_engineering"] = 1.0
        total = sum(scores.values())
        dist = {k: v / total for k, v in scores.items() if v > 0}
        return dist or {"software_engineering": 1.0}

    def _enforce_high_stakes_floor(self, dist: Dict[str, float]) -> Dict[str, float]:
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
        all_fields = set(current.keys())
        for h in self.turn_history:
            all_fields.update(h.keys())
        ema: Dict[str, float] = {}
        for turn in self.turn_history + [current]:
            for field in all_fields:
                prev = ema.get(field, 0.0)
                ema[field] = (1 - HISTORY_EMA_ALPHA) * prev + HISTORY_EMA_ALPHA * turn.get(field, 0.0)
        total = sum(ema.values())
        if total == 0:
            return current
        return {k: v / total for k, v in ema.items() if v / total > 0.02}

    def _apply_entropy_fallback(self, dist: Dict[str, float]) -> Dict[str, float]:
        ratio = self._entropy_ratio(dist)
        if ratio > ENTROPY_CONSERVATIVE_THRESHOLD:
            most_conservative = max(
                dist, key=lambda f: FIELD_CONFIGS.get(f, FIELD_CONFIGS["general"]).c_min
            )
            shift = ratio * 0.15
            result = {k: v * (1 - shift) for k, v in dist.items()}
            result[most_conservative] = result.get(most_conservative, 0.0) + shift
            total = sum(result.values())
            dist = {k: v / total for k, v in result.items()}
        return dist

    def _entropy_ratio(self, dist: Dict[str, float]) -> float:
        if not dist:
            return 1.0
        entropy = -sum(p * math.log(p) for p in dist.values() if p > 0)
        max_entropy = math.log(max(len(dist), 1))
        return entropy / max_entropy if max_entropy > 0 else 0.0


_default_classifier = FieldClassifier()

async def classify_field(task: str) -> Dict[str, float]:
    """Module-level convenience function (backwards compatible, stateless)."""
    return await _default_classifier.classify_async(task, update_history=False)
