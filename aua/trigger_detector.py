"""
aua/trigger_detector.py — Implicit correction trigger detection (V-P1.3).

Answers one binary question per user message:
    "Is this a correction/instruction signal?"

The framework ships Layer 1 only (regex keyword match, <1ms, zero deps).
AUA-Veritas additionally runs a spaCy text classifier as Layer 2; that stays
a product concern — plug a classifier in via ``layer2`` if you need one.

Pattern rules carried forward from AUA-Veritas Phase 13:

  * Regex pattern rule: patterns ending in punctuation or space (``actually,``
    ``no,`` ``in fact,``) cannot use a trailing ``\\b`` word boundary — comma
    + space is not a word boundary. CORRECTION_PATTERNS is therefore split
    into two groups: word-terminated (``\\b...\\b``) and punct/space-terminated
    (``\\b`` prefix only).
  * Explicit prefix rule: ``correction: X`` is a preference statement — store
    it regardless of whether a prior AI turn exists. ``is_explicit_prefix()``
    exposes this so the correction handler's early-return can be guarded.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from enum import Enum

log = logging.getLogger("aua.trigger_detector")

# ── Layer 1 — keyword patterns ────────────────────────────────────────────────

CORRECTION_PATTERNS = re.compile(
    r"(?:"
    # Word-terminated patterns — standard \b works fine
    r"\b(?:wrong|incorrect|that'?s not right|that is not right"
    r"|not what i (?:asked|said|meant)"
    r"|going forward|from now on|henceforth"
    r"|we decided|we are not|we're not"
    r"|i prefer|i always want|i want .{0,30} to always"
    r"|use .{0,30} not|instead of|rather than"
    r"|that'?s (?:the )?(?:wrong|incorrect|opposite|backwards)"
    r"|you keep|you misunderstood|let me correct"
    r"|the correct (?:answer|approach|way)"
    r"|specifically said|told you)\b"
    # Patterns followed by punctuation/space — trailing \b fails after comma/space
    r"|\bno[,\s]"
    r"|\balways |\bnever |\bdon'?t |\bavoid |\bstop "
    r"|\bactually[,\s]|\bin fact[,\s]"
    r"|\bremember[,\s]"
    r"|\bcorrection[:\s;,!]+"
    r")",
    re.IGNORECASE,
)

NON_CORRECTION_PATTERNS = re.compile(
    r"^("
    r"(can you |could you |please )?"
    r"(rewrite|summarize|translate|format|convert|refactor|rename|add|remove|sort|clean)"
    r"|what[\s']"
    r"|which\b"
    r"|how (do|does|would|can|should|many|much|long|far|often|old)\b"
    r"|why (is|are|does|do)"
    r"|when (is|are|does|do)"
    r"|where (is|are|does|do)"
    r"|is (it|there|this|that|postgres|sqlite|rust|python|react|redis)\b"
    r"|should (i|we|you)\b"
    r"|thanks?[,.]?$|thank you"
    r"|ok[,.]?$|okay[,.]?$|got it|sounds good|makes sense|understood|noted"
    r"|perfect[,.]?|great[,.]?|excellent[,.]?"
    r"|write (a|an|the)|create (a|an|the)|generate (a|an|the)|implement|build"
    r")",
    re.IGNORECASE,
)

# System-generated error messages must never be treated as the AI turn being
# corrected — they score as correction language ("failed", "unavailable") and
# produce garbage memory entries.
SYSTEM_RESPONSE_PREFIXES = (
    "All selected models are temporarily unavailable",
    "All selected models",
    "No AI models are connected",
    "All specialists are temporarily unavailable",
)


class TriggerResult(Enum):
    CORRECTION = "correction"
    NOT_CORRECTION = "not_correction"
    UNCERTAIN = "uncertain"  # Layer 1 ambiguous → Layer 2 (if plugged in)


def is_explicit_prefix(message: str) -> bool:
    """True when the message starts with the explicit ``correction:`` prefix."""
    return message.strip().lower().startswith("correction:")


def strip_explicit_prefix(message: str) -> str:
    """Return the instruction text after the ``correction:`` prefix."""
    text = message.strip()
    if text.lower().startswith("correction:"):
        return text[len("correction:") :].strip()
    return text


class TriggerDetector:
    """
    Correction trigger detector.

    Layer 1: regex keyword match (always available).
    Layer 2: optional pluggable classifier — callable(text) -> float in [0,1].
             Veritas plugs a spaCy textcat model in here.
    """

    LAYER2_THRESHOLD = 0.5

    def __init__(self, layer2: Callable[[str], float] | None = None) -> None:
        self._layer2 = layer2
        self._last_score = 0.0

    @property
    def last_score(self) -> float:
        return self._last_score

    def detect_layer1(self, message: str) -> TriggerResult:
        """Regex-only detection. Returns UNCERTAIN when no pattern matches."""
        text = message.strip()
        if not text:
            return TriggerResult.NOT_CORRECTION
        if is_explicit_prefix(text):
            return TriggerResult.CORRECTION
        if NON_CORRECTION_PATTERNS.match(text):
            return TriggerResult.NOT_CORRECTION
        if CORRECTION_PATTERNS.search(text):
            return TriggerResult.CORRECTION
        return TriggerResult.UNCERTAIN

    def detect(self, message: str) -> bool:
        """
        Detect whether a user message is a correction/instruction signal.

        Layer 1 decides where it can; ambiguous cases go to Layer 2 when one
        is plugged in, otherwise default to False (conservative).
        """
        result = self.detect_layer1(message)
        if result is TriggerResult.CORRECTION:
            self._last_score = 1.0
            return True
        if result is TriggerResult.NOT_CORRECTION:
            self._last_score = 0.0
            return False
        if self._layer2 is not None:
            try:
                self._last_score = float(self._layer2(message))
                return self._last_score >= self.LAYER2_THRESHOLD
            except Exception as e:  # noqa: BLE001
                log.debug("Layer 2 classifier failed: %s", e)
        self._last_score = 0.0
        return False
