"""
aua/guard.py — Assertions engine for AUA Framework.

Assertions are user-defined checks that run against specialist responses
before they are returned to the caller. They implement the "Policy as
Curriculum" pattern: blocking bad output in real-time (Layer 1), feeding
reputation signals to the routing system (Layer 2), and marking gold-standard
sessions for DPO export (Layer 3).

Three assertion levels:

    BLOCKING — Fails → error injected back into prompt → specialist retried
               up to max_retries (default 3). If all retries fail, the
               response passes with a U penalty recorded. The user never
               sees a response that violates a blocking assertion unless
               every retry is exhausted.

    SOFT     — Fails → logged to assertion_events, response passes through.
               No U change. Use for soft guardrails you want to track but
               not enforce.

    INFO     — Always passes. When condition is met (returns a message),
               applies a +bonus to the Efficacy (E) score for this session.
               E_final = min(1.0, E_base + Σ individual bonuses).
               Use for positive/incentive assertions that reward gold-standard
               behaviour.

Usage:

    from aua.guard import assertion, AssertionLevel

    # Negative guardrail
    @assertion(name="PythonSyntaxCheck", level=AssertionLevel.BLOCKING)
    def validate_syntax(output: str, context: dict) -> tuple[bool, str | None]:
        try:
            compile(output, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e.msg} at line {e.lineno}"

    # Positive incentive
    @assertion(name="AnalogyBonus", level=AssertionLevel.INFO, bonus=0.10)
    def reward_analogy(output: str, context: dict) -> tuple[bool, str | None]:
        if any(p in output.lower() for p in ["like a", "similar to", "imagine"]):
            return True, "Positive: analogy used for clarity"
        return True, None   # neutral — no bonus

    # Load into a Policy
    from aua.policy import Policy
    policy = Policy(name="SafeCoding", max_total_bonus=0.30)
    policy.add(validate_syntax)
    policy.add(reward_analogy)
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)

# Registry of all decorated assertions — populated by @assertion
_REGISTRY: dict[str, AssertionFn] = {}


class AssertionLevel(str, Enum):
    BLOCKING = "blocking"  # retry on fail
    SOFT = "soft"  # log on fail, pass through
    INFO = "info"  # positive bonus on success signal


@dataclass
class AssertionResult:
    """Result of running a single assertion against a response."""

    assertion_name: str
    level: AssertionLevel
    passed: bool
    message: str | None = None  # failure reason or positive signal text
    bonus_applied: float = 0.0  # > 0 only for INFO assertions that fired
    retries_used: int = 0
    latency_ms: float = 0.0


@dataclass
class PolicyResult:
    """Aggregate result of running all assertions in a Policy."""

    passed: bool  # False only if BLOCKING exhausted
    e_bonus: float  # total E bonus to add (Option B, capped)
    u_penalty: float  # penalty if BLOCKING exhausted
    results: list[AssertionResult] = field(default_factory=list)
    retries_total: int = 0
    gold_standard: bool = False  # True = all INFO fired, no BLOCKING fail


class AssertionFn:
    """
    Wrapper produced by @assertion decorator.

    Callable: call like the original function.
    Carries metadata used by Policy to build pipelines.
    """

    def __init__(
        self,
        fn: Callable,
        name: str,
        level: AssertionLevel,
        bonus: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self.fn = fn
        self.name = name
        self.level = level
        self.bonus = bonus  # only used for INFO
        self.max_retries = max_retries
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__

    def __call__(self, output: str, context: dict) -> tuple[bool, str | None]:
        return self.fn(output, context)

    def __repr__(self) -> str:
        return f"AssertionFn(name={self.name!r}, level={self.level.value})"


def assertion(
    name: str,
    level: AssertionLevel | str = AssertionLevel.SOFT,
    bonus: float = 0.0,
    max_retries: int = 3,
) -> Callable:
    """
    Decorator to register a function as an AUA assertion.

    Args:
        name:        Unique identifier shown in logs, metrics, and DPO export.
        level:       AssertionLevel.BLOCKING | SOFT | INFO  (or string equiv.)
        bonus:       E-score bonus for INFO assertions when condition fires.
                     Ignored for BLOCKING / SOFT.
        max_retries: Maximum retry attempts for BLOCKING assertions.

    Returns a callable with the same signature but registered in the global
    assertion registry.

    Example::

        @assertion(name="NoBannedWords", level=AssertionLevel.BLOCKING)
        def check_banned(output: str, context: dict) -> tuple[bool, str | None]:
            banned = ["synergy", "paradigm shift"]
            found = [w for w in banned if w in output.lower()]
            if found:
                return False, f"Banned words: {found}"
            return True, None
    """
    if isinstance(level, str):
        level = AssertionLevel(level.lower())

    def decorator(fn: Callable) -> AssertionFn:
        wrapper = AssertionFn(fn, name=name, level=level, bonus=bonus, max_retries=max_retries)
        _REGISTRY[name] = wrapper
        log.debug("Registered assertion: %s (level=%s)", name, level.value)
        return wrapper

    return decorator


def load_assertion(import_path: str) -> AssertionFn:
    """
    Load an assertion from 'module.path:function_name'.

    The decorated function must already be registered via @assertion.

    Example::
        fn = load_assertion("mypackage.policies:validate_syntax")
    """
    if ":" not in import_path:
        raise ValueError(f"import_path must be 'module:function', got {import_path!r}")
    module_path, attr = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, attr)
    if not isinstance(fn, AssertionFn):
        raise TypeError(
            f"{import_path!r} is not decorated with @assertion. " f"Got {type(fn).__name__}."
        )
    return fn


def list_assertions() -> list[dict[str, Any]]:
    """Return metadata for all registered assertions."""
    return [
        {
            "name": a.name,
            "level": a.level.value,
            "bonus": a.bonus,
            "max_retries": a.max_retries,
            "doc": (a.__doc__ or "").strip().splitlines()[0] if a.__doc__ else "",
        }
        for a in _REGISTRY.values()
    ]


# ── Built-in assertions ───────────────────────────────────────────────────────


@assertion(name="NoRefusal", level=AssertionLevel.SOFT)
def no_refusal(output: str, context: dict) -> tuple[bool, str | None]:
    """Soft-flags responses that appear to be outright refusals."""
    refusal_phrases = [
        "i cannot",
        "i can't",
        "i am unable",
        "i'm unable",
        "as an ai",
        "as a language model",
    ]
    lower = output.lower()
    found = next((p for p in refusal_phrases if p in lower), None)
    if found:
        return False, f"Possible refusal detected: '{found}'"
    return True, None


@assertion(name="MinLength", level=AssertionLevel.SOFT)
def min_length(output: str, context: dict) -> tuple[bool, str | None]:
    """Soft-flags very short responses (< 20 chars) as likely incomplete."""
    if len(output.strip()) < 20:
        return False, f"Response too short ({len(output.strip())} chars)"
    return True, None


@assertion(name="PythonSyntaxCheck", level=AssertionLevel.BLOCKING)
def python_syntax_check(output: str, context: dict) -> tuple[bool, str | None]:
    """
    Blocks responses containing a Python code block with syntax errors.

    Only activates when the output contains a ```python ... ``` block.
    Does not block non-code responses.
    """
    import ast
    import re

    blocks = re.findall(r"```python(.*?)```", output, re.DOTALL)
    if not blocks:
        return True, None
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError as e:
            return False, f"Python syntax error: {e.msg} at line {e.lineno}"
    return True, None


@assertion(name="AnalogyBonus", level=AssertionLevel.INFO, bonus=0.08)
def analogy_bonus(output: str, context: dict) -> tuple[bool, str | None]:
    """Rewards responses that use analogies to explain concepts."""
    phrases = ["like a", "similar to", "imagine a", "think of it as", "just like", "analogous to"]
    if any(p in output.lower() for p in phrases):
        return True, "Positive: analogy used for clarity"
    return True, None


@assertion(name="ConciseBonus", level=AssertionLevel.INFO, bonus=0.06)
def concise_bonus(output: str, context: dict) -> tuple[bool, str | None]:
    """Rewards responses under 150 words — signal for high information density."""
    if len(output.split()) < 150:
        return True, f"Positive: concise response ({len(output.split())} words)"
    return True, None
