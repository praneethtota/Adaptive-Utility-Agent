"""
aua/policy.py — Policy system for AUA Framework.

A Policy is a named, versioned bundle of assertions that defines:

  1. Guardrails  — BLOCKING/SOFT assertions that prevent bad output
  2. Incentives  — INFO assertions that reward gold-standard behaviour
  3. Utility overrides — field weight shifts when this policy is active

Policies are the mechanism for "teaching the framework what good looks like."
Over time, a policy creates gravity:

  Layer 1  — Bad output blocked and retried in real-time
  Layer 2  — Specialists that fail assertions accumulate lower U scores
             and don't get promoted via blue-green
  Layer 3  — Sessions where every INFO assertion fired become your
             highest-quality DPO training data when you run
             ``aua calibrate --layer 3``

Define in Python::

    from aua.policy import Policy
    from aua.guard import assertion, AssertionLevel

    @assertion(name="NoBannedWords", level=AssertionLevel.BLOCKING)
    def check_banned(output, context):
        ...

    policy = Policy(name="BrandVoice", max_total_bonus=0.30)
    policy.add(check_banned)
    policy.apply(config)   # sets as active policy in config

Or in YAML (``policies/brand_voice.yaml``)::

    name: BrandVoice
    version: "1.0"
    max_retries: 3
    max_total_bonus: 0.30
    assertions:
      - import_path: mypackage.policies:check_banned
        level: blocking
      - import_path: mypackage.policies:reward_analogy
        level: info
        bonus: 0.10
    utility_overrides:
      w_k: 0.35   # raise curiosity weight when this policy is active

Apply via CLI::

    aua policy apply policies/brand_voice.yaml
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aua.guard import AssertionFn, AssertionLevel, AssertionResult, PolicyResult, load_assertion

log = logging.getLogger(__name__)

# Default max retries for BLOCKING assertions if not set on the assertion itself
_DEFAULT_MAX_RETRIES = 3

# Hard ceiling on total E bonus regardless of how many INFO assertions fire
_HARD_BONUS_CEILING = 0.50


@dataclass
class Policy:
    """
    Named bundle of assertions with utility weight overrides.

    Args:
        name:             Human-readable identifier shown in logs and metrics.
        version:          Semver string for tracking policy evolution.
        max_retries:      Default max retries for BLOCKING assertions.
        max_total_bonus:  Cap on total E bonus from INFO assertions (Option B).
                          Each assertion adds its declared bonus independently;
                          the sum is capped here. Default: 0.30.
        utility_overrides: Optional dict of field weight overrides applied when
                          this policy is active, e.g. {"w_k": 0.35}.

    Example::

        policy = Policy(name="SafeCoding", max_total_bonus=0.30)
        policy.add(validate_syntax)       # BLOCKING
        policy.add(reward_analogy)        # INFO, bonus=0.10
    """

    name: str
    version: str = "1.0"
    max_retries: int = _DEFAULT_MAX_RETRIES
    max_total_bonus: float = 0.30
    utility_overrides: dict[str, float] = field(default_factory=dict)
    _assertions: list[AssertionFn] = field(default_factory=list, repr=False)

    def add(self, fn: AssertionFn, **kwargs: Any) -> Policy:
        """
        Add an assertion to this policy.

        Accepts an AssertionFn produced by @assertion.
        Returns self for chaining: policy.add(fn1).add(fn2)

        Keyword overrides (level, bonus, max_retries) take precedence over
        the values set on the decorator.
        """
        if not isinstance(fn, AssertionFn):
            raise TypeError(
                f"Expected an AssertionFn decorated with @assertion, got {type(fn).__name__}. "
                "Did you forget the @assertion decorator?"
            )
        if kwargs:
            # Clone with overrides
            import copy

            fn = copy.copy(fn)
            for k, v in kwargs.items():
                setattr(fn, k, v)
        self._assertions.append(fn)
        return self

    def run(
        self,
        output: str,
        context: dict[str, Any],
        retry_fn: Any | None = None,
    ) -> PolicyResult:
        """
        Run all assertions against a response.

        For BLOCKING assertions that fail, calls ``retry_fn(error_message)``
        to get a new response, up to assertion.max_retries times.

        Args:
            output:    The specialist response text.
            context:   Request metadata dict (query, session_id, domain, ...).
            retry_fn:  Sync callable(error_message: str) -> str | None.
                       If None, BLOCKING failures are recorded but not retried.

        Returns:
            PolicyResult with aggregate pass/fail, E bonus, and per-assertion
            results.
        """
        results: list[AssertionResult] = []
        blocking_exhausted = False
        total_bonus = 0.0
        all_info_fired = True  # tracks gold-standard status
        retries_total = 0

        for fn in self._assertions:
            t0 = time.time()
            max_retries = fn.max_retries if fn.level == AssertionLevel.BLOCKING else 1
            retries = 0
            current_output = output

            while True:
                passed, message = fn(current_output, context)
                retries_used = retries

                if fn.level == AssertionLevel.BLOCKING and not passed:
                    if retries < max_retries and retry_fn is not None:
                        retries += 1
                        retries_total += 1
                        error_injection = (
                            f"[ASSERTION FAILED: {fn.name}] {message or 'Retry required.'} "
                            "Please correct this in your response."
                        )
                        new_output = retry_fn(error_injection)
                        if new_output:
                            current_output = new_output
                        continue
                    else:
                        # All retries exhausted
                        blocking_exhausted = True
                        break
                break  # passed or non-blocking

            bonus_applied = 0.0
            if fn.level == AssertionLevel.INFO:
                if passed and message:
                    # INFO fires bonus only when condition is met (message not None)
                    bonus_applied = fn.bonus
                    total_bonus += bonus_applied
                else:
                    all_info_fired = False  # condition not met — not gold standard
            else:
                if not passed:
                    all_info_fired = False

            results.append(
                AssertionResult(
                    assertion_name=fn.name,
                    level=fn.level,
                    passed=passed,
                    message=message,
                    bonus_applied=bonus_applied,
                    retries_used=retries_used,
                    latency_ms=round((time.time() - t0) * 1000, 2),
                )
            )

        # Apply caps
        capped_bonus = min(total_bonus, self.max_total_bonus, _HARD_BONUS_CEILING)
        u_penalty = 0.15 if blocking_exhausted else 0.0

        overall_passed = not blocking_exhausted
        gold_standard = (
            overall_passed
            and all_info_fired
            and len([f for f in self._assertions if f.level == AssertionLevel.INFO]) > 0
        )

        return PolicyResult(
            passed=overall_passed,
            e_bonus=round(capped_bonus, 4),
            u_penalty=round(u_penalty, 4),
            results=results,
            retries_total=retries_total,
            gold_standard=gold_standard,
        )

    @property
    def assertions(self) -> list[AssertionFn]:
        return list(self._assertions)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "max_retries": self.max_retries,
            "max_total_bonus": self.max_total_bonus,
            "utility_overrides": self.utility_overrides,
            "assertions": [
                {
                    "name": a.name,
                    "level": a.level.value,
                    "bonus": a.bonus,
                }
                for a in self._assertions
            ],
        }


# ── YAML loader ───────────────────────────────────────────────────────────────


def load_policy(path: str | Path) -> Policy:
    """
    Load a Policy from a YAML file.

    Expected format::

        name: SafeCodingPolicy
        version: "1.0"
        max_retries: 3
        max_total_bonus: 0.30
        assertions:
          - import_path: mypackage.policies:validate_syntax
          - import_path: mypackage.policies:reward_analogy
            bonus: 0.12          # override decorator default
        utility_overrides:
          w_k: 0.35

    The ``import_path`` must point to a function decorated with @assertion.

    Args:
        path: Path to the YAML policy file.

    Returns:
        Policy instance ready to use.

    Raises:
        FileNotFoundError, ValueError, ImportError on bad config.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Policy YAML must be a mapping, got {type(raw).__name__}")

    name = raw.get("name")
    if not name:
        raise ValueError("Policy YAML must have a 'name' field")

    policy = Policy(
        name=name,
        version=str(raw.get("version", "1.0")),
        max_retries=int(raw.get("max_retries", _DEFAULT_MAX_RETRIES)),
        max_total_bonus=float(raw.get("max_total_bonus", 0.30)),
        utility_overrides=raw.get("utility_overrides") or {},
    )

    for entry in raw.get("assertions", []):
        import_path = entry.get("import_path")
        if not import_path:
            raise ValueError(f"Assertion entry missing 'import_path': {entry}")
        fn = load_assertion(import_path)
        overrides: dict[str, Any] = {}
        if "level" in entry:
            overrides["level"] = AssertionLevel(entry["level"].lower())
        if "bonus" in entry:
            overrides["bonus"] = float(entry["bonus"])
        if "max_retries" in entry:
            overrides["max_retries"] = int(entry["max_retries"])
        policy.add(fn, **overrides)

    log.info("Loaded policy %r (%d assertions)", name, len(policy.assertions))
    return policy


def validate_policy_yaml(path: str | Path) -> list[str]:
    """
    Validate a policy YAML file without importing assertion functions.

    Returns a list of error strings. Empty list means valid.
    """
    path = Path(path)
    errors: list[str] = []

    if not path.exists():
        return [f"File not found: {path}"]

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    if not isinstance(raw, dict):
        return [f"Top-level must be a mapping, got {type(raw).__name__}"]

    if not raw.get("name"):
        errors.append("Missing required field: 'name'")

    for i, entry in enumerate(raw.get("assertions", [])):
        if not entry.get("import_path"):
            errors.append(f"assertions[{i}]: missing 'import_path'")
        if "level" in entry:
            try:
                AssertionLevel(entry["level"].lower())
            except ValueError:
                errors.append(
                    f"assertions[{i}]: invalid level {entry['level']!r}. "
                    f"Must be one of: {[lv.value for lv in AssertionLevel]}"
                )
        if "bonus" in entry:
            try:
                b = float(entry["bonus"])
                if not 0.0 <= b <= 1.0:
                    errors.append(f"assertions[{i}]: bonus must be in [0, 1], got {b}")
            except (TypeError, ValueError):
                errors.append(f"assertions[{i}]: bonus must be a float")

    overrides = raw.get("utility_overrides", {})
    valid_keys = {"w_e", "w_c", "w_k"}
    for k in overrides:
        if k not in valid_keys:
            errors.append(f"utility_overrides: unknown key {k!r}. Valid: {valid_keys}")

    return errors
