"""Tests for aua/guard.py — assertions engine."""

import pytest

from aua.guard import (
    AssertionFn,
    AssertionLevel,
    analogy_bonus,
    assertion,
    list_assertions,
    min_length,
    no_refusal,
    python_syntax_check,
)
from aua.policy import Policy

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def ctx():
    return {
        "query": "test",
        "session_id": "s1",
        "domain": "software_engineering",
        "field": "software_engineering",
    }


@pytest.fixture
def good_code():
    return "def binary_search(arr, target):\n    return target in arr"


@pytest.fixture
def bad_code():
    return "```python\ndef foo(\n    broken syntax here\n```"


# ── @assertion decorator ──────────────────────────────────────────────────────


def test_assertion_decorator_creates_fn():
    @assertion(name="TestAssertion", level=AssertionLevel.SOFT)
    def my_check(output, context):
        return True, None

    assert isinstance(my_check, AssertionFn)
    assert my_check.name == "TestAssertion"
    assert my_check.level == AssertionLevel.SOFT


def test_assertion_string_level():
    @assertion(name="StringLevel", level="blocking")
    def check(output, context):
        return True, None

    assert check.level == AssertionLevel.BLOCKING


def test_assertion_registered_in_registry():
    from aua.guard import _REGISTRY

    @assertion(name="RegistryTest_unique_xyz", level=AssertionLevel.INFO, bonus=0.05)
    def check(output, context):
        return True, None

    assert "RegistryTest_unique_xyz" in _REGISTRY


def test_assertion_callable(ctx, good_code):
    passed, msg = no_refusal(good_code, ctx)
    assert isinstance(passed, bool)


# ── AssertionLevel values ─────────────────────────────────────────────────────


def test_blocking_level():
    assert AssertionLevel.BLOCKING.value == "blocking"


def test_soft_level():
    assert AssertionLevel.SOFT.value == "soft"


def test_info_level():
    assert AssertionLevel.INFO.value == "info"


# ── Built-in assertions ───────────────────────────────────────────────────────


def test_python_syntax_check_passes_good_code(ctx, good_code):
    passed, msg = python_syntax_check(good_code, ctx)
    assert passed is True
    assert msg is None


def test_python_syntax_check_fails_bad_code(ctx, bad_code):
    passed, msg = python_syntax_check(bad_code, ctx)
    assert passed is False
    assert msg is not None
    assert "syntax" in msg.lower() or "error" in msg.lower()


def test_python_syntax_check_passes_non_code(ctx):
    passed, msg = python_syntax_check("This is just text, no code block.", ctx)
    assert passed is True


def test_analogy_bonus_fires_on_analogy(ctx):
    passed, msg = analogy_bonus("Think of it as a hash table for names.", ctx)
    assert passed is True
    assert msg is not None  # bonus fires
    assert "analogy" in msg.lower() or "positive" in msg.lower()


def test_analogy_bonus_neutral_without_analogy(ctx):
    passed, msg = analogy_bonus("Use a dictionary to store values.", ctx)
    assert passed is True
    assert msg is None  # neutral — no bonus


def test_no_refusal_soft_flags(ctx):
    passed, msg = no_refusal("I cannot help with that request.", ctx)
    assert passed is False
    assert msg is not None


def test_no_refusal_passes_normal(ctx):
    passed, msg = no_refusal("Here's the binary search implementation.", ctx)
    assert passed is True


def test_min_length_soft_flags_short(ctx):
    passed, msg = min_length("Too short.", ctx)
    assert passed is False


def test_min_length_passes_normal(ctx):
    passed, msg = min_length(
        "This is a sufficiently long response that should pass the minimum length check.", ctx
    )
    assert passed is True


def test_list_assertions_returns_list():
    items = list_assertions()
    assert isinstance(items, list)
    assert len(items) > 0
    assert all("name" in a and "level" in a for a in items)


# ── Policy.run() — Option B bonus cap ────────────────────────────────────────


def test_policy_run_info_bonus_applied(ctx):
    @assertion(name="BonusTest1", level=AssertionLevel.INFO, bonus=0.10)
    def bonus_check(output, context):
        return True, "Positive signal"

    policy = Policy(name="BonusPolicy", max_total_bonus=0.30)
    policy.add(bonus_check)
    result = policy.run("Like a tree structure, it branches.", ctx)

    assert result.e_bonus == pytest.approx(0.10, abs=0.001)
    assert result.passed is True


def test_policy_run_multiple_bonuses_sum(ctx):
    @assertion(name="BonusA", level=AssertionLevel.INFO, bonus=0.12)
    def ba(output, context):
        return True, "A fires"

    @assertion(name="BonusB", level=AssertionLevel.INFO, bonus=0.10)
    def bb(output, context):
        return True, "B fires"

    policy = Policy(name="MultiBonus", max_total_bonus=0.30)
    policy.add(ba).add(bb)
    result = policy.run("output text", ctx)

    assert result.e_bonus == pytest.approx(0.22, abs=0.001)


def test_policy_run_bonus_capped_by_max_total(ctx):
    @assertion(name="BigBonus1", level=AssertionLevel.INFO, bonus=0.20)
    def b1(output, context):
        return True, "fires"

    @assertion(name="BigBonus2", level=AssertionLevel.INFO, bonus=0.20)
    def b2(output, context):
        return True, "fires"

    policy = Policy(name="CappedPolicy", max_total_bonus=0.25)
    policy.add(b1).add(b2)
    result = policy.run("output", ctx)

    assert result.e_bonus <= 0.25  # capped at max_total_bonus


def test_policy_run_no_bonus_if_neutral(ctx):
    @assertion(name="NeutralInfo", level=AssertionLevel.INFO, bonus=0.10)
    def neutral(output, context):
        return True, None  # no message = neutral, no bonus

    policy = Policy(name="NeutralPolicy")
    policy.add(neutral)
    result = policy.run("output", ctx)

    assert result.e_bonus == 0.0


def test_policy_run_blocking_pass(ctx):
    @assertion(name="BlockingPass", level=AssertionLevel.BLOCKING)
    def check(output, context):
        return True, None

    policy = Policy(name="BlockingPassPol")
    policy.add(check)
    result = policy.run("fine output", ctx)

    assert result.passed is True
    assert result.u_penalty == 0.0


def test_policy_run_blocking_fail_no_retry_fn(ctx):
    @assertion(name="BlockingFail", level=AssertionLevel.BLOCKING)
    def check(output, context):
        return False, "Failed"

    policy = Policy(name="BlockingFailPol")
    policy.add(check)
    result = policy.run("bad output", ctx, retry_fn=None)

    assert result.passed is False
    assert result.u_penalty > 0.0


def test_policy_run_blocking_retry_succeeds(ctx):
    call_count = {"n": 0}

    @assertion(name="RetrySucceeds", level=AssertionLevel.BLOCKING, max_retries=3)
    def check(output, context):
        call_count["n"] += 1
        if call_count["n"] < 2:
            return False, "Try again"
        return True, None

    policy = Policy(name="RetryPolicy")
    policy.add(check)

    def retry_fn(error):
        return "improved output"

    result = policy.run("initial output", ctx, retry_fn=retry_fn)
    assert result.passed is True
    assert result.retries_total >= 1


def test_policy_gold_standard_flag(ctx):
    @assertion(name="GoldInfo", level=AssertionLevel.INFO, bonus=0.08)
    def info_check(output, context):
        return True, "fires"

    policy = Policy(name="GoldPolicy")
    policy.add(info_check)
    result = policy.run("good output", ctx)

    assert result.gold_standard is True


def test_policy_not_gold_standard_if_blocking_failed(ctx):
    @assertion(name="GoldInfo2", level=AssertionLevel.INFO, bonus=0.08)
    def info_check(output, context):
        return True, "fires"

    @assertion(name="BlockFail2", level=AssertionLevel.BLOCKING)
    def block_check(output, context):
        return False, "Failed"

    policy = Policy(name="NotGoldPolicy")
    policy.add(info_check).add(block_check)
    result = policy.run("bad output", ctx, retry_fn=None)

    assert result.gold_standard is False


def test_policy_chaining():
    policy = Policy(name="ChainTest")
    result = policy.add(no_refusal).add(min_length)
    assert result is policy  # returns self for chaining
    assert len(policy.assertions) == 2


def test_policy_summary():
    policy = Policy(name="SummaryTest", version="2.0", max_total_bonus=0.25)
    policy.add(no_refusal)
    summary = policy.summary()
    assert summary["name"] == "SummaryTest"
    assert summary["version"] == "2.0"
    assert len(summary["assertions"]) == 1
