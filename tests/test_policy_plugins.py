"""
tests/test_policy_plugins.py — Tests for wired arbiter_policy and
promotion_policy plugins.

arbiter_policy:
  Plugin replaces built-in LLM arbitration call in _arbitrate()
  Returns {"winner": "A"|"B"|"both_wrong", "reason": str, ...}
  Falls back to built-in when plugin raises
  _custom_arbiter_policy slot exists and defaults to None

promotion_policy (simple — should_promote()):
  Plugin replaces u_delta >= threshold in _evaluate_green()
  Receives (specialist, blue_mean_u, green_mean_u, n_queries, metadata)
  Falls back to built-in when plugin raises

FullPromotionPolicyPlugin (full context — should_promote_full()):
  Plugin receives complete context dict
  Context includes: u_delta, mean_delta, n_queries, min_queries, threshold,
    shadow_scores, shadow_std_delta, regression_result, dry, source
  should_promote_full() takes priority over should_promote()
  Falls back to should_promote() if should_promote_full() raises
  Falls back to built-in if both raise
  Non-linear promotion functions: CI-based, adaptive threshold, multi-factor

Protocol:
  FullPromotionPolicyPlugin is runtime_checkable
  should_promote_full() required method
  Registered in _PROTOCOL_MAP under "full_promotion_policy"
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_router(tmp_path: Path):
    from aua.config import load_config
    from aua.router import Router

    cfg = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
  - name: math
    model: fake/math
    port: 9002
    field: mathematics
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
  specialist_timeout: 5.0
  fanout_threshold: 0.30
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(cfg)
    return Router.from_config(load_config(p))


# ── FullPromotionPolicyPlugin Protocol ───────────────────────────────────────


class TestFullPromotionPolicyProtocol:
    def test_is_runtime_checkable(self) -> None:
        from aua.plugins.interfaces import FullPromotionPolicyPlugin

        class Good:
            def should_promote_full(self, context):
                return True

        assert isinstance(Good(), FullPromotionPolicyPlugin)

    def test_missing_method_fails(self) -> None:
        from aua.plugins.interfaces import FullPromotionPolicyPlugin

        assert not isinstance(object(), FullPromotionPolicyPlugin)

    def test_in_protocol_map(self) -> None:
        from aua.plugins.registry import _PROTOCOL_MAP

        assert "full_promotion_policy" in _PROTOCOL_MAP

    def test_in_known_plugin_kinds(self) -> None:
        from aua.config import _KNOWN_PLUGIN_KINDS

        assert "full_promotion_policy" in _KNOWN_PLUGIN_KINDS
        assert "promotion_policy" in _KNOWN_PLUGIN_KINDS
        assert "arbiter_policy" in _KNOWN_PLUGIN_KINDS


# ── Router slots ─────────────────────────────────────────────────────────────


class TestRouterSlots:
    def test_arbiter_policy_slot_exists(self, tmp_path: Path) -> None:
        r = _make_router(tmp_path)
        assert hasattr(r, "_custom_arbiter_policy")
        assert r._custom_arbiter_policy is None

    def test_promotion_policy_slot_exists(self, tmp_path: Path) -> None:
        r = _make_router(tmp_path)
        assert hasattr(r, "_custom_promotion_policy")
        assert r._custom_promotion_policy is None


# ── arbiter_policy wiring ─────────────────────────────────────────────────────


class TestArbiterPolicyPlugin:
    def test_plugin_replaces_llm_call(self, tmp_path: Path) -> None:
        """When arbiter_policy is set, _arbitrate() calls the plugin, not the LLM."""
        router = _make_router(tmp_path)
        llm_calls = []

        class DeterministicArbiter:
            def arbitrate(self, subject, domain, output_a, output_b, metadata):
                return {
                    "winner": "B",
                    "reason": "B is more complete",
                    "external_response": "A should include time complexity",
                    "case": "case_1",
                    "correct_a": True,
                    "correct_b": False,
                }

        router._custom_arbiter_policy = DeterministicArbiter()

        # Patch _call to track LLM usage
        async def tracked_call(url, *a, **kw):
            llm_calls.append(url)
            return ("response", 0.8)

        router._call = tracked_call

        spec_a = router._config.specialists[0]
        spec_b = router._config.specialists[1]

        verdict_text, winner = asyncio.run(
            router._arbitrate("test query", spec_a, "response A", spec_b, "response B")
        )

        # Plugin was used — no LLM arbiter call
        arbiter_calls = [c for c in llm_calls if "9003" in str(c) or "arbiter" in str(c).lower()]
        assert len(arbiter_calls) == 0
        # Winner should be spec_b.field (plugin returned "B")
        assert winner == spec_b.field
        assert "VERDICT: B" in verdict_text
        assert "B is more complete" in verdict_text

    def test_plugin_a_wins(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path)

        class AlwaysAWins:
            def arbitrate(self, subject, domain, output_a, output_b, metadata):
                return {"winner": "A", "reason": "A is correct", "external_response": ""}

        router._custom_arbiter_policy = AlwaysAWins()
        router._call = MagicMock(return_value=("unused", 0.8))

        spec_a = router._config.specialists[0]
        spec_b = router._config.specialists[1]

        _, winner = asyncio.run(router._arbitrate("query", spec_a, "text A", spec_b, "text B"))
        assert winner == spec_a.field

    def test_plugin_both_wrong(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path)

        class BothWrongArbiter:
            def arbitrate(self, subject, domain, output_a, output_b, metadata):
                return {"winner": "BOTH_WRONG", "reason": "both incorrect", "external_response": ""}

        router._custom_arbiter_policy = BothWrongArbiter()
        router._call = MagicMock(return_value=("unused", 0.8))

        spec_a = router._config.specialists[0]
        spec_b = router._config.specialists[1]

        _, winner = asyncio.run(router._arbitrate("query", spec_a, "A", spec_b, "B"))
        assert winner == "both_wrong"

    def test_plugin_metadata_contains_domain_info(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path)
        received_meta = []

        class MetaCapture:
            def arbitrate(self, subject, domain, output_a, output_b, metadata):
                received_meta.append(metadata)
                return {"winner": "A", "reason": "", "external_response": ""}

        router._custom_arbiter_policy = MetaCapture()
        router._call = MagicMock(return_value=("unused", 0.8))

        spec_a = router._config.specialists[0]
        spec_b = router._config.specialists[1]

        asyncio.run(router._arbitrate("query", spec_a, "A", spec_b, "B"))

        assert len(received_meta) == 1
        meta = received_meta[0]
        assert "domain_a" in meta
        assert "domain_b" in meta
        assert meta["specialist_a"] == "swe"
        assert meta["specialist_b"] == "math"

    def test_plugin_fallback_on_exception(self, tmp_path: Path) -> None:
        """When plugin raises, falls back to built-in LLM call."""
        router = _make_router(tmp_path)
        llm_called = []

        class CrashingArbiter:
            def arbitrate(self, *a, **kw):
                raise RuntimeError("arbiter exploded")

        router._custom_arbiter_policy = CrashingArbiter()

        async def fake_call(url, prompt, *a, **kw):
            llm_called.append(True)
            return ("VERDICT: A\nREASON: A is better\nCORRECTION: none", 0.8)

        router._call = fake_call

        spec_a = router._config.specialists[0]
        spec_b = router._config.specialists[1]

        _, winner = asyncio.run(router._arbitrate("query", spec_a, "A", spec_b, "B"))
        # Built-in LLM path was used as fallback
        assert len(llm_called) >= 1
        assert winner == spec_a.field  # "VERDICT: A" → spec_a wins


# ── promotion_policy — simple mode (should_promote) ──────────────────────────


class TestSimplePromotionPolicy:
    def test_plugin_blocks_promotion(self, tmp_path: Path) -> None:
        """A plugin that always returns False blocks promotion regardless of delta."""
        router = _make_router(tmp_path)

        class NeverPromote:
            def should_promote(self, specialist, blue_mean_u, green_mean_u, n_queries, metadata):
                return False  # never promote, even with positive delta

        router._custom_promotion_policy = NeverPromote()

        from aua.endpoints import DeployGreenRequest

        req = DeployGreenRequest(
            specialist="swe",
            green_model="./green",
            green_endpoint="http://localhost:9011/v1/chat/completions",
        )

        # Mock _call to return high green scores
        async def good_call(url, *a, **kw):
            return ("def binary_search(): pass", 0.95)

        router._call = good_call

        result = asyncio.run(router._evaluate_green(req))
        # Plugin blocked it despite positive delta
        assert result.promoted is False

    def test_plugin_forces_promotion(self, tmp_path: Path) -> None:
        """A plugin that always returns True promotes even with negative delta."""
        router = _make_router(tmp_path)

        class AlwaysPromote:
            def should_promote(self, specialist, blue_mean_u, green_mean_u, n_queries, metadata):
                return True

        router._custom_promotion_policy = AlwaysPromote()

        from aua.endpoints import DeployGreenRequest

        req = DeployGreenRequest(
            specialist="swe",
            green_model="./green",
            green_endpoint="http://localhost:9011/v1/chat/completions",
        )

        call_count = [0]

        async def alternating_call(url, *a, **kw):
            # Blue gets 0.7, green gets 0.6 (negative delta)
            call_count[0] += 1
            score = 0.7 if call_count[0] % 2 == 1 else 0.6
            return (f"response {call_count[0]}", score)

        router._call = alternating_call

        result = asyncio.run(router._evaluate_green(req))
        assert result.promoted is True

    def test_simple_plugin_fallback_on_exception(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path)

        class CrashingPromoter:
            def should_promote(self, *a, **kw):
                raise RuntimeError("promoter crashed")

        router._custom_promotion_policy = CrashingPromoter()

        from aua.endpoints import DeployGreenRequest

        req = DeployGreenRequest(
            specialist="swe",
            green_model="./green",
            green_endpoint="http://localhost:9011/v1/chat/completions",
        )

        async def good_call(url, *a, **kw):
            return ("response", 0.85)

        router._call = good_call

        # Must not raise — falls back to built-in u_delta >= threshold
        result = asyncio.run(router._evaluate_green(req))
        assert result is not None


# ── FullPromotionPolicyPlugin — full context mode ─────────────────────────────


class TestFullPromotionPolicyPlugin:
    def test_should_promote_full_receives_full_context(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path)
        received_context = []

        class ContextCapture:
            def should_promote(self, specialist, blue_mean_u, green_mean_u, n_queries, metadata):
                return True

            def should_promote_full(self, context):
                received_context.append(context.copy())
                return True

        router._custom_promotion_policy = ContextCapture()

        from aua.endpoints import DeployGreenRequest

        req = DeployGreenRequest(
            specialist="swe",
            green_model="./green",
            green_endpoint="http://localhost:9011/v1/chat/completions",
        )
        router._call = MagicMock(return_value=("response", 0.8))

        asyncio.run(router._evaluate_green(req))

        assert len(received_context) == 1
        ctx = received_context[0]
        # Verify all required context keys are present
        assert "specialist" in ctx
        assert "blue_u" in ctx
        assert "green_u" in ctx
        assert "u_delta" in ctx
        assert "mean_delta" in ctx
        assert "n_queries" in ctx
        assert "min_queries" in ctx
        assert "threshold" in ctx
        assert "shadow_scores" in ctx
        assert "shadow_std_delta" in ctx
        assert "regression_result" in ctx
        assert "dry" in ctx
        assert "source" in ctx
        assert "specialist_config" in ctx
        assert "bg_config" in ctx
        assert ctx["specialist"] == "swe"

    def test_should_promote_full_takes_priority_over_should_promote(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path)
        calls = []

        class PriorityTest:
            def should_promote(self, specialist, blue_mean_u, green_mean_u, n_queries, metadata):
                calls.append("should_promote")
                return False

            def should_promote_full(self, context):
                calls.append("should_promote_full")
                return True  # opposite of should_promote

        router._custom_promotion_policy = PriorityTest()
        router._call = MagicMock(return_value=("response", 0.8))

        from aua.endpoints import DeployGreenRequest

        req = DeployGreenRequest(
            specialist="swe",
            green_model="./green",
            green_endpoint="http://localhost:9011/v1/chat/completions",
        )

        result = asyncio.run(router._evaluate_green(req))
        # should_promote_full was called, not should_promote
        assert "should_promote_full" in calls
        assert "should_promote" not in calls
        assert result.promoted is True

    def test_should_promote_full_fallback_to_should_promote(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path)
        calls = []

        class FallbackTest:
            def should_promote(self, specialist, blue_mean_u, green_mean_u, n_queries, metadata):
                calls.append("should_promote")
                return True

            def should_promote_full(self, context):
                raise RuntimeError("full context not supported")

        router._custom_promotion_policy = FallbackTest()
        router._call = MagicMock(return_value=("response", 0.8))

        from aua.endpoints import DeployGreenRequest

        req = DeployGreenRequest(
            specialist="swe",
            green_model="./green",
            green_endpoint="http://localhost:9011/v1/chat/completions",
        )

        result = asyncio.run(router._evaluate_green(req))
        assert "should_promote" in calls
        assert result.promoted is True

    def test_ci_based_promotion(self) -> None:
        """Confidence-interval gate: mean_delta must exceed 2× std_dev."""

        class CIBasedPromoter:
            def should_promote(self, specialist, blue_mean_u, green_mean_u, n_queries, metadata):
                return True

            def should_promote_full(self, context):
                std = context["shadow_std_delta"]
                mean = context["mean_delta"]
                if std == 0:
                    return mean > 0
                return mean > 2 * std

        p = CIBasedPromoter()
        # Low std, high mean → promote
        assert p.should_promote_full(
            {"mean_delta": 0.10, "shadow_std_delta": 0.02, "n_queries": 50, "threshold": 0.025}
        )
        # High std relative to mean → don't promote
        assert not p.should_promote_full(
            {"mean_delta": 0.03, "shadow_std_delta": 0.05, "n_queries": 10, "threshold": 0.025}
        )

    def test_adaptive_threshold_promotion(self) -> None:
        """Sample-size adaptive: require larger delta when n is small."""

        class AdaptivePromoter:
            def should_promote(self, *a, **kw):
                return False

            def should_promote_full(self, context):
                n = context["n_queries"]
                adaptive_threshold = 0.025 + 0.5 / max(n, 1)
                return context["mean_delta"] >= adaptive_threshold

        p = AdaptivePromoter()
        # n=5: threshold = 0.025 + 0.1 = 0.125 — large delta needed
        assert not p.should_promote_full({"mean_delta": 0.05, "n_queries": 5, "threshold": 0.025})
        # n=100: threshold = 0.025 + 0.005 = 0.030 — small delta sufficient
        assert p.should_promote_full({"mean_delta": 0.04, "n_queries": 100, "threshold": 0.025})

    def test_multi_factor_gate(self) -> None:
        """Multi-factor: regression + delta + minimum n must all pass."""

        class MultiFactorPromoter:
            def should_promote(self, *a, **kw):
                return False

            def should_promote_full(self, context):
                if context.get("regression_result") and context["regression_result"].get(
                    "regressed"
                ):
                    return False
                if context["n_queries"] < context["min_queries"]:
                    return False
                return context["mean_delta"] >= context["threshold"]

        p = MultiFactorPromoter()

        # All conditions met
        assert p.should_promote_full(
            {
                "mean_delta": 0.05,
                "n_queries": 60,
                "min_queries": 50,
                "threshold": 0.025,
                "regression_result": None,
            }
        )
        # Regression detected → block
        assert not p.should_promote_full(
            {
                "mean_delta": 0.05,
                "n_queries": 60,
                "min_queries": 50,
                "threshold": 0.025,
                "regression_result": {"regressed": True, "verdict": "REGRESSION"},
            }
        )
        # Too few queries → block
        assert not p.should_promote_full(
            {
                "mean_delta": 0.10,
                "n_queries": 20,
                "min_queries": 50,
                "threshold": 0.025,
                "regression_result": None,
            }
        )
        # Delta too small → block
        assert not p.should_promote_full(
            {
                "mean_delta": 0.01,
                "n_queries": 60,
                "min_queries": 50,
                "threshold": 0.025,
                "regression_result": None,
            }
        )

    def test_cobb_douglas_style_promotion(self) -> None:
        """Geometric mean of blue_u ratio and n_ratio must exceed threshold."""

        class CobbDouglasPromoter:
            def should_promote(self, *a, **kw):
                return False

            def should_promote_full(self, context):
                # Combine n-sufficiency and delta using geometric mean
                n_ratio = min(1.0, context["n_queries"] / context["min_queries"])
                delta_ratio = context["mean_delta"] / max(context["threshold"], 0.001)
                combined = (n_ratio * delta_ratio) ** 0.5  # geometric mean
                return combined >= 1.0

        p = CobbDouglasPromoter()
        # n=50/50, delta=0.025/0.025 → both ratios=1.0 → combined=1.0 → promote
        assert p.should_promote_full(
            {"mean_delta": 0.025, "n_queries": 50, "min_queries": 50, "threshold": 0.025}
        )
        # n=25/50=0.5, delta=0.10/0.025=4 → geometric mean = sqrt(2.0)=1.41 → promote
        assert p.should_promote_full(
            {"mean_delta": 0.10, "n_queries": 25, "min_queries": 50, "threshold": 0.025}
        )
        # n=10/50=0.2, delta=0.05/0.025=2 → geometric mean = sqrt(0.4)=0.63 → block
        assert not p.should_promote_full(
            {"mean_delta": 0.05, "n_queries": 10, "min_queries": 50, "threshold": 0.025}
        )
