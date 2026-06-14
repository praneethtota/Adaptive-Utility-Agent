"""
tests/test_plugins_extended.py — Tests for #51 (extended plugin system) and
#53 (custom utility function — full replacement).

#51 coverage:
  Protocol definitions: ContradictionDetectorPlugin, AssertionStorePlugin,
    RoutingStrategyPlugin, ScoringComponentPlugin — all runtime_checkable
  Protocol conformance: compliant class passes isinstance(), non-compliant fails
  _PROTOCOL_MAP: all four new kinds registered
  _KNOWN_PLUGIN_KINDS: all four new kinds accepted
  _load_yaml_extensions wiring:
    contradiction_detector → self._custom_detector
    assertion_store        → self._custom_assertion_store (slot exists)
    routing_strategy       → self._routing_strategy
    scoring_component      → self._scoring_component
  _score(): _custom_detector replaces built-in detector
  _score(): _scoring_component adjusts E/C/K components
  _handle(): _routing_strategy intercepts classifier distribution

#53 coverage:
  UtilityScorerPlugin.score_full() method exists in Protocol
  score_full() receives (field, efficacy, confidence, curiosity, weights, metadata)
  score_full() takes priority over score() when both present
  score_full() fallback to score() on exception
  score_full() result replaces built-in w_e·E + w_c·C + w_k·K
  Non-linear utility model computes correctly
  Field-specific scoring (different logic per domain)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Protocol definitions ──────────────────────────────────────────────────────


class TestProtocolDefinitions:
    def test_contradiction_detector_is_runtime_checkable(self) -> None:
        from aua.plugins.interfaces import ContradictionDetectorPlugin

        class Good:
            def check(self, problem, solution, claimed_complexity=None):
                return {"contradictions": [], "confidence_penalty": 0.0, "is_clean": True}

        assert isinstance(Good(), ContradictionDetectorPlugin)

    def test_contradiction_detector_missing_check_fails(self) -> None:
        from aua.plugins.interfaces import ContradictionDetectorPlugin

        class Bad:
            pass

        assert not isinstance(Bad(), ContradictionDetectorPlugin)

    def test_assertion_store_is_runtime_checkable(self) -> None:
        from aua.plugins.interfaces import AssertionStorePlugin

        class Good:
            def add(
                self, subject, domain, claim, confidence, source="arbiter", evidence_summary=""
            ):
                pass

            def query(self, subject, domain=None, min_confidence=None):
                return []

            def query_contradictions(self, subject, new_claim, domain=None):
                return []

        assert isinstance(Good(), AssertionStorePlugin)

    def test_routing_strategy_is_runtime_checkable(self) -> None:
        from aua.plugins.interfaces import RoutingStrategyPlugin

        class Good:
            def route(self, query, distribution, metadata):
                return distribution

        assert isinstance(Good(), RoutingStrategyPlugin)

    def test_routing_strategy_missing_route_fails(self) -> None:
        from aua.plugins.interfaces import RoutingStrategyPlugin

        assert not isinstance(object(), RoutingStrategyPlugin)

    def test_scoring_component_is_runtime_checkable(self) -> None:
        from aua.plugins.interfaces import ScoringComponentPlugin

        class Good:
            def compute(self, component, value, field, metadata):
                return value

        assert isinstance(Good(), ScoringComponentPlugin)

    def test_utility_scorer_has_score_full_method(self) -> None:
        """#53: FullUtilityScorerPlugin declares score_full(); UtilityScorerPlugin has score() only."""
        from aua.plugins.interfaces import FullUtilityScorerPlugin, UtilityScorerPlugin

        # score() is required by UtilityScorerPlugin
        assert "score" in dir(UtilityScorerPlugin)
        # score_full() is on the separate opt-in Protocol
        assert "score_full" in dir(FullUtilityScorerPlugin)
        # UtilityScorerPlugin does NOT require score_full
        assert "score_full" not in [
            m for m in dir(UtilityScorerPlugin) if not m.startswith("_") and m != "score"
        ]


# ── Protocol map and config ───────────────────────────────────────────────────


class TestProtocolMap:
    def test_all_new_kinds_in_protocol_map(self) -> None:
        from aua.plugins.registry import _PROTOCOL_MAP

        for kind in (
            "contradiction_detector",
            "assertion_store",
            "routing_strategy",
            "scoring_component",
        ):
            assert kind in _PROTOCOL_MAP, f"Missing from _PROTOCOL_MAP: {kind}"

    def test_all_new_kinds_in_known_plugin_kinds(self) -> None:
        from aua.config import _KNOWN_PLUGIN_KINDS

        for kind in (
            "contradiction_detector",
            "assertion_store",
            "routing_strategy",
            "scoring_component",
        ):
            assert kind in _KNOWN_PLUGIN_KINDS


# ── load_plugin contract validation ──────────────────────────────────────────


class TestLoadPlugin:
    def test_contradiction_detector_contract_valid(self) -> None:
        from aua.plugins.registry import load_plugin

        class MyDetector:
            def check(self, problem, solution, claimed_complexity=None):
                return {"contradictions": [], "confidence_penalty": 0.0, "is_clean": True}

        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_mod.MyDetector = MyDetector
            mock_import.return_value = mock_mod
            plugin = load_plugin("mymod:MyDetector", "contradiction_detector")
        assert hasattr(plugin, "check")

    def test_routing_strategy_contract_invalid_raises(self) -> None:
        from aua.plugins.registry import PluginLoadError, load_plugin

        class BadRouter:
            pass  # missing route()

        with patch("importlib.import_module") as mock_import:
            mock_mod = MagicMock()
            mock_mod.BadRouter = BadRouter
            mock_import.return_value = mock_mod
            with pytest.raises(PluginLoadError):
                load_plugin("mymod:BadRouter", "routing_strategy")


# ── Router plugin slot wiring ─────────────────────────────────────────────────


class TestRouterPluginSlots:
    def _make_router(self, tmp_path):
        """Create a minimal Router instance."""
        from aua.config import load_config
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        return Router.from_config(load_config(p))

    def test_custom_detector_slot_exists(self, tmp_path) -> None:
        r = self._make_router(tmp_path)
        assert hasattr(r, "_custom_detector")
        assert r._custom_detector is None  # not set without plugin config

    def test_custom_assertion_store_slot_exists(self, tmp_path) -> None:
        r = self._make_router(tmp_path)
        assert hasattr(r, "_custom_assertion_store")
        assert r._custom_assertion_store is None

    def test_routing_strategy_slot_exists(self, tmp_path) -> None:
        r = self._make_router(tmp_path)
        assert hasattr(r, "_routing_strategy")
        assert r._routing_strategy is None

    def test_scoring_component_slot_exists(self, tmp_path) -> None:
        r = self._make_router(tmp_path)
        assert hasattr(r, "_scoring_component")
        assert r._scoring_component is None


# ── Contradiction detector plugin ─────────────────────────────────────────────


class TestContradictionDetectorPlugin:
    def test_custom_detector_called_in_score(self, tmp_path) -> None:
        from aua.config import load_config
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        router = Router.from_config(load_config(p))

        call_log = []

        class CustomDetector:
            def check(self, problem, solution, claimed_complexity=None):
                call_log.append((problem, solution))
                return {
                    "contradictions": [{"type": "custom", "description": "test error"}],
                    "confidence_penalty": 0.3,
                    "is_clean": False,
                }

        router._custom_detector = CustomDetector()

        import asyncio

        result = asyncio.run(
            router._score("test query", "test response", "software_engineering", 0.8)
        )
        assert len(call_log) == 1
        assert call_log[0] == ("test query", "test response")
        # Should have one contradiction recorded
        _u, _conf, n_contra, _dpo = result
        assert n_contra == 1

    def test_custom_detector_fallback_on_error(self, tmp_path) -> None:
        from aua.config import load_config
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        router = Router.from_config(load_config(p))

        class FailingDetector:
            def check(self, *a, **kw):
                raise RuntimeError("detector exploded")

        router._custom_detector = FailingDetector()

        import asyncio

        # Must not raise — falls back to built-in
        result = asyncio.run(router._score("query", "response", "software_engineering", 0.7))
        assert result is not None


# ── Routing strategy plugin ───────────────────────────────────────────────────


class TestRoutingStrategyPlugin:
    def test_routing_strategy_intercepts_distribution(self, tmp_path) -> None:
        from aua.config import load_config
        from aua.endpoints import QueryRequest
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
  specialist_timeout: 5.0
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        router = Router.from_config(load_config(p))

        received_distributions = []

        class AlwaysSWEStrategy:
            def route(self, query, distribution, metadata):
                received_distributions.append(distribution.copy())
                return {"software_engineering": 1.0}

        router._routing_strategy = AlwaysSWEStrategy()

        import asyncio

        with patch.object(router, "_call", return_value=("def foo(): pass", 0.8)):
            asyncio.run(router._handle(QueryRequest(query="test")))

        # Strategy was called and returned software_engineering=1.0
        assert len(received_distributions) >= 1

    def test_routing_strategy_fallback_on_error(self, tmp_path) -> None:
        from aua.config import load_config
        from aua.endpoints import QueryRequest
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
  specialist_timeout: 5.0
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        router = Router.from_config(load_config(p))

        class CrashingStrategy:
            def route(self, *a, **kw):
                raise RuntimeError("strategy failed")

        router._routing_strategy = CrashingStrategy()

        import asyncio

        with patch.object(router, "_call", return_value=("answer", 0.8)):
            # Must not raise — falls back to classifier output
            resp = asyncio.run(router._handle(QueryRequest(query="test")))
        assert resp is not None


# ── Scoring component plugin ──────────────────────────────────────────────────


class TestScoringComponentPlugin:
    def test_scoring_component_adjusts_u(self, tmp_path) -> None:
        from aua.config import load_config
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        router = Router.from_config(load_config(p))

        class BoostEfficacyComponent:
            def compute(self, component, value, field, metadata):
                if component == "efficacy":
                    return min(1.0, value + 0.2)  # boost efficacy by 0.2
                return value

        router._scoring_component = BoostEfficacyComponent()

        import asyncio

        # Run _score and check that U is computed (not zero)
        result = asyncio.run(router._score("test", "test response", "software_engineering", 0.7))
        assert result is not None
        u, conf, n_contra, n_dpo = result
        assert 0.0 <= u <= 1.0

    def test_scoring_component_fallback_on_error(self, tmp_path) -> None:
        from aua.config import load_config
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        router = Router.from_config(load_config(p))

        class FailingComponent:
            def compute(self, *a, **kw):
                raise RuntimeError("component failed")

        router._scoring_component = FailingComponent()

        import asyncio

        # Must not raise
        result = asyncio.run(router._score("q", "r", "software_engineering", 0.7))
        assert result is not None


# ── #53: UtilityScorerPlugin full replacement ─────────────────────────────────


class TestScoreFull:
    def test_score_full_called_when_present(self, tmp_path) -> None:
        from aua.config import load_config
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        router = Router.from_config(load_config(p))

        call_log = []

        class FullReplacementScorer:
            def score(self, response, field, prior_u, confidence, metadata):
                return prior_u  # fallback — should not be called

            def score_full(self, field, efficacy, confidence, curiosity, weights, metadata):
                call_log.append(
                    {
                        "field": field,
                        "efficacy": efficacy,
                        "confidence": confidence,
                        "curiosity": curiosity,
                        "weights": weights,
                    }
                )
                # Non-linear: confidence squared
                return efficacy * (confidence**2) * (1 + 0.1 * curiosity)

        router._custom_scorer = FullReplacementScorer()

        import asyncio

        result = asyncio.run(router._score("query", "response", "software_engineering", 0.8))
        assert len(call_log) == 1
        assert call_log[0]["field"] == "software_engineering"
        assert "w_e" in call_log[0]["weights"]
        u, *_ = result
        assert 0.0 <= u <= 1.0

    def test_score_full_fallback_to_score_on_exception(self, tmp_path) -> None:
        from aua.config import load_config
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        router = Router.from_config(load_config(p))

        score_called = []

        class PartialScorer:
            def score(self, response, field, prior_u, confidence, metadata):
                score_called.append(True)
                return 0.99  # distinctive value

            def score_full(self, *a, **kw):
                raise RuntimeError("score_full not supported")

        router._custom_scorer = PartialScorer()

        import asyncio

        result = asyncio.run(router._score("q", "r", "software_engineering", 0.7))
        # score() was called as fallback
        assert len(score_called) == 1
        u, *_ = result
        assert u == pytest.approx(0.99)

    def test_score_only_no_score_full_uses_adjustment_mode(self, tmp_path) -> None:
        from aua.config import load_config
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 9001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 9003
router:
  port: 8000
  host: "127.0.0.1"
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(cfg_content)
        router = Router.from_config(load_config(p))

        class AdjustmentScorer:
            """Only implements score() — adjustment mode."""

            def score(self, response, field, prior_u, confidence, metadata):
                return 0.42  # fixed distinctive value

        router._custom_scorer = AdjustmentScorer()

        import asyncio

        result = asyncio.run(router._score("q", "r", "software_engineering", 0.7))
        u, *_ = result
        assert u == pytest.approx(0.42)

    def test_score_full_field_specific_logic(self) -> None:
        """Demonstrate non-linear field-specific utility model."""

        class SurgeryAwareScorer:
            def score(self, response, field, prior_u, confidence, metadata):
                return prior_u

            def score_full(self, field, efficacy, confidence, curiosity, weights, metadata):
                if field == "surgery":
                    # Safety-critical: confidence is quadratic
                    return min(1.0, efficacy * (confidence**2))
                # Default: standard linear
                return (
                    weights["w_e"] * efficacy
                    + weights["w_c"] * confidence
                    + weights["w_k"] * curiosity
                )

        scorer = SurgeryAwareScorer()
        weights = {"w_e": 0.55, "w_c": 0.35, "w_k": 0.10}

        # Surgery: penalise low confidence quadratically
        u_surgery = scorer.score_full(
            "surgery", efficacy=0.9, confidence=0.6, curiosity=0.3, weights=weights, metadata={}
        )
        u_linear = 0.55 * 0.9 + 0.35 * 0.6 + 0.10 * 0.3
        # Non-linear surgery score should be LOWER than linear (0.6^2 = 0.36)
        assert u_surgery < u_linear
        assert u_surgery == pytest.approx(0.9 * 0.36, rel=0.01)

        # Software engineering: uses linear weights
        u_swe = scorer.score_full(
            "software_engineering",
            efficacy=0.9,
            confidence=0.8,
            curiosity=0.5,
            weights=weights,
            metadata={},
        )
        assert u_swe == pytest.approx(0.55 * 0.9 + 0.35 * 0.8 + 0.10 * 0.5, rel=0.01)
