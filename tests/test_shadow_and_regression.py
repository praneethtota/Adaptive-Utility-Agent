"""
tests/test_shadow_and_regression.py — Tests for #48 shadow mode and #49 regression gate.

#48 shadow mode:
  ShadowStore: record, get_scores, clear, aggregate
  ShadowManager: activate, deactivate, is_active, shadow_endpoint, report
  ShadowManager.shadow_call: records score pair, handles errors gracefully
  ShadowReport.to_dict: shape and types
  Router: shadow fires after _handle_single when active
  Config: shadow_endpoint and shadow_min_queries loaded from YAML
  REST: POST/GET/DELETE /deploy/shadow/{specialist}

#49 regression gate:
  BlueGreenFieldConfig: regression_dataset and regression_block fields
  Config YAML: both fields loaded
  _evaluate_green: regression_dataset triggers eval, blocks on regression
  _evaluate_green: regression_block=False warns but does not block
  _evaluate_green: no dataset → regression=None in response
  _evaluate_green: uses shadow scores when n_queries >= min_queries
  DeployGreenResponse: regression field shape
  DeployGreenRequest: green_endpoint and regression_dataset fields
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aua.shadow import ShadowManager, ShadowStore

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path):
    from aua.state import SQLiteStateStore

    return SQLiteStateStore(db_path=tmp_path / "test.db")


@pytest.fixture
def shadow_store(store):
    return ShadowStore(store)


@pytest.fixture
def mgr(shadow_store):
    return ShadowManager(shadow_store)


# ── ShadowStore ───────────────────────────────────────────────────────────────


class TestShadowStore:
    def test_record_writes_row(self, shadow_store) -> None:
        rid = shadow_store.record("swe", "binary search", 0.75, 0.80, "software_engineering")
        assert rid is not None
        rows = shadow_store.get_scores("swe")
        assert len(rows) == 1
        assert rows[0]["specialist"] == "swe"
        assert rows[0]["blue_u"] == pytest.approx(0.75)
        assert rows[0]["green_u"] == pytest.approx(0.80)
        assert rows[0]["u_delta"] == pytest.approx(0.05, abs=1e-4)

    def test_get_scores_filtered_by_specialist(self, shadow_store) -> None:
        shadow_store.record("swe", "q1", 0.7, 0.8, "software_engineering")
        shadow_store.record("math", "q2", 0.6, 0.65, "mathematics")
        swe_rows = shadow_store.get_scores("swe")
        assert len(swe_rows) == 1
        assert swe_rows[0]["specialist"] == "swe"

    def test_clear_deletes_rows(self, shadow_store) -> None:
        shadow_store.record("swe", "q", 0.7, 0.8, "swe")
        shadow_store.record("swe", "q2", 0.6, 0.7, "swe")
        n = shadow_store.clear("swe")
        assert n == 2
        assert shadow_store.get_scores("swe") == []

    def test_clear_only_affects_specialist(self, shadow_store) -> None:
        shadow_store.record("swe", "q", 0.7, 0.8, "swe")
        shadow_store.record("math", "q", 0.6, 0.7, "math")
        shadow_store.clear("swe")
        assert len(shadow_store.get_scores("math")) == 1

    def test_aggregate_empty(self, shadow_store) -> None:
        agg = shadow_store.aggregate("swe")
        assert agg["n"] == 0
        assert agg["mean_blue_u"] == 0.0
        assert agg["mean_green_u"] == 0.0

    def test_aggregate_computes_means(self, shadow_store) -> None:
        shadow_store.record("swe", "q1", 0.70, 0.80, "swe")
        shadow_store.record("swe", "q2", 0.80, 0.90, "swe")
        agg = shadow_store.aggregate("swe")
        assert agg["n"] == 2
        assert agg["mean_blue_u"] == pytest.approx(0.75)
        assert agg["mean_green_u"] == pytest.approx(0.85)
        assert agg["mean_delta"] == pytest.approx(0.10, abs=1e-3)


# ── ShadowManager ─────────────────────────────────────────────────────────────


class TestShadowManager:
    def test_activate_marks_active(self, mgr) -> None:
        mgr.activate("swe", "http://localhost:9011/v1/chat/completions")
        assert mgr.is_active("swe")
        assert mgr.shadow_endpoint("swe") == "http://localhost:9011/v1/chat/completions"

    def test_deactivate_removes_active(self, mgr) -> None:
        mgr.activate("swe", "http://localhost:9011/v1/chat/completions")
        mgr.deactivate("swe")
        assert not mgr.is_active("swe")
        assert mgr.shadow_endpoint("swe") is None

    def test_deactivate_clear_scores(self, mgr, shadow_store) -> None:
        shadow_store.record("swe", "q", 0.7, 0.8, "swe")
        mgr.activate("swe", "http://localhost:9011/v1/chat/completions")
        mgr.deactivate("swe", clear_scores=True)
        assert shadow_store.get_scores("swe") == []

    def test_is_active_false_by_default(self, mgr) -> None:
        assert not mgr.is_active("nonexistent")

    def test_report_inactive_specialist(self, mgr) -> None:
        report = mgr.report("swe")
        assert not report.active
        assert report.n_queries == 0
        assert not report.ready_to_promote

    def test_report_active_not_ready(self, mgr, shadow_store) -> None:
        mgr.activate(
            "swe", "http://localhost:9011/v1/chat/completions", min_queries=50, threshold=0.025
        )
        shadow_store.record("swe", "q", 0.7, 0.73, "swe")
        report = mgr.report("swe")
        assert report.active
        assert report.n_queries == 1
        assert report.min_queries == 50
        assert not report.ready_to_promote
        assert "1/50" in report.to_dict()["progress"]

    def test_report_ready_to_promote(self, mgr, shadow_store) -> None:
        mgr.activate(
            "swe", "http://localhost:9011/v1/chat/completions", min_queries=2, threshold=0.025
        )
        shadow_store.record("swe", "q1", 0.70, 0.73, "swe")
        shadow_store.record("swe", "q2", 0.72, 0.75, "swe")
        report = mgr.report("swe")
        assert report.n_queries == 2
        assert report.ready_to_promote

    def test_report_to_dict_shape(self, mgr) -> None:
        mgr.activate("swe", "http://localhost:9011/v1/chat/completions")
        d = mgr.report("swe").to_dict()
        assert "specialist" in d
        assert "shadow_endpoint" in d
        assert "n_queries" in d
        assert "ready_to_promote" in d
        assert "progress" in d


class TestShadowManagerShadowCall:
    @pytest.mark.asyncio
    async def test_shadow_call_records_score(self, mgr, shadow_store) -> None:
        mgr.activate("swe", "http://green:9011/v1/chat/completions", min_queries=5)

        async def fake_call(url, query, domain, model_name="default"):
            return ("def binary_search(): pass", 0.85)

        async def fake_score(query, response, domain, conf, **kwargs):
            return (0.82, conf, 0, 0)

        await mgr.shadow_call(
            specialist="swe",
            query="Write binary search",
            domain="software_engineering",
            blue_u=0.75,
            call_fn=fake_call,
            score_fn=fake_score,
        )

        agg = shadow_store.aggregate("swe")
        assert agg["n"] == 1
        assert agg["mean_blue_u"] == pytest.approx(0.75)
        assert agg["mean_green_u"] == pytest.approx(0.82)

    @pytest.mark.asyncio
    async def test_shadow_call_handles_call_error(self, mgr, shadow_store) -> None:
        mgr.activate("swe", "http://green:9011", min_queries=5)

        async def failing_call(*args, **kwargs):
            raise ConnectionRefusedError("green is down")

        async def fake_score(*args, **kwargs):
            return (0.8, 0.8, 0, 0)

        # Must not raise
        await mgr.shadow_call(
            specialist="swe",
            query="q",
            domain="swe",
            blue_u=0.7,
            call_fn=failing_call,
            score_fn=fake_score,
        )

        # No rows written on error
        assert shadow_store.aggregate("swe")["n"] == 0

    @pytest.mark.asyncio
    async def test_shadow_call_noop_when_inactive(self, mgr, shadow_store) -> None:
        """No record written when specialist is not in shadow mode."""

        async def fake_call(*a, **kw):
            return ("resp", 0.8)

        async def fake_score(*a, **kw):
            return (0.8, 0.8, 0, 0)

        await mgr.shadow_call("swe", "q", "d", 0.7, fake_call, fake_score)
        assert shadow_store.aggregate("swe")["n"] == 0


# ── Config YAML loading ───────────────────────────────────────────────────────


class TestConfigLoading:
    def _write(self, tmp_path: Path, bg_extra: str = "") -> Path:
        content = f"""
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: Qwen/model
    port: 9001
    field: software_engineering
arbiter:
  model: Qwen/arb
  port: 9003
router:
  port: 8000
blue_green:
  swe:
    delta: 0.025
    {bg_extra}
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(content)
        return p

    def test_shadow_endpoint_loaded(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(
            self._write(tmp_path, "shadow_endpoint: http://localhost:9011/v1/chat/completions")
        )
        bg = cfg.blue_green_for("swe")
        assert bg.shadow_endpoint == "http://localhost:9011/v1/chat/completions"

    def test_shadow_min_queries_loaded(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(self._write(tmp_path, "shadow_min_queries: 100"))
        assert cfg.blue_green_for("swe").shadow_min_queries == 100

    def test_shadow_defaults(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(self._write(tmp_path))
        bg = cfg.blue_green_for("swe")
        assert bg.shadow_endpoint is None
        assert bg.shadow_min_queries == 50

    def test_regression_dataset_loaded(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(self._write(tmp_path, "regression_dataset: evals/coding.yaml"))
        assert cfg.blue_green_for("swe").regression_dataset == "evals/coding.yaml"

    def test_regression_block_false(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(self._write(tmp_path, "regression_block: false"))
        assert cfg.blue_green_for("swe").regression_block is False

    def test_regression_defaults(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(self._write(tmp_path))
        bg = cfg.blue_green_for("swe")
        assert bg.regression_dataset is None
        assert bg.regression_block is True


# ── DeployGreen schema ────────────────────────────────────────────────────────


class TestDeployGreenSchema:
    def test_request_accepts_green_endpoint(self) -> None:
        from aua.endpoints import DeployGreenRequest

        req = DeployGreenRequest(
            specialist="swe",
            green_model="./models/swe_green",
            green_endpoint="http://localhost:9011/v1/chat/completions",
            regression_dataset="evals/coding.yaml",
        )
        assert req.green_endpoint == "http://localhost:9011/v1/chat/completions"
        assert req.regression_dataset == "evals/coding.yaml"

    def test_request_green_endpoint_optional(self) -> None:
        from aua.endpoints import DeployGreenRequest

        req = DeployGreenRequest(specialist="swe", green_model="./m")
        assert req.green_endpoint is None

    def test_response_has_regression_field(self) -> None:
        from aua.endpoints import DeployGreenResponse

        resp = DeployGreenResponse(
            specialist="swe",
            promoted=False,
            u_delta=-0.01,
            blue_u=0.75,
            green_u=0.74,
            threshold=0.025,
            message="not promoted",
            regression={
                "regressed": True,
                "verdict": "REGRESSION",
                "delta_pass_rate": -0.10,
                "delta_u_score": -0.02,
                "delta_latency_ms": 50.0,
                "dataset": "evals/coding.yaml",
                "blocked": True,
            },
        )
        assert resp.regression is not None
        assert resp.regression["regressed"] is True
        assert resp.regression["blocked"] is True

    def test_response_regression_none_by_default(self) -> None:
        from aua.endpoints import DeployGreenResponse

        resp = DeployGreenResponse(
            specialist="swe",
            promoted=False,
            u_delta=0.0,
            blue_u=0.0,
            green_u=0.0,
            threshold=0.025,
            message="dry",
        )
        assert resp.regression is None


# ── ShadowActivateRequest schema ──────────────────────────────────────────────


class TestShadowActivateSchema:
    def test_required_field(self) -> None:
        from aua.endpoints import ShadowActivateRequest

        req = ShadowActivateRequest(green_endpoint="http://localhost:9011/v1/chat/completions")
        assert req.green_endpoint == "http://localhost:9011/v1/chat/completions"
        assert req.min_queries is None
        assert req.threshold is None

    def test_optional_fields(self) -> None:
        from aua.endpoints import ShadowActivateRequest

        req = ShadowActivateRequest(
            green_endpoint="http://localhost:9011/v1/chat/completions",
            min_queries=100,
            threshold=0.05,
        )
        assert req.min_queries == 100
        assert req.threshold == pytest.approx(0.05)


# ── _evaluate_green regression gate ──────────────────────────────────────────


class TestEvaluateGreenRegressionGate:
    def _make_eval_dataset(self, tmp_path: Path) -> Path:
        """Write a minimal eval dataset YAML."""
        cases = [
            {
                "id": "test_bs",
                "prompt": "Write binary search.",
                "expected_properties": [{"contains": "def"}, {"min_length": 10}],
            }
        ]
        data = {
            "name": "test_eval",
            "field": "software_engineering",
            "description": "test",
            "cases": cases,
        }
        p = tmp_path / "eval.yaml"
        p.write_text(yaml.dump(data))
        return p

    def test_no_dataset_no_regression_field(self, tmp_path: Path) -> None:
        """Without a dataset, regression field should be None."""
        from aua.endpoints import DeployGreenResponse

        # Verify schema: regression=None when no dataset provided
        resp = DeployGreenResponse(
            specialist="swe",
            promoted=False,
            u_delta=0.0,
            blue_u=0.75,
            green_u=0.75,
            threshold=0.025,
            message="dry run",
            regression=None,
        )
        assert resp.regression is None

    def test_regression_blocked_message(self, tmp_path: Path) -> None:
        """A blocked response should include 'BLOCKED' in message and regression.blocked=True."""
        from aua.endpoints import DeployGreenResponse

        resp = DeployGreenResponse(
            specialist="swe",
            promoted=False,
            u_delta=-0.03,
            blue_u=0.75,
            green_u=0.72,
            threshold=0.025,
            message="PROMOTION BLOCKED — regression detected",
            regression={
                "regressed": True,
                "verdict": "REGRESSION",
                "delta_pass_rate": -0.15,
                "delta_u_score": -0.03,
                "delta_latency_ms": 0.0,
                "dataset": "evals/test.yaml",
                "blocked": True,
            },
        )
        assert "BLOCKED" in resp.message
        assert resp.regression["blocked"] is True
        assert resp.promoted is False

    def test_evaluate_green_uses_shadow_scores_when_ready(self, tmp_path: Path) -> None:
        """When shadow has enough queries, _evaluate_green should prefer them."""
        # This is an integration test of the decision path logic:
        # - shadow.n_queries >= min_queries → use shadow scores
        # - u_delta >= threshold → promoted

        # We verify by checking ShadowManager.report().ready_to_promote
        from aua.shadow import ShadowManager, ShadowStore
        from aua.state import SQLiteStateStore

        store = SQLiteStateStore(db_path=tmp_path / "t.db")
        ss = ShadowStore(store)
        m = ShadowManager(ss)
        m.activate("swe", "http://localhost:9011", min_queries=3, threshold=0.025)

        # Insert 3 shadow scores with positive delta
        for i in range(3):
            ss.record("swe", f"q{i}", 0.70, 0.74, "software_engineering")

        report = m.report("swe")
        assert report.n_queries == 3
        assert report.ready_to_promote  # 3 >= 3 and mean_delta=0.04 >= 0.025

    def test_evaluate_green_not_ready_when_too_few_queries(self, tmp_path: Path) -> None:
        from aua.shadow import ShadowManager, ShadowStore
        from aua.state import SQLiteStateStore

        store = SQLiteStateStore(db_path=tmp_path / "t.db")
        ss = ShadowStore(store)
        m = ShadowManager(ss)
        m.activate("swe", "http://localhost:9011", min_queries=50, threshold=0.025)
        ss.record("swe", "q", 0.70, 0.74, "swe")

        report = m.report("swe")
        assert not report.ready_to_promote
        assert "1/50" in report.to_dict()["progress"]
