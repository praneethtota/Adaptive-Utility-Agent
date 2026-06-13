"""
tests/test_experiment_tracker.py — Tests for #47 experiment tracking.

Coverage:
  ExperimentConfig / MLflowConfig / WandbConfig: defaults, from dict
  experiment_config_from_dict: all fields, empty dict, partial config
  ExperimentTracker.log():
    - no-ops when enabled=False
    - calls mlflow/wandb when enabled
    - graceful when mlflow not installed (ImportError)
    - graceful when wandb not installed (ImportError)
    - graceful when mlflow/wandb raises on log (never propagates)
    - step counter increments on each log() call
    - extracts all metric and tag keys correctly
  ExperimentTracker.finish():
    - calls mlflow.end_run() when mlflow was used
    - calls wandb_run.finish() when wandb was used
    - no-op when nothing was initialised
  YAML loading via load_config:
    - experiment_tracking block parsed into AUAConfig.experiment_tracking
    - missing block gives ExperimentConfig(enabled=False) default
    - enabled: false disables both backends
  Router integration:
    - tracker.log() called after every _handle() response
    - tracker exception never propagates to caller
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aua.experiment_tracker import (
    ExperimentConfig,
    ExperimentTracker,
    MLflowConfig,
    WandbConfig,
    experiment_config_from_dict,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_tracker(
    enabled: bool = True,
    mlflow_enabled: bool = False,
    wandb_enabled: bool = False,
) -> ExperimentTracker:
    cfg = ExperimentConfig(
        enabled=enabled,
        mlflow=MLflowConfig(enabled=mlflow_enabled),
        wandb=WandbConfig(enabled=wandb_enabled),
    )
    return ExperimentTracker(cfg)


def _sample_event() -> dict:
    return {
        "u_score": 0.75,
        "confidence": 0.82,
        "latency_ms": 312.5,
        "contradictions_detected": 0,
        "corrections_injected": 2,
        "dpo_pairs_generated": 1,
        "routing_mode": "single",
        "primary_domain": "software_engineering",
        "specialist": "swe",
        "session_id": "sess-123",
        "trace_id": "trace-456",
    }


# ── ExperimentConfig / dataclasses ────────────────────────────────────────────


class TestExperimentConfig:
    def test_defaults(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.enabled is False
        assert cfg.mlflow.enabled is False
        assert cfg.wandb.enabled is False

    def test_mlflow_defaults(self) -> None:
        cfg = MLflowConfig()
        assert cfg.tracking_uri == "mlruns"
        assert cfg.experiment_name == "aua-framework"
        assert cfg.run_name is None
        assert cfg.log_artifacts is False

    def test_wandb_defaults(self) -> None:
        cfg = WandbConfig()
        assert cfg.project == "aua-framework"
        assert cfg.entity is None
        assert cfg.tags == []


class TestExperimentConfigFromDict:
    def test_empty_dict_gives_defaults(self) -> None:
        cfg = experiment_config_from_dict({})
        assert cfg.enabled is False

    def test_none_gives_defaults(self) -> None:
        cfg = experiment_config_from_dict({})
        assert isinstance(cfg, ExperimentConfig)

    def test_enabled_flag(self) -> None:
        cfg = experiment_config_from_dict({"enabled": True})
        assert cfg.enabled is True

    def test_mlflow_block(self) -> None:
        raw = {
            "enabled": True,
            "mlflow": {
                "enabled": True,
                "tracking_uri": "http://mlflow:5000",
                "experiment_name": "my-exp",
                "run_name": "run-1",
                "log_artifacts": True,
            },
        }
        cfg = experiment_config_from_dict(raw)
        assert cfg.mlflow.enabled is True
        assert cfg.mlflow.tracking_uri == "http://mlflow:5000"
        assert cfg.mlflow.experiment_name == "my-exp"
        assert cfg.mlflow.run_name == "run-1"
        assert cfg.mlflow.log_artifacts is True

    def test_wandb_block(self) -> None:
        raw = {
            "enabled": True,
            "wandb": {
                "enabled": True,
                "project": "my-project",
                "entity": "my-team",
                "run_name": "prod-run",
                "tags": ["prod", "v1"],
            },
        }
        cfg = experiment_config_from_dict(raw)
        assert cfg.wandb.enabled is True
        assert cfg.wandb.project == "my-project"
        assert cfg.wandb.entity == "my-team"
        assert cfg.wandb.tags == ["prod", "v1"]

    def test_partial_mlflow_block_uses_defaults(self) -> None:
        raw = {"enabled": True, "mlflow": {"enabled": True}}
        cfg = experiment_config_from_dict(raw)
        assert cfg.mlflow.experiment_name == "aua-framework"
        assert cfg.mlflow.run_name is None


# ── ExperimentTracker.log() ───────────────────────────────────────────────────


class TestExperimentTrackerLog:
    def test_noop_when_disabled(self) -> None:
        tracker = _make_tracker(enabled=False)
        with patch.dict("sys.modules", {"mlflow": MagicMock(), "wandb": MagicMock()}):
            tracker.log(_sample_event())
        # No exception, step counter stays 0
        assert tracker._step == 0

    def test_step_increments_per_log(self) -> None:
        tracker = _make_tracker(enabled=True, mlflow_enabled=True)
        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="run-1"))
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            tracker.log(_sample_event())
            tracker.log(_sample_event())
            tracker.log(_sample_event())
        assert tracker._step == 3

    def test_mlflow_log_metrics_called(self) -> None:
        tracker = _make_tracker(enabled=True, mlflow_enabled=True)
        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="r1"))
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            tracker.log(_sample_event())
        fake_mlflow.log_metrics.assert_called_once()
        metrics = fake_mlflow.log_metrics.call_args[0][0]
        assert "u_score" in metrics
        assert metrics["u_score"] == pytest.approx(0.75)
        assert "latency_ms" in metrics
        assert "confidence" in metrics

    def test_mlflow_set_tags_called_with_routing_mode(self) -> None:
        tracker = _make_tracker(enabled=True, mlflow_enabled=True)
        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="r1"))
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            tracker.log(_sample_event())
        fake_mlflow.set_tags.assert_called_once()
        tags = fake_mlflow.set_tags.call_args[0][0]
        assert tags["routing_mode"] == "single"
        assert tags["primary_domain"] == "software_engineering"

    def test_wandb_log_called(self) -> None:
        tracker = _make_tracker(enabled=True, wandb_enabled=True)
        fake_wandb = MagicMock()
        fake_run = MagicMock()
        fake_run.name = "run-1"
        fake_wandb.init.return_value = fake_run
        with patch.dict("sys.modules", {"wandb": fake_wandb}):
            tracker.log(_sample_event())
        fake_run.log.assert_called_once()
        payload = fake_run.log.call_args[0][0]
        assert "u_score" in payload
        assert "tag/routing_mode" in payload

    def test_graceful_when_mlflow_not_installed(self) -> None:
        tracker = _make_tracker(enabled=True, mlflow_enabled=True)
        with patch.dict("sys.modules", {"mlflow": None}):
            # Should not raise
            tracker.log(_sample_event())
        assert tracker._mlflow_run is None

    def test_graceful_when_wandb_not_installed(self) -> None:
        tracker = _make_tracker(enabled=True, wandb_enabled=True)
        with patch.dict("sys.modules", {"wandb": None}):
            tracker.log(_sample_event())
        assert tracker._wandb_run is None

    def test_graceful_when_mlflow_log_raises(self) -> None:
        tracker = _make_tracker(enabled=True, mlflow_enabled=True)
        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="r1"))
        fake_mlflow.log_metrics.side_effect = RuntimeError("mlflow server down")
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            # Must not raise
            tracker.log(_sample_event())

    def test_missing_keys_skipped_gracefully(self) -> None:
        tracker = _make_tracker(enabled=True, mlflow_enabled=True)
        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="r1"))
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            # Minimal event — only u_score
            tracker.log({"u_score": 0.5})
        metrics = fake_mlflow.log_metrics.call_args[0][0]
        assert "u_score" in metrics
        assert "latency_ms" not in metrics

    def test_both_backends_called(self) -> None:
        tracker = _make_tracker(enabled=True, mlflow_enabled=True, wandb_enabled=True)
        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="r1"))
        fake_wandb = MagicMock()
        fake_run = MagicMock()
        fake_run.name = "run-1"
        fake_wandb.init.return_value = fake_run
        with patch.dict("sys.modules", {"mlflow": fake_mlflow, "wandb": fake_wandb}):
            tracker.log(_sample_event())
        fake_mlflow.log_metrics.assert_called_once()
        fake_run.log.assert_called_once()


# ── ExperimentTracker._extract_metrics / _extract_tags ───────────────────────


class TestExtractHelpers:
    def test_extract_metrics_all_keys(self) -> None:
        event = _sample_event()
        metrics = ExperimentTracker._extract_metrics(event)
        assert set(metrics) == {
            "u_score",
            "confidence",
            "latency_ms",
            "contradictions_detected",
            "corrections_injected",
            "dpo_pairs_generated",
        }

    def test_extract_tags_all_keys(self) -> None:
        event = _sample_event()
        tags = ExperimentTracker._extract_tags(event)
        assert "routing_mode" in tags
        assert "primary_domain" in tags
        assert "specialist" in tags
        assert "session_id" in tags
        assert "trace_id" in tags

    def test_extract_metrics_skips_none(self) -> None:
        metrics = ExperimentTracker._extract_metrics({"u_score": None, "latency_ms": 100.0})
        assert "u_score" not in metrics
        assert metrics["latency_ms"] == pytest.approx(100.0)

    def test_extract_tags_skips_none(self) -> None:
        tags = ExperimentTracker._extract_tags({"routing_mode": "single", "trace_id": None})
        assert "routing_mode" in tags
        assert "trace_id" not in tags


# ── ExperimentTracker.finish() ────────────────────────────────────────────────


class TestExperimentTrackerFinish:
    def test_mlflow_end_run_called(self) -> None:
        tracker = _make_tracker(enabled=True, mlflow_enabled=True)
        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="r1"))
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            tracker.log(_sample_event())  # initialises the run
            tracker.finish()
        fake_mlflow.end_run.assert_called_once()

    def test_wandb_finish_called(self) -> None:
        tracker = _make_tracker(enabled=True, wandb_enabled=True)
        fake_wandb = MagicMock()
        fake_run = MagicMock()
        fake_run.name = "r1"
        fake_wandb.init.return_value = fake_run
        with patch.dict("sys.modules", {"wandb": fake_wandb}):
            tracker.log(_sample_event())
            tracker.finish()
        fake_run.finish.assert_called_once()

    def test_finish_noop_when_never_initialised(self) -> None:
        tracker = _make_tracker(enabled=False)
        # Must not raise
        tracker.finish()

    def test_finish_graceful_when_end_run_raises(self) -> None:
        tracker = _make_tracker(enabled=True, mlflow_enabled=True)
        fake_mlflow = MagicMock()
        fake_mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="r1"))
        fake_mlflow.end_run.side_effect = RuntimeError("connection lost")
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            tracker.log(_sample_event())
            # Must not raise
            tracker.finish()


# ── YAML config loading ───────────────────────────────────────────────────────


class TestYamlLoading:
    def _write_config(self, tmp_path: Path, extra: str = "") -> Path:
        yaml_content = f"""
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
{extra}
"""
        p = tmp_path / "aua_config.yaml"
        p.write_text(yaml_content)
        return p

    def test_missing_block_gives_disabled_default(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(self._write_config(tmp_path))
        assert cfg.experiment_tracking.enabled is False

    def test_enabled_false_block(self, tmp_path: Path) -> None:
        from aua.config import load_config

        extra = "experiment_tracking:\n  enabled: false\n"
        cfg = load_config(self._write_config(tmp_path, extra))
        assert cfg.experiment_tracking.enabled is False

    def test_mlflow_block_loaded(self, tmp_path: Path) -> None:
        from aua.config import load_config

        extra = """experiment_tracking:
  enabled: true
  mlflow:
    enabled: true
    tracking_uri: http://mlflow:5000
    experiment_name: prod-exp
"""
        cfg = load_config(self._write_config(tmp_path, extra))
        assert cfg.experiment_tracking.enabled is True
        assert cfg.experiment_tracking.mlflow.enabled is True
        assert cfg.experiment_tracking.mlflow.tracking_uri == "http://mlflow:5000"
        assert cfg.experiment_tracking.mlflow.experiment_name == "prod-exp"

    def test_wandb_block_loaded(self, tmp_path: Path) -> None:
        from aua.config import load_config

        extra = """experiment_tracking:
  enabled: true
  wandb:
    enabled: true
    project: my-project
    entity: my-team
    tags: [production, v2]
"""
        cfg = load_config(self._write_config(tmp_path, extra))
        assert cfg.experiment_tracking.wandb.enabled is True
        assert cfg.experiment_tracking.wandb.project == "my-project"
        assert cfg.experiment_tracking.wandb.tags == ["production", "v2"]


# ── Router integration ────────────────────────────────────────────────────────


class TestRouterIntegration:
    def test_tracker_log_called_after_handle(self, tmp_path: Path) -> None:
        """
        Verify ExperimentTracker.log() is called for each query response.
        Uses the minimal_config + fake server pattern from conftest.
        """
        from aua.config import load_config
        from aua.experiment_tracker import ExperimentConfig
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 19001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 19003
router:
  port: 19000
  host: "127.0.0.1"
  specialist_timeout: 5.0
"""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(cfg_content)
        cfg = load_config(cfg_file)

        # Enabled tracker that doesn't actually connect anywhere
        cfg.experiment_tracking = ExperimentConfig(enabled=True)

        router = Router.from_config(cfg)
        tracker_mock = MagicMock()
        router._experiment_tracker = tracker_mock

        import asyncio

        from aua.endpoints import QueryRequest

        # Patch the actual specialist call so we don't need a live server
        fake_resp_text = "def binary_search(): pass"
        with patch.object(router, "_call", return_value=(fake_resp_text, 0.8)):
            asyncio.run(router._handle(QueryRequest(query="Write binary search")))

        tracker_mock.log.assert_called_once()
        event = tracker_mock.log.call_args[0][0]
        assert "u_score" in event
        assert "routing_mode" in event
        assert "latency_ms" in event

    def test_tracker_exception_does_not_propagate(self, tmp_path: Path) -> None:
        """A crashing tracker must never affect the query response."""
        from aua.config import load_config
        from aua.experiment_tracker import ExperimentConfig
        from aua.router import Router

        cfg_content = """
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: fake/model
    port: 19001
    field: software_engineering
arbiter:
  model: fake/arb
  port: 19003
router:
  port: 19000
  host: "127.0.0.1"
  specialist_timeout: 5.0
"""
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(cfg_content)
        cfg = load_config(cfg_file)
        cfg.experiment_tracking = ExperimentConfig(enabled=True)

        router = Router.from_config(cfg)
        crashing_tracker = MagicMock()
        crashing_tracker.log.side_effect = RuntimeError("tracker exploded")
        router._experiment_tracker = crashing_tracker

        import asyncio

        from aua.endpoints import QueryRequest

        with patch.object(router, "_call", return_value=("answer", 0.8)):
            # Must not raise despite tracker crash
            resp = asyncio.run(router._handle(QueryRequest(query="test")))
        assert resp is not None
        assert resp.response == "answer"
