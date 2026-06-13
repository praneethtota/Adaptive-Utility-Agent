"""
aua/experiment_tracker.py — Experiment tracking integration (#47).

Logs per-query metrics to MLflow and/or Weights & Biases automatically
after every routing decision. Both backends are lazy-imported so neither
is a hard dependency — the tracker no-ops cleanly when neither is
configured or installed.

Metrics logged per query:
    u_score            — utility score U = w_e·E + w_c·C + w_k·K
    confidence         — updated confidence after contradiction check
    latency_ms         — wall-clock latency from query receipt to response
    contradictions     — number of contradictions detected
    corrections        — number of prior corrections injected
    dpo_pairs          — DPO pairs generated from this query
    routing_mode       — "single" | "fanout" | "arbiter" | "vcg"
    primary_domain     — field classifier primary domain

Tags/dimensions (not metrics, used for filtering in MLflow/W&B UI):
    specialist         — winning specialist name
    session_id         — request session ID
    trace_id           — request trace ID

Configuration (aua_config.yaml):

    experiment_tracking:
      enabled: true

      mlflow:
        enabled: true
        tracking_uri: http://localhost:5000   # or file:///path/to/mlruns
        experiment_name: aua-production       # created if it doesn't exist
        run_name: aua-router                  # optional; auto-named if omitted
        log_artifacts: false                  # log response text as artifact

      wandb:
        enabled: true
        project: aua-framework
        entity: my-team                       # optional; uses default entity
        run_name: aua-router                  # optional
        tags: [production, v1]                # optional list of tags

Usage (programmatic):

    from aua.experiment_tracker import ExperimentTracker, ExperimentConfig
    tracker = ExperimentTracker(config)
    tracker.log(response)           # RouterResponse or dict
    tracker.finish()                # call on shutdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ── Config dataclasses ────────────────────────────────────────────────────────


@dataclass
class MLflowConfig:
    enabled: bool = False
    tracking_uri: str = "mlruns"  # local file store by default
    experiment_name: str = "aua-framework"
    run_name: str | None = None
    log_artifacts: bool = False  # log response text as artifact (slow)


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "aua-framework"
    entity: str | None = None  # defaults to logged-in user
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """
    Top-level experiment tracking config.

    YAML key: experiment_tracking
    """

    enabled: bool = False
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


# ── Tracker ───────────────────────────────────────────────────────────────────


class ExperimentTracker:
    """
    Logs per-query AUA metrics to MLflow and/or W&B.

    Instantiate once (at router startup) and call log() after each query.
    The tracker initialises the backend connections lazily on the first
    log() call so startup time is not affected.

    Thread-safe: MLflow and W&B both use thread-local run state.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._mlflow_run: Any = None
        self._wandb_run: Any = None
        self._mlflow_init = False
        self._wandb_init = False
        self._step = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, event: dict[str, Any]) -> None:
        """
        Log metrics from a post_response hook event or RouterResponse dict.

        Args:
            event: dict with keys from the post_response hook payload:
                   u_score, confidence, latency_ms, contradictions_detected,
                   corrections_injected, dpo_pairs_generated, routing_mode,
                   primary_domain, specialist, session_id, trace_id.
                   Missing keys are silently skipped.
        """
        if not self._config.enabled:
            return

        metrics = self._extract_metrics(event)
        tags = self._extract_tags(event)
        self._step += 1

        if self._config.mlflow.enabled:
            self._log_mlflow(metrics, tags)

        if self._config.wandb.enabled:
            self._log_wandb(metrics, tags)

    def finish(self) -> None:
        """
        Close backend connections. Call on router shutdown.
        """
        if self._mlflow_run is not None:
            try:
                import mlflow

                mlflow.end_run()
            except Exception as e:
                log.debug("MLflow end_run failed: %s", e)

        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception as e:
                log.debug("W&B finish failed: %s", e)

    # ── Metric extraction ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_metrics(event: dict[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, dtype in [
            ("u_score", float),
            ("confidence", float),
            ("latency_ms", float),
            ("contradictions_detected", float),
            ("corrections_injected", float),
            ("dpo_pairs_generated", float),
        ]:
            if key in event and event[key] is not None:
                try:
                    metrics[key] = dtype(event[key])
                except (TypeError, ValueError):
                    pass
        return metrics

    @staticmethod
    def _extract_tags(event: dict[str, Any]) -> dict[str, str]:
        tags: dict[str, str] = {}
        for key in ("routing_mode", "primary_domain", "specialist", "session_id", "trace_id"):
            if key in event and event[key] is not None:
                tags[key] = str(event[key])
        return tags

    # ── MLflow backend ────────────────────────────────────────────────────────

    def _ensure_mlflow(self) -> bool:
        """Initialise MLflow run on first call. Returns False if unavailable."""
        if self._mlflow_init:
            return self._mlflow_run is not None

        self._mlflow_init = True
        cfg = self._config.mlflow
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg.tracking_uri)
            mlflow.set_experiment(cfg.experiment_name)
            self._mlflow_run = mlflow.start_run(run_name=cfg.run_name)
            log.info(
                "ExperimentTracker: MLflow run started — %s (experiment: %s, uri: %s)",
                self._mlflow_run.info.run_id,
                cfg.experiment_name,
                cfg.tracking_uri,
            )
        except ImportError:
            log.warning(
                "ExperimentTracker: mlflow not installed — "
                "run: pip install mlflow. MLflow logging disabled."
            )
            self._mlflow_run = None
        except Exception as e:
            log.warning("ExperimentTracker: MLflow init failed: %s. Logging disabled.", e)
            self._mlflow_run = None

        return self._mlflow_run is not None

    def _log_mlflow(self, metrics: dict[str, float], tags: dict[str, str]) -> None:
        if not self._ensure_mlflow():
            return
        try:
            import mlflow

            mlflow.log_metrics(metrics, step=self._step)
            if tags:
                mlflow.set_tags(tags)
        except Exception as e:
            log.debug("MLflow log_metrics failed: %s", e)

    # ── W&B backend ───────────────────────────────────────────────────────────

    def _ensure_wandb(self) -> bool:
        """Initialise W&B run on first call. Returns False if unavailable."""
        if self._wandb_init:
            return self._wandb_run is not None

        self._wandb_init = True
        cfg = self._config.wandb
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                name=cfg.run_name,
                tags=cfg.tags or None,
                reinit=True,
            )
            log.info(
                "ExperimentTracker: W&B run started — %s (project: %s)",
                self._wandb_run.name,
                cfg.project,
            )
        except ImportError:
            log.warning(
                "ExperimentTracker: wandb not installed — "
                "run: pip install wandb. W&B logging disabled."
            )
            self._wandb_run = None
        except Exception as e:
            log.warning("ExperimentTracker: W&B init failed: %s. Logging disabled.", e)
            self._wandb_run = None

        return self._wandb_run is not None

    def _log_wandb(self, metrics: dict[str, float], tags: dict[str, str]) -> None:
        if not self._ensure_wandb():
            return
        try:
            payload = {**metrics, **{f"tag/{k}": v for k, v in tags.items()}}
            self._wandb_run.log(payload, step=self._step)
        except Exception as e:
            log.debug("W&B log failed: %s", e)


# ── Config builder (called from config.py YAML loader) ───────────────────────


def experiment_config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """
    Parse the `experiment_tracking:` YAML block into an ExperimentConfig.

    All keys are optional — missing keys use their dataclass defaults.
    """
    if not raw:
        return ExperimentConfig()

    enabled = bool(raw.get("enabled", False))

    raw_mlf = raw.get("mlflow", {}) or {}
    mlflow_cfg = MLflowConfig(
        enabled=bool(raw_mlf.get("enabled", False)),
        tracking_uri=str(raw_mlf.get("tracking_uri", "mlruns")),
        experiment_name=str(raw_mlf.get("experiment_name", "aua-framework")),
        run_name=raw_mlf.get("run_name") or None,
        log_artifacts=bool(raw_mlf.get("log_artifacts", False)),
    )

    raw_wb = raw.get("wandb", {}) or {}
    wandb_cfg = WandbConfig(
        enabled=bool(raw_wb.get("enabled", False)),
        project=str(raw_wb.get("project", "aua-framework")),
        entity=raw_wb.get("entity") or None,
        run_name=raw_wb.get("run_name") or None,
        tags=list(raw_wb.get("tags", [])),
    )

    return ExperimentConfig(
        enabled=enabled,
        mlflow=mlflow_cfg,
        wandb=wandb_cfg,
    )
