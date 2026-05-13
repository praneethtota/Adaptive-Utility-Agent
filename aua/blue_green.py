"""
aua/blue_green.py — Blue-green deployment controller.

Manages model promotions: when a GREEN candidate achieves sufficient utility
improvement over BLUE, BlueGreenDeployment orchestrates the traffic shift,
records the promotion event, and provides rollback capability.

Status: v0.6-alpha stub — interface defined, full harness is roadmap #14.
The rollback log (record_promotion / run_rollback) is fully operational.
The automated U-score evaluation harness will be added in v0.7.

Usage:
    from aua import BlueGreenDeployment

    bg = BlueGreenDeployment(config, specialist_name="swe")
    bg.register_green(model_path="models/swe-green/")
    summary = await bg.evaluate()          # runs N_eval queries, computes ΔU
    if bg.should_promote(summary):
        bg.promote()                       # updates config + records event
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aua.config import AUAConfig, BlueGreenFieldConfig


@dataclass
class EvaluationSummary:
    """Result of a green-vs-blue evaluation run."""

    specialist: str
    blue_model: str
    green_model: str
    blue_mean_u: float = 0.0
    green_mean_u: float = 0.0
    u_delta: float = 0.0
    n_queries: int = 0
    promoted: bool = False
    dry_run: bool = True  # True until full harness is built (roadmap #14)


class BlueGreenDeployment:
    """
    Blue-green deployment controller for a single specialist.

    Tracks the BLUE (production) and GREEN (candidate) model for one specialist,
    evaluates GREEN against the configured utility threshold, and promotes
    when criteria are met.

    Full evaluation harness is roadmap #14. Current capabilities:
      - Record promotions / rollbacks (fully operational via aua.rollback)
      - Dry-run evaluation (returns dummy summary)
      - Interface stable for downstream use
    """

    def __init__(
        self,
        config: AUAConfig,
        specialist_name: str,
        project_dir: str = ".",
    ) -> None:
        self._config = config
        self._specialist_name = specialist_name
        self._project_dir = project_dir
        self._green_model: str | None = None

        spec = config.specialist(specialist_name)
        self._blue_model = spec.model
        self._bg_config: BlueGreenFieldConfig = config.blue_green_for(specialist_name)

    @property
    def specialist_name(self) -> str:
        return self._specialist_name

    @property
    def blue_model(self) -> str:
        return self._blue_model

    @property
    def green_model(self) -> str | None:
        return self._green_model

    @property
    def delta_threshold(self) -> float:
        """Minimum U improvement required to promote GREEN → BLUE."""
        return self._bg_config.delta

    @property
    def t_min(self) -> int:
        """Minimum interactions before a promotion decision."""
        return self._bg_config.T_min

    def register_green(self, model_path: str) -> None:
        """Register a GREEN candidate model for evaluation."""
        self._green_model = model_path

    async def evaluate(
        self,
        n_queries: int = 10,
        router: Any | None = None,
    ) -> EvaluationSummary:
        """
        Evaluate GREEN model against BLUE baseline.

        Args:
            n_queries: number of test queries to run
            router:    running Router instance (required for live evaluation)

        Returns:
            EvaluationSummary with U scores and promotion decision.

        Note:
            Full harness not yet implemented (roadmap #14).
            Returns a dry_run=True summary until v0.7.
        """
        if self._green_model is None:
            raise ValueError("No GREEN model registered. Call register_green() first.")

        # Stub: full evaluation harness is roadmap #14
        return EvaluationSummary(
            specialist=self._specialist_name,
            blue_model=self._blue_model,
            green_model=self._green_model,
            n_queries=n_queries,
            dry_run=True,  # full harness in roadmap #14
        )

    def should_promote(self, summary: EvaluationSummary) -> bool:
        """Return True if GREEN should be promoted based on evaluation summary."""
        if summary.dry_run:
            return False
        return summary.u_delta >= self.delta_threshold and summary.n_queries >= self.t_min

    def promote(self, yes: bool = False, restart: bool = True) -> bool:
        """
        Promote GREEN to BLUE: update config and record the promotion event.

        Returns True on success, False if no green model is registered.
        """
        from aua.rollback import record_promotion

        if self._green_model is None:
            return False

        promoted_from = self._blue_model
        promoted_to = self._green_model

        record_promotion(
            specialist=self._specialist_name,
            from_model=self._blue_model,
            to_model=self._green_model,
            project_dir=self._project_dir,
        )
        self._blue_model = self._green_model
        self._green_model = None

        # ── on_promotion hook (background — non-blocking) ─────────────────
        try:

            from aua.hooks import get_hook_runner

            get_hook_runner().fire_background(
                "on_promotion",
                {
                    "session_id": "",
                    "trace_id": "",
                    "specialist": self._specialist_name,
                    "promoted_from": promoted_from,
                    "promoted_to": promoted_to,
                    "project_dir": self._project_dir,
                },
            )
        except Exception:
            pass  # hooks are always fail-open

        return True

    def rollback(self, yes: bool = False, restart: bool = False) -> int:
        """Revert the last promotion. Returns 0 on success, 1 on failure."""
        from aua.rollback import run_rollback

        return run_rollback(
            config_path=f"{self._project_dir}/aua_config.yaml",
            specialist=self._specialist_name,
            yes=yes,
            restart=restart,
        )
