"""
aua — Adaptive Utility Agents: a deployable specialist framework.

Install:
    pip install aua                  # runtime only
    pip install aua[train]           # + LoRA training dependencies

Quickstart:
    from aua import Router
    from aua.config import load_config

    config = load_config("aua_config.yaml")
    router = Router.from_config(config)
    result = await router.query("Write binary search. State time complexity.")

CLI:
    aua serve              # start all specialists + router
    aua doctor             # check endpoints, VRAM, deps
    aua status             # live terminal dashboard
    aua init               # scaffold a new project
    aua rollback           # revert to the previous BLUE model
"""

# ── Version ────────────────────────────────────────────────────────────────────
__version__ = "0.5.0"

# ── Arbitration ────────────────────────────────────────────────────────────────
from aua.arbiter import ArbiterAgent, ArbiterVerdict, VerdictCase
from aua.assertions_store import AssertionMatch, AssertionsStore
from aua.confidence_updater import ConfidenceUpdater

# ── Config ─────────────────────────────────────────────────────────────────────
from aua.config import (
    FIELD_CONFIGS,
    ArbiterConfig,
    AUAConfig,
    BlueGreenFieldConfig,
    FieldConfig,
    RouterConfig,
    SpecialistConfig,
    get_effective_config,
    load_config,
)
from aua.contradiction_detector import ContradictionDetector, ContradictionResult

# ── Core pipeline components ───────────────────────────────────────────────────
from aua.field_classifier import FieldClassifier

# ── Routing ────────────────────────────────────────────────────────────────────
from aua.router import Router
from aua.utility_scorer import DomainState, TaskScore, UtilityScorer

__all__ = [
    # version
    "__version__",
    # config
    "load_config",
    "AUAConfig",
    "SpecialistConfig",
    "ArbiterConfig",
    "RouterConfig",
    "BlueGreenFieldConfig",
    "FieldConfig",
    "FIELD_CONFIGS",
    "get_effective_config",
    # pipeline
    "FieldClassifier",
    "UtilityScorer",
    "TaskScore",
    "DomainState",
    "ContradictionDetector",
    "ContradictionResult",
    "AssertionsStore",
    "AssertionMatch",
    "ConfidenceUpdater",
    # arbitration
    "ArbiterAgent",
    "VerdictCase",
    "ArbiterVerdict",
    # routing
    "Router",
]
