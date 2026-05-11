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

# ── Core pipeline components ───────────────────────────────────────────────────
from aua.field_classifier import FieldClassifier
from aua.utility_scorer import UtilityScorer, TaskScore, DomainState
from aua.contradiction_detector import ContradictionDetector, ContradictionResult
from aua.assertions_store import AssertionsStore, AssertionMatch
from aua.confidence_updater import ConfidenceUpdater

# ── Arbitration ────────────────────────────────────────────────────────────────
from aua.arbiter import ArbiterAgent, VerdictCase, ArbiterVerdict

# ── Routing ────────────────────────────────────────────────────────────────────
from aua.router import Router

# ── Config ─────────────────────────────────────────────────────────────────────
from aua.config import (
    load_config,
    AUAConfig,
    SpecialistConfig,
    ArbiterConfig,
    RouterConfig,
    BlueGreenFieldConfig,
    FieldConfig,
    FIELD_CONFIGS,
    get_effective_config,
)

# ── REST endpoint models (#09) ─────────────────────────────────────────────────
from aua.endpoints import (
    QueryRequest,
    RouterResponse,
    BatchQueryRequest,
    BatchQueryResponse,
    CorrectionRequest,
    CorrectionResponse,
    ConfigResponse,
    DeployGreenRequest,
    DeployGreenResponse,
    HealthLiveResponse,
    HealthReadyResponse,
    HealthStartupResponse,
)

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
    # endpoint models
    "QueryRequest",
    "RouterResponse",
    "BatchQueryRequest",
    "BatchQueryResponse",
    "CorrectionRequest",
    "CorrectionResponse",
    "ConfigResponse",
    "DeployGreenRequest",
    "DeployGreenResponse",
    "HealthLiveResponse",
    "HealthReadyResponse",
    "HealthStartupResponse",
]