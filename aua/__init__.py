"""
aua — Adaptive Utility Agents: a deployable specialist framework.

Install:
    pip install adaptive-utility-agent          # runtime only
    pip install "adaptive-utility-agent[vllm]"  # + GPU serving
    pip install "adaptive-utility-agent[dev]"   # + dev tools

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

from aua.arbiter import ArbiterAgent, ArbiterVerdict, VerdictCase
from aua.assertions_store import AssertionMatch, AssertionsStore
from aua.blue_green import BlueGreenDeployment, EvaluationSummary
from aua.confidence_updater import ConfidenceUpdater
from aua.config import (
    AVAILABLE_TIERS,
    FIELD_CONFIGS,
    TIER_ALIASES,
    ArbiterConfig,
    AUAConfig,
    BlueGreenFieldConfig,
    FieldConfig,
    RouterConfig,
    RuntimeConfig,
    SpecialistConfig,
    get_effective_config,
    load_config,
    load_tier,
)
from aua.contradiction_detector import ContradictionDetector, ContradictionResult
from aua.correction_loop import CollectionSummary, CorrectionLoop, DPOPair
from aua.endpoints import (
    BatchQueryRequest,
    BatchQueryResponse,
    ConfigResponse,
    CorrectionRequest,
    CorrectionResponse,
    DeployGreenRequest,
    DeployGreenResponse,
    ErrorResponse,
    HealthLiveResponse,
    HealthReadyResponse,
    HealthStartupResponse,
    QueryRequest,
    RouterResponse,
    StreamChunkEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
)
from aua.field_classifier import FieldClassifier
from aua.router import Router
from aua.utility_scorer import DomainState, TaskScore, UtilityScorer
from aua.version import __version__

# ── Aliases ────────────────────────────────────────────────────────────────────
# Arbiter is the canonical public name; ArbiterAgent is the implementation class.
Arbiter = ArbiterAgent

__all__ = [
    # version
    "__version__",
    # config
    "load_config",
    "load_tier",
    "AUAConfig",
    "SpecialistConfig",
    "ArbiterConfig",
    "RouterConfig",
    "BlueGreenFieldConfig",
    "FieldConfig",
    "RuntimeConfig",
    "FIELD_CONFIGS",
    "AVAILABLE_TIERS",
    "TIER_ALIASES",
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
    "Arbiter",
    "VerdictCase",
    "ArbiterVerdict",
    # routing
    "Router",
    # deployment & learning
    "BlueGreenDeployment",
    "EvaluationSummary",
    "CorrectionLoop",
    "DPOPair",
    "CollectionSummary",
    # endpoint models
    "QueryRequest",
    "RouterResponse",
    "BatchQueryRequest",
    "BatchQueryResponse",
    "StreamStartEvent",
    "StreamChunkEvent",
    "StreamDoneEvent",
    "StreamErrorEvent",
    "CorrectionRequest",
    "CorrectionResponse",
    "ErrorResponse",
    "ConfigResponse",
    "DeployGreenRequest",
    "DeployGreenResponse",
    "HealthLiveResponse",
    "HealthReadyResponse",
    "HealthStartupResponse",
]
