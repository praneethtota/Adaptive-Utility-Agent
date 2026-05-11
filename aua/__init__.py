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

# ── Version ────────────────────────────────────────────────────────────────────
# ── Arbitration ────────────────────────────────────────────────────────────────
from aua.arbiter import ArbiterAgent, ArbiterVerdict, VerdictCase
from aua.assertions_store import AssertionMatch, AssertionsStore
from aua.confidence_updater import ConfidenceUpdater

# ── Config ─────────────────────────────────────────────────────────────────────
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

# ── REST endpoint models (#09 / #10) ──────────────────────────────────────────
from aua.endpoints import (
    BatchQueryRequest,
    BatchQueryResponse,
    ConfigResponse,
    CorrectionRequest,
    CorrectionResponse,
    DeployGreenRequest,
    DeployGreenResponse,
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

# ── Core pipeline components ───────────────────────────────────────────────────
from aua.field_classifier import FieldClassifier

# ── Routing ────────────────────────────────────────────────────────────────────
from aua.router import Router
from aua.utility_scorer import DomainState, TaskScore, UtilityScorer
from aua.version import __version__

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
    "load_tier",
    "AVAILABLE_TIERS",
    "TIER_ALIASES",
    "RuntimeConfig",
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
    "StreamStartEvent",
    "StreamChunkEvent",
    "StreamDoneEvent",
    "StreamErrorEvent",
    "CorrectionRequest",
    "CorrectionResponse",
    "ConfigResponse",
    "DeployGreenRequest",
    "DeployGreenResponse",
    "HealthLiveResponse",
    "HealthReadyResponse",
    "HealthStartupResponse",
]
