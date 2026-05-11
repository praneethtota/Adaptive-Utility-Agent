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
from aua.auth import AUAToken, TokenManager, get_token_manager, init_token_manager
from aua.blue_green import BlueGreenDeployment, EvaluationSummary
from aua.certs import generate_dev_certs, inspect_certs
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
from aua.hooks import HookRunner, get_hook_runner
from aua.hot_reload import HotReloader, ReloadResult
from aua.logging_config import configure_logging
from aua.metrics import AUAMetrics, get_metrics
from aua.middleware import AuditMiddleware, MiddlewarePipeline, PIIRedactionMiddleware
from aua.plugins.errors import ALL_ERROR_CODES, AUAErrorCode, get_error_code
from aua.plugins.interfaces import (
    ArbiterPolicyPlugin,
    CorrectionStorePlugin,
    FieldClassifierPlugin,
    HookPlugin,
    ModelBackendPlugin,
    PromotionPolicyPlugin,
    StateStorePlugin,
    UtilityScorerPlugin,
)
from aua.plugins.registry import PluginLoadError, PluginRegistry, get_registry, load_plugin
from aua.presets import AVAILABLE_PRESETS, PRESETS, PresetSpec, get_preset
from aua.router import Router
from aua.safety import SafetyConfig, SafetyPolicy
from aua.secrets import SecretsManager, get_secrets_manager, resolve_secret
from aua.session import SessionContext, get_current, get_current_or_none, new_session_context
from aua.state import FilesStateStore, SQLiteStateStore, get_state_store
from aua.templates.registry import get_template, list_templates, render_template
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
    # v0.8 — plugin interfaces
    "FieldClassifierPlugin",
    "UtilityScorerPlugin",
    "ArbiterPolicyPlugin",
    "PromotionPolicyPlugin",
    "CorrectionStorePlugin",
    "ModelBackendPlugin",
    "StateStorePlugin",
    "HookPlugin",
    # v0.8 — plugin registry
    "load_plugin",
    "PluginRegistry",
    "PluginLoadError",
    "get_registry",
    # v0.8 — error codes
    "AUAErrorCode",
    "ALL_ERROR_CODES",
    "get_error_code",
    # v0.8 — hooks
    "HookRunner",
    "get_hook_runner",
    # v0.8 — middleware
    "MiddlewarePipeline",
    "PIIRedactionMiddleware",
    "AuditMiddleware",
    # v0.8 — state store
    "SQLiteStateStore",
    "FilesStateStore",
    "get_state_store",
    # v0.8 — safety
    "SafetyPolicy",
    "SafetyConfig",
    # v0.9 — security & auth
    "AUAToken",
    "TokenManager",
    "get_token_manager",
    "init_token_manager",
    # v0.9 — certs
    "generate_dev_certs",
    "inspect_certs",
    # v0.9 — session
    "SessionContext",
    "new_session_context",
    "get_current",
    "get_current_or_none",
    # v0.9 — secrets
    "SecretsManager",
    "get_secrets_manager",
    "resolve_secret",
    # v0.9 — metrics
    "AUAMetrics",
    "get_metrics",
    # v0.9 — logging
    "configure_logging",
    # v0.8 — templates
    "get_template",
    "render_template",
    "list_templates",
    # hot reload
    "HotReloader",
    "PRESETS",
    "AVAILABLE_PRESETS",
    "PresetSpec",
    "get_preset",
    "ReloadResult",
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
