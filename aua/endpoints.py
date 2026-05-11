"""
aua/endpoints.py — Pydantic request/response models for all AUA REST endpoints.

Separating models from routing logic keeps router.py focused on routing,
makes models independently testable, and gives a single place to version
the API contract.

Imported by:
    aua/router.py   — mounts these models on the FastAPI app
    aua/__init__.py — re-exports for library consumers

Model groups:
    Query       QueryRequest, RouterResponse
    Batch       BatchQueryRequest, BatchQueryResponse
    Stream      StreamStartEvent, StreamChunkEvent, StreamDoneEvent, StreamErrorEvent
    Corrections CorrectionRequest, CorrectionResponse, CorrectionListItem, CorrectionListResponse
    Config      ConfigResponse, SpecialistInfo, ArbiterInfo, RouterInfo
    Deploy      DeployGreenRequest, DeployGreenResponse
    Health      HealthLiveResponse, HealthReadyResponse, HealthStartupResponse
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Route a single query through the Micro-Expert Architecture."""

    query: str = Field(
        ...,
        description="The query to route through the specialist graph.",
        examples=["Write a binary search function in Python. State the time complexity."],
    )
    session_id: Optional[str] = Field(
        "default",
        description="Session identifier. Used to scope cross-session assertions.",
    )
    conversation_history: Optional[List[dict]] = Field(
        [],
        description="Prior turns as a list of {role: str, content: str} dicts.",
    )
    force_domain: Optional[str] = Field(
        None,
        description=(
            "Pin routing to a specific domain (e.g. 'software_engineering'). "
            "Bypasses the field classifier. Useful for testing."
        ),
    )

    model_config = {"json_schema_extra": {
        "example": {
            "query": "Implement merge sort in Python. State time and space complexity.",
            "session_id": "user_42",
            "conversation_history": [],
            "force_domain": None,
        }
    }}


class RouterResponse(BaseModel):
    """Full response from the routing pipeline, including scoring and telemetry."""

    query: str
    routing_mode: str = Field(
        ...,
        description="How the query was routed: 'single' | 'fanout' | 'arbiter'.",
    )
    domain_distribution: Dict[str, float] = Field(
        ...,
        description="Field classifier output: {domain: probability}.",
    )
    primary_domain: str = Field(
        ...,
        description=(
            "Domain of the winning specialist "
            "(or 'general' for arbiter fallback)."
        ),
    )
    response: str = Field(..., description="The specialist's response text.")
    u_score: float = Field(
        ..., description="Utility score U = w_e·E + w_c·C + w_k·K."
    )
    confidence: float = Field(
        ..., description="Updated confidence after contradiction check."
    )
    contradictions_detected: int
    dpo_pairs_generated: int
    latency_ms: float
    specialist_responses: Optional[List[dict]] = Field(
        None,
        description=(
            "Per-specialist responses "
            "(only populated for fanout routing)."
        ),
    )


# ── Batch ─────────────────────────────────────────────────────────────────────

class BatchQueryRequest(BaseModel):
    """Route a list of queries concurrently."""

    queries: List[str] = Field(
        ...,
        description="Queries to process. Each is routed independently.",
        min_length=1,
        max_length=100,
    )
    session_id: Optional[str] = Field(
        "default",
        description="Shared session ID for all queries in the batch.",
    )
    max_parallel: Optional[int] = Field(
        4,
        ge=1,
        le=32,
        description="Maximum concurrent specialist calls. Default 4.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "queries": [
                "Write binary search in Python.",
                "Implement quicksort in Python.",
            ],
            "session_id": "batch_session_1",
            "max_parallel": 4,
        }
    }}


class BatchQueryResponse(BaseModel):
    """Results for a batch query request."""

    results: List[RouterResponse] = Field(
        ..., description="One result per successful query."
    )
    total_latency_ms: float = Field(
        ..., description="Wall-clock time for the entire batch."
    )
    n_queries: int = Field(..., description="Number of queries submitted.")
    n_errors: int = Field(
        ...,
        description="Number of queries that failed (excluded from results).",
    )


# ── Stream ────────────────────────────────────────────────────────────────────
# Server-Sent Events payload models for POST /query/stream.
# Wire format per event:  data: {json}\n\n
# Content-Type:           text/event-stream

class StreamStartEvent(BaseModel):
    """
    First SSE event. Confirms routing decision before any tokens arrive.
    Clients can use this to display 'Routing to {primary_domain}…' UI.
    """

    type: str = Field("start", description="Always 'start'.")
    routing_mode: str = Field(
        ..., description="'single' | 'fanout' | 'arbiter'."
    )
    primary_domain: str = Field(
        ..., description="Domain of the specialist that will respond."
    )
    domain_distribution: Dict[str, float] = Field(
        ..., description="Full field classifier output."
    )


class StreamChunkEvent(BaseModel):
    """
    One token (or small delta) from the specialist.
    Emitted as tokens arrive from the vLLM / Ollama backend.
    """

    type: str = Field("chunk", description="Always 'chunk'.")
    text: str = Field(..., description="Token text. May be empty string.")
    index: int = Field(..., description="Zero-based token index within this response.")


class StreamDoneEvent(BaseModel):
    """
    Final SSE event. Carries complete metadata — identical fields to RouterResponse.
    Clients should use this to update their UI with scores and telemetry.
    After this event the stream closes.
    """

    type: str = Field("done", description="Always 'done'.")
    full_response: str = Field(
        ..., description="Complete specialist response (all chunks concatenated)."
    )
    routing_mode: str
    primary_domain: str
    domain_distribution: Dict[str, float]
    u_score: float = Field(..., description="U = w_e·E + w_c·C + w_k·K.")
    confidence: float
    contradictions_detected: int
    dpo_pairs_generated: int
    latency_ms: float


class StreamErrorEvent(BaseModel):
    """
    Emitted if the specialist is unreachable or returns an error mid-stream.
    After this event the stream closes.
    """

    type: str = Field("error", description="Always 'error'.")
    code: int = Field(..., description="HTTP-equivalent error code.")
    message: str = Field(..., description="Human-readable error description.")


# ── Corrections ───────────────────────────────────────────────────────────────

class CorrectionRequest(BaseModel):
    """
    Manually inject a verified fact into the cross-session assertions store.

    Injected corrections are used to prime specialist prompts on future queries
    to the same domain/subject, reducing repeated errors without retraining.
    """

    subject: str = Field(
        ...,
        description="Short subject label, e.g. 'bubble_sort_complexity'.",
        max_length=200,
    )
    domain: str = Field(
        ...,
        description=(
            "Field name, e.g. 'software_engineering'. "
            "Must match a key in FIELD_CONFIGS."
        ),
    )
    claim: str = Field(
        ...,
        description=(
            "The correct claim to store, "
            "e.g. 'Bubble sort is O(n²) average case.'"
        ),
        max_length=2000,
    )
    confidence: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in this correction (0.0–1.0). "
            "Defaults to 0.9 for manual entries."
        ),
    )
    source: Optional[str] = Field(
        "manual",
        description="Source label: 'manual' | 'arbiter' | 'external_api' | etc.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "subject": "bubble_sort_complexity",
            "domain": "software_engineering",
            "claim": "Bubble sort has O(n²) average and worst-case time complexity.",
            "confidence": 0.95,
            "source": "manual",
        }
    }}


class CorrectionResponse(BaseModel):
    """Confirmation that a correction was stored."""

    stored: bool
    subject: str
    domain: str
    claim: str
    confidence: float = Field(
        ..., description="Effective confidence at write time."
    )
    decay_class: str = Field(
        ...,
        description=(
            "Assigned decay class: "
            "A (no decay) | B (10yr) | C (3yr) | D (6mo)."
        ),
    )


class CorrectionListItem(BaseModel):
    subject: str
    domain: str
    claim: str
    effective_confidence: float
    decay_class: str
    source: str


class CorrectionListResponse(BaseModel):
    total: int
    returned: int
    corrections: List[CorrectionListItem]


# ── Config ────────────────────────────────────────────────────────────────────

class SpecialistInfo(BaseModel):
    name: str
    model: str
    port: int
    field: str
    endpoint: str


class ArbiterInfo(BaseModel):
    model: str
    port: int
    endpoint: str


class RouterInfo(BaseModel):
    port: int
    single_domain_threshold: float
    fanout_threshold: float
    specialist_timeout: float


class ConfigResponse(BaseModel):
    """Read-only view of the loaded aua_config.yaml."""

    version: str
    mode: str
    backend: str
    specialists: List[SpecialistInfo]
    arbiter: ArbiterInfo
    router: RouterInfo


# ── Deploy / blue-green ───────────────────────────────────────────────────────

class DeployGreenRequest(BaseModel):
    """Trigger a blue-green promotion evaluation."""

    specialist: str = Field(
        ...,
        description="Specialist name to evaluate, e.g. 'swe'.",
    )
    green_model: str = Field(
        ...,
        description="Path or HuggingFace ID of the GREEN model candidate.",
    )
    n_eval_queries: Optional[int] = Field(
        10,
        ge=1,
        le=100,
        description="Number of evaluation queries to run. Default 10.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "specialist": "swe",
            "green_model": "./models/swe_green_v1",
            "n_eval_queries": 10,
        }
    }}


class DeployGreenResponse(BaseModel):
    """Result of a blue-green promotion evaluation."""

    specialist: str
    promoted: bool = Field(
        ..., description="True if GREEN was promoted to production."
    )
    u_delta: float = Field(
        ...,
        description="U_green - U_blue. Must exceed threshold to promote.",
    )
    blue_u: float
    green_u: float
    threshold: float = Field(
        ..., description="Minimum U_delta required for promotion."
    )
    message: str = Field(
        ...,
        description="Human-readable explanation of the promotion decision.",
    )


# ── Health ────────────────────────────────────────────────────────────────────

class HealthLiveResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field(
        "alive", description="Always 'alive' if the process is running."
    )
    uptime_s: float


class HealthReadyResponse(BaseModel):
    """Readiness probe response."""

    status: str = Field(
        ...,
        description="'ready' if all specialists are reachable.",
    )
    specialists: Dict[str, str] = Field(
        ...,
        description=(
            "Per-specialist status: "
            "'ok' | 'unreachable' | 'http_NNN'."
        ),
    )


class HealthStartupResponse(BaseModel):
    """Startup probe response."""

    status: str = Field(
        ...,
        description="'started' once the first readiness check passed.",
    )
    uptime_s: float