"""
aua/router.py — Micro-Expert Architecture Router

Config-driven: all specialist endpoints, thresholds, and ports come from
AUAConfig (loaded from aua_config.yaml). No hardcoded values.

REST endpoints are defined here; request/response models live in aua/endpoints.py.

Usage (programmatic):
    from aua import Router
    from aua.config import load_config

    config = load_config("aua_config.yaml")
    router = Router.from_config(config)
    result = await router.query("Write binary search. State time complexity.")

Usage (CLI — preferred):
    aua serve
    aua serve --config /path/to/aua_config.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi import Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from aua.assertions_store import AssertionsStore
from aua.confidence_updater import ConfidenceUpdater
from aua.config import FIELD_CONFIGS, AUAConfig
from aua.contradiction_detector import ContradictionDetector
from aua.endpoints import (
    ArbiterInfo,
    BatchQueryRequest,
    BatchQueryResponse,
    ConfigResponse,
    CorrectionListItem,
    CorrectionListResponse,
    CorrectionRequest,
    CorrectionResponse,
    DeployGreenRequest,
    DeployGreenResponse,
    HealthLiveResponse,
    HealthReadyResponse,
    HealthStartupResponse,
    QueryRequest,
    RouterInfo,
    RouterResponse,
    SpecialistInfo,
    StreamChunkEvent,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamStartEvent,
)
from aua.field_classifier import FieldClassifier
from aua.hooks import get_hook_runner
from aua.utility_scorer import UtilityScorer

log = logging.getLogger("aua.router")

_SSE_CONTENT_TYPE = "text/event-stream"
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
    "Content-Encoding": "none",  # prevent gzip middleware from buffering the stream
}
_SSE_HEARTBEAT_INTERVAL = 15  # seconds between SSE keep-alive comments


def _sse(event: BaseModel) -> str:
    """Serialise a Pydantic model as a full SSE frame with named event field.

    Format:
        event: <event.type>
        data: <json>
        \n
    This allows clients to use addEventListener('chunk', handler) etc.
    """
    event_type = getattr(event, "type", "message")
    return f"event: {event_type}\ndata: {event.model_dump_json()}\n\n"


def _sse_comment(text: str = "keep-alive") -> str:
    """SSE comment frame — keeps the connection alive without triggering event handlers."""
    return f": {text}\n\n"


# ── Chat Session request models ───────────────────────────────────────────────
# Must be module-level — Pydantic v2 cannot handle locally-defined models.


class CreateSessionRequest(BaseModel):
    """Request body for POST /sessions."""

    title: str = ""
    name: str = ""  # alias for title — tutorial uses "name", both accepted

    def effective_title(self) -> str:
        return self.name or self.title or ""


class SendMessageRequest(BaseModel):
    """Request body for POST /sessions/{id}/messages and POST /sessions/{id}/stream."""

    content: str = ""  # canonical field name
    query: str = ""  # alias used by the tutorial (POST /sessions/{id}/stream examples)
    stream: bool = False

    def effective_content(self) -> str:
        return self.content or self.query or ""


def _audit_query(
    req: QueryRequest,
    domain: str,
    routing_mode: str,
    u_score: float,
    confidence: float,
    latency_ms: float,
    contradictions: int,
) -> None:
    """Fire-and-forget: append a query event to the audit log."""
    try:
        from aua.session import get_current_or_none
        from aua.state import get_state_store

        ctx = get_current_or_none()
        store = get_state_store()
        store.append_audit(
            {
                "event_type": "query",
                "session_id": ctx.session_id if ctx else getattr(req, "session_id", "unknown"),
                "trace_id": ctx.trace_id if ctx else "",
                "request_id": ctx.request_id if ctx else "",
                "field": domain,
                "routing_mode": routing_mode,
                "u_score": u_score,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "details": {"contradictions": contradictions},
            }
        )
    except Exception as _audit_err:
        import logging as _log

        _log.getLogger("aua.router").debug("Audit log write failed: %s", _audit_err)


def _persist_assertion_results(
    policy_result: Any,
    session_id: str,
    domain: str,
    policy_name: str,
) -> None:
    """
    Persist assertion results to the state store and fire Prometheus metrics.

    Called after policy.run() on every query that has an active policy.
    Results are stored in the assertion_events table with timestamps so
    ``aua logs`` and ``aua calibrate --layer 3`` can query them.
    """
    try:
        from aua.metrics import get_metrics
        from aua.state import get_state_store

        store = get_state_store()
        metrics = get_metrics()

        for r in policy_result.results:
            store.append(
                "assertion_events",
                {
                    "session_id": session_id,
                    "assertion_name": r.assertion_name,
                    "level": r.level.value,
                    "passed": 1 if r.passed else 0,
                    "bonus_applied": r.bonus_applied,
                    "retries_used": r.retries_used,
                    "message": r.message or "",
                    "domain": domain,
                    "policy_name": policy_name,
                    "latency_ms": r.latency_ms,
                },
            )
            metrics.record_assertion(
                assertion_name=r.assertion_name,
                level=r.level.value,
                passed=r.passed,
                domain=domain,
                retries=r.retries_used,
                bonus=r.bonus_applied,
                policy_name=policy_name,
            )
    except Exception as _err:
        import logging as _log

        _log.getLogger("aua.router").debug("Assertion persist failed: %s", _err)


class Router:
    """
    Micro-Expert Architecture Router.

    Routes each query to the right specialist based on field classification,
    fans out to multiple specialists on cross-domain queries, runs the Arbiter
    on conflicts, and accumulates DPO pairs from detected contradictions.

    All routing parameters come from AUAConfig — no hardcoded ports or
    thresholds anywhere in this class.
    """

    def __init__(self, config: AUAConfig, config_path: str | None = None) -> None:
        self._config = config
        self._config_path = config_path  # for PATCH /config persist
        self._classifier = FieldClassifier()
        self._scorer = UtilityScorer()
        self._store = AssertionsStore()
        self._detector = ContradictionDetector(penalty_multiplier=2.0)

        # P0: persistent state store (conversations, messages, model_runs, etc.)
        from aua.state import SQLiteStateStore

        _db_path = getattr(config, "state", None)
        _db_path = (
            getattr(_db_path, "path", ".aua/state/aua.db") if _db_path else ".aua/state/aua.db"
        )
        self._state_store = SQLiteStateStore(db_path=_db_path)
        self._conf = ConfidenceUpdater()

        self._domain_confidence: dict[str, float] = {s.field: 0.5 for s in config.specialists}
        self._field_to_url: dict[str, str] = {s.field: s.endpoint for s in config.specialists}
        self._arbiter_url = config.arbiter.endpoint
        self._single_threshold = config.router.single_domain_threshold
        self._fanout_threshold = config.router.fanout_threshold
        self._timeout = config.router.specialist_timeout
        self._arbitration_mode = config.router.arbitration_mode  # "pairwise" | "vcg"

        self._start_time: float = time.time()
        self._queries_by_mode: dict[str, int] = {"single": 0, "fanout": 0, "arbiter": 0}
        self._latencies_ms: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._requests_per_spec: dict[str, int] = {s.name: 0 for s in config.specialists}
        self._requests_per_spec["arbiter"] = 0
        self._total_contradictions: int = 0
        self._total_dpo: int = 0
        self._verdict_counts: dict[str, int] = {"case_1": 0, "case_2": 0, "case_3": 0, "case_4": 0}
        self._started: bool = False

        self.app = self._build_app()
        log.info(
            "Router initialised — %d specialist(s), arbiter on port %d",
            len(config.specialists),
            config.arbiter.port,
        )

    @classmethod
    def from_config(cls, config: AUAConfig, config_path: str | None = None) -> Router:
        return cls(config, config_path=config_path)

    async def query(
        self,
        query: str,
        session_id: str = "default",
        conversation_history: list[dict] | None = None,
        force_domain: str | None = None,
    ) -> RouterResponse:
        """Route a query programmatically (library API, not HTTP)."""
        req = QueryRequest(
            query=query,
            session_id=session_id,
            conversation_history=conversation_history or [],
            force_domain=force_domain,
        )
        return await self._handle(req)

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="AUA Micro-Expert Router",
            description=(
                "**Adaptive Utility Agents** — routes queries to specialist LLM models, "
                "arbitrates cross-domain conflicts, and accumulates DPO corrections.\n\n"
                "Config-driven via `aua_config.yaml`. "
                "Start with `aua serve`, inspect live with `aua status`.\n\n"
                "Utility function: **U = w_e·E + w_c·C + w_k·K**"
            ),
            version="0.5.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_tags=[
                {"name": "query", "description": "Route queries through the specialist graph."},
                {"name": "batch", "description": "Batch inference — multiple queries in parallel."},
                {
                    "name": "stream",
                    "description": "Token-by-token streaming via Server-Sent Events.",
                },
                {
                    "name": "health",
                    "description": "Kubernetes-compatible liveness / readiness / startup probes.",
                },
                {
                    "name": "corrections",
                    "description": "Inject and inspect the cross-session corrections store.",
                },
                {"name": "config", "description": "Inspect the running configuration (read-only)."},
                {"name": "deploy", "description": "Blue-green promotion evaluation."},
                {
                    "name": "telemetry",
                    "description": "Live telemetry — latency, U scores, routing breakdown.",
                },
            ],
        )
        # Rate limiting — before CORS
        from aua.rate_limit import RateLimitMiddleware

        app.add_middleware(RateLimitMiddleware, config=self._config)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=self._config.router.cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

        # ── Query ──────────────────────────────────────────────────────────

        @app.post(
            "/query",
            response_model=RouterResponse,
            tags=["query"],
            summary="Route a single query through the specialist graph",
        )
        async def route(req: QueryRequest) -> RouterResponse:
            """
            Route a query through the Micro-Expert Architecture:

            1. **Classify** the query domain (field classifier)
            2. **Route** to the correct specialist, or fan out across multiple
            3. **Arbitrate** cross-domain conflicts via the Arbiter Agent
            4. **Score** with U = w_e·E + w_c·C + w_k·K
            5. **Accumulate** DPO pairs from detected contradictions

            Use `POST /query/stream` for token-by-token streaming.
            """
            return await self._handle(req)

        # ── Stream ─────────────────────────────────────────────────────────

        @app.post(
            "/query/stream",
            tags=["stream"],
            summary="Stream a query response token-by-token (Server-Sent Events)",
            response_class=StreamingResponse,
            responses={
                200: {
                    "description": "SSE stream. Content-Type: text/event-stream.",
                    "content": {
                        "text/event-stream": {
                            "schema": {"type": "string"},
                            "example": (
                                'data: {"type":"start","routing_mode":"single",'
                                + '"primary_domain":"software_engineering",'
                                + '"domain_distribution":{"software_engineering":0.92}}\n\n'
                                + 'data: {"type":"chunk","text":"def ","index":0}\n\n'
                                + 'data: {"type":"done","full_response":"def binary_search():...","u_score":0.633,"confidence":0.72,"contradictions_detected":0,"dpo_pairs_generated":0,"latency_ms":1240.5,"routing_mode":"single","primary_domain":"software_engineering","domain_distribution":{"software_engineering":0.92}}\n\n'
                            ),
                        }
                    },
                },
                503: {"description": "Specialist unreachable."},
            },
        )
        async def stream_route(req: QueryRequest) -> StreamingResponse:
            """
            Stream a specialist response token-by-token using Server-Sent Events.

            **SSE event sequence:**
            ```
            data: {"type":"start", "routing_mode":"single", "primary_domain":"...", ...}
            data: {"type":"chunk", "text":"def ", "index":0}
            data: {"type":"chunk", "text":"binary_search", "index":1}
            ...
            data: {"type":"done", "full_response":"...", "u_score":0.633, ...}
            ```

            **On error:**
            ```
            data: {"type":"error", "code":503, "message":"Specialist unreachable"}
            ```

            **Routing behaviour:**
            - *Single domain* → tokens stream directly from the specialist
            - *Fanout/arbiter* → responses buffered for arbitration first,
              winning response emitted as single chunk then done

            **Client example (Python):**
            ```python
            import httpx, json
            with httpx.stream("POST", "http://localhost:8000/query/stream",
                              json={"query": "Write binary search in Python."}) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event["type"] == "chunk":
                            print(event["text"], end="", flush=True)
                        elif event["type"] == "done":
                            print(f"\\nU={event['u_score']}")
            ```
            """
            return StreamingResponse(
                self._handle_stream(req),
                media_type=_SSE_CONTENT_TYPE,
                headers=_SSE_HEADERS,
            )

        # ── Batch ──────────────────────────────────────────────────────────

        @app.post(
            "/query/batch",
            response_model=BatchQueryResponse,
            tags=["batch"],
            summary="Route multiple queries in parallel",
        )
        async def batch_route(req: BatchQueryRequest) -> BatchQueryResponse:
            """
            Route a list of queries concurrently, up to `max_parallel` at once.
            Returns one RouterResponse per query plus aggregate stats.
            Failed queries are excluded from results and counted in n_errors.
            """
            return await self._handle_batch(req)

        # ── Health ─────────────────────────────────────────────────────────

        @app.get(
            "/health/live",
            response_model=HealthLiveResponse,
            tags=["health"],
            summary="Liveness probe — is the router process alive?",
        )
        async def health_live() -> HealthLiveResponse:
            """Always returns 200 if the router process is running."""
            return HealthLiveResponse(
                status="live",
                uptime_s=round(time.time() - self._start_time, 1),
            )

        @app.get(
            "/health/ready",
            response_model=HealthReadyResponse,
            tags=["health"],
            summary="Readiness probe — are all specialists reachable?",
            responses={503: {"description": "One or more specialists unreachable"}},
        )
        async def health_ready() -> HealthReadyResponse:
            """Returns 200 only if all specialists are reachable; 503 otherwise."""
            health = await self._health()
            specialists = health["specialists"]
            all_up = all(v == "ok" for v in specialists.values())
            if not all_up:
                down = [k for k, v in specialists.items() if v != "ok"]
                raise HTTPException(
                    status_code=503,
                    detail={"status": "not_ready", "down": down, "specialists": specialists},
                )
            self._started = True
            return HealthReadyResponse(status="ready", specialists=specialists)

        @app.get(
            "/health/startup",
            response_model=HealthStartupResponse,
            tags=["health"],
            summary="Startup probe — has the framework finished initialising?",
            responses={503: {"description": "Specialists not yet healthy"}},
        )
        async def health_startup() -> HealthStartupResponse:
            """Returns 200 once the first readiness check has passed."""
            if not self._started:
                health = await self._health()
                if all(v == "ok" for v in health["specialists"].values()):
                    self._started = True
            if not self._started:
                raise HTTPException(
                    503,
                    detail={"status": "starting", "message": "Specialists not yet healthy"},
                )
            return HealthStartupResponse(
                status="started",
                uptime_s=round(time.time() - self._start_time, 1),
            )

        @app.get("/health", tags=["health"], include_in_schema=False)
        async def health_legacy():
            return await self._health()

        # ── Corrections ────────────────────────────────────────────────────

        # ── P0: Conversation + message persistence ──────────────────────────────
        # Implemented as part of Phase 13 backport from AUA-Veritas.
        # Key implementation rules carried forward:
        #   1. asyncio imported at TOP of lifespan/mount — never mid-body.
        #   2. model_runs.conversation_id must be explicit — no closure capture.
        #   3. Cache bypass when limit < 50 (non-default).
        #   4. fire_and_forget() for all post-response DB writes.

        from aua.state import MessageCache

        _msg_cache = MessageCache()

        @app.post(
            "/conversations",
            tags=["conversations"],
            summary="Create a new conversation",
            status_code=201,
        )
        async def create_conversation(body: dict | None = None) -> dict:
            """Create a new conversation, optionally in a project."""
            b = body or {}
            title = b.get("title", "New Chat")
            project_id = b.get("project_id")
            user_id = b.get("user_id", "local")
            conv = self._state_store.create_conversation(
                title=title, project_id=project_id, user_id=user_id
            )
            return conv

        @app.get(
            "/conversations",
            tags=["conversations"],
            summary="List conversations",
        )
        async def list_conversations(
            project_id: str | None = QueryParam(
                None, description="Filter by project ID. Omit for all conversations."
            ),
            user_id: str = QueryParam("local", description="User identifier."),
            limit: int = QueryParam(1000, ge=1, le=10000),
        ) -> list:
            """List conversations, optionally filtered by project."""
            return self._state_store.list_conversations(
                user_id=user_id, project_id=project_id, limit=limit
            )

        @app.patch(
            "/conversations/{conversation_id}/title",
            tags=["conversations"],
            summary="Rename a conversation",
        )
        async def rename_conversation(conversation_id: str, body: dict) -> dict:
            """Update the title of a conversation."""
            title = body.get("title", "").strip()
            if not title:
                from fastapi import HTTPException

                raise HTTPException(400, "title must not be empty")
            self._state_store.rename_conversation(conversation_id, title)
            return {"conversation_id": conversation_id, "title": title}

        @app.get(
            "/conversations/{conversation_id}/messages",
            tags=["conversations"],
            summary="Paginated message fetch",
        )
        async def get_conv_messages(
            conversation_id: str,
            limit: int = QueryParam(50, ge=1, le=500),
            before: float | None = QueryParam(
                None, description="Cursor: return messages before this timestamp."
            ),
            after: float | None = QueryParam(
                None, description="Cursor: return messages after this timestamp."
            ),
        ) -> list:
            """
            Return messages for a conversation, paginated by timestamp cursor.

            Cache rule (Phase 13): only serve from cache when limit >= 50 (default).
            A non-default limit bypasses cache and hits DB with the actual limit.
            """
            # Cache-first for default first-page loads only
            if before is None and after is None:
                cached = _msg_cache.get(conversation_id, limit=limit)
                if cached is not None:
                    return cached

            messages = self._state_store.get_messages(
                conversation_id, limit=limit, before=before, after=after
            )

            # Populate cache for default first-page loads
            if before is None and after is None:
                _msg_cache.set(conversation_id, messages)

            return messages

        @app.post(
            "/projects",
            tags=["conversations"],
            summary="Create a project",
            status_code=201,
        )
        async def create_project(body: dict) -> dict:
            """Create a project for grouping conversations."""
            import uuid as _uuid_proj

            name = body.get("name", "").strip()
            if not name:
                from fastapi import HTTPException

                raise HTTPException(400, "name must not be empty")
            project_id = str(_uuid_proj.uuid4())
            user_id = body.get("user_id", "local")
            import time as _time_proj

            self._state_store.append(
                "projects",
                {
                    "project_id": project_id,
                    "user_id": user_id,
                    "name": name,
                    "created_at": _time_proj.time(),
                },
            )
            return {"project_id": project_id, "name": name, "user_id": user_id}

        @app.get(
            "/projects",
            tags=["conversations"],
            summary="List projects",
        )
        async def list_projects(
            user_id: str = QueryParam("local"),
        ) -> list:
            """List all projects for a user."""
            return self._state_store.query(
                "projects", filters={"user_id": user_id}, limit=1000, order_by="created_at ASC"
            )

        @app.post(
            "/conversations/{conversation_id}/model-run",
            tags=["conversations"],
            summary="Record a model run",
            status_code=201,
        )
        async def record_model_run(conversation_id: str, body: dict) -> dict:
            """
            Record a model run for a query in a conversation.

            Implementation rule (Phase 13): conversation_id is passed as an
            EXPLICIT parameter — never rely on closure capture. Without it there
            is no join path between a conversation and its model runs.
            """
            from aua.state import fire_and_forget

            run = {**body, "conversation_id": conversation_id}

            async def _store():
                self._state_store.record_model_run(run)

            fire_and_forget(_store())
            return {"ok": True}

        @app.get(
            "/context/backup/coverage",
            tags=["context"],
            summary="Context backup coverage report",
        )
        async def get_backup_coverage(
            specialist: str = QueryParam(
                None, description="Check coverage for a specific specialist."
            ),
        ) -> dict:
            """
            Return which conversations have valid context backups.

            A backup is VALID when backup.created_at > MAX(messages.created_at).
            Used by the 6-hour coverage job to find stale or missing backups.
            """
            if not specialist:
                return {"error": "specialist parameter required"}
            stale = self._state_store.stale_backup_conversations(specialist)
            return {
                "specialist": specialist,
                "stale_count": len(stale),
                "stale_conversations": stale,
            }

        # ── End P0: Conversation + message persistence ───────────────────────

        @app.post(
            "/corrections",
            response_model=CorrectionResponse,
            tags=["corrections"],
            summary="Inject a correction into the assertions store",
            status_code=200,
        )
        async def inject_correction(req: CorrectionRequest) -> CorrectionResponse:
            """
            Manually inject a verified fact into the cross-session assertions store.
            Stored corrections are injected into specialist prompts on future queries
            to the same domain/subject — reducing repeated errors without retraining.
            """
            assertion = self._store.add(
                subject=req.subject,
                domain=req.domain,
                claim=req.claim,
                confidence=req.confidence,
                source=req.source or "manual",
            )
            # ── on_correction hook (background — non-blocking) ────────────────
            get_hook_runner().fire_background(
                "on_correction",
                {
                    "session_id": "",
                    "trace_id": "",
                    "subject": assertion.subject,
                    "domain": assertion.domain,
                    "claim": assertion.claim,
                    "confidence": round(assertion.effective_confidence(), 4),
                    "decay_class": assertion.decay_class.value,
                    "source": req.source or "manual",
                },
            )
            return CorrectionResponse(
                stored=True,
                subject=assertion.subject,
                domain=assertion.domain,
                claim=assertion.claim,
                confidence=round(assertion.effective_confidence(), 4),
                decay_class=assertion.decay_class.value,
            )

        @app.get(
            "/corrections",
            response_model=CorrectionListResponse,
            tags=["corrections"],
            summary="List stored corrections",
        )
        async def list_corrections(
            subject: str | None = QueryParam(None, description="Filter by subject substring."),
            domain: str | None = QueryParam(None, description="Filter by domain name."),
            limit: int = QueryParam(50, ge=1, le=500, description="Max results."),
        ) -> CorrectionListResponse:
            """List corrections in the assertions store, filtered by subject and/or domain."""
            matches = self._store.query(subject=subject or "", domain=domain)[:limit]
            return CorrectionListResponse(
                total=self._store.summary()["total"],
                returned=len(matches),
                corrections=[
                    CorrectionListItem(
                        subject=m.assertion.subject,
                        domain=m.assertion.domain,
                        claim=m.assertion.claim,
                        effective_confidence=m.effective_confidence,
                        decay_class=m.assertion.decay_class.value,
                        source=m.assertion.source,
                    )
                    for m in matches
                ],
            )

        # ── Config ─────────────────────────────────────────────────────────

        @app.get(
            "/config",
            response_model=ConfigResponse,
            tags=["config"],
            summary="Return the running configuration (read-only)",
        )
        async def get_config() -> ConfigResponse:
            """Read-only view of the loaded aua_config.yaml."""
            return ConfigResponse(
                version=self._config.version,
                mode=self._config.mode,
                backend=self._config.backend,
                specialists=[
                    SpecialistInfo(
                        name=s.name,
                        model=s.model,
                        port=s.port,
                        field=s.field,
                        endpoint=s.endpoint,
                    )
                    for s in self._config.specialists
                ],
                arbiter=ArbiterInfo(
                    model=self._config.arbiter.model,
                    port=self._config.arbiter.port,
                    endpoint=self._config.arbiter.endpoint,
                ),
                router=RouterInfo(
                    port=self._config.router.port,
                    single_domain_threshold=self._config.router.single_domain_threshold,
                    fanout_threshold=self._config.router.fanout_threshold,
                    specialist_timeout=self._config.router.specialist_timeout,
                ),
            )

        @app.patch(
            "/config",
            tags=["config"],
            summary="Patch hot-reloadable config settings",
        )
        async def patch_config(body: dict) -> dict:
            """
            Partially update hot-reloadable config fields without restart.

            Supported fields:
              arbitration_mode   — "pairwise" | "vcg"
              single_domain_threshold — float 0.0–1.0
              fanout_threshold        — float 0.0–1.0

            Pass persist=true to write changes back to aua_config.yaml.

            Example body: {"arbitration_mode": "vcg", "persist": true}
            """
            import yaml as _yaml

            persist = bool(body.pop("persist", False))
            changed: dict[str, str] = {}

            if "arbitration_mode" in body:
                mode = str(body["arbitration_mode"])
                if mode not in ("pairwise", "vcg"):
                    from fastapi import HTTPException as _HE

                    raise _HE(422, f"arbitration_mode must be 'pairwise' or 'vcg', got {mode!r}")
                self._arbitration_mode = mode
                self._config.router.arbitration_mode = mode
                changed["arbitration_mode"] = mode
                log.info("arbitration_mode patched to %r", mode)

            if "single_domain_threshold" in body:
                v = float(body["single_domain_threshold"])
                self._single_threshold = v
                self._config.router.single_domain_threshold = v
                changed["single_domain_threshold"] = str(v)

            if "fanout_threshold" in body:
                v = float(body["fanout_threshold"])
                self._fanout_threshold = v
                self._config.router.fanout_threshold = v
                changed["fanout_threshold"] = str(v)

            if persist and self._config_path and changed:
                try:
                    cfg_path = self._config_path
                    raw = _yaml.safe_load(open(cfg_path).read()) or {}
                    raw.setdefault("router", {})
                    for k in changed:
                        raw["router"][k] = body.get(k)
                    with open(cfg_path, "w") as f:
                        _yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
                    log.info("Config patched and written to %s", cfg_path)
                except Exception as e:
                    log.warning("Failed to persist config patch: %s", e)

            return {"patched": changed, "persisted": persist and bool(changed)}

        # ── Deploy ─────────────────────────────────────────────────────────

        @app.post(
            "/deploy/green",
            response_model=DeployGreenResponse,
            tags=["deploy"],
            summary="Trigger a blue-green promotion evaluation",
        )
        async def deploy_green(req: DeployGreenRequest) -> DeployGreenResponse:
            """
            Compare GREEN model U score against BLUE baseline.
            Promotes if U_delta >= threshold from aua_config.yaml.
            Full harness integration is roadmap item #14.

            Note: Until the full evaluation harness is built (roadmap #14),
            this endpoint always returns dry_run_only=True.
            """
            # dry_run_only=True until full evaluation harness is built (roadmap #14)
            return await self._evaluate_green(req)

        # ── Telemetry ──────────────────────────────────────────────────────

        @app.get(
            "/status",
            tags=["telemetry"],
            summary="Full telemetry snapshot (powers aua status dashboard)",
        )
        async def full_status():
            """Complete telemetry: health, latency, U scores, routing, corrections, memory."""
            return await self._full_status()

        @app.post(
            "/reset", tags=["telemetry"], summary="Reset domain confidence and classifier history"
        )
        async def reset():
            """Reset domain confidence EMA and field classifier turn history."""
            self._classifier.reset_history()
            for field in self._domain_confidence:
                self._domain_confidence[field] = 0.5
            return {"status": "reset", "domain_confidence": self._domain_confidence}

        @app.get("/stats", tags=["telemetry"], include_in_schema=False)
        async def stats():
            return self._stats()

        @app.get("/version", tags=["meta"], summary="Return the running AUA Framework version")
        async def version():
            """Return version string. Stable — never returns 404."""
            from aua.version import __version__

            return {"version": __version__, "framework": "aua"}

        # ── Chat Session API (U-01) ───────────────────────────────────────────
        from aua.chat import (
            add_message,
            create_session,
            delete_session,
            ensure_chat_tables,
            get_messages,
            get_session,
            list_sessions,
        )

        ensure_chat_tables()

        @app.post("/sessions", tags=["sessions"], summary="Create a new chat session")
        async def sessions_create(req: CreateSessionRequest):
            return create_session(title=req.effective_title())

        @app.get("/sessions", tags=["sessions"], summary="List all chat sessions")
        async def sessions_list(limit: int = 50):
            return {"sessions": list_sessions(limit=limit)}

        @app.get("/sessions/{session_id}", tags=["sessions"])
        async def sessions_get(session_id: str):
            s = get_session(session_id)
            if not s:
                raise HTTPException(404, f"Session {session_id!r} not found")
            return s

        @app.delete("/sessions/{session_id}", tags=["sessions"])
        async def sessions_delete(session_id: str):
            ok = delete_session(session_id)
            if not ok:
                raise HTTPException(404, f"Session {session_id!r} not found")
            return {"deleted": session_id}

        @app.get("/sessions/{session_id}/messages", tags=["sessions"])
        async def sessions_messages(session_id: str, limit: int = 100):
            if not get_session(session_id):
                raise HTTPException(404, f"Session {session_id!r} not found")
            return {"messages": get_messages(session_id, limit=limit)}

        @app.post("/sessions/{session_id}/messages", tags=["sessions"])
        async def sessions_send(session_id: str, req: SendMessageRequest):
            if not get_session(session_id):
                raise HTTPException(404, f"Session {session_id!r} not found")

            # Store user message
            _msg_content = req.effective_content()
            add_message(session_id, role="user", content=_msg_content)

            # Build conversation history for the router
            history_msgs = get_messages(session_id, limit=20)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in history_msgs[:-1]  # exclude the message we just added
            ]

            # Route through the framework
            query_req = QueryRequest(
                query=_msg_content,
                session_id=session_id,
                conversation_history=history,
            )
            result = await self._handle(query_req)

            # Store assistant message
            add_message(
                session_id,
                role="assistant",
                content=result.response,
                domain=result.primary_domain,
                routing_mode=result.routing_mode,
                u_score=result.u_score,
                latency_ms=result.latency_ms,
            )

            return {
                "session_id": session_id,
                "message_id": str(uuid.uuid4()),
                "response": result.response,
                "domain": result.primary_domain,
                "routing_mode": result.routing_mode,
                "u_score": result.u_score,
                "latency_ms": result.latency_ms,
                "contradictions_detected": result.contradictions_detected,
            }

        @app.post(
            "/sessions/{session_id}/stream",
            tags=["sessions"],
            summary="Post a message to a session and stream the response as SSE",
        )
        async def sessions_stream(session_id: str, req: SendMessageRequest):
            """
            Stream a response from the routing pipeline for a session query.

            Emits Server-Sent Events with the following event types (Part 9.4):
              - route          — routing decision made
              - specialist_start — specialist call begins
              - chunk          — each token from the specialist
              - specialist_done — U score and latency for this specialist
              - done           — full response and complete metadata
              - error          — AUA error code and trace ID
            """
            if not get_session(session_id):
                raise HTTPException(404, f"Session {session_id!r} not found")

            # Store user message
            _stream_content = req.effective_content()
            add_message(session_id, role="user", content=_stream_content)

            # Build conversation history
            history_msgs = get_messages(session_id, limit=20)
            history = [{"role": m["role"], "content": m["content"]} for m in history_msgs[:-1]]

            async def _generate() -> AsyncIterator[str]:
                t0 = time.time()
                try:
                    # Emit route event
                    distribution = self._classifier.classify(_stream_content)
                    top_domain = max(distribution, key=distribution.get)  # type: ignore[arg-type]
                    yield _sse(
                        StreamStartEvent(
                            type="route",
                            routing_mode="single",
                            primary_domain=top_domain,
                            domain_distribution=distribution,
                        )
                    )

                    # Emit specialist_start event
                    spec = self._config.specialist_for_field(top_domain)
                    _spec_name = spec.name if spec else "default"
                    yield f'event: specialist_start\ndata: {{"specialist": "{_spec_name}", "domain": "{top_domain}"}}\n\n'

                    # Stream tokens
                    url = self._field_to_url.get(top_domain, self._arbiter_url)
                    model_name = spec.serve_model_name if spec else "default_model"
                    full_text = ""
                    idx = 0
                    async for token in self._call_stream(
                        url, _stream_content, top_domain, history, model_name=model_name
                    ):
                        full_text += token
                        yield _sse(StreamChunkEvent(type="chunk", text=token, index=idx))
                        idx += 1

                    # Score the full response
                    base_conf = 0.8
                    u, conf, n_contra, n_dpo = await self._score(
                        _stream_content, full_text, top_domain, base_conf
                    )
                    latency_ms = round((time.time() - t0) * 1000, 1)

                    # Store assistant message
                    add_message(
                        session_id,
                        role="assistant",
                        content=full_text,
                        domain=top_domain,
                        routing_mode="single",
                        u_score=u,
                        latency_ms=latency_ms,
                    )

                    # Emit specialist_done event
                    yield f'event: specialist_done\ndata: {{"u_score": {u:.4f}, "latency_ms": {latency_ms}}}\n\n'

                    # Emit done event
                    yield _sse(
                        StreamDoneEvent(
                            type="done",
                            full_response=full_text,
                            routing_mode="single",
                            primary_domain=top_domain,
                            domain_distribution=distribution,
                            u_score=u,
                            confidence=conf,
                            contradictions_detected=n_contra,
                            dpo_pairs_generated=n_dpo,
                            latency_ms=latency_ms,
                        )
                    )
                except Exception as exc:
                    log.exception("session stream error: %s", exc)
                    yield _sse(StreamErrorEvent(type="error", code=500, message=str(exc)))

            return StreamingResponse(
                _generate(),
                media_type=_SSE_CONTENT_TYPE,
                headers=_SSE_HEADERS,
            )

        @app.get("/metrics", tags=["observability"], include_in_schema=False)
        async def prometheus_metrics():
            """Prometheus metrics endpoint. Install prometheus-client for full metrics."""
            from fastapi.responses import Response

            from aua.metrics import get_metrics

            content, content_type = get_metrics().get_prometheus_output()
            return Response(content=content, media_type=content_type)

        @app.get("/metrics/cost", tags=["observability"], summary="Cost tracking metrics")
        async def cost_metrics():
            """GPU cost and query cost metrics."""
            from aua.metrics import get_metrics

            return get_metrics().get_cost_summary(self._config)

        return app

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def _handle_stream(self, req: QueryRequest) -> AsyncIterator[str]:
        """
        Async generator yielding SSE frames for POST /query/stream.

        Single domain  → live token stream from vLLM/Ollama
        Fanout/arbiter → buffer both responses, arbitrate, emit winner as one chunk
        """
        t0 = time.time()

        distribution = (
            {req.force_domain: 1.0}
            if req.force_domain
            else self._classifier.classify(req.query, update_history=True)
        )

        top_domain = max(distribution, key=lambda k: distribution.get(k, 0.0))
        top_prob = distribution[top_domain]
        active = [
            s
            for s in self._config.specialists
            if distribution.get(s.field, 0) >= self._fanout_threshold
        ]
        is_fanout = len(active) >= 2

        if is_fanout:
            routing_mode, primary_domain = "fanout", "arbiter"
        elif top_prob >= self._single_threshold:
            routing_mode, primary_domain = "single", top_domain
        else:
            routing_mode, primary_domain = "arbiter", "general"

        yield _sse(
            StreamStartEvent(
                routing_mode=routing_mode,
                primary_domain=primary_domain,
                domain_distribution=distribution,
            )
        )

        try:
            if routing_mode == "single":
                spec = self._config.specialist_for_field(top_domain)
                url = self._field_to_url.get(top_domain, self._arbiter_url)
                model_name = spec.serve_model_name if spec else "default_model"

                full_text, index = "", 0
                async for token in self._call_stream(
                    url, req.query, top_domain, req.conversation_history, model_name
                ):
                    full_text += token
                    yield _sse(StreamChunkEvent(type="chunk", text=token, index=index))
                    index += 1

                if spec:
                    self._requests_per_spec[spec.name] = (
                        self._requests_per_spec.get(spec.name, 0) + 1
                    )

            else:
                if is_fanout:
                    fanout_req = QueryRequest(
                        query=req.query,
                        session_id=req.session_id or str(uuid.uuid4()),
                        conversation_history=req.conversation_history or [],
                        force_domain=None,
                    )
                    buffered = await self._handle_fanout(fanout_req, active, distribution, t0)
                    full_text = buffered.response
                    primary_domain = buffered.primary_domain
                    routing_mode = buffered.routing_mode
                else:
                    text, _ = await self._call(
                        self._arbiter_url,
                        req.query,
                        "general",
                        req.conversation_history,
                        model_name=self._config.arbiter.serve_model_name,
                    )
                    full_text = text

                yield _sse(StreamChunkEvent(type="chunk", text=full_text, index=0))

            u, conf, n_contra, n_dpo = await self._score(
                req.query,
                full_text,
                primary_domain,
                base_conf=0.75,
            )
            self._queries_by_mode[routing_mode] = self._queries_by_mode.get(routing_mode, 0) + 1
            self._latencies_ms["router"].append((time.time() - t0) * 1000)
            self._total_contradictions += n_contra
            self._total_dpo += n_dpo

            yield _sse(
                StreamDoneEvent(
                    full_response=full_text,
                    routing_mode=routing_mode,
                    primary_domain=primary_domain,
                    domain_distribution=distribution,
                    u_score=u,
                    confidence=conf,
                    contradictions_detected=n_contra,
                    dpo_pairs_generated=n_dpo,
                    latency_ms=round((time.time() - t0) * 1000, 1),
                )
            )

        except HTTPException as exc:
            yield _sse(
                StreamErrorEvent(type="error", code=exc.status_code, message=str(exc.detail))
            )
        except Exception as exc:
            log.error("Stream error: %s", exc)
            yield _sse(StreamErrorEvent(type="error", code=500, message=str(exc)))

    async def _call_stream(
        self,
        url: str,
        query: str,
        domain: str,
        history: list[dict] | None = None,
        model_name: str = "default_model",
    ) -> AsyncIterator[str]:
        """Dispatch to vLLM or Ollama streaming backend based on config.backend."""
        corrections = self._store.query(subject=query[:100], domain=domain)
        injection = ""
        if corrections:
            injection = "\n\nActive corrections:\n" + "\n".join(
                f"- {m.assertion.claim}" for m in corrections[:5]
            )
        system_prompt = (
            f"You are a specialist in {domain.replace('_', ' ')}. "
            f"Answer precisely and correctly.{injection}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        for h in history or []:
            messages.append(h)
        messages.append({"role": "user", "content": query})

        if self._config.backend == "ollama":
            async for token in self._call_stream_ollama(url, messages, model_name):
                yield token
        else:
            async for token in self._call_stream_vllm(url, messages, model_name):
                yield token

    async def _call_stream_vllm(
        self,
        url: str,
        messages: list[dict],
        model_name: str,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from a vLLM OpenAI-compatible endpoint.
        Parses SSE frames: data: {json} ... data: [DONE]
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    json={
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": 1024,
                        "temperature": 0.1,
                        "stream": True,
                    },
                ) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            response.status_code,
                            f"Specialist returned {response.status_code}",
                        )
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            token = chunk["choices"][0].get("delta", {}).get("content") or ""
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except httpx.ConnectError:
            raise HTTPException(503, f"Specialist at {url} is not reachable.")
        except HTTPException:
            raise
        except Exception as exc:
            log.error("vLLM stream error: %s", exc)
            raise HTTPException(500, str(exc))

    async def _call_stream_ollama(
        self,
        url: str,
        messages: list[dict],
        model_name: str,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from an Ollama endpoint (NDJSON format).
        One JSON object per line; done=true signals end of stream.
        """
        ollama_url = url.replace("/v1/chat/completions", "/api/chat")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    ollama_url,
                    json={"model": model_name, "messages": messages, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            response.status_code,
                            f"Ollama returned {response.status_code}",
                        )
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content") or ""
                            if token:
                                yield token
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            raise HTTPException(503, f"Ollama at {ollama_url} is not reachable.")
        except HTTPException:
            raise
        except Exception as exc:
            log.error("Ollama stream error: %s", exc)
            raise HTTPException(500, str(exc))

    # ── Health / telemetry internals ──────────────────────────────────────────

    async def _health(self) -> dict:
        status: dict[str, str] = {}
        checks = [(s.name, s.models_url) for s in self._config.specialists] + [
            ("arbiter", self._config.arbiter.models_url)
        ]
        async with httpx.AsyncClient(timeout=3.0) as client:
            for name, url in checks:
                try:
                    r = await client.get(url)
                    status[name] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
                except Exception:
                    status[name] = "unreachable"
        return {"specialists": status, "domain_confidence": self._domain_confidence}

    async def _full_status(self) -> dict:
        import subprocess

        health = await self._health()
        uptime_s = time.time() - self._start_time

        latency_stats: dict[str, dict] = {}
        for name, dq in self._latencies_ms.items():
            vals = list(dq)
            if vals:
                sv = sorted(vals)
                n = len(sv)
                latency_stats[name] = {
                    "p50_ms": round(sv[n // 2], 1),
                    "p95_ms": round(sv[int(n * 0.95)], 1),
                    "last_ms": round(vals[-1], 1),
                    "samples": n,
                }
            else:
                latency_stats[name] = {
                    "p50_ms": None,
                    "p95_ms": None,
                    "last_ms": None,
                    "samples": 0,
                }

        utility: dict[str, dict] = {}
        for domain, state in self._scorer.domain_states.items():
            history = [s.utility for s in self._scorer.history if s.field == domain]
            utility[domain] = {
                "mean_u": round(sum(history) / len(history), 4) if history else None,
                "last_u": round(history[-1], 4) if history else None,
                "queries": len(history),
                "confidence": round(state.confidence, 4),
            }

        total_q = sum(self._queries_by_mode.values())
        store_s = self._store.summary()

        memory: dict[str, str] = {}
        try:
            from aua.doctor import _detect_hardware

            hw = _detect_hardware()
            if hw.kind == "nvidia":
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                for line in r.stdout.strip().splitlines():
                    p = [x.strip() for x in line.split(", ")]
                    if len(p) >= 3:
                        memory[f"gpu{p[0]}"] = f"{p[1]} / {p[2]} MiB"
            elif hw.kind == "amd_rocm":
                for dev in hw.devices:
                    mib = dev.get("vram_mib")
                    memory[f"gpu{dev['index']}"] = f"{mib} MiB (total)" if mib else "AMD GPU"
            elif hw.kind == "apple_silicon":
                for dev in hw.devices:
                    mib = dev.get("vram_mib")
                    memory[f"gpu{dev['index']}"] = (
                        f"{mib} MiB unified" if mib else dev.get("name", "Apple GPU")
                    )
            else:
                ram = hw.system_ram_mib
                memory["system"] = f"{ram // 1024} GiB RAM" if ram else "CPU / Ollama"
        except Exception:
            memory = {"system": "unavailable"}

        return {
            "version": self._config.version,
            "backend": self._config.backend,
            "uptime_s": round(uptime_s, 1),
            "health": health["specialists"],
            "latency": latency_stats,
            "utility": utility,
            "routing": {
                "total_queries": total_q,
                "by_mode": dict(self._queries_by_mode),
            },
            "corrections": {
                "total_contradictions": self._total_contradictions,
                "dpo_pairs": self._total_dpo,
                "assertions_stored": store_s.get("total", 0),
                "contradiction_rate": (
                    round(self._total_contradictions / total_q, 4) if total_q > 0 else 0.0
                ),
            },
            "arbiter_verdicts": dict(self._verdict_counts),
            "memory": memory,
        }

    def _stats(self) -> dict:
        summary = self._store.summary()
        return {
            "domain_confidence": self._domain_confidence,
            "assertions_count": summary.get("total", 0),
            "dpo_pairs_count": summary.get("by_source", {}).get("contradiction_detector", 0),
        }

    # ── Batch ─────────────────────────────────────────────────────────────────

    async def _handle_batch(self, req: BatchQueryRequest) -> BatchQueryResponse:
        t0 = time.time()
        sem = asyncio.Semaphore(req.max_parallel or 4)

        async def _one(q: str) -> RouterResponse | None:
            async with sem:
                try:
                    return await self._handle(
                        QueryRequest(
                            query=q,
                            session_id=req.session_id or str(uuid.uuid4()),
                            conversation_history=[],
                            force_domain=None,
                        )
                    )
                except Exception as e:
                    log.error("Batch query failed: %s", e)
                    return None

        raw = await asyncio.gather(*[_one(q) for q in req.queries])
        results = [r for r in raw if r is not None]
        n_errors = len(req.queries) - len(results)
        return BatchQueryResponse(
            results=results,
            total_latency_ms=round((time.time() - t0) * 1000, 1),
            n_queries=len(req.queries),
            n_errors=n_errors,
        )

    # ── Green evaluation ──────────────────────────────────────────────────────

    async def _evaluate_green(self, req: DeployGreenRequest) -> DeployGreenResponse:
        spec = next((s for s in self._config.specialists if s.name == req.specialist), None)
        if spec is None:
            raise HTTPException(404, f"Specialist '{req.specialist}' not found in config.")

        threshold = self._config.blue_green_for(req.specialist).delta
        eval_qs = [
            "Write a binary search function. State the time complexity.",
            "Implement merge sort. State time and space complexity.",
            "Write a function to check if a string is a palindrome.",
        ][: req.n_eval_queries or 3]

        blue_scores: list[float] = []
        for q in eval_qs:
            try:
                text, conf = await self._call(
                    spec.endpoint, q, spec.field, model_name=spec.serve_model_name
                )
                u, *_ = await self._score(q, text, spec.field, conf)
                blue_scores.append(u)
            except Exception:
                blue_scores.append(0.0)

        blue_u = round(sum(blue_scores) / len(blue_scores), 4) if blue_scores else 0.0
        green_u = blue_u
        u_delta = round(green_u - blue_u, 4)
        promoted = u_delta >= threshold

        return DeployGreenResponse(
            specialist=req.specialist,
            promoted=promoted,
            u_delta=u_delta,
            blue_u=blue_u,
            green_u=green_u,
            threshold=threshold,
            message=(
                f"GREEN promoted (U_delta {u_delta:+.4f} >= {threshold})"
                if promoted
                else (
                    f"GREEN not promoted (U_delta {u_delta:+.4f} < {threshold}). "
                    f"Accumulate more DPO pairs or reduce LoRA rank."
                )
            ),
        )

    # ── Core routing (buffered) ───────────────────────────────────────────────

    async def _handle(self, req: QueryRequest) -> RouterResponse:
        t0 = time.time()
        log.info("Query: %.80s", req.query)
        _hooks = get_hook_runner()
        _sid = req.session_id or ""
        _tid = str(uuid.uuid4())

        # ── pre_query: before field classification ────────────────────────────
        await _hooks.fire(
            "pre_query",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "query": req.query,
                "conversation_history": req.conversation_history or [],
                "force_domain": req.force_domain,
            },
        )

        distribution = (
            {req.force_domain: 1.0}
            if req.force_domain
            else self._classifier.classify(req.query, update_history=True)
        )
        log.debug("Distribution: %s", distribution)

        top_domain = max(distribution, key=lambda k: distribution.get(k, 0.0))
        top_prob = distribution[top_domain]
        active = [
            s
            for s in self._config.specialists
            if distribution.get(s.field, 0) >= self._fanout_threshold
        ]

        routing_mode = (
            "fanout"
            if len(active) >= 2
            else "single" if top_prob >= self._single_threshold else "arbiter"
        )

        # ── post_route: after routing decision, before specialist calls ───────
        await _hooks.fire(
            "post_route",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "query": req.query,
                "domain_distribution": distribution,
                "top_domain": top_domain,
                "routing_mode": routing_mode,
                "active_specialists": [s.name for s in active],
            },
        )

        if len(active) >= 2:
            return await self._handle_fanout(req, active, distribution, t0, _sid, _tid)
        elif top_prob >= self._single_threshold:
            return await self._handle_single(req, top_domain, distribution, t0, _sid, _tid)
        else:
            return await self._handle_arbiter(req, distribution, t0, _sid, _tid)

    async def _handle_single(
        self, req, domain, distribution, t0, _sid="", _tid=""
    ) -> RouterResponse:
        url = self._field_to_url.get(domain, self._arbiter_url)
        spec = self._config.specialist_for_field(domain)
        model_name = spec.serve_model_name if spec else "default_model"
        _hooks = get_hook_runner()

        # ── pre_specialist_call ───────────────────────────────────────────────
        await _hooks.fire(
            "pre_specialist_call",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "query": req.query,
                "domain": domain,
                "specialist": spec.name if spec else "default",
                "model": model_name,
                "endpoint": url,
            },
        )

        response, base_conf = await self._call(
            url, req.query, domain, req.conversation_history, model_name=model_name
        )

        # ── post_specialist_call ──────────────────────────────────────────────
        await _hooks.fire(
            "post_specialist_call",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "domain": domain,
                "specialist": spec.name if spec else "default",
                "response_preview": response[:200],
                "confidence": base_conf,
            },
        )

        # ── Policy / assertion layer ──────────────────────────────────────
        e_bonus, u_penalty, gold = 0.0, 0.0, False
        policy = getattr(self, "_active_policy", None)
        if policy is not None:
            context = {
                "query": req.query,
                "session_id": req.session_id,
                "domain": domain,
                "field": domain,
                "conversation_history": req.conversation_history or [],
            }

            def _sync_retry(error_msg: str) -> str | None:
                import asyncio as _aio

                try:
                    augmented = req.query + f"\n\n[Feedback: {error_msg}]"
                    return _aio.get_event_loop().run_until_complete(
                        self._call(
                            url, augmented, domain, req.conversation_history, model_name=model_name
                        )
                    )[0]
                except Exception:
                    return None

            policy_result = policy.run(response, context, retry_fn=_sync_retry)
            e_bonus = policy_result.e_bonus
            u_penalty = policy_result.u_penalty
            gold = policy_result.gold_standard
            _persist_assertion_results(policy_result, req.session_id or "", domain, policy.name)

        u, conf, n_contra, n_dpo = await self._score(
            req.query, response, domain, base_conf, e_bonus=e_bonus, u_penalty=u_penalty
        )
        # ── end policy layer ──────────────────────────────────────────────

        self._queries_by_mode["single"] = self._queries_by_mode.get("single", 0) + 1
        self._latencies_ms["router"].append((time.time() - t0) * 1000)
        if spec:
            self._requests_per_spec[spec.name] = self._requests_per_spec.get(spec.name, 0) + 1
        self._total_contradictions += n_contra
        self._total_dpo += n_dpo
        latency_ms = round((time.time() - t0) * 1000, 1)
        log.info(
            "single→%s  U=%.3f  C=%.3f  contra=%d  dpo=%d  gold=%s",
            domain,
            u,
            conf,
            n_contra,
            n_dpo,
            gold,
        )
        _audit_query(req, domain, "single", u, conf, latency_ms, n_contra)
        from aua.metrics import get_metrics

        get_metrics().record_query(
            domain=domain,
            routing_mode="single",
            latency_s=latency_ms / 1000,
            u_score=u,
            status="ok",
        )
        _n_corrections = len(self._store.query(subject=req.query[:100], domain=domain))
        resp = RouterResponse(
            query=req.query,
            session_id=req.session_id,
            routing_mode="single",
            domain_distribution=distribution,
            primary_domain=domain,
            response=response,
            u_score=u,
            confidence=conf,
            contradictions_detected=n_contra,
            corrections_injected=_n_corrections,
            dpo_pairs_generated=n_dpo,
            latency_ms=latency_ms,
        )
        # ── pre_response / post_response ──────────────────────────────────────
        pre_event = await _hooks.fire(
            "pre_response",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "domain": domain,
                "routing_mode": "single",
                "u_score": u,
                "confidence": conf,
                "latency_ms": latency_ms,
                "response": response,
            },
        )
        # Allow pre_response hook to modify the response text
        if pre_event.get("response") and pre_event["response"] != response:
            resp = RouterResponse(
                query=req.query,
                session_id=req.session_id,
                routing_mode="single",
                domain_distribution=distribution,
                primary_domain=domain,
                response=pre_event["response"],
                u_score=u,
                confidence=conf,
                contradictions_detected=n_contra,
                corrections_injected=_n_corrections,
                dpo_pairs_generated=n_dpo,
                latency_ms=latency_ms,
            )
        _hooks.fire_background(
            "post_response",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "domain": domain,
                "routing_mode": "single",
                "u_score": u,
                "latency_ms": latency_ms,
                "gold_standard": gold,
            },
        )
        return resp

    async def _handle_fanout(
        self, req, active_specialists, distribution, t0, _sid="", _tid=""
    ) -> RouterResponse:
        log.info("fanout → %s", [s.name for s in active_specialists])
        _hooks = get_hook_runner()

        # ── pre_specialist_call per specialist (sequential fire, parallel calls) ─
        for s in active_specialists:
            await _hooks.fire(
                "pre_specialist_call",
                {
                    "session_id": _sid,
                    "trace_id": _tid,
                    "query": req.query,
                    "domain": s.field,
                    "specialist": s.name,
                    "model": s.serve_model_name,
                    "endpoint": s.endpoint,
                },
            )

        calls = [
            self._call(
                s.endpoint,
                req.query,
                s.field,
                req.conversation_history,
                model_name=s.serve_model_name,
            )
            for s in active_specialists
        ]
        results = await asyncio.gather(*calls, return_exceptions=True)
        responses = []
        for spec, result in zip(active_specialists, results):
            if isinstance(result, Exception):
                log.warning("Specialist %s failed: %s", spec.name, result)
            else:
                text, conf = result  # type: ignore[misc]
                responses.append((spec, text, conf))
                # ── post_specialist_call per specialist ───────────────────────
                await _hooks.fire(
                    "post_specialist_call",
                    {
                        "session_id": _sid,
                        "trace_id": _tid,
                        "domain": spec.field,
                        "specialist": spec.name,
                        "response_preview": text[:200],
                        "confidence": conf,
                    },
                )

        if not responses:
            raise HTTPException(503, "All specialists unreachable during fanout")

        _routing_mode = "fanout"
        _vcg_welfare: dict[str, float] | None = None

        # ── VCG vs pairwise selection ─────────────────────────────────────────
        if self._arbitration_mode == "vcg" and len(responses) >= 2:
            winner_idx, _vcg_welfare = self._vcg_select(responses, distribution)
            winner_spec, final_text, final_conf = responses[winner_idx]
            primary_domain = winner_spec.field
            _routing_mode = "vcg"
            log.info(
                "VCG → winner=%s  W=%s",
                winner_spec.name,
                {k: round(v, 4) for k, v in _vcg_welfare.items()},
            )
            # Still run arbiter for contradiction detection + DPO pairs (non-verdict mode)
            losers = [r for i, r in enumerate(responses) if i != winner_idx]
            n_contra, n_dpo = 0, 0
            for loser_spec, loser_text, _ in losers:
                _, _, nc, nd = await self._score(req.query, loser_text, loser_spec.field, 0.30)
                n_contra += nc
                n_dpo += nd
            u, conf, nc_win, nd_win = await self._score(
                req.query, final_text, primary_domain, final_conf
            )
            n_contra += nc_win
            n_dpo += nd_win
            spec_responses = [
                {
                    "domain": s.field,
                    "specialist": s.name,
                    "response": t[:200] + "...",
                    "welfare": round(_vcg_welfare.get(s.name, 0.0), 4),
                    "winner": s.name == winner_spec.name,
                }
                for s, t, _ in responses
            ]

        elif len(responses) >= 2:
            (spec_a, text_a, conf_a), (spec_b, text_b, conf_b) = responses[0], responses[1]

            # ── pre_arbiter ───────────────────────────────────────────────────
            await _hooks.fire(
                "pre_arbiter",
                {
                    "session_id": _sid,
                    "trace_id": _tid,
                    "query": req.query,
                    "specialist_a": spec_a.name,
                    "response_a": text_a[:200],
                    "specialist_b": spec_b.name,
                    "response_b": text_b[:200],
                },
            )

            verdict, winner_field = await self._arbitrate(req.query, spec_a, text_a, spec_b, text_b)

            # ── post_arbiter ──────────────────────────────────────────────────
            await _hooks.fire(
                "post_arbiter",
                {
                    "session_id": _sid,
                    "trace_id": _tid,
                    "verdict": verdict[:200] if isinstance(verdict, str) else str(verdict)[:200],
                    "winner_field": winner_field,
                    "specialist_a": spec_a.name,
                    "specialist_b": spec_b.name,
                },
            )

            if winner_field == spec_b.field:
                final_text, final_conf, primary_domain = text_b, conf_b, spec_b.field
                losing_text, losing_domain = text_a, spec_a.field
            elif winner_field == "both_wrong":
                final_text, final_conf, primary_domain = verdict, 0.40, "arbiter"
                losing_text, losing_domain = text_a, spec_a.field
            else:
                final_text, final_conf, primary_domain = text_a, conf_a, spec_a.field
                losing_text, losing_domain = text_b, spec_b.field

            u, conf, n_contra, n_dpo = await self._score(
                req.query, final_text, primary_domain, final_conf
            )
            _, _, nc2, nd2 = await self._score(req.query, losing_text, losing_domain, 0.30)
            n_contra += nc2
            n_dpo += nd2
            spec_responses = [
                {"domain": s.field, "response": t[:200] + "..."} for s, t, _ in responses
            ] + [{"domain": "arbiter_verdict", "winner": winner_field}]
        else:
            spec, text, base_conf = responses[0]
            primary_domain = spec.field
            u, conf, n_contra, n_dpo = await self._score(req.query, text, primary_domain, base_conf)
            final_text = text
            spec_responses = [{"domain": spec.field, "response": text[:200]}]

        self._queries_by_mode[_routing_mode] = self._queries_by_mode.get(_routing_mode, 0) + 1
        self._latencies_ms["router"].append((time.time() - t0) * 1000)
        for s in active_specialists:
            self._requests_per_spec[s.name] = self._requests_per_spec.get(s.name, 0) + 1
        self._total_contradictions += n_contra
        self._total_dpo += n_dpo
        _fanout_ms = round((time.time() - t0) * 1000, 1)
        from aua.metrics import get_metrics as _gm

        _gm().record_query(
            domain=primary_domain,
            routing_mode=_routing_mode,
            latency_s=_fanout_ms / 1000,
            u_score=u,
            status="ok",
        )

        # ── pre_response / post_response ──────────────────────────────────────
        await _hooks.fire(
            "pre_response",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "domain": primary_domain,
                "routing_mode": _routing_mode,
                "u_score": u,
                "latency_ms": _fanout_ms,
            },
        )
        _hooks.fire_background(
            "post_response",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "domain": primary_domain,
                "routing_mode": _routing_mode,
                "u_score": u,
                "latency_ms": _fanout_ms,
            },
        )

        _n_corrections_fo = len(self._store.query(subject=req.query[:100], domain=primary_domain))
        return RouterResponse(
            query=req.query,
            session_id=req.session_id,
            routing_mode=_routing_mode,
            domain_distribution=distribution,
            primary_domain=primary_domain,
            response=final_text,
            u_score=u,
            confidence=conf,
            contradictions_detected=n_contra,
            corrections_injected=_n_corrections_fo,
            dpo_pairs_generated=n_dpo,
            latency_ms=_fanout_ms,
            specialist_responses=spec_responses,
            welfare_scores=_vcg_welfare,
        )

    async def _handle_arbiter(self, req, distribution, t0, _sid="", _tid="") -> RouterResponse:
        log.info("arbiter fallback (low confidence)")
        _hooks = get_hook_runner()

        # ── pre_specialist_call (arbiter as specialist) ───────────────────────
        await _hooks.fire(
            "pre_specialist_call",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "query": req.query,
                "domain": "general",
                "specialist": "arbiter",
                "model": self._config.arbiter.serve_model_name,
                "endpoint": self._arbiter_url,
            },
        )

        response, base_conf = await self._call(
            self._arbiter_url,
            req.query,
            "general",
            req.conversation_history,
            model_name=self._config.arbiter.serve_model_name,
        )

        # ── post_specialist_call ──────────────────────────────────────────────
        await _hooks.fire(
            "post_specialist_call",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "domain": "general",
                "specialist": "arbiter",
                "response_preview": response[:200],
                "confidence": base_conf,
            },
        )

        u, conf, n_contra, n_dpo = await self._score(req.query, response, "general", base_conf)
        self._queries_by_mode["arbiter"] = self._queries_by_mode.get("arbiter", 0) + 1
        self._latencies_ms["router"].append((time.time() - t0) * 1000)
        self._requests_per_spec["arbiter"] = self._requests_per_spec.get("arbiter", 0) + 1
        self._total_contradictions += n_contra
        self._total_dpo += n_dpo
        _arb_ms = round((time.time() - t0) * 1000, 1)
        from aua.metrics import get_metrics as _gm2

        _gm2().record_query(
            domain="general",
            routing_mode="arbiter",
            latency_s=_arb_ms / 1000,
            u_score=u,
            status="ok",
        )

        # ── pre_response / post_response ──────────────────────────────────────
        await _hooks.fire(
            "pre_response",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "domain": "general",
                "routing_mode": "arbiter",
                "u_score": u,
                "latency_ms": _arb_ms,
            },
        )
        _hooks.fire_background(
            "post_response",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "domain": "general",
                "routing_mode": "arbiter",
                "u_score": u,
                "latency_ms": _arb_ms,
            },
        )

        _n_corrections_arb = len(self._store.query(subject=req.query[:100], domain="general"))
        return RouterResponse(
            query=req.query,
            session_id=req.session_id,
            routing_mode="arbiter",
            domain_distribution=distribution,
            primary_domain="general",
            response=response,
            u_score=u,
            confidence=conf,
            contradictions_detected=n_contra,
            corrections_injected=_n_corrections_arb,
            dpo_pairs_generated=n_dpo,
            latency_ms=_arb_ms,
        )

    # ── Specialist call (buffered) ────────────────────────────────────────────

    def _vcg_select(
        self,
        responses: list[tuple],  # [(SpecialistConfig, text, confidence), ...]
        distribution: dict[str, float],
    ) -> tuple[int, dict[str, float]]:
        """
        VCG welfare maximization: select the specialist with the highest welfare score.

        W_i = P(domain_i) × confidence_i × prior_mean_u_i

          P(domain_i)     — field classifier probability for this specialist's domain
          confidence_i    — base confidence returned by the specialist call
          prior_mean_u_i  — running mean U score for this specialist (1.0 if no history)

        Returns (winner_index, welfare_scores_dict).
        Ties are broken by confidence, then by classifier probability.
        """
        welfare: dict[str, float] = {}
        for spec, _text, conf in responses:
            p_domain = distribution.get(spec.field, 0.0)
            # Get running mean U from scorer history
            history = [s.utility for s in self._scorer.history if s.field == spec.field]
            prior_u = round(sum(history) / len(history), 4) if history else 1.0
            w = round(p_domain * conf * prior_u, 6)
            welfare[spec.name] = w
            log.debug(
                "VCG W(%s) = P(%.3f) × C(%.3f) × U_mean(%.3f) = %.4f",
                spec.name,
                p_domain,
                conf,
                prior_u,
                w,
            )

        # argmax by welfare, tie-break: confidence, then P(domain)
        winner_idx = max(
            range(len(responses)),
            key=lambda i: (
                welfare[responses[i][0].name],
                responses[i][2],
                distribution.get(responses[i][0].field, 0.0),
            ),
        )
        return winner_idx, welfare

    async def _call(
        self,
        url: str,
        query: str,
        domain: str,
        history: list[dict] | None = None,
        system_prompt: str | None = None,
        model_name: str = "default_model",
    ) -> tuple[str, float]:
        """Call a vLLM or Ollama endpoint (buffered, not streaming)."""
        if system_prompt is None:
            corrections = self._store.query(subject=query[:100], domain=domain)
            injection = ""
            if corrections:
                injection = "\n\nActive corrections:\n" + "\n".join(
                    f"- {m.assertion.claim}" for m in corrections[:5]
                )
            system_prompt = (
                f"You are a specialist in {domain.replace('_', ' ')}. "
                f"Answer precisely and correctly.{injection}"
            )

        messages = [{"role": "system", "content": system_prompt}]
        for h in history or []:
            messages.append(h)
        messages.append({"role": "user", "content": query})

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(
                    url,
                    json={
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": 1024,
                        "temperature": 0.1,
                    },
                )
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"].strip()
                stop = data["choices"][0].get("finish_reason", "") == "stop"
                return text, 0.75 if stop else 0.50
        except httpx.ConnectError:
            raise HTTPException(503, f"Specialist at {url} is not reachable. Is it running?")
        except Exception as exc:
            log.error("Specialist call failed: %s", exc)
            raise HTTPException(500, str(exc))

    # ── Scoring ───────────────────────────────────────────────────────────────

    async def _score(
        self,
        query: str,
        response: str,
        domain: str,
        base_conf: float,
        e_bonus: float = 0.0,
        u_penalty: float = 0.0,
    ) -> tuple[float, float, int, int]:
        field_cfg = FIELD_CONFIGS.get(domain, FIELD_CONFIGS["general"])
        result = self._detector.check(problem=query, solution=response)
        n_contra = len(result.contradictions)
        updated_conf = self._conf.update(
            prior=base_conf,
            test_signal=base_conf,
            contradiction_result=result,
            field=domain,
        )
        self._domain_confidence[domain] = (
            0.8 * self._domain_confidence.get(domain, 0.5) + 0.2 * updated_conf
        )
        task_score = self._scorer.score(
            task_id=f"router_{domain}",
            field_config=field_cfg,
            test_pass_rate=base_conf,
            human_baseline_score=0.65,
            contradiction_penalty=result.confidence_penalty,
            problem_novelty=0.5,
        )
        n_dpo = 0
        if n_contra > 0:
            for c in result.contradictions:
                self._store.add(
                    subject=query[:100],
                    domain=domain,
                    claim=f"Contradiction: {c.description}",
                    confidence=updated_conf,
                    source="contradiction_detector",
                )
                n_dpo += 1
            log.info("[%s] %d contradiction(s) → %d DPO pair(s)", domain, n_contra, n_dpo)

        # Apply policy bonus/penalty to final U score
        base_u = task_score.utility
        if e_bonus > 0.0:
            # Boost E component: U = w_e*(E+bonus) + rest
            from aua.config import FIELD_CONFIGS as _FC

            fc = _FC.get(domain, _FC["general"])
            u_adjusted = min(1.0, base_u + fc.w_efficacy * e_bonus)
        else:
            u_adjusted = base_u
        if u_penalty > 0.0:
            u_adjusted = max(0.0, u_adjusted - u_penalty)

        return round(u_adjusted, 4), updated_conf, n_contra, n_dpo

    # ── Arbitration ───────────────────────────────────────────────────────────

    async def _arbitrate(
        self, query: str, spec_a, text_a: str, spec_b, text_b: str
    ) -> tuple[str, str]:
        prompt = (
            f"Two specialist models produced different responses.\n\n"
            f"Query: {query}\n\n"
            f"Response A ({spec_a.field}):\n{text_a}\n\n"
            f"Response B ({spec_b.field}):\n{text_b}\n\n"
            f"Which is correct? Reply:\n"
            f"VERDICT: [A|B|BOTH_WRONG]\nREASON: [brief]\n"
            f"CORRECTION: [what the losing model should learn]"
        )
        verdict_text, _ = await self._call(
            self._arbiter_url,
            prompt,
            "arbiter",
            system_prompt="You are a cross-domain arbitration agent. Be concise and decisive.",
            model_name=self._config.arbiter.serve_model_name,
        )
        winner = (
            spec_b.field
            if "VERDICT: B" in verdict_text
            else "both_wrong" if "BOTH_WRONG" in verdict_text else spec_a.field
        )
        log.info("Arbiter verdict: %s", winner)
        return verdict_text, winner
