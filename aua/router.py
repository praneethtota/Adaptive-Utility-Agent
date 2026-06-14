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
import os
import re
import statistics
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from pathlib import Path
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
    BatchSubmitRequest,
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
    ShadowActivateRequest,
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

        # ── F-09/F-10/F-11: YAML-registered extensions ───────────────────────
        # The expert path: plugins, hooks, and middleware register in
        # aua_config.yaml — users never edit AUA source files. Each entry is
        # loaded and contract-validated via aua.plugins.registry.load_plugin;
        # a bad import_path fails fast at startup with a clear error.
        from aua.middleware import MiddlewarePipeline

        self._middleware = MiddlewarePipeline()
        self._custom_scorer: Any | None = None  # utility_scorer plugin
        # #51: extended plugin slots
        self._custom_detector: Any | None = None  # contradiction_detector plugin
        self._custom_assertion_store: Any | None = None  # assertion_store plugin
        self._routing_strategy: Any | None = None  # routing_strategy plugin
        self._scoring_component: Any | None = None  # scoring_component plugin
        self._custom_arbiter_policy: Any | None = None  # arbiter_policy plugin
        self._custom_promotion_policy: Any | None = None  # promotion_policy plugin
        self._load_yaml_extensions()

        # ── v1.1-veritas P1–P3 components ────────────────────────────────────
        from aua.context_backup import ContextBackupManager
        from aua.domain_tree import DomainTree, OntologyJob
        from aua.keywords import KeywordIndex
        from aua.remote_config import RemoteModelConfig
        from aua.trigger_detector import TriggerDetector

        self._keyword_index = KeywordIndex(self._state_store)  # V-P1.1
        self._backup_mgr = ContextBackupManager(self._state_store)  # V-P1.2/1.4
        self._trigger = TriggerDetector()  # V-P1.3
        self._pending_implicit: dict[str, dict] = {}  # conv_id → pending correction
        self._remote_models = RemoteModelConfig(self._state_store)  # V-P1.6
        self._domain_tree = DomainTree(self._state_store)  # V-P3.4
        self._ontology_job = OntologyJob(self._domain_tree, self._state_store)
        self._crash_session_id: str | None = None  # V-P1.5
        self._background_tasks: list = []  # cancelled on shutdown

        # #56: persistent batch queue + background worker
        from aua.batch_queue import BatchQueue
        from aua.batch_queue import BatchWorker as _BatchWorker

        self._batch_queue = BatchQueue(self._state_store)
        self._batch_worker: _BatchWorker | None = None  # started in lifespan

        # #47: experiment tracking
        from aua.experiment_tracker import ExperimentTracker

        self._experiment_tracker = ExperimentTracker(config.experiment_tracking)

        # #48: shadow mode
        from aua.shadow import ShadowManager, ShadowStore

        self._shadow_store = ShadowStore(self._state_store)
        self._shadow_mgr = ShadowManager(self._shadow_store)
        # Activate shadow mode from config for any specialist that has shadow_endpoint set
        for _spec in config.specialists:
            _bg = config.blue_green_for(_spec.name)
            if _bg.shadow_endpoint:
                self._shadow_mgr.activate(
                    _spec.name,
                    _bg.shadow_endpoint,
                    min_queries=_bg.shadow_min_queries,
                    threshold=_bg.delta,
                )

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
        # Implementation rule (Phase 13): asyncio imported at the TOP of the
        # lifespan/mount body — never mid-body. Startup code that runs before
        # a lazy import crashes with NameError (bit Veritas in 2 places).
        import asyncio as _asyncio
        from contextlib import asynccontextmanager

        from aua import crash_reporter as _crash

        @asynccontextmanager
        async def _lifespan(app: FastAPI):
            # ── Startup ───────────────────────────────────────────────────────
            try:
                # V-P1.5: detect previous crash BEFORE writing the new sentinel
                # (writing first makes the current session self-report as
                # crashed), then write the sentinel, then report in background.
                _prev_crash = _crash.detect_crash(self._state_store)
                self._crash_session_id = _crash.record_startup(self._state_store)

                async def _crash_startup() -> None:
                    try:
                        if _prev_crash:
                            await _crash.report_previous_crash(self._state_store, crash=_prev_crash)
                        await _crash.flush_pending_errors(self._state_store)
                    except Exception as e:  # noqa: BLE001
                        log.debug("Crash report on startup skipped: %s", e)

                self._background_tasks.append(_asyncio.create_task(_crash_startup()))

                # V-P1.1: keyword worker + index build/backfill (off the loop)
                self._keyword_index.start()

                async def _build_index() -> None:
                    loop = _asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._keyword_index.build_from_db)

                self._background_tasks.append(_asyncio.create_task(_build_index()))

                # V-P1.2: 6-hour context backup coverage job
                self._background_tasks.append(
                    _asyncio.create_task(
                        self._backup_mgr.coverage_job(
                            specialists_provider=lambda: [s.name for s in self._config.specialists],
                            generator=self._generate_backup_text,
                        )
                    )
                )

                # V-P1.6: remote model config — fetch now, refresh every 24h
                async def _remote_cfg() -> None:
                    try:
                        await self._remote_models.refresh()
                    except Exception as e:  # noqa: BLE001
                        log.debug("Remote model config initial refresh skipped: %s", e)

                self._background_tasks.append(_asyncio.create_task(_remote_cfg()))
                self._background_tasks.append(
                    _asyncio.create_task(self._remote_models.refresh_job())
                )

                # V-P3.4: hourly ontology maintenance
                self._ontology_job.start()

                # #56: persistent batch worker — recover interrupted jobs then start
                from aua.batch_queue import BatchWorker as _BW

                self._batch_queue.recover_interrupted()
                self._batch_worker = _BW(self._batch_queue, self._handle)
                self._batch_worker.start()
            except Exception as e:  # noqa: BLE001
                log.warning("v1.1 startup tasks failed (continuing): %s", e)

            yield

            # ── Shutdown ──────────────────────────────────────────────────────
            self._keyword_index.stop()
            self._ontology_job.stop()
            if self._batch_worker is not None:
                self._batch_worker.stop()
            for task in self._background_tasks:
                task.cancel()
            self._background_tasks.clear()
            self._experiment_tracker.finish()
            if self._crash_session_id:
                _crash.record_shutdown(self._state_store, self._crash_session_id)

        app = FastAPI(
            lifespan=_lifespan,
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

        # ── #15: session/trace/request ID middleware ─────────────────────────
        # Every HTTP request gets a SessionContext (client-supplied IDs
        # honored, UUIDs generated otherwise) stored in a contextvar so the
        # router, classifier, specialist calls, arbiter, correction loop,
        # hooks, middleware, logs, metrics, and audit log can all read it.
        # The three IDs are returned on EVERY response as headers.
        from aua.session import new_session_context

        @app.middleware("http")
        async def _session_id_middleware(request, call_next):
            ctx = new_session_context(
                session_id=request.headers.get("X-Session-ID"),
                trace_id=request.headers.get("X-Trace-ID") or request.headers.get("traceparent"),
                request_id=request.headers.get("X-Request-ID"),
            )
            response = await call_next(request)
            for k, v in ctx.as_headers().items():
                response.headers[k] = v
            return response

        # security.cors_origins (tutorial Part 2/15) overrides router.cors_origins
        _cors_origins = (
            self._config.security.cors_origins
            if getattr(self._config, "security", None) and self._config.security.cors_origins
            else self._config.router.cors_origins
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins,
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

        # ── Persistent batch queue (#56) ────────────────────────────────────

        @app.post(
            "/batch/jobs",
            tags=["batch"],
            summary="Submit a batch job — returns job_id immediately",
            status_code=202,
        )
        async def batch_submit(req: BatchSubmitRequest) -> dict:
            loop = _asyncio.get_event_loop()
            job_id = await loop.run_in_executor(
                None,
                lambda: self._batch_queue.submit(
                    queries=req.queries,
                    priority=req.priority,
                    session_id=req.session_id,
                    max_parallel=req.max_parallel,
                    meta=req.meta,
                ),
            )
            return {"job_id": job_id, "status": "pending", "n_queries": len(req.queries)}

        @app.get(
            "/batch/jobs/{job_id}",
            tags=["batch"],
            summary="Poll a batch job — status and partial results",
        )
        async def batch_poll(job_id: str) -> dict:
            loop = _asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: self._batch_queue.get_job(job_id),
            )
            if data is None:
                raise HTTPException(404, f"Batch job '{job_id}' not found")
            return data

        @app.get(
            "/batch/jobs",
            tags=["batch"],
            summary="List recent batch jobs",
        )
        async def batch_list(
            status: str | None = None,
            limit: int = 50,
        ) -> dict:
            loop = _asyncio.get_event_loop()
            jobs = await loop.run_in_executor(
                None,
                lambda: self._batch_queue.list_jobs(status=status, limit=limit),
            )
            return {"jobs": jobs, "n": len(jobs)}

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

            # Direct INSERT — the generic append() injects an `id` column,
            # but the projects table keys on project_id (bug found by the
            # v1.1 E2E suite: "table projects has no column named id").
            with self._state_store._connect() as _conn_proj:
                _conn_proj.execute(
                    "INSERT INTO projects (project_id, user_id, name, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (project_id, user_id, name, _time_proj.time()),
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

        # ── v1.1-veritas P1–P3 endpoints ─────────────────────────────────────
        self._mount_v11_endpoints(app, _msg_cache)

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
            # V-P2.4: dual-write to the state store so the correction has a
            # persistent ID for PATCH/DELETE/evidence.
            _correction_id = self._persist_correction(assertion, source=req.source or "manual")
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
                correction_id=_correction_id,
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

        # ── Shadow mode (#48) ─────────────────────────────────────────────

        @app.post(
            "/deploy/shadow/{specialist}",
            tags=["deploy"],
            summary="Activate shadow mode — GREEN receives traffic silently",
            status_code=200,
        )
        async def shadow_activate(specialist: str, body: ShadowActivateRequest) -> dict:
            spec = next((s for s in self._config.specialists if s.name == specialist), None)
            if spec is None:
                raise HTTPException(404, f"Specialist '{specialist}' not found.")
            bg = self._config.blue_green_for(specialist)
            self._shadow_mgr.activate(
                specialist,
                body.green_endpoint,
                min_queries=body.min_queries or bg.shadow_min_queries,
                threshold=body.threshold or bg.delta,
            )
            return {
                "specialist": specialist,
                "green_endpoint": body.green_endpoint,
                "min_queries": body.min_queries or bg.shadow_min_queries,
                "status": "shadow_active",
            }

        @app.get(
            "/deploy/shadow/{specialist}",
            tags=["deploy"],
            summary="Report accumulated shadow scores for a specialist",
        )
        async def shadow_report(specialist: str) -> dict:
            return self._shadow_mgr.report(specialist).to_dict()

        @app.delete(
            "/deploy/shadow/{specialist}",
            tags=["deploy"],
            summary="Deactivate shadow mode and optionally clear scores",
        )
        async def shadow_deactivate(specialist: str, clear_scores: bool = False) -> dict:
            self._shadow_mgr.deactivate(specialist, clear_scores=clear_scores)
            return {
                "specialist": specialist,
                "status": "shadow_inactive",
                "scores_cleared": clear_scores,
            }

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

            # ── V-P1.3: correction trigger detection ──────────────────────────
            # Explicit prefix rule: "correction: X" is a preference statement —
            # store it regardless of whether a prior AI turn exists. Without
            # this guard, corrections sent at the start of a conversation are
            # silently discarded.
            from aua.trigger_detector import is_explicit_prefix, strip_explicit_prefix

            _prior_msgs = get_messages(session_id, limit=20)
            _has_prior_ai = any(m["role"] == "assistant" for m in _prior_msgs[:-1])
            _correction_stored: str | None = None
            _implicit_pending = False
            if is_explicit_prefix(_msg_content):
                _instruction = strip_explicit_prefix(_msg_content)
                if _instruction:
                    _assertion = self._store.add(
                        subject=_instruction[:100],
                        domain="general",
                        claim=_instruction,
                        confidence=0.85,
                        source="explicit_prefix",
                    )
                    _correction_stored = self._persist_correction(
                        _assertion, source="explicit_prefix", session_id=session_id
                    )
            elif _has_prior_ai and self._trigger.detect(_msg_content):
                # Implicit correction — queue for Accept/Reject instead of
                # asking the user to re-type their intent.
                self._pending_implicit[session_id] = {
                    "query": _msg_content,
                    "domain": "general",
                    "detected_at": time.time(),
                }
                _implicit_pending = True

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
                "review_notes": result.review_notes,
                "correction_stored": _correction_stored,
                "implicit_correction_pending": _implicit_pending,
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

    def _mount_v11_endpoints(self, app: FastAPI, _msg_cache: Any) -> None:
        """
        Mount the v1.1-veritas P1–P3 backport endpoints.

        Persistence (V-P0 complement), search (V-P1.1), backup coverage
        (V-P1.2), implicit corrections (V-P1.3), analytics suite (V-P2.2),
        update management (V-P2.3), corrections CRUD (V-P2.4), bug reporting
        (V-P3.1), local model management (V-P3.3), domain tree (V-P3.4).
        """
        from aua.state import fire_and_forget

        # ── Message write (completes the V-P0 persistence API; feeds V-P1.1) ──

        @app.post(
            "/conversations/{conversation_id}/messages",
            tags=["conversations"],
            summary="Append a message to a conversation",
            status_code=201,
        )
        async def post_conv_message(conversation_id: str, body: dict) -> dict:
            """
            Persist a message and index its keywords.

            The keyword extraction is enqueued to the background worker (never
            blocks the response path); when the worker isn't running (e.g. in
            tests without lifespan) it falls back to synchronous indexing.
            """
            role = body.get("role", "user")
            content = (body.get("content") or "").strip()
            if role not in ("user", "assistant"):
                raise HTTPException(422, "role must be 'user' or 'assistant'")
            if not content:
                raise HTTPException(422, "content must not be empty")
            message_id = self._state_store.add_message(
                conversation_id,
                role=role,
                content=content,
                models_used=body.get("models_used"),
            )
            _msg_cache.invalidate(conversation_id, reason="new_message")
            if self._keyword_index._queue is not None:
                self._keyword_index.enqueue(message_id, conversation_id, role, content)
            else:
                self._keyword_index.index_message_now(message_id, conversation_id, role, content)
            return {"message_id": message_id, "conversation_id": conversation_id}

        # ── V-P1.1: full-text keyword search ──────────────────────────────────

        @app.get("/search", tags=["conversations"], summary="Keyword search over messages")
        async def search_messages(
            q: str = QueryParam("", description="Space-separated keywords (AND semantics)."),
            limit: int = QueryParam(500, ge=1, le=1000),
        ) -> list:
            """
            Message-level search (Cmd+F model): one entry per matching MESSAGE.
            In-memory inverted index when built; DB fallback otherwise.
            """
            if not q.strip():
                return []
            t0 = time.time()
            if self._keyword_index.ready:
                hits = self._keyword_index.search(q, limit=limit)
            else:
                hits = self._keyword_index.search_db_fallback(q, limit=limit)
            if not hits:
                return []
            conv_ids = list(dict.fromkeys(h["conversation_id"] for h in hits))
            placeholders = ",".join("?" * len(conv_ids))
            with self._state_store._connect() as conn:
                rows = conn.execute(
                    f"SELECT conversation_id, title FROM conversations"
                    f" WHERE conversation_id IN ({placeholders})",
                    conv_ids,
                ).fetchall()
            titles = {r[0]: (r[1] or "New Chat") for r in rows}
            latency_ms = round((time.time() - t0) * 1000, 2)
            log.debug("Search %r: %d hits in %.2fms", q, len(hits), latency_ms)
            return [
                {
                    "conversation_id": h["conversation_id"],
                    "title": titles.get(h["conversation_id"], "New Chat"),
                    "message_id": h["message_id"],
                    "match_message_id": h["message_id"],
                    "match_message_ts": h["ts"],
                }
                for h in hits
            ]

        # ── V-P1.2: context backup coverage job ───────────────────────────────

        @app.post(
            "/context/backup/run-coverage-job",
            tags=["context"],
            summary="Trigger a backup coverage sweep now",
        )
        async def run_coverage_job(
            specialist: str | None = QueryParam(
                None, description="Limit the sweep to one specialist."
            ),
        ) -> dict:
            """Start an immediate coverage sweep in the background."""
            specialists = [specialist] if specialist else [s.name for s in self._config.specialists]
            unknown = [
                s for s in specialists if s not in {sp.name for sp in self._config.specialists}
            ]
            if unknown:
                raise HTTPException(404, f"Unknown specialist(s): {unknown}")

            async def _run() -> None:
                try:
                    result = await self._backup_mgr.run_coverage_sweep(
                        specialists, self._generate_backup_text
                    )
                    log.info("Manual coverage sweep: %s", result)
                except Exception as e:  # noqa: BLE001
                    log.error("Manual coverage sweep failed: %s", e)

            fire_and_forget(_run())
            return {
                "ok": True,
                "message": (
                    f"Coverage job started for {len(specialists)} specialist(s). "
                    "Check GET /context/backup/coverage for results."
                ),
                "specialists": specialists,
            }

        # ── V-P1.3: implicit correction confirm/reject ────────────────────────

        @app.post(
            "/corrections/confirm-implicit",
            tags=["corrections"],
            summary="Accept or reject a detected implicit correction",
        )
        async def confirm_implicit(body: dict) -> dict:
            """
            Accept/Reject buttons on the implicit-correction modal.
            Body: { "conversation_id": str, "action": "accept" | "reject" }
            """
            conv_id = body.get("conversation_id", "")
            action = body.get("action", "reject")
            pending = self._pending_implicit.get(conv_id)
            if not pending:
                return {"ok": False, "error": "no pending correction for this conversation"}
            del self._pending_implicit[conv_id]
            if action != "accept":
                return {"ok": True, "stored": False, "message": "Correction discarded."}
            assertion = self._store.add(
                subject=pending["query"][:100],
                domain=pending.get("domain", "general"),
                claim=pending["query"],
                confidence=0.75,
                source="implicit_confirmed",
            )
            correction_id = self._persist_correction(
                assertion, source="implicit_confirmed", session_id=conv_id
            )
            return {
                "ok": True,
                "stored": True,
                "correction_id": correction_id,
                "message": "Correction saved.",
            }

        # ── V-P2.2: analytics / reliability / usage / pricing ─────────────────

        @app.get("/analytics", tags=["telemetry"], summary="Analytics dashboard payload")
        async def get_analytics() -> dict:
            """Session stats, agreement, domain distribution, correction stats."""
            runs = self._state_store.query("model_runs", limit=5000)
            corrs = self._state_store.query("corrections", limit=1000)
            convs = self._state_store.query("conversations", limit=1000, order_by="updated_at DESC")
            answer_runs = [r for r in runs if r.get("round") == "answer"]
            winner_runs = [r for r in answer_runs if r.get("vcg_winner")]

            per_spec: dict[str, dict] = {}
            for r in answer_runs:
                spec = r.get("specialist", "")
                if not spec:
                    continue
                s = per_spec.setdefault(
                    spec,
                    {
                        "specialist": spec,
                        "total_runs": 0,
                        "winner_count": 0,
                        "latencies": [],
                        "welfare": [],
                        "confidence": [],
                    },
                )
                s["total_runs"] += 1
                if r.get("vcg_winner"):
                    s["winner_count"] += 1
                if r.get("latency_ms") is not None:
                    s["latencies"].append(r["latency_ms"])
                if r.get("vcg_welfare_score") is not None:
                    s["welfare"].append(r["vcg_welfare_score"])
                if r.get("confidence_score") is not None:
                    s["confidence"].append(r["confidence_score"])
            specialists_out = []
            for s in per_spec.values():
                n = s["total_runs"] or 1
                specialists_out.append(
                    {
                        "specialist": s["specialist"],
                        "total_runs": s["total_runs"],
                        "winner_count": s["winner_count"],
                        "win_rate_pct": round(s["winner_count"] / n * 100, 1),
                        "avg_latency_ms": (
                            round(sum(s["latencies"]) / len(s["latencies"]), 1)
                            if s["latencies"]
                            else None
                        ),
                        "avg_welfare_score": (
                            round(sum(s["welfare"]) / len(s["welfare"]), 3)
                            if s["welfare"]
                            else None
                        ),
                    }
                )
            specialists_out.sort(key=lambda m: -m["total_runs"])

            conf_hi = conf_med = conf_lo = 0
            for r in winner_runs:
                cs = r.get("confidence_score") or 0.5
                if cs >= 0.75:
                    conf_hi += 1
                elif cs >= 0.50:
                    conf_med += 1
                else:
                    conf_lo += 1

            active_corrs = [c for c in corrs if c.get("scope") != "superseded"]
            corr_by_domain: dict[str, int] = {}
            for c in active_corrs:
                d = c.get("domain", "general")
                corr_by_domain[d] = corr_by_domain.get(d, 0) + 1

            domain_dist: dict[str, int] = {}
            for r in answer_runs:
                d = r.get("domain") or "general"
                domain_dist[d] = domain_dist.get(d, 0) + 1

            welfare = [
                r["vcg_welfare_score"]
                for r in answer_runs
                if r.get("vcg_welfare_score") is not None
            ]
            return {
                "specialists": specialists_out,
                "confidence_dist": {
                    "high": conf_hi,
                    "medium": conf_med,
                    "uncertain": conf_lo,
                    "total": len(winner_runs),
                },
                "correction_stats": {
                    "total_active": len(active_corrs),
                    "by_domain": corr_by_domain,
                },
                "domain_dist": domain_dist,
                "welfare_summary": {
                    "avg": round(sum(welfare) / len(welfare), 3) if welfare else None,
                    "max": round(max(welfare), 3) if welfare else None,
                    "min": round(min(welfare), 3) if welfare else None,
                    "total_scored": len(welfare),
                },
                "total_conversations": len(convs),
                "total_model_runs": len(runs),
            }

        @app.get(
            "/reliability",
            tags=["telemetry"],
            summary="Per-specialist win rate and welfare trajectory",
        )
        async def get_reliability() -> list:
            """Per-specialist win rate and effective-u trajectory (V-P2.2)."""
            runs = self._state_store.query("model_runs", limit=2000, order_by="created_at ASC")
            by_spec: dict[str, list[dict]] = {}
            for r in runs:
                if r.get("round") != "answer":
                    continue
                spec = r.get("specialist", "")
                if spec:
                    by_spec.setdefault(spec, []).append(r)
            result = []
            for spec, rs in by_spec.items():
                wins = sum(1 for r in rs if r.get("vcg_winner"))
                trajectory = [
                    {
                        "created_at": r.get("created_at"),
                        "welfare": r.get("vcg_welfare_score"),
                        "winner": bool(r.get("vcg_winner")),
                    }
                    for r in rs[-20:]
                ]
                recent = [t["welfare"] for t in trajectory if t["welfare"] is not None]
                trend = "flat"
                if len(recent) >= 2:
                    trend = (
                        "up"
                        if recent[-1] > recent[-2]
                        else "down" if recent[-1] < recent[-2] else "flat"
                    )
                result.append(
                    {
                        "specialist": spec,
                        "total_runs": len(rs),
                        "win_rate_pct": round(wins / len(rs) * 100, 1),
                        "trend": trend,
                        "trajectory": trajectory,
                    }
                )
            result.sort(key=lambda m: -float(m["win_rate_pct"]))  # type: ignore[arg-type]
            return result

        @app.get("/usage", tags=["telemetry"], summary="Per-specialist usage and cost")
        async def get_usage() -> dict:
            """Per-specialist query counts and estimated costs (V-P2.2)."""
            runs = self._state_store.query("model_runs", limit=5000)
            counts: dict[str, dict] = {}
            for r in runs:
                spec = r.get("specialist", "")
                if not spec:
                    continue
                c = counts.setdefault(spec, {"count": 0, "last_used": None})
                c["count"] += 1
                ts = r.get("created_at")
                if ts and (c["last_used"] is None or ts > c["last_used"]):
                    c["last_used"] = ts
            pricing = self._pricing_table()
            out, total_cost, total_queries = [], 0.0, 0
            for spec, c in counts.items():
                per_query = pricing.get(spec, {}).get("estimated_cost_per_query", 0.001)
                cost = c["count"] * per_query
                total_cost += cost
                total_queries += c["count"]
                out.append(
                    {
                        "specialist": spec,
                        "query_count": c["count"],
                        "estimated_cost": round(cost, 6),
                        "last_used": c["last_used"],
                    }
                )
            out.sort(key=lambda x: -x["query_count"])
            return {
                "specialists": out,
                "total_cost": round(total_cost, 6),
                "total_queries": total_queries,
            }

        @app.get("/pricing", tags=["telemetry"], summary="Per-specialist token pricing")
        async def get_pricing() -> dict:
            """Per-specialist pricing for cost estimation (V-P2.2)."""
            return {"pricing": self._pricing_table(), "source": self._remote_models.source}

        # ── V-P2.3: update management ─────────────────────────────────────────

        @app.get(
            "/extensions",
            tags=["meta"],
            summary="List loaded plugins, hooks, and middleware",
        )
        async def list_extensions() -> dict:
            """
            Server truth for "did my extension load?" — the CLI's
            `aua extensions list` runs in a fresh process and cannot see
            what the running server loaded.
            """
            from aua.config import _KNOWN_PLUGIN_KINDS

            plugins: dict[str, str | None] = {k: None for k in sorted(_KNOWN_PLUGIN_KINDS)}
            for kind, spec in (self._config.plugins or {}).items():
                plugins[kind] = spec.import_path
            return {
                "plugins": plugins,  # null = using the built-in
                "hooks": get_hook_runner().registered_hooks(),
                "middleware": self._middleware.registered(),
            }

        @app.get("/version/check", tags=["meta"], summary="Check GitHub for a newer release")
        async def version_check() -> dict:
            """Compare the running version against the latest GitHub release."""
            from aua.version import __version__

            repo = os.environ.get("AUA_RELEASES_REPO", "praneethtota/Adaptive-Utility-Agent")
            latest, url = None, None
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(
                        f"https://api.github.com/repos/{repo}/releases/latest",
                        headers={"Accept": "application/vnd.github+json"},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    latest = (data.get("tag_name") or "").lstrip("v") or None
                    url = data.get("html_url")
            except Exception as e:  # noqa: BLE001
                log.debug("version check failed: %s", e)
            skipped = self._state_store.meta_get("skipped_version")
            update_available = bool(latest) and latest != __version__
            return {
                "current": __version__,
                "latest": latest,
                "update_available": update_available,
                "skipped": skipped,
                "show_banner": update_available and latest != skipped,
                "release_url": url,
            }

        @app.post("/update/skip", tags=["meta"], summary="Skip an update version")
        async def update_skip(body: dict) -> dict:
            """Persist a skipped version so the update banner stays hidden."""
            version_str = (body.get("version") or "").strip()
            if not version_str:
                raise HTTPException(422, "version must not be empty")
            self._state_store.meta_set("skipped_version", version_str)
            return {"ok": True, "skipped_version": version_str}

        @app.get("/update/skipped", tags=["meta"], summary="Return the skipped version")
        async def update_skipped() -> dict:
            return {"skipped_version": self._state_store.meta_get("skipped_version")}

        # ── V-P2.4: corrections CRUD + evidence ───────────────────────────────

        @app.get(
            "/corrections/evidence",
            tags=["corrections"],
            summary="Per-correction evidence and application history",
        )
        async def corrections_evidence(
            correction_id: str | None = QueryParam(None),
            include_superseded: bool = QueryParam(False),
            limit: int = QueryParam(100, ge=1, le=1000),
        ) -> dict:
            """Corrections from the state store joined with their event history."""
            rows: list[dict[str, Any]]
            if correction_id:
                row = self._state_store.get("corrections", correction_id)
                rows = [row] if row else []
            else:
                rows = self._state_store.query("corrections", limit=limit)
            if not include_superseded:
                rows = [r for r in rows if r.get("scope") != "superseded"]
            out = []
            for r in rows:
                events = self._state_store.query(
                    "correction_events",
                    filters={"correction_id": r["id"]},
                    limit=100,
                    order_by="created_at ASC",
                )
                out.append({**r, "events": events, "application_count": len(events)})
            return {"total": len(out), "corrections": out}

        @app.patch(
            "/corrections/{correction_id}",
            tags=["corrections"],
            summary="Edit a stored correction",
        )
        async def patch_correction(correction_id: str, body: dict) -> dict:
            """Update the correction text. Logs an 'edited' event."""
            row = self._state_store.get("corrections", correction_id)
            if not row:
                raise HTTPException(404, f"Correction {correction_id!r} not found")
            new_claim = (body.get("claim") or "").strip()
            if not new_claim:
                raise HTTPException(422, "claim must not be empty")
            old_claim = row.get("claim", "")
            with self._state_store._connect() as conn:
                conn.execute(
                    "UPDATE corrections SET claim=? WHERE id=?", (new_claim, correction_id)
                )
            # Best-effort sync of the in-memory prompt-injection store
            for a in self._store.assertions:
                if a.subject == row.get("subject") and a.claim == old_claim:
                    a.claim = new_claim
            self._state_store.append(
                "correction_events",
                {
                    "correction_id": correction_id,
                    "event": "edited",
                    "details": json.dumps({"old": old_claim[:200], "new": new_claim[:200]}),
                },
            )
            return {"ok": True, "correction_id": correction_id, "claim": new_claim}

        @app.delete(
            "/corrections/{correction_id}",
            tags=["corrections"],
            summary="Soft-delete a correction",
        )
        async def delete_correction(correction_id: str) -> dict:
            """
            Soft-delete: sets scope='superseded'. The row stays in the DB for
            audit but is excluded from retrieval and evidence by default.
            """
            row = self._state_store.get("corrections", correction_id)
            if not row:
                raise HTTPException(404, f"Correction {correction_id!r} not found")
            with self._state_store._connect() as conn:
                conn.execute(
                    "UPDATE corrections SET scope='superseded' WHERE id=?", (correction_id,)
                )
            self._store.assertions = [
                a
                for a in self._store.assertions
                if not (a.subject == row.get("subject") and a.claim == row.get("claim"))
            ]
            self._state_store.append(
                "correction_events",
                {"correction_id": correction_id, "event": "superseded"},
            )
            return {"ok": True, "correction_id": correction_id, "scope": "superseded"}

        # ── V-P3.1: bug reporting ─────────────────────────────────────────────

        @app.post("/bug-report", tags=["meta"], summary="Submit a structured bug report")
        async def submit_bug_report(body: dict) -> dict:
            """
            Assemble and submit a bug report via the GitHub Contents API.
            Falls back gracefully when no PAT is configured — returns 200 with
            an error message, never a 500.
            """
            from aua import bug_reporter as _bugs

            report = _bugs.build_report(
                user_token=_bugs.generate_user_token(),
                comment=body.get("comment", ""),
                kind="bug",
                system_log_tail=body.get("system_log_tail", ""),
                api_log_tail=body.get("api_log_tail", ""),
                console_errors=body.get("console_errors", []),
                include_messages=bool(body.get("include_messages", False)),
                last_messages=body.get("last_messages", []),
                user_email=body.get("user_email"),
            )
            pat = _bugs.get_bugs_pat(self._state_store)
            ok, msg = await _bugs.submit_report(report, pat or "")
            return {"ok": ok, "report_id": report["report_id"], "message": msg}

        # ── V-P3.3: local model management ────────────────────────────────────

        @app.get("/local/models", tags=["local"], summary="List registered local models")
        async def list_local_models(user_id: str = QueryParam("local")) -> list:
            rows = self._state_store.query("local_models", filters={"user_id": user_id}, limit=200)
            return rows

        @app.post(
            "/local/models",
            tags=["local"],
            summary="Register a local model",
            status_code=201,
        )
        async def register_local_model(body: dict) -> dict:
            local_model_id = (body.get("local_model_id") or "").strip()
            if not local_model_id:
                raise HTTPException(422, "local_model_id must not be empty")
            now = time.time()
            record = {
                "local_model_id": local_model_id,
                "user_id": body.get("user_id", "local"),
                "ollama_name": body.get("ollama_name") or local_model_id,
                "nickname": body.get("nickname") or local_model_id,
                "base_url": body.get("base_url", "http://localhost:11434"),
                "runtime": body.get("runtime", "ollama"),
                "connected": 1,
                "specialist_domain": body.get("specialist_domain"),
                "specialist_depth": int(body.get("specialist_depth", 0)),
                "created_at": now,
                "updated_at": now,
            }
            cols = ", ".join(record.keys())
            ph = ", ".join("?" for _ in record)
            updates = ", ".join(
                f"{k}=excluded.{k}" for k in record if k not in ("local_model_id", "created_at")
            )
            with self._state_store._connect() as conn:
                conn.execute(
                    f"INSERT INTO local_models ({cols}) VALUES ({ph})"
                    f" ON CONFLICT(local_model_id) DO UPDATE SET {updates}",
                    list(record.values()),
                )
            return record

        @app.patch(
            "/local/specialist/{local_model_id}",
            tags=["local"],
            summary="Tag a local model as a domain specialist",
        )
        async def set_local_specialist(local_model_id: str, body: dict) -> dict:
            """Tag a local model for a domain node; specialist_domain=null untags."""
            rows = self._state_store.query(
                "local_models", filters={"local_model_id": local_model_id}, limit=1
            )
            if not rows:
                raise HTTPException(404, f"Local model {local_model_id!r} not found")
            specialist_domain = body.get("specialist_domain")
            specialist_depth = int(body.get("specialist_depth", 0))
            with self._state_store._connect() as conn:
                conn.execute(
                    "UPDATE local_models"
                    " SET specialist_domain=?, specialist_depth=?, updated_at=?"
                    " WHERE local_model_id=?",
                    (specialist_domain, specialist_depth, time.time(), local_model_id),
                )
            return {
                "ok": True,
                "local_model_id": local_model_id,
                "specialist_domain": specialist_domain,
                "specialist_depth": specialist_depth,
            }

        @app.get("/local/settings", tags=["local"], summary="Read local model settings")
        async def get_local_settings() -> dict:
            raw = self._state_store.meta_get("local_settings")
            return json.loads(raw) if raw else {}

        @app.post("/local/settings", tags=["local"], summary="Write local model settings")
        async def set_local_settings(body: dict) -> dict:
            self._state_store.meta_set("local_settings", json.dumps(body))
            return {"ok": True, "settings": body}

        # ── V-P3.4: domain ontology ───────────────────────────────────────────

        @app.get("/domain-tree", tags=["telemetry"], summary="Domain ontology tree")
        async def get_domain_tree() -> dict:
            """Full ontology with node stats and the candidate queue."""
            nodes = [
                {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "depth": n.depth,
                    "display_name": n.display_name,
                    "alias_count": len(n.aliases),
                    "query_count": n.query_count,
                    "is_l0_root": n.is_l0_root,
                    "promoted_from": n.promoted_from,
                }
                for n in sorted(self._domain_tree.all_nodes(), key=lambda n: (n.depth, n.node_id))
            ]
            candidates = [
                {
                    "raw_string": c.raw_string,
                    "nearest_node": c.nearest_node,
                    "similarity": round(c.similarity, 3),
                    "query_count": c.query_count,
                    "model_count": len(c.model_sources),
                }
                for c in sorted(self._domain_tree.candidates(), key=lambda c: -c.query_count)[:50]
            ]
            return {"nodes": nodes, "candidates": candidates}

    def _pricing_table(self) -> dict[str, dict]:
        """Per-specialist pricing derived from the live model registry (V-P2.2)."""
        pricing: dict[str, dict] = {}
        models = self._remote_models.models
        for spec in self._config.specialists:
            entry = models.get(spec.model, {})
            input_cost = entry.get("input_cost_per_1m")
            output_cost = entry.get("output_cost_per_1m")
            if input_cost is not None or output_cost is not None:
                # ~1K in + 1K out per query as the estimation basis
                est = ((input_cost or 0.0) + (output_cost or 0.0)) / 1000.0
            else:
                est = 0.0  # local/self-hosted models: no per-token cost
            pricing[spec.name] = {
                "model": spec.model,
                "input_cost_per_1m": input_cost,
                "output_cost_per_1m": output_cost,
                "estimated_cost_per_query": round(est, 6),
            }
        return pricing

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

        # #51: routing strategy plugin — may reorder or override distribution
        if self._routing_strategy is not None and not req.force_domain:
            try:
                _route_meta = {
                    "session_id": req.session_id,
                }
                distribution = self._routing_strategy.route(req.query, distribution, _route_meta)
            except Exception as _rs_err:  # noqa: BLE001
                log.debug("Routing strategy plugin failed, using classifier: %s", _rs_err)

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

    async def _evaluate_green(self, req: DeployGreenRequest) -> DeployGreenResponse:  # noqa: C901
        """
        Evaluate GREEN vs BLUE with regression gate (#49) and shadow score support (#48).

        Decision path:
          1. Run regression dataset against BLUE (baseline) and GREEN (candidate).
             Block promotion if regression detected and regression_block=True.
          2. If enough shadow scores have accumulated, use them for U comparison
             instead of a fresh synthetic eval run.
          3. If no green_endpoint and no shadow scores: score BLUE only (dry run).
        """
        spec = next((s for s in self._config.specialists if s.name == req.specialist), None)
        if spec is None:
            raise HTTPException(404, f"Specialist '{req.specialist}' not found in config.")

        bg_cfg = self._config.blue_green_for(req.specialist)
        threshold = bg_cfg.delta

        # ── #49: regression gate ──────────────────────────────────────────────
        regression_result: dict | None = None
        dataset_path = req.regression_dataset or bg_cfg.regression_dataset
        green_endpoint = req.green_endpoint or bg_cfg.shadow_endpoint

        if dataset_path:
            import pathlib

            from aua.eval import run_dataset

            ds_path = pathlib.Path(dataset_path)
            if not ds_path.exists():
                raise HTTPException(400, f"Regression dataset not found: {dataset_path}")

            # Run against BLUE (production)
            blue_report = run_dataset(
                ds_path, router_url=f"http://127.0.0.1:{self._config.router.port}"
            )

            # Run against GREEN if endpoint available
            if green_endpoint:
                # Temporarily patch the router URL to GREEN's base URL to hit it directly
                # GREEN is a vLLM-compatible server — use its /v1/chat/completions
                # We pass the raw GREEN endpoint as the router_url for the dataset runner
                # (run_dataset uses /query on the router, so we can't directly hit
                # the specialist endpoint — instead score GREEN via the shadow call path)
                # Use accumulated shadow scores if available and sufficient
                shadow_agg = self._shadow_mgr.report(req.specialist)
                if shadow_agg.n_queries >= bg_cfg.shadow_min_queries:
                    # Use shadow scores
                    green_u_for_regression = shadow_agg.green_mean_u
                    _blue_u_for_regression = shadow_agg.blue_mean_u  # used in EvalReport below
                else:
                    # Fresh eval against GREEN directly (score only, no routing)
                    green_scores: list[float] = []
                    for case in (blue_report.cases or [])[: min(10, len(blue_report.cases))]:
                        try:
                            text, conf = await self._call(
                                green_endpoint,
                                case.get("id", case.get("prompt", "")),
                                spec.field,
                                model_name="green",
                            )
                            gu, *_ = await self._score(case.get("id", ""), text, spec.field, conf)
                            green_scores.append(float(gu))
                        except Exception:
                            green_scores.append(0.0)
                    green_u_for_regression = (
                        sum(green_scores) / len(green_scores)
                        if green_scores
                        else blue_report.mean_u_score
                    )

                # Build a synthetic green report for regression_vs
                from aua.eval import EvalReport

                green_report_mock = EvalReport(
                    dataset_name="green",
                    field=blue_report.field,
                    description="GREEN evaluation",
                    run_at=blue_report.run_at,
                    router_url=blue_report.router_url,
                    total=blue_report.total,
                    passed=blue_report.passed,
                    failed=blue_report.failed,
                    error=blue_report.error,
                    mean_u_score=green_u_for_regression,
                    mean_latency_ms=blue_report.mean_latency_ms,
                    pass_rate=blue_report.pass_rate,
                    cases=blue_report.cases,
                )
                reg = green_report_mock.regression_vs(blue_report)
            else:
                # No green endpoint — can only check BLUE against itself (always OK)
                reg = {
                    "regressed": False,
                    "verdict": "SKIPPED — no green_endpoint provided",
                    "delta_pass_rate": 0.0,
                    "delta_u_score": 0.0,
                    "delta_latency_ms": 0.0,
                }

            regression_result = {
                **reg,
                "dataset": dataset_path,
                "blocked": reg.get("regressed", False) and bg_cfg.regression_block,
            }
            log.info(
                "Regression check for %s: verdict=%s regressed=%s blocked=%s",
                req.specialist,
                reg.get("verdict"),
                reg.get("regressed"),
                regression_result["blocked"],
            )

            if regression_result["blocked"]:
                return DeployGreenResponse(
                    specialist=req.specialist,
                    promoted=False,
                    u_delta=reg.get("delta_u_score", 0.0),
                    blue_u=blue_report.mean_u_score,
                    green_u=blue_report.mean_u_score + reg.get("delta_u_score", 0.0),
                    threshold=threshold,
                    dry_run_only=False,
                    message=(
                        f"PROMOTION BLOCKED — regression detected on '{dataset_path}'. "
                        f"Pass rate delta: {reg.get('delta_pass_rate', 0):+.3f}, "
                        f"U delta: {reg.get('delta_u_score', 0):+.4f}. "
                        f"Set regression_block: false to warn-only."
                    ),
                    regression=regression_result,
                )

        # ── U score comparison (shadow or synthetic) ──────────────────────────
        shadow_report = self._shadow_mgr.report(req.specialist)
        if shadow_report.n_queries >= bg_cfg.shadow_min_queries:
            # Use accumulated shadow scores — real traffic, most reliable
            blue_u = shadow_report.blue_mean_u
            green_u = shadow_report.green_mean_u
            source = f"shadow ({shadow_report.n_queries} queries)"
            dry = False
        elif green_endpoint:
            # Fresh synthetic eval against GREEN
            eval_qs = [
                "Write a binary search function. State the time complexity.",
                "Implement merge sort. State time and space complexity.",
                "Write a function to check if a string is a palindrome.",
                "Reverse a linked list. State the space complexity.",
                "Implement a stack using two queues.",
            ][: req.n_eval_queries or 3]

            blue_scores: list[float] = []
            green_scores_list: list[float] = []
            for q in eval_qs:
                try:
                    bt, bc = await self._call(
                        spec.endpoint, q, spec.field, model_name=spec.serve_model_name
                    )
                    bu, *_ = await self._score(q, bt, spec.field, bc)
                    blue_scores.append(float(bu))
                except Exception:
                    blue_scores.append(0.0)
                try:
                    gt, gc = await self._call(green_endpoint, q, spec.field, model_name="green")
                    gu, *_ = await self._score(q, gt, spec.field, gc)
                    green_scores_list.append(float(gu))
                except Exception:
                    green_scores_list.append(0.0)

            blue_u = round(sum(blue_scores) / len(blue_scores), 4) if blue_scores else 0.0
            green_u = (
                round(sum(green_scores_list) / len(green_scores_list), 4)
                if green_scores_list
                else 0.0
            )
            source = f"synthetic eval ({len(eval_qs)} queries)"
            dry = False
        else:
            # No endpoint, no shadow — dry run only
            eval_qs = [
                "Write a binary search function. State the time complexity.",
                "Implement merge sort. State time and space complexity.",
                "Write a function to check if a string is a palindrome.",
            ][: req.n_eval_queries or 3]
            blue_scores = []
            for q in eval_qs:
                try:
                    text, conf = await self._call(
                        spec.endpoint, q, spec.field, model_name=spec.serve_model_name
                    )
                    u_val, *_ = await self._score(q, text, spec.field, conf)
                    blue_scores.append(float(u_val))
                except Exception:
                    blue_scores.append(0.0)
            blue_u = round(sum(blue_scores) / len(blue_scores), 4) if blue_scores else 0.0
            green_u = blue_u
            source = "dry-run (no green_endpoint)"
            dry = True

        u_delta = round(green_u - blue_u, 4)

        # promotion_policy plugin — replaces the built-in u_delta >= threshold gate
        if self._custom_promotion_policy is not None and not dry:
            try:
                # Build full context for should_promote_full() / should_promote()
                _shadow_rows = self._shadow_store.get_scores(req.specialist)
                _shadow_deltas = [r["u_delta"] for r in _shadow_rows]
                _shadow_std = statistics.stdev(_shadow_deltas) if len(_shadow_deltas) >= 2 else 0.0
                _promotion_context = {
                    "specialist": req.specialist,
                    "blue_u": blue_u,
                    "green_u": green_u,
                    "u_delta": u_delta,
                    "mean_delta": (
                        shadow_report.mean_delta if not shadow_report.n_queries == 0 else u_delta
                    ),
                    "n_queries": (
                        shadow_report.n_queries
                        if shadow_report.n_queries > 0
                        else (req.n_eval_queries or 3)
                    ),
                    "min_queries": bg_cfg.shadow_min_queries,
                    "threshold": threshold,
                    "shadow_scores": _shadow_rows,
                    "shadow_std_delta": _shadow_std,
                    "regression_result": regression_result,
                    "dry": dry,
                    "source": source,
                    "specialist_config": spec,
                    "bg_config": bg_cfg,
                }
                if hasattr(self._custom_promotion_policy, "should_promote_full") and callable(
                    getattr(self._custom_promotion_policy, "should_promote_full")
                ):
                    try:
                        promoted = bool(
                            self._custom_promotion_policy.should_promote_full(_promotion_context)
                        )
                    except Exception as _sf_err:  # noqa: BLE001
                        log.debug(
                            "should_promote_full() failed, falling back to should_promote(): %s",
                            _sf_err,
                        )
                        promoted = bool(
                            self._custom_promotion_policy.should_promote(
                                specialist=req.specialist,
                                blue_mean_u=blue_u,
                                green_mean_u=green_u,
                                n_queries=_promotion_context["n_queries"],
                                metadata=_promotion_context,
                            )
                        )
                else:
                    promoted = bool(
                        self._custom_promotion_policy.should_promote(
                            specialist=req.specialist,
                            blue_mean_u=blue_u,
                            green_mean_u=green_u,
                            n_queries=_promotion_context["n_queries"],
                            metadata=_promotion_context,
                        )
                    )
            except Exception as _pp_err:  # noqa: BLE001
                log.debug("Promotion policy plugin failed, using built-in: %s", _pp_err)
                promoted = not dry and u_delta >= threshold
        else:
            promoted = not dry and u_delta >= threshold

        shadow_note = (
            f" Shadow: {shadow_report.n_queries}/{bg_cfg.shadow_min_queries} queries."
            if not shadow_report.ready_to_promote and shadow_report.active
            else ""
        )

        return DeployGreenResponse(
            specialist=req.specialist,
            promoted=promoted,
            u_delta=u_delta,
            blue_u=blue_u,
            green_u=green_u,
            threshold=threshold,
            dry_run_only=dry,
            message=(
                f"GREEN promoted via {source} (U_delta {u_delta:+.4f} >= {threshold})"
                if promoted
                else (
                    f"GREEN not promoted via {source} "
                    f"(U_delta {u_delta:+.4f} < {threshold} or dry run).{shadow_note}"
                )
            ),
            regression=regression_result,
        )

    # ── Core routing (buffered) ───────────────────────────────────────────────

    def _load_yaml_extensions(self) -> None:
        """
        Load `plugins:`, `hooks:`, and `middleware:` from aua_config.yaml.

        Wiring per kind:
          field_classifier — replaces self._classifier (classify() shim keeps
                             the built-in update_history kwarg compatible)
          utility_scorer   — overrides the final U score via the plugin's
                             score(response, field, prior_u, confidence,
                             metadata) after the built-in pipeline runs
          correction_store — replaces self._store when it exposes the
                             AssertionsStore-compatible add()/query() surface
          hooks            — registered on the global HookRunner per entry
          middleware       — ordered before_query/after_response pipeline

        arbiter_policy, promotion_policy, model_backend, and state_store load
        and contract-validate (fail-fast on typos) but wire programmatically —
        see tutorial How-to 13 for the assignment points.
        """
        # Make project-local plugin modules importable. `aua` is an installed
        # entry point, so the project directory is NOT on sys.path the way it
        # is for `python manage.py` — insert the config file's directory (or
        # CWD) first, exactly like Django does for the project root.
        import sys as _sys

        from aua.plugins.registry import load_plugin

        _project_dir = (
            str(Path(self._config_path).resolve().parent) if self._config_path else os.getcwd()
        )
        if _project_dir not in _sys.path:
            _sys.path.insert(0, _project_dir)

        class _ClassifierShim:
            """Adapt a FieldClassifierPlugin to the built-in call signature."""

            def __init__(self, plugin: Any) -> None:
                self._plugin = plugin

            def classify(self, query: str, update_history: bool = False) -> dict[str, float]:
                return self._plugin.classify(query)

        for kind, spec in (self._config.plugins or {}).items():
            plugin = load_plugin(spec.import_path, kind, spec.config)
            if kind == "field_classifier":
                self._classifier = _ClassifierShim(plugin)  # type: ignore[assignment]
            elif kind == "utility_scorer":
                self._custom_scorer = plugin
            elif kind == "correction_store" and hasattr(plugin, "query"):
                self._store = plugin
            # #51: wire new plugin types
            elif kind == "contradiction_detector":
                self._custom_detector = plugin
                log.info("Contradiction detector replaced by plugin: %s", spec.import_path)
            elif kind == "assertion_store":
                self._custom_assertion_store = plugin
                log.info("Assertion store replaced by plugin: %s", spec.import_path)
            elif kind == "routing_strategy":
                self._routing_strategy = plugin
                log.info("Routing strategy plugin registered: %s", spec.import_path)
            elif kind == "scoring_component":
                self._scoring_component = plugin
                log.info("Scoring component plugin registered: %s", spec.import_path)
            elif kind == "arbiter_policy":
                self._custom_arbiter_policy = plugin
                log.info("Arbiter policy replaced by plugin: %s", spec.import_path)
            elif kind in ("promotion_policy", "full_promotion_policy"):
                self._custom_promotion_policy = plugin
                log.info("Promotion policy replaced by plugin: %s", spec.import_path)
            log.info("Plugin loaded from config: %s ← %s", kind, spec.import_path)

        runner = get_hook_runner()
        for h in self._config.hooks or []:
            hook = load_plugin(h.import_path, "hook", h.config)
            runner.register(h.hook_point, hook, fail_closed=h.fail_closed)
            log.info("Hook registered from config: %s ← %s", h.hook_point, h.import_path)

        for m in self._config.middleware or []:
            mw = load_plugin(m.import_path, "middleware", m.config)
            self._middleware.add(mw)
            log.info("Middleware registered from config: %s", m.import_path)

    async def _handle(self, req: QueryRequest) -> RouterResponse:
        t0 = time.time()
        log.info("Query: %.80s", req.query)
        _hooks = get_hook_runner()

        # ── #15: session/trace/request IDs ────────────────────────────────────
        # The HTTP middleware sets a SessionContext per request; the library
        # API (Router.query) creates one here. Body session_id wins over the
        # header; when neither is supplied the generated UUID is echoed back
        # so clients can adopt it for the rest of the conversation.
        from aua.session import get_current_or_none, new_session_context

        _ctx = get_current_or_none()
        if _ctx is None:
            _ctx = new_session_context(session_id=req.session_id)
        if req.session_id:
            _ctx.session_id = req.session_id
        else:
            req.session_id = _ctx.session_id
        _sid = req.session_id
        _tid = _ctx.trace_id

        # ── F-11: middleware before_query (may rewrite the query) ────────────
        if self._middleware.registered():
            _mw_req = await self._middleware.before_query(
                {
                    "query": req.query,
                    "session_id": _sid,
                    "conversation_history": req.conversation_history or [],
                }
            )
            req.query = _mw_req.get("query", req.query)

        # ── pre_query: before field classification ────────────────────────────
        await _hooks.fire(
            "pre_query",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "request_id": _ctx.request_id,
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

        # #51: routing strategy plugin — may reorder or override distribution
        if self._routing_strategy is not None and not req.force_domain:
            try:
                _route_meta = {"session_id": req.session_id}
                distribution = self._routing_strategy.route(req.query, distribution, _route_meta)
            except Exception as _rs_err:  # noqa: BLE001
                log.debug("Routing strategy plugin failed, using classifier: %s", _rs_err)

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
        # #15: enrich the context so logs/audit carry domain + routing mode
        _ctx.domain = top_domain
        _ctx.routing_mode = routing_mode

        # ── post_route: after routing decision, before specialist calls ───────
        await _hooks.fire(
            "post_route",
            {
                "session_id": _sid,
                "trace_id": _tid,
                "request_id": _ctx.request_id,
                "query": req.query,
                "domain_distribution": distribution,
                "top_domain": top_domain,
                "routing_mode": routing_mode,
                "active_specialists": [s.name for s in active],
            },
        )

        if len(active) >= 2:
            resp = await self._handle_fanout(req, active, distribution, t0, _sid, _tid)
        elif top_prob >= self._single_threshold:
            resp = await self._handle_single(req, top_domain, distribution, t0, _sid, _tid)
        else:
            resp = await self._handle_arbiter(req, distribution, t0, _sid, _tid)

        # ── F-11: middleware after_response (reverse order, may rewrite) ─────
        if self._middleware.registered():
            _mw_resp = await self._middleware.after_response(resp.model_dump())
            resp = RouterResponse(**_mw_resp)

        # #47: experiment tracking — fire-and-forget, never blocks the response
        try:
            self._experiment_tracker.log(
                {
                    "u_score": resp.u_score,
                    "confidence": resp.confidence,
                    "latency_ms": resp.latency_ms,
                    "contradictions_detected": resp.contradictions_detected,
                    "corrections_injected": resp.corrections_injected,
                    "dpo_pairs_generated": resp.dpo_pairs_generated,
                    "routing_mode": resp.routing_mode,
                    "primary_domain": resp.primary_domain,
                    "session_id": resp.session_id,
                    "trace_id": resp.trace_id,
                }
            )
        except Exception as _et_err:  # noqa: BLE001
            log.debug("experiment_tracker.log failed: %s", _et_err)

        return resp

    def _response_ids(self, trace_id: str) -> dict[str, str | None]:
        """#15: trace/request IDs echoed in every RouterResponse."""
        from aua.session import get_current_or_none

        ctx = get_current_or_none()
        return {"trace_id": trace_id, "request_id": ctx.request_id if ctx else None}

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
            **self._response_ids(_tid),
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
                **self._response_ids(_tid),
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

        # #48: shadow call — fire GREEN silently, never blocks the response
        if spec and self._shadow_mgr.is_active(spec.name):
            from aua.state import fire_and_forget

            fire_and_forget(
                self._shadow_mgr.shadow_call(
                    specialist=spec.name,
                    query=req.query,
                    domain=domain,
                    blue_u=u,
                    call_fn=self._call,
                    score_fn=self._score,
                    model_name="green",
                )
            )

        return resp

    async def _handle_fanout(
        self, req, active_specialists, distribution, t0, _sid="", _tid=""
    ) -> RouterResponse:
        log.info("fanout → %s", [s.name for s in active_specialists])
        _hooks = get_hook_runner()
        _review_notes: str | None = None  # V-P2.1: populated when arbiter flags issues

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
            # ── Persist per-specialist VCG run data to model_runs ────────────
            # vcg_winner, vcg_welfare_score, specialist, domain, confidence_score
            # are queried by /analytics and /reliability. fire_and_forget keeps
            # the response path non-blocking.
            from aua.state import fire_and_forget as _faf

            _vcg_run_time = time.time()
            for _s, _t, _c in responses:
                _is_winner = _s.name == winner_spec.name
                _wf = _vcg_welfare.get(_s.name, 0.0)

                async def _write_run(
                    sname=_s.name, sfield=_s.field, is_w=_is_winner, wf=_wf, c=_c
                ) -> None:
                    self._state_store.record_model_run(
                        {
                            "conversation_id": req.session_id or "",
                            "specialist": sname,
                            "domain": sfield,
                            "round": "answer",
                            "vcg_winner": 1 if is_w else 0,
                            "vcg_welfare_score": round(wf, 6),
                            "confidence_score": round(c, 4),
                            "latency_ms": round((time.time() - _vcg_run_time) * 1000, 1),
                            "created_at": _vcg_run_time,
                        }
                    )

                _faf(_write_run())

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

            # V-P2.1: surface reviewer findings instead of discarding them
            _review_notes = self._parse_review_notes(verdict, reviewer=self._config.arbiter.model)

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
            review_notes=_review_notes,
            **self._response_ids(_tid),
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
            **self._response_ids(_tid),
        )

    # ── Specialist call (buffered) ────────────────────────────────────────────

    # ── VCG constants (§10.6.7.1 / Appendix B §B.8) ──────────────────────────
    _VCG_N_CLIFF: int = 10  # Efron-Morris pseudo-count; see Lemma B.8.1
    _VCG_GLOBAL_PRIOR: float = 0.65  # cross-domain win-rate prior

    def _vcg_effective_u(self, specialist_name: str, domain: str) -> float:
        """
        Shrinkage-corrected win-rate for specialist i in domain j (§10.6.7.1 Point 2).

        eu(i, j) = (n_ij * u_hat_ij + N_cliff * u_bar) / (n_ij + N_cliff)

        where:
          n_ij    — observations of specialist i answering queries in domain j
          u_hat_ij — raw win-rate (fraction of queries where this specialist won)
          N_cliff  — Efron-Morris pseudo-count (default 10)
          u_bar    — global cross-domain prior (default 0.65)

        When n_ij >= N_cliff the estimate converges to the raw win-rate.
        When n_ij < N_cliff it is pulled toward the prior, preventing extreme
        values from dominating the welfare sum (Lemma B.8.1).

        Data source: model_runs table (vcg_winner flag, specialist + domain columns).
        Falls back to the prior when no data is available.
        """
        N_c = self._VCG_N_CLIFF
        u_bar = self._VCG_GLOBAL_PRIOR
        try:
            runs = self._state_store.query(
                "model_runs",
                filters={"specialist": specialist_name, "domain": domain, "round": "answer"},
                limit=500,
            )
            n = len(runs)
            if n == 0:
                return u_bar
            wins = sum(1 for r in runs if r.get("vcg_winner"))
            u_hat = wins / n
            return (n * u_hat + N_c * u_bar) / (n + N_c)
        except Exception as _eu_err:
            log.debug("effective_u lookup failed for %s/%s: %s", specialist_name, domain, _eu_err)
            return u_bar

    def _vcg_welfare(
        self,
        spec_name: str,
        distribution: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """
        Multi-domain welfare score (§10.6.7.1 Point 1):

        W_i(q) = Σ_j  p(j|q) · effective_u(i, j)

        Only domains with p(j|q) >= 0.05 contribute (below that the term is
        negligible and the DB lookup is wasteful). The result is a convex
        combination of per-domain utilities (Proposition B.8.3 P3).

        Returns (W_i, per_domain_breakdown) for logging.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        breakdown: dict[str, float] = {}
        for domain, p in distribution.items():
            if p < 0.05:
                continue
            eu = self._vcg_effective_u(spec_name, domain)
            breakdown[domain] = round(eu, 4)
            weighted_sum += p * eu
            total_weight += p
        # Re-normalise if we dropped low-probability domains
        if total_weight > 0.0:
            welfare = weighted_sum / total_weight
        else:
            welfare = self._VCG_GLOBAL_PRIOR
        return round(welfare, 6), breakdown

    def _vcg_select(
        self,
        responses: list[tuple],  # [(SpecialistConfig, text, confidence), ...]
        distribution: dict[str, float],
    ) -> tuple[int, dict[str, float]]:
        """
        VCG welfare maximization: select the specialist with the highest
        multi-domain welfare score (§10.6.7.1, Theorems S1-S3, Appendix B §B.8).

        W_i(q) = Σ_j  p(j|q) · effective_u(i, j)

          effective_u(i, j) — shrinkage-corrected win-rate (Lemma B.8.1)
          p(j|q)            — field classifier domain probability

        Returns (winner_index, welfare_scores_dict).
        Ties are broken by raw confidence, then by top-domain classifier probability.
        """
        welfare: dict[str, float] = {}
        for spec, _text, conf in responses:
            w, breakdown = self._vcg_welfare(spec.name, distribution)
            welfare[spec.name] = w
            log.debug(
                "VCG W(%s) = %.4f  breakdown=%s  conf=%.3f",
                spec.name,
                w,
                breakdown,
                conf,
            )

        # argmax by welfare, tie-break: confidence, then P(top domain)
        winner_idx = max(
            range(len(responses)),
            key=lambda i: (
                welfare[responses[i][0].name],
                responses[i][2],
                distribution.get(responses[i][0].field, 0.0),
            ),
        )
        return winner_idx, welfare

    # ── v1.1-veritas helpers ──────────────────────────────────────────────────

    async def _generate_backup_text(
        self,
        specialist: str,
        conversation_id: str,
        prompt: str,
        history: list[dict],
    ) -> str:
        """
        Ask a specialist to produce a structured context backup (V-P1.2/1.4).

        Full-history rule (V-P0.5): `history` comes from the canonical DB read
        in ContextBackupManager.build_backup_context(), never the request body.
        """
        spec = next((s for s in self._config.specialists if s.name == specialist), None)
        if spec is None:
            return ""
        chat_history = [
            {"role": m.get("role", "user"), "content": m.get("content") or ""}
            for m in history
            if m.get("content")
        ]
        text, _conf = await self._call(
            spec.endpoint,
            "Write the context handoff note now.",
            spec.field,
            history=chat_history,
            system_prompt=prompt,
            model_name=spec.serve_model_name,
        )
        return text

    @staticmethod
    def _parse_review_notes(verdict_text: str, reviewer: str) -> str | None:
        """
        Surface reviewer findings to the client (V-P2.1).

        Parses REASON:/ISSUES: and CORRECTION: structured sections from the
        arbiter output. Returns None when nothing useful was flagged —
        previously the framework discarded these after parsing the verdict.
        """
        if not verdict_text:
            return None
        sections: list[str] = []
        for label in ("ISSUES", "REASON", "CORRECTION"):
            m = re.search(
                rf"{label}:\s*(.+?)(?=\n[A-Z]+:|\Z)", verdict_text, re.DOTALL | re.IGNORECASE
            )
            if m:
                body = m.group(1).strip()
                if body and body.lower() not in ("none", "n/a", "-"):
                    sections.append(f"{label}: {body[:400]}")
        if not sections:
            return None
        return f"Reviewer: {reviewer}. " + " | ".join(sections)

    def _persist_correction(
        self,
        assertion: Any,
        source: str,
        session_id: str = "",
    ) -> str:
        """
        Dual-write a correction to the state store (V-P2.4) so it has a
        persistent ID for PATCH/DELETE/evidence, alongside the in-memory
        AssertionsStore used for prompt injection.
        """
        correction_id = self._state_store.append(
            "corrections",
            {
                "subject": assertion.subject,
                "domain": assertion.domain,
                "claim": assertion.claim,
                "rejected": "",
                "confidence": assertion.confidence_at_write,
                "source": source,
                "effective_confidence": round(assertion.effective_confidence(), 4),
                "decay_class": assertion.decay_class.value,
                "scope": "global",
            },
        )
        self._state_store.append(
            "correction_events",
            {
                "correction_id": correction_id,
                "event": "created",
                "session_id": session_id,
                "details": json.dumps({"source": source}),
            },
        )
        return correction_id

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

        # #15: propagate session/trace/request IDs to downstream services
        from aua.session import get_current_or_none as _gcon

        _prop_ctx = _gcon()
        _prop_headers = _prop_ctx.as_headers() if _prop_ctx else {}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(
                    url,
                    headers=_prop_headers,
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
        # #51: custom contradiction detector plugin
        if self._custom_detector is not None:
            try:
                _det_raw = self._custom_detector.check(problem=query, solution=response)
                # Wrap dict result in a ContradictionResult-compatible object
                from aua.contradiction_detector import Contradiction, ContradictionResult

                result = ContradictionResult()
                for c in _det_raw.get("contradictions", []):
                    if isinstance(c, Contradiction):
                        result.contradictions.append(c)
                    else:
                        result.contradictions.append(
                            Contradiction(
                                type=c.get("type", "custom"),
                                description=c.get("description", str(c)),
                                severity=float(c.get("severity", 0.5)),
                            )
                        )
                result.confidence_penalty = float(_det_raw.get("confidence_penalty", 0.0))
            except Exception as _det_err:  # noqa: BLE001
                log.debug("Custom detector failed, using built-in: %s", _det_err)
                result = self._detector.check(problem=query, solution=response)
        else:
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

        # #51: scoring component plugin — adjusts individual E/C/K components
        base_u = task_score.utility
        if self._scoring_component is not None:
            try:
                _comp_meta = {
                    "query": query,
                    "response": response,
                    "pass_rate": base_conf,
                    "contradiction_penalty": result.confidence_penalty,
                }
                _e = self._scoring_component.compute(
                    "efficacy", task_score.efficacy_ema, domain, _comp_meta
                )
                _c = self._scoring_component.compute(
                    "confidence", task_score.confidence, domain, _comp_meta
                )
                _k = self._scoring_component.compute(
                    "curiosity", task_score.curiosity_effective, domain, _comp_meta
                )
                fc = FIELD_CONFIGS.get(domain, FIELD_CONFIGS["general"])
                base_u = max(
                    0.0,
                    min(
                        1.0,
                        fc.w_efficacy * float(_e)
                        + fc.w_confidence * float(_c)
                        + fc.w_curiosity * float(_k),
                    ),
                )
            except Exception as _sc_err:  # noqa: BLE001
                log.debug("Scoring component plugin failed, using built-in: %s", _sc_err)
                base_u = task_score.utility

        # Apply policy bonus/penalty to final U score
        base_u = base_u  # reassignment for clarity after component override
        if e_bonus > 0.0:
            # Boost E component: U = w_e*(E+bonus) + rest
            from aua.config import FIELD_CONFIGS as _FC

            fc = _FC.get(domain, _FC["general"])
            u_adjusted = min(1.0, base_u + fc.w_efficacy * e_bonus)
        else:
            u_adjusted = base_u
        if u_penalty > 0.0:
            u_adjusted = max(0.0, u_adjusted - u_penalty)

        # F-09 / #53: utility_scorer plugin — adjustment or full replacement.
        # score_full() (#53): receives raw E/C/K + weights, replaces built-in U.
        # score() (F-09):     receives prior_u (built-in result), adjusts it.
        if self._custom_scorer is not None:
            _scorer_meta = {
                "query": query,
                "response": response,
                "contradictions": n_contra,
                "task_score": task_score,
            }
            if hasattr(self._custom_scorer, "score_full") and callable(
                getattr(self._custom_scorer, "score_full")
            ):
                # #53: full replacement — skip built-in U, compute from components
                try:
                    fc = FIELD_CONFIGS.get(domain, FIELD_CONFIGS["general"])
                    u_adjusted = float(
                        self._custom_scorer.score_full(
                            field=domain,
                            efficacy=task_score.efficacy_ema,
                            confidence=task_score.confidence,
                            curiosity=task_score.curiosity_effective,
                            weights={
                                "w_e": fc.w_efficacy,
                                "w_c": fc.w_confidence,
                                "w_k": fc.w_curiosity,
                            },
                            metadata=_scorer_meta,
                        )
                    )
                    u_adjusted = max(0.0, min(1.0, u_adjusted))
                except Exception as _sf_err:  # noqa: BLE001
                    log.debug("score_full() failed, falling back to score(): %s", _sf_err)
                    try:
                        u_adjusted = float(
                            self._custom_scorer.score(
                                response=response,
                                field=domain,
                                prior_u=u_adjusted,
                                confidence=updated_conf,
                                metadata=_scorer_meta,
                            )
                        )
                        u_adjusted = max(0.0, min(1.0, u_adjusted))
                    except Exception as _scorer_err:  # noqa: BLE001
                        log.error("Custom scorer failed (using built-in U): %s", _scorer_err)
            else:
                # Adjustment mode: plugin receives built-in U and adjusts
                try:
                    u_adjusted = float(
                        self._custom_scorer.score(
                            response=response,
                            field=domain,
                            prior_u=u_adjusted,
                            confidence=updated_conf,
                            metadata=_scorer_meta,
                        )
                    )
                    u_adjusted = max(0.0, min(1.0, u_adjusted))
                except Exception as _scorer_err:  # noqa: BLE001
                    log.error("Custom utility scorer failed (using built-in U): %s", _scorer_err)

        return round(u_adjusted, 4), updated_conf, n_contra, n_dpo

    # ── Arbitration ───────────────────────────────────────────────────────────

    async def _arbitrate(
        self, query: str, spec_a, text_a: str, spec_b, text_b: str
    ) -> tuple[str, str]:
        # arbiter_policy plugin — replaces the built-in LLM arbitration call
        if self._custom_arbiter_policy is not None:
            try:
                _arb_meta = {
                    "domain_a": spec_a.field,
                    "domain_b": spec_b.field,
                    "specialist_a": spec_a.name,
                    "specialist_b": spec_b.name,
                }
                _arb_result = self._custom_arbiter_policy.arbitrate(
                    subject=query[:100],
                    domain=spec_a.field,
                    output_a=text_a,
                    output_b=text_b,
                    metadata=_arb_meta,
                )
                _winner_key = _arb_result.get("winner", "A").upper()
                _winner_field = (
                    spec_b.field
                    if _winner_key == "B"
                    else "both_wrong" if _winner_key == "BOTH_WRONG" else spec_a.field
                )
                _verdict_text = (
                    f"VERDICT: {_winner_key}\n"
                    f"REASON: {_arb_result.get('reason', '')}\n"
                    f"CORRECTION: {_arb_result.get('external_response', '')}"
                )
                log.info("Custom arbiter policy verdict: %s", _winner_field)
                return _verdict_text, _winner_field
            except Exception as _arb_err:  # noqa: BLE001
                log.debug("Arbiter policy plugin failed, using built-in: %s", _arb_err)

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
