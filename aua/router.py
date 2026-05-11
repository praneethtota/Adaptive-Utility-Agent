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


class Router:
    """
    Micro-Expert Architecture Router.

    Routes each query to the right specialist based on field classification,
    fans out to multiple specialists on cross-domain queries, runs the Arbiter
    on conflicts, and accumulates DPO pairs from detected contradictions.

    All routing parameters come from AUAConfig — no hardcoded ports or
    thresholds anywhere in this class.
    """

    def __init__(self, config: AUAConfig) -> None:
        self._config = config
        self._classifier = FieldClassifier()
        self._scorer = UtilityScorer()
        self._store = AssertionsStore()
        self._detector = ContradictionDetector(penalty_multiplier=2.0)
        self._conf = ConfidenceUpdater()

        self._domain_confidence: dict[str, float] = {s.field: 0.5 for s in config.specialists}
        self._field_to_url: dict[str, str] = {s.field: s.endpoint for s in config.specialists}
        self._arbiter_url = config.arbiter.endpoint
        self._single_threshold = config.router.single_domain_threshold
        self._fanout_threshold = config.router.fanout_threshold
        self._timeout = config.router.specialist_timeout

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
    def from_config(cls, config: AUAConfig) -> Router:
        return cls(config)

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
                status="alive",
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

        @app.post(
            "/corrections",
            response_model=CorrectionResponse,
            tags=["corrections"],
            summary="Inject a correction into the assertions store",
            status_code=201,
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

        if len(active) >= 2:
            return await self._handle_fanout(req, active, distribution, t0)
        elif top_prob >= self._single_threshold:
            return await self._handle_single(req, top_domain, distribution, t0)
        else:
            return await self._handle_arbiter(req, distribution, t0)

    async def _handle_single(self, req, domain, distribution, t0) -> RouterResponse:
        url = self._field_to_url.get(domain, self._arbiter_url)
        spec = self._config.specialist_for_field(domain)
        model_name = spec.serve_model_name if spec else "default_model"
        response, base_conf = await self._call(
            url, req.query, domain, req.conversation_history, model_name=model_name
        )
        u, conf, n_contra, n_dpo = await self._score(req.query, response, domain, base_conf)
        self._queries_by_mode["single"] = self._queries_by_mode.get("single", 0) + 1
        self._latencies_ms["router"].append((time.time() - t0) * 1000)
        if spec:
            self._requests_per_spec[spec.name] = self._requests_per_spec.get(spec.name, 0) + 1
        self._total_contradictions += n_contra
        self._total_dpo += n_dpo
        latency_ms = round((time.time() - t0) * 1000, 1)
        log.info("single→%s  U=%.3f  C=%.3f  contra=%d  dpo=%d", domain, u, conf, n_contra, n_dpo)
        _audit_query(req, domain, "single", u, conf, latency_ms, n_contra)
        from aua.metrics import get_metrics
        get_metrics().record_query(
            domain=domain,
            routing_mode="single",
            latency_s=latency_ms / 1000,
            u_score=u,
            status="ok",
        )
        return RouterResponse(
            query=req.query,
            routing_mode="single",
            domain_distribution=distribution,
            primary_domain=domain,
            response=response,
            u_score=u,
            confidence=conf,
            contradictions_detected=n_contra,
            dpo_pairs_generated=n_dpo,
            latency_ms=latency_ms,
        )

    async def _handle_fanout(self, req, active_specialists, distribution, t0) -> RouterResponse:
        log.info("fanout → %s", [s.name for s in active_specialists])
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

        if not responses:
            raise HTTPException(503, "All specialists unreachable during fanout")

        if len(responses) >= 2:
            (spec_a, text_a, conf_a), (spec_b, text_b, conf_b) = responses[0], responses[1]
            verdict, winner_field = await self._arbitrate(req.query, spec_a, text_a, spec_b, text_b)
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

        self._queries_by_mode["fanout"] = self._queries_by_mode.get("fanout", 0) + 1
        self._latencies_ms["router"].append((time.time() - t0) * 1000)
        for s in active_specialists:
            self._requests_per_spec[s.name] = self._requests_per_spec.get(s.name, 0) + 1
        self._total_contradictions += n_contra
        self._total_dpo += n_dpo
        _fanout_ms = round((time.time() - t0) * 1000, 1)
        from aua.metrics import get_metrics as _gm
        _gm().record_query(domain=primary_domain, routing_mode="fanout", latency_s=_fanout_ms / 1000, u_score=u, status="ok")
        return RouterResponse(
            query=req.query,
            routing_mode="fanout",
            domain_distribution=distribution,
            primary_domain=primary_domain,
            response=final_text,
            u_score=u,
            confidence=conf,
            contradictions_detected=n_contra,
            dpo_pairs_generated=n_dpo,
            latency_ms=_fanout_ms,
            specialist_responses=spec_responses,
        )

    async def _handle_arbiter(self, req, distribution, t0) -> RouterResponse:
        log.info("arbiter fallback (low confidence)")
        response, base_conf = await self._call(
            self._arbiter_url,
            req.query,
            "general",
            req.conversation_history,
            model_name=self._config.arbiter.serve_model_name,
        )
        u, conf, n_contra, n_dpo = await self._score(req.query, response, "general", base_conf)
        self._queries_by_mode["arbiter"] = self._queries_by_mode.get("arbiter", 0) + 1
        self._latencies_ms["router"].append((time.time() - t0) * 1000)
        self._requests_per_spec["arbiter"] = self._requests_per_spec.get("arbiter", 0) + 1
        self._total_contradictions += n_contra
        self._total_dpo += n_dpo
        _arb_ms = round((time.time() - t0) * 1000, 1)
        from aua.metrics import get_metrics as _gm2
        _gm2().record_query(domain="general", routing_mode="arbiter", latency_s=_arb_ms / 1000, u_score=u, status="ok")
        return RouterResponse(
            query=req.query,
            routing_mode="arbiter",
            domain_distribution=distribution,
            primary_domain="general",
            response=response,
            u_score=u,
            confidence=conf,
            contradictions_detected=n_contra,
            dpo_pairs_generated=n_dpo,
            latency_ms=_arb_ms,
        )

    # ── Specialist call (buffered) ────────────────────────────────────────────

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
        self, query: str, response: str, domain: str, base_conf: float
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
        return task_score.utility, updated_conf, n_contra, n_dpo

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
