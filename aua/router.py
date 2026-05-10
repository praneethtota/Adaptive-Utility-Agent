"""
aua/router.py — Micro-Expert Architecture Router

Config-driven: all specialist endpoints, thresholds, and ports come from
AUAConfig (loaded from aua_config.yaml). No hardcoded values.

Usage (programmatic):
    from aua import Router
    from aua.config import load_config

    config = load_config("aua_config.yaml")
    router = Router.from_config(config)
    # router.app is the FastAPI app — mount or run directly

Usage (CLI — preferred):
    aua serve                    # reads aua_config.yaml from cwd
    aua serve --config /path/to/aua_config.yaml
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from statistics import median, quantiles
from typing import Counter as CounterT, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aua.config import AUAConfig, FieldConfig, FIELD_CONFIGS
from aua.field_classifier import FieldClassifier
from aua.utility_scorer import UtilityScorer
from aua.assertions_store import AssertionsStore
from aua.contradiction_detector import ContradictionDetector
from aua.confidence_updater import ConfidenceUpdater

log = logging.getLogger("aua.router")


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    conversation_history: Optional[List[dict]] = []
    force_domain: Optional[str] = None     # override routing for testing


class RouterResponse(BaseModel):
    query: str
    routing_mode: str                          # "single" | "fanout" | "arbiter"
    domain_distribution: Dict[str, float]
    primary_domain: str
    response: str
    u_score: float
    confidence: float
    contradictions_detected: int
    dpo_pairs_generated: int
    latency_ms: float
    specialist_responses: Optional[List[dict]] = None


# ── Router ────────────────────────────────────────────────────────────────────

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
        self._config     = config
        self._classifier = FieldClassifier()
        self._scorer     = UtilityScorer()
        self._store      = AssertionsStore()
        self._detector   = ContradictionDetector(penalty_multiplier=2.0)
        self._conf       = ConfidenceUpdater()

        # Domain-level confidence EMA (persists across requests in this process)
        self._domain_confidence: Dict[str, float] = {
            s.field: 0.5 for s in config.specialists
        }

        # Build field → endpoint lookup from config
        # Primary mapping: field name → specialist endpoint URL
        self._field_to_url: Dict[str, str] = {
            s.field: s.endpoint for s in config.specialists
        }
        self._arbiter_url = config.arbiter.endpoint

        self._single_threshold = config.router.single_domain_threshold
        self._fanout_threshold = config.router.fanout_threshold
        self._timeout = config.router.specialist_timeout

        # ── Telemetry tracking ──────────────────────────────────────────
        self._start_time: float = time.time()
        self._queries_by_mode: Dict[str, int] = {"single": 0, "fanout": 0, "arbiter": 0}
        self._latencies_ms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._requests_per_spec: Dict[str, int] = {s.name: 0 for s in config.specialists}
        self._requests_per_spec["arbiter"] = 0
        self._total_contradictions: int = 0
        self._total_dpo: int = 0
        # Arbiter verdict distribution (case_1..case_4) tracked via AssertionsStore
        # but we also keep a local counter for the /status endpoint
        self._verdict_counts: Dict[str, int] = {
            "case_1": 0, "case_2": 0, "case_3": 0, "case_4": 0
        }

        self.app = self._build_app()
        log.info(
            "Router initialised — %d specialist(s), arbiter on port %d",
            len(config.specialists), config.arbiter.port,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: AUAConfig) -> "Router":
        """Build a Router from a loaded AUAConfig."""
        return cls(config)

    # ── Public query API ──────────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        session_id: str = "default",
        conversation_history: Optional[List[dict]] = None,
        force_domain: Optional[str] = None,
    ) -> RouterResponse:
        """
        Route a query through the specialist graph and return a scored response.
        This is the primary programmatic API.
        """
        req = QueryRequest(
            query=query,
            session_id=session_id,
            conversation_history=conversation_history or [],
            force_domain=force_domain,
        )
        return await self._handle(req)

    # ── FastAPI app ───────────────────────────────────────────────────────────

    def _build_app(self) -> FastAPI:
        app = FastAPI(
            title="AUA Micro-Expert Router",
            description=(
                "Routes queries to specialist LLM models and arbitrates "
                "cross-domain conflicts. Config-driven via aua_config.yaml."
            ),
            version="0.5.0",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/health")
        async def health():
            return await self._health()

        @app.get("/stats")
        async def stats():
            return self._stats()

        @app.post("/query", response_model=RouterResponse)
        async def route(req: QueryRequest):
            return await self._handle(req)

        @app.get("/status")
        async def full_status():
            return await self._full_status()

        @app.post("/reset")
        async def reset():
            self._classifier.reset_history()
            for field in self._domain_confidence:
                self._domain_confidence[field] = 0.5
            return {"status": "reset", "domain_confidence": self._domain_confidence}

        return app

    # ── Health / stats ────────────────────────────────────────────────────────

    async def _health(self) -> dict:
        """Ping each specialist and the arbiter, return reachability map."""
        status: Dict[str, str] = {}
        checks = [
            (s.name, s.models_url) for s in self._config.specialists
        ] + [("arbiter", self._config.arbiter.models_url)]

        async with httpx.AsyncClient(timeout=3.0) as client:
            for name, url in checks:
                try:
                    r = await client.get(url)
                    status[name] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
                except Exception:
                    status[name] = "unreachable"
        return {"specialists": status, "domain_confidence": self._domain_confidence}


    async def _full_status(self) -> dict:
        """Comprehensive status for aua status dashboard."""
        import os, subprocess

        # Health check
        health = await self._health()

        # Uptime
        uptime_s = time.time() - self._start_time

        # Latency percentiles per specialist
        latency_stats: Dict[str, dict] = {}
        for name, dq in self._latencies_ms.items():
            vals = list(dq)
            if vals:
                sorted_vals = sorted(vals)
                n = len(sorted_vals)
                p50 = sorted_vals[n // 2]
                p95 = sorted_vals[int(n * 0.95)]
                latency_stats[name] = {
                    "p50_ms": round(p50, 1),
                    "p95_ms": round(p95, 1),
                    "last_ms": round(vals[-1], 1),
                    "samples": n,
                }
            else:
                latency_stats[name] = {"p50_ms": None, "p95_ms": None, "last_ms": None, "samples": 0}

        # U score history per domain
        utility: Dict[str, dict] = {}
        for domain, state in self._scorer.domain_states.items():
            history = [s.utility for s in self._scorer.history if s.field == domain]
            utility[domain] = {
                "mean_u":     round(sum(history) / len(history), 4) if history else None,
                "last_u":     round(history[-1], 4) if history else None,
                "queries":    len(history),
                "confidence": round(state.confidence, 4),
            }

        # Routing stats
        total_q = sum(self._queries_by_mode.values())
        routing = {
            "total_queries": total_q,
            "by_mode": dict(self._queries_by_mode),
        }

        # Corrections
        store_summary = self._store.summary()
        corrections = {
            "total_contradictions": self._total_contradictions,
            "dpo_pairs":            self._total_dpo,
            "assertions_stored":    store_summary.get("total", 0),
            "contradiction_rate":   round(
                self._total_contradictions / total_q, 4
            ) if total_q > 0 else 0.0,
        }

        # Memory info — hardware-agnostic via _detect_hardware()
        memory: Dict[str, str] = {}
        try:
            from aua.doctor import _detect_hardware
            hw = _detect_hardware()
            if hw.kind == "nvidia":
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3
                )
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(", ")]
                    if len(parts) >= 3:
                        memory[f"gpu{parts[0]}"] = f"{parts[1]} / {parts[2]} MiB"
            elif hw.kind == "amd_rocm":
                for dev in hw.devices:
                    mib = dev.get("vram_mib")
                    memory[f"gpu{dev['index']}"] = f"{mib} MiB (total)" if mib else "AMD GPU"
            elif hw.kind == "apple_silicon":
                for dev in hw.devices:
                    mib = dev.get("vram_mib")
                    label = f"{mib} MiB unified" if mib else dev.get("name", "Apple GPU")
                    memory[f"gpu{dev['index']}"] = label
            else:
                # CPU / Ollama — show system RAM
                ram = hw.system_ram_mib
                memory["system"] = f"{ram // 1024} GiB RAM" if ram else "CPU / Ollama"
        except Exception:
            memory = {"system": "unavailable"}

        return {
            "version":   self._config.version,
            "backend":   self._config.backend,
            "uptime_s":  round(uptime_s, 1),
            "health":    health["specialists"],
            "latency":   latency_stats,
            "utility":   utility,
            "routing":   routing,
            "corrections": corrections,
            "arbiter_verdicts": dict(self._verdict_counts),
            "memory":    memory,
        }

    def _stats(self) -> dict:
        summary = self._store.summary()
        return {
            "domain_confidence":  self._domain_confidence,
            "assertions_count":   summary.get("total", 0),
            "dpo_pairs_count":    summary.get("by_source", {}).get("contradiction_detector", 0),
        }

    # ── Core routing ──────────────────────────────────────────────────────────

    async def _handle(self, req: QueryRequest) -> RouterResponse:
        t0 = time.time()
        log.info("Query: %.80s", req.query)

        # ── 1. Classify ──────────────────────────────────────────────────────
        if req.force_domain:
            distribution = {req.force_domain: 1.0}
        else:
            distribution = self._classifier.classify(req.query, update_history=True)

        log.debug("Distribution: %s", distribution)

        # ── 2. Decide routing mode ────────────────────────────────────────────
        top_domain = max(distribution, key=distribution.get)
        top_prob   = distribution[top_domain]

        # Check if multiple specialists are active above fanout threshold
        active = [
            s for s in self._config.specialists
            if distribution.get(s.field, 0) >= self._fanout_threshold
        ]
        is_fanout = len(active) >= 2

        # ── 3. Route ──────────────────────────────────────────────────────────
        if is_fanout:
            return await self._handle_fanout(req, active, distribution, t0)
        elif top_prob >= self._single_threshold:
            return await self._handle_single(req, top_domain, distribution, t0)
        else:
            return await self._handle_arbiter(req, distribution, t0)

    async def _handle_single(
        self,
        req: QueryRequest,
        domain: str,
        distribution: Dict[str, float],
        t0: float,
    ) -> RouterResponse:
        url = self._field_to_url.get(domain, self._arbiter_url)
        spec = self._config.specialist_for_field(domain)
        model_name = spec.serve_model_name if spec else "default_model"
        response, base_conf = await self._call(url, req.query, domain,
                                               req.conversation_history,
                                               model_name=model_name)
        u, conf, n_contra, n_dpo = await self._score(req.query, response, domain, base_conf)
        self._queries_by_mode["single"] = self._queries_by_mode.get("single", 0) + 1
        self._latencies_ms["router"].append((time.time() - t0) * 1000)
        spec_obj = self._config.specialist_for_field(domain)
        if spec_obj:
            self._requests_per_spec[spec_obj.name] = self._requests_per_spec.get(spec_obj.name, 0) + 1
        self._total_contradictions += n_contra
        self._total_dpo += n_dpo
        log.info("single→%s  U=%.3f  C=%.3f  contra=%d  dpo=%d", domain, u, conf, n_contra, n_dpo)
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
            latency_ms=(time.time() - t0) * 1000,
        )

    async def _handle_fanout(
        self,
        req: QueryRequest,
        active_specialists,
        distribution: Dict[str, float],
        t0: float,
    ) -> RouterResponse:
        log.info("fanout → %s", [s.name for s in active_specialists])

        # Call all active specialists in parallel
        calls = [
            self._call(s.endpoint, req.query, s.field, req.conversation_history,
                       model_name=s.serve_model_name)
            for s in active_specialists
        ]
        results = await asyncio.gather(*calls, return_exceptions=True)

        responses = []
        for spec, result in zip(active_specialists, results):
            if isinstance(result, Exception):
                log.warning("Specialist %s failed: %s", spec.name, result)
            else:
                text, conf = result
                responses.append((spec, text, conf))

        if not responses:
            raise HTTPException(503, "All specialists unreachable during fanout")

        # Arbitrate if we got 2+ responses
        if len(responses) >= 2:
            (spec_a, text_a, conf_a), (spec_b, text_b, conf_b) = responses[0], responses[1]
            verdict, winner_field = await self._arbitrate(
                req.query, spec_a, text_a, spec_b, text_b
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

            # Score winner + accumulate DPO from loser
            u, conf, n_contra, n_dpo = await self._score(
                req.query, final_text, primary_domain, final_conf
            )
            _, _, nc2, nd2 = await self._score(
                req.query, losing_text, losing_domain, 0.30
            )
            n_contra += nc2; n_dpo += nd2

            specialist_responses = [
                {"domain": s.field, "response": t[:200] + "..."}
                for s, t, _ in responses
            ] + [{"domain": "arbiter_verdict", "winner": winner_field}]
        else:
            # Only one response came back — treat as single
            spec, text, base_conf = responses[0]
            primary_domain = spec.field
            u, conf, n_contra, n_dpo = await self._score(
                req.query, text, primary_domain, base_conf
            )
            final_text = text
            specialist_responses = [{"domain": spec.field, "response": text[:200]}]

        self._queries_by_mode["fanout"] = self._queries_by_mode.get("fanout", 0) + 1
        self._latencies_ms["router"].append((time.time() - t0) * 1000)
        for s in active_specialists:
            self._requests_per_spec[s.name] = self._requests_per_spec.get(s.name, 0) + 1
        self._total_contradictions += n_contra
        self._total_dpo += n_dpo
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
            latency_ms=(time.time() - t0) * 1000,
            specialist_responses=specialist_responses,
        )

    async def _handle_arbiter(
        self,
        req: QueryRequest,
        distribution: Dict[str, float],
        t0: float,
    ) -> RouterResponse:
        log.info("arbiter fallback (low confidence)")
        response, base_conf = await self._call(
            self._arbiter_url, req.query, "general", req.conversation_history,
            model_name=self._config.arbiter.serve_model_name,
        )
        u, conf, n_contra, n_dpo = await self._score(req.query, response, "general", base_conf)
        self._queries_by_mode["arbiter"] = self._queries_by_mode.get("arbiter", 0) + 1
        self._latencies_ms["router"].append((time.time() - t0) * 1000)
        self._requests_per_spec["arbiter"] = self._requests_per_spec.get("arbiter", 0) + 1
        self._total_contradictions += n_contra
        self._total_dpo += n_dpo
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
            latency_ms=(time.time() - t0) * 1000,
        )

    # ── Specialist call ───────────────────────────────────────────────────────

    async def _call(
        self,
        url: str,
        query: str,
        domain: str,
        history: Optional[List[dict]] = None,
        system_prompt: Optional[str] = None,
        model_name: str = "default_model",
    ) -> Tuple[str, float]:
        """Call a vLLM or Ollama-compatible OpenAI endpoint, return (text, base_confidence)."""
        if system_prompt is None:
            corrections = self._store.query(subject=query[:100], domain=domain)
            injection = ""
            if corrections:
                injection = "\n\nActive corrections:\n" + \
                    "\n".join(f"- {m.assertion.claim}" for m in corrections[:5])
            system_prompt = (
                f"You are a specialist in {domain.replace('_', ' ')}. "
                f"Answer precisely and correctly.{injection}"
            )

        messages = [{"role": "system", "content": system_prompt}]
        for h in (history or []):
            messages.append(h)
        messages.append({"role": "user", "content": query})

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(url, json={
                    "model":       model_name,
                    "messages":    messages,
                    "max_tokens":  1024,
                    "temperature": 0.1,
                })
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
    ) -> Tuple[float, float, int, int]:
        """Full AUA scoring pipeline. Returns (u, conf, n_contradictions, n_dpo)."""
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

    async def _arbitrate(self, query, spec_a, text_a, spec_b, text_b) -> Tuple[str, str]:
        prompt = (
            f"Two specialist models produced different responses.\n\n"
            f"Query: {query}\n\n"
            f"Response A ({spec_a.field}):\n{text_a}\n\n"
            f"Response B ({spec_b.field}):\n{text_b}\n\n"
            f"Which is correct? Reply:\n"
            f"VERDICT: [A|B|BOTH_WRONG]\n"
            f"REASON: [brief]\n"
            f"CORRECTION: [what the losing model should learn]"
        )
        verdict_text, _ = await self._call(
            self._arbiter_url, prompt, "arbiter",
            system_prompt="You are a cross-domain arbitration agent. Be concise and decisive.",
            model_name=self._config.arbiter.serve_model_name,
        )
        if "VERDICT: B" in verdict_text:
            winner = spec_b.field
        elif "BOTH_WRONG" in verdict_text:
            winner = "both_wrong"
        else:
            winner = spec_a.field
        log.info("Arbiter verdict: %s", winner)
        return verdict_text, winner
