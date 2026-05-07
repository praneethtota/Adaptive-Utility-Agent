"""
router.py — Micro-Expert Architecture Router
Phase 0 / Stage 1 POC

Routes queries to the correct specialist model based on field classification.
Handles single-domain routing and cross-domain fan-out with arbitration.

Specialist ports:
  8001 — Software Engineering (DeepSeek-Coder)
  8002 — Mathematics (Qwen2.5-Math)
  8003 — Arbiter / General (Phi-3.5-mini or Llama-3.2-3B)

Usage:
  python router.py                    # starts on port 8000
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "write a binary search function in Python"}'
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Local imports (must be run from agent/ directory) ─────────────────────────
from config import FIELD_CONFIGS
from field_classifier import FieldClassifier
from utility_scorer import UtilityScorer
from assertions_store import AssertionsStore
from contradiction_detector import ContradictionDetector
from confidence_updater import ConfidenceUpdater

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("router")

# ── Specialist endpoint map ───────────────────────────────────────────────────
SPECIALISTS: Dict[str, str] = {
    "software_engineering": "http://localhost:8001/v1/chat/completions",
    "mathematics":          "http://localhost:8002/v1/chat/completions",
    "arbiter":              "http://localhost:8003/v1/chat/completions",
    # Fallback mapping — classifier fields → specialist ports
    "stem_research":        "http://localhost:8002/v1/chat/completions",
    "general":              "http://localhost:8003/v1/chat/completions",
    "education":            "http://localhost:8001/v1/chat/completions",
    "general_knowledge":    "http://localhost:8003/v1/chat/completions",
}

# Fields that route to each specialist
SWE_FIELDS   = {"software_engineering", "education", "general_knowledge"}
MATH_FIELDS  = {"mathematics", "stem_research", "physics", "chemistry"}
ARBITER_PORT = "http://localhost:8003/v1/chat/completions"

# Routing thresholds
SINGLE_DOMAIN_CONFIDENCE = 0.75   # above this → send to one specialist only
FANOUT_THRESHOLD          = 0.30   # both domains above this → fan out

# Request timeout per specialist call
SPECIALIST_TIMEOUT = 60.0

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AUA Micro-Expert Router",
    description="Routes queries to specialist LLM models and arbitrates cross-domain conflicts",
    version="0.1.0-poc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ──────────────────────────────────────────────────────────────
classifier      = FieldClassifier()
scorer          = UtilityScorer()
store           = AssertionsStore()
detector        = ContradictionDetector(penalty_multiplier=2.0)
conf_updater    = ConfidenceUpdater()

# Session-level domain confidence (persists across requests in this process)
domain_confidence: Dict[str, float] = {
    "software_engineering": 0.5,
    "mathematics":          0.5,
}

# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    conversation_history: Optional[list] = []
    force_domain: Optional[str] = None     # override routing for testing


class SpecialistResponse(BaseModel):
    domain: str
    response: str
    confidence: float
    u_score: float
    port: str


class RouterResponse(BaseModel):
    query: str
    routing_mode: str                    # "single" | "fanout" | "arbiter"
    domain_distribution: Dict[str, float]
    primary_domain: str
    response: str
    u_score: float
    confidence: float
    contradictions_detected: int
    dpo_pairs_generated: int
    latency_ms: float
    specialist_responses: Optional[list] = None   # populated in fanout mode


# ── Core routing logic ─────────────────────────────────────────────────────────

def get_specialist_url(domain: str) -> str:
    """Map a domain name to its specialist endpoint URL."""
    if domain in SWE_FIELDS:
        return SPECIALISTS["software_engineering"]
    if domain in MATH_FIELDS:
        return SPECIALISTS["mathematics"]
    return SPECIALISTS.get(domain, ARBITER_PORT)


async def call_specialist(
    url: str,
    query: str,
    domain: str,
    history: list = [],
    system_prompt: Optional[str] = None,
) -> Tuple[str, float]:
    """
    Call a vLLM-compatible OpenAI endpoint and return (response_text, confidence).
    The confidence is a raw estimate based on response length and coherence.
    Full confidence scoring happens in score_response().
    """
    messages = []

    # Build system prompt with domain context and active corrections
    if system_prompt is None:
        corrections = store.query(subject=query[:100], domain=domain)
        correction_text = ""
        if corrections:
            correction_text = "\n\nActive corrections from prior sessions:\n" + \
                "\n".join(f"- {m.assertion.claim}" for m in corrections[:5])
        system_prompt = (
            f"You are a specialist in {domain.replace('_', ' ')}. "
            f"Answer precisely and correctly.{correction_text}"
        )

    messages.append({"role": "system", "content": system_prompt})
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": query})

    try:
        async with httpx.AsyncClient(timeout=SPECIALIST_TIMEOUT) as client:
            resp = await client.post(url, json={
                "model":      "default_model",
                "messages":   messages,
                "max_tokens": 1024,
                "temperature": 0.1,
            })
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()

            # Simple confidence proxy: finish_reason stop = full response
            finish_reason = data["choices"][0].get("finish_reason", "")
            base_conf = 0.75 if finish_reason == "stop" else 0.50

            return text, base_conf

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Specialist at {url} is not reachable. Is it running?"
        )
    except Exception as e:
        log.error(f"Specialist call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def score_response(
    query: str,
    response: str,
    domain: str,
    base_confidence: float,
) -> Tuple[float, float, int, int]:
    """
    Score a response using the full AUA pipeline.
    Returns: (u_score, updated_confidence, n_contradictions, n_dpo_pairs)
    """
    # 1. Detect contradictions
    contradiction_result = detector.check(
        problem=query,
        solution=response,
    )
    n_contradictions = len(contradiction_result.contradictions)

    # 2. Update confidence with contradiction penalty
    updated_conf = conf_updater.update(
        prior=base_confidence,
        test_signal=base_confidence,
        contradiction_result=contradiction_result,
        field=domain,
    )

    # 3. Update domain-level confidence EMA
    domain_confidence[domain] = (
        0.8 * domain_confidence.get(domain, 0.5) + 0.2 * updated_conf
    )

    # 4. Score utility via actual scorer API
    field_config = FIELD_CONFIGS.get(domain, FIELD_CONFIGS["general"])
    task_score = scorer.score(
        task_id=f"router_{domain}",
        field_config=field_config,
        test_pass_rate=base_confidence,
        human_baseline_score=0.65,
        contradiction_penalty=contradiction_result.confidence_penalty,
        problem_novelty=0.5,
    )
    u_score = task_score.utility

    # 5. Accumulate DPO pairs if contradiction detected
    n_dpo = 0
    if n_contradictions > 0:
        for c in contradiction_result.contradictions:
            store.add(
                subject=query[:100],
                domain=domain,
                claim=f"Contradiction: {c.description}",
                confidence=updated_conf,
                source="contradiction_detector",
            )
            n_dpo += 1
        log.info(
            f"[{domain}] {n_contradictions} contradiction(s) detected — "
            f"{n_dpo} DPO pair(s) accumulated"
        )

    return u_score, updated_conf, n_contradictions, n_dpo


async def arbitrate(
    query: str,
    swe_response: str,
    math_response: str,
) -> Tuple[str, str]:
    """
    Send both responses to the Arbiter specialist for cross-domain resolution.
    Returns: (arbiter_verdict, winning_domain)
    """
    arbiter_prompt = (
        f"Two specialist models produced different responses to the same query. "
        f"Determine which is correct, or if both are wrong.\n\n"
        f"Query: {query}\n\n"
        f"Response A (Software Engineering specialist):\n{swe_response}\n\n"
        f"Response B (Mathematics specialist):\n{math_response}\n\n"
        f"Evaluate:\n"
        f"1. Does Response A contradict its own premises? (logical check)\n"
        f"2. Are any mathematical or complexity claims provably wrong? (math check)\n"
        f"3. Which response is more correct? State A, B, or BOTH_WRONG.\n"
        f"4. What correction should be applied to the losing response?\n\n"
        f"Format your response as:\n"
        f"VERDICT: [A|B|BOTH_WRONG]\n"
        f"REASON: [brief explanation]\n"
        f"CORRECTION: [what the losing model should learn]"
    )

    arbiter_response, _ = await call_specialist(
        ARBITER_PORT,
        arbiter_prompt,
        "arbiter",
        system_prompt="You are a cross-domain arbitration agent. Be precise and decisive."
    )

    # Parse verdict
    winner = "software_engineering"  # default
    if "VERDICT: B" in arbiter_response or "VERDICT:B" in arbiter_response:
        winner = "mathematics"
    elif "BOTH_WRONG" in arbiter_response:
        winner = "both_wrong"

    log.info(f"[Arbiter] Verdict: {winner}")
    return arbiter_response, winner


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check which specialist servers are reachable."""
    status = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in [
            ("swe",     "http://localhost:8001/v1/models"),
            ("math",    "http://localhost:8002/v1/models"),
            ("arbiter", "http://localhost:8003/v1/models"),
        ]:
            try:
                r = await client.get(url)
                status[name] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
            except Exception:
                status[name] = "unreachable"
    return {"specialists": status, "domain_confidence": domain_confidence}


@app.post("/query", response_model=RouterResponse)
async def route_query(req: QueryRequest):
    """
    Main routing endpoint.

    Routing logic:
      1. Classify query domain
      2. If top domain confidence > 0.75 → single specialist
      3. If SWE + Math both > 0.30 → fan out to both + arbitrate
      4. Otherwise → top specialist with Arbiter as fallback
    """
    t_start = time.time()
    log.info(f"[Router] Query: {req.query[:80]}...")

    # ── Step 1: classify ──────────────────────────────────────────────────────
    if req.force_domain:
        distribution = {req.force_domain: 1.0}
        log.info(f"[Router] Force-routed to: {req.force_domain}")
    else:
        distribution = classifier.classify(req.query, update_history=True)

    log.info(f"[Router] Distribution: {distribution}")

    top_domain = max(distribution, key=distribution.get)
    top_prob   = distribution[top_domain]

    # Detect if this is a cross-domain SWE + Math query
    swe_prob  = sum(distribution.get(f, 0) for f in SWE_FIELDS)
    math_prob = sum(distribution.get(f, 0) for f in MATH_FIELDS)
    is_cross_domain = swe_prob > FANOUT_THRESHOLD and math_prob > FANOUT_THRESHOLD

    # ── Step 2: route ─────────────────────────────────────────────────────────
    if is_cross_domain:
        # Fan out to both specialists, then arbitrate
        log.info(f"[Router] Cross-domain fan-out (SWE={swe_prob:.2f}, Math={math_prob:.2f})")

        results = await asyncio.gather(
            call_specialist(
                SPECIALISTS["software_engineering"],
                req.query, "software_engineering",
                req.conversation_history
            ),
            call_specialist(
                SPECIALISTS["mathematics"],
                req.query, "mathematics",
                req.conversation_history
            )
        )
        swe_text, swe_conf   = results[0]
        math_text, math_conf = results[1]

        # Arbitrate
        arbiter_verdict, winner = await arbitrate(req.query, swe_text, math_text)

        # Use winning response (or arbiter's synthesis on BOTH_WRONG)
        if winner == "mathematics":
            final_response = math_text
            primary_domain = "mathematics"
            base_conf = math_conf
        elif winner == "both_wrong":
            final_response = arbiter_verdict
            primary_domain = "arbiter"
            base_conf = 0.40  # both were wrong — lower confidence
        else:
            final_response = swe_text
            primary_domain = "software_engineering"
            base_conf = swe_conf

        # Score the final response
        u_score, final_conf, n_contra, n_dpo = await score_response(
            req.query, final_response, primary_domain, base_conf
        )

        # Also score the losing response to accumulate DPO signal
        losing_domain   = "mathematics" if winner != "mathematics" else "software_engineering"
        losing_response = math_text     if winner != "mathematics" else swe_text
        _, _, n_contra_losing, n_dpo_losing = await score_response(
            req.query, losing_response, losing_domain, 0.30
        )

        routing_mode = "fanout"
        specialist_responses = [
            {"domain": "software_engineering", "response": swe_text[:200] + "..."},
            {"domain": "mathematics",          "response": math_text[:200] + "..."},
            {"domain": "arbiter_verdict",      "verdict": winner,
             "arbiter_response": arbiter_verdict[:200] + "..."},
        ]
        total_contra = n_contra + n_contra_losing
        total_dpo    = n_dpo + n_dpo_losing

    elif top_prob >= SINGLE_DOMAIN_CONFIDENCE:
        # Single specialist — high confidence routing
        specialist_url = get_specialist_url(top_domain)
        log.info(f"[Router] Single specialist: {top_domain} ({top_prob:.2f}) → {specialist_url}")

        final_response, base_conf = await call_specialist(
            specialist_url, req.query, top_domain, req.conversation_history
        )
        primary_domain = top_domain
        u_score, final_conf, total_contra, total_dpo = await score_response(
            req.query, final_response, top_domain, base_conf
        )
        routing_mode         = "single"
        specialist_responses = None

    else:
        # Low confidence in any domain — route to Arbiter as general handler
        log.info(f"[Router] Low confidence ({top_prob:.2f}) — routing to Arbiter")
        final_response, base_conf = await call_specialist(
            ARBITER_PORT, req.query, "general", req.conversation_history
        )
        primary_domain = "general"
        u_score, final_conf, total_contra, total_dpo = await score_response(
            req.query, final_response, "general", base_conf
        )
        routing_mode         = "arbiter"
        specialist_responses = None

    latency_ms = (time.time() - t_start) * 1000
    log.info(
        f"[Router] Done — mode={routing_mode} domain={primary_domain} "
        f"U={u_score:.3f} C={final_conf:.3f} "
        f"contra={total_contra} dpo={total_dpo} "
        f"latency={latency_ms:.0f}ms"
    )

    return RouterResponse(
        query=req.query,
        routing_mode=routing_mode,
        domain_distribution=distribution,
        primary_domain=primary_domain,
        response=final_response,
        u_score=u_score,
        confidence=final_conf,
        contradictions_detected=total_contra,
        dpo_pairs_generated=total_dpo,
        latency_ms=latency_ms,
        specialist_responses=specialist_responses,
    )


@app.get("/stats")
async def stats():
    """Return current domain confidence, DPO pair count, and assertion store size."""
    summary = store.summary()
    return {
        "domain_confidence": domain_confidence,
        "assertions_count":  summary.get("total", 0),
        "dpo_pairs_count":   summary.get("by_source", {}).get("contradiction_detector", 0),
    }


@app.post("/reset")
async def reset_session():
    """Reset conversation history and domain confidence for a new session."""
    classifier.reset_history()
    domain_confidence["software_engineering"] = 0.5
    domain_confidence["mathematics"] = 0.5
    return {"status": "reset", "domain_confidence": domain_confidence}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n=== AUA Micro-Expert Router ===")
    print("Specialist endpoints:")
    print("  SWE     → http://localhost:8001")
    print("  Math    → http://localhost:8002")
    print("  Arbiter → http://localhost:8003")
    print("\nRouter API:")
    print("  POST http://localhost:8000/query")
    print("  GET  http://localhost:8000/health")
    print("  GET  http://localhost:8000/stats")
    print("  GET  http://localhost:8000/docs  (Swagger UI)\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
