#!/usr/bin/env python3
"""
arbiter_session.py — Cross-domain arbitration runner for AUA POC Phase 5.

Fires a single query against two domain specialists simultaneously,
passes both responses to the ArbiterAgent, and exports the full
evidence chain: which model was wrong, which check caught it,
what correction was issued, and the DPO pair created.

Usage:
    python arbiter_session.py \
        --swe     http://localhost:9001 \
        --math    http://localhost:9002 \
        --arbiter http://localhost:9003 \
        --query   "Write a Python function for drug dosage. O(n) complexity required." \
        --export  results/arbitration_evidence.json

    # Run the full canonical cross-domain battery (5 queries)
    python arbiter_session.py --battery --export results/arbitration_evidence.json
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from arbiter import ArbiterAgent, VerdictCase
from assertions_store import AssertionsStore
from contradiction_detector import ContradictionDetector
from config import FIELD_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("arbiter_session")


BATTERY_QUERIES = [
    {
        "id": "xd_01_dosage",
        "query": (
            "Write a Python function calculate_dosage(patient_weight_kg, drug_db) "
            "that looks up a drug dosage from a list-based drug database. "
            "The database is an unsorted list of (min_weight, max_weight, dose_mg) tuples. "
            "The function must scan the full list to find the right weight range. "
            "State the time complexity. Include a worked example."
        ),
        "subject": "drug_dosage_complexity",
        "domain": "software_engineering",
        "swe_expected_complexity": "O(n)",
        "math_expected_complexity": "O(n)",
        "why_cross_domain": "Code correctness (SWE) + complexity verification (Math)",
    },
    {
        "id": "xd_02_gradient_descent",
        "query": (
            "Implement gradient_descent(X, y, lr, n_iter) for linear regression in Python. "
            "X is an (n, k) matrix, y is length-n. "
            "State the time complexity per iteration in terms of n (samples) and k (features). "
            "Include an assert that the loss decreases after 100 iterations."
        ),
        "subject": "gradient_descent_complexity",
        "domain": "software_engineering",
        "swe_expected_complexity": "O(n*k)",
        "math_expected_complexity": "O(n*k)",
        "why_cross_domain": "Matrix multiply implementation (SWE) + per-iteration complexity (Math)",
    },
    {
        "id": "xd_03_dijkstra",
        "query": (
            "Implement Dijkstra's shortest-path algorithm in Python using a min-heap. "
            "The graph has V vertices and E edges represented as an adjacency list. "
            "State and verify the worst-case time complexity."
        ),
        "subject": "dijkstra_complexity",
        "domain": "software_engineering",
        "swe_expected_complexity": "O((V+E) log V)",
        "math_expected_complexity": "O((V+E) log V)",
        "why_cross_domain": "Heap-based graph search (SWE) + complexity derivation (Math)",
    },
    {
        "id": "xd_04_matmul",
        "query": (
            "Implement matrix multiplication for two nxn matrices in Python without numpy. "
            "State the time complexity. "
            "Then prove using the recurrence relation why Strassen's algorithm is faster."
        ),
        "subject": "matmul_strassen",
        "domain": "software_engineering",
        "swe_expected_complexity": "O(n^3)",
        "math_expected_complexity": "O(n^3)",
        "why_cross_domain": "Naive matmul implementation (SWE) + Strassen recurrence proof (Math)",
    },
    {
        "id": "xd_05_merge_sort_proof",
        "query": (
            "Implement merge sort in Python. "
            "Then derive its time complexity using the Master Theorem: "
            "T(n) = 2T(n/2) + O(n). Solve it formally and state the result."
        ),
        "subject": "merge_sort_master_theorem",
        "domain": "software_engineering",
        "swe_expected_complexity": "O(n log n)",
        "math_expected_complexity": "O(n log n)",
        "why_cross_domain": "Merge sort implementation (SWE) + Master Theorem application (Math)",
    },
]


async def call_specialist(client, endpoint, role, query,
                          model_id="swe", max_tokens=600, temperature=0.1):
    system_prompts = {
        "swe":  "You are a specialist software engineering assistant. "
                "Write correct, well-documented Python code. "
                "Always state time and space complexity explicitly.",
        "math": "You are a specialist mathematics and algorithm analysis assistant. "
                "Verify all complexity claims rigorously. "
                "Use formal notation and derive results step by step.",
    }
    system = system_prompts.get(role, system_prompts["swe"])
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    try:
        resp = await client.post(url, json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": query},
            ],
            "max_tokens": max_tokens, "temperature": temperature,
        }, timeout=90.0)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return {"role": role, "endpoint": endpoint, "model_id": model_id,
                "response": text,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_ms": round((time.time() - t0) * 1000, 1), "error": None}
    except Exception as e:
        return {"role": role, "endpoint": endpoint, "response": "",
                "error": str(e), "latency_ms": round((time.time() - t0) * 1000, 1)}


async def call_arbiter_llm(client, endpoint, query, response_swe, response_math,
                           model_id="arbiter"):
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    system = (
        "You are an expert arbitration judge comparing two answers to the same question. "
        "Identify which answer is more correct, more precise, and better reasoned. "
        "If one answer has a wrong complexity claim, flag it explicitly. "
        "Reply in this exact format:\n"
        "VERDICT: A_BETTER | B_BETTER | BOTH_WRONG | INCONCLUSIVE\n"
        "WINNER_CONFIDENCE: 0.XX\n"
        "REASON: one sentence"
    )
    user_prompt = (
        f"QUESTION:\n{query}\n\n"
        f"ANSWER A (SWE specialist):\n{response_swe[:800]}\n\n"
        f"ANSWER B (Math specialist):\n{response_math[:800]}\n\n"
        f"Which answer is more correct?"
    )
    t0 = time.time()
    try:
        resp = await client.post(url, json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": 200, "temperature": 0.1,
        }, timeout=60.0)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        verdict_str, conf, reason = "INCONCLUSIVE", 0.0, text
        for line in text.split("\n"):
            if line.startswith("VERDICT:"):
                verdict_str = line.split(":", 1)[1].strip()
            elif line.startswith("WINNER_CONFIDENCE:"):
                try: conf = float(line.split(":", 1)[1].strip())
                except: pass
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        return {"raw_response": text, "verdict": verdict_str, "confidence": conf,
                "reason": reason, "latency_ms": round((time.time() - t0)*1000,1), "error": None}
    except Exception as e:
        return {"verdict": "INCONCLUSIVE", "error": str(e), "confidence": 0.0,
                "reason": f"Arbiter call failed: {e}"}


async def run_session(args, query_spec):
    query   = query_spec["query"]
    subject = query_spec.get("subject", query[:40].replace(" ", "_"))
    domain  = query_spec.get("domain", "software_engineering")
    qid     = query_spec.get("id", "xd_query")

    log.info(f"\n{'='*60}")
    log.info(f"Query ID: {qid} | Subject: {subject}")
    log.info(f"Query: {query[:100]}...")
    log.info(f"{'='*60}")

    async with httpx.AsyncClient() as client:
        # Step 1: parallel fan-out
        log.info("Step 1: Fan-out to SWE and Math specialists...")
        t0 = time.time()
        swe_r, math_r = await asyncio.gather(
            call_specialist(client, args.swe,  "swe",  query,
                            model_id=getattr(args,'swe_model','swe')),
            call_specialist(client, args.math, "math", query,
                            model_id=getattr(args,'math_model','math')),
        )
        fanout_ms = round((time.time()-t0)*1000, 1)

        log.info(f"SWE  ({swe_r['latency_ms']:.0f}ms): {swe_r['response'][:120]}...")
        log.info(f"Math ({math_r['latency_ms']:.0f}ms): {math_r['response'][:120]}...")

        # Step 2: structured ArbiterAgent checks
        log.info("Step 2: Structured arbiter checks (logical, mathematical, cross-session)...")
        store   = AssertionsStore()
        agent   = ArbiterAgent(assertions_store=store,
                               field_penalty_multipliers={domain: 2.0})
        detector = ContradictionDetector(penalty_multiplier=2.0)

        import re
        def extract_complexity(text):
            m = re.findall(r'O\([^)]+\)', text)
            return m[0] if m else None

        swe_complexity  = extract_complexity(swe_r["response"])
        math_complexity = extract_complexity(math_r["response"])

        verdict = agent.arbitrate(
            subject=subject, domain=domain,
            output_A=swe_r["response"], output_B=math_r["response"],
            field_penalty_multiplier=2.0,
            claimed_complexity_A=swe_complexity,
            claimed_complexity_B=math_complexity,
        )
        log.info(f"Structured verdict: {verdict.case.value} (conf={verdict.arbiter_confidence:.2f})")
        for chk in verdict.checks_run:
            s = '✓' if chk.converged else '·'
            log.info(f"  [{s}] {chk.check_type:<15} winner={chk.winner} "
                     f"conf={chk.confidence:.2f}: {chk.explanation[:80]}")

        # Step 3: LLM empirical check
        log.info("Step 3: LLM arbiter empirical check...")
        arb_llm = await call_arbiter_llm(
            client, args.arbiter, query,
            swe_r["response"], math_r["response"],
            model_id=getattr(args,'arbiter_model','arbiter')
        )
        log.info(f"LLM verdict: {arb_llm['verdict']} "
                 f"(conf={arb_llm['confidence']:.2f}): {arb_llm['reason']}")

        # Step 4: contradiction detection
        log.info("Step 4: Contradiction detection on each response...")
        swe_contra  = detector.check(problem=query, solution=swe_r["response"])
        math_contra = detector.check(problem=query, solution=math_r["response"])
        log.info(f"SWE  contradictions: {len(swe_contra.contradictions)} — {swe_contra.contradictions}")
        log.info(f"Math contradictions: {len(math_contra.contradictions)} — {math_contra.contradictions}")

        # Step 5: DPO pair
        log.info("Step 5: Building DPO pair...")
        dpo_pair = None
        if verdict.case == VerdictCase.CASE_1:
            dpo_pair = {
                "source": "arbiter_cross_domain", "query_id": qid,
                "subject": subject, "domain": domain, "prompt": query,
                "chosen": swe_r["response"], "rejected": math_r["response"],
                "weight": 2.0 * 1.3, "wrong_model": "math",
                "arbiter_case": verdict.case.value,
                "arbiter_confidence": verdict.arbiter_confidence,
                "evidence_summary": verdict.evidence_summary,
            }
            log.info(f"DPO pair created: math was wrong — weight={dpo_pair['weight']:.2f}")
        elif verdict.case == VerdictCase.CASE_2:
            dpo_pair = {
                "source": "arbiter_cross_domain", "query_id": qid,
                "subject": subject, "domain": domain, "prompt": query,
                "chosen": math_r["response"], "rejected": swe_r["response"],
                "weight": 2.0 * 1.3, "wrong_model": "swe",
                "arbiter_case": verdict.case.value,
                "arbiter_confidence": verdict.arbiter_confidence,
                "evidence_summary": verdict.evidence_summary,
            }
            log.info(f"DPO pair created: swe was wrong — weight={dpo_pair['weight']:.2f}")
        elif verdict.case == VerdictCase.CASE_3:
            log.info("Case 3: both wrong — gap bonus activated, no DPO pair")
        else:
            log.info("Case 4: inconclusive — flagged for escalation")

    return {
        "query_id": qid, "subject": subject, "domain": domain,
        "query": query, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fanout_latency_ms": fanout_ms,
        "swe": {
            "endpoint": swe_r["endpoint"], "response": swe_r["response"],
            "claimed_complexity": swe_complexity,
            "n_contradictions": len(swe_contra.contradictions),
            "contradictions": swe_contra.contradictions,
            "latency_ms": swe_r["latency_ms"],
            "prompt_tokens": swe_r.get("prompt_tokens", 0),
            "completion_tokens": swe_r.get("completion_tokens", 0),
            "error": swe_r["error"],
        },
        "math": {
            "endpoint": math_r["endpoint"], "response": math_r["response"],
            "claimed_complexity": math_complexity,
            "n_contradictions": len(math_contra.contradictions),
            "contradictions": math_contra.contradictions,
            "latency_ms": math_r["latency_ms"],
            "prompt_tokens": math_r.get("prompt_tokens", 0),
            "completion_tokens": math_r.get("completion_tokens", 0),
            "error": math_r["error"],
        },
        "structured_arbiter": {
            "case": verdict.case.value,
            "confidence": verdict.arbiter_confidence,
            "correct_A": verdict.correct_A,
            "correct_B": verdict.correct_B,
            "needs_escalation": verdict.needs_escalation,
            "gap_bonus_active": verdict.gap_bonus_active,
            "gap_multiplier": verdict.gap_multiplier,
            "evidence_summary": verdict.evidence_summary,
            "checks": [
                {"check_type": c.check_type, "converged": c.converged,
                 "winner": c.winner, "confidence": c.confidence,
                 "explanation": c.explanation}
                for c in verdict.checks_run
            ],
        },
        "llm_arbiter": arb_llm,
        "dpo_pair": dpo_pair,
        "dpo_created": dpo_pair is not None,
        "why_cross_domain": query_spec.get("why_cross_domain", ""),
        "swe_expected": query_spec.get("swe_expected_complexity", ""),
        "math_expected": query_spec.get("math_expected_complexity", ""),
    }


def print_summary(records):
    print(f"\n{'='*70}")
    print("CROSS-DOMAIN ARBITRATION SESSION SUMMARY")
    print(f"{'='*70}")
    case_counts, dpo_created = {}, 0
    total_swe_c = total_math_c = 0
    for r in records:
        case = r["structured_arbiter"]["case"]
        case_counts[case] = case_counts.get(case, 0) + 1
        if r["dpo_created"]: dpo_created += 1
        total_swe_c  += r["swe"]["n_contradictions"]
        total_math_c += r["math"]["n_contradictions"]

    case_labels = {
        "case_1": "Case 1 (SWE correct, Math wrong)",
        "case_2": "Case 2 (Math correct, SWE wrong)",
        "case_3": "Case 3 (Both wrong — gap bonus)",
        "case_4": "Case 4 (Inconclusive — escalate)",
    }
    for k, label in case_labels.items():
        count = case_counts.get(k, 0)
        print(f"  {label:<40} {count:2d} ({count/len(records)*100:.0f}%) {'█'*count}")

    print(f"\nDPO pairs created: {dpo_created}/{len(records)}")
    print(f"SWE total contradictions:  {total_swe_c}")
    print(f"Math total contradictions: {total_math_c}")
    print(f"\n{'ID':<22} {'Struct':<10} {'LLM':<16} {'SWE_C':<8} {'Math_C':<8} DPO")
    print("-" * 70)
    for r in records:
        print(f"{r['query_id']:<22} {r['structured_arbiter']['case']:<10} "
              f"{r['llm_arbiter'].get('verdict','?')[:15]:<16} "
              f"{r['swe']['n_contradictions']:<8} {r['math']['n_contradictions']:<8} "
              f"{'✓' if r['dpo_created'] else '·'}")

    any_c = total_swe_c > 0 or total_math_c > 0
    caught = any(r["structured_arbiter"]["case"] in ("case_1","case_2") for r in records)
    corrected = any(r["structured_arbiter"].get("correct_A") or
                    r["structured_arbiter"].get("correct_B") for r in records)
    print(f"\nPOC Phase 5 checklist:")
    print(f"  At least one contradiction detected:  {'PASS' if any_c else 'FAIL'}")
    print(f"  Mathematical check caught a mismatch: {'PASS' if caught else 'FAIL'}")
    print(f"  Correction issued by arbiter:         {'PASS' if corrected else 'FAIL'}")
    print(f"  DPO pairs created (source=arbiter):   {'PASS' if dpo_created > 0 else 'FAIL'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--swe",     default="http://localhost:9001")
    p.add_argument("--math",    default="http://localhost:9002")
    p.add_argument("--arbiter", default="http://localhost:9003")
    p.add_argument("--query",   default=None)
    p.add_argument("--subject", default="cross_domain_query")
    p.add_argument("--battery", action="store_true")
    p.add_argument("--export",  default="results/arbitration_evidence.json")
    p.add_argument("--swe-model",     default="swe")
    p.add_argument("--math-model",    default="math")
    p.add_argument("--arbiter-model", default="arbiter")
    return p.parse_args()


async def main():
    args = parse_args()
    if args.battery:
        queries = BATTERY_QUERIES
    elif args.query:
        queries = [{"id": "custom_query", "query": args.query,
                    "subject": args.subject, "domain": "software_engineering"}]
    else:
        log.error("Provide --query TEXT or --battery")
        sys.exit(1)

    log.info(f"Verifying endpoints...")
    async with httpx.AsyncClient() as client:
        for name, ep in [("SWE", args.swe), ("Math", args.math), ("Arbiter", args.arbiter)]:
            try:
                r = await client.get(ep.rstrip("/")+"/v1/models", timeout=10.0)
                ids = [m.get("id") for m in r.json().get("data",[])]
                log.info(f"  {name}: UP — models: {ids}")
            except Exception as e:
                log.warning(f"  {name}: {e}")

    records = []
    for i, q in enumerate(queries):
        log.info(f"\nQuery {i+1}/{len(queries)}: {q['id']}")
        records.append(await run_session(args, q))

    print_summary(records)

    out = Path(args.export)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        def _dc_default(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return dataclasses.asdict(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        json.dump({"session_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "n_queries": len(records),
                   "swe_endpoint": args.swe,
                   "math_endpoint": args.math,
                   "arbiter_endpoint": args.arbiter,
                   "records": records}, f, indent=2, default=_dc_default)
    log.info(f"Evidence chain saved to: {out}")


if __name__ == "__main__":
    asyncio.run(main())
