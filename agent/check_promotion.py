"""
check_promotion.py — Promotion criteria checker and gradual traffic shift
Phase 3 POC

Two modes:

1. CHECK mode: reads canary or shift results, applies the whitepaper's
   promotion criteria, and outputs a YES/NO promotion decision.

   Promotion criteria (from §6.4, Definition 2):
     |U_green - U_blue| > δ(f)  for T(f) consecutive interactions
     where:
       δ(f) = δ₀ / µ(f)  with δ₀ = 0.05
       T(f) from power analysis: T ≥ (z_{α/2} · σ̂ / δ(f))²
       σ̂ ≈ 0.04, z_{0.025} = 1.96
     
     SWE:            δ=0.025, T≥10
     Surgery:        δ=0.005, T≥246
     Creative writing: δ=0.050, T≥2

2. SHIFT mode: runs gradual traffic shift using softmax utility routing,
   ramping GREEN traffic from 5% to up to 95% based on U scores.

   traffic_green = clip(exp(U_green/τ) / (exp(U_green/τ) + exp(U_blue/τ)), 0.05, 0.95)
   τ(SWE) = 0.20  (conservative)
   τ(creative) = 0.50 (aggressive)

Usage:
    # Check promotion from canary results
    python check_promotion.py --check \
        --results results/canary_cycle1.json \
        --field swe

    # Run gradual shift
    python check_promotion.py --shift \
        --blue  http://localhost:9001 \
        --green http://localhost:9011 \
        --field swe \
        --queries benchmark_swe.json \
        --n 100 \
        --output results/shift_cycle1.json

    # Check promotion from shift results
    python check_promotion.py --check \
        --results results/shift_cycle1.json \
        --field swe
"""

import argparse
import asyncio
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from utility_scorer import UtilityScorer
from contradiction_detector import ContradictionDetector
from confidence_updater import ConfidenceUpdater
from config import FIELD_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("check_promotion")


# ── Promotion thresholds (from whitepaper §6.4) ───────────────────────────────

FIELD_THRESHOLDS = {
    # field: (delta, T_min, tau)
    # delta  = U deviation threshold to trigger promotion
    # T_min  = minimum sustained interactions above threshold
    # tau    = softmax temperature for traffic routing
    "surgery":               (0.005, 246, 0.05),
    "aviation":              (0.005, 246, 0.05),
    "law":                   (0.010,  61, 0.10),
    "structural_engineering":(0.0125, 39, 0.10),
    "software_engineering":  (0.025,  10, 0.20),
    "swe":                   (0.025,  10, 0.20),
    "stem_research":         (0.025,  10, 0.20),
    "education":             (0.033,   6, 0.30),
    "art":                   (0.050,   2, 0.50),
    "creative_writing":      (0.050,   2, 0.50),
    "general":               (0.033,   6, 0.30),
}

# σ̂ from simulation (whitepaper §6.4)
SIGMA_HAT  = 0.04
Z_ALPHA_2  = 1.96


def compute_T_min(delta: float) -> int:
    """Power-analysis derived T_min: T ≥ (z_{α/2} · σ̂ / δ)²"""
    return max(2, math.ceil((Z_ALPHA_2 * SIGMA_HAT / delta) ** 2))


def softmax_traffic(u_green: float, u_blue: float, tau: float) -> float:
    """
    Self-regulating traffic split from whitepaper Equation 15.
    traffic_green = clip(exp(U_green/τ) / (exp(U_green/τ) + exp(U_blue/τ)), 0.05, 0.95)
    """
    exp_g = math.exp(u_green / tau)
    exp_b = math.exp(u_blue  / tau)
    raw = exp_g / (exp_g + exp_b)
    return max(0.05, min(0.95, raw))


# ── Check promotion ───────────────────────────────────────────────────────────

def check_promotion(results_path: str, field: str) -> dict:
    """
    Read evaluation/canary/shift results and apply promotion criteria.
    Returns a structured promotion decision.
    """
    with open(results_path) as f:
        data = json.load(f)

    delta, T_min, tau = FIELD_THRESHOLDS.get(field, FIELD_THRESHOLDS["general"])
    T_min_actual = compute_T_min(delta)  # use formula to verify

    # Extract U scores — handle different result formats
    blue_u  = data.get("blue_mean_u",  data.get("mean_u"))
    green_u = data.get("green_mean_u", data.get("candidate_mean_u"))
    n_green = data.get("n_green_calls", data.get("n_queries", 0))

    # For shift results, use sustained_above_threshold count if available
    t_sustained = data.get("t_sustained", n_green)

    if blue_u is None or green_u is None:
        print("ERROR: Could not find blue_mean_u and green_mean_u in results file")
        print(f"Available keys: {list(data.keys())}")
        return {"promote": False, "reason": "missing_data"}

    u_delta = green_u - blue_u
    delta_ok  = abs(u_delta) > delta and u_delta > 0
    t_ok      = t_sustained >= T_min

    promote = delta_ok and t_ok

    decision = {
        "field":          field,
        "blue_mean_u":    blue_u,
        "green_mean_u":   green_u,
        "u_delta":        round(u_delta, 4),
        "threshold_delta": delta,
        "threshold_T":    T_min,
        "t_sustained":    t_sustained,
        "delta_ok":       delta_ok,
        "t_ok":           t_ok,
        "promote":        promote,
        "tau":            tau,
    }

    print(f"\n{'='*50}")
    print(f"PROMOTION DECISION — field: {field}")
    print(f"{'='*50}")
    print(f"BLUE  mean U:      {blue_u:.4f}")
    print(f"GREEN mean U:      {green_u:.4f}")
    print(f"U delta:           {u_delta:+.4f}  (threshold: δ={delta})")
    print(f"Delta criterion:   {'PASS ✓' if delta_ok else 'FAIL ✗'}  ({abs(u_delta):.4f} {'>' if abs(u_delta) > delta else '<='} {delta})")
    print(f"T sustained:       {t_sustained}  (threshold: T≥{T_min})")
    print(f"T criterion:       {'PASS ✓' if t_ok else 'FAIL ✗'}  ({t_sustained} {'≥' if t_ok else '<'} {T_min})")
    print()
    if promote:
        print("✓ PROMOTE GREEN → new BLUE")
        print()
        print("Next steps:")
        print("  1. Stop BLUE server (port 9001)")
        print("  2. Copy GREEN model: cp -r ./models/swe_green_v1 ./models/swe_v2")
        print("  3. Start new BLUE on port 9001 pointing at swe_v2")
        print("  4. Stop GREEN server (port 9011)")
        print("  5. Update baseline: python evaluate.py --endpoint http://localhost:9001")
        print("     --label blue_cycle2 --output results/blue_cycle2_baseline.json")
    else:
        reasons = []
        if not delta_ok:
            reasons.append(f"U delta {u_delta:+.4f} < threshold {delta}")
        if not t_ok:
            reasons.append(f"T sustained {t_sustained} < T_min {T_min}")
        print(f"✗ DO NOT PROMOTE — {'; '.join(reasons)}")
        if u_delta > 0 and not t_ok:
            still_needed = T_min - t_sustained
            print(f"   Need {still_needed} more interactions above threshold")
            print(f"   Continue running shift with more queries")

    return decision


# ── Gradual shift ─────────────────────────────────────────────────────────────

def is_correct(response: str, expected: Optional[str]) -> bool:
    if not expected:
        return True
    resp_lower = response.lower().replace(" ", "")
    exp = expected.lower().replace("o(", "").replace(")", "").replace(" ", "").strip()
    variants = {exp}
    if "logn" in exp:
        variants.update(["nlogn", "n*log(n)", "nlog(n)"])
    if "n^3" in exp or "n3" in exp:
        variants.update(["n^3", "n³", "n**3"])
    if "n^2" in exp or "n2" in exp:
        variants.update(["n^2", "n²", "n**2"])
    return any(v in resp_lower for v in variants)


_MODEL_NAME_CACHE: dict = {}

async def _get_model_name(client: httpx.AsyncClient, endpoint: str) -> str:
    if endpoint not in _MODEL_NAME_CACHE:
        try:
            r = await client.get(endpoint + "/v1/models", timeout=5.0)
            _MODEL_NAME_CACHE[endpoint] = r.json()["data"][0]["id"]
        except Exception:
            _MODEL_NAME_CACHE[endpoint] = "swe"
    return _MODEL_NAME_CACHE[endpoint]


async def call_model(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: str,
) -> tuple[str, float]:
    model_name = await _get_model_name(client, endpoint)
    t0 = time.time()
    resp = await client.post(
        endpoint + "/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a specialist software engineering assistant."},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens":  512,
            "temperature": 0.1,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return text, (time.time() - t0) * 1000


def score(response: str, prompt: str, correct: bool, domain: str) -> tuple[float, float]:
    detector = ContradictionDetector(penalty_multiplier=2.0)
    updater  = ConfidenceUpdater()
    scorer   = UtilityScorer()
    cr = detector.check(problem=prompt, solution=response)
    n_contra = len(cr.contradictions)
    base_conf = 0.75 if correct else 0.45
    conf = updater.update(prior=base_conf, test_signal=base_conf,
                          contradiction_result=cr, field=domain)
    field_cfg = FIELD_CONFIGS.get(domain, FIELD_CONFIGS["general"])
    score_result = scorer.score(
        task_id="promotion_check",
        field_config=field_cfg,
        test_pass_rate=1.0 if correct else 0.0,
        human_baseline_score=0.65,
        contradiction_penalty=float(n_contra) * field_cfg.penalty_multiplier * 0.05,
        problem_novelty=0.1,
    )
    return score_result.utility, conf


DEFAULT_QUERIES = [
    {"id": "s01", "prompt": "Write a Python function binary_search(nums, target). State time complexity.",
     "expected_complexity": "O(log n)", "domain": "software_engineering"},
    {"id": "s02", "prompt": "Write efficient two_sum(nums, target) returning indices. State complexity.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "s03", "prompt": "Implement merge sort in Python. State time and space complexity.",
     "expected_complexity": "O(n log n)", "domain": "software_engineering"},
    {"id": "s04", "prompt": "Implement Kadane's algorithm. State complexity.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "s05", "prompt": "LRU cache with O(1) get/put in Python. Explain data structures.",
     "expected_complexity": "O(1)", "domain": "software_engineering"},
    {"id": "s06", "prompt": "Check valid parentheses in Python. State complexity.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "s07", "prompt": "Implement quicksort. Give average and worst-case complexity.",
     "expected_complexity": "O(n log n)", "domain": "software_engineering"},
    {"id": "s08", "prompt": "Reverse a singly linked list in-place in Python. State complexity.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "s09", "prompt": "Fibonacci with dynamic programming. Compare naive vs DP.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "s10", "prompt": "Matrix multiplication without numpy. State time complexity.",
     "expected_complexity": "O(n^3)", "domain": "software_engineering"},
]


async def run_shift(args) -> dict:
    if args.queries and Path(args.queries).exists():
        with open(args.queries) as f:
            all_queries = json.load(f)
    else:
        all_queries = DEFAULT_QUERIES

    # Expand to N queries
    queries = []
    while len(queries) < args.n:
        queries.extend(all_queries)
    queries = queries[:args.n]
    random.seed(args.seed)
    random.shuffle(queries)

    delta, T_min, tau = FIELD_THRESHOLDS.get(args.field, FIELD_THRESHOLDS["general"])

    log.info(f"=== GRADUAL SHIFT ===")
    log.info(f"BLUE:     {args.blue}")
    log.info(f"GREEN:    {args.green}")
    log.info(f"Field:    {args.field} | δ={delta} | T≥{T_min} | τ={tau}")
    log.info(f"Queries:  {args.n}")
    log.info("")

    records = []
    blue_u_window,  green_u_window  = [], []
    t_sustained = 0
    promotion_triggered = False

    # Running U means
    all_blue_u, all_green_u = [], []

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(queries):
            qid      = q.get("id", f"q{i+1}")
            prompt   = q.get("prompt", q.get("query", ""))
            domain   = q.get("domain", "software_engineering")
            expected = q.get("expected_complexity")

            # Always call BLUE
            blue_resp, blue_lat = await call_model(client, args.blue, prompt)
            blue_correct = is_correct(blue_resp, expected)
            blue_u, blue_conf = score(blue_resp, prompt, blue_correct, domain)
            all_blue_u.append(blue_u)

            # Compute current running means
            cur_blue_mean  = sum(all_blue_u)  / len(all_blue_u)
            cur_green_mean = sum(all_green_u) / len(all_green_u) if all_green_u else None

            # Compute GREEN traffic fraction
            if cur_green_mean is not None:
                traffic_g = softmax_traffic(cur_green_mean, cur_blue_mean, tau)
            else:
                traffic_g = 0.05  # start at 5% before any GREEN data

            # Route to GREEN with computed probability
            route_green = random.random() < traffic_g
            green_u, green_conf, green_correct, green_lat = None, None, None, None

            if route_green:
                try:
                    green_resp, green_lat = await call_model(client, args.green, prompt)
                    green_correct = is_correct(green_resp, expected)
                    green_u, green_conf = score(green_resp, prompt, green_correct, domain)
                    all_green_u.append(green_u)

                    # Update sustained counter
                    if cur_green_mean is not None:
                        u_diff = cur_green_mean - cur_blue_mean
                        if u_diff > delta:
                            t_sustained += 1
                        else:
                            t_sustained = 0  # reset if drops below threshold
                except Exception as e:
                    log.warning(f"[{qid}] GREEN call failed: {e}")

            cur_green_mean_str = f"{sum(all_green_u)/len(all_green_u):.3f}" if all_green_u else "N/A"
            log.info(
                f"[{i+1:3d}/{args.n}] {qid} → {'GREEN' if route_green else 'BLUE '} "
                f"traffic_g={traffic_g:.0%} "
                f"blue_U={cur_blue_mean:.3f} green_U={cur_green_mean_str} "
                f"T_sus={t_sustained}"
            )

            records.append({
                "i":              i+1,
                "id":             qid,
                "routed_to":      "green" if route_green else "blue",
                "traffic_green":  round(traffic_g, 3),
                "blue_u":         round(blue_u, 4),
                "blue_conf":      round(blue_conf, 4),
                "blue_correct":   blue_correct,
                "green_u":        round(green_u, 4) if green_u else None,
                "green_conf":     round(green_conf, 4) if green_conf else None,
                "green_correct":  green_correct,
                "t_sustained":    t_sustained,
            })

            # Check promotion criteria
            if t_sustained >= T_min and not promotion_triggered:
                promotion_triggered = True
                log.info(f"")
                log.info(f"🎉 PROMOTION CRITERIA MET at query {i+1}")
                log.info(f"   T_sustained={t_sustained} ≥ T_min={T_min}")
                log.info(f"   U_green={sum(all_green_u)/len(all_green_u):.4f} > U_blue={cur_blue_mean:.4f}")
                log.info(f"")

    final_blue_u  = sum(all_blue_u)  / len(all_blue_u)  if all_blue_u  else 0
    final_green_u = sum(all_green_u) / len(all_green_u) if all_green_u else None
    u_delta       = (final_green_u - final_blue_u) if final_green_u is not None else None

    summary = {
        "phase":            "gradual_shift",
        "field":            args.field,
        "blue_endpoint":    args.blue,
        "green_endpoint":   args.green,
        "n_queries":        args.n,
        "n_green_calls":    len(all_green_u),
        "n_blue_calls":     len(all_blue_u),
        "blue_mean_u":      round(final_blue_u, 4),
        "green_mean_u":     round(final_green_u, 4) if final_green_u else None,
        "u_delta":          round(u_delta, 4) if u_delta is not None else None,
        "t_sustained":      t_sustained,
        "threshold_delta":  delta,
        "threshold_T":      T_min,
        "tau":              tau,
        "promote":          promotion_triggered,
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "records":          records,
    }

    log.info("")
    log.info("=== GRADUAL SHIFT SUMMARY ===")
    log.info(f"BLUE  mean U:   {final_blue_u:.4f}")
    log.info(f"GREEN mean U:   {f'{final_green_u:.4f}' if final_green_u is not None else 'N/A'}")
    log.info(f"U delta:        {f'{u_delta:+.4f}' if u_delta is not None else 'N/A'}")
    log.info(f"T sustained:    {t_sustained} / {T_min} required")
    log.info(f"Promote:        {'YES ✓' if promotion_triggered else 'NO ✗'}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Results saved to: {args.output}")

    # Auto-run promotion check
    log.info("")
    check_promotion_from_dict(summary, args.field)

    return summary


def check_promotion_from_dict(data: dict, field: str):
    """Run promotion check directly from a results dict (no file I/O)."""
    delta, T_min, tau = FIELD_THRESHOLDS.get(field, FIELD_THRESHOLDS["general"])
    blue_u    = data.get("blue_mean_u", 0)
    green_u   = data.get("green_mean_u")
    t_sus     = data.get("t_sustained", 0)

    if green_u is None:
        print("Cannot check promotion — no GREEN data")
        return

    u_delta   = green_u - blue_u
    delta_ok  = u_delta > delta
    t_ok      = t_sus >= T_min
    promote   = delta_ok and t_ok

    print(f"\n{'='*50}")
    print(f"PROMOTION DECISION — {field}")
    print(f"{'='*50}")
    print(f"U delta: {u_delta:+.4f}  (need >{delta})  {'✓' if delta_ok else '✗'}")
    print(f"T sustained: {t_sus}  (need ≥{T_min})  {'✓' if t_ok else '✗'}")
    print(f"Decision: {'PROMOTE ✓' if promote else 'HOLD ✗'}")


def parse_args():
    p = argparse.ArgumentParser(description="AUA promotion checker and gradual shift runner")
    p.add_argument("--check",  action="store_true", help="Check promotion from saved results")
    p.add_argument("--shift",  action="store_true", help="Run gradual traffic shift")
    p.add_argument("--results",  default=None,                     help="Results JSON for --check")
    p.add_argument("--blue",     default="http://localhost:9001",   help="BLUE endpoint")
    p.add_argument("--green",    default="http://localhost:9011",   help="GREEN endpoint")
    p.add_argument("--field",    default="swe",                     help="Field name")
    p.add_argument("--queries",  default=None,                      help="Benchmark queries JSON")
    p.add_argument("--n",        type=int, default=100,             help="Total queries for shift")
    p.add_argument("--output",   default="results/shift.json",      help="Output path")
    p.add_argument("--seed",     type=int, default=42,              help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.check:
        if not args.results:
            print("--check requires --results path")
            sys.exit(1)
        check_promotion(args.results, args.field)
    elif args.shift:
        asyncio.run(run_shift(args))
    else:
        print("Specify --check or --shift")
        print("Examples:")
        print("  python check_promotion.py --check --results results/canary_cycle1.json --field swe")
        print("  python check_promotion.py --shift --blue http://localhost:9001 --green http://localhost:9011 --field swe --n 100")
        sys.exit(1)
