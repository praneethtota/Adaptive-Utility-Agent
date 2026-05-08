"""
canary.py — Blue-green canary traffic split runner
Phase 3 POC

Sends a configurable fraction of queries to GREEN (new model) and the rest
to BLUE (current model), recording per-query U scores for both.

The canary phase is the first validation gate before gradual traffic shift.
A 5% canary over 50 queries gives 2–3 GREEN responses — enough to confirm
the model responds coherently, not enough to commit to promotion.

Usage:
    # Start canary: 5% to GREEN, 95% to BLUE, 50 queries
    python canary.py \
        --blue  http://localhost:9001 \
        --green http://localhost:9011 \
        --traffic-green 0.05 \
        --n 50 \
        --queries benchmark_swe.json \
        --output results/canary_cycle1.json

    # After canary: check if GREEN is ready for gradual shift
    python canary.py --check --results results/canary_cycle1.json
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
log = logging.getLogger("canary")


# ── Default benchmark (same as evaluate.py) ───────────────────────────────────

DEFAULT_QUERIES = [
    {"id": "c01", "prompt": "Write a Python function binary_search(nums, target). State time complexity.",
     "expected_complexity": "O(log n)", "domain": "software_engineering"},
    {"id": "c02", "prompt": "Write an efficient two_sum(nums, target) returning indices. State complexity.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "c03", "prompt": "Implement merge sort in Python with time and space complexity.",
     "expected_complexity": "O(n log n)", "domain": "software_engineering"},
    {"id": "c04", "prompt": "Write Python to check valid parentheses. State time and space complexity.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "c05", "prompt": "Implement Kadane's algorithm for max subarray sum. State complexity.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "c06", "prompt": "Write Python for LRU cache with O(1) get/put. Explain data structures.",
     "expected_complexity": "O(1)", "domain": "software_engineering"},
    {"id": "c07", "prompt": "Implement quicksort in Python. Give average and worst-case complexity.",
     "expected_complexity": "O(n log n)", "domain": "software_engineering"},
    {"id": "c08", "prompt": "Write Python to reverse a singly linked list in-place. State complexity.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "c09", "prompt": "Fibonacci with dynamic programming in Python. Compare with naive recursion.",
     "expected_complexity": "O(n)", "domain": "software_engineering"},
    {"id": "c10", "prompt": "Implement matrix multiplication without numpy. State time complexity.",
     "expected_complexity": "O(n^3)", "domain": "software_engineering"},
]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def is_correct(response: str, expected: Optional[str]) -> bool:
    if not expected:
        return True
    resp_lower = response.lower().replace(" ", "")
    exp = expected.lower().replace("o(", "").replace(")", "").replace(" ", "").strip()
    variants = {exp}
    if "logn" in exp or "nlogn" in exp:
        variants.update(["nlogn", "n*log(n)", "nlog(n)"])
    if "n^3" in exp or "n3" in exp:
        variants.update(["n^3", "n³", "n**3", "cubic"])
    if "n^2" in exp or "n2" in exp:
        variants.update(["n^2", "n²", "n**2", "quadratic"])
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


async def call(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: str,
) -> tuple[str, float]:
    """Returns (response_text, latency_ms)."""
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


def score_response(response: str, prompt: str, correct: bool, domain: str) -> tuple[float, float]:
    """Returns (u_score, confidence)."""
    detector = ContradictionDetector(penalty_multiplier=2.0)
    updater  = ConfidenceUpdater()
    scorer   = UtilityScorer()

    contradiction_result = detector.check(problem=prompt, solution=response)
    n_contra = len(contradiction_result.contradictions)
    base_conf = 0.75 if correct else 0.45
    conf = updater.update(
        prior=base_conf,
        test_signal=base_conf,
        contradiction_result=contradiction_result,
        field=domain,
    )
    field_cfg = FIELD_CONFIGS.get(domain, FIELD_CONFIGS["general"])
    score_result = scorer.score(
        task_id="canary",
        field_config=field_cfg,
        test_pass_rate=1.0 if correct else 0.0,
        human_baseline_score=0.65,
        contradiction_penalty=float(n_contra) * field_cfg.penalty_multiplier * 0.05,
        problem_novelty=0.1,
    )
    return score_result.utility, conf


# ── Main canary loop ──────────────────────────────────────────────────────────

async def run_canary(args) -> dict:
    # Load queries
    if args.queries and Path(args.queries).exists():
        with open(args.queries) as f:
            all_queries = json.load(f)
    else:
        all_queries = DEFAULT_QUERIES

    # Sample N queries (cycle through if N > len)
    queries = []
    while len(queries) < args.n:
        queries.extend(all_queries)
    queries = queries[:args.n]
    random.seed(args.seed)
    random.shuffle(queries)

    log.info(f"=== CANARY PHASE ===")
    log.info(f"BLUE:          {args.blue}")
    log.info(f"GREEN:         {args.green}")
    log.info(f"Traffic GREEN: {args.traffic_green:.0%}")
    log.info(f"Queries:       {args.n}")
    log.info(f"Expected GREEN calls: ~{int(args.n * args.traffic_green)}")
    log.info("")

    records = []
    blue_u_scores, green_u_scores = [], []
    blue_brier_data, green_brier_data = [], []

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(queries):
            qid    = q.get("id", f"q{i+1}")
            prompt = q.get("prompt", q.get("query", ""))
            domain = q.get("domain", "software_engineering")
            expected = q.get("expected_complexity")

            # Always call BLUE
            try:
                blue_resp, blue_lat = await call(client, args.blue, prompt)
                blue_correct = is_correct(blue_resp, expected)
                blue_u, blue_conf = score_response(blue_resp, prompt, blue_correct, domain)
                blue_u_scores.append(blue_u)
                blue_brier_data.append((blue_conf, blue_correct))
            except Exception as e:
                log.error(f"[{qid}] BLUE call failed: {e}")
                continue

            # Stochastically call GREEN
            route_to_green = random.random() < args.traffic_green
            green_u, green_conf, green_correct, green_lat = None, None, None, None

            if route_to_green:
                try:
                    green_resp, green_lat = await call(client, args.green, prompt)
                    green_correct = is_correct(green_resp, expected)
                    green_u, green_conf = score_response(green_resp, prompt, green_correct, domain)
                    green_u_scores.append(green_u)
                    green_brier_data.append((green_conf, green_correct))
                    log.info(
                        f"[{i+1}/{args.n}] {qid} → GREEN "
                        f"U={green_u:.3f} C={green_conf:.3f} "
                        f"{'✓' if green_correct else '✗'} {green_lat:.0f}ms"
                    )
                except Exception as e:
                    log.error(f"[{qid}] GREEN call failed: {e}")
                    route_to_green = False
            else:
                log.info(
                    f"[{i+1}/{args.n}] {qid} → BLUE  "
                    f"U={blue_u:.3f} C={blue_conf:.3f} "
                    f"{'✓' if blue_correct else '✗'} {blue_lat:.0f}ms"
                )

            records.append({
                "id":             qid,
                "routed_to":      "green" if route_to_green else "blue",
                "blue_u":         round(blue_u, 4),
                "blue_conf":      round(blue_conf, 4),
                "blue_correct":   blue_correct,
                "blue_latency":   round(blue_lat, 1),
                "green_u":        round(green_u, 4) if green_u is not None else None,
                "green_conf":     round(green_conf, 4) if green_conf is not None else None,
                "green_correct":  green_correct,
                "green_latency":  round(green_lat, 1) if green_lat is not None else None,
            })

    # Aggregate
    def brier(data):
        if not data:
            return None
        return sum((c - (1.0 if ok else 0.0))**2 for c, ok in data) / len(data)

    blue_mean_u  = sum(blue_u_scores) / len(blue_u_scores) if blue_u_scores else 0
    green_mean_u = sum(green_u_scores) / len(green_u_scores) if green_u_scores else None
    blue_brier   = brier(blue_brier_data)
    green_brier  = brier(green_brier_data)

    u_delta = (green_mean_u - blue_mean_u) if green_mean_u is not None else None

    summary = {
        "phase":            "canary",
        "blue_endpoint":    args.blue,
        "green_endpoint":   args.green,
        "traffic_green":    args.traffic_green,
        "n_queries":        args.n,
        "n_green_calls":    len(green_u_scores),
        "n_blue_calls":     len(blue_u_scores),
        "blue_mean_u":      round(blue_mean_u, 4),
        "green_mean_u":     round(green_mean_u, 4) if green_mean_u else None,
        "u_delta":          round(u_delta, 4) if u_delta is not None else None,
        "blue_brier":       round(blue_brier, 4) if blue_brier else None,
        "green_brier":      round(green_brier, 4) if green_brier else None,
        "canary_passed":    (u_delta is not None and u_delta > 0),
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "records":          records,
    }

    log.info("")
    log.info("=== CANARY SUMMARY ===")
    log.info(f"BLUE  mean U:  {blue_mean_u:.4f} ({len(blue_u_scores)} calls)")
    if green_mean_u is not None:
        log.info(f"GREEN mean U:  {green_mean_u:.4f} ({len(green_u_scores)} calls)")
        log.info(f"U delta:       {u_delta:+.4f}")
        log.info(f"Canary passed: {'YES ✓' if u_delta > 0 else 'NO ✗'}")
    else:
        log.info("GREEN had no successful calls")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Results saved to: {args.output}")

    return summary


def check_canary(results_path: str):
    """Print canary pass/fail assessment from saved results."""
    with open(results_path) as f:
        r = json.load(f)

    print("\n=== CANARY CHECK ===")
    print(f"BLUE mean U:   {r.get('blue_mean_u', 'N/A')}")
    print(f"GREEN mean U:  {r.get('green_mean_u', 'N/A')}")
    print(f"U delta:       {r.get('u_delta', 'N/A')}")
    print(f"GREEN calls:   {r.get('n_green_calls', 0)}")
    print()

    u_delta = r.get("u_delta")
    n_green = r.get("n_green_calls", 0)

    if n_green < 3:
        print(f"⚠️  Only {n_green} GREEN calls — insufficient for confident assessment")
        print("   Recommendation: run more canary queries or increase traffic_green")
    elif u_delta is None:
        print("✗ Could not compute U delta")
    elif u_delta > 0:
        print(f"✓ CANARY PASSED — GREEN U is {u_delta:+.4f} vs BLUE")
        print("  Recommendation: proceed to gradual_shift.py")
    else:
        print(f"✗ CANARY FAILED — GREEN U is {u_delta:+.4f} vs BLUE")
        print("  Recommendation: do not proceed to gradual shift")


def parse_args():
    p = argparse.ArgumentParser(description="AUA blue-green canary runner")
    p.add_argument("--blue",           default="http://localhost:9001", help="BLUE endpoint")
    p.add_argument("--green",          default="http://localhost:9011", help="GREEN endpoint")
    p.add_argument("--traffic-green",  type=float, default=0.05,       help="Fraction of traffic to GREEN (0.05 = 5%%)")
    p.add_argument("--n",              type=int,   default=50,         help="Total queries to run")
    p.add_argument("--queries",        default=None,                   help="Benchmark queries JSON")
    p.add_argument("--output",         default="results/canary.json",  help="Output path")
    p.add_argument("--seed",           type=int,   default=42,         help="Random seed for routing decisions")
    p.add_argument("--check",          action="store_true",            help="Check canary results instead of running")
    p.add_argument("--results",        default=None,                   help="Results JSON for --check mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check:
        if not args.results:
            print("--check requires --results")
            sys.exit(1)
        check_canary(args.results)
    else:
        asyncio.run(run_canary(args))
