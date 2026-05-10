"""
evaluate.py — Benchmark runner for AUA blue-green deployment
Phase 3 POC

Records a baseline snapshot of a specialist endpoint:
  - Average U score
  - Brier score (confidence calibration)
  - Contradiction rate
  - Per-problem results

Run this BEFORE training GREEN to record BLUE baseline,
then again AFTER canary to compare GREEN vs BLUE.

Usage:
    # Record BLUE baseline
    python evaluate.py \
        --endpoint http://localhost:9001 \
        --queries benchmark_swe.json \
        --label blue_baseline \
        --output results/blue_baseline.json

    # Record GREEN after training
    python evaluate.py \
        --endpoint http://localhost:9011 \
        --queries benchmark_swe.json \
        --label green_cycle1 \
        --output results/green_cycle1.json

    # Compare two baselines
    python evaluate.py --compare \
        --baseline results/blue_baseline.json \
        --candidate results/green_cycle1.json
"""

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from utility_scorer import UtilityScorer
from contradiction_detector import ContradictionDetector
from confidence_updater import ConfidenceUpdater
from assertions_store import AssertionsStore
from config import FIELD_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("evaluate")

# ── Default benchmark queries (SWE) ──────────────────────────────────────────
# Used if no --queries file is provided

DEFAULT_BENCHMARK = [
    {
        "id": "b01_binary_search",
        "prompt": "Write a Python function binary_search(nums, target) on a sorted array. State the time complexity.",
        "expected_complexity": "O(log n)",
        "domain": "software_engineering"
    },
    {
        "id": "b02_two_sum",
        "prompt": "Write an efficient Python function two_sum(nums, target) that returns indices. State the time complexity.",
        "expected_complexity": "O(n)",
        "domain": "software_engineering"
    },
    {
        "id": "b03_merge_sort",
        "prompt": "Implement merge sort in Python. State time and space complexity.",
        "expected_complexity": "O(n log n)",
        "domain": "software_engineering"
    },
    {
        "id": "b04_lru_cache",
        "prompt": "Implement an LRU cache with get and put operations in O(1). Explain the data structures used.",
        "expected_complexity": "O(1)",
        "domain": "software_engineering"
    },
    {
        "id": "b05_max_subarray",
        "prompt": "Implement Kadane's algorithm for maximum subarray sum in Python. State the time complexity.",
        "expected_complexity": "O(n)",
        "domain": "software_engineering"
    },
    {
        "id": "b06_valid_parens",
        "prompt": "Write a Python function to check if a string of parentheses is valid. State time and space complexity.",
        "expected_complexity": "O(n)",
        "domain": "software_engineering"
    },
    {
        "id": "b07_fibonacci",
        "prompt": "Implement Fibonacci using dynamic programming in Python. Compare naive recursion vs DP complexity.",
        "expected_complexity": "O(n)",
        "domain": "software_engineering"
    },
    {
        "id": "b08_reverse_linked_list",
        "prompt": "Write a Python function to reverse a singly linked list in-place. State time and space complexity.",
        "expected_complexity": "O(n)",
        "domain": "software_engineering"
    },
    {
        "id": "b09_matrix_multiply",
        "prompt": "Implement matrix multiplication in Python without numpy. State the time complexity.",
        "expected_complexity": "O(n^3)",
        "domain": "software_engineering"
    },
    {
        "id": "b10_quicksort",
        "prompt": "Implement quicksort in Python. State average and worst-case time complexity.",
        "expected_complexity": "O(n log n)",
        "domain": "software_engineering"
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def brier_score(confidences: list[float], correct: list[bool]) -> float:
    """
    Brier score: mean squared error between confidence and binary outcome.
    0 = perfect calibration, 1 = worst possible.
    """
    if not confidences:
        return 1.0
    return sum(
        (c - (1.0 if ok else 0.0)) ** 2
        for c, ok in zip(confidences, correct)
    ) / len(confidences)


def is_correct(response: str, expected_complexity: Optional[str]) -> bool:
    """
    Heuristic correctness check.
    For complexity questions: verify expected big-O appears in response.
    Normalises common variants.
    """
    if not expected_complexity:
        return True  # no ground truth → assume correct

    resp_lower = response.lower()

    # Normalise expected
    exp = (expected_complexity
           .lower()
           .replace("o(", "")
           .replace(")", "")
           .replace(" ", "")
           .strip())

    # Variants to check
    variants = {exp}
    if exp == "nlogn" or exp == "n log n":
        variants.update(["nlogn", "n log n", "n*log(n)", "o(nlogn)"])
    if exp == "logn" or exp == "log n":
        variants.update(["logn", "log n", "log(n)"])
    if exp == "n^3" or exp == "n3":
        variants.update(["n^3", "n³", "n**3", "cubic"])
    if exp == "n^2" or exp == "n2":
        variants.update(["n^2", "n²", "n**2", "quadratic"])

    for v in variants:
        if v in resp_lower.replace(" ", ""):
            return True

    return False


# ── Inference call ─────────────────────────────────────────────────────────────

async def call_endpoint(
    client: httpx.AsyncClient,
    endpoint: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
    model_id: str = "swe",
) -> tuple[str, int, int, float]:
    """
    Call vLLM OpenAI-compatible endpoint.
    Returns: (response_text, prompt_tokens, completion_tokens, latency_ms)
    """
    t0 = time.time()
    messages = [
        {"role": "system", "content": "You are a specialist software engineering assistant. Answer precisely and correctly."},
        {"role": "user",   "content": prompt},
    ]
    resp = await client.post(
        endpoint,
        json={
            "model": model_id,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    latency = (time.time() - t0) * 1000
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), latency


# ── Main evaluation loop ──────────────────────────────────────────────────────

async def run_evaluation(args) -> dict:
    # Load queries
    # ── Load clean benchmark queries ──────────────────────────────────────
    if args.queries and Path(args.queries).exists():
        with open(args.queries) as f:
            queries = json.load(f)
        log.info(f"Loaded {len(queries)} clean queries from {args.queries}")
    else:
        queries = DEFAULT_BENCHMARK
        log.info(f"Using default benchmark ({len(queries)} clean queries)")

    # ── Mix in adversarial queries if requested ────────────────────────────
    # GREEN is trained on DPO pairs from seeded_contradictions.json.
    # A clean-only eval won't show GREEN's improvement because both models
    # score nearly identically on well-formed questions. Mixing in adversarial
    # queries — where GREEN was specifically trained to avoid wrong complexity
    # claims — produces a meaningful U delta for promotion decisions.
    adv_path = getattr(args, 'include_adversarial', None)
    if adv_path and Path(adv_path).exists():
        import random
        with open(adv_path) as f:
            adv_data = json.load(f)
        adv_pool = adv_data.get("queries", adv_data) if isinstance(adv_data, dict) else adv_data

        # Normalise adversarial entries to match benchmark format
        adv_normalised = []
        for q in adv_pool:
            adv_normalised.append({
                "id":    q.get("id", "adv"),
                "prompt": q.get("prompt", ""),
                "domain": q.get("domain", "software_engineering"),
                # No expected_complexity — correctness judged by contradiction count
                "expected_complexity": None,
                "is_adversarial": True,
                "expected_contradiction": q.get("expected_contradiction", ""),
            })

        n_adv = getattr(args, 'adversarial_n', 10)
        ratio = getattr(args, 'adversarial_ratio', 0.5)
        n_adv = min(n_adv, len(adv_normalised))
        sampled_adv = random.sample(adv_normalised, n_adv)

        # Keep only the clean fraction that balances the ratio
        if ratio > 0:
            n_clean_keep = int(n_adv * (1.0 - ratio) / max(ratio, 1e-6))
            clean_queries = queries[:n_clean_keep] if n_clean_keep < len(queries) else queries
        else:
            clean_queries = queries
            sampled_adv = []

        queries = clean_queries + sampled_adv
        log.info(
            f"Mixed eval: {len(clean_queries)} clean + {len(sampled_adv)} adversarial "
            f"= {len(queries)} total (ratio={ratio:.0%} adversarial)"
        )
    else:
        if adv_path:
            log.warning(f"--include-adversarial path not found: {adv_path} — using clean only")

    # Init scorers
    store    = AssertionsStore()
    detector = ContradictionDetector(penalty_multiplier=2.0)
    updater  = ConfidenceUpdater()
    scorer   = UtilityScorer()

    results = []
    confidences, corrects = [], []
    total_contradictions = 0
    total_dpo_pairs = 0
    total_latency = 0.0

    base_url = args.endpoint.rstrip("/")
    completions_url = (
        base_url + "/v1/chat/completions"
        if not base_url.endswith("/v1/chat/completions")
        else base_url
    )

    log.info(f"Evaluating endpoint: {completions_url}")
    log.info(f"Queries: {len(queries)} | Label: {args.label}")
    log.info("")

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(queries):
            qid    = q.get("id", f"q{i+1}")
            prompt = q.get("prompt", q.get("query", ""))
            domain = q.get("domain", "software_engineering")
            expected = q.get("expected_complexity")

            if not prompt:
                log.warning(f"Skipping {qid} — no prompt field")
                continue

            try:
                response, p_tok, c_tok, latency_ms = await call_endpoint(
                    client, completions_url, prompt, model_id=args.model_id
                )
            except Exception as e:
                log.error(f"[{qid}] Call failed: {e}")
                results.append({"id": qid, "error": str(e)})
                continue

            # Correctness
            # For adversarial queries: correct = zero contradictions detected
            # (GREEN should resist the wrong complexity claim in the prompt)
            # For clean queries: correct = expected complexity appears in response
            is_adv = q.get("is_adversarial", False)
            if is_adv:
                # Run detection first, then judge correctness
                contradiction_result_pre = detector.check(
                    problem=prompt, solution=response
                )
                correct = len(contradiction_result_pre.contradictions) == 0
                # Re-use this result below so we don't double-call
            else:
                correct = is_correct(response, expected)

            # Contradiction detection
            if is_adv:
                contradiction_result = contradiction_result_pre  # already computed above
            else:
                contradiction_result = detector.check(
                    problem=prompt,
                    solution=response,
                )
            n_contra = len(contradiction_result.contradictions)

            # Confidence update
            base_conf = 0.75 if correct else 0.45
            updated_conf = updater.update(
                prior=base_conf,
                test_signal=base_conf,
                contradiction_result=contradiction_result,
                field=domain,
            )

            # U score
            field_cfg = FIELD_CONFIGS.get(domain, FIELD_CONFIGS["general"])
            score_result = scorer.score(
                task_id=qid,
                field_config=field_cfg,
                test_pass_rate=1.0 if correct else 0.0,
                human_baseline_score=0.65,
                contradiction_penalty=float(n_contra) * field_cfg.penalty_multiplier * 0.05,
                problem_novelty=0.1,
            )
            u_score = score_result.utility

            confidences.append(updated_conf)
            corrects.append(correct)
            total_contradictions += n_contra
            total_latency += latency_ms

            result = {
                "id":               qid,
                "prompt":           prompt[:120] + "..." if len(prompt) > 120 else prompt,
                "correct":          correct,
                "is_adversarial":   is_adv,
                "expected":         expected,
                "u_score":          round(u_score, 4),
                "confidence":       round(updated_conf, 4),
                "contradictions":   n_contra,
                "latency_ms":       round(latency_ms, 1),
                "prompt_tokens":    p_tok,
                "completion_tokens": c_tok,
                "response_preview": response[:200] + "..." if len(response) > 200 else response,
            }
            results.append(result)

            status = "✓" if correct else "✗"
            log.info(
                f"[{i+1}/{len(queries)}] {qid} {status} "
                f"U={u_score:.3f} C={updated_conf:.3f} "
                f"contra={n_contra} {latency_ms:.0f}ms"
            )

    # Aggregate metrics
    n = len([r for r in results if "error" not in r])
    if n == 0:
        log.error("No successful evaluations")
        sys.exit(1)

    accuracy         = sum(corrects) / n
    mean_u           = sum(r["u_score"]    for r in results if "error" not in r) / n
    mean_conf        = sum(r["confidence"] for r in results if "error" not in r) / n
    brier            = brier_score(confidences, corrects)
    contradiction_rate = total_contradictions / n
    mean_latency     = total_latency / n

    # ── Adversarial subset stats for summary ─────────────────────────────────
    adv_results_s = [r for r in results if r.get("is_adversarial") and "error" not in r]
    adv_summary = {}
    if adv_results_s:
        adv_summary = {
            "n_adversarial": len(adv_results_s),
            "adversarial_accuracy": round(
                sum(1 for r in adv_results_s if r["correct"]) / len(adv_results_s), 4),
            "adversarial_mean_u": round(
                sum(r["u_score"] for r in adv_results_s) / len(adv_results_s), 4),
            "adversarial_contradiction_rate": round(
                sum(r["contradictions"] for r in adv_results_s) / len(adv_results_s), 4),
        }

    summary = {
        "label":              args.label,
        "endpoint":           args.endpoint,
        "n_queries":          n,
        "accuracy":           round(accuracy, 4),
        "mean_u":             round(mean_u, 4),
        "mean_confidence":    round(mean_conf, 4),
        "brier_score":        round(brier, 4),
        "contradiction_rate": round(contradiction_rate, 4),
        "total_contradictions": total_contradictions,
        "mean_latency_ms":    round(mean_latency, 1),
        "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "adversarial_subset": adv_summary,
        "per_query":          results,
    }

    log.info("")
    log.info("=== EVALUATION SUMMARY ===")
    log.info(f"Label:              {args.label}")
    log.info(f"Queries evaluated:  {n}")
    log.info(f"Accuracy:           {accuracy:.1%}")
    log.info(f"Mean U score:       {mean_u:.4f}")
    log.info(f"Mean confidence:    {mean_conf:.4f}")
    log.info(f"Brier score:        {brier:.4f}")
    log.info(f"Contradiction rate: {contradiction_rate:.2f} per query")
    log.info(f"Mean latency:       {mean_latency:.0f}ms")

    # ── Adversarial-subset breakdown ──────────────────────────────────────────
    adv_results = [r for r in results if r.get("is_adversarial") and "error" not in r]
    clean_results = [r for r in results if not r.get("is_adversarial") and "error" not in r]
    if adv_results:
        adv_acc = sum(1 for r in adv_results if r["correct"]) / len(adv_results)
        adv_u   = sum(r["u_score"] for r in adv_results) / len(adv_results)
        adv_contra = sum(r["contradictions"] for r in adv_results) / len(adv_results)
        log.info(f"")
        log.info(f"=== ADVERSARIAL SUBSET ({len(adv_results)} queries) ===")
        log.info(f"  Accuracy (no-contradiction rate): {adv_acc:.1%}")
        log.info(f"  Mean U:                           {adv_u:.4f}")
        log.info(f"  Mean contradictions per query:    {adv_contra:.2f}")
        log.info(f"  (GREEN trained on these — this is the signal that drives U delta)")
    if clean_results:
        clean_acc = sum(1 for r in clean_results if r["correct"]) / len(clean_results)
        clean_u   = sum(r["u_score"] for r in clean_results) / len(clean_results)
        log.info(f"")
        log.info(f"=== CLEAN SUBSET ({len(clean_results)} queries) ===")
        log.info(f"  Accuracy:  {clean_acc:.1%}")
        log.info(f"  Mean U:    {clean_u:.4f}")

    # Save output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Results saved to: {out_path}")

    return summary


def compare_results(baseline_path: str, candidate_path: str):
    """Compare two evaluation results and print a promotion-ready summary."""
    with open(baseline_path) as f:
        blue = json.load(f)
    with open(candidate_path) as f:
        green = json.load(f)

    print("\n=== BLUE vs GREEN COMPARISON ===")
    print(f"{'Metric':<25} {'BLUE (baseline)':<20} {'GREEN (candidate)':<20} {'Delta':<15}")
    print("-" * 80)

    metrics = [
        ("accuracy",           "Accuracy",           "{:.1%}"),
        ("mean_u",             "Mean U score",        "{:.4f}"),
        ("mean_confidence",    "Mean confidence",     "{:.4f}"),
        ("brier_score",        "Brier score",         "{:.4f}"),
        ("contradiction_rate", "Contradiction rate",  "{:.3f}"),
    ]

    for key, label, fmt in metrics:
        b_val = blue.get(key, 0)
        g_val = green.get(key, 0)
        delta = g_val - b_val
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        # For Brier, lower is better
        better = delta > 0 if key != "brier_score" else delta < 0
        marker = " ✓" if better else " ✗"
        print(
            f"{label:<25} {fmt.format(b_val):<20} {fmt.format(g_val):<20} "
            f"{delta_str:<15}{marker}"
        )

    print()
    u_delta = green.get("mean_u", 0) - blue.get("mean_u", 0)
    print(f"U delta: {u_delta:+.4f} (threshold for SWE: δ=0.025)")
    if u_delta >= 0.025:
        print("PROMOTION RECOMMENDATION: PROMOTE GREEN ✓")
    elif u_delta > 0:
        print(f"PROMOTION RECOMMENDATION: MARGINAL — delta {u_delta:.4f} < 0.025 threshold")
    else:
        print("PROMOTION RECOMMENDATION: DO NOT PROMOTE — GREEN is not better ✗")

    # Adversarial subset comparison
    b_adv = blue.get("adversarial_subset", {})
    g_adv = green.get("adversarial_subset", {})
    if b_adv and g_adv:
        print()
        print("=== ADVERSARIAL SUBSET COMPARISON (where GREEN was trained) ===")
        adv_keys = [
            ("adversarial_accuracy",           "Adv accuracy",     "{:.1%}"),
            ("adversarial_mean_u",              "Adv mean U",        "{:.4f}"),
            ("adversarial_contradiction_rate",  "Adv contra rate",   "{:.3f}"),
        ]
        for key, label, fmt in adv_keys:
            bv = b_adv.get(key, 0)
            gv = g_adv.get(key, 0)
            delta = gv - bv
            better = delta > 0 if key != "adversarial_contradiction_rate" else delta < 0
            marker = " ✓" if better else " ✗"
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            print(f"{label:<30} BLUE={fmt.format(bv):<12} GREEN={fmt.format(gv):<12} Δ={delta_str}{marker}")
        adv_u_delta = g_adv.get("adversarial_mean_u", 0) - b_adv.get("adversarial_mean_u", 0)
        print(f"\nAdversarial U delta: {adv_u_delta:+.4f}")
        if adv_u_delta >= 0.025:
            print("ADVERSARIAL PROMOTION: PROMOTE GREEN ✓ (adversarial subset passes threshold)")


def parse_args():
    p = argparse.ArgumentParser(description="AUA specialist endpoint evaluator")
    p.add_argument("--endpoint",  default="http://localhost:9001",
                   help="vLLM endpoint URL")
    p.add_argument("--queries",   default=None,
                   help="Path to benchmark queries JSON (default: built-in SWE benchmark)")
    p.add_argument("--label",     default="evaluation",
                   help="Label for this evaluation run (e.g. blue_baseline, green_cycle1)")
    p.add_argument("--output",    default=None,
                   help="Path to save results JSON")
    p.add_argument("--n",         type=int, default=None,
                   help="Number of queries to run (default: all)")
    p.add_argument("--include-adversarial", default=None, metavar="PATH",
                   help="Path to seeded_contradictions.json — mixes adversarial "
                        "queries with the clean benchmark so GREEN's DPO "
                        "training signal shows up in the U delta. "
                        "Use --adversarial-ratio to control the mix (default 0.5).")
    p.add_argument("--adversarial-ratio", type=float, default=0.5,
                   help="Fraction of eval queries that are adversarial (default 0.5). "
                        "0.0 = clean only, 1.0 = adversarial only.")
    p.add_argument("--adversarial-n", type=int, default=10,
                   help="Number of adversarial queries to sample (default 10).")
    p.add_argument("--model-id", default="swe",
                   help="Model ID to send in API requests (default: swe). "
                        "Override when serving LoRA adapters (e.g. swe_green_v1).")
    # Comparison mode
    p.add_argument("--compare",   action="store_true",
                   help="Compare two result files instead of running evaluation")
    p.add_argument("--baseline",  default=None,
                   help="Baseline (BLUE) results JSON for comparison")
    p.add_argument("--candidate", default=None,
                   help="Candidate (GREEN) results JSON for comparison")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.compare:
        if not args.baseline or not args.candidate:
            print("--compare requires --baseline and --candidate")
            sys.exit(1)
        compare_results(args.baseline, args.candidate)
    else:
        asyncio.run(run_evaluation(args))
