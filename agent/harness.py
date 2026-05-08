"""
MVP test harness — Adaptive Utility Agent v0.4.

Runs LeetCode-style problems through the full pipeline. Default backend is the
local vLLM SWE specialist (Stage 1 cloud); legacy Anthropic backend is preserved
for off-cloud use.

v0.4 features (all preserved):
    - Efficacy EMA accumulation
    - Dynamic difficulty routing based on domain confidence
    - Arbiter Agent for cross-problem consistency checks
    - Assertions store with decay
    - Trust manager
    - DPO pair export after each cycle

v0.4-cloud additions:
    - argparse CLI: --endpoint, --model, --cycles, --queries, --export-dpo, --field
    - Multi-cycle paired DPO output: cycle-1 contradicting solution = rejected;
      later cycle's improved solution = chosen.

Usage (cloud, default — Stage 1 SWE specialist on RunPod):
    python harness.py --cycles 2 --queries seeded_contradictions.json \\
                       --export-dpo dpo_pairs/cycle1.json

Usage (legacy, Anthropic API):
    export ANTHROPIC_API_KEY=sk-ant-...
    python harness.py --endpoint https://api.anthropic.com --model claude-haiku-4-5-20251001
"""

import os
import json
import httpx
import asyncio
import argparse
from datetime import datetime
from typing import Callable, Optional

from config import FIELD_CONFIGS, get_effective_config
from field_classifier import FieldClassifier
from contradiction_detector import ContradictionDetector
from utility_scorer import UtilityScorer
from personality_manager import PersonalityManager
from assertions_store import AssertionsStore
from arbiter import ArbiterAgent, VerdictCase
from trust_manager import TrustManager


# ── Problem bank ──────────────────────────────────────────────────────────────

PROBLEMS = {
    "easy": [
        {
            "id": "two_sum",
            "prompt": "Write a Python function two_sum(nums, target) that returns indices of two numbers that add to target. Use a hash map. State time complexity.",
            "baseline": 0.72,
            "novelty": 0.8,
        },
        {
            "id": "is_palindrome",
            "prompt": "Write a Python function is_palindrome(s) that returns True if s is a palindrome. Include a test case. State time complexity.",
            "baseline": 0.65,
            "novelty": 0.7,
        },
        {
            "id": "valid_parentheses",
            "prompt": "Write a Python function valid_parentheses(s) that checks if brackets are balanced. Use a stack. State time complexity.",
            "baseline": 0.70,
            "novelty": 0.6,
        },
    ],
    "medium": [
        {
            "id": "max_subarray",
            "prompt": "Write a Python function max_subarray(nums) using Kadane's algorithm. Explain why it works. State time complexity.",
            "baseline": 0.68,
            "novelty": 0.7,
        },
        {
            "id": "binary_search",
            "prompt": "Write a Python function binary_search(nums, target) on a sorted array. State time complexity and explain the loop invariant.",
            "baseline": 0.75,
            "novelty": 0.6,
        },
        {
            "id": "flatten_nested",
            "prompt": "Write a Python function flatten(lst) that recursively flattens a nested list. State time complexity.",
            "baseline": 0.62,
            "novelty": 0.7,
        },
    ],
    "hard": [
        {
            "id": "lru_cache",
            "prompt": "Implement an LRU cache in Python with get(key) and put(key, value) in O(1) time. Explain your data structure choice.",
            "baseline": 0.58,
            "novelty": 0.9,
        },
        {
            "id": "merge_intervals",
            "prompt": "Write merge_intervals(intervals) that merges overlapping intervals. Explain the sorting step and state time complexity.",
            "baseline": 0.63,
            "novelty": 0.8,
        },
    ],
}


def get_problems_for_difficulty(difficulty: str):
    return PROBLEMS.get(difficulty, PROBLEMS["easy"])


async def call_vllm(
    prompt: str,
    system_prompt: str,
    endpoint: str,
    model: str = "swe",
    timeout: float = 120.0,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """Call a vLLM OpenAI-compatible /v1/chat/completions endpoint."""
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
        })
        try:
            resp.raise_for_status()
        except Exception as e:
            return f"[vLLM error: {type(e).__name__}: {str(e)[:200]}]"
        body = resp.json()
        if "choices" in body and body["choices"]:
            return body["choices"][0]["message"].get("content", "") or ""
        return f"[vLLM unexpected response: {str(body)[:200]}]"


async def call_claude(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    """Call Claude API and return response text."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        data = response.json()
        if "content" in data and data["content"]:
            return data["content"][0].get("text", "")
        return f"[API error: {data.get('error', {}).get('message', 'unknown')}]"


async def run_problem(
    problem: dict,
    cycle: int,
    scorer: UtilityScorer,
    detector: ContradictionDetector,
    arbiter: ArbiterAgent,
    personality: PersonalityManager,
    call_fn: Callable[[str, str], "asyncio.Future[str]"],
    field: str,
    field_config,
    active_corrections: list,
    prior_solution: Optional[str] = None,
) -> dict:
    """Run one problem through the full pipeline.

    `call_fn(prompt, system_prompt) -> str` is an async callable abstracting the
    backend (vLLM or Anthropic). Bound at the top of main().
    """

    # Build system prompt with personality and active corrections
    traits = personality.get_active_weights(field)
    corrections_block = ""
    if active_corrections:
        corrections_str = "\n".join(f"  - {c}" for c in active_corrections[-20:])
        corrections_block = f"\nACTIVE CORRECTIONS (verified — do not repeat these errors):\n{corrections_str}\n"

    system_prompt = f"""You are a software engineering assistant.
Minimum confidence standard: {field_config.c_min:.0%}
If unsure, say so explicitly rather than guessing.
{corrections_block}
Approach: analytical_rigor={traits.get('analytical_rigor', 0.6):.2f}, caution={traits.get('caution', 0.5):.2f}

Always include: working code, time complexity claim, at least one assert statement."""

    # Call the bound LLM backend
    print(f"  Calling LLM for {problem['id']}...", end=" ", flush=True)
    solution = await call_fn(problem["prompt"], system_prompt)
    print("done")

    # Extract claimed complexity from response
    import re
    complexity_match = re.search(r"O\([^)]+\)", solution)
    claimed_complexity = complexity_match.group(0) if complexity_match else None

    # Contradiction detection
    cd_result = detector.check(
        problem=problem["prompt"],
        solution=solution,
        claimed_complexity=claimed_complexity,
    )

    # Arbiter (compare against prior cycle's solution if available)
    arbiter_verdict = None
    gap_subject = None
    if prior_solution and cycle > 0:
        arbiter_verdict = arbiter.arbitrate(
            subject=problem["id"],
            domain=field,
            output_A=solution,
            output_B=prior_solution,
            field_penalty_multiplier=field_config.penalty_multiplier,
            claimed_complexity_A=claimed_complexity,
        )
        if arbiter_verdict.case == VerdictCase.CASE_3:
            gap_subject = problem["id"]
            print(f"    ⚡ Arbiter Case 3: gap bonus on '{problem['id']}'")
        elif arbiter_verdict.case == VerdictCase.CASE_4:
            print(f"    ❓ Arbiter inconclusive for '{problem['id']}'")

    # Score interaction.
    # test_pass_rate is now derived from contradiction count rather than a
    # hardcoded floor that climbed with cycle index. Each contradiction docks
    # the pass rate by 0.15 (capped at 0.1 floor); zero contradictions = 1.0.
    n_contra = len(cd_result.contradictions)
    test_pass_rate = 1.0 if n_contra == 0 else max(0.1, 1.0 - n_contra * 0.15)

    task_score = scorer.score(
        task_id=problem["id"],
        field_config=field_config,
        test_pass_rate=test_pass_rate,
        human_baseline_score=problem["baseline"],
        contradiction_penalty=cd_result.confidence_penalty,
        problem_novelty=problem["novelty"],
        active_gap_subject=gap_subject,
    )

    # Session corrections
    new_corrections = []
    for c in cd_result.contradictions:
        correction = f"[{field}:{problem['id']}] {c.type}: {c.description}"
        new_corrections.append(correction)
        active_corrections.append(correction)

    return {
        "problem_id": problem["id"],
        "cycle": cycle + 1,
        "solution_preview": solution[:200] + "..." if len(solution) > 200 else solution,
        "claimed_complexity": claimed_complexity,
        "utility": task_score.utility,
        "efficacy_ema": task_score.efficacy_ema,
        "confidence": task_score.confidence,
        "gap_bonus": task_score.gap_bonus,
        "contradictions": len(cd_result.contradictions),
        "contradiction_details": [c.description for c in cd_result.contradictions],
        "arbiter_case": arbiter_verdict.case.value if arbiter_verdict else None,
        "below_minimum": task_score.below_minimum,
        "recommended_difficulty": task_score.recommended_difficulty,
        "new_corrections": new_corrections,
        "solution": solution,   # stored for Arbiter comparison next cycle
    }


def _load_queries(path: Optional[str]) -> list:
    """Load problems from a JSON file (list of {id, prompt, baseline, novelty}) or
    flatten the built-in PROBLEMS dict if path is None."""
    if path:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "queries" in data:
            return data["queries"]
        return data
    return [p for tier in PROBLEMS.values() for p in tier]


_ERROR_MARKERS = (
    "vLLM error", "HTTPStatusError", "Client error '", "Server error '",
    "Connection refused", "TimeoutError", "ConnectError", "ReadError",
    "Error '4", "Error '5", "raise_for_status",
)

def _is_valid_solution(text: str) -> bool:
    """Return False if text is a server error string rather than a model response."""
    if not text or len(text) < 10:
        return False
    return not any(m in text for m in _ERROR_MARKERS)


def _build_paired_dpo(per_problem_history: dict, field: str, weight: float) -> list:
    """Produce paired (chosen, rejected) DPO entries from cross-cycle history.

    For each problem, find the earliest cycle with contradictions (rejected) and
    a later cycle with strictly fewer contradictions (chosen). If no improvement
    occurs, emit an unpaired rejected-only entry so the contradiction is still
    recorded.

    Server error strings (vLLM 400/500, connection refused, etc.) are filtered
    out and never recorded as chosen or rejected.
    """
    pairs = []
    for pid, history in per_problem_history.items():
        # history: list of {"cycle", "solution", "n_contradictions", "prompt", "details"}
        # Filter out error-string entries entirely
        history = [h for h in history if _is_valid_solution(h.get("solution", ""))]
        if not history:
            continue

        rejected_idx = None
        for i, h in enumerate(history):
            if h["n_contradictions"] > 0:
                rejected_idx = i
                break
        if rejected_idx is None:
            continue

        rejected = history[rejected_idx]
        chosen = None
        for h in history[rejected_idx + 1 :]:
            if h["n_contradictions"] < rejected["n_contradictions"]:
                chosen = h
                break

        if chosen is not None:
            pairs.append({
                "task_id":               pid,
                "field":                 field,
                "prompt":                rejected["prompt"],
                "chosen":                chosen["solution"],
                "rejected":              rejected["solution"],
                "weight":                weight,
                "source":                "cross_cycle_improvement",
                "rejected_cycle":        rejected["cycle"],
                "chosen_cycle":          chosen["cycle"],
                "rejected_contradictions": rejected["details"],
                "rejected_n_contradictions": rejected["n_contradictions"],
                "chosen_n_contradictions":   chosen["n_contradictions"],
            })
        else:
            pairs.append({
                "task_id":               pid,
                "field":                 field,
                "prompt":                rejected["prompt"],
                "chosen":                None,
                "rejected":              rejected["solution"],
                "weight":                weight,
                "source":                "contradiction_only_no_improvement",
                "rejected_cycle":        rejected["cycle"],
                "rejected_contradictions": rejected["details"],
                "rejected_n_contradictions": rejected["n_contradictions"],
            })
    return pairs


async def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Utility Agent harness — runs problems through the full pipeline and exports DPO pairs.")
    parser.add_argument("--endpoint", default="http://localhost:9001",
                        help="LLM endpoint base URL. Default: vLLM SWE specialist on port 9001. Use https://api.anthropic.com for legacy Claude path (requires ANTHROPIC_API_KEY).")
    parser.add_argument("--model", default="swe",
                        help="Served model name. For vLLM, matches --served-model-name from server start. (default: %(default)s)")
    parser.add_argument("--cycles", type=int, default=2,
                        help="Number of cycles to run per problem (default: %(default)s)")
    parser.add_argument("--queries", default=None,
                        help="Path to JSON of {id, prompt, baseline, novelty} problems. Default: built-in PROBLEMS bank.")
    parser.add_argument("--export-dpo", default=None,
                        help="Path to write paired DPO JSON (default: dpo_pairs/cycle1_<timestamp>.json)")
    parser.add_argument("--field", default="software_engineering",
                        help="Field name (must exist in FIELD_CONFIGS). Default: %(default)s")
    parser.add_argument("--out", default=None,
                        help="Path for full harness results JSON (default: harness_results_<timestamp>.json)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature passed to vLLM (default: %(default)s)")
    parser.add_argument("--append", action="store_true",
                        help="Append new DPO pairs to existing --export-dpo file rather than overwriting")
    args = parser.parse_args()

    field = args.field
    field_config = FIELD_CONFIGS[field]
    num_cycles = args.cycles

    # Bind backend ───────────────────────────────────────────────────────────
    is_anthropic = "anthropic.com" in args.endpoint
    if is_anthropic:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set; required for endpoint pointing at Anthropic.")
            return
        async def call_fn(prompt, sys_p):
            return await call_claude(prompt, sys_p, api_key, model=args.model)
        backend_label = f"Anthropic ({args.model})"
    else:
        async def call_fn(prompt, sys_p):
            return await call_vllm(prompt, sys_p, args.endpoint, model=args.model,
                                   temperature=args.temperature)
        backend_label = f"vLLM ({args.endpoint}, model={args.model}, temp={args.temperature})"

    # Components ─────────────────────────────────────────────────────────────
    assertions_store = AssertionsStore(confidence_threshold=0.5)
    arbiter = ArbiterAgent(assertions_store=assertions_store)
    scorer = UtilityScorer(arbiter=arbiter)
    detector = ContradictionDetector(penalty_multiplier=field_config.penalty_multiplier)
    personality = PersonalityManager()
    trust_manager = TrustManager()

    active_corrections = []
    prior_solutions = {}
    all_results = []
    per_problem_history = {}    # pid -> list of cycle entries (for paired DPO)

    queries = _load_queries(args.queries)

    print(f"\n{'='*60}")
    print("Adaptive Utility Agent Harness v0.4-cloud")
    print(f"  Backend: {backend_label}")
    print(f"  Field:   {field}")
    print(f"  Cycles:  {num_cycles}")
    print(f"  Queries: {len(queries)}  (from {args.queries or 'built-in PROBLEMS bank'})")
    print(f"{'='*60}\n")

    for cycle in range(num_cycles):
        print(f"\n── Cycle {cycle+1} {'─'*45}")
        domain_summary = scorer.get_domain_summary(field)
        current_confidence = domain_summary.get("confidence", 0.5)
        recommended_diff = scorer._recommended_difficulty(current_confidence)
        print(f"   Domain confidence: {current_confidence:.3f} → routing to '{recommended_diff}' problems\n")

        cycle_results = []
        for problem in queries:
            result = await run_problem(
                problem=problem,
                cycle=cycle,
                scorer=scorer,
                detector=detector,
                arbiter=arbiter,
                personality=personality,
                call_fn=call_fn,
                field=field,
                field_config=field_config,
                active_corrections=active_corrections,
                prior_solution=prior_solutions.get(problem["id"]),
            )
            if _is_valid_solution(result["solution"]):
                prior_solutions[problem["id"]] = result["solution"]
            else:
                # Server returned an error string — skip history recording
                continue

            per_problem_history.setdefault(problem["id"], []).append({
                "cycle":             cycle + 1,
                "solution":          result["solution"],
                "n_contradictions":  result["contradictions"],
                "details":           result["contradiction_details"],
                "prompt":            problem["prompt"],
                "utility":           result["utility"],
                "claimed_complexity": result["claimed_complexity"],
            })

            print(
                f"   {problem['id']:<28} U={result['utility']:.4f} "
                f"E_ema={result['efficacy_ema']:.4f} "
                f"C={result['confidence']:.4f}"
                + (f" gap={result['gap_bonus']:.3f}" if result["gap_bonus"] > 0 else "")
                + (f" ⚠ x{result['contradictions']}" if result["contradictions"] > 0 else "")
                + (" 🔴 ABSTAIN" if result["below_minimum"] else "")
            )
            cycle_results.append(result)
            all_results.append(result)

        if not cycle_results:
            print(f"\n   Cycle {cycle+1}: no valid responses (all filtered — likely context-length overflow)")
            continue
        avg_U = sum(r["utility"] for r in cycle_results) / len(cycle_results)
        avg_E = sum(r["efficacy_ema"] for r in cycle_results) / len(cycle_results)
        avg_C = sum(r["confidence"] for r in cycle_results) / len(cycle_results)
        total_contra = sum(r["contradictions"] for r in cycle_results)
        print(f"\n   Cycle {cycle+1}: avg U={avg_U:.4f} | E_ema={avg_E:.4f} | C={avg_C:.4f} | contradictions={total_contra}")

        utility_trend = scorer.get_utility_trend(field)
        domain_sum = scorer.get_domain_summary(field)
        personality.evolve(
            utility_history=utility_trend,
            contradiction_rate=domain_sum.get("contradiction_rate", 0.0),
            domain=field,
        )
        traits = personality.get_trait_summary()
        print(f"   Personality: curiosity={traits.get('curiosity', 0):.2f} "
              f"caution={traits.get('caution', 0):.2f} "
              f"analytical_rigor={traits.get('analytical_rigor', 0):.2f}")

    # Build paired DPO output ────────────────────────────────────────────────
    dpo_pairs = _build_paired_dpo(per_problem_history, field, field_config.penalty_multiplier)
    n_paired   = sum(1 for p in dpo_pairs if p.get("chosen") is not None)
    n_unpaired = len(dpo_pairs) - n_paired

    print(f"\n{'='*60}")
    print("HARNESS COMPLETE")
    print(f"{'='*60}")
    print(f"  Arbiter verdicts:        {arbiter.total_verdicts}")
    print(f"  Arbiter corrections:     {arbiter.total_corrections_issued}")
    print(f"  Correction rate:         {arbiter.correction_rate():.1%}")
    print(f"  DPO pairs (total):       {len(dpo_pairs)}")
    print(f"    paired chosen+rejected: {n_paired}")
    print(f"    rejected-only:          {n_unpaired}")
    print(f"  Assertions stored:       {assertions_store.summary()['total']}")
    print(f"  Active corrections:      {len(active_corrections)}")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    out_path = args.out or f"harness_results_{timestamp}.json"
    out = {
        "timestamp":         timestamp,
        "backend":           backend_label,
        "endpoint":          args.endpoint,
        "model":             args.model,
        "field":             field,
        "cycles":            num_cycles,
        "n_queries":         len(queries),
        "results": [
            {k: v for k, v in r.items() if k != "solution"}
            for r in all_results
        ],
        "dpo_summary":       { "total": len(dpo_pairs), "paired": n_paired, "rejected_only": n_unpaired },
        "arbiter_status":    arbiter.status(),
        "assertions_store":  assertions_store.summary(),
        "personality_final": personality.get_trait_summary(),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results: {out_path}")

    dpo_path = args.export_dpo or f"dpo_pairs/cycle1_{timestamp}.json"
    os.makedirs(os.path.dirname(dpo_path) or ".", exist_ok=True)

    if args.append and os.path.exists(dpo_path):
        with open(dpo_path) as f:
            prior = json.load(f)
        prior_pairs = prior if isinstance(prior, list) else prior.get("pairs", prior.get("entries", []))
        combined = prior_pairs + dpo_pairs
        with open(dpo_path, "w") as f:
            json.dump(combined, f, indent=2)
        n_total  = len(combined)
        n_paired_total = sum(1 for p in combined if p.get("chosen") is not None)
        print(f"  DPO pairs:    {dpo_path}  (appended; total {n_total}, paired {n_paired_total})")
    else:
        with open(dpo_path, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "field":     field,
                "cycles":    num_cycles,
                "endpoint":  args.endpoint,
                "model":     args.model,
                "temperature": args.temperature,
                "n_pairs":   len(dpo_pairs),
                "n_paired":  n_paired,
                "n_rejected_only": n_unpaired,
                "pairs":     dpo_pairs,
            }, f, indent=2)
        print(f"  DPO pairs:    {dpo_path}")


if __name__ == "__main__":
    asyncio.run(main())
