"""
MVP test harness — Adaptive Utility Agent v0.4.

Runs LeetCode-style problems through the full pipeline with a live Claude API call.
Incorporates all v0.4 changes:
    - Efficacy EMA accumulation
    - Dynamic difficulty routing based on domain confidence
    - Arbiter Agent for cross-problem consistency checks
    - Assertions store with decay
    - Trust manager
    - DPO pair export after each cycle

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python harness.py
"""

import os
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional

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
    api_key: str,
    field: str,
    field_config,
    active_corrections: list,
    prior_solution: Optional[str] = None,
) -> dict:
    """Run one problem through the full pipeline."""

    # Build system prompt with personality and active corrections
    traits = personality.get_active_weights(field)
    corrections_block = ""
    if active_corrections:
        corrections_str = "\n".join(f"  - {c}" for c in active_corrections[-5:])
        corrections_block = f"\nACTIVE CORRECTIONS (verified — do not repeat these errors):\n{corrections_str}\n"

    system_prompt = f"""You are a software engineering assistant.
Minimum confidence standard: {field_config.c_min:.0%}
If unsure, say so explicitly rather than guessing.
{corrections_block}
Approach: analytical_rigor={traits.get('analytical_rigor', 0.6):.2f}, caution={traits.get('caution', 0.5):.2f}

Always include: working code, time complexity claim, at least one assert statement."""

    # Call Claude
    print(f"  Calling API for {problem['id']}...", end=" ", flush=True)
    solution = await call_claude(problem["prompt"], system_prompt, api_key)
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

    # Score interaction
    task_score = scorer.score(
        task_id=problem["id"],
        field_config=field_config,
        test_pass_rate=0.85 + (cycle * 0.03),   # improves per cycle in live system
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


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-...")
        return

    field = "software_engineering"
    field_config = FIELD_CONFIGS[field]
    num_cycles = 3

    # Initialize all components
    assertions_store = AssertionsStore(confidence_threshold=0.5)
    arbiter = ArbiterAgent(assertions_store=assertions_store)
    scorer = UtilityScorer(arbiter=arbiter)
    detector = ContradictionDetector(penalty_multiplier=field_config.penalty_multiplier)
    personality = PersonalityManager()
    trust_manager = TrustManager()

    active_corrections = []
    prior_solutions = {}     # problem_id → solution from last cycle
    all_results = []
    dpo_pairs = []

    print(f"\n{'='*60}")
    print(f"Adaptive Utility Agent Harness v0.4")
    print(f"Field: {field}  |  Cycles: {num_cycles}")
    print(f"{'='*60}\n")

    for cycle in range(num_cycles):
        print(f"\n── Cycle {cycle+1} {'─'*45}")

        # Determine difficulty routing from current domain confidence
        domain_summary = scorer.get_domain_summary(field)
        current_confidence = domain_summary.get("confidence", 0.5)
        recommended_diff = scorer._recommended_difficulty(current_confidence)

        print(f"   Domain confidence: {current_confidence:.3f} → routing to '{recommended_diff}' problems\n")

        # Run all problems regardless of tier for harness coverage
        # (In production, the router would select by difficulty tier)
        cycle_results = []
        all_problems = [p for tier in PROBLEMS.values() for p in tier]

        for problem in all_problems:
            result = await run_problem(
                problem=problem,
                cycle=cycle,
                scorer=scorer,
                detector=detector,
                arbiter=arbiter,
                personality=personality,
                api_key=api_key,
                field=field,
                field_config=field_config,
                active_corrections=active_corrections,
                prior_solution=prior_solutions.get(problem["id"]),
            )
            prior_solutions[problem["id"]] = result["solution"]

            # Collect DPO pairs for contradictions
            if result["contradictions"] > 0:
                dpo_pairs.append({
                    "task_id": problem["id"],
                    "field": field,
                    "rejected_preview": result["solution_preview"],
                    "reason": result["contradiction_details"],
                    "weight": field_config.penalty_multiplier,
                    "cycle": cycle + 1,
                })

            print(
                f"   {problem['id']:<22} U={result['utility']:.4f} "
                f"E_ema={result['efficacy_ema']:.4f} "
                f"C={result['confidence']:.4f}"
                + (f" gap={result['gap_bonus']:.3f}" if result["gap_bonus"] > 0 else "")
                + (" ⚠" if result["contradictions"] > 0 else "")
                + (" 🔴 ABSTAIN" if result["below_minimum"] else "")
            )

            cycle_results.append(result)
            all_results.append(result)

        # Cycle summary
        avg_U = sum(r["utility"] for r in cycle_results) / len(cycle_results)
        avg_E = sum(r["efficacy_ema"] for r in cycle_results) / len(cycle_results)
        avg_C = sum(r["confidence"] for r in cycle_results) / len(cycle_results)
        total_contradictions = sum(r["contradictions"] for r in cycle_results)

        print(f"\n   Cycle {cycle+1}: avg U={avg_U:.4f} | E_ema={avg_E:.4f} | C={avg_C:.4f} | contradictions={total_contradictions}")

        # Personality evolution
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

    # Final report
    print(f"\n{'='*60}")
    print("HARNESS COMPLETE")
    print(f"{'='*60}")
    print(f"  Arbiter verdicts:      {arbiter.total_verdicts}")
    print(f"  Arbiter corrections:   {arbiter.total_corrections_issued}")
    print(f"  Correction rate:       {arbiter.correction_rate():.1%}")
    print(f"  DPO pairs collected:   {len(dpo_pairs)}")
    print(f"  Assertions stored:     {assertions_store.summary()['total']}")
    print(f"  Active corrections:    {len(active_corrections)}")

    # Save results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": timestamp,
        "field": field,
        "cycles": num_cycles,
        "results": [
            {k: v for k, v in r.items() if k != "solution"}  # omit full solution from JSON
            for r in all_results
        ],
        "dpo_pairs": dpo_pairs,
        "arbiter_status": arbiter.status(),
        "assertions_store": assertions_store.summary(),
        "personality_final": personality.get_trait_summary(),
    }

    out_path = f"harness_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
