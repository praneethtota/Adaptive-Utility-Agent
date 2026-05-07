"""
Self-contained simulation for Adaptive Utility Agent v0.4.

No API calls needed. Uses synthetic responses of increasing quality
to simulate what a real model produces after DPO calibration.

Fixes from A.5:
    1. Efficacy now uses EMA accumulation across cycles (not fixed per-interaction)
    2. Difficulty escalates as domain confidence rises (resets novelty counter)

New in v0.4:
    - Arbiter Agent validates cross-model consistency
    - Assertions store persists verified facts with decay
    - Gap bonus applied when Arbiter detects both-wrong (Case 3)
    - DPO pairs collected and exported after each cycle
"""

import json
import random
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import FIELD_CONFIGS
from field_classifier import FieldClassifier
from contradiction_detector import ContradictionDetector, ContradictionResult
from utility_scorer import UtilityScorer, DIFFICULTY_THRESHOLDS
from personality_manager import PersonalityManager
from assertions_store import AssertionsStore
from arbiter import ArbiterAgent, VerdictCase


# ── Problem bank with difficulty tiers ───────────────────────────────────────

PROBLEMS = {
    "easy": [
        {"id": "two_sum",           "topic": "array", "baseline": 0.72},
        {"id": "is_palindrome",     "topic": "string", "baseline": 0.65},
        {"id": "valid_parentheses", "topic": "stack",  "baseline": 0.70},
    ],
    "medium": [
        {"id": "max_subarray",      "topic": "dp",         "baseline": 0.68},
        {"id": "binary_search",     "topic": "search",     "baseline": 0.75},
        {"id": "flatten_nested",    "topic": "recursion",  "baseline": 0.62},
    ],
    "hard": [
        {"id": "lru_cache",         "topic": "design",     "baseline": 0.58},
        {"id": "merge_intervals",   "topic": "sorting",    "baseline": 0.63},
    ],
}

# Synthetic responses: quality improves per cycle
CYCLE_RESPONSES = {
    "two_sum": [
        # Cycle 1: nested loop claiming O(n) — mathematical contradiction, detected
        {"pass_rate": 0.60, "complexity": "O(n)", "novelty": 0.8, "has_contradiction": True},
        # Cycle 2: hash-map solution — O(n) correct, contradiction resolved
        {"pass_rate": 0.87, "complexity": "O(n)", "novelty": 0.4, "has_contradiction": False},
        # Cycle 3: further improved
        {"pass_rate": 0.92, "complexity": "O(n)", "novelty": 0.2, "has_contradiction": False},
    ],
    "is_palindrome": [
        {"pass_rate": 0.70, "complexity": "O(n)", "novelty": 0.7, "has_contradiction": False},
        {"pass_rate": 0.85, "complexity": "O(n)", "novelty": 0.3, "has_contradiction": False},
        {"pass_rate": 0.90, "complexity": "O(n)", "novelty": 0.1, "has_contradiction": False},
    ],
    "max_subarray": [
        {"pass_rate": 0.75, "complexity": "O(n)", "novelty": 0.6, "has_contradiction": False},
        {"pass_rate": 0.88, "complexity": "O(n)", "novelty": 0.3, "has_contradiction": False},
        {"pass_rate": 0.92, "complexity": "O(n)", "novelty": 0.2, "has_contradiction": False},
    ],
    "binary_search": [
        {"pass_rate": 0.78, "complexity": "O(log n)", "novelty": 0.6, "has_contradiction": False},
        {"pass_rate": 0.89, "complexity": "O(log n)", "novelty": 0.2, "has_contradiction": False},
        {"pass_rate": 0.93, "complexity": "O(log n)", "novelty": 0.1, "has_contradiction": False},
    ],
    "flatten_nested": [
        {"pass_rate": 0.80, "complexity": "O(n)", "novelty": 0.7, "has_contradiction": False},
        {"pass_rate": 0.88, "complexity": "O(n)", "novelty": 0.3, "has_contradiction": False},
        {"pass_rate": 0.92, "complexity": "O(n)", "novelty": 0.2, "has_contradiction": False},
    ],
    "lru_cache": [
        {"pass_rate": 0.82, "complexity": "O(1)", "novelty": 0.9, "has_contradiction": False},
        {"pass_rate": 0.90, "complexity": "O(1)", "novelty": 0.4, "has_contradiction": False},
        {"pass_rate": 0.94, "complexity": "O(1)", "novelty": 0.2, "has_contradiction": False},
    ],
    "valid_parentheses": [
        {"pass_rate": 0.80, "complexity": "O(n)", "novelty": 0.5, "has_contradiction": False},
        {"pass_rate": 0.88, "complexity": "O(n)", "novelty": 0.2, "has_contradiction": False},
        {"pass_rate": 0.91, "complexity": "O(n)", "novelty": 0.1, "has_contradiction": False},
    ],
    "merge_intervals": [
        {"pass_rate": 0.83, "complexity": "O(n log n)", "novelty": 0.8, "has_contradiction": False},
        {"pass_rate": 0.90, "complexity": "O(n log n)", "novelty": 0.3, "has_contradiction": False},
        {"pass_rate": 0.93, "complexity": "O(n log n)", "novelty": 0.2, "has_contradiction": False},
    ],
}


def get_problems_for_difficulty(difficulty: str):
    """Return problem list for given difficulty tier."""
    return PROBLEMS.get(difficulty, PROBLEMS["easy"])


def make_synthetic_solution(problem_id: str, cycle: int, has_contradiction: bool) -> str:
    """
    Generate a synthetic solution string for simulation.
    When has_contradiction=True for two_sum, produces a genuine nested-loop
    solution that claims O(n) — the contradiction detector will catch this.
    """
    if problem_id == "two_sum" and has_contradiction:
        # Seeded contradiction: nested loop (O(n^2)) claiming O(n)
        return """```python
def two_sum(nums, target):
    # Time complexity: O(n)  <-- WRONG: this is actually O(n^2)
    result = []
    for i in range(len(nums)):
        for j in range(len(nums)):  # nested loop
            if nums[i] + nums[j] == target and i != j:
                result.append((i, j))
    return result
```
Time complexity: O(n)
assert two_sum([2, 7, 11, 15], 9) == [(0, 1)]"""

    if problem_id == "two_sum":
        return """```python
def two_sum(nums, target):
    # Time complexity: O(n)
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```
Time complexity: O(n)
assert two_sum([2, 7, 11, 15], 9) == [0, 1]"""

    # Generic solutions for other problems
    complexity_map = {
        "is_palindrome":    ("O(n)",       "return s == s[::-1]"),
        "max_subarray":     ("O(n)",       "# Kadane's algorithm\n    max_sum = cur = nums[0]\n    for n in nums[1:]:\n        cur = max(n, cur+n)\n        max_sum = max(max_sum, cur)\n    return max_sum"),
        "binary_search":    ("O(log n)",   "lo,hi=0,len(nums)-1\n    while lo<=hi:\n        mid=(lo+hi)//2\n        if nums[mid]==target: return mid\n        elif nums[mid]<target: lo=mid+1\n        else: hi=mid-1\n    return -1"),
        "flatten_nested":   ("O(n)",       "return [x for sub in lst for x in (flatten(sub) if isinstance(sub,list) else [sub])]"),
        "lru_cache":        ("O(1)",       "# OrderedDict-based LRU\n    pass"),
        "valid_parentheses": ("O(n)",      "stack=[]\n    for c in s:\n        if c in '([{': stack.append(c)\n        elif not stack: return False\n        else: stack.pop()\n    return not stack"),
        "merge_intervals":  ("O(n log n)", "intervals.sort()\n    merged=[]\n    for iv in intervals:\n        if merged and merged[-1][1]>=iv[0]: merged[-1][1]=max(merged[-1][1],iv[1])\n        else: merged.append(list(iv))\n    return merged"),
    }
    complexity, body = complexity_map.get(problem_id, ("O(n)", "pass"))
    return f"""```python
def solution(input):
    # {problem_id} — cycle {cycle+1}
    # Time complexity: {complexity}
    {body}
```
Time complexity: {complexity}"""


def run_simulation(num_cycles: int = 3, seed: int = 42) -> dict:
    random.seed(seed)

    field = "software_engineering"
    field_config = FIELD_CONFIGS[field]

    # Initialize all components
    assertions_store = AssertionsStore(confidence_threshold=0.5)
    arbiter = ArbiterAgent(assertions_store=assertions_store)
    scorer = UtilityScorer(arbiter=arbiter)
    detector = ContradictionDetector(penalty_multiplier=field_config.penalty_multiplier)
    personality = PersonalityManager()

    results = {
        "field": field,
        "cycles": [],
        "per_problem": {pid: [] for tier in PROBLEMS.values() for p in tier for pid in [p["id"]]},
        "personality_evolution": [],
        "arbiter_stats": [],
        "dpo_pairs": [],
    }

    print(f"\n{'='*60}")
    print(f"Adaptive Utility Agent — Simulation v0.4")
    print(f"Field: {field}  |  Cycles: {num_cycles}")
    print(f"{'='*60}\n")

    # Record initial personality
    results["personality_evolution"].append({
        "cycle": 0,
        "state": personality.get_trait_summary(),
    })

    for cycle in range(num_cycles):
        cycle_scores = []
        cycle_contradictions = 0
        cycle_dpo = []

        print(f"── Cycle {cycle+1} {'─'*45}")

        # Determine which problems to run based on current confidence
        domain_summary = scorer.get_domain_summary(field)
        current_confidence = domain_summary.get("confidence", 0.5)
        recommended_diff = scorer._recommended_difficulty(current_confidence)

        print(f"   Domain confidence: {current_confidence:.3f} → routing to '{recommended_diff}' problems")

        # Get problems for this difficulty tier
        tier_problems = get_problems_for_difficulty(recommended_diff)
        # Also include some problems from adjacent tiers for coverage
        all_problems_flat = [p for tier in PROBLEMS.values() for p in tier]

        for problem in all_problems_flat:
            pid = problem["id"]
            if pid not in CYCLE_RESPONSES or cycle >= len(CYCLE_RESPONSES[pid]):
                continue

            resp = CYCLE_RESPONSES[pid][cycle]

            # Synthetic solution
            solution = make_synthetic_solution(pid, cycle, resp["has_contradiction"])
            alt_solution = make_synthetic_solution(pid, cycle, not resp["has_contradiction"])

            # Contradiction detection
            cd_result = detector.check(
                problem=f"Solve {pid}",
                solution=solution,
                claimed_complexity=resp["complexity"],
            )

            # Arbiter check (comparing our solution vs alternative)
            arbiter_verdict = None
            gap_subject = None
            if cycle > 0:  # Arbiter active from cycle 2
                arbiter_verdict = arbiter.arbitrate(
                    subject=pid,
                    domain=field,
                    output_A=solution,
                    output_B=alt_solution,
                    field_penalty_multiplier=field_config.penalty_multiplier,
                    claimed_complexity_A=resp["complexity"],
                )
                if arbiter_verdict.case == VerdictCase.CASE_3:
                    gap_subject = pid
                    print(f"   ⚡ Case 3 gap bonus activated for '{pid}'")

            # Score interaction
            task_score = scorer.score(
                task_id=pid,
                field_config=field_config,
                test_pass_rate=resp["pass_rate"],
                human_baseline_score=problem["baseline"],
                contradiction_penalty=cd_result.confidence_penalty,
                problem_novelty=resp["novelty"],
                active_gap_subject=gap_subject,
            )

            if cd_result.contradictions:
                cycle_contradictions += 1
                for c in cd_result.contradictions:
                    cycle_dpo.append({
                        "task_id": pid,
                        "field": field,
                        "rejected": solution[:50],
                        "reason": c.description,
                        "weight": field_config.penalty_multiplier,
                    })

            cycle_scores.append(task_score)
            results["per_problem"][pid].append({
                "cycle": cycle + 1,
                "utility": task_score.utility,
                "efficacy_raw": task_score.efficacy,
                "efficacy_ema": task_score.efficacy_ema,
                "confidence": task_score.confidence,
                "gap_bonus": task_score.gap_bonus,
                "recommended_difficulty": task_score.recommended_difficulty,
            })

            print(
                f"   {pid:<22} U={task_score.utility:.4f} "
                f"E_ema={task_score.efficacy_ema:.4f} "
                f"C={task_score.confidence:.4f}"
                + (f" gap={task_score.gap_bonus:.3f}" if task_score.gap_bonus > 0 else "")
                + (" ⚠ CONTRADICTION" if cd_result.contradictions else "")
            )

        # Cycle summary
        avg_U = sum(s.utility for s in cycle_scores) / len(cycle_scores)
        avg_E_ema = sum(s.efficacy_ema for s in cycle_scores) / len(cycle_scores)
        avg_E_raw = sum(s.efficacy for s in cycle_scores) / len(cycle_scores)
        avg_C = sum(s.confidence for s in cycle_scores) / len(cycle_scores)

        print(f"\n   Cycle {cycle+1} summary:")
        print(f"     avg U     = {avg_U:.4f}")
        print(f"     avg E_raw = {avg_E_raw:.4f}  (per-interaction)")
        print(f"     avg E_ema = {avg_E_ema:.4f}  (accumulated — this is what drives U)")
        print(f"     avg C     = {avg_C:.4f}")
        print(f"     contradictions = {cycle_contradictions}")

        results["cycles"].append({
            "cycle": cycle + 1,
            "avg_U": round(avg_U, 4),
            "avg_E_raw": round(avg_E_raw, 4),
            "avg_E_ema": round(avg_E_ema, 4),
            "avg_C": round(avg_C, 4),
            "contradictions": cycle_contradictions,
        })

        results["dpo_pairs"].extend(cycle_dpo)
        results["arbiter_stats"].append({
            "cycle": cycle + 1,
            **arbiter.status(),
        })

        # Personality evolution
        utility_trend = scorer.get_utility_trend(field)
        domain_sum = scorer.get_domain_summary(field)
        personality.evolve(
            utility_history=utility_trend,
            contradiction_rate=domain_sum.get("contradiction_rate", 0.0),
            domain=field,
        )
        results["personality_evolution"].append({
            "cycle": cycle + 1,
            "state": personality.get_trait_summary(),
        })

        print()

    # Final summary
    print(f"{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    c1 = results["cycles"][0]
    cn = results["cycles"][-1]
    delta_U = cn["avg_U"] - c1["avg_U"]
    delta_E = cn["avg_E_ema"] - c1["avg_E_ema"]
    delta_C = cn["avg_C"] - c1["avg_C"]
    print(f"  Total U improvement:     {c1['avg_U']:.4f} → {cn['avg_U']:.4f}  ({delta_U:+.4f})")
    print(f"  Efficacy EMA change:     {c1['avg_E_ema']:.4f} → {cn['avg_E_ema']:.4f}  ({delta_E:+.4f})")
    print(f"  Confidence change:       {c1['avg_C']:.4f} → {cn['avg_C']:.4f}  ({delta_C:+.4f})")
    print(f"  DPO pairs accumulated:   {len(results['dpo_pairs'])}")
    print(f"  Assertions stored:       {assertions_store.summary()['total']}")
    print(f"  Arbiter verdicts:        {arbiter.total_verdicts}")
    print(f"  Arbiter correction rate: {arbiter.correction_rate():.1%}")
    print()

    return results


if __name__ == "__main__":
    results = run_simulation(num_cycles=3)

    out_path = os.path.join(os.path.dirname(__file__), "simulation_results_v04.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")
