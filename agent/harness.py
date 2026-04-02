"""
MVP harness: runs the UtilityAgent against LeetCode-style problems
and tracks utility score over time.

Usage:
    python harness.py

Plots utility trend after running all problems.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from agent import UtilityAgent


# ── Sample problem set ────────────────────────────────────────────────────────
# Format: (problem_description, human_baseline_score, problem_novelty)
# human_baseline_score: normalized score a typical developer achieves (0-1)
# problem_novelty: how novel/unexplored this problem type is for the agent (0-1)

PROBLEMS = [
    (
        """Write a Python function two_sum(nums: list[int], target: int) -> list[int]
        that returns indices of two numbers that add up to target.
        Use a hash map for O(n) time. Include test cases.""",
        0.85,   # humans do well on this classic problem
        0.3     # low novelty — very common problem
    ),
    (
        """Write a Python function is_valid_bst(root) -> bool that checks if a 
        binary tree is a valid BST. Use O(n) time and O(h) space where h is height.
        Define a minimal TreeNode class. Include test cases.""",
        0.75,
        0.5
    ),
    (
        """Write a Python function lru_cache(capacity: int) that implements an LRU 
        cache with get(key) and put(key, value) operations both in O(1) time.
        Use OrderedDict. Include test cases.""",
        0.65,   # harder — humans struggle more here
        0.6
    ),
    (
        """Write a Python function word_break(s: str, wordDict: list[str]) -> bool
        using dynamic programming. Return True if s can be segmented into words
        from wordDict. State the time and space complexity. Include test cases.""",
        0.60,
        0.65
    ),
    (
        """Write a Python function median_of_two_sorted_arrays(nums1, nums2) -> float
        that finds the median of two sorted arrays in O(log(m+n)) time.
        This is a hard problem. State your approach clearly. Include test cases.""",
        0.40,   # humans struggle significantly on this one
        0.8     # high novelty for the agent
    ),
]


async def run_harness():
    agent = UtilityAgent(evolution_interval=3)  # evolve every 3 for demo
    results = []

    print("=" * 60)
    print("UTILITY AGENT — MVP HARNESS")
    print("Field: Software Engineering")
    print("=" * 60)

    for i, (problem, human_baseline, novelty) in enumerate(PROBLEMS):
        print(f"\n{'─'*60}")
        print(f"Problem {i+1}/{len(PROBLEMS)}")
        print(f"Human baseline: {human_baseline:.0%}  |  Novelty: {novelty:.0%}")
        print(f"{'─'*60}")
        print(f"Task: {problem[:100]}...")

        response = await agent.respond(
            task=problem,
            human_baseline_score=human_baseline,
            problem_novelty=novelty,
            run_tests=True
        )

        if response.should_abstain:
            print(f"\n[ABSTAIN] {response.abstain_reason}")
        else:
            # Show first 300 chars of response
            preview = response.content[:300].replace("\n", " ")
            print(f"\nResponse preview: {preview}...")

        results.append({
            "problem": i + 1,
            "utility": response.score.utility,
            "efficacy": response.score.efficacy,
            "confidence": response.score.confidence,
            "curiosity": response.score.curiosity,
            "field": response.score.field,
            "below_minimum": response.score.below_minimum,
        })

        # Small delay to avoid rate limits
        await asyncio.sleep(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n{'Problem':>8} {'U':>7} {'E':>7} {'C':>7} {'K':>7}")
    print(f"{'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for r in results:
        flag = " ⚠" if r["below_minimum"] else ""
        print(
            f"{r['problem']:>8} "
            f"{r['utility']:>7.3f} "
            f"{r['efficacy']:>7.3f} "
            f"{r['confidence']:>7.3f} "
            f"{r['curiosity']:>7.3f}"
            f"{flag}"
        )

    utilities = [r["utility"] for r in results]
    print(f"\nUtility trend: {' → '.join(f'{u:.3f}' for u in utilities)}")
    print(f"Net change: {utilities[-1] - utilities[0]:+.3f}")

    print(f"\nPersonality traits after {len(PROBLEMS)} problems:")
    for name, score in agent.personality.get_trait_summary().items():
        bar = "█" * int(score * 20)
        print(f"  {name:<20} {score:.3f}  {bar}")

    print(f"\nDomain state (software_engineering):")
    summary = agent.scorer.get_domain_summary("software_engineering")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save results
    with open("/mnt/user-data/outputs/harness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to harness_results.json")


if __name__ == "__main__":
    asyncio.run(run_harness())
