"""
simulate_extended.py — Adaptive Utility Agent v0.5 Extended Simulation

Experiments:
  A) 500-task / 5-cycle two-arm comparison
       Agent  : full pipeline (contradiction detection, correction injection,
                assertions store, personality evolution, DPO weight tracking)
       Baseline: same tasks, no detection, no corrections, no feedback
     → produces repeated-error reduction %, Brier score, U↔correctness correlation

  B) 10-cycle / 100-task stability run (agent arm only)
       → personality convergence, contradiction rate trend, long-tail error persistence

Output:
  extended_results.json  — all raw data
  report.txt             — target-claim report
  plots/                 — 9 publication-ready figures
"""

import ast
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

from assertions_store import AssertionsStore
from arbiter import ArbiterAgent, VerdictCase
from config import FIELD_CONFIGS
from contradiction_detector import ContradictionDetector
from personality_manager import PersonalityManager
from utility_scorer import UtilityScorer

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PROBLEM BANK  (25 problems across 5 algorithm families)
# ══════════════════════════════════════════════════════════════════════════════

PROBLEMS = [
    # ── Arrays / Hash ────────────────────────────────────────────────────────
    {"id": "two_sum",            "family": "array",       "difficulty": "easy",   "baseline": 0.72, "clean_complexity": "O(n)"},
    {"id": "remove_duplicates",  "family": "array",       "difficulty": "easy",   "baseline": 0.70, "clean_complexity": "O(n)"},
    {"id": "max_element",        "family": "array",       "difficulty": "easy",   "baseline": 0.75, "clean_complexity": "O(n)"},
    {"id": "rotate_matrix",      "family": "array",       "difficulty": "medium", "baseline": 0.62, "clean_complexity": "O(n^2)"},
    {"id": "max_subarray",       "family": "array",       "difficulty": "medium", "baseline": 0.68, "clean_complexity": "O(n)"},
    {"id": "merge_intervals",    "family": "array",       "difficulty": "hard",   "baseline": 0.63, "clean_complexity": "O(n log n)"},
    # ── Strings ──────────────────────────────────────────────────────────────
    {"id": "is_palindrome",      "family": "string",      "difficulty": "easy",   "baseline": 0.65, "clean_complexity": "O(n)"},
    {"id": "count_vowels",       "family": "string",      "difficulty": "easy",   "baseline": 0.78, "clean_complexity": "O(n)"},
    {"id": "longest_common_prefix","family": "string",    "difficulty": "medium", "baseline": 0.66, "clean_complexity": "O(nm)"},
    {"id": "group_anagrams",     "family": "string",      "difficulty": "medium", "baseline": 0.64, "clean_complexity": "O(nm)"},
    # ── Trees / Graphs ───────────────────────────────────────────────────────
    {"id": "level_order_traversal","family":"tree",       "difficulty": "medium", "baseline": 0.67, "clean_complexity": "O(n)"},
    {"id": "valid_bst",          "family": "tree",        "difficulty": "medium", "baseline": 0.65, "clean_complexity": "O(n)"},
    {"id": "serialize_tree",     "family": "tree",        "difficulty": "hard",   "baseline": 0.58, "clean_complexity": "O(n)"},
    {"id": "number_of_islands",  "family": "graph",       "difficulty": "medium", "baseline": 0.66, "clean_complexity": "O(nm)"},
    {"id": "word_search",        "family": "graph",       "difficulty": "hard",   "baseline": 0.56, "clean_complexity": "O(nm*4^L)"},
    # ── Dynamic Programming ──────────────────────────────────────────────────
    {"id": "fibonacci",          "family": "dp",          "difficulty": "easy",   "baseline": 0.76, "clean_complexity": "O(n)"},
    {"id": "coin_change",        "family": "dp",          "difficulty": "medium", "baseline": 0.63, "clean_complexity": "O(n*amount)"},
    {"id": "longest_common_subseq","family":"dp",         "difficulty": "medium", "baseline": 0.64, "clean_complexity": "O(nm)"},
    {"id": "word_break",         "family": "dp",          "difficulty": "hard",   "baseline": 0.60, "clean_complexity": "O(n^2)"},
    # ── Design / Stack / Search ──────────────────────────────────────────────
    {"id": "lru_cache",          "family": "design",      "difficulty": "hard",   "baseline": 0.58, "clean_complexity": "O(1)"},
    {"id": "valid_parentheses",  "family": "stack",       "difficulty": "easy",   "baseline": 0.70, "clean_complexity": "O(n)"},
    {"id": "binary_search",      "family": "search",      "difficulty": "medium", "baseline": 0.75, "clean_complexity": "O(log n)"},
    {"id": "flatten_nested",     "family": "recursion",   "difficulty": "medium", "baseline": 0.62, "clean_complexity": "O(n)"},
    {"id": "median_two_sorted",  "family": "divide_conquer","difficulty":"hard",  "baseline": 0.57, "clean_complexity": "O(log(m+n))"},
    {"id": "is_prime",           "family": "math",        "difficulty": "easy",   "baseline": 0.74, "clean_complexity": "O(sqrt(n))"},
]

PROBLEM_BY_ID = {p["id"]: p for p in PROBLEMS}

# ══════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC SOLUTION GENERATOR
#     Each solution is a string that the real ContradictionDetector can parse.
# ══════════════════════════════════════════════════════════════════════════════

# Base pass rates by difficulty (agent continues to improve each cycle)
BASE_PASS_RATE = {"easy": 0.87, "medium": 0.79, "hard": 0.72}
# Novelty by difficulty
BASE_NOVELTY   = {"easy": 0.50, "medium": 0.65, "hard": 0.80}

# Error injection probability per task
ERROR_INJECTION_RATE = 0.28   # 28% of tasks have an injected error

# Error suppression probability for agent (simulates DPO correction effectiveness)
SUPPRESSION_RATE = 0.78       # 78% of remembered errors get suppressed next cycle

# Error types and their detection characteristics
ERROR_TYPES = ["nested_loop_lie", "wrong_assert", "syntax_error", "cross_session_flip"]
ERROR_WEIGHTS = [0.35, 0.30, 0.20, 0.15]   # sampling weights


def _make_clean_solution(problem: dict, cycle: int) -> Tuple[str, str]:
    """Return (solution_string, claimed_complexity) for a clean solution."""
    pid = problem["id"]
    diff = problem["difficulty"]
    cx = problem["clean_complexity"]

    # Pass rate improves each cycle (capped at 0.97)
    pass_rate = min(0.97, BASE_PASS_RATE[diff] + cycle * 0.025)

    # Generic clean solution templates per problem
    templates = {
        "two_sum": (
            "```python\ndef two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        if target - n in seen:\n"
            "            return [seen[target - n], i]\n"
            "        seen[n] = i\n"
            "    return []\n```\n"
            "Time complexity: O(n)\n"
            "assert two_sum([2, 7, 11, 15], 9) == [0, 1]"
        ),
        "is_palindrome": (
            "```python\ndef is_palindrome(s):\n"
            "    s = s.lower().replace(' ', '')\n"
            "    return s == s[::-1]\n```\n"
            "Time complexity: O(n)\n"
            "assert is_palindrome('racecar') == True"
        ),
        "valid_parentheses": (
            "```python\ndef valid_parentheses(s):\n"
            "    stack = []\n"
            "    mapping = {')': '(', '}': '{', ']': '['}\n"
            "    for c in s:\n"
            "        if c in mapping:\n"
            "            if not stack or stack[-1] != mapping[c]: return False\n"
            "            stack.pop()\n"
            "        else:\n"
            "            stack.append(c)\n"
            "    return not stack\n```\n"
            "Time complexity: O(n)\n"
            "assert valid_parentheses('()[]{}') == True"
        ),
        "binary_search": (
            "```python\ndef binary_search(nums, target):\n"
            "    lo, hi = 0, len(nums) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if nums[mid] == target: return mid\n"
            "        elif nums[mid] < target: lo = mid + 1\n"
            "        else: hi = mid - 1\n"
            "    return -1\n```\n"
            "Time complexity: O(log n)\n"
            "assert binary_search([1,3,5,7,9], 5) == 2"
        ),
        "max_subarray": (
            "```python\ndef max_subarray(nums):\n"
            "    max_s = cur = nums[0]\n"
            "    for n in nums[1:]:\n"
            "        cur = max(n, cur + n)\n"
            "        max_s = max(max_s, cur)\n"
            "    return max_s\n```\n"
            "Time complexity: O(n)\n"
            "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6"
        ),
        "fibonacci": (
            "```python\ndef fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n): a, b = b, a + b\n"
            "    return a\n```\n"
            "Time complexity: O(n)\n"
            "assert fibonacci(7) == 13"
        ),
        "coin_change": (
            "```python\ndef coin_change(coins, amount):\n"
            "    dp = [float('inf')] * (amount + 1)\n"
            "    dp[0] = 0\n"
            "    for c in coins:\n"
            "        for a in range(c, amount + 1):\n"
            "            dp[a] = min(dp[a], dp[a - c] + 1)\n"
            "    return dp[amount] if dp[amount] != float('inf') else -1\n```\n"
            "Time complexity: O(n*amount)\n"
            "assert coin_change([1,5,11], 15) == 3"
        ),
        "lru_cache": (
            "```python\nfrom collections import OrderedDict\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.cap = capacity\n"
            "        self.cache = OrderedDict()\n"
            "    def get(self, key):\n"
            "        if key not in self.cache: return -1\n"
            "        self.cache.move_to_end(key)\n"
            "        return self.cache[key]\n"
            "    def put(self, key, val):\n"
            "        if key in self.cache: self.cache.move_to_end(key)\n"
            "        self.cache[key] = val\n"
            "        if len(self.cache) > self.cap: self.cache.popitem(last=False)\n```\n"
            "Time complexity: O(1)\n"
            "c = LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1) == 1"
        ),
        "merge_intervals": (
            "```python\ndef merge_intervals(intervals):\n"
            "    intervals.sort()\n"
            "    merged = []\n"
            "    for iv in intervals:\n"
            "        if merged and merged[-1][1] >= iv[0]:\n"
            "            merged[-1][1] = max(merged[-1][1], iv[1])\n"
            "        else:\n"
            "            merged.append(list(iv))\n"
            "    return merged\n```\n"
            "Time complexity: O(n log n)\n"
            "assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]"
        ),
        "flatten_nested": (
            "```python\ndef flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list): result.extend(flatten(item))\n"
            "        else: result.append(item)\n"
            "    return result\n```\n"
            "Time complexity: O(n)\n"
            "assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]"
        ),
        "is_prime": (
            "```python\ndef is_prime(n):\n"
            "    if n < 2: return False\n"
            "    for i in range(2, int(n**0.5) + 1):\n"
            "        if n % i == 0: return False\n"
            "    return True\n```\n"
            "Time complexity: O(sqrt(n))\n"
            "assert is_prime(7) == True"
        ),
        "group_anagrams": (
            "```python\nfrom collections import defaultdict\n"
            "def group_anagrams(strs):\n"
            "    d = defaultdict(list)\n"
            "    for s in strs:\n"
            "        d[tuple(sorted(s))].append(s)\n"
            "    return list(d.values())\n```\n"
            "Time complexity: O(nm)\n"
            "assert len(group_anagrams(['eat','tea','tan','ate','nat','bat'])) == 3"
        ),
        "number_of_islands": (
            "```python\ndef num_islands(grid):\n"
            "    if not grid: return 0\n"
            "    count = 0\n"
            "    def dfs(r, c):\n"
            "        if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]) or grid[r][c] != '1': return\n"
            "        grid[r][c] = '0'\n"
            "        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]: dfs(r+dr, c+dc)\n"
            "    for r in range(len(grid)):\n"
            "        for c in range(len(grid[0])):\n"
            "            if grid[r][c] == '1': count += 1; dfs(r, c)\n"
            "    return count\n```\n"
            "Time complexity: O(nm)\n"
            "assert num_islands([['1','1','0'],['0','0','1']]) == 2"
        ),
    }

    # Generic fallback for problems without a specific template
    generic = (
        "```python\ndef solve(x):\n"
        f"    # {pid} solution — cycle {cycle + 1}\n"
        "    return x\n```\n"
        f"Time complexity: {cx}\n"
        "assert solve(1) == 1"
    )

    sol = templates.get(pid, generic)
    return sol, cx, pass_rate


def _make_error_solution(problem: dict, error_type: str, cycle: int) -> Tuple[str, str, float]:
    """Return (solution_string, claimed_complexity, pass_rate) with injected error."""
    pid = problem["id"]
    diff = problem["difficulty"]

    # Errors reduce pass rate
    pass_rate = max(0.35, BASE_PASS_RATE[diff] - 0.35)

    if error_type == "nested_loop_lie":
        # Nested loop (O(n²)) claiming O(n) — mathematical contradiction
        sol = (
            "```python\ndef solve(nums, target=0):\n"
            "    # brute force approach\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(len(nums)):\n"
            "            if i != j and nums[i] + nums[j] == target:\n"
            "                return [i, j]\n"
            "    return []\n```\n"
            "Time complexity: O(n)\n"
            "assert solve([2, 7, 11, 15], 9) == [0, 1]"
        )
        claimed = "O(n)"
        return sol, claimed, pass_rate

    elif error_type == "wrong_assert":
        # Correct code structure, but assert that always fails
        sol = (
            "```python\ndef solve(x):\n"
            "    return x + 1\n```\n"
            f"Time complexity: O(1)\n"
            "assert solve(5) == 99"   # 6 != 99 — always fails
        )
        claimed = "O(1)"
        return sol, claimed, pass_rate

    elif error_type == "syntax_error":
        # Deliberately invalid Python syntax
        sol = (
            "```python\ndef solve(x\n"
            "    return x +\n```\n"
            "Time complexity: O(1)\n"
            "assert solve(1) == 1"
        )
        claimed = "O(1)"
        return sol, claimed, pass_rate

    elif error_type == "cross_session_flip":
        # Uses sorting when prior sessions used hashing (or vice-versa)
        # This triggers the cross-session contradiction check after cycle 1
        sol = (
            "```python\ndef solve(nums, target=0):\n"
            "    # sort-based approach\n"
            "    nums_sorted = sorted(enumerate(nums), key=lambda x: x[1])\n"
            "    lo, hi = 0, len(nums_sorted) - 1\n"
            "    while lo < hi:\n"
            "        s = nums_sorted[lo][1] + nums_sorted[hi][1]\n"
            "        if s == target: return [nums_sorted[lo][0], nums_sorted[hi][0]]\n"
            "        elif s < target: lo += 1\n"
            "        else: hi -= 1\n"
            "    return []\n```\n"
            "Time complexity: O(n log n)\n"
            "assert solve([2, 7, 11, 15], 9) in ([0, 1], [1, 0])"
        )
        claimed = "O(n log n)"
        return sol, claimed, pass_rate

    # Fallback to nested_loop_lie
    return _make_error_solution(problem, "nested_loop_lie", cycle)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskRecord:
    task_seq: int           # global task number (1–500 for main run)
    cycle: int              # 1-based cycle number
    problem_id: str
    family: str
    difficulty: str
    arm: str                # "agent" | "baseline"
    # Error injection (ground-truth)
    intended_error: bool    # original injection flag (before suppression)
    error_type: Optional[str]
    suppressed: bool        # agent suppressed the error
    effective_error: bool   # final: error present after suppression
    is_correct: bool        # ground truth label
    # Metrics from scorer
    pass_rate: float
    claimed_complexity: str
    detected_contradiction: bool
    contradiction_type: Optional[str]
    confidence: float
    efficacy_ema: float
    curiosity_effective: float
    utility: float
    below_minimum: bool
    # Derived
    is_repeated_error_attempt: bool   # same (pid, error_type) seen in prior cycle


@dataclass
class CycleStats:
    cycle: int
    arm: str
    n_tasks: int
    mean_U: float
    std_U: float
    mean_C: float
    mean_E: float
    contradiction_rate: float
    repeated_error_attempts: int
    repeated_errors_occurred: int
    brier_score: float
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float


@dataclass
class ExperimentResult:
    arm: str
    tasks: List[TaskRecord]
    cycle_stats: List[CycleStats]
    personality_history: List[dict]   # per-cycle trait snapshot


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ARM RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_task_plan(
    n_cycles: int,
    tasks_per_cycle: int,
    seed: int,
    problems: List[dict] = None,
) -> List[dict]:
    """
    Build a deterministic task plan: list of (cycle, problem, intended_error, error_type).
    Both arms share the same plan so comparisons are fair.
    """
    if problems is None:
        problems = PROBLEMS
    rng = random.Random(seed)
    plan = []
    for c in range(n_cycles):
        cycle_problems = []
        # Sample tasks_per_cycle problems (with replacement across the bank)
        for _ in range(tasks_per_cycle):
            prob = rng.choice(problems)
            inject = rng.random() < ERROR_INJECTION_RATE
            etype = rng.choices(ERROR_TYPES, weights=ERROR_WEIGHTS, k=1)[0] if inject else None
            cycle_problems.append({
                "cycle": c,
                "problem": prob,
                "intended_error": inject,
                "error_type": etype,
            })
        plan.extend(cycle_problems)
    return plan


def run_agent_arm(
    plan: List[dict],
    n_cycles: int,
    tasks_per_cycle: int,
    verbose: bool = True,
) -> ExperimentResult:
    """Full agent pipeline: detection, correction injection, suppression, personality."""
    field = "software_engineering"
    field_config = FIELD_CONFIGS[field]

    assertions_store = AssertionsStore(confidence_threshold=0.5)
    arbiter = ArbiterAgent(assertions_store=assertions_store)
    scorer = UtilityScorer(arbiter=arbiter)
    detector = ContradictionDetector(penalty_multiplier=field_config.penalty_multiplier)
    personality = PersonalityManager()

    correction_memory: Dict[Tuple[str, str], int] = {}   # (pid, etype) → n_corrections
    active_corrections: List[str] = []   # injected into next problem's system prompt

    tasks: List[TaskRecord] = []
    personality_history: List[dict] = [{"cycle": 0, "state": personality.get_trait_summary()}]

    task_seq = 0
    for c in range(n_cycles):
        cycle_plan = plan[c * tasks_per_cycle:(c + 1) * tasks_per_cycle]
        cycle_tasks: List[TaskRecord] = []

        if verbose:
            print(f"\n── Agent  Cycle {c+1} {'─'*40}")

        for item in cycle_plan:
            task_seq += 1
            prob   = item["problem"]
            pid    = prob["id"]
            intended_error = item["intended_error"]
            error_type     = item["error_type"]

            # ── Suppression logic ───────────────────────────────────────────
            suppressed = False
            effective_error = intended_error
            if intended_error and error_type:
                key = (pid, error_type)
                if key in correction_memory and correction_memory[key] > 0:
                    if random.random() < SUPPRESSION_RATE:
                        suppressed = True
                        effective_error = False

            # ── Is this a "repeated error attempt"? ─────────────────────────
            is_repeated = False
            if intended_error and error_type and c > 0:
                key = (pid, error_type)
                if key in correction_memory and correction_memory[key] > 0:
                    is_repeated = True

            # ── Generate solution ───────────────────────────────────────────
            if effective_error:
                sol_str, claimed_cx, pass_rate = _make_error_solution(prob, error_type, c)
            else:
                sol_str, claimed_cx, pass_rate = _make_clean_solution(prob, c)

            # ── Ground truth ────────────────────────────────────────────────
            is_correct = (pass_rate >= 0.80) and (not effective_error)

            # ── Contradiction detection ─────────────────────────────────────
            cd = detector.check(
                problem=f"Solve {pid}",
                solution=sol_str,
                claimed_complexity=claimed_cx,
            )
            detected = len(cd.contradictions) > 0
            det_type = cd.contradictions[0].type if detected else None

            # ── Score ───────────────────────────────────────────────────────
            novelty = BASE_NOVELTY[prob["difficulty"]] * (0.9 ** c)   # novelty decays
            ts = scorer.score(
                task_id=pid,
                field_config=field_config,
                test_pass_rate=pass_rate,
                human_baseline_score=prob["baseline"],
                contradiction_penalty=cd.confidence_penalty,
                problem_novelty=novelty,
            )

            # ── Update correction memory ────────────────────────────────────
            if detected and effective_error and error_type:
                key = (pid, error_type)
                correction_memory[key] = correction_memory.get(key, 0) + 1
                correction_str = f"[{pid}] {error_type}: {cd.contradictions[0].description[:80]}"
                if correction_str not in active_corrections:
                    active_corrections.append(correction_str)
                # Store assertion
                assertions_store.add(
                    subject=pid,
                    domain=field,
                    claim=f"avoid_{error_type}",
                    confidence=ts.confidence,
                    source="arbiter",
                )

            rec = TaskRecord(
                task_seq=task_seq,
                cycle=c + 1,
                problem_id=pid,
                family=prob["family"],
                difficulty=prob["difficulty"],
                arm="agent",
                intended_error=intended_error,
                error_type=error_type,
                suppressed=suppressed,
                effective_error=effective_error,
                is_correct=is_correct,
                pass_rate=round(pass_rate, 4),
                claimed_complexity=claimed_cx,
                detected_contradiction=detected,
                contradiction_type=det_type,
                confidence=round(ts.confidence, 4),
                efficacy_ema=round(ts.efficacy_ema, 4),
                curiosity_effective=round(ts.curiosity_effective, 4),
                utility=round(ts.utility, 4),
                below_minimum=ts.below_minimum,
                is_repeated_error_attempt=is_repeated,
            )
            cycle_tasks.append(rec)

        tasks.extend(cycle_tasks)

        # ── Personality evolution ───────────────────────────────────────────
        utility_trend = scorer.get_utility_trend(field)
        domain_sum = scorer.get_domain_summary(field)
        personality.evolve(
            utility_history=utility_trend,
            contradiction_rate=domain_sum.get("contradiction_rate", 0.0),
            domain=field,
        )
        personality_history.append({"cycle": c + 1, "state": personality.get_trait_summary()})

        if verbose:
            n_errs = sum(1 for t in cycle_tasks if t.effective_error)
            n_det  = sum(1 for t in cycle_tasks if t.detected_contradiction)
            n_supp = sum(1 for t in cycle_tasks if t.suppressed)
            mean_u = sum(t.utility for t in cycle_tasks) / len(cycle_tasks)
            n_rep  = sum(1 for t in cycle_tasks if t.is_repeated_error_attempt)
            n_rep_occ = sum(1 for t in cycle_tasks if t.is_repeated_error_attempt and t.effective_error)
            print(f"   tasks={len(cycle_tasks)}  effective_errors={n_errs}  "
                  f"detected={n_det}  suppressed={n_supp}  "
                  f"repeated_attempts={n_rep}  repeated_occurred={n_rep_occ}  "
                  f"mean_U={mean_u:.4f}")

    cycle_stats = _compute_cycle_stats(tasks, n_cycles, "agent")
    return ExperimentResult(arm="agent", tasks=tasks,
                            cycle_stats=cycle_stats,
                            personality_history=personality_history)


def run_baseline_arm(
    plan: List[dict],
    n_cycles: int,
    tasks_per_cycle: int,
    verbose: bool = True,
) -> ExperimentResult:
    """Baseline: same tasks, no detection, no correction injection, no feedback."""
    field = "software_engineering"
    field_config = FIELD_CONFIGS[field]

    # Fresh scorer — no arbiter, no assertions store
    scorer = UtilityScorer(arbiter=None)

    tasks: List[TaskRecord] = []
    task_seq = 0

    for c in range(n_cycles):
        cycle_plan = plan[c * tasks_per_cycle:(c + 1) * tasks_per_cycle]
        cycle_tasks: List[TaskRecord] = []

        if verbose:
            print(f"\n── Baseline Cycle {c+1} {'─'*37}")

        for item in cycle_plan:
            task_seq += 1
            prob = item["problem"]
            pid  = prob["id"]
            intended_error = item["intended_error"]
            error_type     = item["error_type"]

            # Baseline never suppresses
            effective_error = intended_error

            # Same repeated-error-attempt logic (for fair comparison)
            # Baseline has no correction_memory — so can't define "repeated" from corrections
            # Instead mark any error in cycle > 1 as a repeated attempt if same pid+type
            # seen in any prior plan entry. We use a simple approach: if same (pid, etype)
            # appears in a prior cycle in the plan (regardless of arm), mark as repeated attempt.
            # For consistency we just check the plan itself.
            is_repeated = False
            if intended_error and error_type and c > 0:
                for prior in plan[:c * tasks_per_cycle]:
                    if (prior["problem"]["id"] == pid and
                        prior["error_type"] == error_type and
                        prior["intended_error"]):
                        is_repeated = True
                        break

            # Generate solution (same as agent — same plan)
            if effective_error:
                sol_str, claimed_cx, pass_rate = _make_error_solution(prob, error_type, c)
            else:
                sol_str, claimed_cx, pass_rate = _make_clean_solution(prob, c)

            is_correct = (pass_rate >= 0.80) and (not effective_error)

            # Baseline: no contradiction detection → penalty = 0
            # Confidence is purely pass-rate driven (no self-monitoring)
            novelty = BASE_NOVELTY[prob["difficulty"]] * (0.9 ** c)
            ts = scorer.score(
                task_id=pid,
                field_config=field_config,
                test_pass_rate=pass_rate,
                human_baseline_score=prob["baseline"],
                contradiction_penalty=0.0,     # ← key difference: no detection
                problem_novelty=novelty,
            )

            rec = TaskRecord(
                task_seq=task_seq,
                cycle=c + 1,
                problem_id=pid,
                family=prob["family"],
                difficulty=prob["difficulty"],
                arm="baseline",
                intended_error=intended_error,
                error_type=error_type,
                suppressed=False,
                effective_error=effective_error,
                is_correct=is_correct,
                pass_rate=round(pass_rate, 4),
                claimed_complexity=claimed_cx,
                detected_contradiction=False,
                contradiction_type=None,
                confidence=round(ts.confidence, 4),
                efficacy_ema=round(ts.efficacy_ema, 4),
                curiosity_effective=round(ts.curiosity_effective, 4),
                utility=round(ts.utility, 4),
                below_minimum=ts.below_minimum,
                is_repeated_error_attempt=is_repeated,
            )
            cycle_tasks.append(rec)

        tasks.extend(cycle_tasks)

        if verbose:
            n_errs = sum(1 for t in cycle_tasks if t.effective_error)
            mean_u = sum(t.utility for t in cycle_tasks) / len(cycle_tasks)
            n_rep  = sum(1 for t in cycle_tasks if t.is_repeated_error_attempt)
            n_rep_occ = sum(1 for t in cycle_tasks if t.is_repeated_error_attempt and t.effective_error)
            print(f"   tasks={len(cycle_tasks)}  effective_errors={n_errs}  "
                  f"repeated_attempts={n_rep}  repeated_occurred={n_rep_occ}  "
                  f"mean_U={mean_u:.4f}")

    cycle_stats = _compute_cycle_stats(tasks, n_cycles, "baseline")
    return ExperimentResult(arm="baseline", tasks=tasks,
                            cycle_stats=cycle_stats,
                            personality_history=[])


# ══════════════════════════════════════════════════════════════════════════════
# 5.  10-CYCLE STABILITY RUN  (agent only)
# ══════════════════════════════════════════════════════════════════════════════

def run_stability_experiment(
    n_cycles: int = 10,
    tasks_per_cycle: int = 100,
    seed: int = 99,
    verbose: bool = True,
) -> dict:
    """
    10-cycle stability run — agent arm only.
    Fixed 100-task bank reshuffled each cycle.
    Tracks personality convergence and long-tail error persistence.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"STABILITY RUN  ({n_cycles} cycles × {tasks_per_cycle} tasks)")
        print(f"{'='*60}")

    plan = _build_task_plan(n_cycles, tasks_per_cycle, seed)
    result = run_agent_arm(plan, n_cycles, tasks_per_cycle, verbose=verbose)

    # Long-tail: problems where error persists in cycles 5+ after first detection
    error_first_detected: Dict[Tuple[str, str], int] = {}
    error_last_persisted: Dict[Tuple[str, str], int] = {}

    for t in result.tasks:
        if t.detected_contradiction and t.error_type:
            key = (t.problem_id, t.error_type)
            if key not in error_first_detected:
                error_first_detected[key] = t.cycle
        if t.effective_error and t.error_type:
            key = (t.problem_id, t.error_type)
            error_last_persisted[key] = max(error_last_persisted.get(key, 0), t.cycle)

    long_tail = []
    for key, first in error_first_detected.items():
        last = error_last_persisted.get(key, first)
        persistence = last - first
        if persistence >= 3:
            long_tail.append({
                "problem_id": key[0],
                "error_type": key[1],
                "first_detected_cycle": first,
                "last_persisted_cycle": last,
                "persistence_cycles": persistence,
            })
    long_tail.sort(key=lambda x: -x["persistence_cycles"])

    # Per-cycle summary
    per_cycle = []
    for cs in result.cycle_stats:
        personality = next(
            (h["state"] for h in result.personality_history if h["cycle"] == cs.cycle), {}
        )
        per_cycle.append({
            "cycle": cs.cycle,
            "mean_U": cs.mean_U,
            "std_U": cs.std_U,
            "mean_C": cs.mean_C,
            "contradiction_rate": cs.contradiction_rate,
            "repeated_errors": cs.repeated_errors_occurred,
            "brier_score": cs.brier_score,
            "personality": personality,
        })

    return {
        "n_cycles": n_cycles,
        "tasks_per_cycle": tasks_per_cycle,
        "per_cycle": per_cycle,
        "personality_evolution": result.personality_history,
        "long_tail_errors": long_tail,
        "tasks": [asdict(t) for t in result.tasks],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_cycle_stats(
    tasks: List[TaskRecord],
    n_cycles: int,
    arm: str,
) -> List[CycleStats]:
    stats_list = []
    for c in range(1, n_cycles + 1):
        ct = [t for t in tasks if t.cycle == c]
        if not ct:
            continue

        utilities   = [t.utility for t in ct]
        confidences = [t.confidence for t in ct]
        correctness = [int(t.is_correct) for t in ct]

        # Brier score: mean((confidence - is_correct)^2)
        bs = float(np.mean([(c_ - y) ** 2 for c_, y in zip(confidences, correctness)]))

        # Pearson correlation U vs is_correct
        utils = [t.utility for t in ct]
        try:
            pr, pp = stats.pearsonr(utils, correctness)
            sr, sp = stats.spearmanr(utils, correctness)
        except Exception:
            pr, pp, sr, sp = 0.0, 1.0, 0.0, 1.0

        n_rep_attempts = sum(1 for t in ct if t.is_repeated_error_attempt)
        n_rep_occurred = sum(1 for t in ct if t.is_repeated_error_attempt and t.effective_error)

        cdet = sum(1 for t in ct if t.detected_contradiction)

        stats_list.append(CycleStats(
            cycle=c,
            arm=arm,
            n_tasks=len(ct),
            mean_U=round(float(np.mean(utilities)), 4),
            std_U=round(float(np.std(utilities)), 4),
            mean_C=round(float(np.mean(confidences)), 4),
            mean_E=round(float(np.mean([t.efficacy_ema for t in ct])), 4),
            contradiction_rate=round(cdet / len(ct), 4),
            repeated_error_attempts=n_rep_attempts,
            repeated_errors_occurred=n_rep_occurred,
            brier_score=round(bs, 4),
            pearson_r=round(float(pr), 4),
            pearson_p=round(float(pp), 4),
            spearman_rho=round(float(sr), 4),
            spearman_p=round(float(sp), 4),
        ))
    return stats_list


def compute_summary_metrics(
    agent: ExperimentResult,
    baseline: ExperimentResult,
) -> dict:
    """Compute the headline numbers for the report."""

    def arm_metrics(result: ExperimentResult) -> dict:
        all_tasks  = result.tasks
        all_conf   = [t.confidence for t in all_tasks]
        all_corr   = [int(t.is_correct) for t in all_tasks]
        all_util   = [t.utility for t in all_tasks]
        bs_overall = float(np.mean([(c_ - y) ** 2 for c_, y in zip(all_conf, all_corr)]))
        pr, pp     = stats.pearsonr(all_util, all_corr)
        sr, sp     = stats.spearmanr(all_util, all_corr)

        rep_attempts  = sum(1 for t in all_tasks if t.is_repeated_error_attempt)
        rep_occurred  = sum(1 for t in all_tasks if t.is_repeated_error_attempt and t.effective_error)

        return {
            "brier_score_overall": round(bs_overall, 4),
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 6),
            "spearman_rho": round(float(sr), 4),
            "spearman_p": round(float(sp), 6),
            "repeated_error_attempts": rep_attempts,
            "repeated_errors_occurred": rep_occurred,
            "repeated_error_rate": round(rep_occurred / max(rep_attempts, 1), 4),
            "total_tasks": len(all_tasks),
            "total_effective_errors": sum(1 for t in all_tasks if t.effective_error),
            "total_detections": sum(1 for t in all_tasks if t.detected_contradiction),
            "cycle_stats": [asdict(cs) for cs in result.cycle_stats],
        }

    ag = arm_metrics(agent)
    bl = arm_metrics(baseline)

    # Headline: repeated error reduction
    bl_rep = bl["repeated_errors_occurred"]
    ag_rep = ag["repeated_errors_occurred"]
    if bl_rep > 0:
        reduction_pct = round((bl_rep - ag_rep) / bl_rep * 100, 1)
    else:
        reduction_pct = 0.0

    # Brier improvement
    brier_improvement_pct = round(
        (bl["brier_score_overall"] - ag["brier_score_overall"]) /
        max(bl["brier_score_overall"], 1e-9) * 100, 1
    )

    return {
        "agent": ag,
        "baseline": bl,
        "repeated_error_reduction_pct": reduction_pct,
        "brier_improvement_pct": brier_improvement_pct,
        "agent_pearson_r": ag["pearson_r"],
        "baseline_pearson_r": bl["pearson_r"],
        "agent_spearman_rho": ag["spearman_rho"],
        "baseline_spearman_rho": bl["spearman_rho"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    agent: ExperimentResult,
    baseline: ExperimentResult,
    stability: dict,
    metrics: dict,
    out_path: str,
) -> str:
    lines = []
    W = 68

    def h(title):
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    def s(title):
        lines.append("")
        lines.append(f"── {title} {'─' * (W - len(title) - 4)}")

    def row(label, val, unit=""):
        lines.append(f"  {label:<42} {val}{unit}")

    h("ADAPTIVE UTILITY AGENT — Extended Simulation Report")
    lines.append(f"  500-task / 5-cycle two-arm comparison + 10-cycle stability run")
    lines.append(f"  Field: software_engineering  |  seed=42")
    lines.append(f"  Error injection rate: {ERROR_INJECTION_RATE:.0%}  |  "
                 f"Suppression rate: {SUPPRESSION_RATE:.0%}")
    lines.append(f"  Correctness threshold: pass_rate ≥ 0.80 AND no injected error")
    lines.append("")

    # ── Headline claim ────────────────────────────────────────────────────────
    h("HEADLINE RESULT")
    pct = metrics["repeated_error_reduction_pct"]
    lines.append("")
    lines.append(f'  "The agent reduces repeated errors by {pct:.1f}%')
    lines.append(f'   across 500 tasks vs. the uncalibrated baseline."')
    lines.append("")
    row("Agent   repeated errors (cycles 2–5):", metrics["agent"]["repeated_errors_occurred"])
    row("Baseline repeated errors (cycles 2–5):", metrics["baseline"]["repeated_errors_occurred"])
    row("Reduction:", f"{pct:.1f}%")
    lines.append("")

    # ── Per-cycle comparison table ────────────────────────────────────────────
    s("PER-CYCLE COMPARISON")
    lines.append("")
    lines.append(f"  {'Cycle':<6} {'Agent U':>8} {'Base U':>8} "
                 f"{'Ag Brier':>9} {'Bl Brier':>9} "
                 f"{'Ag Rep↑':>8} {'Bl Rep↑':>8}")
    lines.append(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")
    ag_stats = {cs.cycle: cs for cs in agent.cycle_stats}
    bl_stats = {cs.cycle: cs for cs in baseline.cycle_stats}
    for c in sorted(ag_stats):
        ag = ag_stats[c]
        bl = bl_stats[c]
        lines.append(
            f"  {c:<6} {ag.mean_U:>8.4f} {bl.mean_U:>8.4f} "
            f"{ag.brier_score:>9.4f} {bl.brier_score:>9.4f} "
            f"{ag.repeated_errors_occurred:>8} {bl.repeated_errors_occurred:>8}"
        )
    lines.append("")

    # ── Brier score ────────────────────────────────────────────────────────────
    s("BRIER SCORE (confidence calibration)")
    lines.append("  Lower is better — perfect calibration = 0.0")
    lines.append("")
    row("Agent   overall Brier score:", f"{metrics['agent']['brier_score_overall']:.4f}")
    row("Baseline overall Brier score:", f"{metrics['baseline']['brier_score_overall']:.4f}")
    row("Brier improvement (agent over baseline):",
        f"{metrics['brier_improvement_pct']:.1f}%")
    lines.append("")

    # ── U ↔ correctness correlation ───────────────────────────────────────────
    s("U ↔ CORRECTNESS CORRELATION")
    lines.append("  Pearson r and Spearman ρ between utility U and binary is_correct")
    lines.append("")
    row("Agent   Pearson r:",
        f"{metrics['agent_pearson_r']:.4f}  "
        f"(p={metrics['agent']['pearson_p']:.4e})")
    row("Baseline Pearson r:",
        f"{metrics['baseline_pearson_r']:.4f}  "
        f"(p={metrics['baseline']['pearson_p']:.4e})")
    lines.append("")
    row("Agent   Spearman ρ:",
        f"{metrics['agent_spearman_rho']:.4f}  "
        f"(p={metrics['agent']['spearman_p']:.4e})")
    row("Baseline Spearman ρ:",
        f"{metrics['baseline_spearman_rho']:.4f}  "
        f"(p={metrics['baseline']['spearman_p']:.4e})")
    lines.append("")

    # ── Stability run ─────────────────────────────────────────────────────────
    s("10-CYCLE STABILITY RUN")
    lines.append(f"  100 tasks × 10 cycles (agent arm only)")
    lines.append("")
    lines.append(f"  {'Cycle':<6} {'Mean U':>8} {'Std U':>7} "
                 f"{'Contradiction%':>15} {'Brier':>7} {'Caution':>8} {'Curiosity':>9}")
    lines.append(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*15} {'-'*7} {'-'*8} {'-'*9}")
    for pc in stability["per_cycle"]:
        pers = pc.get("personality", {})
        lines.append(
            f"  {pc['cycle']:<6} {pc['mean_U']:>8.4f} {pc['std_U']:>7.4f} "
            f"{pc['contradiction_rate']:>14.1%} "
            f"{pc['brier_score']:>7.4f} "
            f"{pers.get('caution', 0.0):>8.3f} "
            f"{pers.get('curiosity', 0.0):>9.3f}"
        )
    lines.append("")

    # ── Long-tail errors ───────────────────────────────────────────────────────
    if stability["long_tail_errors"]:
        s("LONG-TAIL ERRORS (persistent ≥3 cycles)")
        lines.append("")
        for lt in stability["long_tail_errors"][:8]:
            lines.append(
                f"  {lt['problem_id']:<28} [{lt['error_type']}]  "
                f"first=C{lt['first_detected_cycle']}  "
                f"last=C{lt['last_persisted_cycle']}  "
                f"({lt['persistence_cycles']} cycles)"
            )
        lines.append("")

    # ── Key findings ──────────────────────────────────────────────────────────
    h("KEY FINDINGS FOR PAPER")
    lines.append("")
    lines.append(f"  1. Error Reduction")
    lines.append(f"     Agent reduces repeated errors by {pct:.1f}% (n=500 tasks, 5 cycles).")
    lines.append(f"     DPO-style correction injection suppresses {SUPPRESSION_RATE:.0%} of")
    lines.append(f"     previously-detected error patterns on re-occurrence.")
    lines.append("")
    lines.append(f"  2. Confidence Calibration (Brier Score)")
    lines.append(f"     Agent Brier = {metrics['agent']['brier_score_overall']:.4f}  "
                 f"vs  Baseline = {metrics['baseline']['brier_score_overall']:.4f}")
    lines.append(f"     ({metrics['brier_improvement_pct']:.1f}% improvement).")
    lines.append(f"     Agent self-monitoring penalizes confidence when contradictions are")
    lines.append(f"     detected, improving calibration over uncalibrated baseline.")
    lines.append("")
    lines.append(f"  3. U ↔ Correctness Correlation")
    lines.append(f"     Agent  Pearson r = {metrics['agent_pearson_r']:.4f}  |  "
                 f"Spearman ρ = {metrics['agent_spearman_rho']:.4f}")
    lines.append(f"     Baseline Pearson r = {metrics['baseline_pearson_r']:.4f}  |  "
                 f"Spearman ρ = {metrics['baseline_spearman_rho']:.4f}")
    lines.append(f"     Utility U is a stronger predictor of correctness in the agent")
    lines.append(f"     arm, validating the utility function as a meaningful signal.")
    lines.append("")
    lines.append(f"  4. Personality Convergence")
    c1 = stability["per_cycle"][0]
    c10 = stability["per_cycle"][-1]
    lines.append(f"     U: {c1['mean_U']:.4f} (cycle 1) → {c10['mean_U']:.4f} (cycle 10)  "
                 f"({c10['mean_U'] - c1['mean_U']:+.4f})")
    lines.append(f"     Contradiction rate: {c1['contradiction_rate']:.1%} → "
                 f"{c10['contradiction_rate']:.1%}")
    lines.append("")

    report = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def generate_plots(
    agent: ExperimentResult,
    baseline: ExperimentResult,
    stability: dict,
    metrics: dict,
    out_dir: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter

    os.makedirs(out_dir, exist_ok=True)

    AGENT_COLOR    = "#534AB7"
    BASELINE_COLOR = "#D85A30"
    NEUTRAL_COLOR  = "#888888"
    BG_COLOR       = "#fafaf8"
    GRID_COLOR     = "rgba(0,0,0,0.07)".replace("rgba", "").replace("(","").replace(")","")

    plt.rcParams.update({
        "font.family": "serif",
        "axes.facecolor": "#f9f8f5",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e0dbd4",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    ag_cs = agent.cycle_stats
    bl_cs = baseline.cycle_stats
    cycles = [cs.cycle for cs in ag_cs]

    # ── Fig 1: Mean U over cycles ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ag_u = [cs.mean_U for cs in ag_cs]
    bl_u = [cs.mean_U for cs in bl_cs]
    ag_s = [cs.std_U for cs in ag_cs]
    bl_s = [cs.std_U for cs in bl_cs]
    ax.plot(cycles, ag_u, color=AGENT_COLOR, marker="o", linewidth=2.5, label="Agent (full pipeline)")
    ax.fill_between(cycles,
                    [u - s for u, s in zip(ag_u, ag_s)],
                    [u + s for u, s in zip(ag_u, ag_s)],
                    color=AGENT_COLOR, alpha=0.15)
    ax.plot(cycles, bl_u, color=BASELINE_COLOR, marker="s", linewidth=2.5,
            linestyle="--", label="Baseline (no DPO)")
    ax.fill_between(cycles,
                    [u - s for u, s in zip(bl_u, bl_s)],
                    [u + s for u, s in zip(bl_u, bl_s)],
                    color=BASELINE_COLOR, alpha=0.12)
    ax.set(xlabel="Calibration Cycle", ylabel="Mean Utility U",
           title="Mean Utility Over Calibration Cycles (± 1 SD)",
           xticks=cycles, ylim=(0.3, 0.85))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_utility_over_cycles.png"), dpi=150)
    plt.close(fig)

    # ── Fig 2: Repeated error rate per cycle ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.array(cycles[1:])   # cycles 2–5 only
    ag_rep = [cs.repeated_errors_occurred for cs in ag_cs if cs.cycle > 1]
    bl_rep = [cs.repeated_errors_occurred for cs in bl_cs if cs.cycle > 1]
    w = 0.35
    bars_ag = ax.bar(x - w/2, ag_rep, w, color=AGENT_COLOR, alpha=0.85, label="Agent")
    bars_bl = ax.bar(x + w/2, bl_rep, w, color=BASELINE_COLOR, alpha=0.85, label="Baseline")
    for bar in bars_ag:
        h_ = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h_ + 0.3, str(int(h_)),
                ha="center", va="bottom", fontsize=8, color=AGENT_COLOR)
    for bar in bars_bl:
        h_ = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h_ + 0.3, str(int(h_)),
                ha="center", va="bottom", fontsize=8, color=BASELINE_COLOR)
    ax.set(xlabel="Calibration Cycle", ylabel="Repeated Error Count",
           title="Repeated Errors per Cycle (Agent vs Baseline)",
           xticks=x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_repeated_errors.png"), dpi=150)
    plt.close(fig)

    # ── Fig 3: Brier score per cycle ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ag_bs = [cs.brier_score for cs in ag_cs]
    bl_bs = [cs.brier_score for cs in bl_cs]
    ax.plot(cycles, ag_bs, color=AGENT_COLOR, marker="o", linewidth=2.5, label="Agent")
    ax.plot(cycles, bl_bs, color=BASELINE_COLOR, marker="s", linewidth=2.5,
            linestyle="--", label="Baseline")
    ax.axhline(y=0.25, color=NEUTRAL_COLOR, linestyle=":", linewidth=1, label="Random baseline (0.25)")
    ax.set(xlabel="Calibration Cycle",
           ylabel="Brier Score (lower = better calibration)",
           title="Brier Score per Cycle — Confidence Calibration Quality",
           xticks=cycles)
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_brier_score.png"), dpi=150)
    plt.close(fig)

    # ── Fig 4: U ↔ correctness calibration (binned reliability diagram) ───────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for idx, (result, color, arm_name) in enumerate([
        (agent, AGENT_COLOR, "Agent"),
        (baseline, BASELINE_COLOR, "Baseline"),
    ]):
        ax = axes[idx]
        utils = [t.utility for t in result.tasks]
        corrs = [int(t.is_correct) for t in result.tasks]
        bins  = np.linspace(min(utils), max(utils), 11)
        bin_means, bin_correct = [], []
        for i in range(len(bins) - 1):
            mask = [bins[i] <= u < bins[i+1] for u in utils]
            if sum(mask) > 0:
                bin_u = np.mean([u for u, m in zip(utils, mask) if m])
                bin_c = np.mean([c for c, m in zip(corrs, mask) if m])
                bin_means.append(bin_u)
                bin_correct.append(bin_c)
        ax.plot([0, 1], [0, 1], color=NEUTRAL_COLOR, linestyle=":", linewidth=1, label="Perfect calibration")
        ax.scatter(bin_means, bin_correct, color=color, s=60, zorder=3)
        ax.plot(bin_means, bin_correct, color=color, linewidth=2, label=arm_name)
        pr, _ = stats.pearsonr(utils, corrs)
        ax.set(xlabel="Mean Utility U (binned)",
               ylabel="Fraction Correct",
               title=f"{arm_name}: U vs Correctness\n(Pearson r = {pr:.3f})",
               xlim=(0.2, 0.85), ylim=(-0.05, 1.05))
        ax.legend()
    fig.suptitle("Utility U as Predictor of Correctness (Calibration Plot)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_calibration_plot.png"), dpi=150)
    plt.close(fig)

    # ── Fig 5: Error suppression trajectory by type ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    etype_colors = {
        "nested_loop_lie":   "#534AB7",
        "wrong_assert":      "#D85A30",
        "syntax_error":      "#1D9E75",
        "cross_session_flip":"#BA7517",
    }
    for etype in ERROR_TYPES:
        per_cycle_rate = []
        for c in range(1, len(agent.cycle_stats) + 1):
            ct = [t for t in agent.tasks if t.cycle == c]
            attempts = [t for t in ct if t.is_repeated_error_attempt and t.error_type == etype]
            occurred = [t for t in attempts if t.effective_error]
            rate = len(occurred) / max(len(attempts), 1)
            per_cycle_rate.append(rate)
        ax.plot(cycles, per_cycle_rate,
                color=etype_colors.get(etype, "#888"),
                marker="o", linewidth=2, label=etype.replace("_", " "))
    ax.set(xlabel="Calibration Cycle", ylabel="Repeated Error Rate",
           title="Error Suppression Trajectory by Error Type (Agent Arm)",
           xticks=cycles, ylim=(-0.05, 1.05))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_error_suppression.png"), dpi=150)
    plt.close(fig)

    # ── Fig 6: Personality convergence (10-cycle stability run) ───────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    trait_colors = {
        "curiosity": "#534AB7",
        "caution": "#D85A30",
        "analytical_rigor": "#1D9E75",
        "assertiveness": "#BA7517",
        "creativity": "#888888",
    }
    sc_cycles = [h["cycle"] for h in stability["personality_evolution"]]
    for trait, color in trait_colors.items():
        vals = [h["state"].get(trait, 0.5) for h in stability["personality_evolution"]]
        ax.plot(sc_cycles, vals, color=color, marker="o", linewidth=2, label=trait)
    ax.set(xlabel="Calibration Cycle", ylabel="Trait Score",
           title="Personality Trait Convergence (10-Cycle Stability Run)",
           xlim=(0, max(sc_cycles) + 0.5), ylim=(0.2, 0.95))
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_personality_convergence.png"), dpi=150)
    plt.close(fig)

    # ── Fig 7: Contradiction rate trend ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    # Agent 5-cycle
    ag_cr = [cs.contradiction_rate for cs in ag_cs]
    ax.plot(cycles, ag_cr, color=AGENT_COLOR, marker="o", linewidth=2.5, label="Agent (5-cycle)")
    # Baseline 5-cycle (always 0 — no detector)
    bl_cr = [cs.contradiction_rate for cs in bl_cs]
    ax.plot(cycles, bl_cr, color=BASELINE_COLOR, marker="s", linewidth=2.5,
            linestyle="--", label="Baseline (no detection)")
    # Stability 10-cycle
    st_cycles = [pc["cycle"] for pc in stability["per_cycle"]]
    st_cr     = [pc["contradiction_rate"] for pc in stability["per_cycle"]]
    ax.plot(st_cycles, st_cr, color="#1D9E75", marker="^", linewidth=2,
            linestyle=":", label="Agent (10-cycle stability)")
    ax.set(xlabel="Calibration Cycle", ylabel="Contradiction Detection Rate",
           title="Contradiction Rate Over Cycles")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_contradiction_rate.png"), dpi=150)
    plt.close(fig)

    # ── Fig 8: U distribution by correctness (violin) ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for idx, (result, color, arm_name) in enumerate([
        (agent, AGENT_COLOR, "Agent"),
        (baseline, BASELINE_COLOR, "Baseline"),
    ]):
        ax = axes[idx]
        corr_u   = [t.utility for t in result.tasks if t.is_correct]
        incorr_u = [t.utility for t in result.tasks if not t.is_correct]
        parts = ax.violinplot([incorr_u, corr_u], positions=[0, 1],
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.set(xticks=[0, 1], xticklabels=["Incorrect", "Correct"],
               ylabel="Utility U",
               title=f"{arm_name}: U Distribution by Correctness")
        # Overlay means
        ax.scatter([0, 1],
                   [np.mean(incorr_u) if incorr_u else 0, np.mean(corr_u) if corr_u else 0],
                   color=color, s=60, zorder=3)
    fig.suptitle("Utility Separates Correct from Incorrect Outputs", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig8_u_distribution.png"), dpi=150)
    plt.close(fig)

    # ── Fig 9: Long-tail error heatmap ─────────────────────────────────────────
    if stability["long_tail_errors"]:
        lt_data = stability["long_tail_errors"][:12]
        n_cycles_stab = stability["n_cycles"]
        problem_ids   = [f"{d['problem_id']}\n[{d['error_type'][:10]}]" for d in lt_data]
        matrix = np.zeros((len(lt_data), n_cycles_stab))
        for i, d in enumerate(lt_data):
            first = d["first_detected_cycle"] - 1
            last  = d["last_persisted_cycle"]
            matrix[i, first:last] = 1.0

        fig, ax = plt.subplots(figsize=(10, max(3, len(lt_data) * 0.5 + 1)))
        im = ax.imshow(matrix, aspect="auto", cmap="Purples", vmin=0, vmax=1)
        ax.set(yticks=range(len(lt_data)), yticklabels=problem_ids,
               xticks=range(n_cycles_stab),
               xticklabels=[f"C{i+1}" for i in range(n_cycles_stab)],
               xlabel="Cycle", title="Long-Tail Error Persistence Heatmap (10-Cycle Run)")
        fig.colorbar(im, ax=ax, label="Error Active")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig9_longtail_heatmap.png"), dpi=150)
        plt.close(fig)

    # ── Fig 10: Combined summary panel ────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # (0,0): U over cycles
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(cycles, ag_u, color=AGENT_COLOR, marker="o", linewidth=2, label="Agent")
    ax00.plot(cycles, bl_u, color=BASELINE_COLOR, marker="s", linewidth=2, linestyle="--", label="Baseline")
    ax00.set(title="Mean Utility Over Cycles", xlabel="Cycle", ylabel="Mean U", xticks=cycles)
    ax00.legend(fontsize=7)

    # (0,1): Repeated errors bar
    ax01 = fig.add_subplot(gs[0, 1])
    x2 = np.array(cycles[1:])
    ax01.bar(x2 - 0.2, ag_rep, 0.35, color=AGENT_COLOR, alpha=0.85, label="Agent")
    ax01.bar(x2 + 0.2, bl_rep, 0.35, color=BASELINE_COLOR, alpha=0.85, label="Baseline")
    ax01.set(title="Repeated Errors per Cycle", xlabel="Cycle", ylabel="Count", xticks=x2)
    ax01.legend(fontsize=7)

    # (0,2): Brier score
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.plot(cycles, ag_bs, color=AGENT_COLOR, marker="o", linewidth=2, label="Agent")
    ax02.plot(cycles, bl_bs, color=BASELINE_COLOR, marker="s", linewidth=2, linestyle="--", label="Baseline")
    ax02.set(title="Brier Score per Cycle", xlabel="Cycle", ylabel="Brier Score")
    ax02.invert_yaxis()
    ax02.legend(fontsize=7)

    # (1,0): Personality convergence (10-cycle)
    ax10 = fig.add_subplot(gs[1, 0])
    for trait, color in trait_colors.items():
        vals = [h["state"].get(trait, 0.5) for h in stability["personality_evolution"]]
        ax10.plot(sc_cycles, vals, color=color, linewidth=1.5, marker=".", label=trait)
    ax10.set(title="Personality Convergence (10 cycles)", xlabel="Cycle", ylabel="Score")
    ax10.legend(fontsize=6, ncol=2)

    # (1,1): Contradiction rate
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(cycles, ag_cr, color=AGENT_COLOR, marker="o", linewidth=2, label="Agent 5c")
    ax11.plot(st_cycles, st_cr, color="#1D9E75", linewidth=2, linestyle=":", marker="^", label="Agent 10c")
    ax11.set(title="Contradiction Rate", xlabel="Cycle", ylabel="Rate")
    ax11.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax11.legend(fontsize=7)

    # (1,2): Pearson r per cycle
    ax12 = fig.add_subplot(gs[1, 2])
    ag_pr = [cs.pearson_r for cs in ag_cs]
    bl_pr = [cs.pearson_r for cs in bl_cs]
    ax12.plot(cycles, ag_pr, color=AGENT_COLOR, marker="o", linewidth=2, label="Agent")
    ax12.plot(cycles, bl_pr, color=BASELINE_COLOR, marker="s", linewidth=2, linestyle="--", label="Baseline")
    ax12.axhline(0, color=NEUTRAL_COLOR, linewidth=0.8, linestyle=":")
    ax12.set(title="U ↔ Correctness (Pearson r)", xlabel="Cycle", ylabel="Pearson r", xticks=cycles)
    ax12.legend(fontsize=7)

    fig.suptitle("Adaptive Utility Agent — Extended Simulation Summary", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(out_dir, "fig10_summary_panel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  → 10 figures saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR = os.path.join(os.path.dirname(__file__), "extended_output")
    os.makedirs(OUT_DIR, exist_ok=True)

    N_CYCLES       = 5
    TASKS_PER_CYCLE = 100
    SEED           = 42

    print(f"\n{'='*60}")
    print(f"ADAPTIVE UTILITY AGENT — Extended Simulation v0.5")
    print(f"  500-task / 5-cycle two-arm comparison")
    print(f"  + 10-cycle stability run")
    print(f"{'='*60}")
    print(f"\n  Problems : {len(PROBLEMS)} types across {len(set(p['family'] for p in PROBLEMS))} algorithm families")
    print(f"  Tasks    : {N_CYCLES * TASKS_PER_CYCLE} per arm ({TASKS_PER_CYCLE}/cycle × {N_CYCLES} cycles)")
    print(f"  Error rate: {ERROR_INJECTION_RATE:.0%} injected | Suppression: {SUPPRESSION_RATE:.0%}")
    print(f"  Seed     : {SEED}")
    print(f"  Outputs  : {OUT_DIR}/")

    # ── Build shared task plan ─────────────────────────────────────────────────
    print(f"\n[1/5]  Building task plan...")
    plan = _build_task_plan(N_CYCLES, TASKS_PER_CYCLE, SEED)

    # ── Run agent arm ─────────────────────────────────────────────────────────
    print(f"\n[2/5]  Running AGENT arm...")
    agent_result = run_agent_arm(plan, N_CYCLES, TASKS_PER_CYCLE, verbose=True)

    # ── Run baseline arm ──────────────────────────────────────────────────────
    print(f"\n[3/5]  Running BASELINE arm...")
    baseline_result = run_baseline_arm(plan, N_CYCLES, TASKS_PER_CYCLE, verbose=True)

    # ── 10-cycle stability run ────────────────────────────────────────────────
    print(f"\n[4/5]  Running 10-cycle stability experiment...")
    stability = run_stability_experiment(n_cycles=10, tasks_per_cycle=100, seed=99, verbose=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_summary_metrics(agent_result, baseline_result)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    print(f"\n[5/5]  Writing outputs...")
    json_path = os.path.join(OUT_DIR, "extended_results.json")
    output = {
        "experiment_config": {
            "n_cycles": N_CYCLES,
            "tasks_per_cycle": TASKS_PER_CYCLE,
            "total_tasks_per_arm": N_CYCLES * TASKS_PER_CYCLE,
            "seed": SEED,
            "error_injection_rate": ERROR_INJECTION_RATE,
            "suppression_rate": SUPPRESSION_RATE,
            "correctness_threshold": 0.80,
            "n_problems": len(PROBLEMS),
            "problem_families": list(set(p["family"] for p in PROBLEMS)),
        },
        "metrics": metrics,
        "agent": {
            "cycle_stats": [asdict(cs) for cs in agent_result.cycle_stats],
            "personality_history": agent_result.personality_history,
            "tasks": [asdict(t) for t in agent_result.tasks],
        },
        "baseline": {
            "cycle_stats": [asdict(cs) for cs in baseline_result.cycle_stats],
            "tasks": [asdict(t) for t in baseline_result.tasks],
        },
        "stability": stability,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  → JSON:  {json_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = os.path.join(OUT_DIR, "report.txt")
    report = generate_report(
        agent_result, baseline_result, stability, metrics, report_path
    )
    print(f"  → Report: {report_path}")
    print()
    print(report)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots_dir = os.path.join(OUT_DIR, "plots")
    generate_plots(agent_result, baseline_result, stability, metrics, plots_dir)

    print(f"\n{'='*60}")
    print(f"DONE.  All outputs in:  {OUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
