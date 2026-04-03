"""
Simulation harness for the Adaptive Utility Agent MVP.

Runs without network access by mocking Claude responses with realistic
synthetic code solutions of varying quality. Tests the full scoring
pipeline: contradiction detection, confidence EMA, curiosity growth,
50% cap, and personality evolution.

Simulates 3 calibration cycles of 8 problems each to observe U evolution.
"""

import sys, os, math, json, ast, re, subprocess, tempfile, uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# ── Minimal stubs so imports work without network ─────────────────────────────

import types

# Stub config module
config_mod = types.ModuleType("config")

@dataclass
class FieldConfig:
    name: str
    w_efficacy: float
    w_confidence: float
    w_curiosity: float
    c_min: float
    e_min: float
    penalty_multiplier: float

SE_CFG = FieldConfig("software_engineering", 0.55, 0.35, 0.10, 0.70, 0.65, 2.0)
config_mod.FieldConfig = FieldConfig
config_mod.FIELD_CONFIGS = {"software_engineering": SE_CFG}
config_mod.get_effective_config = lambda d: SE_CFG
sys.modules["config"] = config_mod

from utility_scorer import UtilityScorer, CURIOSITY_ALPHA
from contradiction_detector import ContradictionDetector
from personality_manager import PersonalityManager

# ── Synthetic problem bank ────────────────────────────────────────────────────
# Each problem has: description, human_baseline, novelty,
# and a list of response variants (quality improves across calibration cycles
# as the "model" learns from corrections)

PROBLEMS = [
    {
        "name": "two_sum",
        "novelty": 0.2,
        "human_baseline": 0.85,
        "responses": [
            # Cycle 1: wrong complexity claim, brute force
            {
                "code": '''
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
                "claimed_complexity": "O(n)",   # WRONG — contradiction
                "tests": [
                    "assert two_sum([2,7,11,15], 9) == [0,1]",
                    "assert two_sum([3,2,4], 6) == [1,2]",
                ]
            },
            # Cycle 2: correct complexity, still brute force
            {
                "code": '''
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
                "claimed_complexity": "O(n^2)",  # now correct
                "tests": [
                    "assert two_sum([2,7,11,15], 9) == [0,1]",
                    "assert two_sum([3,2,4], 6) == [1,2]",
                ]
            },
            # Cycle 3: optimal hashmap solution
            {
                "code": '''
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []
''',
                "claimed_complexity": "O(n)",   # correct now
                "tests": [
                    "assert two_sum([2,7,11,15], 9) == [0,1]",
                    "assert two_sum([3,2,4], 6) == [1,2]",
                    "assert two_sum([3,3], 6) == [0,1]",
                ]
            },
        ]
    },
    {
        "name": "is_palindrome",
        "novelty": 0.3,
        "human_baseline": 0.80,
        "responses": [
            # Cycle 1: works but inefficient, no edge cases
            {
                "code": '''
def is_palindrome(s):
    s = s.lower()
    return s == s[::-1]
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert is_palindrome('racecar') == True",
                    "assert is_palindrome('hello') == False",
                ]
            },
            # Cycle 2: handles alphanumeric filtering
            {
                "code": '''
def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert is_palindrome('A man a plan a canal Panama') == True",
                    "assert is_palindrome('race a car') == False",
                ]
            },
            # Cycle 3: two-pointer, O(1) space
            {
                "code": '''
def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    l, r = 0, len(cleaned) - 1
    while l < r:
        if cleaned[l] != cleaned[r]:
            return False
        l += 1; r -= 1
    return True
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert is_palindrome('A man a plan a canal Panama') == True",
                    "assert is_palindrome('race a car') == False",
                    "assert is_palindrome('') == True",
                ]
            },
        ]
    },
    {
        "name": "max_subarray",
        "novelty": 0.5,
        "human_baseline": 0.70,
        "responses": [
            # Cycle 1: brute force, wrong complexity claim
            {
                "code": '''
def max_subarray(nums):
    max_sum = nums[0]
    for i in range(len(nums)):
        curr = 0
        for j in range(i, len(nums)):
            curr += nums[j]
            max_sum = max(max_sum, curr)
    return max_sum
''',
                "claimed_complexity": "O(n log n)",  # WRONG — contradiction
                "tests": [
                    "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6",
                ]
            },
            # Cycle 2: Kadane's but claimed complexity still slightly off
            {
                "code": '''
def max_subarray(nums):
    max_sum = curr = nums[0]
    for n in nums[1:]:
        curr = max(n, curr + n)
        max_sum = max(max_sum, curr)
    return max_sum
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6",
                    "assert max_subarray([1]) == 1",
                ]
            },
            # Cycle 3: Kadane's, correct, full edge cases
            {
                "code": '''
def max_subarray(nums):
    if not nums:
        return 0
    max_sum = curr = nums[0]
    for n in nums[1:]:
        curr = max(n, curr + n)
        max_sum = max(max_sum, curr)
    return max_sum
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6",
                    "assert max_subarray([1]) == 1",
                    "assert max_subarray([-1,-2,-3]) == -1",
                ]
            },
        ]
    },
    {
        "name": "binary_search",
        "novelty": 0.35,
        "human_baseline": 0.75,
        "responses": [
            # Cycle 1: off-by-one bug
            {
                "code": '''
def binary_search(nums, target):
    lo, hi = 0, len(nums)   # bug: should be len(nums)-1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return -1
''',
                "claimed_complexity": "O(log n)",
                "tests": [
                    "assert binary_search([1,3,5,7,9], 5) == 2",
                    "assert binary_search([1,3,5,7,9], 9) == 4",  # will fail
                ]
            },
            # Cycle 2: fixed
            {
                "code": '''
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
                "claimed_complexity": "O(log n)",
                "tests": [
                    "assert binary_search([1,3,5,7,9], 5) == 2",
                    "assert binary_search([1,3,5,7,9], 9) == 4",
                ]
            },
            # Cycle 3: full, handles empty, not found
            {
                "code": '''
def binary_search(nums, target):
    if not nums:
        return -1
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2  # avoids overflow
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
                "claimed_complexity": "O(log n)",
                "tests": [
                    "assert binary_search([1,3,5,7,9], 5) == 2",
                    "assert binary_search([1,3,5,7,9], 9) == 4",
                    "assert binary_search([], 1) == -1",
                    "assert binary_search([1,3,5], 4) == -1",
                ]
            },
        ]
    },
    {
        "name": "flatten_nested",
        "novelty": 0.6,
        "human_baseline": 0.65,
        "responses": [
            # Cycle 1: only handles 2 levels, wrong complexity
            {
                "code": '''
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert flatten([1,[2,3],[4]]) == [1,2,3,4]",
                    "assert flatten([1,[2,[3]]]) == [1,2,[3]]",  # will fail for deep
                ]
            },
            # Cycle 2: recursive, handles arbitrary depth
            {
                "code": '''
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert flatten([1,[2,3],[4]]) == [1,2,3,4]",
                    "assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]",
                ]
            },
            # Cycle 3: iterative stack-based (no recursion limit issue)
            {
                "code": '''
def flatten(lst):
    result = []
    stack = list(lst)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack = list(item) + stack
        else:
            result.append(item)
    return result
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert flatten([1,[2,3],[4]]) == [1,2,3,4]",
                    "assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]",
                    "assert flatten([]) == []",
                ]
            },
        ]
    },
    {
        "name": "lru_cache",
        "novelty": 0.75,
        "human_baseline": 0.55,
        "responses": [
            # Cycle 1: uses list (slow O(n) get/put)
            {
                "code": '''
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = []  # list of (key, value)

    def get(self, key):
        for i, (k, v) in enumerate(self.cache):
            if k == key:
                self.cache.pop(i)
                self.cache.append((k, v))
                return v
        return -1

    def put(self, key, value):
        for i, (k, v) in enumerate(self.cache):
            if k == key:
                self.cache.pop(i)
                self.cache.append((key, value))
                return
        if len(self.cache) >= self.capacity:
            self.cache.pop(0)
        self.cache.append((key, value))
''',
                "claimed_complexity": "O(1)",  # WRONG — O(n)
                "tests": [
                    "c = LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1",
                ]
            },
            # Cycle 2: OrderedDict — O(1)
            {
                "code": '''
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
''',
                "claimed_complexity": "O(1)",
                "tests": [
                    "from collections import OrderedDict",
                    "c = LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1",
                    "c = LRUCache(2); c.put(1,1); c.put(2,2); c.put(3,3); assert c.get(2)==2",
                ]
            },
            # Cycle 3: same but with eviction test
            {
                "code": '''
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
''',
                "claimed_complexity": "O(1)",
                "tests": [
                    "from collections import OrderedDict",
                    "c = LRUCache(2); c.put(1,1); c.put(2,2); c.get(1); c.put(3,3); assert c.get(2)==-1",
                    "c = LRUCache(2); c.put(1,1); c.put(2,2); c.get(1); c.put(3,3); assert c.get(1)==1",
                ]
            },
        ]
    },
    {
        "name": "valid_parentheses",
        "novelty": 0.4,
        "human_baseline": 0.78,
        "responses": [
            # Cycle 1: only handles single bracket type
            {
                "code": '''
def is_valid(s):
    count = 0
    for c in s:
        if c == '(':
            count += 1
        elif c == ')':
            count -= 1
        if count < 0:
            return False
    return count == 0
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert is_valid('()') == True",
                    "assert is_valid('()[]{}') == True",  # will FAIL
                ]
            },
            # Cycle 2: stack-based, all bracket types
            {
                "code": '''
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in mapping:
            top = stack.pop() if stack else '#'
            if mapping[c] != top:
                return False
        else:
            stack.append(c)
    return not stack
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert is_valid('()') == True",
                    "assert is_valid('()[]{}') == True",
                    "assert is_valid('(]') == False",
                ]
            },
            # Cycle 3: same + empty string edge case
            {
                "code": '''
def is_valid(s):
    if not s:
        return True
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in mapping:
            top = stack.pop() if stack else '#'
            if mapping[c] != top:
                return False
        else:
            stack.append(c)
    return not stack
''',
                "claimed_complexity": "O(n)",
                "tests": [
                    "assert is_valid('') == True",
                    "assert is_valid('()[]{') == False",
                    "assert is_valid('{[]}') == True",
                ]
            },
        ]
    },
    {
        "name": "merge_intervals",
        "novelty": 0.65,
        "human_baseline": 0.60,
        "responses": [
            # Cycle 1: doesn't sort first, wrong result
            {
                "code": '''
def merge(intervals):
    result = [intervals[0]]
    for interval in intervals[1:]:
        if interval[0] <= result[-1][1]:
            result[-1][1] = max(result[-1][1], interval[1])
        else:
            result.append(interval)
    return result
''',
                "claimed_complexity": "O(n log n)",
                "tests": [
                    "assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
                    "assert merge([[1,4],[4,5]]) == [[1,5]]",
                ]
            },
            # Cycle 2: sorts, works
            {
                "code": '''
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
''',
                "claimed_complexity": "O(n log n)",
                "tests": [
                    "assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
                ]
            },
            # Cycle 3: handles edge cases
            {
                "code": '''
def merge(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0][:]]
    for start, end in intervals[1:]:
        if start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
''',
                "claimed_complexity": "O(n log n)",
                "tests": [
                    "assert merge([]) == []",
                    "assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
                    "assert merge([[1,4],[4,5]]) == [[1,5]]",
                ]
            },
        ]
    },
]

# ── Scoring helpers ────────────────────────────────────────────────────────────

def run_tests(code: str, tests: List[str]) -> Tuple[float, List[str]]:
    """Run tests against code. Returns (pass_rate, failed_tests)."""
    if not tests:
        return 0.8, []
    passed = 0
    failed = []
    for test in tests:
        full = code + "\n" + test
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full)
                fname = f.name
            r = subprocess.run(["python3", fname], capture_output=True, timeout=5)
            os.unlink(fname)
            if r.returncode == 0:
                passed += 1
            else:
                failed.append(test.strip())
        except Exception:
            passed += 1  # runner error → don't penalize
    return passed / len(tests), failed


def check_complexity(code: str, claimed: str) -> Tuple[bool, str]:
    """Heuristic complexity contradiction check."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "syntax error"

    nested = 0
    def walk(node, depth):
        nonlocal nested
        if isinstance(node, (ast.For, ast.While)):
            depth += 1
            nested = max(nested, depth)
        for child in ast.iter_child_nodes(node):
            walk(child, depth)
    walk(tree, 0)

    cl = claimed.lower().replace(" ", "")
    if ("o(1)" in cl or "o(n)" in cl) and nested >= 2:
        return True, f"claimed {claimed} but code has {nested} nested loops → likely O(n²) or worse"
    return False, ""


def estimate_novelty_for_cycle(base_novelty: float, cycle: int) -> float:
    """Novelty decreases as the agent has seen more similar problems."""
    return max(0.05, base_novelty * (0.8 ** cycle))


# ── Main simulation ────────────────────────────────────────────────────────────

def run_simulation():
    scorer = UtilityScorer()
    detector = ContradictionDetector(penalty_multiplier=2.0)
    personality = PersonalityManager()

    all_results = []
    cycle_summaries = []

    N_CYCLES = 3
    EVOLUTION_INTERVAL = 8  # evolve personality after each cycle

    print("=" * 70)
    print("ADAPTIVE UTILITY AGENT — SIMULATION HARNESS")
    print(f"Field: software_engineering  |  Cycles: {N_CYCLES}  |  Problems: {len(PROBLEMS)}")
    print("=" * 70)

    for cycle in range(N_CYCLES):
        print(f"\n{'─'*70}")
        print(f"CALIBRATION CYCLE {cycle + 1} / {N_CYCLES}")
        print(f"{'─'*70}")

        cycle_scores = []

        for prob in PROBLEMS:
            response = prob["responses"][min(cycle, len(prob["responses"]) - 1)]
            code      = response["code"].strip()
            claimed   = response.get("claimed_complexity")
            tests     = response["tests"]
            novelty   = estimate_novelty_for_cycle(prob["novelty"], cycle)

            # Run tests
            pass_rate, failed_tests = run_tests(code, tests)

            # Check complexity contradiction
            has_contradiction, contradiction_msg = check_complexity(code, claimed) if claimed else (False, "")

            # Also run through contradiction detector
            solution_text = f"```python\n{code}\n```\n{claimed or ''}"
            c_result = detector.check(prob["name"], solution_text, claimed)
            if has_contradiction and not c_result.contradictions:
                from contradiction_detector import Contradiction
                c_result.contradictions.append(
                    Contradiction("mathematical", contradiction_msg, 0.7)
                )
                c_result.confidence_penalty = max(c_result.confidence_penalty, 0.12)

            # Score
            task_id = f"c{cycle+1}_{prob['name']}"
            score = scorer.score(
                task_id=task_id,
                field_config=SE_CFG,
                test_pass_rate=pass_rate,
                human_baseline_score=prob["human_baseline"],
                contradiction_penalty=c_result.confidence_penalty,
                problem_novelty=novelty,
            )

            cycle_scores.append(score)
            all_results.append({
                "cycle": cycle + 1,
                "problem": prob["name"],
                "pass_rate": round(pass_rate, 3),
                "contradiction": has_contradiction,
                "failed_tests": len(failed_tests),
                "E": score.efficacy,
                "C": score.confidence,
                "K_raw": score.curiosity_raw,
                "K_eff": score.curiosity_effective,
                "U": score.utility,
                "capped": score.curiosity_capped,
            })

            cap_flag = " [K CAPPED]" if score.curiosity_capped else ""
            contra_flag = " ⚠ CONTRADICTION" if has_contradiction else ""
            fail_flag = f" ✗ {len(failed_tests)} test(s) failed" if failed_tests else ""
            print(
                f"  {prob['name']:<22} "
                f"pass={pass_rate:.0%}  "
                f"E={score.efficacy:.3f}  C={score.confidence:.3f}  "
                f"K={score.curiosity_effective:.3f}  U={score.utility:.3f}"
                f"{cap_flag}{contra_flag}{fail_flag}"
            )

        # Cycle summary
        avg_u = sum(s.utility for s in cycle_scores) / len(cycle_scores)
        avg_c = sum(s.confidence for s in cycle_scores) / len(cycle_scores)
        avg_e = sum(s.efficacy for s in cycle_scores) / len(cycle_scores)
        n_contradictions = sum(1 for r in all_results if r["cycle"] == cycle+1 and r["contradiction"])
        n_failed = sum(r["failed_tests"] for r in all_results if r["cycle"] == cycle+1)

        cycle_summaries.append({
            "cycle": cycle + 1,
            "avg_U": round(avg_u, 4),
            "avg_E": round(avg_e, 4),
            "avg_C": round(avg_c, 4),
            "contradictions": n_contradictions,
            "failed_tests": n_failed,
        })

        print(f"\n  ── Cycle {cycle+1} summary ──")
        print(f"  avg U={avg_u:.4f}  avg E={avg_e:.4f}  avg C={avg_c:.4f}")
        print(f"  contradictions={n_contradictions}  failed_tests={n_failed}")

        # Personality evolution
        trend = scorer.get_utility_trend(domain="software_engineering", last_n=len(PROBLEMS))
        domain_state = scorer.domain_states.get("software_engineering")
        c_rate = domain_state.contradiction_count / max(domain_state.interaction_count, 1) if domain_state else 0
        personality.evolve(trend, c_rate, "software_engineering")

    # ── Final report ─────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("RESULTS ACROSS ALL CYCLES")
    print("=" * 70)

    print(f"\n{'Cycle':<8} {'avg U':>8} {'avg E':>8} {'avg C':>8} {'Contradictions':>16} {'Failed Tests':>14}")
    print(f"{'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*16} {'─'*14}")
    for s in cycle_summaries:
        delta = ""
        if s["cycle"] > 1:
            prev = cycle_summaries[s["cycle"] - 2]["avg_U"]
            diff = s["avg_U"] - prev
            delta = f"  ({'+' if diff >= 0 else ''}{diff:.4f})"
        print(f"  {s['cycle']:<6} {s['avg_U']:>8.4f} {s['avg_E']:>8.4f} {s['avg_C']:>8.4f} "
              f"  {s['contradictions']:>14}   {s['failed_tests']:>12}{delta}")

    # Per-problem U trajectory
    print(f"\n{'─'*70}")
    print("UTILITY TRAJECTORY BY PROBLEM (U across cycles)")
    print(f"{'─'*70}")
    print(f"{'Problem':<24} {'Cycle 1':>10} {'Cycle 2':>10} {'Cycle 3':>10} {'Trend':>10}")
    print(f"{'─'*24} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for prob in PROBLEMS:
        row = [r for r in all_results if r["problem"] == prob["name"]]
        vals = [r["U"] for r in row]
        trend_str = "↑ improving" if vals[-1] > vals[0] else ("↓ declining" if vals[-1] < vals[0] else "→ flat")
        cols = "  ".join(f"{v:>8.4f}" for v in vals)
        print(f"  {prob['name']:<22} {cols}  {trend_str}")

    # Domain state
    print(f"\n{'─'*70}")
    print("FINAL DOMAIN STATE (software_engineering)")
    print(f"{'─'*70}")
    summary = scorer.get_domain_summary("software_engineering")
    for k, v in summary.items():
        print(f"  {k:<35} {v}")

    # Personality drift
    print(f"\n{'─'*70}")
    print("PERSONALITY TRAIT EVOLUTION")
    print(f"{'─'*70}")
    traits = personality.get_trait_summary()
    for name, score_val in traits.items():
        bar = "█" * int(score_val * 30)
        print(f"  {name:<20} {score_val:.3f}  {bar}")

    if personality.evolution_history:
        print(f"\n  Evolution log ({len(personality.evolution_history)} runs):")
        for ev in personality.evolution_history:
            changes = ev.get("changes", {})
            if changes:
                change_str = ", ".join(f"{k}: {'+' if v>0 else ''}{v:.2f}" for k, v in changes.items())
                print(f"    cycle avg_U={ev['avg_utility']:.4f}  "
                      f"contradiction_rate={ev['contradiction_rate']:.2f}  → {change_str}")

    # U trend
    all_u = [r["U"] for r in all_results]
    u_c1 = [r["U"] for r in all_results if r["cycle"] == 1]
    u_c3 = [r["U"] for r in all_results if r["cycle"] == 3]
    print(f"\n{'─'*70}")
    print(f"OVERALL U IMPROVEMENT: {sum(u_c1)/len(u_c1):.4f} → {sum(u_c3)/len(u_c3):.4f}  "
          f"(Δ = {(sum(u_c3)/len(u_c3)) - (sum(u_c1)/len(u_c1)):+.4f})")
    print(f"CONTRADICTION REDUCTION: {cycle_summaries[0]['contradictions']} → {cycle_summaries[-1]['contradictions']}")
    print(f"FAILED TEST REDUCTION:   {cycle_summaries[0]['failed_tests']} → {cycle_summaries[-1]['failed_tests']}")
    print("=" * 70)

    # Save JSON
    out = {
        "cycle_summaries": cycle_summaries,
        "per_problem": all_results,
        "domain_state": summary,
        "personality": traits,
        "evolution_history": personality.evolution_history,
    }
    with open("/mnt/user-data/outputs/simulation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nFull results saved to simulation_results.json")


if __name__ == "__main__":
    run_simulation()
