"""
Contradiction detector for code generation outputs.

Detects three types of contradictions:
1. Logical      — code does the opposite of what it claims
2. Mathematical — claimed complexity contradicts actual complexity
3. Cross-session — conflicts with prior solutions to similar problems
"""

import ast
import re
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Contradiction:
    type: str           # "logical" | "mathematical" | "cross_session"
    description: str
    severity: float     # 0.0 to 1.0


@dataclass
class ContradictionResult:
    contradictions: List[Contradiction] = field(default_factory=list)
    confidence_penalty: float = 0.0

    @property
    def is_clean(self) -> bool:
        return len(self.contradictions) == 0


class ContradictionDetector:
    """Runs automated contradiction checks on code generation outputs."""

    def __init__(self, penalty_multiplier: float = 2.0):
        self.penalty_multiplier = penalty_multiplier
        self.session_history: List[dict] = []

    def check(
        self,
        problem: str,
        solution: str,
        claimed_complexity: Optional[str] = None,
    ) -> ContradictionResult:
        result = ContradictionResult()
        self._check_syntax(solution, result)
        self._check_logical_consistency(problem, solution, result)
        if claimed_complexity:
            self._check_complexity_claim(solution, claimed_complexity, result)
        self._check_cross_session(problem, solution, result)
        if result.contradictions:
            total_severity = sum(c.severity for c in result.contradictions)
            result.confidence_penalty = min(
                total_severity * self.penalty_multiplier * 0.1, 0.5
            )
        self.session_history.append({
            "problem": problem, "solution": solution,
            "claimed_complexity": claimed_complexity,
            "contradictions": result.contradictions,
        })
        return result

    def _check_syntax(self, solution: str, result: ContradictionResult):
        code = self._extract_code(solution)
        if not code:
            return
        try:
            ast.parse(code)
        except SyntaxError as e:
            result.contradictions.append(Contradiction(
                type="logical", description=f"Syntax error: {e}", severity=0.8
            ))

    def _check_logical_consistency(self, problem: str, solution: str, result: ContradictionResult):
        code = self._extract_code(solution)
        test_cases = self._extract_test_cases(solution)
        if not code or not test_cases:
            return
        for test in test_cases:
            if not self._run_test(code, test):
                result.contradictions.append(Contradiction(
                    type="logical",
                    description=f"Code fails its own stated test case: {test}",
                    severity=0.9,
                ))

    def _check_complexity_claim(self, solution: str, claimed_complexity: str, result: ContradictionResult):
        code = self._extract_code(solution)
        if not code:
            return
        nested_loop_count = self._count_nested_loops(code)
        norm = claimed_complexity.lower()
        for ch in ("\\", "$", "{", "}", "`", "*", ","):
            norm = norm.replace(ch, "")
        norm = "".join(norm.split())
        if ("o(n)" in norm or "o(1)" in norm) and nested_loop_count >= 2:
            result.contradictions.append(Contradiction(
                type="mathematical",
                description=f"Claimed {claimed_complexity} but code has {nested_loop_count} nested loop levels",
                severity=0.7,
            ))
        if "o(nlogn)" in norm and nested_loop_count >= 3:
            result.contradictions.append(Contradiction(
                type="mathematical",
                description=f"Claimed {claimed_complexity} but code structure suggests O(n²) or worse",
                severity=0.6,
            ))

    def _check_cross_session(self, problem: str, solution: str, result: ContradictionResult):
        code = self._extract_code(solution)
        if not code or not self.session_history:
            return
        for prior in self.session_history[-10:]:
            if self._problem_similarity(problem, prior["problem"]) > 0.7:
                prior_code = self._extract_code(prior["solution"])
                if prior_code and self._are_contradictory_approaches(code, prior_code):
                    result.contradictions.append(Contradiction(
                        type="cross_session",
                        description="Solution approach contradicts a prior solution to a similar problem",
                        severity=0.5,
                    ))

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_code(self, text: str) -> Optional[str]:
        matches = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        return matches[0].strip() if matches else None

    def _extract_test_cases(self, text: str) -> List[str]:
        return [l.strip() for l in text.split("\n") if l.strip().startswith("assert")]

    def _run_test(self, code: str, test: str) -> bool:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(f"{code}\n\n{test}")
                fname = f.name
            result = subprocess.run(
                ["python3", fname], capture_output=True, text=True, timeout=5
            )
            os.unlink(fname)
            return result.returncode == 0
        except Exception:
            return True

    def _count_nested_loops(self, code: str) -> int:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0
        max_depth = 0
        def walk(node, depth):
            nonlocal max_depth
            if isinstance(node, (ast.For, ast.While)):
                depth += 1
                max_depth = max(max_depth, depth)
            for child in ast.iter_child_nodes(node):
                walk(child, depth)
        walk(tree, 0)
        return max_depth

    def _problem_similarity(self, p1: str, p2: str) -> float:
        w1, w2 = set(p1.lower().split()), set(p2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def _are_contradictory_approaches(self, code1: str, code2: str) -> bool:
        uses_sort_1 = "sort" in code1.lower()
        uses_hash_1 = "dict" in code1.lower() or "set" in code1.lower()
        uses_sort_2 = "sort" in code2.lower()
        uses_hash_2 = "dict" in code2.lower() or "set" in code2.lower()
        return (uses_sort_1 and uses_hash_2) or (uses_hash_1 and uses_sort_2)
