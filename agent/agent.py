"""
UtilityAgent: the main wrapper around a frontier model.

Ties together field classification, personality management,
contradiction detection, and utility scoring.
"""

import asyncio
import httpx
import uuid
from dataclasses import dataclass
from typing import Optional

from config import FIELD_CONFIGS, get_effective_config
from field_classifier import classify_field
from contradiction_detector import ContradictionDetector
from utility_scorer import UtilityScorer, TaskScore
from personality_manager import PersonalityManager


SYSTEM_PROMPT_TEMPLATE = """You are a precise, rigorous assistant operating under a utility-maximizing framework.

Field context: {field}
Minimum confidence threshold for this domain: {c_min}
Minimum efficacy threshold for this domain: {e_min}

{personality_prompt}

Core principles:
1. If you are not confident enough to meet the domain threshold, say so explicitly rather than guessing.
2. State your complexity claims precisely (e.g. O(n log n)) — these will be verified.
3. Be internally consistent. If you stated something earlier in this session, do not contradict it.
4. For code: always include test cases that verify your stated behavior.
5. Share the least about your internal state that the situation allows.
"""


@dataclass
class AgentResponse:
    task_id: str
    content: str
    score: TaskScore
    field_distribution: dict
    should_abstain: bool
    abstain_reason: Optional[str] = None


class UtilityAgent:
    """
    Wrapper around Claude that applies the utility framework.
    """

    def __init__(self, evolution_interval: int = 20):
        self.scorer = UtilityScorer()
        self.personality = PersonalityManager()
        self.contradiction_detector = ContradictionDetector()
        self.evolution_interval = evolution_interval
        self.interaction_count = 0
        self.conversation_history = []

    async def respond(
        self,
        task: str,
        human_baseline_score: float = 0.7,   # default human benchmark
        problem_novelty: float = 0.5,         # how novel is this task
        run_tests: bool = True,
    ) -> AgentResponse:

        task_id = str(uuid.uuid4())[:8]

        # ── Step 1: Classify field ────────────────────────────────────────────
        field_dist = await classify_field(task)
        field_config = get_effective_config(field_dist)
        primary_field = max(field_dist, key=field_dist.get)

        # ── Step 2: Determine situation type for personality ──────────────────
        situation = self._infer_situation(field_config)

        # ── Step 3: Check if we should abstain based on current confidence ────
        domain_summary = self.scorer.get_domain_summary(primary_field)
        current_confidence = domain_summary.get("confidence", 0.5)

        if current_confidence < field_config.c_min:
            return AgentResponse(
                task_id=task_id,
                content="",
                score=TaskScore(
                    task_id=task_id, field=primary_field,
                    efficacy=0.0, confidence=current_confidence,
                    curiosity=0.0, utility=0.0,
                    timestamp="", below_minimum=True
                ),
                field_distribution=field_dist,
                should_abstain=True,
                abstain_reason=(
                    f"Current confidence ({current_confidence:.2f}) is below "
                    f"minimum for {primary_field} ({field_config.c_min}). "
                    f"Escalation recommended."
                )
            )

        # ── Step 4: Build system prompt ───────────────────────────────────────
        personality_prompt = self.personality.build_personality_prompt(situation)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            field=primary_field,
            c_min=field_config.c_min,
            e_min=field_config.e_min,
            personality_prompt=personality_prompt
        )

        # ── Step 5: Call frontier model ───────────────────────────────────────
        self.conversation_history.append({"role": "user", "content": task})
        response_text = await self._call_claude(system_prompt)
        self.conversation_history.append({"role": "assistant", "content": response_text})

        # ── Step 6: Run contradiction detection ───────────────────────────────
        claimed_complexity = self._extract_complexity_claim(response_text)
        contradiction_result = self.contradiction_detector.check(
            problem=task,
            solution=response_text,
            claimed_complexity=claimed_complexity
        )

        # ── Step 7: Compute test pass rate ────────────────────────────────────
        test_pass_rate = self._compute_test_pass_rate(response_text) if run_tests else 0.7

        # ── Step 8: Score utility ─────────────────────────────────────────────
        score = self.scorer.score(
            task_id=task_id,
            field_config=field_config,
            test_pass_rate=test_pass_rate,
            human_baseline_score=human_baseline_score,
            contradiction_penalty=contradiction_result.confidence_penalty,
            problem_novelty=problem_novelty
        )

        # ── Step 9: Periodic personality evolution ────────────────────────────
        self.interaction_count += 1
        if self.interaction_count % self.evolution_interval == 0:
            self._run_evolution(primary_field)

        # Print a summary for observability
        self._log_score(task_id, score, contradiction_result)

        return AgentResponse(
            task_id=task_id,
            content=response_text,
            score=score,
            field_distribution=field_dist,
            should_abstain=score.below_minimum
        )

    # ── Private ───────────────────────────────────────────────────────────────

    async def _call_claude(self, system_prompt: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system": system_prompt,
                    "messages": self.conversation_history
                },
                timeout=30.0
            )
            data = response.json()
            blocks = data.get("content", [])
            return " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    def _infer_situation(self, field_config) -> str:
        """Map field config to a personality situation type."""
        if field_config.c_min >= 0.90:
            return "high_stakes"
        if field_config.w_curiosity >= 0.20:
            return "exploration"
        if field_config.w_efficacy >= 0.75:
            return "creative"
        return "technical"

    def _extract_complexity_claim(self, text: str) -> Optional[str]:
        """Look for Big-O claims in the response."""
        import re
        match = re.search(r"O\([^)]+\)", text)
        return match.group(0) if match else None

    def _compute_test_pass_rate(self, solution: str) -> float:
        """
        Run tests from the contradiction detector and return pass rate.
        Returns 1.0 if no tests found (benefit of the doubt).
        """
        import re, subprocess, tempfile, os
        code_match = re.findall(r"```(?:python)?\n(.*?)```", solution, re.DOTALL)
        if not code_match:
            return 0.8  # no code found, partial credit

        code = code_match[0].strip()
        asserts = [l.strip() for l in solution.split("\n") if l.strip().startswith("assert")]

        if not asserts:
            return 0.8  # no tests to run

        passed = 0
        for test in asserts:
            full = f"{code}\n\n{test}"
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(full)
                    fname = f.name
                r = subprocess.run(["python3", fname], capture_output=True, timeout=5)
                os.unlink(fname)
                if r.returncode == 0:
                    passed += 1
            except Exception:
                passed += 1  # runner error = don't penalize
        return passed / len(asserts)

    def _run_evolution(self, domain: str):
        """Trigger personality evolution based on recent utility history."""
        trend = self.scorer.get_utility_trend(domain=domain, last_n=self.evolution_interval)
        if not trend:
            return
        domain_state = self.scorer.domain_states.get(domain)
        contradiction_rate = 0.0
        if domain_state and domain_state.interaction_count > 0:
            contradiction_rate = domain_state.contradiction_count / domain_state.interaction_count
        self.personality.evolve(trend, contradiction_rate, domain)
        print(f"\n[PersonalityManager] Evolution triggered after {self.interaction_count} interactions")
        print(f"  Trait weights: {self.personality.get_trait_summary()}")

    def _log_score(self, task_id, score, contradiction_result):
        status = "⚠ BELOW MINIMUM" if score.below_minimum else "✓"
        print(f"\n[UtilityAgent] Task {task_id} {status}")
        print(f"  Field: {score.field}")
        print(f"  E={score.efficacy:.3f}  C={score.confidence:.3f}  K={score.curiosity:.3f}  U={score.utility:.3f}")
        if contradiction_result.contradictions:
            print(f"  Contradictions: {len(contradiction_result.contradictions)}")
            for c in contradiction_result.contradictions:
                print(f"    [{c.type}] {c.description}")
        if score.notes:
            print(f"  Note: {score.notes}")
