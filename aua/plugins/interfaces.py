"""
aua/plugins/interfaces.py — Formal Python Protocol interfaces for all AUA plugin types.

These are the stable public extension points. v1.x guarantees backward compatibility
for all Protocols defined here. Breaking changes require a major version bump.

Usage:
    from aua.plugins.interfaces import UtilityScorerPlugin

    class MyScorer:
        def score(self, response: str, field: str, prior_u: float) -> float:
            return 0.8

    # Implement the protocol — no inheritance required
    # Register in aua_config.yaml:
    #   utility_scorer:
    #     import_path: my_module:MyScorer

Stability guarantee:
    All method signatures in this file are stable from v0.8 onwards.
    Deprecated methods will have one minor-version warning period before removal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

# ── Field Classifier ──────────────────────────────────────────────────────────


@runtime_checkable
class FieldClassifierPlugin(Protocol):
    """
    Replaces the built-in field classifier.

    Receives the raw query string, returns a probability distribution over
    all known field names. Probabilities must sum to ≤ 1.0 (remainder = unknown).
    """

    def classify(self, query: str) -> dict[str, float]:
        """
        Classify a query into field probabilities.

        Args:
            query: raw user query string

        Returns:
            dict mapping field_name → probability (0.0–1.0)
            Example: {"software_engineering": 0.82, "mathematics": 0.14}
        """
        ...


# ── Utility Scorer ────────────────────────────────────────────────────────────


@runtime_checkable
class UtilityScorerPlugin(Protocol):
    """
    Replaces the built-in U = w_e·E + w_c·C + w_k·K scorer.

    Adjustment mode: the plugin receives the built-in U as prior_u and may
    adjust it. The built-in pipeline still runs; the plugin gets the last word.

    For full U replacement (bypassing the built-in w_e·E + w_c·C + w_k·K
    computation), also implement FullUtilityScorerPlugin.score_full(). The
    router checks for score_full() at call time via hasattr(), so the two
    Protocols are independent — you can implement both on the same class.
    """

    def score(
        self,
        response: str,
        field: str,
        prior_u: float,
        confidence: float,
        metadata: dict[str, Any],
    ) -> float:
        """
        Adjust the built-in U score (mode 1 — adjustment).

        Args:
            response:   the specialist's text output
            field:      field name (e.g. "software_engineering")
            prior_u:    the built-in U score (w_e·E + w_c·C + w_k·K result)
            confidence: Kalman-filtered confidence estimate (0.0–1.0)
            metadata:   context: {query, contradictions, latency_ms, ...}

        Returns:
            U score in [0.0, 1.0]
        """
        ...


# ── Arbiter Policy ────────────────────────────────────────────────────────────


@runtime_checkable
class ArbiterPolicyPlugin(Protocol):
    """
    Replaces the built-in 4-check arbitration policy.

    Receives two specialist outputs and returns a verdict dict.
    """

    def arbitrate(
        self,
        subject: str,
        domain: str,
        output_a: str,
        output_b: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Arbitrate between two specialist outputs.

        Args:
            subject:   short subject identifier (e.g. "bubble_sort_complexity")
            domain:    field name
            output_a:  first specialist's response
            output_b:  second specialist's response
            metadata:  context (field_penalty_multiplier, session_id, etc.)

        Returns:
            dict with keys:
                case (str): "case_1" | "case_2" | "case_3" | "case_4"
                correct_a (bool): True if A should receive a correction
                correct_b (bool): True if B should receive a correction
                verified_claim (str | None): the correct answer, or None
                external_response (str): what the user sees
        """
        ...


# ── Promotion Policy ──────────────────────────────────────────────────────────


@runtime_checkable
class PromotionPolicyPlugin(Protocol):
    """
    Decides whether a GREEN candidate should be promoted to BLUE.

    Replaces the built-in delta+T_min threshold policy.
    """

    def should_promote(
        self,
        specialist: str,
        blue_mean_u: float,
        green_mean_u: float,
        n_queries: int,
        metadata: dict[str, Any],
    ) -> bool:
        """
        Decide whether GREEN should replace BLUE.

        Args:
            specialist:   specialist name
            blue_mean_u:  BLUE model's mean utility score
            green_mean_u: GREEN candidate's mean utility score
            n_queries:    number of canary queries evaluated
            metadata:     config values (delta, T_min, tau, etc.)

        Returns:
            True to promote GREEN → BLUE, False to keep BLUE
        """
        ...


# ── Full Promotion Policy (#51 wire-up) ──────────────────────────────────────


@runtime_checkable
class FullPromotionPolicyPlugin(Protocol):
    """
    Optional extension of PromotionPolicyPlugin for arbitrary promotion functions.

    Mirrors the FullUtilityScorerPlugin pattern: PromotionPolicyPlugin.should_promote()
    gives a simple boolean based on pre-computed scalars. This Protocol adds
    should_promote_full() which receives the complete promotion context — raw
    scores, shadow history, regression results, field config — so any function
    of those inputs can drive the decision.

    The router checks for should_promote_full() via hasattr() at call time.
    Implement should_promote_full() to opt into full context mode.
    Implement should_promote() as a fallback if should_promote_full() raises.

    Non-linear promotion functions this enables:

        Confidence-interval gate — require green_u > blue_u + 2*std_dev:
            def should_promote_full(self, context):
                std = context["shadow_std_delta"]
                return context["mean_delta"] > 2 * std

        Sample-size adaptive threshold — stricter when n is small:
            def should_promote_full(self, context):
                n = context["n_queries"]
                # Wilson-style: require larger delta when sample is small
                adaptive = 0.025 + 0.5 / max(n, 1)
                return context["mean_delta"] >= adaptive

        Multi-factor gate — regression + delta + minimum n:
            def should_promote_full(self, context):
                if context["regression_result"] and context["regression_result"]["regressed"]:
                    return False
                if context["n_queries"] < context["min_queries"]:
                    return False
                return context["mean_delta"] >= context["threshold"]

    YAML: register as promotion_policy — no separate key needed.

        plugins:
          promotion_policy:
            import_path: my_plugins:AdaptivePromoter
    """

    def should_promote_full(self, context: dict[str, Any]) -> bool:
        """
        Decide whether GREEN should replace BLUE given full promotion context.

        Args:
            context: dict with the following keys:
                specialist (str):         specialist name
                blue_u (float):           BLUE model mean U score
                green_u (float):          GREEN candidate mean U score
                u_delta (float):          green_u - blue_u
                mean_delta (float):       mean U_delta across shadow queries
                n_queries (int):          number of shadow/eval queries completed
                min_queries (int):        configured minimum (shadow_min_queries)
                threshold (float):        configured delta threshold (bg_cfg.delta)
                shadow_scores (list):     raw shadow score dicts from ShadowStore
                shadow_std_delta (float): std dev of U_delta across shadow queries
                regression_result (dict|None): regression gate output (or None)
                dry (bool):               True if no green_endpoint was available
                source (str):             "shadow (N queries)" | "synthetic eval" | "dry-run"
                specialist_config (Any):  SpecialistConfig for this specialist
                bg_config (Any):          BlueGreenFieldConfig for this specialist

        Returns:
            True to promote GREEN → BLUE, False to keep BLUE.
        """
        ...


# ── Correction Store ──────────────────────────────────────────────────────────


@runtime_checkable
class CorrectionStorePlugin(Protocol):
    """
    Replaces the built-in in-memory AssertionsStore.

    Persistent, queryable store for verified claims and DPO pairs.
    """

    def store(self, subject: str, domain: str, claim: str, confidence: float) -> None:
        """Persist a verified claim."""
        ...

    def query(self, subject: str, domain: str) -> list[dict[str, Any]]:
        """Return stored claims matching subject + domain."""
        ...

    def export_dpo_pairs(self, domain: str | None, limit: int) -> list[dict[str, Any]]:
        """Export DPO pairs for training, optionally filtered by domain."""
        ...


# ── Model Backend ─────────────────────────────────────────────────────────────


@runtime_checkable
class ModelBackendPlugin(Protocol):
    """
    Replaces the built-in vLLM/Ollama HTTP backend.

    Implement this to connect AUA to any LLM serving infrastructure.
    """

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Send a completion request and return the full response.

        Args:
            request: OpenAI-compatible dict with model, messages, temperature, etc.

        Returns:
            OpenAI-compatible response dict with choices[0].message.content
        """
        ...

    async def stream(self, request: dict[str, Any]) -> AsyncIterator[str]:
        """
        Send a streaming completion request.

        Yields token strings as they arrive.
        """
        ...

    async def health(self) -> dict[str, Any]:
        """
        Return health status of this backend.

        Returns:
            dict with at least: {"status": "ok" | "error", "latency_ms": float}
        """
        ...


# ── State Store ───────────────────────────────────────────────────────────────


@runtime_checkable
class StateStorePlugin(Protocol):
    """
    Pluggable persistent state store.

    Default implementation: SQLite (aua/state/sqlite.py).
    Alternatives: Postgres, Redis (community plugins).
    """

    def get(self, table: str, key: str) -> dict[str, Any] | None:
        """Retrieve a record by table and key."""
        ...

    def set(self, table: str, key: str, value: dict[str, Any]) -> None:
        """Upsert a record."""
        ...

    def append(self, table: str, record: dict[str, Any]) -> str:
        """Append a record to an append-only table. Returns the record ID."""
        ...

    def query(
        self,
        table: str,
        filters: dict[str, Any],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query records matching filters."""
        ...


# ── Hook ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class HookPlugin(Protocol):
    """
    Lifecycle hook. Fires at named points in the request pipeline.

    Hooks receive an event dict and return a (possibly modified) event dict.
    Return the event unchanged to pass through without modification.
    """

    async def __call__(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        Process a lifecycle event.

        Args:
            event: dict with at minimum:
                type (str): hook point name (e.g. "on_correction")
                session_id (str): query session ID
                trace_id (str): distributed trace ID

        Returns:
            (possibly modified) event dict
        """
        ...


# ── Middleware ────────────────────────────────────────────────────────────────


@runtime_checkable
class AUAMiddleware(Protocol):
    """
    Request/response middleware. Runs before and after the query pipeline.

    Middleware is ordered (YAML list order). Each middleware receives the
    output of the previous one.
    """

    async def before_query(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Process a request before it enters the query pipeline.

        Return the (possibly modified) request dict.
        Raise an exception to short-circuit (abort the request).
        """
        ...

    async def after_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Process a response after the query pipeline completes.

        Return the (possibly modified) response dict.
        """
        ...


# ── Full Utility Function Replacement (#53) ───────────────────────────────────


@runtime_checkable
class FullUtilityScorerPlugin(Protocol):
    """
    Optional extension of UtilityScorerPlugin for full U replacement (#53).

    Implement this Protocol in addition to (or instead of) UtilityScorerPlugin
    to bypass the built-in w_e·E + w_c·C + w_k·K computation entirely.

    The router checks for score_full() via hasattr() at call time, so any
    class that implements score_full() will use full replacement mode even
    without inheriting or registering this Protocol explicitly.

    Use case: non-linear utility models, field-specific scoring architectures,
    multi-objective utility functions.

    YAML: register the class as utility_scorer — no separate config key needed.

    Example (surgery domain penalises low confidence quadratically):

        class SurgeryAwareScorer:
            # score() required by UtilityScorerPlugin
            def score(self, response, field, prior_u, confidence, metadata):
                return prior_u

            # score_full() opts into full replacement (#53)
            def score_full(self, field, efficacy, confidence, curiosity,
                           weights, metadata):
                if field == "surgery":
                    return min(1.0, efficacy * (confidence ** 2))
                return (weights["w_e"]*efficacy
                        + weights["w_c"]*confidence
                        + weights["w_k"]*curiosity)
    """

    def score_full(
        self,
        field: str,
        efficacy: float,
        confidence: float,
        curiosity: float,
        weights: dict[str, float],
        metadata: dict[str, Any],
    ) -> float:
        """
        Replace the built-in utility function entirely.

        Args:
            field:      field name (e.g. "surgery", "software_engineering")
            efficacy:   E component from the built-in efficacy pipeline (0.0–1.0)
            confidence: C component from the built-in confidence pipeline (0.0–1.0)
            curiosity:  K component from the built-in curiosity pipeline (0.0–1.0)
            weights:    field config weights {"w_e": float, "w_c": float, "w_k": float}
            metadata:   context dict — query, response, pass_rate, task_score, etc.

        Returns:
            U score in [0.0, 1.0]
        """
        ...


# ── Contradiction Detector (#51) ──────────────────────────────────────────────


@runtime_checkable
class ContradictionDetectorPlugin(Protocol):
    """
    Replaces the built-in contradiction detector.

    Receives a (problem, solution) pair and returns a result dict indicating
    any contradictions found and the total confidence penalty to apply.

    YAML:
        plugins:
          contradiction_detector:
            import_path: my_plugins:MyContradictionDetector
    """

    def check(
        self,
        problem: str,
        solution: str,
        claimed_complexity: str | None = None,
    ) -> dict[str, Any]:
        """
        Check a solution for contradictions.

        Args:
            problem:             the original query / problem statement
            solution:            the specialist's response text
            claimed_complexity:  optional big-O claim extracted from the solution

        Returns:
            dict with keys:
                contradictions (list[dict]): each dict has 'type', 'description'
                confidence_penalty (float):  total penalty to subtract from confidence
                is_clean (bool):             True when no contradictions found
        """
        ...


# ── Assertion Store (#51) ─────────────────────────────────────────────────────


@runtime_checkable
class AssertionStorePlugin(Protocol):
    """
    Replaces the built-in in-memory AssertionsStore.

    Provides a persistent, queryable store for verified claims (assertions)
    used to inject prior knowledge into specialist prompts.

    Note: this is distinct from CorrectionStorePlugin which stores DPO pairs.
    AssertionStorePlugin stores claim-level knowledge; CorrectionStorePlugin
    stores training signal.

    YAML:
        plugins:
          assertion_store:
            import_path: my_plugins:PostgresAssertionStore
            config:
              dsn: postgresql://localhost/aua
    """

    def add(
        self,
        subject: str,
        domain: str,
        claim: str,
        confidence: float,
        source: str = "arbiter",
        evidence_summary: str = "",
    ) -> Any:
        """
        Persist a verified claim.

        Args:
            subject:          short identifier (e.g. "bubble_sort_complexity")
            domain:           field name (e.g. "software_engineering")
            claim:            the verified claim text
            confidence:       claim confidence 0.0–1.0
            source:           who produced this claim ("arbiter" | "empirical" | etc.)
            evidence_summary: optional provenance note

        Returns:
            implementation-defined assertion object (stored and returned)
        """
        ...

    def query(
        self,
        subject: str,
        domain: str | None = None,
        min_confidence: float | None = None,
    ) -> list[Any]:
        """
        Retrieve active assertions matching subject.

        Args:
            subject:        partial-match subject string
            domain:         optional domain filter
            min_confidence: optional minimum effective confidence

        Returns:
            list of assertion objects (implementation-defined), highest confidence first
        """
        ...

    def query_contradictions(
        self,
        subject: str,
        new_claim: str,
        domain: str | None = None,
    ) -> list[tuple[Any, str]]:
        """
        Check if new_claim contradicts any stored assertion.

        Returns:
            list of (assertion_match, contradiction_description) tuples
        """
        ...


# ── Routing Strategy (#51) ────────────────────────────────────────────────────


@runtime_checkable
class RoutingStrategyPlugin(Protocol):
    """
    Post-classifier routing hook. Receives the field classifier's probability
    distribution and may reorder, override, or augment it before the router
    decides which specialist(s) to call.

    Use cases:
      - Force traffic to a specific specialist based on external state
      - Blend classifier output with a business-rule layer
      - Implement A/B routing at the distribution level

    YAML:
        plugins:
          routing_strategy:
            import_path: my_plugins:BusinessRuleRouter
            config:
              tenant_overrides:
                tenant-a: software_engineering
    """

    def route(
        self,
        query: str,
        distribution: dict[str, float],
        metadata: dict[str, Any],
    ) -> dict[str, float]:
        """
        Adjust the domain probability distribution for a query.

        Args:
            query:        the raw user query
            distribution: classifier output {domain: probability}
            metadata:     context dict with session_id, tenant_id, trace_id, etc.

        Returns:
            adjusted distribution {domain: probability}
            Must sum to ≤ 1.0. Return distribution unchanged to pass through.
        """
        ...


# ── Scoring Component (#51) ───────────────────────────────────────────────────


@runtime_checkable
class ScoringComponentPlugin(Protocol):
    """
    Inject into a specific sub-score within the built-in U pipeline.

    Unlike UtilityScorerPlugin which replaces the entire scorer,
    ScoringComponentPlugin replaces ONE component (E, C, or K) while
    leaving the others to the built-in implementation.

    YAML:
        plugins:
          scoring_component:
            import_path: my_plugins:DomainAwareEfficacyScorer
            config:
              component: efficacy   # "efficacy" | "confidence" | "curiosity"
              baseline_source: external_api
    """

    def compute(
        self,
        component: str,
        value: float,
        field: str,
        metadata: dict[str, Any],
    ) -> float:
        """
        Compute or adjust one scoring component.

        Args:
            component: which component this is ("efficacy" | "confidence" | "curiosity")
            value:     the built-in computed value for this component
            field:     field name (e.g. "software_engineering")
            metadata:  context: {query, response, pass_rate, contradiction_penalty, ...}

        Returns:
            adjusted component value in [0.0, 1.0]
        """
        ...
