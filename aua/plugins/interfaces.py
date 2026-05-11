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

    Receives a specialist response and domain context, returns a scalar U score.
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
        Compute utility score for a specialist response.

        Args:
            response:   the specialist's text output
            field:      field name (e.g. "software_engineering")
            prior_u:    the running mean U for this specialist
            confidence: Kalman-filtered confidence estimate (0.0–1.0)
            metadata:   arbitrary context (session_id, query, latency_ms, etc.)

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
