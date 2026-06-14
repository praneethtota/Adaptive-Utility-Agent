"""
aua/shadow.py — Shadow mode for blue-green deployment (#48).

Shadow mode lets a GREEN specialist receive real production traffic silently:
the user always gets BLUE's response, but GREEN is called in the background
and its U score is recorded. Once enough shadow queries accumulate, the
promotion decision is based on real traffic rather than a synthetic eval.

Architecture
────────────
  ShadowStore   — thin wrapper around SQLiteStateStore for shadow_scores table
  ShadowManager — per-specialist controller; fires background shadow calls
                  and reports accumulated scores

Flow per query (when shadow is active for a specialist):
  1. Router returns BLUE's response to the user (no latency impact)
  2. fire_and_forget(_shadow_call()) dispatches GREEN in the background
  3. GREEN's response is scored with the same _score() logic as BLUE
  4. (blue_u, green_u, delta) written to shadow_scores table
  5. GET /deploy/shadow/{specialist} reports aggregated scores at any time
  6. POST /deploy/green uses accumulated shadow scores instead of a
     synthetic eval run when shadow_n_queries >= shadow_min_queries

Shadow score table schema (shadow_scores):
  id, specialist, query, blue_u, green_u, u_delta, domain, created_at

Configuration (aua_config.yaml):

    blue_green:
      swe:
        delta: 0.025
        shadow_endpoint: http://localhost:9011/v1/chat/completions
        shadow_min_queries: 50   # queries before promotion is allowed

REST endpoints (added by Router):
  POST /deploy/shadow/{specialist}  — register/update the GREEN endpoint
  GET  /deploy/shadow/{specialist}  — report accumulated shadow scores
  DELETE /deploy/shadow/{specialist} — deactivate shadow mode, clear scores
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aua.state import SQLiteStateStore

log = logging.getLogger(__name__)


# ── Shadow report ─────────────────────────────────────────────────────────────


@dataclass
class ShadowReport:
    """Aggregated shadow score report for one specialist."""

    specialist: str
    shadow_endpoint: str | None
    n_queries: int
    min_queries: int
    blue_mean_u: float
    green_mean_u: float
    mean_delta: float
    ready_to_promote: bool  # n_queries >= min_queries AND mean_delta >= threshold
    threshold: float
    active: bool  # shadow endpoint is currently registered

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialist": self.specialist,
            "shadow_endpoint": self.shadow_endpoint,
            "n_queries": self.n_queries,
            "min_queries": self.min_queries,
            "blue_mean_u": round(self.blue_mean_u, 4),
            "green_mean_u": round(self.green_mean_u, 4),
            "mean_delta": round(self.mean_delta, 4),
            "ready_to_promote": self.ready_to_promote,
            "threshold": self.threshold,
            "active": self.active,
            "progress": f"{self.n_queries}/{self.min_queries} shadow queries",
        }


# ── ShadowStore ───────────────────────────────────────────────────────────────


class ShadowStore:
    """
    Thin wrapper around SQLiteStateStore for shadow_scores reads/writes.
    The table is created by the migration in state.py _MIGRATIONS.
    """

    def __init__(self, store: SQLiteStateStore) -> None:
        self._store = store

    def record(
        self,
        specialist: str,
        query: str,
        blue_u: float,
        green_u: float,
        domain: str,
    ) -> str:
        """Write one shadow score pair. Returns the row id."""
        return self._store.append(
            "shadow_scores",
            {
                "specialist": specialist,
                "query": query[:500],  # cap query length
                "blue_u": blue_u,
                "green_u": green_u,
                "u_delta": round(green_u - blue_u, 6),
                "domain": domain,
            },
        )

    def get_scores(self, specialist: str, limit: int = 1000) -> list[dict[str, Any]]:
        """Return all shadow score rows for a specialist, newest first."""
        return self._store.query(
            "shadow_scores",
            filters={"specialist": specialist},
            limit=limit,
            order_by="created_at DESC",
        )

    def clear(self, specialist: str) -> int:
        """Delete all shadow scores for a specialist. Returns rows deleted."""
        with self._store._connect() as conn:
            cursor = conn.execute("DELETE FROM shadow_scores WHERE specialist = ?", (specialist,))
            return cursor.rowcount

    def aggregate(self, specialist: str) -> dict[str, Any]:
        """Return aggregate stats: n, mean_blue_u, mean_green_u, mean_delta."""
        rows = self.get_scores(specialist)
        if not rows:
            return {
                "n": 0,
                "mean_blue_u": 0.0,
                "mean_green_u": 0.0,
                "mean_delta": 0.0,
            }
        n = len(rows)
        mean_blue = sum(r["blue_u"] for r in rows) / n
        mean_green = sum(r["green_u"] for r in rows) / n
        mean_delta = sum(r["u_delta"] for r in rows) / n
        return {
            "n": n,
            "mean_blue_u": round(mean_blue, 4),
            "mean_green_u": round(mean_green, 4),
            "mean_delta": round(mean_delta, 4),
        }


# ── ShadowManager ─────────────────────────────────────────────────────────────


class ShadowManager:
    """
    Per-router controller for shadow mode.

    Tracks which specialists have an active GREEN shadow endpoint, fires
    background shadow calls after every BLUE response, and produces
    ShadowReport objects for the /deploy/shadow/{specialist} endpoint.

    Thread-safe: all state mutations use a dict with GIL protection;
    the background _shadow_call coroutines are fire-and-forget and never
    mutate shared mutable state (they only write to the DB).
    """

    def __init__(self, shadow_store: ShadowStore) -> None:
        self._store = shadow_store
        # specialist_name → {"endpoint": str, "min_queries": int, "threshold": float}
        self._active: dict[str, dict[str, Any]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def activate(
        self,
        specialist: str,
        endpoint: str,
        min_queries: int = 50,
        threshold: float = 0.025,
    ) -> None:
        """
        Register a GREEN endpoint for a specialist. Shadow calls will begin
        firing on the next query routed to that specialist.
        """
        self._active[specialist] = {
            "endpoint": endpoint,
            "min_queries": min_queries,
            "threshold": threshold,
        }
        log.info(
            "ShadowManager: shadow mode activated for %s → %s (min=%d, threshold=%.3f)",
            specialist,
            endpoint,
            min_queries,
            threshold,
        )

    def deactivate(self, specialist: str, clear_scores: bool = False) -> None:
        """Deactivate shadow mode for a specialist."""
        self._active.pop(specialist, None)
        if clear_scores:
            n = self._store.clear(specialist)
            log.info("ShadowManager: cleared %d shadow scores for %s", n, specialist)
        log.info("ShadowManager: shadow mode deactivated for %s", specialist)

    def is_active(self, specialist: str) -> bool:
        return specialist in self._active

    def shadow_endpoint(self, specialist: str) -> str | None:
        return self._active.get(specialist, {}).get("endpoint")

    # ── Report ────────────────────────────────────────────────────────────────

    def report(self, specialist: str) -> ShadowReport:
        """Build a ShadowReport for the /deploy/shadow/{specialist} endpoint."""
        cfg = self._active.get(specialist, {})
        agg = self._store.aggregate(specialist)
        min_q = cfg.get("min_queries", 50)
        threshold = cfg.get("threshold", 0.025)
        ready = agg["n"] >= min_q and agg["mean_delta"] >= threshold
        return ShadowReport(
            specialist=specialist,
            shadow_endpoint=cfg.get("endpoint"),
            n_queries=agg["n"],
            min_queries=min_q,
            blue_mean_u=agg["mean_blue_u"],
            green_mean_u=agg["mean_green_u"],
            mean_delta=agg["mean_delta"],
            ready_to_promote=ready,
            threshold=threshold,
            active=specialist in self._active,
        )

    # ── Background shadow call (called from router via fire_and_forget) ────────

    async def shadow_call(
        self,
        specialist: str,
        query: str,
        domain: str,
        blue_u: float,
        call_fn: Any,  # router._call coroutine
        score_fn: Any,  # router._score coroutine
        model_name: str = "green",
    ) -> None:
        """
        Fire a shadow call to GREEN and record the score pair.
        Never raises — all errors are logged and swallowed.
        Called via fire_and_forget so it never blocks the user response.
        """
        cfg = self._active.get(specialist)
        if cfg is None:
            return  # shadow deactivated between dispatch and execution

        endpoint = cfg["endpoint"]
        try:
            t0 = time.time()
            text, conf = await call_fn(endpoint, query, domain, model_name=model_name)
            latency_ms = (time.time() - t0) * 1000
            green_u, *_ = await score_fn(query, text, domain, conf)
            self._store.record(
                specialist=specialist,
                query=query,
                blue_u=blue_u,
                green_u=float(green_u),
                domain=domain,
            )
            log.debug(
                "Shadow %s: blue_u=%.4f green_u=%.4f delta=%.4f latency=%.0fms",
                specialist,
                blue_u,
                green_u,
                green_u - blue_u,
                latency_ms,
            )
        except Exception as e:
            log.debug("Shadow call failed for %s: %s", specialist, e)
