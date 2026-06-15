"""
aua/batch_queue.py — Persistent batch inference queue (#56).

Architecture
────────────
  POST /batch/jobs        → enqueue a batch job, return job_id immediately
  GET  /batch/jobs/{id}  → poll status and partial results
  GET  /batch/jobs        → list recent jobs

Each job holds N queries that are dispatched to the router independently,
but tracked under a single job_id. Results stream into the DB as they
complete — the caller never has to wait for the whole batch.

Priority lanes
──────────────
  "high"   — dispatched first, bypass normal queue
  "normal" — default
  "low"    — cost-optimised; run after high + normal are drained

The background worker (BatchWorker) runs as an asyncio task inside the
FastAPI lifespan. It pulls the next pending item off the priority queue and
dispatches it to the router _handle() coroutine, writing results atomically.

Persistence
───────────
  batch_jobs   — one row per submitted job
  batch_items  — one row per query inside a job (holds result when done)

Both tables are added to the SQLite state store via _BATCH_SCHEMA migration.
The worker survives router restarts: on startup it re-queues any items that
were "running" when the server last stopped (they were interrupted mid-flight).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aua.state import SQLiteStateStore

log = logging.getLogger(__name__)

# ── Schema additions (applied via migration in state.py) ──────────────────────

BATCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_jobs (
    job_id       TEXT PRIMARY KEY,
    priority     TEXT NOT NULL DEFAULT 'normal',   -- 'high' | 'normal' | 'low'
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
    n_queries    INTEGER NOT NULL,
    n_done       INTEGER NOT NULL DEFAULT 0,
    n_errors     INTEGER NOT NULL DEFAULT 0,
    session_id   TEXT,
    max_parallel INTEGER NOT NULL DEFAULT 4,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    meta         TEXT DEFAULT '{}'               -- JSON: caller-supplied metadata
);

CREATE TABLE IF NOT EXISTS batch_items (
    item_id      TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL,
    query        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending', -- pending|running|done|error
    result       TEXT,                            -- JSON RouterResponse or null
    error        TEXT,                            -- error message if status=error
    created_at   REAL NOT NULL,
    finished_at  REAL
);

CREATE INDEX IF NOT EXISTS idx_batch_items_job  ON batch_items(job_id);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status, priority, created_at);
"""

# Priority ordering for the queue: lower number = higher priority
_PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}


# ── Public data objects ───────────────────────────────────────────────────────


class BatchJob:
    """In-memory view of a batch_jobs row."""

    __slots__ = (
        "job_id",
        "priority",
        "status",
        "n_queries",
        "n_done",
        "n_errors",
        "session_id",
        "max_parallel",
        "created_at",
        "started_at",
        "finished_at",
        "meta",
    )

    def __init__(self, row: dict[str, Any]) -> None:
        self.job_id = row["job_id"]
        self.priority = row.get("priority", "normal")
        self.status = row.get("status", "pending")
        self.n_queries = row["n_queries"]
        self.n_done = row.get("n_done", 0)
        self.n_errors = row.get("n_errors", 0)
        self.session_id = row.get("session_id")
        self.max_parallel = row.get("max_parallel", 4)
        self.created_at = row.get("created_at", 0.0)
        self.started_at = row.get("started_at")
        self.finished_at = row.get("finished_at")
        self.meta = json.loads(row.get("meta") or "{}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "priority": self.priority,
            "status": self.status,
            "n_queries": self.n_queries,
            "n_done": self.n_done,
            "n_errors": self.n_errors,
            "session_id": self.session_id,
            "max_parallel": self.max_parallel,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "meta": self.meta,
        }


# ── BatchQueue — the DB-backed queue ─────────────────────────────────────────


class BatchQueue:
    """
    Enqueue, poll, and manage batch jobs backed by the SQLite state store.

    All methods are synchronous (called from async contexts via run_in_executor
    or from the sync worker loop). Thread-safe because SQLiteStateStore uses
    WAL mode with per-connection locking.
    """

    def __init__(self, store: SQLiteStateStore) -> None:
        self._store = store
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Apply batch tables if they don't exist yet."""
        import sqlite3

        try:
            with self._store._connect() as conn:
                conn.executescript(BATCH_SCHEMA)
        except sqlite3.OperationalError as e:
            log.warning("BatchQueue schema init warning: %s", e)

    # ── Submit ────────────────────────────────────────────────────────────────

    def submit(
        self,
        queries: list[str],
        priority: str = "normal",
        session_id: str | None = None,
        max_parallel: int = 4,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """
        Enqueue a batch of queries. Returns job_id immediately.

        Args:
            queries:      list of query strings (1–500)
            priority:     'high' | 'normal' | 'low'
            session_id:   shared session for all queries in this job
            max_parallel: max concurrent specialist calls within this job
            meta:         arbitrary caller metadata stored with the job

        Returns:
            job_id (UUID string)
        """
        if priority not in _PRIORITY_ORDER:
            priority = "normal"
        if not 1 <= len(queries) <= 500:
            raise ValueError(f"queries must be 1–500 items, got {len(queries)}")

        job_id = str(uuid.uuid4())
        now = time.time()

        with self._store._connect() as conn:
            conn.execute(
                """
                INSERT INTO batch_jobs
                    (job_id, priority, status, n_queries, n_done, n_errors,
                     session_id, max_parallel, created_at, meta)
                VALUES (?, ?, 'pending', ?, 0, 0, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    priority,
                    len(queries),
                    session_id,
                    max_parallel,
                    now,
                    json.dumps(meta or {}),
                ),
            )
            conn.executemany(
                """
                INSERT INTO batch_items (item_id, job_id, query, status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                [(str(uuid.uuid4()), job_id, q, now) for q in queries],
            )

        log.info("BatchQueue.submit job_id=%s n=%d priority=%s", job_id, len(queries), priority)
        return job_id

    # ── Poll ──────────────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """
        Return the full job status plus all item results collected so far.

        Returns None if job_id is unknown.
        """
        with self._store._connect() as conn:
            row = conn.execute("SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            job = dict(row)

            items = conn.execute(
                "SELECT * FROM batch_items WHERE job_id = ? ORDER BY created_at ASC",
                (job_id,),
            ).fetchall()

        results = []
        errors = []
        pending_count = 0
        running_count = 0

        for item in items:
            item = dict(item)
            if item["status"] == "done" and item["result"]:
                try:
                    results.append(json.loads(item["result"]))
                except json.JSONDecodeError:
                    pass
            elif item["status"] == "error":
                errors.append(
                    {"item_id": item["item_id"], "query": item["query"], "error": item["error"]}
                )
            elif item["status"] == "pending":
                pending_count += 1
            elif item["status"] == "running":
                running_count += 1

        return {
            **job,
            "meta": json.loads(job.get("meta") or "{}"),
            "results": results,
            "errors": errors,
            "n_pending": pending_count,
            "n_running": running_count,
        }

    def list_jobs(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List recent jobs, optionally filtered by status."""
        with self._store._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM batch_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM batch_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        return [{**dict(r), "meta": json.loads(dict(r).get("meta") or "{}")} for r in rows]

    # ── Internal queue operations (called by BatchWorker) ─────────────────────

    def next_pending_job(self) -> dict[str, Any] | None:
        """
        Return the highest-priority oldest-pending job, or None if queue is empty.
        Priority: high → normal → low, then by created_at ASC within each tier.
        """
        with self._store._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM batch_jobs
                WHERE status = 'pending'
                ORDER BY
                    CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END ASC,
                    created_at ASC
                LIMIT 1
                """,
            ).fetchone()
        return dict(row) if row else None

    def claim_job(self, job_id: str) -> bool:
        """
        Atomically transition a job from pending → running.
        Returns True if the claim succeeded (no other worker took it).
        """
        with self._store._connect() as conn:
            cursor = conn.execute(
                "UPDATE batch_jobs SET status='running', started_at=? WHERE job_id=? AND status='pending'",
                (time.time(), job_id),
            )
            return cursor.rowcount == 1

    def pending_items(self, job_id: str) -> list[dict[str, Any]]:
        """Return all pending items for a job, ordered by created_at."""
        with self._store._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM batch_items WHERE job_id=? AND status='pending' ORDER BY created_at ASC",
                (job_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_item_running(self, item_id: str) -> None:
        with self._store._connect() as conn:
            conn.execute(
                "UPDATE batch_items SET status='running' WHERE item_id=?",
                (item_id,),
            )

    def mark_item_done(self, item_id: str, result: dict[str, Any]) -> None:
        with self._store._connect() as conn:
            conn.execute(
                "UPDATE batch_items SET status='done', result=?, finished_at=? WHERE item_id=?",
                (json.dumps(result), time.time(), item_id),
            )
            conn.execute(
                "UPDATE batch_jobs SET n_done = n_done + 1 WHERE job_id = "
                "(SELECT job_id FROM batch_items WHERE item_id=?)",
                (item_id,),
            )

    def mark_item_error(self, item_id: str, error: str) -> None:
        with self._store._connect() as conn:
            conn.execute(
                "UPDATE batch_items SET status='error', error=?, finished_at=? WHERE item_id=?",
                (error, time.time(), item_id),
            )
            conn.execute(
                "UPDATE batch_jobs SET n_errors = n_errors + 1 WHERE job_id = "
                "(SELECT job_id FROM batch_items WHERE item_id=?)",
                (item_id,),
            )

    def finish_job(self, job_id: str) -> None:
        with self._store._connect() as conn:
            conn.execute(
                "UPDATE batch_jobs SET status='done', finished_at=? WHERE job_id=?",
                (time.time(), job_id),
            )

    def fail_job(self, job_id: str, reason: str = "") -> None:
        with self._store._connect() as conn:
            conn.execute(
                "UPDATE batch_jobs SET status='failed', finished_at=?, meta=json_set(meta, '$.fail_reason', ?) WHERE job_id=?",
                (time.time(), reason, job_id),
            )

    def recover_interrupted(self) -> int:
        """
        On startup: reset any jobs/items left in 'running' state back to 'pending'.
        These were interrupted mid-flight when the server last stopped.
        Returns the number of jobs reset.
        """
        with self._store._connect() as conn:
            cursor = conn.execute(
                "UPDATE batch_jobs SET status='pending', started_at=NULL WHERE status='running'"
            )
            conn.execute("UPDATE batch_items SET status='pending' WHERE status='running'")
            n = cursor.rowcount
        if n:
            log.info("BatchQueue.recover_interrupted: reset %d interrupted job(s) to pending", n)
        return n


# ── BatchWorker — the async background dispatcher ─────────────────────────────


class BatchWorker:
    """
    Asyncio background task that drains the batch queue.

    Runs inside the FastAPI lifespan. Picks the highest-priority pending job,
    dispatches all its items concurrently (bounded by max_parallel), writes
    results as they arrive, then moves to the next job.

    The _handle callable is router._handle — injected at construction time
    to avoid a circular import.
    """

    POLL_INTERVAL = 2.0  # seconds between queue-empty polls

    def __init__(
        self,
        queue: BatchQueue,
        handle_fn: Any,  # Callable[[QueryRequest], Awaitable[RouterResponse]]
        middleware: Any | None = None,  # MiddlewarePipeline — for before/after_batch hooks (#52)
    ) -> None:
        self._queue = queue
        self._handle = handle_fn
        self._middleware = middleware  # injected at router startup
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._running = True
        self._task = asyncio.get_event_loop().create_task(self._loop())
        log.info("BatchWorker started")

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        log.info("BatchWorker stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                job_row = self._queue.next_pending_job()
                if job_row is None:
                    await asyncio.sleep(self.POLL_INTERVAL)
                    continue

                job_id = job_row["job_id"]
                if not self._queue.claim_job(job_id):
                    # Another worker claimed it first (future multi-worker setup)
                    await asyncio.sleep(0.1)
                    continue

                await self._dispatch_job(job_id, job_row)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("BatchWorker loop error: %s", e, exc_info=True)
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _dispatch_job(self, job_id: str, job_row: dict[str, Any]) -> None:
        from aua.endpoints import QueryRequest

        items = self._queue.pending_items(job_id)
        if not items:
            self._queue.finish_job(job_id)
            return

        max_parallel = job_row.get("max_parallel", 4)
        session_id = job_row.get("session_id") or str(uuid.uuid4())
        sem = asyncio.Semaphore(max_parallel)

        # #52: before_batch middleware hook
        _job_meta = dict(job_row)
        _job_meta["job_id"] = job_id
        _job_meta["n_queries"] = len(items)
        if self._middleware and self._middleware.registered():
            _job_meta = await self._middleware.before_batch(_job_meta)

        log.info(
            "BatchWorker dispatching job_id=%s n_items=%d max_parallel=%d",
            job_id,
            len(items),
            max_parallel,
        )

        _results: list[dict[str, Any]] = []

        async def _run_item(item: dict[str, Any]) -> None:
            item_id = item["item_id"]
            async with sem:
                self._queue.mark_item_running(item_id)
                try:
                    req = QueryRequest(
                        query=item["query"],
                        session_id=session_id,
                        conversation_history=[],
                        force_domain=None,
                    )
                    resp = await self._handle(req)
                    result = resp.model_dump()
                    self._queue.mark_item_done(item_id, result)
                    _results.append(result)
                except Exception as e:
                    log.error("BatchWorker item %s failed: %s", item_id, e)
                    _results.append({"error": str(e), "item_id": item_id})
                    self._queue.mark_item_error(item_id, str(e))

        await asyncio.gather(*[_run_item(item) for item in items])

        # #52: after_batch middleware hook
        if self._middleware and self._middleware.registered():
            _results = await self._middleware.after_batch(_job_meta, _results)

        self._queue.finish_job(job_id)
        log.info("BatchWorker finished job_id=%s", job_id)
