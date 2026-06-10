"""
aua/keywords.py — Full-text keyword search (v1.1-veritas backport, V-P1.1).

Two parts:

  1. ``extract()`` — pure-Python keyword extraction (~8µs/call, no spaCy).
     Splits on whitespace/punctuation, filters stopwords, boosts technical
     identifiers (CamelCase / snake_case), keeps years and multi-digit numbers.

  2. ``KeywordIndex`` — in-memory inverted index over ``message_keywords``:
       keyword → {(conversation_id, message_id, created_at), ...}
     backed by a sorted keyword list for O(log n) prefix matching via bisect.
     Multi-word queries use AND semantics at the message level (Cmd+F model).

Production rules carried forward from AUA-Veritas Phase 13:

  * Closure-scope trap: the async worker imports ``time`` locally inside the
    coroutine — a module imported later in an enclosing body is NOT in scope
    when the task first runs. This bug silently broke search for 3 months.
  * Startup backfill required: the async worker is killed on process restart
    before flushing. ``build_from_db()`` scans ``messages`` for rows not yet
    in ``message_keywords`` and indexes them; without this, search returns
    empty after every rebuild.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aua.state import SQLiteStateStore

log = logging.getLogger("aua.keywords")

# ── Keyword extraction ────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "not",
        "no",
        "with",
        "from",
        "by",
        "about",
        "than",
        "then",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "all",
        "any",
        "some",
        "more",
        "very",
        "just",
        "also",
        "so",
        "up",
        "out",
        "if",
        "into",
        "as",
        "well",
        "here",
        "there",
        "get",
        "use",
        "make",
        "want",
        "need",
        "let",
        "know",
        "think",
        "say",
        "tell",
        "ask",
        "give",
        "show",
        "work",
        "try",
        "look",
        "seem",
        "become",
        "certainly",
        "sure",
        "yes",
        "please",
        "like",
        "new",
        "one",
        "two",
        "hi",
        "cant",
        "dont",
        "doesnt",
        "isnt",
        "arent",
        "wasnt",
        "werent",
    }
)

_TECH_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:_[a-z]+)+)\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NUM_RE = re.compile(r"\b\d{2,}\b")
_SPLIT_RE = re.compile(r"[\s\.,!?;:()\[\]{}\"'`\-/\\|@#$%\^&*+=<>~]+")


def extract(text: str, max_keywords: int = 40) -> list[str]:
    """Extract normalized keywords. Pure Python, no external deps."""
    if not text or len(text.strip()) < 3:
        return []
    seen: set[str] = set()
    result: list[str] = []

    def add(token: str) -> None:
        t = token.lower().strip(".,!?;:'\"()-_")
        if len(t) >= 3 and t not in _STOPWORDS and t not in seen and not t.isdigit():
            seen.add(t)
            result.append(t)

    # Technical identifiers first so they survive the max_keywords cap
    for tech in _TECH_RE.findall(text):
        add(tech)
    for tok in _SPLIT_RE.split(text):
        if tok:
            add(tok)
    # Years and multi-digit numbers bypass the isdigit filter in add() —
    # "error 404" and "2024 report" must be searchable. (In Veritas the
    # year path was dead code because add() rejected pure-digit tokens.)
    for num in _YEAR_RE.findall(text) + _NUM_RE.findall(text):
        if num not in seen:
            seen.add(num)
            result.append(num)
    return result[:max_keywords]


def _prefix_upper_bound(prefix: str) -> str:
    """Smallest string strictly greater than every string with this prefix."""
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


# ── In-memory inverted index ──────────────────────────────────────────────────


class KeywordIndex:
    """
    Message-level keyword inverted index with an async extraction queue.

    Hot path:    enqueue() — O(1), never blocks the response path.
    Background:  worker() — drains the queue in 50ms/20-item batches,
                 single DB transaction per batch.
    Startup:     build_from_db() — load existing rows + backfill unindexed
                 messages (required: the worker dies unflushed on restart).
    Search:      search() — pure in-memory set ops, AND semantics per message.
    """

    BATCH_MAX = 20  # flush after this many items
    BATCH_WAIT = 0.05  # or after this many seconds (50ms)
    BACKFILL_CAP = 5000  # cap so startup isn't delayed on huge DBs

    def __init__(self, state_store: SQLiteStateStore) -> None:
        self._state = state_store
        self._index: dict[str, set[tuple[str, str, float]]] = {}
        self._sorted: list[str] = []
        self._queue: asyncio.Queue[tuple[str, str, str, str]] | None = None
        self._worker_task: asyncio.Task[None] | None = None

    # ── Index mutation ────────────────────────────────────────────────────────

    def add(
        self,
        keywords: list[str],
        conversation_id: str,
        message_id: str = "",
        created_at: float | None = None,
    ) -> None:
        """Add keyword → (conv, msg, ts) mappings to the in-memory index."""
        ts = created_at or time.time()
        needs_resort = False
        for kw in keywords:
            if kw not in self._index:
                self._index[kw] = set()
                needs_resort = True
            self._index[kw].add((conversation_id, message_id, ts))
        if needs_resort:
            self._sorted = sorted(self._index.keys())

    # ── Search ────────────────────────────────────────────────────────────────

    def _prefix_hits(self, prefix: str) -> set[tuple[str, str, float]]:
        """All (conv_id, msg_id, ts) tuples for keywords starting with prefix."""
        result: set[tuple[str, str, float]] = set()
        if not prefix:
            return result
        if prefix in self._index:
            result.update(self._index[prefix])
        lo = bisect.bisect_left(self._sorted, prefix)
        hi = bisect.bisect_left(self._sorted, _prefix_upper_bound(prefix))
        for i in range(lo, min(hi, len(self._sorted))):
            kw = self._sorted[i]
            if not kw.startswith(prefix):
                break
            result.update(self._index[kw])
        return result

    def search(self, query: str, limit: int = 500) -> list[dict[str, Any]]:
        """
        Message-level search (Cmd+F model): one result per matching message.
        AND semantics — every query word must appear in the same message.
        Results sorted newest-first.
        """
        words = [w.lower().strip() for w in query.split() if len(w.strip()) >= 2]
        if not words:
            return []
        hit_sets = [self._prefix_hits(word) for word in words]
        if not hit_sets or not all(hit_sets):
            return []
        if len(hit_sets) > 1:
            msg_id_sets = [{h[1] for h in s} for s in hit_sets]
            common = set.intersection(*msg_id_sets)
            hits = [h for h in hit_sets[0] if h[1] in common]
        else:
            hits = list(hit_sets[0])
        hits.sort(key=lambda h: h[2], reverse=True)
        return [{"conversation_id": h[0], "message_id": h[1], "ts": h[2]} for h in hits[:limit]]

    def search_db_fallback(self, query: str, limit: int = 500) -> list[dict[str, Any]]:
        """DB-backed search used before the in-memory index is built."""
        words = [w.lower().strip() for w in query.split() if len(w.strip()) >= 2]
        if not words:
            return []
        with self._state._connect() as conn:
            msg_maps: list[dict[str, tuple[str, float]]] = []
            for word in words:
                rows = conn.execute(
                    "SELECT DISTINCT message_id, conversation_id, created_at"
                    " FROM message_keywords WHERE keyword >= ? AND keyword < ?"
                    " ORDER BY created_at DESC LIMIT 1000",
                    (word, _prefix_upper_bound(word)),
                ).fetchall()
                msg_maps.append({r[0]: (r[1], r[2]) for r in rows})
        if not msg_maps or not all(msg_maps):
            return []
        common_ids = set(msg_maps[0].keys())
        for m in msg_maps[1:]:
            common_ids &= set(m.keys())
        hits = [
            {"message_id": mid, "conversation_id": msg_maps[0][mid][0], "ts": msg_maps[0][mid][1]}
            for mid in common_ids
        ]
        hits.sort(key=lambda h: float(h["ts"]), reverse=True)  # type: ignore[arg-type]
        return hits[:limit]

    @property
    def ready(self) -> bool:
        """True once the in-memory index holds at least one keyword."""
        return bool(self._index)

    def size(self) -> int:
        return len(self._index)

    # ── Async extraction pipeline ─────────────────────────────────────────────

    def enqueue(self, message_id: str, conversation_id: str, role: str, text: str) -> None:
        """Enqueue a message for keyword extraction. Returns immediately."""
        if self._queue is not None and text and len(text.strip()) >= 3:
            try:
                self._queue.put_nowait((message_id, conversation_id, role, text))
            except Exception:  # noqa: BLE001 — queue full / loop closed: drop, never block
                pass

    def _process_batch(self, batch: list[tuple[str, str, str, str]]) -> None:
        """Extract + persist a batch in a single DB transaction, then index."""
        now = time.time()
        extracted: list[tuple[str, str, str, str, float]] = []
        for message_id, conversation_id, role, text in batch:
            try:
                for kw in extract(text):
                    extracted.append((kw, message_id, conversation_id, role, now))
            except Exception as e:  # noqa: BLE001
                log.debug("Keyword extract failed for msg %s: %s", message_id, e)
        if not extracted:
            return
        try:
            with self._state._connect() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO message_keywords"
                    " (keyword, message_id, conversation_id, role, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    extracted,
                )
        except Exception as e:  # noqa: BLE001
            log.warning("Keyword batch DB write failed: %s", e)
        by_msg: dict[tuple[str, str], list[str]] = {}
        for kw, mid, cid, _role, _ts in extracted:
            by_msg.setdefault((cid, mid), []).append(kw)
        for (cid, mid), kws in by_msg.items():
            self.add(kws, cid, message_id=mid, created_at=now)

    async def worker(self) -> None:
        """
        Drain the extraction queue as a background asyncio task.

        Batches DB writes: accumulates items for up to 50ms or 20 messages,
        whichever comes first, then flushes in a single transaction.
        """
        # Closure-scope rule (Phase 13): import time locally — a module
        # imported later in an enclosing body is not in scope here.
        import time as _kw_tm

        assert self._queue is not None
        while True:
            batch: list[tuple[str, str, str, str]] = [await self._queue.get()]
            deadline = _kw_tm.time() + self.BATCH_WAIT
            while len(batch) < self.BATCH_MAX:
                remaining = deadline - _kw_tm.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            self._process_batch(batch)

    def start(self) -> None:
        """Create the queue and spawn the background worker (event loop required)."""
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self.worker())

    def stop(self) -> None:
        """Cancel the background worker (clean shutdown)."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None

    # ── Startup: load + backfill ──────────────────────────────────────────────

    def build_from_db(self) -> dict[str, int]:
        """
        Load the index from message_keywords, then backfill any messages not
        yet indexed (the async worker dies unflushed on process restart —
        without this, search returns empty after every rebuild).
        """
        loaded = backfilled = 0
        now = time.time()
        try:
            with self._state._connect() as conn:
                rows = conn.execute(
                    "SELECT keyword, conversation_id, message_id, created_at"
                    " FROM message_keywords"
                ).fetchall()
            for kw, cid, mid, ts in rows:
                if kw not in self._index:
                    self._index[kw] = set()
                self._index[kw].add((cid, mid or "", ts or now))
            loaded = len(rows)

            with self._state._connect() as conn:
                unindexed = conn.execute(
                    "SELECT message_id, conversation_id, role, content FROM messages"
                    " WHERE message_id NOT IN"
                    "   (SELECT DISTINCT message_id FROM message_keywords)"
                    " AND content IS NOT NULL AND length(content) > 3"
                    " LIMIT ?",
                    (self.BACKFILL_CAP,),
                ).fetchall()
            if unindexed:
                inserts: list[tuple[str, str, str, str, float]] = []
                for mid, cid, role, content in unindexed:
                    text = content or ""
                    if len(text.strip()) < 3:
                        continue
                    kws = extract(text, max_keywords=30)
                    for kw in kws:
                        inserts.append((kw, mid, cid, role or "user", now))
                    self.add(kws, cid, message_id=mid, created_at=now)
                if inserts:
                    with self._state._connect() as conn:
                        conn.executemany(
                            "INSERT OR IGNORE INTO message_keywords"
                            " (keyword, message_id, conversation_id, role, created_at)"
                            " VALUES (?, ?, ?, ?, ?)",
                            inserts,
                        )
                backfilled = len(unindexed)
            self._sorted = sorted(self._index.keys())
        except Exception as e:  # noqa: BLE001
            log.warning("Keyword index load skipped: %s", e)
        log.info(
            "Keyword index built: %d rows loaded, %d messages backfilled, %d unique keywords",
            loaded,
            backfilled,
            len(self._index),
        )
        return {"loaded": loaded, "backfilled": backfilled, "unique_keywords": len(self._index)}

    # ── Synchronous immediate indexing (used in tests / non-async contexts) ──

    def index_message_now(
        self, message_id: str | None, conversation_id: str, role: str, text: str
    ) -> int:
        """Extract + persist keywords for one message synchronously."""
        mid = message_id or str(uuid.uuid4())
        self._process_batch([(mid, conversation_id, role, text)])
        return len(extract(text))
