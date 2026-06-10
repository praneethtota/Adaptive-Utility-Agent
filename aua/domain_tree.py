"""
aua/domain_tree.py — Dynamic domain ontology (v1.1-veritas backport, V-P3.4).

Three-tier architecture:
  ALIAS:     raw string maps to an existing node (same effective-utility cell)
  CANDIDATE: queued, accumulating evidence, not yet resolved
  NODE:      promoted candidate with its own effective-utility cell

Normalization is two-stage:
  Stage 1: alias map lookup (O(1), no computation)
  Stage 2: edit-distance similarity against all node names + aliases
           (pure Python — no embedding model dependency)

L0 roots are fixed anchors — never deleted, always present. Everything below
L0 is dynamic and grows from specialist self-reports.

Promotion criteria (evaluated by :class:`OntologyJob` in the background):
  Gate 1: query volume    — candidate seen on ≥ K_MIN distinct queries
  Gate 2: model diversity — reported by ≥ K_MIN_MODELS distinct specialists
  Gate 3: not covered     — re-run of alias lookup still misses
  Gate 4: divergence      — mean per-specialist win-rate divergence between
                            the candidate domain and its nearest node exceeds
                            δ(d) = 0.10 + 0.05·d (branch-relative threshold)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aua.state import SQLiteStateStore

log = logging.getLogger("aua.domain_tree")

# ── Tuning constants (production-validated in AUA-Veritas) ───────────────────

ALIAS_HIGH_THRESHOLD = 0.80  # ≥ this → treat as alias, attach to node
ALIAS_LOW_THRESHOLD = 0.55  # < this → genuinely novel, candidate queue only
K_MIN = 5  # min distinct queries before a candidate is evaluated
K_MIN_MODELS = 2  # min distinct specialists that must have reported it
CANDIDATE_MAX_AGE_DAYS = 30  # prune low-evidence candidates older than this
ONTOLOGY_JOB_INTERVAL_SECONDS = 3600  # one maintenance cycle per hour

# ── L0 root seeds ─────────────────────────────────────────────────────────────
# Inserted at startup if not already present. Aligned with the framework's
# built-in field registry (software_engineering, mathematics, general, ...).

L0_ROOTS: list[dict[str, Any]] = [
    {
        "node_id": "software_engineering",
        "display_name": "Software Engineering",
        "aliases": [
            "code",
            "coding",
            "programming",
            "software development",
            "computer science",
            "software",
            "scripting",
            "swe",
        ],
    },
    {
        "node_id": "mathematics",
        "display_name": "Mathematics",
        "aliases": [
            "math",
            "maths",
            "calculus",
            "algebra",
            "statistics",
            "linear algebra",
            "discrete math",
            "probability",
        ],
    },
    {
        "node_id": "research",
        "display_name": "Research",
        "aliases": ["science", "physics", "chemistry", "biology", "scientific", "natural science"],
    },
    {
        "node_id": "law",
        "display_name": "Law",
        "aliases": [
            "legal",
            "lawyer",
            "attorney",
            "legislation",
            "regulation",
            "compliance",
            "court",
        ],
    },
    {
        "node_id": "medicine",
        "display_name": "Medicine",
        "aliases": [
            "medical",
            "health",
            "clinical",
            "healthcare",
            "doctor",
            "diagnosis",
            "treatment",
            "pharmacy",
        ],
    },
    {
        "node_id": "finance",
        "display_name": "Finance",
        "aliases": [
            "financial",
            "economics",
            "investing",
            "accounting",
            "money",
            "banking",
            "trading",
            "markets",
        ],
    },
    {
        "node_id": "writing",
        "display_name": "Writing",
        "aliases": [
            "composition",
            "editing",
            "grammar",
            "essay",
            "content",
            "copywriting",
            "proofreading",
            "drafting",
        ],
    },
    {
        "node_id": "analysis",
        "display_name": "Analysis",
        "aliases": [
            "reasoning",
            "evaluation",
            "critique",
            "comparison",
            "assessment",
            "strategy",
            "research analysis",
        ],
    },
    {
        "node_id": "history",
        "display_name": "History",
        "aliases": [
            "historical",
            "past",
            "ancient",
            "medieval",
            "modern history",
            "world history",
            "civilization",
        ],
    },
    {
        "node_id": "general",
        "display_name": "General",
        "aliases": ["misc", "miscellaneous", "other", "unknown", "various"],
    },
]


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class DomainNode:
    node_id: str
    parent_id: str | None
    depth: int
    display_name: str
    aliases: list[str]
    query_count: int = 0
    is_l0_root: bool = False
    promoted_from: str | None = None


@dataclass
class DomainCandidate:
    raw_string: str
    nearest_node: str
    similarity: float
    query_count: int = 1
    model_sources: set[str] = field(default_factory=set)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


# ── Edit-distance similarity (Stage 2 fallback) ───────────────────────────────


def _edit_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * lb
        for j, cb in enumerate(b):
            curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1))
        prev = curr
    return prev[lb]


def _edit_similarity(a: str, b: str) -> float:
    """Normalised similarity in [0, 1] based on edit distance."""
    dist = _edit_distance(a.lower(), b.lower())
    return 1.0 - dist / max(len(a), len(b), 1)


def _best_alias_similarity(raw: str, node: DomainNode) -> float:
    """Max edit-distance similarity of raw against node name + all aliases."""
    candidates = [node.display_name.lower(), node.node_id.replace("_", " ")] + [
        a.lower() for a in node.aliases
    ]
    return max(_edit_similarity(raw.lower(), c) for c in candidates)


# ── DomainTree ────────────────────────────────────────────────────────────────


class DomainTree:
    """
    In-memory domain ontology tree backed by the domain_nodes table.

    Loaded at startup from the DB. Alias additions mutate memory immediately;
    DB writes are batched and flushed by :class:`OntologyJob`.
    """

    def __init__(self, state: SQLiteStateStore) -> None:
        self._state = state
        self._nodes: dict[str, DomainNode] = {}
        self._alias_map: dict[str, str] = {}
        self._candidates: dict[str, DomainCandidate] = {}
        self._pending_alias_writes: list[tuple[str, str]] = []

        self._load_from_db()
        self._seed_l0_roots()
        self._rebuild_alias_index()
        self._load_candidates_from_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def find(self, raw: str, specialist: str = "") -> DomainNode | None:
        """
        Resolve a raw domain string to a node.

        Stage 1: exact alias-map lookup. Stage 2: best edit-distance match.

        Side effects:
          - similarity ≥ ALIAS_HIGH_THRESHOLD → added to the alias map
          - below that → added/updated in the candidate queue
        """
        normalised = raw.strip().lower().replace("-", " ").replace("_", " ")
        if not normalised:
            return self._nodes.get("general")

        for key in (normalised, raw.lower()):
            if key in self._alias_map:
                node = self._nodes.get(self._alias_map[key])
                if node:
                    return node

        best_node: DomainNode | None = None
        best_sim = 0.0
        for node in self._nodes.values():
            sim = _best_alias_similarity(normalised, node)
            if sim > best_sim:
                best_sim, best_node = sim, node

        if best_node is None:
            return self._nodes.get("general")

        if best_sim >= ALIAS_HIGH_THRESHOLD:
            self.add_alias(best_node.node_id, normalised)
            return best_node

        self._update_candidate(normalised, best_node.node_id, best_sim, specialist)

        if best_sim >= ALIAS_LOW_THRESHOLD:
            return best_node  # best-effort closest match
        return self._nodes.get("general")  # genuinely novel

    def walk_up(self, node_id: str) -> list[str]:
        """Path from node_id up to its L0 root: [node_id, ..., l0_id]."""
        path: list[str] = []
        current: str | None = node_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            path.append(current)
            node = self._nodes.get(current)
            if node is None or node.parent_id is None:
                break
            current = node.parent_id
        return path

    def is_descendant(self, node_id: str, ancestor_id: str) -> bool:
        return ancestor_id in self.walk_up(node_id)

    def get(self, node_id: str) -> DomainNode | None:
        return self._nodes.get(node_id)

    def all_nodes(self) -> list[DomainNode]:
        return list(self._nodes.values())

    def candidates(self) -> list[DomainCandidate]:
        return list(self._candidates.values())

    def delta_threshold(self, depth: int) -> float:
        """Branch-relative divergence threshold: δ(d) = 0.10 + 0.05·d."""
        return 0.10 + 0.05 * depth

    def add_alias(self, node_id: str, alias: str) -> None:
        """Add an alias to a node. Memory now; DB write queued for the job."""
        alias = alias.strip().lower()
        if not alias or alias in self._alias_map:
            return
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.aliases.append(alias)
        self._alias_map[alias] = node_id
        self._pending_alias_writes.append((node_id, alias))
        log.debug("Alias added: '%s' → %s", alias, node_id)

    def create_node(
        self,
        raw_string: str,
        parent_id: str,
        display_name: str | None = None,
    ) -> DomainNode:
        """
        Promote a candidate to a full node. Writes to DB immediately —
        promotion is a significant event, not batched.
        """
        parent = self._nodes.get(parent_id)
        depth = (parent.depth + 1) if parent else 1
        node_id = raw_string.strip().lower().replace(" ", "_").replace("-", "_")
        base, suffix = node_id, 0
        while node_id in self._nodes:
            suffix += 1
            node_id = f"{base}_{suffix}"

        node = DomainNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            display_name=display_name or raw_string.title(),
            aliases=[raw_string.lower()],
            is_l0_root=False,
            promoted_from=raw_string,
        )
        self._nodes[node_id] = node
        self._alias_map[raw_string.lower()] = node_id
        self._alias_map[node_id.replace("_", " ")] = node_id

        try:
            with self._state._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO domain_nodes"
                    " (node_id, parent_id, depth, display_name, aliases,"
                    "  query_count, is_l0_root, promoted_from, created_at)"
                    " VALUES (?,?,?,?,?,0,0,?,?)",
                    (
                        node_id,
                        parent_id,
                        depth,
                        node.display_name,
                        json.dumps(node.aliases),
                        raw_string,
                        time.time(),
                    ),
                )
                conn.execute(
                    "DELETE FROM domain_candidates WHERE raw_string=?", (raw_string.lower(),)
                )
        except Exception as e:  # noqa: BLE001
            log.error("Failed to persist new node %s: %s", node_id, e)

        self._candidates.pop(raw_string.lower(), None)
        log.info(
            "Node promoted: '%s' → %s (depth=%d, parent=%s)",
            raw_string,
            node_id,
            depth,
            parent_id,
        )
        return node

    # ── Batched persistence (flushed by OntologyJob) ──────────────────────────

    def flush_alias_writes(self) -> int:
        """Flush pending alias DB writes. Returns count written."""
        if not self._pending_alias_writes:
            return 0
        batch, self._pending_alias_writes = self._pending_alias_writes[:], []
        by_node: dict[str, list[str]] = {}
        for nid, alias in batch:
            by_node.setdefault(nid, []).append(alias)
        written = 0
        try:
            with self._state._connect() as conn:
                for nid, aliases in by_node.items():
                    row = conn.execute(
                        "SELECT aliases FROM domain_nodes WHERE node_id=?", (nid,)
                    ).fetchone()
                    if row:
                        existing = json.loads(row[0] or "[]")
                        merged = list(dict.fromkeys(existing + aliases))
                        conn.execute(
                            "UPDATE domain_nodes SET aliases=? WHERE node_id=?",
                            (json.dumps(merged), nid),
                        )
                        written += len(aliases)
        except Exception as e:  # noqa: BLE001
            log.error("Alias flush failed: %s", e)
        return written

    def flush_candidates(self) -> int:
        """Persist the candidate queue to DB. Returns count written."""
        if not self._candidates:
            return 0
        written = 0
        try:
            with self._state._connect() as conn:
                for c in self._candidates.values():
                    conn.execute(
                        "INSERT INTO domain_candidates"
                        " (raw_string, nearest_node, similarity, query_count,"
                        "  model_sources, first_seen, last_seen)"
                        " VALUES (?,?,?,?,?,?,?)"
                        " ON CONFLICT(raw_string) DO UPDATE SET"
                        "   query_count=excluded.query_count,"
                        "   model_sources=excluded.model_sources,"
                        "   last_seen=excluded.last_seen,"
                        "   similarity=excluded.similarity",
                        (
                            c.raw_string,
                            c.nearest_node,
                            c.similarity,
                            c.query_count,
                            json.dumps(sorted(c.model_sources)),
                            c.first_seen,
                            c.last_seen,
                        ),
                    )
                    written += 1
        except Exception as e:  # noqa: BLE001
            log.error("Candidate flush failed: %s", e)
        return written

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_candidate(
        self, raw: str, nearest_node: str, similarity: float, specialist: str
    ) -> None:
        c = self._candidates.get(raw)
        if c is None:
            c = DomainCandidate(raw_string=raw, nearest_node=nearest_node, similarity=similarity)
            self._candidates[raw] = c
        else:
            c.query_count += 1
            c.last_seen = time.time()
            if similarity > c.similarity:
                c.similarity, c.nearest_node = similarity, nearest_node
        if specialist:
            c.model_sources.add(specialist)

    def _load_from_db(self) -> None:
        try:
            with self._state._connect() as conn:
                rows = conn.execute("SELECT * FROM domain_nodes").fetchall()
            for r in rows:
                d = dict(r)
                self._nodes[d["node_id"]] = DomainNode(
                    node_id=d["node_id"],
                    parent_id=d["parent_id"],
                    depth=d.get("depth", 0) or 0,
                    display_name=d.get("display_name") or d["node_id"],
                    aliases=json.loads(d.get("aliases") or "[]"),
                    query_count=d.get("query_count", 0) or 0,
                    is_l0_root=bool(d.get("is_l0_root", 0)),
                    promoted_from=d.get("promoted_from"),
                )
        except Exception as e:  # noqa: BLE001
            log.warning("DomainTree load failed: %s", e)

    def _seed_l0_roots(self) -> None:
        """Insert the fixed L0 anchors if not already present."""
        for seed in L0_ROOTS:
            if seed["node_id"] in self._nodes:
                continue
            node = DomainNode(
                node_id=seed["node_id"],
                parent_id=None,
                depth=0,
                display_name=seed["display_name"],
                aliases=list(seed["aliases"]),
                is_l0_root=True,
            )
            self._nodes[node.node_id] = node
            try:
                with self._state._connect() as conn:
                    conn.execute(
                        "INSERT OR IGNORE INTO domain_nodes"
                        " (node_id, parent_id, depth, display_name, aliases,"
                        "  query_count, is_l0_root, created_at)"
                        " VALUES (?,NULL,0,?,?,0,1,?)",
                        (node.node_id, node.display_name, json.dumps(node.aliases), time.time()),
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("L0 seed failed for %s: %s", node.node_id, e)

    def _rebuild_alias_index(self) -> None:
        self._alias_map.clear()
        for node in self._nodes.values():
            self._alias_map[node.node_id.replace("_", " ")] = node.node_id
            self._alias_map[node.display_name.lower()] = node.node_id
            for alias in node.aliases:
                self._alias_map.setdefault(alias.lower(), node.node_id)

    def _load_candidates_from_db(self) -> None:
        try:
            with self._state._connect() as conn:
                rows = conn.execute("SELECT * FROM domain_candidates").fetchall()
            for r in rows:
                d = dict(r)
                self._candidates[d["raw_string"]] = DomainCandidate(
                    raw_string=d["raw_string"],
                    nearest_node=d["nearest_node"],
                    similarity=d.get("similarity", 0.0) or 0.0,
                    query_count=d.get("query_count", 1) or 1,
                    model_sources=set(json.loads(d.get("model_sources") or "[]")),
                    first_seen=d.get("first_seen") or time.time(),
                    last_seen=d.get("last_seen") or time.time(),
                )
        except Exception as e:  # noqa: BLE001
            log.warning("Candidate load failed: %s", e)


# ── OntologyJob ───────────────────────────────────────────────────────────────


class OntologyJob:
    """
    Background asyncio task that maintains the domain ontology:
    flush aliases → flush candidates → evaluate promotions → prune stale.
    CPU work runs in the default executor so the event loop never blocks.
    """

    def __init__(self, tree: DomainTree, state: SQLiteStateStore) -> None:
        self._tree = tree
        self._state = state
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the background loop. Safe to call multiple times."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            log.info("OntologyJob started")

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(ONTOLOGY_JOB_INTERVAL_SECONDS)
            try:
                await self.run_once()
            except Exception as e:  # noqa: BLE001
                log.error("OntologyJob error: %s", e, exc_info=True)

    async def run_once(self) -> dict[str, int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run_sync)

    def run_sync(self) -> dict[str, int]:
        """One full maintenance cycle (synchronous — call from executor)."""
        t0 = time.time()
        aliases_written = self._tree.flush_alias_writes()
        candidates_written = self._tree.flush_candidates()
        promoted = self._evaluate_candidates()
        pruned = self._prune_candidates()
        log.info(
            "OntologyJob cycle: aliases=%d candidates=%d promoted=%d pruned=%d (%.1fms)",
            aliases_written,
            candidates_written,
            promoted,
            pruned,
            (time.time() - t0) * 1000,
        )
        return {
            "aliases_written": aliases_written,
            "candidates_written": candidates_written,
            "promoted": promoted,
            "pruned": pruned,
        }

    def _evaluate_candidates(self) -> int:
        """Check each candidate against the four promotion gates."""
        promoted = 0
        for c in list(self._tree.candidates()):
            if c.query_count < K_MIN:
                continue  # Gate 1: query volume
            if len(c.model_sources) < K_MIN_MODELS:
                continue  # Gate 2: model diversity
            # Gate 3: not already covered by an alias
            existing = self._tree.get(self._tree._alias_map.get(c.raw_string, ""))
            if existing is not None and existing.node_id != "general":
                self._tree._candidates.pop(c.raw_string, None)
                continue
            parent = self._tree.get(c.nearest_node)
            if parent is None:
                continue
            if not self._divergence_test_passes(c, parent):
                continue  # Gate 4: divergence
            try:
                self._tree.create_node(raw_string=c.raw_string, parent_id=c.nearest_node)
                promoted += 1
            except Exception as e:  # noqa: BLE001
                log.error("Promotion failed for '%s': %s", c.raw_string, e)
        return promoted

    def _specialist_win_rate(self, specialist: str, domain: str) -> float | None:
        """Win rate of a specialist at a domain node (effective-u proxy)."""
        try:
            with self._state._connect() as conn:
                rows = conn.execute(
                    "SELECT vcg_winner FROM model_runs"
                    " WHERE specialist=? AND (domain=? OR domain_l0=?)"
                    "   AND round='answer'",
                    (specialist, domain, domain),
                ).fetchall()
            if not rows:
                return None
            return sum(1 for r in rows if r[0] == 1) / len(rows)
        except Exception:  # noqa: BLE001
            return None

    def _divergence_test_passes(self, c: DomainCandidate, parent: DomainNode) -> bool:
        """
        True when the mean per-specialist win-rate divergence between the
        candidate domain and its parent node exceeds δ(parent.depth).
        """
        delta = self._tree.delta_threshold(parent.depth)
        divergences: list[float] = []
        try:
            with self._state._connect() as conn:
                for specialist in c.model_sources:
                    rows = conn.execute(
                        "SELECT vcg_winner FROM model_runs"
                        " WHERE specialist=? AND (domain=? OR domain_path LIKE ?)"
                        "   AND round='answer'",
                        (specialist, c.raw_string, f"%{c.raw_string}%"),
                    ).fetchall()
                    if not rows:
                        continue
                    u_candidate = sum(1 for r in rows if r[0] == 1) / len(rows)
                    u_parent = self._specialist_win_rate(specialist, parent.node_id)
                    if u_parent is None:
                        continue
                    divergences.append(abs(u_candidate - u_parent))
        except Exception as e:  # noqa: BLE001
            log.warning("Divergence test error for '%s': %s", c.raw_string, e)
            return False
        if not divergences:
            return False
        mean_div = sum(divergences) / len(divergences)
        passes = mean_div > delta
        log.debug(
            "Divergence '%s' vs %s: mean=%.3f threshold=%.3f → %s",
            c.raw_string,
            parent.node_id,
            mean_div,
            delta,
            "PROMOTE" if passes else "keep",
        )
        return passes

    def _prune_candidates(self) -> int:
        """Remove candidates that are old and have low evidence."""
        max_age = CANDIDATE_MAX_AGE_DAYS * 86400
        now = time.time()
        to_prune = [
            c.raw_string
            for c in self._tree.candidates()
            if c.query_count < 2 and (now - c.first_seen) > max_age
        ]
        for raw in to_prune:
            self._tree._candidates.pop(raw, None)
        if to_prune:
            try:
                with self._state._connect() as conn:
                    conn.execute(
                        "DELETE FROM domain_candidates" " WHERE query_count < 2 AND first_seen < ?",
                        (now - max_age,),
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("Candidate pruning DB error: %s", e)
        return len(to_prune)
