"""
aua/state.py — Pluggable state store with SQLite default.

The state store is the single source of truth for all persistent AUA state:
  - Promotion log (replaces .aua/state/promotions.jsonl)
  - Corrections / DPO pairs (replaces dpo_pairs/*.jsonl)
  - Assertions (replaces in-memory AssertionsStore)
  - Sessions (new in v0.8)
  - Audit log (new in v0.8, append-only with hash chain)

Configuration:
    state:
      backend: sqlite          # "sqlite" | "files" (v0.7 compat) | "postgres"
      path: .aua/state/aua.db  # SQLite only
      url: postgres://...      # Postgres only (v0.9+)

Migration from v0.7 flat files:
    aua config migrate --from 0.7 --to 0.8

Usage:
    from aua.state import get_state_store
    store = get_state_store(config)
    store.append("promotions", {"specialist": "swe", "event": "promote", ...})
    records = store.query("corrections", {"domain": "software_engineering"}, limit=50)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS promotions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    specialist TEXT NOT NULL,
    event TEXT NOT NULL,  -- "promote" | "rollback"
    from_model TEXT,
    to_model TEXT,
    reverted INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    subject TEXT NOT NULL,
    domain TEXT NOT NULL,
    claim TEXT NOT NULL,
    rejected TEXT DEFAULT '',
    confidence REAL NOT NULL,
    source TEXT DEFAULT 'arbiter',
    effective_confidence REAL NOT NULL,
    decay_class TEXT DEFAULT 'C',
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    domain TEXT,
    query_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    event_type TEXT NOT NULL,
    session_id TEXT,
    trace_id TEXT,
    request_id TEXT,
    routing_mode TEXT,
    token_id TEXT,
    field TEXT,
    specialist TEXT,
    u_score REAL,
    confidence REAL,
    latency_ms REAL,
    details TEXT DEFAULT '{}',
    prev_hash TEXT,
    curr_hash TEXT
);

CREATE TABLE IF NOT EXISTS assertion_events (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    session_id TEXT NOT NULL,
    assertion_name TEXT NOT NULL,
    level TEXT NOT NULL,       -- "blocking" | "soft" | "info"
    passed INTEGER NOT NULL,   -- 1 = passed, 0 = failed
    bonus_applied REAL DEFAULT 0.0,
    retries_used INTEGER DEFAULT 0,
    message TEXT,
    domain TEXT,
    policy_name TEXT,
    latency_ms REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_corrections_domain ON corrections(domain);
CREATE INDEX IF NOT EXISTS idx_promotions_specialist ON promotions(specialist);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_assertion_events_session ON assertion_events(session_id);
CREATE INDEX IF NOT EXISTS idx_assertion_events_name ON assertion_events(assertion_name);
CREATE INDEX IF NOT EXISTS idx_assertion_events_created ON assertion_events(created_at);

-- ── P0: Conversation + message persistence (Phase 13 backport) ──────────────

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'local',
    title           TEXT NOT NULL DEFAULT 'New Chat',
    project_id      TEXT DEFAULT NULL,     -- NULL = global (no project)
    created_at      REAL NOT NULL DEFAULT (unixepoch('now')),
    updated_at      REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    message_id      TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,         -- 'user' | 'assistant'
    content         TEXT,                  -- AES-encrypted in production
    callout_type    TEXT DEFAULT NULL,     -- 'disagreement' | 'correction' | NULL
    models_used     TEXT DEFAULT NULL,     -- JSON list of model IDs
    accuracy_level  TEXT DEFAULT NULL,
    confidence      TEXT DEFAULT NULL,
    created_at      REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id              TEXT PRIMARY KEY,
    query_id            TEXT,              -- loose reference, no FK constraint
    conversation_id     TEXT,              -- which conversation this query belongs to
    specialist          TEXT NOT NULL,     -- specialist / model identifier
    round               TEXT NOT NULL DEFAULT 'answer',  -- 'answer' | 'peer_review'
    raw_response        TEXT,
    utility_score       REAL,
    confidence_score    REAL,
    vcg_welfare_score   REAL,
    vcg_winner          INTEGER DEFAULT 0,
    corrections_applied TEXT,              -- JSON list of correction IDs
    latency_ms          REAL,
    domain              TEXT DEFAULT 'general',
    domain_l0           TEXT,              -- top-level domain node
    domain_path         TEXT,              -- full domain path (l0.l1.l2)
    created_at          REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS token_counters (
    counter_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'local',
    specialist      TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    token_estimate  INTEGER DEFAULT 0,     -- chars / 4 (heuristic)
    message_count   INTEGER DEFAULT 0,
    thread_number   INTEGER DEFAULT 1,
    updated_at      REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS message_keywords (
    keyword         TEXT NOT NULL,
    message_id      TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    role            TEXT DEFAULT 'user',
    created_at      REAL NOT NULL DEFAULT (unixepoch('now')),
    PRIMARY KEY (keyword, message_id)
);

CREATE TABLE IF NOT EXISTS context_backups (
    backup_id       TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    specialist      TEXT NOT NULL,
    trigger         TEXT NOT NULL,         -- 'manual' | 'coverage_job' | 'message_count' etc.
    thread_number   INTEGER DEFAULT 1,
    backup_text     TEXT,
    token_estimate  INTEGER DEFAULT 0,
    created_at      REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS projects (
    project_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'local',
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL DEFAULT (unixepoch('now'))
);

-- Indexes for P0 tables
CREATE INDEX IF NOT EXISTS idx_conv_project    ON conversations(project_id);
CREATE INDEX IF NOT EXISTS idx_conv_user       ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_msg_conv        ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msg_created     ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_conv       ON model_runs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_runs_specialist ON model_runs(specialist);
CREATE INDEX IF NOT EXISTS idx_counters_conv   ON token_counters(conversation_id);
CREATE INDEX IF NOT EXISTS idx_kw_conv         ON message_keywords(conversation_id);
CREATE INDEX IF NOT EXISTS idx_backup_conv     ON context_backups(conversation_id);

-- ── v1.1-veritas P1–P3 backport tables ───────────────────────────────────────

-- V-P1.5: crash detection sentinel
CREATE TABLE IF NOT EXISTS crash_sentinel (
    session_id  TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'running',  -- 'running' | 'clean'
    started_at  REAL NOT NULL,
    ended_at    REAL,
    system_log_snippet TEXT,
    api_log_snippet    TEXT
);

-- V-P1.5 / V-P3.1: errors queued from crashed sessions, sent on next launch
CREATE TABLE IF NOT EXISTS pending_error_reports (
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'error',    -- 'error' | 'crash' | 'bug'
    payload     TEXT DEFAULT '{}',
    sent        INTEGER DEFAULT 0
);

-- V-P1.6: remote model config cache (last successful fetch, kept 7 days)
CREATE TABLE IF NOT EXISTS remote_config_cache (
    id          TEXT PRIMARY KEY,                 -- cache key, e.g. 'models'
    created_at  REAL NOT NULL,
    fetched_at  REAL NOT NULL,
    payload     TEXT NOT NULL
);

-- V-P2.3: small app-level key/value store (skipped update versions, etc.)
CREATE TABLE IF NOT EXISTS app_meta (
    id          TEXT PRIMARY KEY,
    created_at  REAL,
    value       TEXT
);

-- V-P2.4: per-correction application/edit history (evidence endpoint)
CREATE TABLE IF NOT EXISTS correction_events (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    correction_id TEXT NOT NULL,
    event         TEXT NOT NULL DEFAULT 'applied', -- created|applied|edited|superseded
    session_id    TEXT,
    details       TEXT DEFAULT '{}'
);

-- V-P3.3: local (Ollama-class) model management
CREATE TABLE IF NOT EXISTS local_models (
    local_model_id    TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL DEFAULT 'local',
    ollama_name       TEXT,
    nickname          TEXT,
    base_url          TEXT,
    runtime           TEXT DEFAULT 'ollama',
    connected         INTEGER DEFAULT 1,
    specialist_domain TEXT,
    specialist_depth  INTEGER DEFAULT 0,
    created_at        REAL NOT NULL DEFAULT (unixepoch('now')),
    updated_at        REAL NOT NULL DEFAULT (unixepoch('now'))
);

-- V-P3.4: dynamic domain ontology (candidate queue + promotion)
CREATE TABLE IF NOT EXISTS domain_nodes (
    node_id       TEXT PRIMARY KEY,
    parent_id     TEXT,
    depth         INTEGER NOT NULL DEFAULT 0,
    display_name  TEXT NOT NULL,
    aliases       TEXT DEFAULT '[]',              -- JSON list of alias strings
    query_count   INTEGER DEFAULT 0,
    is_l0_root    INTEGER DEFAULT 0,
    promoted_from TEXT,
    created_at    REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS domain_candidates (
    raw_string    TEXT PRIMARY KEY,
    nearest_node  TEXT NOT NULL,
    similarity    REAL DEFAULT 0.0,
    query_count   INTEGER DEFAULT 1,
    model_sources TEXT DEFAULT '[]',              -- JSON list of specialist names
    first_seen    REAL NOT NULL,
    last_seen     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_corr_events_corr ON correction_events(correction_id);
CREATE INDEX IF NOT EXISTS idx_sentinel_status  ON crash_sentinel(status);
CREATE INDEX IF NOT EXISTS idx_domain_parent    ON domain_nodes(parent_id);
"""

# Column additions applied to databases created before this version.
# Each runs in a try/except — sqlite raises OperationalError if the column
# already exists, which is the expected steady state.
_MIGRATIONS = [
    "ALTER TABLE corrections ADD COLUMN scope TEXT DEFAULT 'global'",
    # #15: audit events carry request_id; without this column the audit
    # INSERT failed silently (swallowed by the fire-and-forget try/except)
    "ALTER TABLE audit_log ADD COLUMN request_id TEXT",
    "ALTER TABLE audit_log ADD COLUMN routing_mode TEXT",
    "ALTER TABLE token_counters ADD COLUMN last_backup_at REAL",
    # #44: multi-tenancy — tenant_id column on data tables
    "ALTER TABLE corrections ADD COLUMN tenant_id TEXT DEFAULT NULL",
    "ALTER TABLE promotions ADD COLUMN tenant_id TEXT DEFAULT NULL",
    "ALTER TABLE audit_log ADD COLUMN tenant_id TEXT DEFAULT NULL",
    "ALTER TABLE model_runs ADD COLUMN tenant_id TEXT DEFAULT NULL",
]


# ── SQLite state store ────────────────────────────────────────────────────────


class SQLiteStateStore:
    """
    SQLite-backed state store. Default for v0.8+.

    Thread-safe via WAL mode. Supports concurrent readers + one writer.
    """

    def __init__(self, db_path: str | os.PathLike = ".aua/state/aua.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            for migration in _MIGRATIONS:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass  # column already exists — expected steady state

    # ── App meta key/value (V-P2.3) ───────────────────────────────────────────

    def meta_get(self, key: str) -> str | None:
        """Read a value from the app_meta key/value table."""
        row = self.get("app_meta", key)
        return row["value"] if row else None

    def meta_set(self, key: str, value: str) -> None:
        """Write a value to the app_meta key/value table (upsert)."""
        self.set("app_meta", key, {"created_at": time.time(), "value": value})

    def get(self, table: str, key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (key,)).fetchone()
            return dict(row) if row else None

    def set(self, table: str, key: str, value: dict[str, Any]) -> None:
        value = {**value, "id": key}
        cols = ", ".join(value.keys())
        placeholders = ", ".join("?" for _ in value)
        updates = ", ".join(f"{k}=excluded.{k}" for k in value if k != "id")
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                list(value.values()),
            )

    # Tables that carry tenant_id for multi-tenancy (#44)
    _TENANT_SCOPED_TABLES = frozenset({"corrections", "promotions", "audit_log", "model_runs"})

    def append(self, table: str, record: dict[str, Any]) -> str:
        # #44: auto-inject tenant_id from contextvar for scoped tables
        if table in self._TENANT_SCOPED_TABLES and "tenant_id" not in record:
            try:
                from aua.tenancy import get_tenant_id

                tid = get_tenant_id()
                if tid:
                    record = {**record, "tenant_id": tid}
            except Exception:
                pass  # tenancy module unavailable — skip silently

        record_id = record.get("id") or str(uuid.uuid4())
        record = {
            **record,
            "id": record_id,
            "created_at": record.get("created_at") or time.time(),
        }

        # Serialize nested dicts to JSON strings
        serialized = {}
        for k, v in record.items():
            if isinstance(v, (dict, list)):
                serialized[k] = json.dumps(v)
            else:
                serialized[k] = v

        cols = ", ".join(serialized.keys())
        placeholders = ", ".join("?" for _ in serialized)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})",
                list(serialized.values()),
            )
        return record_id

    def query(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        order_by: str = "created_at DESC",
    ) -> list[dict[str, Any]]:
        filters = filters or {}
        where_clauses = [f"{k} = ?" for k in filters]
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"SELECT * FROM {table} {where} ORDER BY {order_by} LIMIT ?"
        with self._connect() as conn:
            rows = conn.execute(sql, list(filters.values()) + [limit]).fetchall()
        return [dict(row) for row in rows]

    def append_audit(self, event: dict[str, Any]) -> str:
        """Append to audit log with hash chain for tamper detection."""
        # #44: inject tenant_id from context when present
        from aua.tenancy import get_tenant_id

        tenant_id = get_tenant_id()
        if tenant_id and "tenant_id" not in event:
            event = {**event, "tenant_id": tenant_id}

        # Get last hash
        rows = self.query("audit_log", limit=1, order_by="created_at DESC")
        prev_hash = rows[0]["curr_hash"] if rows else "genesis"

        event_json = json.dumps(event, sort_keys=True, default=str)
        curr_hash = hashlib.sha256(f"{prev_hash}:{event_json}".encode()).hexdigest()

        return self.append(
            "audit_log",
            {**event, "prev_hash": prev_hash, "curr_hash": curr_hash},
        )

    def migrate_from_files(self, project_dir: str = ".") -> dict[str, int]:
        """
        Migrate v0.7 flat files into SQLite.

        Reads:
          .aua/state/promotions.jsonl → promotions table
          dpo_pairs/*.jsonl           → corrections table

        Returns count of migrated records per table.
        """
        counts: dict[str, int] = {"promotions": 0, "corrections": 0}
        base = Path(project_dir)

        # Migrate promotions.jsonl
        promotions_file = base / ".aua" / "state" / "promotions.jsonl"
        if promotions_file.exists():
            for line in promotions_file.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        self.append("promotions", record)
                        counts["promotions"] += 1
                    except (json.JSONDecodeError, Exception):
                        pass

        # Migrate dpo_pairs/*.jsonl
        dpo_dir = base / "dpo_pairs"
        if dpo_dir.exists():
            for jsonl_file in sorted(dpo_dir.glob("*.jsonl")):
                for line in jsonl_file.read_text().splitlines():
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            # Map DPO pair fields to corrections schema
                            self.append(
                                "corrections",
                                {
                                    "subject": record.get("prompt", ""),
                                    "domain": record.get("domain", "general"),
                                    "claim": record.get("chosen", ""),
                                    "rejected": record.get("rejected", ""),
                                    "confidence": record.get("confidence", 0.7),
                                    "effective_confidence": record.get("confidence", 0.7),
                                    "source": record.get("source", "arbiter"),
                                },
                            )
                            counts["corrections"] += 1
                        except (json.JSONDecodeError, Exception):
                            pass

        return counts

    # ── P0: Conversation + message persistence ────────────────────────────────

    def create_conversation(
        self, title: str = "New Chat", project_id: str | None = None, user_id: str = "local"
    ) -> dict[str, Any]:
        """Create a new conversation, returning the full record."""
        conv_id = str(uuid.uuid4())
        now = time.time()
        record = {
            "conversation_id": conv_id,
            "user_id": user_id,
            "title": title,
            "project_id": project_id,
            "created_at": now,
            "updated_at": now,
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (conversation_id, user_id, title, project_id, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, user_id, title, project_id, now, now),
            )
        return record

    def list_conversations(
        self,
        user_id: str = "local",
        project_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """List conversations, optionally filtered by project."""
        with self._connect() as conn:
            if project_id is not None:
                rows = conn.execute(
                    "SELECT * FROM conversations WHERE user_id=? AND project_id=?"
                    " ORDER BY updated_at DESC LIMIT ?",
                    (user_id, project_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM conversations WHERE user_id=?"
                    " ORDER BY updated_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        """Update the title of a conversation."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE conversation_id=?",
                (title, time.time(), conversation_id),
            )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        callout_type: str | None = None,
        models_used: list[str] | None = None,
        accuracy_level: str | None = None,
        confidence: str | None = None,
    ) -> str:
        """Append a message to a conversation. Returns message_id."""
        msg_id = str(uuid.uuid4())
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (message_id, conversation_id, role, content,"
                " callout_type, models_used, accuracy_level, confidence, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    msg_id,
                    conversation_id,
                    role,
                    content,
                    callout_type,
                    json.dumps(models_used) if models_used else None,
                    accuracy_level,
                    confidence,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                (now, conversation_id),
            )
        return msg_id

    def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
        before: float | None = None,
        after: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Paginated message fetch.
        before: load older messages (created_at < before), newest-first.
        after:  load newer messages (created_at > after), oldest-first.
        No cursor: most recent `limit` messages, oldest-first.
        """
        with self._connect() as conn:
            if after is not None:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id=? AND created_at>?"
                    " ORDER BY created_at ASC LIMIT ?",
                    (conversation_id, after, limit),
                ).fetchall()
                return [dict(r) for r in rows]
            elif before is not None:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id=? AND created_at<?"
                    " ORDER BY created_at DESC LIMIT ?",
                    (conversation_id, before, limit),
                ).fetchall()
                return [dict(r) for r in reversed(rows)]
            else:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id=?"
                    " ORDER BY created_at DESC LIMIT ?",
                    (conversation_id, limit),
                ).fetchall()
                return [dict(r) for r in reversed(rows)]

    def record_model_run(self, run: dict[str, Any]) -> str:
        """
        Store a model run record.

        Implementation rule (Phase 13): every model_run must carry conversation_id
        as an explicit field — do not rely on closure capture. Without it there is no
        join path between a conversation and its model runs.

        run must include: specialist, conversation_id, round.
        Optional: query_id, utility_score, confidence_score, domain, domain_l0,
                  domain_path, vcg_welfare_score, vcg_winner, corrections_applied,
                  latency_ms, raw_response.
        """
        # #44: inject tenant_id from context when present
        from aua.tenancy import get_tenant_id

        tenant_id = get_tenant_id()
        if tenant_id and "tenant_id" not in run:
            run = {**run, "tenant_id": tenant_id}

        run_id = run.get("run_id") or str(uuid.uuid4())
        record = {**run, "run_id": run_id, "created_at": run.get("created_at") or time.time()}
        serialized = {
            k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in record.items()
        }
        cols = ", ".join(serialized.keys())
        placeholders = ", ".join("?" for _ in serialized)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO model_runs ({cols}) VALUES ({placeholders})",
                list(serialized.values()),
            )
        return run_id

    def update_token_counter(
        self,
        specialist: str,
        conversation_id: str,
        tokens_added: int,
        user_id: str = "local",
    ) -> None:
        """Increment token counter for a specialist in a conversation."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT counter_id, token_estimate, message_count FROM token_counters"
                " WHERE specialist=? AND conversation_id=?",
                (specialist, conversation_id),
            ).fetchone()
            now = time.time()
            if existing:
                conn.execute(
                    "UPDATE token_counters SET token_estimate=?, message_count=?, updated_at=?"
                    " WHERE counter_id=?",
                    (
                        existing["token_estimate"] + tokens_added,
                        existing["message_count"] + 1,
                        now,
                        existing["counter_id"],
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO token_counters"
                    " (counter_id, user_id, specialist, conversation_id,"
                    "  token_estimate, message_count, thread_number, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, 1, 1, ?)",
                    (str(uuid.uuid4()), user_id, specialist, conversation_id, tokens_added, now),
                )

    def store_context_backup(
        self,
        conversation_id: str,
        specialist: str,
        backup_text: str,
        trigger: str,
        thread_number: int = 1,
    ) -> str:
        """Store a context backup. Returns backup_id."""
        backup_id = str(uuid.uuid4())
        token_estimate = max(1, len(backup_text) // 4)
        record = {
            "backup_id": backup_id,
            "conversation_id": conversation_id,
            "specialist": specialist,
            "trigger": trigger,
            "thread_number": thread_number,
            "backup_text": backup_text,
            "token_estimate": token_estimate,
            "created_at": time.time(),
        }
        cols = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR IGNORE INTO context_backups ({cols}) VALUES ({placeholders})",
                list(record.values()),
            )
        return backup_id

    def get_latest_backup(self, conversation_id: str, specialist: str) -> str | None:
        """Return the most recent backup text for a conversation+specialist pair."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT backup_text FROM context_backups"
                " WHERE conversation_id=? AND specialist=?"
                " ORDER BY created_at DESC LIMIT 1",
                (conversation_id, specialist),
            ).fetchone()
        return row["backup_text"] if row else None

    def backup_is_valid(self, conversation_id: str, specialist: str) -> bool:
        """
        A backup is VALID when:
            MAX(context_backups.created_at WHERE specialist=X, conversation_id=Y)
                > MAX(messages.created_at WHERE conversation_id=Y)

        i.e. the most recent backup is NEWER than the most recent message.
        """
        with self._connect() as conn:
            backup_ts = conn.execute(
                "SELECT MAX(created_at) ts FROM context_backups"
                " WHERE conversation_id=? AND specialist=?",
                (conversation_id, specialist),
            ).fetchone()
            last_msg_ts = conn.execute(
                "SELECT MAX(created_at) ts FROM messages WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
        bt = backup_ts["ts"] if backup_ts else None
        lt = last_msg_ts["ts"] if last_msg_ts else None
        if bt is None or lt is None:
            return False
        return bt > lt

    def stale_backup_conversations(
        self, specialist: str, min_messages: int = 5
    ) -> list[dict[str, Any]]:
        """
        Return conversations that need a backup for the given specialist.
        Used by the 6-hour coverage job (Phase 13).

        A conversation needs a backup when:
          - It has >= min_messages messages AND
          - No backup exists for this specialist OR backup is older than last message
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                WITH conv_stats AS (
                    SELECT c.conversation_id, COUNT(m.message_id) AS msg_count,
                           MAX(m.created_at) AS last_msg_ts
                    FROM conversations c
                    LEFT JOIN messages m ON m.conversation_id = c.conversation_id
                    GROUP BY c.conversation_id
                    HAVING msg_count >= ?
                ),
                latest_backup AS (
                    SELECT conversation_id, MAX(created_at) AS backup_ts
                    FROM context_backups WHERE specialist=?
                    GROUP BY conversation_id
                )
                SELECT cs.conversation_id, cs.msg_count, cs.last_msg_ts, lb.backup_ts
                FROM conv_stats cs
                LEFT JOIN latest_backup lb ON lb.conversation_id = cs.conversation_id
                WHERE lb.backup_ts IS NULL OR lb.backup_ts < cs.last_msg_ts
                ORDER BY cs.last_msg_ts DESC
                """,
                (min_messages, specialist),
            ).fetchall()
        return [dict(r) for r in rows]


# ── P0: LRU message cache ─────────────────────────────────────────────────────


class MessageCache:
    """
    LRU in-memory cache for conversation message lists.

    Design rules (Phase 13):
      - Use collections.OrderedDict + move_to_end() on every hit (true LRU).
      - FIFO dict evicts the most-accessed conversation — wrong behavior.
      - Bypass cache when limit < default (50): custom limits must hit DB.
        Otherwise GET /messages?limit=1 silently returns all cached messages.

    Capacity: 500 conversations (~21 MB at typical message sizes).
    """

    DEFAULT_LIMIT = 50
    MAX_CONVS = 500

    def __init__(self) -> None:
        from collections import OrderedDict

        self._cache: OrderedDict[str, list[dict]] = OrderedDict()

    def get(self, conversation_id: str, limit: int = DEFAULT_LIMIT) -> list[dict] | None:
        """Return cached messages or None. Bypasses cache for non-default limits."""
        if limit < self.DEFAULT_LIMIT:
            return None  # custom limit must hit DB — see bypass rule above
        if conversation_id in self._cache:
            self._cache.move_to_end(conversation_id)  # LRU promotion
            return self._cache[conversation_id]
        return None

    def set(self, conversation_id: str, messages: list[dict]) -> None:
        """Cache messages for a conversation. Evicts LRU if at capacity."""
        if conversation_id in self._cache:
            self._cache.move_to_end(conversation_id)
        self._cache[conversation_id] = messages
        if len(self._cache) > self.MAX_CONVS:
            self._cache.popitem(last=False)  # evict LRU (oldest)

    def invalidate(self, conversation_id: str, reason: str = "new_message") -> None:
        """Invalidate cache for a conversation (call after any write)."""
        self._cache.pop(conversation_id, None)

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


# ── P0: Off-critical-path write helper ───────────────────────────────────────


def fire_and_forget(coro: Coroutine[object, object, object]) -> None:
    """
    Schedule a coroutine as a background task without blocking the caller.

    Implementation rule (Phase 13):
      - asyncio MUST be imported at module level or locally — never rely on
        a lifespan-scope alias being available in endpoint functions.
      - Call this from within a running event loop (inside async request handlers).

    Usage:
        fire_and_forget(_store_model_run(store, run_record))
        fire_and_forget(_update_token_counter(store, specialist, conv_id, tokens))
    """
    import asyncio as _asyncio_ff

    try:
        loop = _asyncio_ff.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # No running loop — run synchronously as fallback (e.g. in tests)
        import asyncio

        asyncio.run(coro)


# ── Files state store (v0.7 compatibility) ────────────────────────────────────
# ── Files state store (v0.7 compatibility) ────────────────────────────────────


class FilesStateStore:
    """
    Flat-file state store. Maintains v0.7 behavior.
    Use this if you can't run SQLite or want the old JSONL files.
    """

    def __init__(self, base_dir: str = ".aua/state") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _table_path(self, table: str) -> Path:
        return self._base / f"{table}.jsonl"

    def get(self, table: str, key: str) -> dict[str, Any] | None:
        for record in self._read_all(table):
            if record.get("id") == key:
                return record
        return None

    def set(self, table: str, key: str, value: dict[str, Any]) -> None:
        records = [r for r in self._read_all(table) if r.get("id") != key]
        records.append({**value, "id": key})
        self._write_all(table, records)

    def append(self, table: str, record: dict[str, Any]) -> str:
        record_id = record.get("id") or str(uuid.uuid4())
        record = {**record, "id": record_id, "created_at": record.get("created_at") or time.time()}
        path = self._table_path(table)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record_id

    def query(
        self,
        table: str,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        order_by: str = "created_at DESC",
    ) -> list[dict[str, Any]]:
        records = self._read_all(table)
        if filters:
            records = [r for r in records if all(r.get(k) == v for k, v in filters.items())]
        reverse = "DESC" in order_by
        key_field = order_by.split()[0]
        records.sort(key=lambda r: r.get(key_field, 0), reverse=reverse)
        return records[:limit]

    def _read_all(self, table: str) -> list[dict[str, Any]]:
        path = self._table_path(table)
        if not path.exists():
            return []
        results = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return results

    def _write_all(self, table: str, records: list[dict[str, Any]]) -> None:
        path = self._table_path(table)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        os.replace(str(tmp), str(path))

    def append_audit(self, event: dict[str, Any]) -> str:
        return self.append("audit_log", event)

    def migrate_from_files(self, project_dir: str = ".") -> dict[str, int]:
        return {}  # Files store IS the files — nothing to migrate


# ── Factory ───────────────────────────────────────────────────────────────────


def get_state_store(config: Any | None = None) -> SQLiteStateStore | FilesStateStore:
    """
    Return the configured state store instance.

    Config example:
        state:
          backend: sqlite
          path: .aua/state/aua.db

    Defaults to SQLite at .aua/state/aua.db if no config provided.
    """
    if config is None:
        return SQLiteStateStore()

    state_cfg = getattr(config, "state", None)
    if state_cfg is None:
        return SQLiteStateStore()

    backend = getattr(state_cfg, "backend", "sqlite")

    if backend == "files":
        base_dir = getattr(state_cfg, "path", ".aua/state")
        return FilesStateStore(base_dir)

    # Default: sqlite
    db_path = getattr(state_cfg, "path", ".aua/state/aua.db")
    return SQLiteStateStore(db_path)
