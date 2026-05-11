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

CREATE INDEX IF NOT EXISTS idx_corrections_domain ON corrections(domain);
CREATE INDEX IF NOT EXISTS idx_promotions_specialist ON promotions(specialist);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
"""


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

    def append(self, table: str, record: dict[str, Any]) -> str:
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
