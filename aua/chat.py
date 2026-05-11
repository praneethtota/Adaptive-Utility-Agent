"""
aua/chat.py — Chat Session API for AUA Framework.

Provides persistent multi-turn conversations backed by the state store.
Each session tracks messages, routing traces, and metadata.

Endpoints (added to router by _build_app):
    POST   /sessions                    Create a new session
    GET    /sessions                    List sessions
    GET    /sessions/{id}               Get session detail
    DELETE /sessions/{id}               Delete a session
    GET    /sessions/{id}/messages      Get messages in a session
    POST   /sessions/{id}/messages      Send a message (buffered)
    POST   /sessions/{id}/stream        Send a message (SSE streaming)

SSE streaming events:
    {"type": "start",            "session_id": "...", "message_id": "..."}
    {"type": "route",            "domain": "...", "routing_mode": "..."}
    {"type": "specialist_start", "specialist": "..."}
    {"type": "chunk",            "text": "..."}
    {"type": "specialist_done",  "specialist": "..."}
    {"type": "done",             "response": "...", "u_score": 0.5, ...}
    {"type": "error",            "message": "..."}

State store tables used:
    chat_sessions   id, created_at, updated_at, title, message_count, metadata
    chat_messages   id, session_id, role, content, created_at, domain,
                    routing_mode, u_score, latency_ms, metadata
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from aua.state import get_state_store

# ── Session helpers ───────────────────────────────────────────────────────────


def create_session(title: str = "", metadata: dict | None = None) -> dict[str, Any]:
    """Create a new chat session. Returns the session dict."""
    store = get_state_store()
    now = time.time()
    session_id = str(uuid.uuid4())
    session: dict[str, Any] = {
        "id": session_id,
        "created_at": now,
        "updated_at": now,
        "title": title or "New Chat",
        "message_count": 0,
        "metadata": json.dumps(metadata or {}),
    }
    store.set("chat_sessions", session_id, session)
    return session


def get_session(session_id: str) -> dict[str, Any] | None:
    """Get a session by ID."""
    return get_state_store().get("chat_sessions", session_id)


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """List all sessions, most recent first."""
    return get_state_store().query("chat_sessions", limit=limit, order_by="updated_at DESC")


def delete_session(session_id: str) -> bool:
    """Delete a session and its messages."""
    store = get_state_store()
    session = store.get("chat_sessions", session_id)
    if not session:
        return False
    # Mark as deleted (soft delete — we don't support hard deletes in append-only tables)
    session["deleted"] = True
    session["updated_at"] = time.time()
    store.set("chat_sessions", session_id, session)
    return True


def get_messages(session_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Get messages for a session, oldest first."""
    return get_state_store().query(
        "chat_messages",
        filters={"session_id": session_id},
        limit=limit,
        order_by="created_at ASC",
    )


def add_message(
    session_id: str,
    role: str,
    content: str,
    domain: str = "",
    routing_mode: str = "",
    u_score: float = 0.0,
    latency_ms: float = 0.0,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Append a message to a session."""
    store = get_state_store()
    msg = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": time.time(),
        "domain": domain,
        "routing_mode": routing_mode,
        "u_score": u_score,
        "latency_ms": latency_ms,
        "metadata": json.dumps(metadata or {}),
    }
    store.append("chat_messages", msg)

    # Update session message count and updated_at
    session = store.get("chat_sessions", session_id)
    if session:
        session["message_count"] = int(session.get("message_count", 0)) + 1
        session["updated_at"] = time.time()
        # Auto-title from first user message
        if session.get("title") == "New Chat" and role == "user":
            session["title"] = content[:60] + ("..." if len(content) > 60 else "")
        store.set("chat_sessions", session_id, session)

    return msg


def ensure_chat_tables() -> None:
    """Ensure chat tables exist in the SQLite state store."""
    from aua.state import SQLiteStateStore, get_state_store

    store = get_state_store()
    if not isinstance(store, SQLiteStateStore):
        return  # Files store — tables not needed

    chat_schema = """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id TEXT PRIMARY KEY,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        title TEXT DEFAULT 'New Chat',
        message_count INTEGER DEFAULT 0,
        deleted INTEGER DEFAULT 0,
        metadata TEXT DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at REAL NOT NULL,
        domain TEXT DEFAULT '',
        routing_mode TEXT DEFAULT '',
        u_score REAL DEFAULT 0.0,
        latency_ms REAL DEFAULT 0.0,
        metadata TEXT DEFAULT '{}'
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_updated ON chat_sessions(updated_at DESC);
    """
    with store._connect() as conn:
        conn.executescript(chat_schema)
