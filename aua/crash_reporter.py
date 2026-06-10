"""
aua/crash_reporter.py — Crash detection + auto-report on next launch (V-P1.5).

Sentinel mechanism:
  On startup:       write a sentinel row with status='running'
  On clean shutdown: update the sentinel to status='clean'
  On next startup:  any row still status='running' → previous session crashed
                    → queue a crash report and send it asynchronously

Runtime errors can also be queued in the ``pending_error_reports`` table; if
the process dies before sending, they are flushed on the next launch.

Reports go through :mod:`aua.bug_reporter` (GitHub Contents API) and degrade
gracefully when no PAT is configured.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Any

from aua import bug_reporter

if TYPE_CHECKING:
    from aua.state import SQLiteStateStore

log = logging.getLogger("aua.crash_reporter")


# ── Sentinel lifecycle ────────────────────────────────────────────────────────


def record_startup(state: SQLiteStateStore) -> str:
    """Write a 'running' sentinel row at startup. Returns the session_id."""
    session_id = str(uuid.uuid4())
    try:
        with state._connect() as conn:
            conn.execute(
                "INSERT INTO crash_sentinel"
                " (session_id, status, started_at, ended_at,"
                "  system_log_snippet, api_log_snippet)"
                " VALUES (?, 'running', ?, NULL, NULL, NULL)",
                (session_id, time.time()),
            )
        log.debug("Crash sentinel written: session=%s", session_id)
    except Exception as e:  # noqa: BLE001
        log.warning("record_startup failed: %s", e)
    return session_id


def record_shutdown(state: SQLiteStateStore, session_id: str) -> None:
    """Mark the current session as a clean shutdown."""
    try:
        with state._connect() as conn:
            conn.execute(
                "UPDATE crash_sentinel SET status='clean', ended_at=? WHERE session_id=?",
                (time.time(), session_id),
            )
        log.debug("Clean shutdown recorded: session=%s", session_id)
    except Exception as e:  # noqa: BLE001
        log.warning("record_shutdown failed: %s", e)


def detect_crash(state: SQLiteStateStore) -> dict[str, Any] | None:
    """
    Check if a previous session crashed (sentinel still status='running').
    Returns crash info dict or None when the last session ended cleanly.
    """
    try:
        with state._connect() as conn:
            row = conn.execute(
                "SELECT session_id, started_at, ended_at,"
                "       system_log_snippet, api_log_snippet"
                " FROM crash_sentinel WHERE status='running'"
                " ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        if row:
            return {
                "session_id": row[0],
                "started_at": row[1],
                "ended_at": row[2],
                "system_log_snippet": row[3] or "",
                "api_log_snippet": row[4] or "",
            }
    except Exception as e:  # noqa: BLE001
        log.warning("detect_crash failed: %s", e)
    return None


def mark_crash_reported(state: SQLiteStateStore, session_id: str) -> None:
    """Close out a crashed sentinel so it isn't reported twice."""
    try:
        with state._connect() as conn:
            conn.execute(
                "UPDATE crash_sentinel SET status='clean', ended_at=? WHERE session_id=?",
                (time.time(), session_id),
            )
    except Exception as e:  # noqa: BLE001
        log.warning("mark_crash_reported failed: %s", e)


# ── Pending error queue ───────────────────────────────────────────────────────


def queue_error_report(
    state: SQLiteStateStore,
    error: BaseException | str,
    kind: str = "error",
    context: dict[str, Any] | None = None,
) -> str:
    """
    Save a runtime error to ``pending_error_reports``. If the process dies
    before sending, the report is flushed on the next startup.
    """
    report_id = str(uuid.uuid4())
    if isinstance(error, BaseException):
        payload: dict[str, Any] = {
            "error": repr(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-4000:],
        }
    else:
        payload = {"error": str(error)}
    if context:
        payload["context"] = context
    try:
        state.append(
            "pending_error_reports",
            {"id": report_id, "kind": kind, "payload": json.dumps(payload), "sent": 0},
        )
    except Exception as e:  # noqa: BLE001
        log.warning("queue_error_report failed: %s", e)
    return report_id


def get_pending_error_reports(state: SQLiteStateStore, limit: int = 10) -> list[dict[str, Any]]:
    try:
        return state.query(
            "pending_error_reports",
            filters={"sent": 0},
            limit=limit,
            order_by="created_at ASC",
        )
    except Exception:  # noqa: BLE001
        return []


def mark_error_sent(state: SQLiteStateStore, report_id: str) -> None:
    try:
        with state._connect() as conn:
            conn.execute("UPDATE pending_error_reports SET sent=1 WHERE id=?", (report_id,))
    except Exception as e:  # noqa: BLE001
        log.warning("mark_error_sent failed: %s", e)


# ── Async reporting on next launch ───────────────────────────────────────────


async def report_previous_crash(
    state: SQLiteStateStore, crash: dict[str, Any] | None = None
) -> bool:
    """
    Report a crash from the previous session asynchronously.

    Pass `crash` when detection already ran (the startup sequence must detect
    BEFORE writing the new 'running' sentinel, or the current session
    self-reports as crashed). Falls back to detect_crash() when omitted.
    Returns True when a crash was detected (whether or not the send worked).
    """
    if crash is None:
        crash = detect_crash(state)
    if not crash:
        return False
    pat = bug_reporter.get_bugs_pat(state)
    report = bug_reporter.build_report(
        user_token=bug_reporter.generate_user_token(),
        comment="Automatic crash report — previous session did not shut down cleanly.",
        kind="crash",
        system_log_tail=crash["system_log_snippet"],
        api_log_tail=crash["api_log_snippet"],
        extra={"crashed_session_id": crash["session_id"], "started_at": crash["started_at"]},
    )
    ok, msg = await bug_reporter.submit_report(report, pat or "")
    if not ok:
        log.info("Crash report not sent (%s) — marking sentinel anyway", msg)
    mark_crash_reported(state, crash["session_id"])
    return True


async def flush_pending_errors(state: SQLiteStateStore) -> int:
    """Send queued error reports from previous sessions. Returns count sent."""
    pat = bug_reporter.get_bugs_pat(state)
    if not pat:
        return 0
    sent = 0
    for pending in get_pending_error_reports(state):
        try:
            payload = json.loads(pending.get("payload") or "{}")
        except Exception:  # noqa: BLE001
            payload = {"error": pending.get("payload", "")}
        report = bug_reporter.build_report(
            user_token=bug_reporter.generate_user_token(),
            comment="Queued error report from a previous session.",
            kind=pending.get("kind", "error"),
            extra=payload,
        )
        ok, _msg = await bug_reporter.submit_report(report, pat)
        if ok:
            mark_error_sent(state, pending["id"])
            sent += 1
    return sent
