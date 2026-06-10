"""
aua/bug_reporter.py — Structured bug reporting via GitHub Contents API (V-P3.1).

Assembles a structured JSON report and pushes it to a dedicated bug-reports
repository via the GitHub Contents API. The PAT should be write-only and
scoped exclusively to that repository — worst-case misuse is spam reports.

Graceful degradation rule (Phase 13): when the PAT is not configured the
endpoint returns 200 with an error message in the body — never a 500. Bug
reporting must not itself be a source of bugs.

Configuration (all optional):
    AUA_BUGS_REPO  — "owner/repo" for the reports repository
    AUA_BUGS_PAT   — write-only PAT for that repository
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import platform
import sys
import time
import uuid
from typing import Any

from aua.version import __version__

log = logging.getLogger("aua.bug_reporter")

_DEFAULT_REPO = "praneethtota/AUA-Bug-Reports"
_pat_cache: str | None = None


def get_bugs_repo() -> str:
    return os.environ.get("AUA_BUGS_REPO", _DEFAULT_REPO).strip() or _DEFAULT_REPO


def get_bugs_pat(state: Any | None = None) -> str | None:
    """
    Read the bug-report PAT. Priority: in-memory cache → AUA_BUGS_PAT env var
    → app_meta['bugs_pat'] in the state store.
    """
    global _pat_cache
    if _pat_cache:
        return _pat_cache
    pat = os.environ.get("AUA_BUGS_PAT", "").strip()
    if pat:
        _pat_cache = pat
        return _pat_cache
    if state is not None:
        try:
            stored = state.meta_get("bugs_pat")
            if stored:
                _pat_cache = stored
                return _pat_cache
        except Exception:  # noqa: BLE001
            pass
    return None


def generate_user_token() -> str:
    """
    SHA-256 of the machine's MAC address, truncated to 8 hex chars.
    Stable across sessions, not personally identifiable.
    """
    return hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()[:8]


def build_report(
    user_token: str,
    comment: str,
    kind: str = "bug",
    system_log_tail: str = "",
    api_log_tail: str = "",
    console_errors: list[dict[str, Any]] | None = None,
    include_messages: bool = False,
    last_messages: list[dict[str, Any]] | None = None,
    user_email: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assemble the report dict.

    Always included (no user content): log tails and console errors.
    Opt-in: the last conversation exchange and a contact email.
    """
    epoch = int(time.time())
    report: dict[str, Any] = {
        "report_id": f"usr_{user_token}_{epoch}",
        "kind": kind,
        "created_at": epoch,
        "framework_version": __version__,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "python": platform.python_version(),
            "frozen": bool(getattr(sys, "frozen", False)),
        },
        "comment": comment,
        "system_log_tail": system_log_tail,
        "api_log_tail": api_log_tail,
        "console_errors": console_errors or [],
    }
    if include_messages and last_messages:
        report["last_messages"] = last_messages
    if user_email:
        report["user_email"] = user_email
    if extra:
        report["extra"] = extra
    return report


async def submit_report(report: dict[str, Any], pat: str) -> tuple[bool, str]:
    """
    Push report JSON to the bug-reports repo via GitHub Contents API.
    Returns (success, message). Never raises.
    """
    if not pat:
        return False, "Bug report PAT not configured (AUA_BUGS_PAT)."
    try:
        import httpx
    except ImportError:
        return False, "httpx not installed — bug reporting unavailable."

    report_id = report["report_id"]
    url = f"https://api.github.com/repos/{get_bugs_repo()}/contents/reports/{report_id}.json"
    payload = {
        "message": f"Bug report {report_id}",
        "content": base64.b64encode(json.dumps(report, indent=2).encode()).decode(),
    }
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.put(url, json=payload, headers=headers)
        if resp.status_code in (200, 201):
            log.info("Bug report submitted: %s", report_id)
            return True, report_id
        log.error("Bug report failed: %d %s", resp.status_code, resp.text[:200])
        return False, f"GitHub API returned {resp.status_code}"
    except Exception as e:  # noqa: BLE001
        log.error("Bug report exception: %s", e)
        return False, str(e)
