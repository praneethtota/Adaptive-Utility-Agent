"""
aua/context_backup.py — Per-specialist context continuity (v1.1-veritas backport).

V-P1.4 — Structured context backup prompt: a 6-section handoff template that
forces the model to capture GOAL / DECISIONS / STATUS / ACTIVE FILE /
PREFERENCES / RESUME INSTRUCTION. Max backup tokens raised 600 → 900 so all
six sections fit without truncation.

V-P1.2 — Coverage job: a 6-hour background sweep that finds all conversations
whose latest backup is stale or missing and regenerates them at 1/s pacing.

    backup valid ⇔ MAX(context_backups.created_at) > MAX(messages.created_at)
                    for that (conversation, specialist) pair

Implementation rules carried forward from AUA-Veritas Phase 13:

  * Context backups read the FULL conversation from the DB (last 60 messages),
    never the request-body slice the client happened to have loaded (V-P0.5).
  * Method signature hygiene: when refactoring signatures, grep all call
    sites — a stale kwarg crashed every coverage job run silently in Veritas.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aua.state import SQLiteStateStore

log = logging.getLogger("aua.context_backup")

# Fraction of the specialist's context window at which a backup triggers
TOKEN_THRESHOLD_PCT = 0.70

# Number of messages between rolling backups (regardless of token count)
MESSAGE_BACKUP_INTERVAL = 30

# Characters-per-token estimate (conservative)
CHARS_PER_TOKEN = 4

# V-P1.4: max tokens for the generated backup — raised 600 → 900 to fit
# the structured 6-section template without truncation.
MAX_BACKUP_TOKENS = 900

# How many DB messages feed the backup generator (full-history rule, V-P0.5)
BACKUP_HISTORY_MESSAGES = 60

# V-P1.4: structured 6-section handoff template
BACKUP_PROMPT = """
You are writing a context handoff note so that a fresh AI window can resume \
this conversation immediately without the user re-explaining anything.

Structure your response EXACTLY as follows (keep each section, skip if truly \
not applicable):

## GOAL
One sentence: what is the user ultimately trying to accomplish and why?
Include any deadline, constraint, or success criterion mentioned.

## DECISIONS MADE
List every significant decision with the reason it was made.
Format: "Decided [X] because [Y]. Rejected [Z] because [W]."
Include technology choices, architectural decisions, approach trade-offs.

## CURRENT STATUS
- Completed: list what is fully done
- In progress: what is being worked on RIGHT NOW (be specific — exact \
function, file, or step)
- Unresolved: open questions or blockers not yet decided

## ACTIVE FILE / CODE CONTEXT
If working on code: exact file path(s), function name, what it does and what \
needs to happen next.
If writing/analysis: exact document section or data being worked on.

## USER PREFERENCES LEARNED
Preferences and corrections from this conversation not yet formally stored
(e.g. naming conventions, tone, format, style choices specific to this work).

## RESUME INSTRUCTION
Single sentence telling the new window exactly how to pick up
(e.g. "Ask the user to confirm the auth middleware location before writing \
any code.").

Rules:
- Max 900 tokens total
- No pleasantries, no meta-commentary about what you are doing
- Only include information that would be LOST if the conversation ended now
- Write in second person ("The user is building..." not "I was helping...")
"""

# Coverage job tuning (V-P1.2)
COVERAGE_INTERVAL_S = 6 * 3600  # 6 hours between sweeps
COVERAGE_FIRST_RUN_DELAY_S = 60  # first run: 60s after startup so models load
COVERAGE_MIN_MESSAGES = 5  # skip trivial conversations
COVERAGE_PACE_S = 1.0  # seconds between backup calls (rate-limit pacing)

# Type of the callable that actually asks a model to produce the backup text.
# Signature: (specialist, conversation_id, prompt, history) -> backup_text
BackupGenerator = Callable[[str, str, str, list[dict[str, Any]]], Awaitable[str]]


class ContextBackupManager:
    """Per-specialist token counting, backup triggering, and coverage sweep."""

    def __init__(self, state: SQLiteStateStore) -> None:
        self._state = state

    # ── Token counting ────────────────────────────────────────────────────────

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)

    def update_counter(
        self,
        specialist: str,
        conversation_id: str,
        new_text: str,
        user_id: str = "local",
    ) -> dict[str, Any]:
        """Add new_text's estimated tokens to the (specialist, conversation) counter."""
        tokens = self.estimate_tokens(new_text)
        try:
            self._state.update_token_counter(
                specialist=specialist,
                conversation_id=conversation_id,
                tokens_added=tokens,
                user_id=user_id,
            )
            rows = self._state.query(
                "token_counters",
                filters={"specialist": specialist, "conversation_id": conversation_id},
                limit=1,
                order_by="updated_at DESC",
            )
            return rows[0] if rows else {}
        except Exception as e:  # noqa: BLE001
            log.warning("Counter update failed for %s/%s: %s", specialist, conversation_id, e)
            return {}

    def should_backup(
        self,
        specialist: str,
        conversation_id: str,
        context_window: int,
        last_message_ts: float | None = None,
    ) -> tuple[bool, str]:
        """
        Returns (should_backup, trigger_reason).

        Triggers:
          token_threshold — context window ≥ 70% full
          message_count   — every MESSAGE_BACKUP_INTERVAL messages
          time_gap        — returning after >24h with ≥5 messages of substance
        """
        try:
            rows = self._state.query(
                "token_counters",
                filters={"specialist": specialist, "conversation_id": conversation_id},
                limit=1,
                order_by="updated_at DESC",
            )
            if not rows:
                return False, ""
            row = rows[0]
            tokens = row.get("token_estimate", 0) or 0
            count = row.get("message_count", 0) or 0
            if tokens >= int(context_window * TOKEN_THRESHOLD_PCT):
                return True, "token_threshold"
            if count > 0 and count % MESSAGE_BACKUP_INTERVAL == 0:
                return True, "message_count"
            if last_message_ts is not None and count >= 5:
                if (time.time() - last_message_ts) / 3600 >= 24:
                    return True, "time_gap"
            return False, ""
        except Exception as e:  # noqa: BLE001
            log.warning("should_backup check failed: %s", e)
            return False, ""

    # ── Backup generation + storage ───────────────────────────────────────────

    def build_backup_context(self, conversation_id: str) -> list[dict[str, Any]]:
        """
        Full-history rule (V-P0.5): the backup is generated from the canonical
        DB history (last 60 messages), never from whatever slice the client
        request happened to contain.
        """
        return self._state.get_messages(conversation_id, limit=BACKUP_HISTORY_MESSAGES)

    def store_backup(
        self,
        specialist: str,
        conversation_id: str,
        backup_text: str,
        trigger: str,
    ) -> str:
        """Store a backup and reset the token counter for the new thread."""
        try:
            rows = self._state.query(
                "token_counters",
                filters={"specialist": specialist, "conversation_id": conversation_id},
                limit=1,
                order_by="updated_at DESC",
            )
            thread_num = rows[0].get("thread_number", 1) if rows else 1
            if trigger == "token_threshold":
                thread_num += 1  # new thread after context reset

            backup_id = self._state.store_context_backup(
                conversation_id=conversation_id,
                specialist=specialist,
                backup_text=backup_text,
                trigger=trigger,
                thread_number=thread_num,
            )

            if rows:
                now = time.time()
                with self._state._connect() as conn:
                    conn.execute(
                        "UPDATE token_counters"
                        " SET token_estimate=0, message_count=0, thread_number=?,"
                        "     last_backup_at=?, updated_at=?"
                        " WHERE specialist=? AND conversation_id=?",
                        (thread_num, now, now, specialist, conversation_id),
                    )
            log.info(
                "Context backup stored: %s/%s (thread=%s trigger=%s)",
                specialist,
                conversation_id[:12],
                thread_num,
                trigger,
            )
            return backup_id
        except Exception as e:  # noqa: BLE001
            log.error("store_backup failed: %s", e)
            return ""

    def get_latest_backup(self, specialist: str, conversation_id: str) -> str | None:
        return self._state.get_latest_backup(conversation_id, specialist)

    # ── Coverage (V-P1.2) ─────────────────────────────────────────────────────

    def coverage_report(self, specialist: str) -> dict[str, Any]:
        """Stale/missing backup report for one specialist."""
        stale = self._state.stale_backup_conversations(
            specialist, min_messages=COVERAGE_MIN_MESSAGES
        )
        return {
            "specialist": specialist,
            "stale_count": len(stale),
            "stale_conversations": stale,
        }

    async def run_coverage_sweep(
        self,
        specialists: list[str],
        generator: BackupGenerator,
        pace_s: float = COVERAGE_PACE_S,
    ) -> dict[str, Any]:
        """
        One coverage pass: for every specialist, find conversations with stale
        or missing backups and regenerate them, paced at `pace_s` per call.
        """
        total_generated = 0
        errors: list[str] = []
        for specialist in specialists:
            try:
                stale = self._state.stale_backup_conversations(
                    specialist, min_messages=COVERAGE_MIN_MESSAGES
                )
            except Exception as e:  # noqa: BLE001
                errors.append(f"coverage query failed for {specialist}: {e}")
                continue
            if not stale:
                continue
            log.info("Coverage sweep: %d conversations need backup for %s", len(stale), specialist)
            for entry in stale:
                conv_id = entry["conversation_id"]
                status = "no_backup" if entry.get("backup_ts") is None else "stale"
                try:
                    history = self.build_backup_context(conv_id)
                    text = await generator(specialist, conv_id, BACKUP_PROMPT, history)
                    if text:
                        self.store_backup(
                            specialist, conv_id, text, trigger=f"coverage_job_{status}"
                        )
                        total_generated += 1
                    await asyncio.sleep(pace_s)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{specialist}:{conv_id[:8]}: {e}")
        return {"generated": total_generated, "errors": errors}

    async def coverage_job(
        self,
        specialists_provider: Callable[[], list[str]],
        generator: BackupGenerator,
        interval_s: float = COVERAGE_INTERVAL_S,
        first_run_delay_s: float = COVERAGE_FIRST_RUN_DELAY_S,
    ) -> None:
        """Background loop: first run after 60s, then every 6 hours."""
        await asyncio.sleep(first_run_delay_s)
        while True:
            try:
                specialists = specialists_provider()
                if specialists:
                    result = await self.run_coverage_sweep(specialists, generator)
                    if result["generated"]:
                        log.info("Coverage job complete: generated %d backups", result["generated"])
            except Exception as e:  # noqa: BLE001
                log.error("Context backup coverage job error: %s", e)
            await asyncio.sleep(interval_s)
