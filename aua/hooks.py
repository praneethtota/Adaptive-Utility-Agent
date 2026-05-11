"""
aua/hooks.py — Lifecycle hook system.

Hooks fire at named points in the request pipeline. They receive an event
dict and return a (possibly modified) event dict. They are called in YAML
registration order.

Hook points (in pipeline order):
    pre_query           Before field classification
    post_route          After routing decision, before specialist calls
    pre_specialist_call Before each individual specialist call
    post_specialist_call After each specialist responds
    pre_arbiter         Before arbiter runs
    post_arbiter        After arbiter issues verdict
    on_correction       After a correction is stored
    pre_response        Before response is sent to client
    post_response       After response is sent (async, non-blocking)
    on_promotion        After a GREEN model is promoted (async)
    on_rollback         After a rollback completes (async)

YAML registration:
    hooks:
      on_correction:
        - import_path: plugins.hooks:SlackNotificationHook
          config:
            webhook_url_secret: SLACK_WEBHOOK_URL
          fail_closed: false   # fail-open (default): log error, continue
          timeout_s: 5.0

Usage:
    from aua.hooks import HookRunner
    runner = HookRunner()
    runner.register("on_correction", my_hook, fail_closed=False)
    event = await runner.fire("on_correction", {"type": "on_correction", ...})
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

HOOK_POINTS = frozenset(
    [
        "pre_query",
        "post_route",
        "pre_specialist_call",
        "post_specialist_call",
        "pre_arbiter",
        "post_arbiter",
        "on_correction",
        "pre_response",
        "post_response",
        "on_promotion",
        "on_rollback",
    ]
)


class HookRegistration:
    def __init__(self, hook: Any, fail_closed: bool = False, timeout_s: float = 5.0) -> None:
        self.hook = hook
        self.fail_closed = fail_closed
        self.timeout_s = timeout_s


class HookRunner:
    """
    Manages hook registration and firing for all hook points.

    Thread-safe read (fire is async). Writes (register) happen at startup only.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookRegistration]] = {}

    def register(
        self,
        hook_point: str,
        hook: Any,
        fail_closed: bool = False,
        timeout_s: float = 5.0,
    ) -> None:
        """Register a hook for a named hook point."""
        if hook_point not in HOOK_POINTS:
            raise ValueError(
                f"Unknown hook point {hook_point!r}. " f"Valid points: {sorted(HOOK_POINTS)}"
            )
        self._hooks.setdefault(hook_point, []).append(
            HookRegistration(hook, fail_closed=fail_closed, timeout_s=timeout_s)
        )
        log.info("Registered hook: %s → %s", hook_point, type(hook).__name__)

    async def fire(self, hook_point: str, event: dict[str, Any]) -> dict[str, Any]:
        """
        Fire all hooks registered for hook_point in order.

        Each hook receives the event dict from the previous hook.
        fail-open hooks: log error and continue.
        fail-closed hooks: re-raise exception, aborting the request.

        Args:
            hook_point: name of the lifecycle point
            event:      event dict (at minimum: type, session_id, trace_id)

        Returns:
            Possibly modified event dict.
        """
        event = {**event, "type": hook_point}
        for reg in self._hooks.get(hook_point, []):
            try:
                result = await asyncio.wait_for(
                    reg.hook(event),
                    timeout=reg.timeout_s,
                )
                if isinstance(result, dict):
                    event = result
            except asyncio.TimeoutError:
                msg = f"Hook {type(reg.hook).__name__!r} timed out after {reg.timeout_s}s"
                log.warning(msg)
                if reg.fail_closed:
                    raise RuntimeError(msg)
            except Exception as exc:
                msg = f"Hook {type(reg.hook).__name__!r} raised: {exc}"
                log.error(msg)
                if reg.fail_closed:
                    raise
        return event

    def fire_background(self, hook_point: str, event: dict[str, Any]) -> None:
        """
        Fire hooks as a background task (for post_response, on_promotion, on_rollback).

        Errors are logged but never propagate to the caller.
        """

        async def _run() -> None:
            try:
                await self.fire(hook_point, event)
            except Exception as exc:
                log.error("Background hook %s failed: %s", hook_point, exc)

        asyncio.ensure_future(_run())

    def registered_hooks(self) -> dict[str, list[str]]:
        """Return {hook_point: [class_names]} for status/doctor output."""
        return {point: [type(r.hook).__name__ for r in regs] for point, regs in self._hooks.items()}


# Global hook runner — populated at startup
_runner: HookRunner | None = None


def get_hook_runner() -> HookRunner:
    global _runner
    if _runner is None:
        _runner = HookRunner()
    return _runner


def reset_hook_runner() -> HookRunner:
    global _runner
    _runner = HookRunner()
    return _runner
