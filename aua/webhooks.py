"""
aua/webhooks.py — Webhook event delivery for AUA Framework.

Sends structured event payloads to configured webhook URLs with
automatic retry and exponential backoff.

Events:
    specialist_promoted         Green model promoted to blue
    rollback_completed          Specialist rolled back
    contradiction_threshold_exceeded  Contradiction rate crossed threshold
    arbiter_inconclusive_spike  Case 4 verdict rate spiked
    specialist_down             Specialist health check failed
    plugin_failure              Plugin raised exception
    security_auth_failure_spike Auth failures spiking

Configuration:
    webhooks:
      slack:
        url_secret: SLACK_WEBHOOK_URL
        events:
          - specialist_promoted
          - specialist_down
          - high_contradiction_rate
      pagerduty:
        url_secret: PAGERDUTY_WEBHOOK_URL
        events:
          - specialist_down

Usage:
    from aua.webhooks import WebhookDispatcher
    dispatcher = WebhookDispatcher.from_config(config)
    await dispatcher.send("specialist_promoted", {"specialist": "swe", ...})
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

VALID_EVENTS = frozenset(
    [
        "specialist_promoted",
        "rollback_completed",
        "contradiction_threshold_exceeded",
        "arbiter_inconclusive_spike",
        "specialist_down",
        "plugin_failure",
        "security_auth_failure_spike",
        "high_contradiction_rate",
        "low_utility_score",
        "hook_failure",
    ]
)


class WebhookTarget:
    """A single webhook endpoint with event filter."""

    def __init__(self, name: str, url: str, events: list[str]) -> None:
        self.name = name
        self.url = url
        self.events = set(events) if events else VALID_EVENTS

    def should_receive(self, event_type: str) -> bool:
        return event_type in self.events or not self.events


class WebhookDispatcher:
    """
    Delivers webhook events to all configured targets.

    Delivery is fire-and-forget with retry (3 attempts, exponential backoff).
    Failures are logged but never propagate to the caller.
    """

    def __init__(self, targets: list[WebhookTarget]) -> None:
        self._targets = targets

    @classmethod
    def from_config(cls, config: Any | None = None) -> WebhookDispatcher:
        if config is None:
            return cls([])

        wh_cfg = getattr(config, "webhooks", None)
        if wh_cfg is None:
            return cls([])

        items = wh_cfg.items() if isinstance(wh_cfg, dict) else []
        targets = []
        for name, cfg in items:
            url_secret = cfg.get("url_secret") if isinstance(cfg, dict) else None
            if not url_secret:
                continue
            url = os.environ.get(url_secret, "")
            if not url:
                log.warning("Webhook %s: secret %s not set — skipping", name, url_secret)
                continue
            events = cfg.get("events", []) if isinstance(cfg, dict) else []
            targets.append(WebhookTarget(name=name, url=url, events=events))
            log.info("Webhook registered: %s (events: %s)", name, events or "all")

        return cls(targets)

    async def send(self, event_type: str, payload: dict[str, Any]) -> None:
        """Fire-and-forget webhook delivery to all subscribed targets."""
        if not self._targets:
            return

        full_payload = {
            "event": event_type,
            "timestamp": time.time(),
            "framework": "aua",
            "version": "0.9.0rc1",
            **payload,
        }

        for target in self._targets:
            if target.should_receive(event_type):
                asyncio.ensure_future(self._deliver(target, full_payload))

    async def _deliver(
        self, target: WebhookTarget, payload: dict[str, Any], max_retries: int = 3
    ) -> None:
        """Deliver with exponential backoff retry."""
        try:
            import httpx
        except ImportError:
            log.warning("httpx not available — webhook delivery skipped")
            return

        body = json.dumps(payload, default=str)
        headers = {"Content-Type": "application/json", "User-Agent": "AUA-Framework/0.9"}

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(target.url, content=body, headers=headers)
                    if resp.status_code < 300:
                        log.info(
                            "Webhook delivered: %s → %s (%d)",
                            payload["event"],
                            target.name,
                            resp.status_code,
                        )
                        return
                    log.warning("Webhook %s returned %d", target.name, resp.status_code)
            except Exception as e:
                log.warning("Webhook %s attempt %d failed: %s", target.name, attempt + 1, e)

            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

        log.error(
            "Webhook %s: all %d attempts failed for event %s",
            target.name,
            max_retries,
            payload["event"],
        )


# Global dispatcher — set at serve startup
_dispatcher: WebhookDispatcher | None = None


def get_webhook_dispatcher() -> WebhookDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = WebhookDispatcher([])
    return _dispatcher


def init_webhook_dispatcher(config: Any) -> WebhookDispatcher:
    global _dispatcher
    _dispatcher = WebhookDispatcher.from_config(config)
    return _dispatcher
