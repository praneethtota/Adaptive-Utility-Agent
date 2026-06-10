"""
aua/remote_config.py — Remote model config with fallback chain (V-P1.6).

Lets model aliases, pricing, and context windows be updated without a code
release. Production motivation from AUA-Veritas: Gemini 1.5 Pro was silently
deprecated upstream with a 404 and required a full rebuild to fix before this
system existed.

Fallback chain:
    remote fetch succeeds → use remote merged onto the built-in registry
    fetch fails (offline) → use DB-cached config from the last good fetch
    no cache in DB        → use the hardcoded built-in registry

What the remote config may update (no release needed):
    full_id, display_name, context_window, input_cost_per_1m,
    output_cost_per_1m, recommended_vram_gb, deprecated list,
    model_id_renames

What stays local (requires a code change):
    backend, provider — these map to actual plugin classes.

Remote JSON shape:
    {"schema_version": 1,
     "models": {"alias": {...updatable fields...}},
     "deprecated": ["old-alias"],
     "model_id_renames": {"old-alias": "new-alias"}}
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aua.state import SQLiteStateStore

log = logging.getLogger("aua.remote_config")

DEFAULT_REMOTE_URL = "https://praneethtota.github.io/Adaptive-Utility-Agent/models.json"
CACHE_TTL_SECONDS = 86400  # refresh every 24 hours
CACHE_KEEP_SECONDS = CACHE_TTL_SECONDS * 7  # serve stale cache up to 7 days
FETCH_TIMEOUT_SECONDS = 8.0
CACHE_KEY = "models"

REMOTE_UPDATABLE_FIELDS = frozenset(
    {
        "full_id",
        "display_name",
        "context_window",
        "input_cost_per_1m",
        "output_cost_per_1m",
        "recommended_vram_gb",
        "recommended_fields",
    }
)


def remote_url() -> str:
    return os.environ.get("AUA_REMOTE_MODELS_URL", DEFAULT_REMOTE_URL).strip()


async def fetch_remote_config(url: str | None = None) -> dict[str, Any] | None:
    """Fetch models.json. Returns the parsed dict, or None on any failure."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS) as client:
            resp = await client.get(url or remote_url())
            resp.raise_for_status()
            data = resp.json()
        if data.get("schema_version") != 1:
            log.warning(
                "Remote model config: unsupported schema_version=%s",
                data.get("schema_version"),
            )
            return None
        return data
    except Exception as e:  # noqa: BLE001
        log.info("Remote model config fetch failed (using cache/fallback): %s", e)
        return None


def merge_remote_into_registry(
    base: dict[str, dict[str, Any]],
    remote: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    Merge a remote config into a copy of the built-in model registry.

    Rules:
      - Remote may update only REMOTE_UPDATABLE_FIELDS on existing aliases.
      - Remote may add new aliases only when they name a provider already in
        the registry (so the backend mapping can be inherited).
      - backend/provider are never overwritten from remote.
    """
    merged = {k: dict(v) for k, v in base.items()}
    provider_to_backend: dict[str, str] = {}
    for spec in base.values():
        if spec.get("provider") and spec.get("backend"):
            provider_to_backend[spec["provider"]] = spec["backend"]

    for alias, remote_spec in (remote.get("models") or {}).items():
        if alias in merged:
            for field in REMOTE_UPDATABLE_FIELDS:
                if field in remote_spec:
                    merged[alias][field] = remote_spec[field]
        else:
            provider = remote_spec.get("provider", "")
            if provider in provider_to_backend:
                entry = {k: v for k, v in remote_spec.items() if k in REMOTE_UPDATABLE_FIELDS}
                entry["provider"] = provider
                entry["backend"] = provider_to_backend[provider]
                merged[alias] = entry
                log.info("Remote model config: added new alias %s (provider=%s)", alias, provider)
            else:
                log.warning(
                    "Remote model config: skipped %s — provider %r unknown", alias, provider
                )

    deprecated = list(remote.get("deprecated") or [])
    return merged, deprecated


def get_renames(remote: dict[str, Any]) -> dict[str, str]:
    """Return the {old_alias: new_alias} rename map from a remote config."""
    return dict(remote.get("model_id_renames") or {})


# ── DB cache ──────────────────────────────────────────────────────────────────


def load_cached_config(state: SQLiteStateStore) -> dict[str, Any] | None:
    """Load the last successfully fetched config (kept up to 7 days)."""
    try:
        row = state.get("remote_config_cache", CACHE_KEY)
        if row and row.get("payload"):
            if time.time() - (row.get("fetched_at") or 0) < CACHE_KEEP_SECONDS:
                return json.loads(row["payload"])
    except Exception as e:  # noqa: BLE001
        log.debug("remote_config_cache read failed: %s", e)
    return None


def save_cached_config(state: SQLiteStateStore, remote: dict[str, Any]) -> None:
    try:
        now = time.time()
        state.set(
            "remote_config_cache",
            CACHE_KEY,
            {"created_at": now, "fetched_at": now, "payload": json.dumps(remote)},
        )
    except Exception as e:  # noqa: BLE001
        log.warning("remote_config_cache write failed: %s", e)


# ── Manager ───────────────────────────────────────────────────────────────────


class RemoteModelConfig:
    """
    Live model registry with remote refresh.

    Usage:
        mgr = RemoteModelConfig(state_store)
        await mgr.refresh()          # at startup and every 24h
        mgr.models                   # merged registry, always valid
        mgr.deprecated, mgr.renames
    """

    def __init__(
        self,
        state: SQLiteStateStore,
        base: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if base is None:
            from aua.defaults.registry import get_defaults

            base = get_defaults("models")
        self._state = state
        self._base = base  # hardcoded fallback — never mutated
        self.models: dict[str, dict[str, Any]] = {k: dict(v) for k, v in base.items()}
        self.deprecated: list[str] = []
        self.renames: dict[str, str] = {}
        self.source: str = "builtin"  # 'remote' | 'cache' | 'builtin'
        self._last_fetch: float = 0.0

    async def refresh(self, force: bool = False) -> bool:
        """Refresh via the fallback chain. Returns True when remote/cache used."""
        if not force and (time.time() - self._last_fetch) < CACHE_TTL_SECONDS:
            return self.source in ("remote", "cache")

        remote = await fetch_remote_config()
        if remote:
            save_cached_config(self._state, remote)
            self._last_fetch = time.time()
            self.models, self.deprecated = merge_remote_into_registry(self._base, remote)
            self.renames = get_renames(remote)
            self.source = "remote"
            log.info(
                "Model config refreshed from remote: %d models, %d deprecated, %d renames",
                len(self.models),
                len(self.deprecated),
                len(self.renames),
            )
            return True

        cached = load_cached_config(self._state)
        if cached:
            self.models, self.deprecated = merge_remote_into_registry(self._base, cached)
            self.renames = get_renames(cached)
            self.source = "cache"
            log.info("Model config loaded from DB cache (remote unavailable)")
            return True

        self.models = {k: dict(v) for k, v in self._base.items()}
        self.deprecated = []
        self.renames = {}
        self.source = "builtin"
        log.warning("Model config using hardcoded fallback (no remote, no cache)")
        return False

    def resolve_alias(self, alias: str) -> str:
        """Apply the rename map — current alias for a possibly-deprecated one."""
        return self.renames.get(alias, alias)

    def is_deprecated(self, alias: str) -> bool:
        return alias in self.deprecated

    async def refresh_job(self, interval_s: float = CACHE_TTL_SECONDS) -> None:
        """Background loop: refresh every 24h so retired model IDs hot-swap."""
        import asyncio

        while True:
            await asyncio.sleep(interval_s)
            try:
                await self.refresh(force=True)
            except Exception as e:  # noqa: BLE001
                log.error("Model config refresh job error: %s", e)
