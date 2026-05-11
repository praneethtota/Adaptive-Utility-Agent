"""
aua/hot_reload.py — Hot-reload support for aua_config.yaml.

Handles SIGHUP-triggered config reload without full restart.

Hot-reloadable (no restart needed):
    routing thresholds, utility weights, promotion thresholds,
    logging level, cors_origins

Requires partial restart (specialist server restart):
    new specialist, changed model, changed port, changed backend

Usage (programmatic):
    from aua.hot_reload import HotReloader
    reloader = HotReloader(config_path, router)
    signal.signal(signal.SIGHUP, reloader.handle_sighup)

CLI:
    aua config reload
"""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Fields that can be reloaded without restarting any process
_HOT_FIELDS = {
    "router.single_domain_threshold",
    "router.fanout_threshold",
    "router.specialist_timeout",
    "router.cors_origins",
    "logging.level",
    "blue_green",  # delta / T_min / tau per specialist
}

# Fields that require a specialist server restart
_RESTART_REQUIRED_FIELDS = {
    "specialists[*].model",
    "specialists[*].port",
    "specialists[*].gpu",
    "specialists[*].gpu_memory_utilization",
    "arbiter.model",
    "arbiter.port",
    "backend",
}


class ReloadResult:
    """Result of a hot-reload attempt."""

    def __init__(self) -> None:
        self.success: bool = False
        self.hot_reloaded: list[str] = []
        self.restart_required: list[str] = []
        self.errors: list[str] = []

    def __str__(self) -> str:
        lines = []
        if self.success:
            lines.append(f"✓ Reloaded: {', '.join(self.hot_reloaded) or 'no changes'}")
        if self.restart_required:
            lines.append(f"⚠ Restart required for: {', '.join(self.restart_required)}")
        if self.errors:
            lines.append(f"✗ Errors: {', '.join(self.errors)}")
        return "\n".join(lines) if lines else "No changes detected."


class HotReloader:
    """
    Watches aua_config.yaml and applies hot-reloadable changes to a running Router.

    Triggered by:
        - SIGHUP signal (Unix standard for config reload)
        - aua config reload CLI command
        - POST /config/reload REST endpoint (future)
    """

    def __init__(self, config_path: str | os.PathLike, router: Any | None = None) -> None:
        self._config_path = Path(config_path)
        self._router = router
        self._last_mtime: float = 0.0
        self._current_config: Any = None

    def attach(self, router: Any) -> None:
        """Attach to a running Router instance."""
        self._router = router

    def register_sighup(self) -> None:
        """Register SIGHUP handler (Unix only — no-op on Windows)."""
        try:
            signal.signal(signal.SIGHUP, self._sighup_handler)
            log.info("SIGHUP handler registered — send SIGHUP to reload config")
        except (OSError, AttributeError):
            log.debug("SIGHUP not available on this platform (Windows?)")

    def _sighup_handler(self, signum: int, frame: Any) -> None:
        log.info("SIGHUP received — reloading config...")
        result = self.reload()
        log.info(str(result))

    def has_changed(self) -> bool:
        """Return True if aua_config.yaml has been modified since last load."""
        try:
            mtime = self._config_path.stat().st_mtime
            return mtime > self._last_mtime
        except OSError:
            return False

    def reload(self) -> ReloadResult:
        """
        Reload aua_config.yaml and apply hot-reloadable changes.

        Returns a ReloadResult describing what was reloaded and what
        requires a restart.
        """
        result = ReloadResult()

        # ── Validate new config first (atomic — don't apply if invalid) ───────
        try:
            from aua.config import load_config

            new_cfg = load_config(self._config_path)
        except Exception as e:
            result.errors.append(f"Config validation failed: {e}")
            log.error("Hot reload aborted — new config is invalid: %s", e)
            return result

        old_cfg = self._current_config

        # ── First load (no diff possible) ─────────────────────────────────────
        if old_cfg is None:
            self._current_config = new_cfg
            self._last_mtime = self._config_path.stat().st_mtime
            result.success = True
            return result

        # ── Diff and apply ────────────────────────────────────────────────────
        changes = _diff_configs(old_cfg, new_cfg)

        for field, (old_val, new_val) in changes.items():
            if _is_hot_reloadable(field):
                _apply_hot(field, new_val, self._router, new_cfg)
                result.hot_reloaded.append(field)
                log.info("Hot-reloaded: %s = %s", field, new_val)
            else:
                result.restart_required.append(field)
                log.warning("Restart required for field: %s (%s → %s)", field, old_val, new_val)

        self._current_config = new_cfg
        self._last_mtime = self._config_path.stat().st_mtime
        result.success = True
        return result


def _diff_configs(old: Any, new: Any) -> dict[str, tuple[Any, Any]]:
    """Return {field_path: (old_value, new_value)} for changed fields."""
    changes: dict[str, tuple[Any, Any]] = {}

    def _check(path: str, o: Any, n: Any) -> None:
        if o != n:
            changes[path] = (o, n)

    # Router thresholds
    _check(
        "router.single_domain_threshold",
        old.router.single_domain_threshold,
        new.router.single_domain_threshold,
    )
    _check("router.fanout_threshold", old.router.fanout_threshold, new.router.fanout_threshold)
    _check(
        "router.specialist_timeout", old.router.specialist_timeout, new.router.specialist_timeout
    )
    _check("router.cors_origins", old.router.cors_origins, new.router.cors_origins)

    # Logging
    _check("logging.level", old.logging.level, new.logging.level)

    # Blue-green thresholds
    for name in set(list(old.blue_green) + list(new.blue_green)):
        o_bg = old.blue_green.get(name)
        n_bg = new.blue_green.get(name)
        if o_bg != n_bg:
            changes[f"blue_green.{name}"] = (o_bg, n_bg)

    # Specialist changes (require restart)
    old_specs = {s.name: s for s in old.specialists}
    new_specs = {s.name: s for s in new.specialists}
    for name in set(list(old_specs) + list(new_specs)):
        o_s = old_specs.get(name)
        n_s = new_specs.get(name)
        if o_s != n_s:
            changes[f"specialists.{name}"] = (o_s, n_s)

    # Arbiter changes (require restart)
    if old.arbiter != new.arbiter:
        changes["arbiter"] = (old.arbiter, new.arbiter)

    # Backend change (requires restart)
    if old.backend != new.backend:
        changes["backend"] = (old.backend, new.backend)

    return changes


def _is_hot_reloadable(field: str) -> bool:
    """Return True if the field can be reloaded without restarting."""
    for hot in _HOT_FIELDS:
        if field.startswith(hot):
            return True
    return False


def _apply_hot(field: str, value: Any, router: Any | None, new_cfg: Any) -> None:
    """Apply a hot-reloadable change to the running router."""
    if router is None:
        return

    if field == "router.single_domain_threshold":
        router._config.router.single_domain_threshold = value
    elif field == "router.fanout_threshold":
        router._config.router.fanout_threshold = value
    elif field == "router.specialist_timeout":
        router._config.router.specialist_timeout = value
    elif field == "router.cors_origins":
        router._config.router.cors_origins = value
        # CORS middleware rebuild requires app restart — log a note
        log.warning("cors_origins changed — restart router process to apply CORS update")
    elif field == "logging.level":
        logging.getLogger("aua").setLevel(value)
        router._config.logging.level = value
    elif field.startswith("blue_green."):
        name = field.split(".", 1)[1]
        router._config.blue_green = new_cfg.blue_green
        log.info("Blue-green thresholds updated for %s", name)
