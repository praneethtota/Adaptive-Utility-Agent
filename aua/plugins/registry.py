"""
aua/plugins/registry.py — Plugin registry and extension import system.

Handles the full plugin loading lifecycle:
  1. Parse import_path: "module.path:ClassName"
  2. Import module
  3. Instantiate with config dict
  4. Validate against Protocol (isinstance check)
  5. Register in global registry

YAML syntax:
    utility_scorer:
      import_path: plugins.custom_utility:RiskWeightedUtilityScorer
      config:
        risk_weight: 0.7

    middleware:
      - import_path: plugins.middleware:PIIRedactionMiddleware
        config:
          patterns: ["\\d{3}-\\d{2}-\\d{4}"]

    hooks:
      on_correction:
        - import_path: plugins.hooks:SlackNotificationHook
          config:
            webhook_url_secret: SLACK_WEBHOOK_URL
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from aua.plugins.errors import AUA_PLUGIN_CONTRACT_INVALID, AUA_PLUGIN_LOAD_FAILED
from aua.plugins.interfaces import (
    ArbiterPolicyPlugin,
    AUAMiddleware,
    CorrectionStorePlugin,
    FieldClassifierPlugin,
    HookPlugin,
    ModelBackendPlugin,
    PromotionPolicyPlugin,
    StateStorePlugin,
    UtilityScorerPlugin,
)

log = logging.getLogger(__name__)

# Map plugin kind → expected Protocol
_PROTOCOL_MAP: dict[str, type] = {
    "field_classifier": FieldClassifierPlugin,
    "utility_scorer": UtilityScorerPlugin,
    "arbiter_policy": ArbiterPolicyPlugin,
    "promotion_policy": PromotionPolicyPlugin,
    "correction_store": CorrectionStorePlugin,
    "model_backend": ModelBackendPlugin,
    "state_store": StateStorePlugin,
    "hook": HookPlugin,
    "middleware": AUAMiddleware,
}


class PluginLoadError(RuntimeError):
    """Raised when a plugin cannot be loaded or fails contract validation."""

    def __init__(self, code: Any, import_path: str, reason: str) -> None:
        self.code = code
        self.import_path = import_path
        self.reason = reason
        super().__init__(f"{code.code}: {import_path} — {reason}")


def load_plugin(
    import_path: str,
    kind: str,
    config: dict[str, Any] | None = None,
) -> Any:
    """
    Load, instantiate, and validate a plugin.

    Args:
        import_path: "module.path:ClassName"
        kind:        plugin type key (see _PROTOCOL_MAP)
        config:      optional config dict passed to the plugin constructor

    Returns:
        Instantiated plugin object.

    Raises:
        PluginLoadError on import failure or protocol mismatch.
    """
    config = config or {}

    # Parse import_path
    if ":" not in import_path:
        raise PluginLoadError(
            AUA_PLUGIN_LOAD_FAILED,
            import_path,
            "import_path must be 'module.path:ClassName'",
        )
    module_path, class_name = import_path.rsplit(":", 1)

    # Import module
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise PluginLoadError(
            AUA_PLUGIN_LOAD_FAILED,
            import_path,
            f"Cannot import module {module_path!r}: {exc}",
        ) from exc

    # Get class
    cls = getattr(module, class_name, None)
    if cls is None:
        raise PluginLoadError(
            AUA_PLUGIN_LOAD_FAILED,
            import_path,
            f"Class {class_name!r} not found in module {module_path!r}",
        )

    # Instantiate
    try:
        instance = cls(**config) if config else cls()
    except TypeError as exc:
        raise PluginLoadError(
            AUA_PLUGIN_LOAD_FAILED,
            import_path,
            f"Failed to instantiate {class_name}: {exc}",
        ) from exc
    except Exception as exc:
        raise PluginLoadError(
            AUA_PLUGIN_LOAD_FAILED,
            import_path,
            f"Constructor raised: {exc}",
        ) from exc

    # Validate protocol
    if kind in _PROTOCOL_MAP:
        protocol = _PROTOCOL_MAP[kind]
        if not isinstance(instance, protocol):
            raise PluginLoadError(
                AUA_PLUGIN_CONTRACT_INVALID,
                import_path,
                f"{class_name!r} does not implement {protocol.__name__}. "
                f"Check that all required methods are present.",
            )

    log.info("Loaded plugin: %s [kind=%s]", import_path, kind)
    return instance


class PluginRegistry:
    """
    Runtime registry for all loaded plugins.

    Populated at startup. On hot reload, a new registry is built and
    swapped atomically.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Any] = {}
        self._middleware: list[Any] = []
        self._hooks: dict[str, list[Any]] = {}

    def register(self, name: str, plugin: Any) -> None:
        self._plugins[name] = plugin
        log.debug("Registered plugin: %s", name)

    def register_middleware(self, mw: Any) -> None:
        self._middleware.append(mw)

    def register_hook(self, hook_point: str, hook: Any) -> None:
        self._hooks.setdefault(hook_point, []).append(hook)

    def get(self, name: str) -> Any | None:
        return self._plugins.get(name)

    def get_middleware(self) -> list[Any]:
        return list(self._middleware)

    def get_hooks(self, hook_point: str) -> list[Any]:
        return list(self._hooks.get(hook_point, []))

    def __repr__(self) -> str:
        return (
            f"PluginRegistry("
            f"plugins={list(self._plugins.keys())}, "
            f"middleware={len(self._middleware)}, "
            f"hooks={dict((k, len(v)) for k, v in self._hooks.items())})"
        )


# Global registry instance — populated at serve startup
_global_registry: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_registry() -> PluginRegistry:
    """Reset and return a fresh registry (used on hot reload)."""
    global _global_registry
    _global_registry = PluginRegistry()
    return _global_registry
