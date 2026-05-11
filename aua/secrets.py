"""
aua/secrets.py — Secrets management for AUA Framework.

Config references secret names, not values. Secrets are resolved at
startup from the configured provider.

Supported providers:
    env      Environment variables (default, always available)
    vault    HashiCorp Vault (requires hvac)
    aws      AWS Secrets Manager (requires boto3)
    gcp      GCP Secret Manager (requires google-cloud-secret-manager)

YAML configuration:
    secrets:
      provider: env          # default

    # Reference a secret by env var name:
    api_key: ${OPENAI_API_KEY}   # resolved at load time

    # Or explicitly:
    secrets:
      provider: aws
      region: us-east-1
    api_key_secret: my-openai-key   # looks up in AWS Secrets Manager

Usage:
    from aua.secrets import resolve_secret, SecretsManager
    mgr = SecretsManager.from_config(config)
    value = mgr.get("OPENAI_API_KEY")

    # Or resolve inline ${VAR} references in strings:
    resolved = resolve_secret("${OPENAI_API_KEY}", mgr)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

# Pattern for ${VAR_NAME} references in config values
_SECRET_REF = re.compile(r"\$\{([^}]+)\}")


class SecretNotFoundError(ValueError):
    """Raised when a secret cannot be resolved from the configured provider."""

    def __init__(self, name: str, provider: str) -> None:
        super().__init__(
            f"Secret {name!r} not found in provider {provider!r}. "
            f"Set the environment variable or check your secrets provider config."
        )
        self.name = name
        self.provider = provider


class SecretsManager:
    """
    Resolves secret references to their actual values.

    Providers are tried in order: configured provider → env fallback.
    Values are never logged or included in GET /config responses.
    """

    def __init__(self, provider: str = "env", **kwargs: Any) -> None:
        self._provider = provider
        self._kwargs = kwargs
        self._cache: dict[str, str] = {}

    @classmethod
    def from_config(cls, config: Any | None = None) -> SecretsManager:
        """Create a SecretsManager from AUAConfig."""
        if config is None:
            return cls(provider="env")

        secrets_cfg = getattr(config, "secrets", None)
        if secrets_cfg is None:
            return cls(provider="env")

        provider = getattr(secrets_cfg, "provider", "env")
        kwargs: dict[str, Any] = {}

        if provider == "aws":
            kwargs["region"] = getattr(secrets_cfg, "region", "us-east-1")
        elif provider == "vault":
            kwargs["url"] = getattr(secrets_cfg, "url", "http://127.0.0.1:8200")
            kwargs["token_env"] = getattr(secrets_cfg, "token_env", "VAULT_TOKEN")

        return cls(provider=provider, **kwargs)

    def get(self, name: str, required: bool = True) -> str | None:
        """
        Resolve a secret by name.

        Args:
            name:     secret name / env var name
            required: if True, raise SecretNotFoundError when not found

        Returns:
            Secret value string, or None if not required and not found.
        """
        if name in self._cache:
            return self._cache[name]

        value = self._resolve(name)

        if value is None and required:
            raise SecretNotFoundError(name, self._provider)

        if value is not None:
            self._cache[name] = value
            log.debug("Resolved secret: %s [provider=%s]", name, self._provider)

        return value

    def _resolve(self, name: str) -> str | None:
        """Try to resolve the secret from the configured provider."""
        # Always try env first (fastest, always available)
        env_val = os.environ.get(name)
        if env_val:
            return env_val

        if self._provider == "env":
            return None

        if self._provider == "aws":
            return self._resolve_aws(name)
        elif self._provider == "vault":
            return self._resolve_vault(name)
        elif self._provider == "gcp":
            return self._resolve_gcp(name)

        return None

    def _resolve_aws(self, name: str) -> str | None:
        try:
            import boto3

            client = boto3.client("secretsmanager", region_name=self._kwargs.get("region"))
            resp = client.get_secret_value(SecretId=name)
            return resp.get("SecretString")
        except ImportError:
            log.warning("boto3 not installed — cannot resolve AWS secret %s", name)
        except Exception as e:
            log.warning("AWS Secrets Manager error for %s: %s", name, e)
        return None

    def _resolve_vault(self, name: str) -> str | None:
        try:
            import hvac

            token = os.environ.get(self._kwargs.get("token_env", "VAULT_TOKEN"))
            client = hvac.Client(url=self._kwargs.get("url"), token=token)
            resp = client.secrets.kv.v2.read_secret_version(path=name)
            data = resp["data"]["data"]
            return data.get("value") or data.get(name)
        except ImportError:
            log.warning("hvac not installed — cannot resolve Vault secret %s", name)
        except Exception as e:
            log.warning("Vault error for %s: %s", name, e)
        return None

    def _resolve_gcp(self, name: str) -> str | None:
        try:
            from google.cloud import secretmanager

            client = secretmanager.SecretManagerServiceClient()
            project = self._kwargs.get("project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            path = f"projects/{project}/secrets/{name}/versions/latest"
            resp = client.access_secret_version(name=path)
            return resp.payload.data.decode("utf-8")
        except ImportError:
            log.warning(
                "google-cloud-secret-manager not installed — cannot resolve GCP secret %s", name
            )
        except Exception as e:
            log.warning("GCP Secret Manager error for %s: %s", name, e)
        return None

    @property
    def provider(self) -> str:
        return self._provider


def resolve_secret(value: str, mgr: SecretsManager | None = None) -> str:
    """
    Resolve ${VAR_NAME} references in a string value.

    Example:
        resolve_secret("Bearer ${OPENAI_API_KEY}", mgr)
        → "Bearer sk-..."
    """
    if not isinstance(value, str) or "${" not in value:
        return value

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        if mgr:
            resolved = mgr.get(name, required=False)
            if resolved:
                return resolved
        # Fall back to env
        return os.environ.get(name, match.group(0))

    return _SECRET_REF.sub(_replace, value)


# Global secrets manager — initialised at serve startup
_manager: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    global _manager
    if _manager is None:
        _manager = SecretsManager()
    return _manager


def init_secrets_manager(config: Any) -> SecretsManager:
    global _manager
    _manager = SecretsManager.from_config(config)
    return _manager
