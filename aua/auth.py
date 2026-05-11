"""
aua/auth.py — Bearer token authentication and scope validation.

Design:
  - Tokens are signed with HMAC-SHA256 using a secret key
  - Each token carries scopes (from the permission matrix in docs/)
  - Token metadata is stored in the state store
  - Revocation is checked on every request via a revocation list in state

Scopes (from docs/permission_scope_matrix.md):
    aua:query, aua:stream, aua:batch, aua:status,
    aua:config:read, aua:config:write,
    aua:corrections:read, aua:corrections:write,
    aua:deploy, aua:rollback,
    aua:extensions:read, aua:extensions:write,
    aua:tokens:read, aua:tokens:write,
    aua:admin

Configuration:
    security:
      auth_enabled: false    # true in production
      token_secret_env: AUA_TOKEN_SECRET
      token_expiry_days: 30

Local dev:
    When auth_enabled=false, all endpoints are open with a WARNING.
    Never set auth_enabled=false on a public endpoint.

CLI:
    aua token create --scope aua:query --expires 30d
    aua token list
    aua token revoke <token-id>
    aua token inspect <token-id>
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# All valid scopes
VALID_SCOPES = frozenset(
    [
        "aua:query",
        "aua:stream",
        "aua:batch",
        "aua:status",
        "aua:config:read",
        "aua:config:write",
        "aua:corrections:read",
        "aua:corrections:write",
        "aua:deploy",
        "aua:rollback",
        "aua:extensions:read",
        "aua:extensions:write",
        "aua:tokens:read",
        "aua:tokens:write",
        "aua:admin",
    ]
)

# Endpoint → required scope
ENDPOINT_SCOPES: dict[str, str] = {
    "POST /query": "aua:query",
    "POST /query/stream": "aua:stream",
    "POST /query/batch": "aua:batch",
    "GET /status": "aua:status",
    "GET /config": "aua:config:read",
    "POST /config/reload": "aua:config:write",
    "GET /corrections": "aua:corrections:read",
    "POST /corrections": "aua:corrections:write",
    "POST /deploy/green": "aua:deploy",
    "POST /deploy/rollback": "aua:rollback",
    "GET /extensions": "aua:extensions:read",
    "POST /extensions/reload": "aua:extensions:write",
    "POST /extensions/test": "aua:extensions:write",
    "GET /metrics": "aua:status",
    "GET /metrics/cost": "aua:status",
}

# Public endpoints (no auth required)
PUBLIC_ENDPOINTS = frozenset(
    [
        "GET /health/live",
        "GET /health/ready",
        "GET /health/startup",
        "GET /version",
        "GET /docs",
        "GET /openapi.json",
    ]
)


class TokenError(ValueError):
    """Raised when a token is invalid, expired, or revoked."""

    pass


class AUAToken:
    """
    A signed AUA access token.

    Format: base64url(json_payload).base64url(signature)
    Signature: HMAC-SHA256(payload, secret)
    """

    def __init__(
        self,
        token_id: str,
        scopes: list[str],
        expires_at: float,
        created_at: float,
        label: str = "",
    ) -> None:
        self.token_id = token_id
        self.scopes = list(scopes)
        self.expires_at = expires_at
        self.created_at = created_at
        self.label = label

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def is_admin(self) -> bool:
        return "aua:admin" in self.scopes

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or self.is_admin

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "scopes": self.scopes,
            "expires_at": self.expires_at,
            "expires_at_human": datetime.fromtimestamp(self.expires_at, timezone.utc).isoformat(),
            "created_at": self.created_at,
            "label": self.label,
            "is_expired": self.is_expired,
        }


class TokenManager:
    """
    Creates, signs, validates, and revokes AUA tokens.

    Token storage uses the state store (SQLite by default).
    """

    TOKEN_TABLE = "tokens"

    def __init__(self, secret: str, store: Any | None = None) -> None:
        self._secret = secret.encode() if isinstance(secret, str) else secret
        self._store = store

    @classmethod
    def from_config(cls, config: Any | None = None) -> TokenManager:
        """Create from AUAConfig, reading secret from env."""
        secret_env = "AUA_TOKEN_SECRET"
        if config:
            sec_cfg = getattr(config, "security", None)
            if sec_cfg:
                secret_env = getattr(sec_cfg, "token_secret_env", secret_env)

        secret = os.environ.get(secret_env, "")
        if not secret:
            # Generate a random secret for dev — NOT suitable for production
            secret = hashlib.sha256(os.urandom(32)).hexdigest()
            log.warning(
                "AUA_TOKEN_SECRET not set — using ephemeral secret. "
                "Tokens will not survive restart. Set AUA_TOKEN_SECRET in production."
            )

        store = None
        if config:
            try:
                from aua.state import get_state_store

                store = get_state_store(config)
            except Exception:
                pass

        return cls(secret=secret, store=store)

    def create(
        self,
        scopes: list[str],
        expires_days: int = 30,
        label: str = "",
    ) -> tuple[AUAToken, str]:
        """
        Create and sign a new token.

        Args:
            scopes:       list of scope strings
            expires_days: days until expiry
            label:        human-readable label

        Returns:
            (AUAToken, token_string) — store the token_string securely
        """
        # Validate scopes
        invalid = set(scopes) - VALID_SCOPES - {"aua:admin"}
        if invalid:
            raise ValueError(f"Invalid scopes: {invalid}. Valid: {sorted(VALID_SCOPES)}")

        token_id = str(uuid.uuid4())
        created_at = time.time()
        expires_at = created_at + expires_days * 86400

        payload = {
            "token_id": token_id,
            "scopes": scopes,
            "expires_at": expires_at,
            "created_at": created_at,
            "label": label,
        }
        payload_json = json.dumps(payload, sort_keys=True)
        sig = hmac.new(self._secret, payload_json.encode(), hashlib.sha256).hexdigest()

        import base64

        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()
        token_str = f"{payload_b64}.{sig}"

        token = AUAToken(
            token_id=token_id,
            scopes=scopes,
            expires_at=expires_at,
            created_at=created_at,
            label=label,
        )

        # Persist to state store
        if self._store:
            try:
                self._store.set(
                    self.TOKEN_TABLE,
                    token_id,
                    {**token.as_dict(), "revoked": False},
                )
            except Exception as e:
                log.warning("Could not persist token to state store: %s", e)

        return token, token_str

    def verify(self, token_str: str) -> AUAToken:
        """
        Verify a token string and return the AUAToken.

        Raises TokenError if invalid, expired, or revoked.
        """
        import base64

        try:
            payload_b64, sig = token_str.rsplit(".", 1)
            payload_json = base64.urlsafe_b64decode(payload_b64 + "==").decode()
        except Exception:
            raise TokenError("Malformed token")

        # Verify signature
        expected_sig = hmac.new(self._secret, payload_json.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            raise TokenError("Invalid token signature")

        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            raise TokenError("Malformed token payload")

        token = AUAToken(
            token_id=payload["token_id"],
            scopes=payload["scopes"],
            expires_at=payload["expires_at"],
            created_at=payload["created_at"],
            label=payload.get("label", ""),
        )

        if token.is_expired:
            raise TokenError(
                f"Token expired at {datetime.fromtimestamp(token.expires_at, timezone.utc).isoformat()}"
            )

        # Check revocation
        if self._store:
            try:
                record = self._store.get(self.TOKEN_TABLE, token.token_id)
                if record and record.get("revoked"):
                    raise TokenError("Token has been revoked")
            except TokenError:
                raise
            except Exception as e:
                log.warning("Could not check token revocation: %s", e)

        return token

    def revoke(self, token_id: str) -> bool:
        """Revoke a token by ID. Returns True if found and revoked."""
        if not self._store:
            log.warning("No state store — token revocation not persisted")
            return False
        try:
            record = self._store.get(self.TOKEN_TABLE, token_id)
            if record is None:
                return False
            record["revoked"] = True
            self._store.set(self.TOKEN_TABLE, token_id, record)
            return True
        except Exception as e:
            log.error("Could not revoke token %s: %s", token_id, e)
            return False

    def list_tokens(self, include_revoked: bool = False) -> list[dict[str, Any]]:
        """List all tokens from the state store."""
        if not self._store:
            return []
        try:
            records = self._store.query(self.TOKEN_TABLE, limit=200)
            if not include_revoked:
                records = [r for r in records if not r.get("revoked")]
            return records
        except Exception as e:
            log.error("Could not list tokens: %s", e)
            return []


# Global token manager — set at serve startup
_manager: TokenManager | None = None


def get_token_manager() -> TokenManager:
    global _manager
    if _manager is None:
        _manager = TokenManager.from_config()
    return _manager


def init_token_manager(config: Any) -> TokenManager:
    global _manager
    _manager = TokenManager.from_config(config)
    return _manager
