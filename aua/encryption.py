"""
aua/encryption.py — AES-256-GCM encryption at rest for sensitive state.

Encrypts: correction payloads, assertion store entries, DPO pairs,
token metadata, and sensitive audit fields before writing to the state store.

Configuration:
    security:
      encryption:
        enabled: false              # true in production
        key_secret: AUA_ENCRYPTION_KEY   # 32-byte hex key in env

Key management:
    Generate a key: python3 -c "import os; print(os.urandom(32).hex())"
    Store in env:   export AUA_ENCRYPTION_KEY=<hex>

Usage:
    from aua.encryption import get_encryptor
    enc = get_encryptor()
    ciphertext = enc.encrypt("sensitive text")
    plaintext = enc.decrypt(ciphertext)
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# Sentinel for "not encrypted"
_PLAINTEXT_PREFIX = "plain:"
_ENCRYPTED_PREFIX = "enc:v1:"


class Encryptor:
    """
    AES-256-GCM encryptor for sensitive state values.

    If the cryptography package is not installed or encryption is disabled,
    operates in pass-through mode (values stored as-is with a plaintext prefix).
    """

    def __init__(self, key_hex: str | None = None, enabled: bool = True) -> None:
        self._enabled = enabled and key_hex is not None
        self._key: bytes | None = None

        if self._enabled and key_hex:
            try:
                self._key = bytes.fromhex(key_hex)
                if len(self._key) != 32:
                    log.warning(
                        "AUA_ENCRYPTION_KEY must be 32 bytes hex (64 chars) — disabling encryption"
                    )
                    self._enabled = False
                    self._key = None
            except ValueError:
                log.warning("AUA_ENCRYPTION_KEY is not valid hex — disabling encryption")
                self._enabled = False

        if not self._enabled:
            log.debug("Encryption at rest: disabled (pass-through mode)")

    @classmethod
    def from_config(cls, config: Any | None = None) -> Encryptor:
        if config is None:
            return cls(enabled=False)

        sec_cfg = getattr(config, "security", None)
        if sec_cfg is None:
            return cls(enabled=False)

        enc_cfg = getattr(sec_cfg, "encryption", None)
        if enc_cfg is None:
            return cls(enabled=False)

        enabled = getattr(enc_cfg, "enabled", False)
        if not enabled:
            return cls(enabled=False)

        key_secret = getattr(enc_cfg, "key_secret", "AUA_ENCRYPTION_KEY")
        key_hex = os.environ.get(key_secret)
        if not key_hex:
            log.warning("Encryption enabled but %s not set — disabling", key_secret)
            return cls(enabled=False)

        return cls(key_hex=key_hex, enabled=True)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string value. Returns a prefixed ciphertext string."""
        if not self._enabled or self._key is None:
            return _PLAINTEXT_PREFIX + plaintext

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = os.urandom(12)
            aesgcm = AESGCM(self._key)
            ciphertext = aesgcm.encrypt(nonce, plaintext.encode(), None)
            encoded = base64.b64encode(nonce + ciphertext).decode()
            return _ENCRYPTED_PREFIX + encoded
        except ImportError:
            log.warning("cryptography not installed — storing value as plaintext")
            return _PLAINTEXT_PREFIX + plaintext
        except Exception as e:
            log.error("Encryption failed: %s — storing plaintext", e)
            return _PLAINTEXT_PREFIX + plaintext

    def decrypt(self, value: str) -> str:
        """Decrypt a previously encrypted value."""
        if value.startswith(_PLAINTEXT_PREFIX):
            return value[len(_PLAINTEXT_PREFIX) :]

        if not value.startswith(_ENCRYPTED_PREFIX):
            return value  # Legacy unencrypted value

        if not self._enabled or self._key is None:
            log.warning("Encrypted value found but encryption is disabled — cannot decrypt")
            return value

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            raw = base64.b64decode(value[len(_ENCRYPTED_PREFIX) :])
            nonce, ciphertext = raw[:12], raw[12:]
            aesgcm = AESGCM(self._key)
            return aesgcm.decrypt(nonce, ciphertext, None).decode()
        except Exception as e:
            log.error("Decryption failed: %s", e)
            return value

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Global encryptor — set at serve startup
_encryptor: Encryptor | None = None


def get_encryptor() -> Encryptor:
    global _encryptor
    if _encryptor is None:
        _encryptor = Encryptor(enabled=False)
    return _encryptor


def init_encryptor(config: Any) -> Encryptor:
    global _encryptor
    _encryptor = Encryptor.from_config(config)
    return _encryptor
