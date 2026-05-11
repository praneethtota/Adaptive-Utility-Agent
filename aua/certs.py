"""
aua/certs.py — mTLS certificate management for AUA Framework.

Generates and manages TLS certificates for encrypted communication
between the router, specialists, and arbiter.

Configuration:
    security:
      mtls:
        enabled: false          # true in production
        cert_dir: .aua/certs
        auto_generate: true     # generate dev certs automatically
        ca_cert: ca.pem         # bring your own CA in production

CLI:
    aua certs generate          # generate dev CA + server/client certs
    aua certs rotate            # rotate existing certs
    aua certs inspect           # show cert expiry dates
    aua doctor --check-certs    # verify certs are valid and not expired

Note: Auto-generated certs are for development only.
      In production, use your own CA and signed certificates.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_CERT_DIR = ".aua/certs"
CERT_VALIDITY_DAYS = 365


def generate_dev_certs(cert_dir: str | os.PathLike = DEFAULT_CERT_DIR) -> dict[str, Path]:
    """
    Generate a self-signed CA and server/client certificate pair for development.

    Requires: cryptography >= 3.0 (pip install cryptography)

    Returns:
        dict with paths: ca_cert, ca_key, server_cert, server_key, client_cert, client_key

    Raises:
        ImportError if cryptography is not installed.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        raise ImportError(
            "The 'cryptography' package is required for cert generation.\n"
            "Install it with: pip install cryptography"
        )

    cert_path = Path(cert_dir)
    cert_path.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now(datetime.timezone.utc)
    expiry = now + datetime.timedelta(days=CERT_VALIDITY_DAYS)

    # ── Generate CA ──────────────────────────────────────────────────────────
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AUA Development CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(expiry)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # ── Generate server cert ─────────────────────────────────────────────────
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "aua-server")])
    san = x509.SubjectAlternativeName(
        [
            x509.DNSName("localhost"),
            x509.DNSName("aua-router"),
            x509.DNSName("aua-ollama"),
        ]
    )
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(expiry)
        .add_extension(san, critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    # ── Generate client cert ─────────────────────────────────────────────────
    client_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "aua-client")])
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(client_name)
        .issuer_name(ca_name)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(expiry)
        .sign(ca_key, hashes.SHA256())
    )

    def _write_cert(path: Path, cert: Any) -> None:
        path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        path.chmod(0o644)

    def _write_key(path: Path, key: Any) -> None:
        path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
        path.chmod(0o600)  # owner-only read

    paths = {
        "ca_cert": cert_path / "ca.pem",
        "ca_key": cert_path / "ca-key.pem",
        "server_cert": cert_path / "server.pem",
        "server_key": cert_path / "server-key.pem",
        "client_cert": cert_path / "client.pem",
        "client_key": cert_path / "client-key.pem",
    }

    _write_cert(paths["ca_cert"], ca_cert)
    _write_key(paths["ca_key"], ca_key)
    _write_cert(paths["server_cert"], server_cert)
    _write_key(paths["server_key"], server_key)
    _write_cert(paths["client_cert"], client_cert)
    _write_key(paths["client_key"], client_key)

    return paths


def inspect_certs(cert_dir: str | os.PathLike = DEFAULT_CERT_DIR) -> list[dict[str, Any]]:
    """
    Return expiry info for all certs in cert_dir.

    Returns list of dicts with: file, subject, expires_at, days_remaining, expired
    """
    try:
        from cryptography import x509
    except ImportError:
        return []

    cert_path = Path(cert_dir)
    results = []

    for pem_file in sorted(cert_path.glob("*.pem")):
        if "key" in pem_file.name:
            continue  # skip private keys
        try:
            cert = x509.load_pem_x509_certificate(pem_file.read_bytes())
            expiry = cert.not_valid_after_utc
            now = datetime.datetime.now(datetime.timezone.utc)
            days_remaining = (expiry - now).days
            results.append(
                {
                    "file": str(pem_file),
                    "subject": cert.subject.rfc4514_string(),
                    "expires_at": expiry.isoformat(),
                    "days_remaining": days_remaining,
                    "expired": days_remaining < 0,
                    "expiring_soon": 0 <= days_remaining <= 30,
                }
            )
        except Exception as e:
            results.append({"file": str(pem_file), "error": str(e)})

    return results
