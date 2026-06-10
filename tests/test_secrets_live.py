"""
tests/test_secrets_live.py — #19: Vault + AWS Secrets Manager live
integration tests.

These exercise the REAL provider clients over their real wire protocols —
no monkeypatching of aua.secrets internals:

  Vault — a wire-faithful KV v2 HTTP server (FastAPI) implementing
          GET /v1/secret/data/{path} with X-Vault-Token auth; the real
          ``hvac`` client talks to it over real HTTP on localhost.

  AWS   — ``moto`` intercepts the real ``boto3`` Secrets Manager client at
          the API level (create_secret / get_secret_value round-trip).

Both suites skip cleanly when the optional dependency is missing, and the
dependencies ship in the ``dev`` extra so CI runs them on every push.
"""

from __future__ import annotations

import threading

import pytest
from fastapi import FastAPI, Request, Response

from aua.secrets import SecretNotFoundError, SecretsManager, resolve_secret

hvac = pytest.importorskip("hvac", reason="#19 Vault tests require hvac")
boto3 = pytest.importorskip("boto3", reason="#19 AWS tests require boto3")
moto = pytest.importorskip("moto", reason="#19 AWS tests require moto")

VAULT_PORT = 18299
VAULT_URL = f"http://127.0.0.1:{VAULT_PORT}"
VAULT_TOKEN = "test-root-token"


# ── Wire-faithful Vault KV v2 server ─────────────────────────────────────────


class _VaultState:
    """Secrets + request counter for the fake Vault server."""

    def __init__(self) -> None:
        self.secrets: dict[str, dict] = {}
        self.read_count = 0


def _make_vault_app(state: _VaultState):
    """KV v2 read endpoint exactly as Vault serves it (incl. auth errors)."""
    # NB: Request/Response must be importable at module level — with
    # `from __future__ import annotations`, FastAPI resolves the string
    # annotations via module globals and silently degrades closure-local
    # imports into required query parameters.
    app = FastAPI()

    @app.get("/v1/secret/data/{path:path}")
    async def read_secret(path: str, request: Request, response: Response):
        token = request.headers.get("X-Vault-Token")
        if token != VAULT_TOKEN:
            response.status_code = 403
            return {"errors": ["permission denied"]}
        state.read_count += 1
        if path not in state.secrets:
            response.status_code = 404
            return {"errors": []}
        return {
            "request_id": "req-1",
            "data": {
                "data": state.secrets[path],
                "metadata": {"version": 1, "destroyed": False},
            },
        }

    return app


@pytest.fixture(scope="module")
def vault_server():
    """Run the Vault wire API on localhost; yield its mutable state."""
    import uvicorn

    state = _VaultState()
    config = uvicorn.Config(
        _make_vault_app(state), host="127.0.0.1", port=VAULT_PORT, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import time

    deadline = time.time() + 5
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "fake Vault server failed to start"
    yield state
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def vault_env(monkeypatch):
    monkeypatch.setenv("VAULT_TOKEN", VAULT_TOKEN)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ── Vault integration ─────────────────────────────────────────────────────────


def test_hvac_client_reads_kv2_secret_directly(vault_server, vault_env):
    """Sanity: the real hvac client speaks to the wire API."""
    vault_server.secrets["sanity"] = {"value": "s3cret"}
    client = hvac.Client(url=VAULT_URL, token=VAULT_TOKEN)
    resp = client.secrets.kv.v2.read_secret_version(path="sanity")
    assert resp["data"]["data"]["value"] == "s3cret"


def test_vault_provider_resolves_value_key(vault_server, vault_env):
    vault_server.secrets["OPENAI_API_KEY"] = {"value": "sk-vault-123"}
    mgr = SecretsManager(provider="vault", url=VAULT_URL, token_env="VAULT_TOKEN")
    assert mgr.get("OPENAI_API_KEY") == "sk-vault-123"
    assert mgr.provider == "vault"


def test_vault_provider_falls_back_to_name_key(vault_server, vault_env):
    vault_server.secrets["DB_PASSWORD"] = {"DB_PASSWORD": "pg-pass"}
    mgr = SecretsManager(provider="vault", url=VAULT_URL, token_env="VAULT_TOKEN")
    assert mgr.get("DB_PASSWORD") == "pg-pass"


def test_vault_env_var_takes_precedence(vault_server, vault_env, monkeypatch):
    """Env is always tried first — fastest and always available."""
    vault_server.secrets["SHARED_KEY"] = {"value": "from-vault"}
    monkeypatch.setenv("SHARED_KEY", "from-env")
    mgr = SecretsManager(provider="vault", url=VAULT_URL, token_env="VAULT_TOKEN")
    assert mgr.get("SHARED_KEY") == "from-env"


def test_vault_missing_secret_raises_when_required(vault_server, vault_env):
    mgr = SecretsManager(provider="vault", url=VAULT_URL, token_env="VAULT_TOKEN")
    with pytest.raises(SecretNotFoundError) as exc:
        mgr.get("NOPE_NOT_THERE")
    assert exc.value.provider == "vault"
    assert mgr.get("NOPE_NOT_THERE", required=False) is None


def test_vault_bad_token_resolves_none_not_crash(vault_server, monkeypatch):
    """403 from Vault degrades to not-found, never an unhandled exception."""
    monkeypatch.setenv("VAULT_TOKEN", "wrong-token")
    monkeypatch.delenv("LOCKED_KEY", raising=False)
    vault_server.secrets["LOCKED_KEY"] = {"value": "nope"}
    mgr = SecretsManager(provider="vault", url=VAULT_URL, token_env="VAULT_TOKEN")
    assert mgr.get("LOCKED_KEY", required=False) is None


def test_vault_resolution_is_cached(vault_server, vault_env):
    """Second get() serves from cache — no extra Vault round-trip."""
    vault_server.secrets["CACHED_KEY"] = {"value": "v1"}
    mgr = SecretsManager(provider="vault", url=VAULT_URL, token_env="VAULT_TOKEN")
    before = vault_server.read_count
    assert mgr.get("CACHED_KEY") == "v1"
    assert mgr.get("CACHED_KEY") == "v1"
    assert vault_server.read_count == before + 1


def test_resolve_secret_inline_refs_via_vault(vault_server, vault_env):
    vault_server.secrets["API_TOKEN"] = {"value": "tok-789"}
    mgr = SecretsManager(provider="vault", url=VAULT_URL, token_env="VAULT_TOKEN")
    assert resolve_secret("Bearer ${API_TOKEN}", mgr) == "Bearer tok-789"
    # Unresolvable refs stay literal (redaction-safe behavior)
    assert resolve_secret("Bearer ${MISSING_REF}", mgr) == "Bearer ${MISSING_REF}"


# ── AWS Secrets Manager integration (moto) ────────────────────────────────────


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@moto.mock_aws
def test_aws_provider_resolves_secret_string(aws_env):
    sm = boto3.client("secretsmanager", region_name="us-east-1")
    sm.create_secret(Name="OPENAI_API_KEY", SecretString="sk-aws-456")
    mgr = SecretsManager(provider="aws", region="us-east-1")
    assert mgr.get("OPENAI_API_KEY") == "sk-aws-456"


@moto.mock_aws
def test_aws_missing_secret_raises_when_required(aws_env):
    mgr = SecretsManager(provider="aws", region="us-east-1")
    with pytest.raises(SecretNotFoundError) as exc:
        mgr.get("DOES_NOT_EXIST")
    assert exc.value.provider == "aws"
    assert mgr.get("DOES_NOT_EXIST", required=False) is None


@moto.mock_aws
def test_aws_env_var_takes_precedence(aws_env, monkeypatch):
    sm = boto3.client("secretsmanager", region_name="us-east-1")
    sm.create_secret(Name="SHARED_AWS_KEY", SecretString="from-aws")
    monkeypatch.setenv("SHARED_AWS_KEY", "from-env")
    mgr = SecretsManager(provider="aws", region="us-east-1")
    assert mgr.get("SHARED_AWS_KEY") == "from-env"


@moto.mock_aws
def test_aws_binary_secret_resolves_none(aws_env):
    """SecretBinary has no SecretString — resolves to not-found, no crash."""
    sm = boto3.client("secretsmanager", region_name="us-east-1")
    sm.create_secret(Name="BIN_KEY", SecretBinary=b"\x00\x01")
    mgr = SecretsManager(provider="aws", region="us-east-1")
    assert mgr.get("BIN_KEY", required=False) is None


@moto.mock_aws
def test_aws_region_scoping(aws_env):
    """Secret created in eu-west-1 is invisible to a us-east-1 manager."""
    sm_eu = boto3.client("secretsmanager", region_name="eu-west-1")
    sm_eu.create_secret(Name="EU_ONLY_KEY", SecretString="eu-value")
    mgr_us = SecretsManager(provider="aws", region="us-east-1")
    assert mgr_us.get("EU_ONLY_KEY", required=False) is None
    mgr_eu = SecretsManager(provider="aws", region="eu-west-1")
    assert mgr_eu.get("EU_ONLY_KEY") == "eu-value"


@moto.mock_aws
def test_aws_resolution_cached_after_secret_deleted(aws_env):
    """Cache survives upstream deletion within a process lifetime."""
    sm = boto3.client("secretsmanager", region_name="us-east-1")
    sm.create_secret(Name="EPHEMERAL", SecretString="v1")
    mgr = SecretsManager(provider="aws", region="us-east-1")
    assert mgr.get("EPHEMERAL") == "v1"
    sm.delete_secret(SecretId="EPHEMERAL", ForceDeleteWithoutRecovery=True)
    assert mgr.get("EPHEMERAL") == "v1"  # served from cache


# ── Cross-provider: config redaction guarantee ────────────────────────────────


@moto.mock_aws
def test_resolved_secret_never_in_repr(aws_env):
    sm = boto3.client("secretsmanager", region_name="us-east-1")
    sm.create_secret(Name="SENSITIVE", SecretString="hunter2")
    mgr = SecretsManager(provider="aws", region="us-east-1")
    mgr.get("SENSITIVE")
    assert "hunter2" not in repr(mgr)
    assert "hunter2" not in str(mgr)
