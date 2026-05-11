# AUA Framework v1 — Deployment Profiles

**Version:** 1.0.0  
**Status:** Canonical. Each profile defines minimum requirements and recommended settings.

---

## Overview

AUA ships with four deployment profiles. Choose the profile that matches your environment, then use `aua init` with the appropriate tier and configure security accordingly.

| Profile | Auth | State | mTLS | Observability | Use case |
|---|---|---|---|---|---|
| Local Developer | Optional | SQLite | No | Optional | Solo dev, experimentation |
| Single GPU Workstation | Recommended | SQLite | No | Optional | Personal GPU server |
| Team Server | Required | Postgres/SQLite | Required | Required | Shared team deployment |
| Enterprise | Required + IAM | Postgres | Required | Required | Production, regulated |

---

## Profile 1 — Local Developer

**Target:** MacBook Pro / laptop. Ollama backend. No GPU required.

```yaml
# aua_config.yaml
aua:
  version: "1.0"
  backend: ollama

security:
  auth_enabled: false   # acceptable for localhost-only

state:
  backend: sqlite
  path: .aua/state/aua.db

logging:
  level: INFO
  format: text           # human-readable for local dev
```

**Setup:**
```bash
brew install ollama
aua init . --tier macbook --preset coding
aua doctor
aua serve
```

**Doctor checks for this profile:**
- Ollama reachable at port 11434
- Required models pulled
- Auth disabled warning (non-fatal on localhost)

**Limitations:**
- Not suitable for network exposure
- Single user only
- No authentication enforced

---

## Profile 2 — Single GPU Workstation

**Target:** RTX 4090 or similar consumer GPU. vLLM backend. Single user or small team on LAN.

```yaml
aua:
  version: "1.0"
  backend: vllm

security:
  auth_enabled: true
  token_secret_env: AUA_TOKEN_SECRET

state:
  backend: sqlite
  path: .aua/state/aua.db

logging:
  level: INFO
  format: json
```

**Setup:**
```bash
export AUA_TOKEN_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
aua init . --tier single-4090 --preset coding
aua token create --scope aua:query --expires 90d --label "primary"
aua doctor --strict
aua serve
```

**Doctor checks for this profile:**
- CUDA available
- VRAM sufficient for configured specialists
- Auth enabled
- Token secret set

---

## Profile 3 — Team Server

**Target:** Dedicated Linux server, RTX 4090 or A100. Shared team access. Prometheus + Grafana monitoring.

```yaml
aua:
  version: "1.0"
  backend: vllm

security:
  auth_enabled: true
  token_secret_env: AUA_TOKEN_SECRET
  mtls:
    enabled: true
    cert_dir: /etc/aua/certs
    auto_generate: false    # use your own CA in production

state:
  backend: sqlite           # or postgres for HA
  path: /var/lib/aua/state/aua.db

logging:
  level: INFO
  format: json
  output: /var/log/aua/router.log

rate_limits:
  aua:query:
    requests_per_minute: 120
  aua:admin:
    requests_per_minute: 10
```

**Setup:**
```bash
# Generate certs (or use your own CA)
aua certs generate --cert-dir /etc/aua/certs

# Create tokens per team member
aua token create --scope aua:query --scope aua:stream --expires 30d --label "team-alice"
aua token create --scope aua:admin --expires 1d --label "ci-deploy"

# Start with observability
docker compose --profile obs up prometheus grafana -d
aua serve
```

**Doctor checks for this profile:**
- Auth enabled (fatal if disabled)
- mTLS certs present and not expired
- Rate limits configured
- Prometheus reachable

---

## Profile 4 — Enterprise

**Target:** Multi-GPU cluster, regulated environment, audit requirements.

```yaml
aua:
  version: "1.0"
  backend: vllm

secrets:
  provider: vault          # or: aws, gcp
  vault_url: https://vault.internal
  token_env: VAULT_TOKEN

security:
  auth_enabled: true
  token_secret_env: AUA_TOKEN_SECRET
  mtls:
    enabled: true
    cert_dir: /etc/aua/certs
    auto_generate: false
  encryption:
    enabled: true
    key_secret: AUA_ENCRYPTION_KEY

state:
  backend: sqlite           # postgres recommended for HA
  path: /var/lib/aua/state/aua.db

logging:
  level: INFO
  format: json
  output: stdout            # forward to ELK/Splunk via log aggregator

rate_limits:
  aua:query:
    requests_per_minute: 300
  aua:admin:
    requests_per_minute: 5

# Disable development features
extensions:
  runtime_import_enabled: false   # never allow runtime plugin loading
  allowlist_only: true
```

**Additional requirements:**
- All secrets via Vault/AWS SM/GCP SM — no plaintext in config
- Encryption at rest enabled (AES-256-GCM)
- Audit log verified via hash chain integrity check
- Extension runtime API disabled
- mTLS between all components
- Prometheus + Grafana + alert routing to PagerDuty/Slack
- Token expiry ≤ 30 days, rotation enforced

**Doctor checks for this profile:**
- All of Profile 3 checks
- Secrets provider reachable
- Encryption key set
- Runtime import disabled
- Audit log hash chain valid

---

## Doctor Profile Validation

Run with `--strict` to enforce profile requirements:

```bash
aua doctor --strict
```

Exit codes:
- `0` — all checks pass
- `1` — one or more checks failed
- `2` — warnings in strict mode (treated as failures)

The doctor automatically detects which profile you're running based on your config and applies the appropriate check set.

---

## Upgrading Between Profiles

Profile 1 → 2: Enable auth, set `AUA_TOKEN_SECRET`, create tokens.  
Profile 2 → 3: Add mTLS, configure rate limits, add observability stack.  
Profile 3 → 4: Add secrets manager, enable encryption, disable runtime imports.

Migration: `aua config migrate --from 0.9 --to 1.0`
