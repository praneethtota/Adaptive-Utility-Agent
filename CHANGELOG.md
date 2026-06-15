# AUA Framework — Changelog

All notable changes to the AUA Framework are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com). Versioning: [SemVer](https://semver.org).

---

## [1.2.0] — 2026-06-15

Resilience, security, the extended plugin system, and the operations toolkit
(#37–#55 block), plus a full pre-release audit.

### Added
- **Retry with exponential backoff (#39)**: per-specialist transport-level
  retry on `ConnectError`/`ReadTimeout`/429/502/503/504 with configurable
  `base_delay_ms`, `max_delay_ms`, `±25%` jitter, and `retryable_status_codes`.
  Non-transient codes (400/500) are never retried. `max_retries: 0` disables it.
- **Circuit breaker per specialist (#37)**: CLOSED/OPEN/HALF_OPEN state machine
  with sliding failure window, automatic HALF_OPEN probe after
  `recovery_timeout_s`, and per-specialist status at `GET /health/ready`.
- **Degraded-mode failover (#38)**: open circuits are excluded from routing;
  responses carry `degraded_mode` and `degraded_specialists` so callers can
  detect partial availability.
- **Bearer token auth wiring**: `security.auth_enabled` activates HMAC-SHA256
  token verification middleware (15 scopes, revocation). Public endpoints
  (health/docs/version) pass through; zero overhead when disabled.
- **mTLS wiring**: `security.mtls.key_file`/`cert_file`/`ca_file` are passed to
  uvicorn; presence of `ca_file` requires client certs (mutual TLS).
- **Extended plugin system (#51)**: four new Protocol interfaces —
  `ContradictionDetectorPlugin`, `AssertionStorePlugin`,
  `RoutingStrategyPlugin`, `ScoringComponentPlugin`.
- **Custom utility function (#53)**: `FullUtilityScorerPlugin.score_full()`
  bypasses the linear form (Axiom A5) for quadratic, multiplicative,
  Cobb-Douglas, Rawlsian-min, and threshold-gate models.
- **Extended middleware (#52)**: `on_chunk` (SSE token interception),
  `before_batch`/`after_batch`, and `on_error` hooks on `AUAMiddleware`.
- **Compatibility matrix (#55)**: model-format × hardware × backend matrix in
  `aua/compat.py`; `aua doctor` check group 6 and `aua doctor --compat-matrix`.
- **Operations toolkit**: `aua test` built-in suites (#54), `aua loadtest`
  (#50), persistent batch queue (#56), model registry + version pinning (#46),
  experiment tracking via MLflow/W&B (#47), shadow mode (#48), regression gate
  (#49), multi-tenancy isolation (#44).
- **ArbiterAgent live pipeline**: the four-check arbitration (logical,
  mathematical, cross-session, empirical via SymPy/arXiv/PubMed) is now the
  default; a simplified LLM-only path is available via `arbitration_mode: "llm"`.
- **tau softmax routing** and **T_min promotion gate** wired into the router.
- Hardware tiers `gaming-pc` and `h100-cluster` added.

### Fixed
- **Version source of truth** corrected to 1.2.0 (was 1.1.0).
- `arbitration_mode` is now validated at config load time (was accepted
  silently; only the runtime PATCH endpoint validated).
- Retry and circuit-breaker numeric fields are validated at load time
  (`max_retries >= 0`, `max_delay_ms >= base_delay_ms`, thresholds `>= 1`).
- `infer_model_format()` is backend-aware: Ollama/llama.cpp tags without a
  suffix resolve to GGUF (previously every Ollama user — including the default
  `aua init --tier macbook` scaffold — saw "model format unknown" warnings).
- Normalised the `aua.version` schema field across tier templates.
- Packaging: explicit `aua/templates/prompts/*.txt` include; removed stale
  committed build artifacts.

---

## [1.1.0] — 2026-06-10

The AUA-Veritas production backport plus the completed expert path.

### Added
- **Persistence & search (V-P1.1)**: message-level keyword search with async
  background indexing, startup backfill, and DB fallback (`GET /search`,
  `POST /conversations/{id}/messages`)
- **Context backups (V-P1.2/1.4)**: 6-section structured handoff notes,
  token/message/time-gap triggers, 6-hour coverage job
  (`POST /context/backup/run-coverage-job`)
- **Correction lifecycle (V-P1.3/2.1/2.4)**: explicit `correction:` prefix,
  implicit detection with Accept/Reject (`POST /corrections/confirm-implicit`),
  CRUD + evidence history (`PATCH/DELETE /corrections/{id}`,
  `GET /corrections/evidence`), arbiter findings surfaced as `review_notes`
- **Self-maintenance (V-P1.5/1.6/2.3/3.1)**: crash sentinel + auto-reporting,
  remote model config with remote→cache→builtin fallback, update management
  (`GET /version/check`, `POST /update/skip`), structured bug reports
  (`POST /bug-report`)
- **Analytics suite (V-P2.2)**: `GET /analytics`, `/reliability`, `/usage`,
  `/pricing`
- **Projects & local models (V-P3.2/3.3)**: conversation grouping,
  Ollama-class model registration and specialist tagging
- **Dynamic domain ontology (V-P3.4)**: 10 L0 roots, alias map + edit-distance
  resolution, 4-gate candidate promotion, hourly maintenance job
  (`GET /domain-tree`)
- **Session IDs (#15)**: session/trace/request IDs on every request —
  client-supplied honored, UUIDs generated, returned as headers on every
  response, propagated to specialists/hooks/audit/logs
- **Secrets (#19)**: `secrets:` config block (env|vault|aws|gcp) and live
  Vault + AWS Secrets Manager integration tests in CI
- **YAML extension wiring (F-09/F-10/F-11)**: `plugins:`, `hooks:`,
  `middleware:`, `state:`, and `security:` config blocks now parse with
  strict validation and wire at startup; `GET /extensions` reports what the
  running server loaded
- Tutorial: Concepts section, bring-your-own-model walkthrough, complete
  config reference, troubleshooting guide, How-to 18 (production ops)

### Fixed
- Audit log writes failed silently (missing `request_id`/`routing_mode`
  columns)
- `POST /projects` failed on an injected `id` column
- Keyword extraction dropped years/numbers (dead code path)
- Crash reporter could self-report the current session
- Hook YAML format and 9 plugin constructor examples in the tutorial matched
  a contract the loader never had

---

## [1.0.0] — 2026-05-11

First public stable release.

### Added
- **v0.6-alpha (P-01–P-12):** Production-hardened core — packaging, CI, config strictness, serve lifecycle, rollback, test suite (132 tests)
- **v0.7-beta (#11–#14, #05A–D):** Docker stack (Dockerfile + docker-compose, 4 profiles), hardware tier templates (macbook/single-4090/quad-4090/a100-cluster), model/field/preset registries, `aua config/models/fields/presets` CLI, hot reload (SIGHUP), tutorial v0.7
- **v0.8-framework-beta (F-01–F-17):** Architecture spec, stable Plugin Protocol interfaces (8 types), AUA_* error taxonomy (17 codes), permission/scope matrix (14 scopes), SQLite state store with WAL + audit hash chain, config versioning + migration, plugin registry + import system, hook system (11 points), middleware pipeline, extension CLI (`aua extensions`), prompt templates, safety/abstention policy, defaults registry, example projects
- **v0.9-rc1 (#15–#32):** Session IDs (session_id/trace_id/request_id), secrets management (env/Vault/AWS/GCP), bearer token auth (HMAC-SHA256, 14 scopes), token CLI, mTLS cert generation, Prometheus metrics (16 metrics + GET /metrics), cost tracking (GET /metrics/cost), structured JSON logging, rate limiting (429 + Retry-After), audit log wiring, webhook events (10 event types), encryption at rest (AES-256-GCM), OTEL instrumentation, Grafana dashboard (20 panels), Datadog preset, alert rules (8 rules)
- **v0.9-rc2 (E-01–E-03, U-01–U-02):** Evaluation harness (`aua eval run/report/compare`), DPO/corrections export, 6 smoke eval datasets, Chat Session API (5 endpoints, persistent sessions), Chat UI (Next.js 14, 3-zone layout, Framework Debugger, AUA Controls drawer, username/password auth via NextAuth)
- **v1.0 (D-01–D-04):** Deployment profiles doc (4 profiles), compatibility matrix, release engineering docs, fresh-clone validation

### Framework metrics (v1.0)
- 132 tests passing
- 10 CLI command groups, 40+ subcommands
- 20 REST API endpoints
- 8 plugin Protocol interfaces
- 14 auth scopes
- 16 Prometheus metrics
- 20 Grafana dashboard panels
- 6 smoke eval datasets

---

## [0.9.0rc2] — 2026-05-11

### Added
- Evaluation harness (`aua eval run/report/compare`)
- DPO/corrections export CLI
- 6 built-in smoke eval datasets
- Chat Session API (POST/GET /sessions, messages)
- Chat UI — Next.js 14, Framework Debugger, AUA Controls, NextAuth credentials auth

---

## [0.9.0rc1] — 2026-05-11

### Added
- Session IDs (session_id, trace_id, request_id) propagated end-to-end
- Secrets management (env/Vault/AWS SM/GCP SM)
- Bearer token authentication (HMAC-SHA256, 14 scopes)
- Token CLI (`aua token create/list/revoke/inspect`)
- mTLS cert management (`aua certs generate/inspect`)
- Prometheus metrics endpoint (GET /metrics, 16 metrics)
- Cost tracking (GET /metrics/cost)
- Structured JSON logging with session ID auto-injection
- Rate limiting (per-scope sliding window, 429 + Retry-After)
- Audit log wiring into router handlers
- Webhook events (10 event types, retry with backoff)
- Encryption at rest (AES-256-GCM)
- OTEL instrumentation (optional aua[otel])
- Grafana dashboard (20 panels, pre-built JSON)
- Datadog OTEL collector preset
- 8 Prometheus alert rules

---

## [0.8.0b0] — 2026-05-11

### Added
- Architecture specification (AUA_Framework_v1_Architecture.md)
- 8 stable Plugin Protocol interfaces
- 17 AUA_* error codes with HTTP/CLI mappings
- 14-scope permission matrix
- SQLite state store (WAL, hash chain)
- Config versioning + migration CLI
- Plugin registry and extension import system
- Hook system (11 lifecycle hook points)
- Middleware pipeline (PIIRedaction, Audit, TenantPolicy)
- Extension CLI (`aua extensions test/list/inspect`)
- Prompt template system (versioned, field-specific)
- Safety/abstention policy for high-risk fields
- Defaults registry (`aua defaults show`)
- Example projects (quickstart, custom utility, custom middleware)

---

## [0.7.0b0] — 2026-05-11

### Added
- Dockerfile + docker-compose (4 profiles: ollama, gpu, obs, secure)
- Hardware tier templates (macbook, single-4090, quad-4090, a100-cluster)
- Model registry with aliases
- Field registry with utility weights
- Preset system (6 built-in presets)
- `aua config validate/expand/reload` CLI
- Hot reload (SIGHUP) for routing thresholds and config
- Tutorial v0.7 (12-part, Django-style)

### Fixed
- Ollama specialist endpoint: use /v1/chat/completions (OpenAI-compat) not /api/chat
- Port conflict detection: skip Ollama ports (all specialists share 11434)

---

## [0.6.0a0] — 2026-05-11

### Added
- Production-complete core (P-01–P-12)
- CI workflow (Python 3.10/3.11/3.12 matrix)
- Config strictness (unknown key validation, duplicate port detection)
- Serve lifecycle hardening (SIGTERM handler, readiness polling)
- API contract hardening (session_id, ErrorResponse, /version)
- Rollback with atomic state writes and file locking
- 132-test suite (fakes, fixtures, contract tests)

---

## [0.5.0] — 2026-03-01

Initial POC release. Core routing, utility scoring, arbitration, correction loop, blue-green deployment.
