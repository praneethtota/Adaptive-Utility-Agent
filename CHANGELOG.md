# AUA Framework — Changelog

All notable changes to the AUA Framework are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com). Versioning: [SemVer](https://semver.org).

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
