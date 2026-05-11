# AUA Framework v1 — Architecture Specification

**Version:** 0.8.0b0  
**Status:** Canonical. Implementation must match this document. Divergence = a bug.

---

## 1. System Overview

AUA (Adaptive Utility Agents) is a multi-specialist LLM routing framework. It routes queries to domain-expert models, scores outputs using a utility function, detects contradictions, resolves them with an arbiter, and feeds verified corrections back into training.

The design goal is **Django for adaptive multi-model LLM systems** — batteries included, deeply configurable, extensible without editing framework internals.

---

## 2. Component Boundaries

```
┌─────────────────────────────────────────────────────┐
│                    AUA Router                        │
│                                                      │
│  ┌──────────┐  ┌───────────┐  ┌─────────────────┐   │
│  │Middleware │  │  Session  │  │  Correction     │   │
│  │ Pipeline │  │  Manager  │  │  Retrieval      │   │
│  └──────────┘  └───────────┘  └─────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │          Field Classifier                    │    │
│  │  (pluggable via FieldClassifierPlugin)       │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────┐  ┌───────────┐  ┌─────────────────┐   │
│  │  Router  │  │ Specialist│  │ Utility Scorer  │   │
│  │ Decision │  │  Calls    │  │ (pluggable)     │   │
│  └──────────┘  └───────────┘  └─────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │              Arbiter Agent                   │    │
│  │  (pluggable policy via ArbiterPolicyPlugin)  │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌──────────┐  ┌───────────┐  ┌─────────────────┐   │
│  │   Hook   │  │Correction │  │  State Store    │   │
│  │ Registry │  │  Logger   │  │  (pluggable)    │   │
│  └──────────┘  └───────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────┘

External:
  Specialist servers  (vLLM / Ollama / custom ModelBackendPlugin)
  Arbiter server      (same backends)
  State store         (files / SQLite / Postgres)
  Observability       (stdout / Prometheus / OTEL)
```

---

## 3. Full Request Lifecycle

Every query follows this pipeline in order. Steps marked `[pluggable]` can be replaced or extended via plugins/hooks.

```
 1. HTTP Request arrives at router
    └─ session_id / trace_id / request_id assigned (UUID if not supplied)

 2. Middleware pipeline — before_query() [pluggable]
    └─ PII redaction, tenant policy, rate limiting, auth check

 3. Session lookup
    └─ Retrieve prior session context from state store (if session_id known)

 4. Correction retrieval
    └─ Load relevant verified claims from AssertionsStore for this domain

 5. Field Classifier [pluggable]
    └─ Scores query against all known fields → domain_distribution dict
    └─ Emits: primary_domain, domain_distribution, routing_mode decision

 6. Routing decision
    ├─ single: one field above single_domain_threshold → one specialist
    ├─ fanout: multiple fields above fanout_threshold → multiple specialists
    └─ force_domain: override from request

 7. Specialist calls [pluggable via ModelBackendPlugin]
    └─ POST to specialist endpoint with correction context injected
    └─ Timeout: specialist_timeout (default 60s) → AUA_SPECIALIST_TIMEOUT

 8. Utility Scoring [pluggable]
    └─ U = w_e·E + w_c·C + w_k·K per specialist response
    └─ Kalman filter updates confidence estimate

 9. Arbiter [pluggable policy]
    └─ Runs if: fanout + contradiction detected
    └─ 4 checks: logical, mathematical, cross-session, empirical
    └─ Issues Case 1/2/3/4 verdict → correction signal

10. Hook registry — on_correction / on_promotion / etc. [pluggable]
    └─ Fire registered hooks for this event type

11. Correction logging
    └─ Store DPO pair to state store (if arbiter issued correction)
    └─ Update AssertionsStore with verified claim

12. Response assembly
    └─ RouterResponse model with session_id, u_score, routing_mode, response

13. Middleware pipeline — after_response() [pluggable]
    └─ Response transformation, audit logging

14. Metrics / Logs / Traces / Audit
    └─ Structured JSON to stdout
    └─ Prometheus metrics (if observability profile enabled)
    └─ OTEL traces (if otel extra installed)
    └─ Audit log entry written to state store (append-only, hash chain)
```

---

## 4. Component Ownership

| Component | Module | Owner interface |
|---|---|---|
| Field classifier | `aua.field_classifier` | `FieldClassifierPlugin` |
| Utility scorer | `aua.utility_scorer` | `UtilityScorerPlugin` |
| Arbiter policy | `aua.arbiter` | `ArbiterPolicyPlugin` |
| Promotion policy | `aua.blue_green` | `PromotionPolicyPlugin` |
| Correction store | `aua.assertions_store` | `CorrectionStorePlugin` |
| Model backend | `aua.router` (http calls) | `ModelBackendPlugin` |
| State store | `aua.state` | `StateStorePlugin` |
| Hooks | `aua.hooks` | `HookPlugin` |
| Middleware | `aua.middleware` | `AUAMiddleware` |

All plugin types are defined in `aua/plugins/interfaces.py` as Python `Protocol` classes.

---

## 5. Plugin Loading Lifecycle

```
1. Config loaded (load_config)
2. For each plugin reference in config:
   a. Resolve import_path: "module.path:ClassName"
   b. Import module
   c. Instantiate class with config dict injected
   d. Validate against Protocol (runtime isinstance check)
   e. Register in plugin registry
3. Router initialised with plugin registry
4. On SIGHUP: reload config → re-run steps 1-5 atomically
```

Plugins are validated at startup. A failed plugin load causes startup to abort with `AUA_PLUGIN_LOAD_FAILED`.

---

## 6. Hook Execution Order

For each hook point, hooks fire in YAML registration order:

```
pre_query → [middleware.before_query] → post_route → pre_specialist_call
→ post_specialist_call → pre_arbiter → post_arbiter → on_correction
→ pre_response → [middleware.after_response] → post_response
→ on_promotion / on_rollback (async, not in request path)
```

Hook failures default to **fail-open** (log + continue). Set `hooks.{name}.fail_closed: true` to abort on failure.

---

## 7. Observability Flow

```
Every request → structured JSON log line (stdout)
             → Prometheus counter/histogram increment (if enabled)
             → OTEL span (if aua[otel] installed)
             → Audit log entry (state store, append-only)

Key metrics:
  aua_queries_total{domain, routing_mode, status}
  aua_query_latency_seconds{domain, routing_mode}
  aua_utility_score{domain}
  aua_contradiction_rate{domain}
  aua_arbiter_verdict_total{case}
  aua_specialist_errors_total{specialist, error_code}
```

---

## 8. Security Boundary

- Only the router port (default 8000) is public-facing.
- Specialist ports are internal — bind to `127.0.0.1` or Docker internal network.
- Extension endpoints (`/extensions/*`) are disabled in production mode.
- All external endpoints require bearer token auth (v0.9+).
- Secrets are never logged, traced, or returned via `GET /config`.
- Audit log is append-only with a hash chain for tamper detection (v0.9+).

---

## 9. State Store

All persistent state goes through the `StateStore` interface:

| Data | v0.7 location | v0.8+ (default) |
|---|---|---|
| Promotion log | `.aua/state/promotions.jsonl` | SQLite: `promotions` table |
| Correction pairs | `dpo_pairs/*.jsonl` | SQLite: `corrections` table |
| Assertions | In-memory (AssertionsStore) | SQLite: `assertions` table |
| Sessions | None | SQLite: `sessions` table |
| Audit log | None | SQLite: `audit_log` table |

Migration from v0.7 flat files: `aua config migrate --from 0.7 --to 0.8`

---

## 10. Extension Points Summary

Users extend AUA by adding YAML entries — never by editing framework source files.

```yaml
# Custom utility scorer
utility_scorer:
  import_path: plugins.custom_utility:RiskWeightedUtilityScorer
  config:
    risk_weight: 0.7

# Custom middleware
middleware:
  - import_path: plugins.middleware:PIIRedactionMiddleware
  - import_path: plugins.middleware:AuditMiddleware

# Custom hook
hooks:
  on_correction:
    - import_path: plugins.hooks:SlackNotificationHook
      config:
        webhook_url_secret: SLACK_WEBHOOK_URL

# Custom backend
backends:
  my_gateway:
    import_path: plugins.backends:GatewayBackend
    base_url: https://gateway.internal
    auth_secret: GATEWAY_API_KEY
```

---

*Document maintained by: Praneeth Tota. Last updated: v0.8.0b0. For implementation questions, check this document first.*
