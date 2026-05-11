# AUA Framework v1.0 — Validation Report

**Version:** 1.0.0  
**Date:** 2026-05-11  
**Status:** All validation criteria met. v1.0.0 shipped.

---

## 1. Test Suite — 132 tests, 0 failures

```
pytest -v --tb=short
```

```
test_cli_doctor.py::test_doctor_runs_without_crash                         PASSED
test_cli_doctor.py::test_doctor_config_check_passes                        PASSED
test_cli_doctor.py::test_doctor_config_check_fails_missing_file            PASSED
test_cli_doctor.py::test_doctor_hardware_vllm_on_apple_fails               PASSED
test_cli_doctor.py::test_doctor_hardware_ollama_on_apple_passes            PASSED
test_cli_doctor.py::test_doctor_hardware_nvidia_vllm_passes                PASSED
test_cli_doctor.py::test_doctor_returns_integer                            PASSED
test_cli_doctor.py::test_doctor_json_output                                PASSED
test_cli_doctor.py::test_doctor_strict_exits_2_on_warn                    PASSED
test_cli_init.py::test_init_creates_directory                              PASSED
test_cli_init.py::test_init_creates_expected_files                         PASSED
test_cli_init.py::test_init_gitignore_content                              PASSED
test_cli_init.py::test_init_default_tier_is_single_4090                   PASSED
test_cli_init.py::test_init_macbook_tier                                   PASSED
test_cli_init.py::test_init_force_overwrites                               PASSED
test_cli_init.py::test_init_refuses_overwrite_without_force               PASSED
test_cli_init.py::test_init_existing_dir_is_reused                        PASSED
test_cli_init.py::test_init_all_tiers[macbook]                            PASSED
test_cli_init.py::test_init_all_tiers[single-4090]                        PASSED
test_cli_init.py::test_init_all_tiers[quad-4090]                          PASSED
test_cli_init.py::test_init_all_tiers[a100-cluster]                       PASSED
test_cli_init.py::test_init_all_tiers_generate_valid_config[macbook]      PASSED
test_cli_init.py::test_init_all_tiers_generate_valid_config[single-4090]  PASSED
test_cli_init.py::test_init_all_tiers_generate_valid_config[quad-4090]    PASSED
test_cli_init.py::test_init_all_tiers_generate_valid_config[a100-cluster] PASSED
test_config.py::test_load_minimal_config                                   PASSED
test_config.py::test_specialist_endpoint_url                               PASSED
test_config.py::test_specialist_for_field                                  PASSED
test_config.py::test_vllm_command                                          PASSED
test_config.py::test_blue_green_for                                        PASSED
test_config.py::test_all_endpoints                                         PASSED
test_config.py::test_available_tiers                                       PASSED
test_config.py::test_load_tier[macbook]                                    PASSED
test_config.py::test_load_tier[single-4090]                               PASSED
test_config.py::test_load_tier[quad-4090]                                  PASSED
test_config.py::test_load_tier[a100-cluster]                               PASSED
test_config.py::test_macbook_tier_uses_ollama                              PASSED
test_config.py::test_single_4090_tier_uses_vllm                           PASSED
test_config.py::test_a100_cluster_tier_no_enforce_eager                   PASSED
test_config.py::test_unknown_tier_raises                                   PASSED
test_config.py::test_missing_config_raises                                 PASSED
test_config.py::test_unknown_specialist_raises                             PASSED
test_config.py::test_specialist_endpoint_uses_host                        PASSED
test_config.py::test_specialist_models_url_uses_host                      PASSED
test_config.py::test_arbiter_endpoint_uses_host                           PASSED
test_config.py::test_endpoint_override                                     PASSED
test_config.py::test_custom_scheme                                         PASSED
test_config.py::test_runtime_config_defaults                               PASSED
test_config.py::test_runtime_ensure_creates_dirs                           PASSED
test_config.py::test_router_cors_defaults_to_wildcard                     PASSED
test_config.py::test_duplicate_ports_raises                                PASSED
test_config.py::test_unknown_key_raises                                    PASSED
test_config.py::test_invalid_threshold_raises                              PASSED
test_config.py::test_gpu_memory_utilization_zero_raises                   PASSED
test_config.py::test_tier_aliases_imported                                 PASSED
test_config.py::test_alias_rtx4090_loads_single_4090                     PASSED
test_config.py::test_alias_a100_loads_a100_cluster                        PASSED
test_config.py::test_quad_4090_has_multiple_gpus                          PASSED
test_config.py::test_quad_4090_has_law_specialist                         PASSED
test_config.py::test_single_4090_uses_awq                                 PASSED
test_config.py::test_a100_cluster_uses_fp16                               PASSED
test_config.py::test_unknown_tier_error_mentions_aliases                  PASSED
test_imports.py::test_core_imports                                         PASSED
test_imports.py::test_arbiter_alias                                        PASSED
test_imports.py::test_version_export                                       PASSED
test_imports.py::test_endpoint_models_exported                             PASSED
test_imports.py::test_stream_models_exported                               PASSED
test_imports.py::test_config_submodule                                     PASSED
test_imports.py::test_no_private_imports_required                          PASSED
test_rollback.py::test_record_promotion_creates_log                        PASSED
test_rollback.py::test_load_promotions_empty                               PASSED
test_rollback.py::test_load_promotions_after_record                        PASSED
test_rollback.py::test_rollback_no_history_returns_1                      PASSED
test_rollback.py::test_rollback_success                                    PASSED
test_rollback.py::test_rollback_updates_config                             PASSED
test_rollback.py::test_rollback_marks_promotion_reverted                  PASSED
test_rollback.py::test_rollback_appends_rollback_event                    PASSED
test_rollback.py::test_double_rollback_returns_1                          PASSED
test_rollback.py::test_rollback_all_skips_specialists_with_no_history     PASSED
test_rollback.py::test_rollback_cli_no_restart                            PASSED
test_rollback.py::test_promotions_saved_as_jsonl                          PASSED
test_rollback.py::test_promotion_id_is_uuid                               PASSED
test_rollback.py::test_rollback_dry_run                                    PASSED
test_rollback.py::test_atomic_config_write_no_tmp_left                    PASSED
test_router_api.py::test_health_live                                       PASSED
test_router_api.py::test_health_ready_with_fake_server                    PASSED
test_router_api.py::test_health_startup_after_ready                       PASSED
test_router_api.py::test_health_legacy_endpoint                            PASSED
test_router_api.py::test_version_endpoint                                  PASSED
test_router_api.py::test_config_endpoint                                   PASSED
test_router_api.py::test_config_does_not_expose_secrets                   PASSED
test_router_api.py::test_post_correction                                   PASSED
test_router_api.py::test_get_corrections_empty                             PASSED
test_router_api.py::test_get_corrections_after_post                       PASSED
test_router_api.py::test_status_endpoint_structure                        PASSED
test_router_api.py::test_query_single_domain                               PASSED
test_router_api.py::test_query_response_contains_text                     PASSED
test_router_api.py::test_query_batch                                       PASSED
test_router_api.py::test_reset_endpoint                                    PASSED
test_router_api.py::test_openapi_json_accessible                          PASSED
test_router_api.py::test_docs_accessible                                   PASSED
test_router_api.py::test_redoc_accessible                                  PASSED
test_router_api.py::test_version_endpoint_returns_correct_version         PASSED
test_router_api.py::test_cors_uses_config_origins                         PASSED
test_router_api.py::test_stream_named_event_fields                        PASSED
test_router_api.py::test_stream_content_encoding_none                     PASSED
test_status.py::test_fmt_uptime[0-0s]                                     PASSED
test_status.py::test_fmt_uptime[45-45s]                                   PASSED
test_status.py::test_fmt_uptime[90-1m 30s]                                PASSED
test_status.py::test_fmt_uptime[3600-1h 0m]                               PASSED
test_status.py::test_fmt_uptime[3723-1h 2m]                               PASSED
test_status.py::test_fmt_uptime[7200-2h 0m]                               PASSED
test_status.py::test_mini_bar_full                                         PASSED
test_status.py::test_mini_bar_empty                                        PASSED
test_status.py::test_mini_bar_half                                         PASSED
test_status.py::test_mini_bar_width                                        PASSED
test_status.py::test_render_returns_panel                                  PASSED
test_status.py::test_render_shows_up_down                                  PASSED
test_status.py::test_render_shows_utility_score                            PASSED
test_status.py::test_render_shows_memory                                   PASSED
test_status.py::test_render_none_shows_error_panel                        PASSED
test_streaming.py::test_stream_returns_200                                 PASSED
test_streaming.py::test_stream_emits_start_event                          PASSED
test_streaming.py::test_stream_emits_chunk_events                         PASSED
test_streaming.py::test_stream_emits_done_event                           PASSED
test_streaming.py::test_stream_event_order                                PASSED
test_streaming.py::test_stream_chunks_concatenate_to_response             PASSED
test_streaming.py::test_stream_sse_headers                                PASSED
test_version.py::test_version_format                                       PASSED
test_version.py::test_init_re_exports_version                             PASSED
test_version.py::test_version_in_all                                       PASSED
test_version.py::test_cli_version                                          PASSED

======================== 132 passed, 6 warnings in 15.69s ========================
```

Matrix: Python 3.10, 3.11, 3.12. All green on CI (GitHub Actions).

---

## 2. REST API — 23 endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Route a single query through the specialist graph |
| `POST` | `/query/stream` | Stream a query response token-by-token (SSE) |
| `POST` | `/query/batch` | Route multiple queries in parallel |
| `GET`  | `/health/live` | Liveness probe — is the router process alive? |
| `GET`  | `/health/ready` | Readiness probe — are all specialists reachable? |
| `GET`  | `/health/startup` | Startup probe — has the framework finished initialising? |
| `GET`  | `/health` | Legacy liveness alias |
| `POST` | `/corrections` | Inject a correction into the assertions store |
| `GET`  | `/corrections` | List stored corrections |
| `GET`  | `/config` | Return the running configuration (read-only) |
| `POST` | `/deploy/green` | Trigger a blue-green promotion evaluation |
| `GET`  | `/status` | Full telemetry snapshot (powers `aua status` dashboard) |
| `POST` | `/reset` | Reset domain confidence and classifier history |
| `GET`  | `/stats` | Telemetry alias (legacy) |
| `GET`  | `/version` | Return the running AUA Framework version |
| `POST` | `/sessions` | Create a new chat session |
| `GET`  | `/sessions` | List all chat sessions |
| `GET`  | `/sessions/{session_id}` | Get session metadata |
| `DELETE` | `/sessions/{session_id}` | Delete a session |
| `GET`  | `/sessions/{session_id}/messages` | List messages in a session |
| `POST` | `/sessions/{session_id}/messages` | Post a message to a session |
| `GET`  | `/metrics` | Prometheus metrics scrape endpoint |
| `GET`  | `/metrics/cost` | Cost tracking metrics (GPU hours, USD per query) |

Interactive docs: `http://localhost:8000/docs` (Swagger UI) · `http://localhost:8000/redoc`

---

## 3. Plugin Protocol Interfaces — 8 protocols + 1 middleware

Defined in `aua/plugins/interfaces.py`. All use Python `typing.Protocol` for structural subtyping — no base class required.

| Protocol | Description |
|----------|-------------|
| `FieldClassifierPlugin` | Replaces the built-in field classifier. Implement `classify(query) -> dict[str, float]` and `top_field(query) -> str`. |
| `UtilityScorerPlugin` | Replaces the built-in `U = w_e·E + w_c·C + w_k·K` scorer. Implement `score(response, field, prior_u) -> float` and `weights(field) -> dict`. |
| `ArbiterPolicyPlugin` | Replaces the built-in 4-check arbitration policy. Implement `arbitrate(claims, context) -> ArbiterVerdict` and `should_escalate(claims, context) -> bool`. |
| `PromotionPolicyPlugin` | Decides whether a GREEN candidate should be promoted to BLUE. Implement `should_promote(blue_stats, green_stats, config) -> bool` and `promotion_reason(…) -> str`. |
| `CorrectionStorePlugin` | Replaces the built-in in-memory `AssertionsStore`. Implement `store(claim)`, `query(subject, domain) -> list`, and `export_dpo(domain) -> list`. |
| `ModelBackendPlugin` | Replaces the built-in vLLM/Ollama HTTP backend. Implement `generate(prompt, model, params) -> str`, `stream(…)`, `health() -> bool`, and `models() -> list`. |
| `StateStorePlugin` | Pluggable persistent state store (SQLite default, Postgres via `asyncpg`). Implement `get`, `set`, `append`, `delete`, and `query`. |
| `HookPlugin` | Lifecycle hook. Fires at 11 named points in the request pipeline. Implement `hook_name() -> str` and `__call__(context) -> None`. |
| `AUAMiddleware` | Request/response middleware. Runs before and after the query pipeline. Implement `before_query`, `after_query`, `before_specialist`, `after_specialist`. |

Register via `aua_config.yaml`:
```yaml
extensions:
  - import_path: "mypackage.myplugin:MyClassifierPlugin"
```

---

## 4. Prometheus Metrics — 15 metrics

Scraped at `GET /metrics`. All metrics prefixed `aua_`.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `aua_queries_total` | Counter | `domain`, `routing_mode`, `status` | Total queries routed |
| `aua_query_latency_seconds` | Histogram | `domain`, `routing_mode` | End-to-end query latency |
| `aua_utility_score` | Gauge | `domain` | Last U score per domain |
| `aua_contradiction_rate` | Gauge | `domain` | Arbiter contradiction rate |
| `aua_routing_field_distribution` | Counter | `field` | Classifier field assignment counts |
| `aua_specialist_confidence` | Gauge | `specialist` | Per-specialist confidence score |
| `aua_correction_count` | Counter | `domain` | Corrections stored per domain |
| `aua_arbiter_verdict_distribution` | Counter | `case` | Verdict cases (A/B/C/D) distribution |
| `aua_dpo_pairs_accumulated` | Gauge | — | Total DPO training pairs in store |
| `aua_token_requests_total` | Counter | `scope`, `status` | Token auth requests |
| `aua_hook_failures_total` | Counter | `hook_point` | Hook execution failures by hook point |
| `aua_plugin_execution_seconds` | Histogram | `plugin`, `kind` | Plugin execution latency |
| `aua_specialist_vram_utilization` | Gauge | `specialist` | VRAM utilisation (0–1) |
| `aua_cost_gpu_hours_total` | Counter | `specialist` | Cumulative GPU hours per specialist |
| `aua_cost_usd_total` | Counter | `specialist` | Cumulative USD cost per specialist |

Grafana dashboard: `docker/grafana/aua_dashboard.json` — 20 panels, pre-provisioned.

---

## 5. CLI — 10 command groups, 40+ subcommands

```
aua --version  # 1.0.0
```

| Group | Subcommands | Description |
|-------|-------------|-------------|
| `aua init` | _(positional: name)_ `--preset` `--tier` `--force` | Scaffold a new AUA project |
| `aua serve` | `--config` `--tier` `--dry-run` `--with-ui` `--ui-port` `--reuse-running` `--router-only` | Start specialists + router |
| `aua doctor` | `--config` `--strict` `--json` | Pre-flight readiness check |
| `aua status` | `--config` `--interval` `--once` | Live terminal dashboard |
| `aua config` | `validate` · `expand` · `reload` | Config management |
| `aua eval` | `run` · `report` · `compare` | Evaluation harness |
| `aua token` | `create` · `list` · `inspect` · `revoke` | API token management |
| `aua certs` | `generate` · `inspect` | mTLS certificate management |
| `aua dpo` | `export` | Export DPO pairs for fine-tuning |
| `aua corrections` | `export` | Export stored corrections |
| `aua rollback` | _(positional: specialist)_ `--all` `--dry-run` `--no-restart` | Blue-green rollback |
| `aua extensions` | `list` · `inspect` · `test` | Plugin/hook management |
| `aua models` | `list` | Model pull status |
| `aua fields` | `list` | Field config introspection |
| `aua presets` | `list` | Preset introspection |
| `aua defaults` | `show` | Framework defaults |
| `aua ui` | `--port` `--install-only` | Chat UI (standalone) |

---

## 6. Fresh-Clone Install

```bash
# Environment: Python 3.11.10 (pyenv), macOS Apple Silicon
git clone https://github.com/praneethtota/Adaptive-Utility-Agent.git
cd Adaptive-Utility-Agent

pip install -e ".[dev]"
# Successfully installed adaptive-utility-agent-1.0.0 ...

aua --version
# aua, version 1.0.0

aua init my-test-project --preset coding --tier macbook
# ✓ Created my-test-project/
# ✓ aua_config.yaml written (tier: macbook)
# ✓ evals/ scaffolded

cd my-test-project && aua doctor
# ✓ Config valid
# ✓ Ollama reachable on port 11434
# ✓ All checks passed

pytest -q
# 132 passed, 6 warnings in 15.69s
```

---

## 7. Docker Compose Validation

```bash
# CPU/Ollama profile (macOS / CPU servers)
docker compose --profile ollama up -d
# ✓ aua-ollama       healthy (30s)
# ✓ aua-model-puller exited 0 (models pulled)
# ✓ aua-router       healthy

curl http://localhost:8000/health/live
# {"status":"ok","version":"1.0.0"}

# Observability stack
docker compose --profile obs up -d
# ✓ aua-prometheus   healthy (port 9090)
# ✓ aua-grafana      healthy (port 3000)
# Dashboard auto-provisioned at http://localhost:3000

# GPU profile (Linux + NVIDIA)
docker compose -f docker-compose.gpu.yml up -d
# ✓ aua-router       healthy (vLLM backend)
```

---

## 8. Chat UI Startup Validation

```bash
# Terminal 1 — AUA router
aua serve --tier macbook
# ✓ ollama healthy (3s)
# ✓ qwen2.5-coder:7b already pulled
# ✓ qwen2.5:7b already pulled
# ✓ qwen2.5:3b already pulled
# INFO: Uvicorn running on http://0.0.0.0:8000

# Terminal 2 — Next.js Chat UI (Node.js 18+)
cd apps/aua_chat && npm install && npm run dev
# ▲ Next.js 14.x
# - Local: http://localhost:3001
# ✓ Ready in 4.2s

# Browser: http://localhost:3001
# Login: admin / aua-admin
# ✓ Three-panel layout: Sidebar | Chat | Framework Debugger
# ✓ AUA Controls drawer opens on click
# ✓ Query routed, debugger shows domain, U score, latency
```

Note: `aua serve --with-ui` attempts to start the Next.js process automatically. On macOS with nvm/homebrew, the manual two-terminal approach above is recommended if `--with-ui` does not produce a `✓ Chat UI` confirmation line. UI startup log: `.aua/logs/ui.log`.

---

## 9. Security Validation

```bash
# Token auth
aua token create --scope aua:query --expires 30d
# Token: aua_tk_...

curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer aua_tk_..." \
  -d '{"query": "test"}'
# ✓ 200 OK

curl -X POST http://localhost:8000/query \
  -d '{"query": "test"}'
# ✓ 401 Unauthorized

# 14 auth scopes: aua:query, aua:query:stream, aua:query:batch,
#   aua:corrections:read, aua:corrections:write, aua:config:read,
#   aua:deploy, aua:status, aua:reset, aua:sessions:read,
#   aua:sessions:write, aua:metrics, aua:tokens:manage, aua:admin

# mTLS certificates
aua certs generate
# ✓ ca.crt, router.crt, router.key written to .aua/certs/

# Encryption at rest (AES-256-GCM)
export AUA_ENCRYPTION_KEY=$(python3 -c "import os; print(os.urandom(32).hex())")
# Corrections, assertions, DPO pairs encrypted in state store

# Config redaction — secrets never exposed via API
curl http://localhost:8000/config | jq '.security'
# {"auth":{"enabled":true},"encryption":{"enabled":true,"key_secret":"[REDACTED]"}}
```

---

## 10. Observability Validation

```bash
# Prometheus scrape
curl http://localhost:8000/metrics | grep "^aua_"
# aua_queries_total{domain="software_engineering",routing_mode="single",status="ok"} 12.0
# aua_query_latency_seconds_bucket{domain="software_engineering",...} ...
# aua_utility_score{domain="software_engineering"} 0.7831
# aua_contradiction_rate{domain="software_engineering"} 0.0
# aua_routing_field_distribution{field="software_engineering"} 10.0
# aua_specialist_confidence{specialist="swe"} 0.823
# aua_correction_count{domain="software_engineering"} 0.0
# aua_arbiter_verdict_distribution{case="A"} 12.0
# aua_dpo_pairs_accumulated 0.0
# aua_cost_gpu_hours_total{specialist="swe"} 0.0114
# aua_cost_usd_total{specialist="swe"} 0.0079

# Live status dashboard
aua status
# ✓ All specialists up, U scores, VRAM, uptime displayed

# OTEL export (optional)
# Set OTEL_EXPORTER_OTLP_ENDPOINT to export traces to Jaeger/Tempo

# Grafana: http://localhost:3000 (admin / aua-admin)
# ✓ 20 pre-configured panels
# ✓ AUA dashboard auto-provisioned from docker/grafana/aua_dashboard.json
```
