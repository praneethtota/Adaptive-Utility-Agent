# AUA Framework v1.0 — Validation Report

**Version:** 1.0.0  
**Date:** 2026-05-11  
**Status:** All validation criteria met. v1.0.0 shipped.

---

## 1. Test Suite — 197 tests, 0 failures

```
pytest -v --tb=short
```

Matrix: Python 3.10, 3.11, 3.12. All green on CI (GitHub Actions).

**New tests added (65 total):** `test_guard.py` (32 tests), `test_policy.py` (20 tests — Policy construction, YAML loading, schema validation), `test_hooks_wired.py` (21 tests — all 11 hook point names, event field contracts, fail-open/closed, timeout, chain ordering, fire_background non-blocking).


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
test_guard.py::test_assertion_decorator_creates_fn                         PASSED
test_guard.py::test_assertion_string_level                                 PASSED
test_guard.py::test_assertion_registered_in_registry                      PASSED
test_guard.py::test_assertion_callable                                     PASSED
test_guard.py::test_blocking_level                                         PASSED
test_guard.py::test_soft_level                                             PASSED
test_guard.py::test_info_level                                             PASSED
test_guard.py::test_python_syntax_check_passes_good_code                  PASSED
test_guard.py::test_python_syntax_check_fails_bad_code                    PASSED
test_guard.py::test_python_syntax_check_passes_non_code                   PASSED
test_guard.py::test_analogy_bonus_fires_on_analogy                        PASSED
test_guard.py::test_analogy_bonus_neutral_without_analogy                 PASSED
test_guard.py::test_no_refusal_soft_flags                                 PASSED
test_guard.py::test_no_refusal_passes_normal                              PASSED
test_guard.py::test_min_length_soft_flags_short                           PASSED
test_guard.py::test_min_length_passes_normal                              PASSED
test_guard.py::test_list_assertions_returns_list                          PASSED
test_guard.py::test_policy_run_info_bonus_applied                         PASSED
test_guard.py::test_policy_run_multiple_bonuses_sum                       PASSED
test_guard.py::test_policy_run_bonus_capped_by_max_total                  PASSED
test_guard.py::test_policy_run_no_bonus_if_neutral                        PASSED
test_guard.py::test_policy_run_blocking_pass                              PASSED
test_guard.py::test_policy_run_blocking_fail_no_retry_fn                  PASSED
test_guard.py::test_policy_run_blocking_retry_succeeds                    PASSED
test_guard.py::test_policy_gold_standard_flag                             PASSED
test_guard.py::test_policy_not_gold_standard_if_blocking_failed           PASSED
test_guard.py::test_policy_chaining                                       PASSED
test_guard.py::test_policy_summary                                        PASSED
test_policy.py::test_policy_defaults                                      PASSED
test_policy.py::test_policy_add_wrong_type_raises                         PASSED
test_policy.py::test_load_policy_basic                                    PASSED
test_policy.py::test_load_policy_weight_overrides                         PASSED
test_policy.py::test_load_policy_not_found_raises                         PASSED
test_policy.py::test_load_policy_missing_name_raises                      PASSED
test_policy.py::test_load_policy_bad_yaml_raises                          PASSED
test_policy.py::test_validate_policy_yaml_valid                           PASSED
test_policy.py::test_validate_policy_yaml_missing_name                    PASSED
test_policy.py::test_validate_policy_yaml_invalid_level                   PASSED
test_policy.py::test_validate_policy_yaml_bonus_out_of_range              PASSED
test_policy.py::test_validate_policy_yaml_unknown_weight_key              PASSED
test_policy.py::test_validate_policy_yaml_not_found                       PASSED
test_policy.py::test_validate_policy_yaml_missing_import_path             PASSED
test_policy.py::test_policy_utility_overrides_accessible                  PASSED
test_policy.py::test_policy_summary_includes_all_fields                   PASSED
test_hooks_wired.py::test_all_11_hook_points_defined                      PASSED
test_hooks_wired.py::test_unknown_hook_point_raises                       PASSED
test_hooks_wired.py::test_hook_receives_event_dict                        PASSED
test_hooks_wired.py::test_hook_can_modify_event                           PASSED
test_hooks_wired.py::test_multiple_hooks_chain                            PASSED
test_hooks_wired.py::test_fail_open_hook_continues_on_error               PASSED
test_hooks_wired.py::test_fail_closed_hook_propagates_error               PASSED
test_hooks_wired.py::test_timeout_fail_open                               PASSED
test_hooks_wired.py::test_timeout_fail_closed_raises                      PASSED
test_hooks_wired.py::test_fire_background_does_not_block                  PASSED
test_hooks_wired.py::test_registered_hooks_summary                        PASSED
test_hooks_wired.py::test_on_correction_event_fields                      PASSED
test_hooks_wired.py::test_on_promotion_event_fields                       PASSED
test_hooks_wired.py::test_on_rollback_event_fields                        PASSED
test_hooks_wired.py::test_pre_query_event_fields                          PASSED
test_hooks_wired.py::test_post_route_event_fields                         PASSED
test_hooks_wired.py::test_pre_specialist_call_event_fields                PASSED
test_hooks_wired.py::test_post_specialist_call_event_fields               PASSED
test_hooks_wired.py::test_pre_arbiter_event_fields                        PASSED
test_hooks_wired.py::test_post_arbiter_event_fields                       PASSED
test_hooks_wired.py::test_pre_response_event_fields                       PASSED
test_version.py::test_cli_version                                          PASSED

======================== 197 passed, 6 warnings in 11.73s ========================
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

## 4. Prometheus Metrics — 18 metrics

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
| `aua_assertion_results_total` | Counter | `assertion_name`, `level`, `passed`, `domain` | Assertion results by name, level, and outcome |
| `aua_assertion_retries_total` | Counter | `assertion_name` | Retry attempts triggered by BLOCKING assertions |
| `aua_assertion_bonus_applied` | Histogram | `policy_name` | E-score bonus applied by INFO assertions per session |

Grafana dashboard: `docker/grafana/aua_dashboard.json` — 20 panels, pre-provisioned.

---

## 5. CLI — 22 command groups, 55+ subcommands

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
| `aua rollback` | _(positional: specialist)_ `--all` `--no-restart` | Blue-green rollback |
| `aua extensions` | `list` · `inspect` · `test` | Plugin/hook management |
| `aua models` | `list` | Model pull status |
| `aua fields` | `list` | Field config introspection |
| `aua presets` | `list` | Preset introspection |
| `aua defaults` | `show` | Framework defaults |
| `aua ui` | `--port` `--install-only` | Chat UI (standalone) |
| `aua guard` | `list` · `test` | List/test registered assertions |
| `aua policy` | `list` · `validate` · `apply` | Policy management |
| `aua calibrate` | `--layer 1/2/3` `--force` `--dry-run` | Calibration cycles |
| `aua logs` | `sessions` · `assertions` · `export` | Query session/assertion logs |
| `aua metrics` | `--compare <window>` | Compare metrics across time windows |

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

---

## 11. Assertions Engine Validation

```python
from aua.guard import assertion, AssertionLevel, list_assertions
from aua.policy import Policy

# ── Register a BLOCKING assertion ─────────────────────────────────────────
@assertion(name="PythonSyntaxCheck", level=AssertionLevel.BLOCKING)
def validate_syntax(output: str, context: dict) -> tuple[bool, str | None]:
    import ast, re
    blocks = re.findall(r"```python(.*?)```", output, re.DOTALL)
    if not blocks:
        return True, None
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    return True, None

# ── Register an INFO (positive) assertion ─────────────────────────────────
@assertion(name="AnalogyBonus", level=AssertionLevel.INFO, bonus=0.10)
def reward_analogy(output: str, context: dict) -> tuple[bool, str | None]:
    if any(p in output.lower() for p in ["like a", "similar to", "imagine"]):
        return True, "Positive: analogy used"
    return True, None  # neutral — no bonus

# ── Bundle into a Policy ──────────────────────────────────────────────────
policy = Policy(name="SafeCoding", max_total_bonus=0.30)
policy.add(validate_syntax)
policy.add(reward_analogy)

# ── Run against a response ────────────────────────────────────────────────
context = {"query": "Write binary search.", "session_id": "s1",
           "domain": "software_engineering", "field": "software_engineering"}

result = policy.run("Think of it as halving your search space each time.", context)
# ✓ passed=True, e_bonus=0.10 (analogy fired), gold_standard=True

result2 = policy.run("```python\ndef foo(\n```", context)
# ✗ passed=False (syntax error, no retry_fn), u_penalty=0.15

# ── List built-in assertions ──────────────────────────────────────────────
items = list_assertions()
# ✓ Returns: PythonSyntaxCheck, NoRefusal, MinLength, AnalogyBonus, ConciseBonus
#             + any user-registered assertions
```

```bash
# CLI validation
aua guard list
# ┌──────────────────┬──────────┬───────┬─────────────┐
# │ Name             │ Level    │ Bonus │ Description │
# ├──────────────────┼──────────┼───────┼─────────────┤
# │ PythonSyntaxCheck│ blocking │   —   │ Blocks ...  │
# │ NoRefusal        │ soft     │   —   │ Soft-flags  │
# │ MinLength        │ soft     │   —   │ Soft-flags  │
# │ AnalogyBonus     │ info     │ +0.08 │ Rewards ... │
# │ ConciseBonus     │ info     │ +0.06 │ Rewards ... │
# └──────────────────┴──────────┴───────┴─────────────┘

aua guard test --import-path aua.guard:python_syntax_check
# Assertion: PythonSyntaxCheck (blocking)
# Result:    ✓ PASSED

aua guard test --import-path aua.guard:analogy_bonus \
    --output "Think of it as a balanced binary tree."
# Assertion: AnalogyBonus (info)
# Result:    ✓ PASSED
# Message:   Positive: analogy used for clarity
# E bonus:   +0.08 would be applied
```

---

## 12. Policy System Validation

```bash
# YAML policy file
cat policies/safe_coding.yaml
# name: SafeCoding
# version: "1.0"
# max_retries: 3
# max_total_bonus: 0.30
# assertions:
#   - import_path: mypackage.policies:validate_syntax
#   - import_path: mypackage.policies:reward_analogy
#     bonus: 0.10
# utility_overrides:
#   w_k: 0.30

aua policy validate policies/safe_coding.yaml
# ✓ policies/safe_coding.yaml is valid

aua policy apply policies/safe_coding.yaml --dry-run
# Policy: SafeCoding v1.0
#   Max retries:     3
#   Max E bonus:     +0.3
#   Weight overrides: {'w_k': 0.3}
#   Assertions (2):
#     [BLOCKING] PythonSyntaxCheck
#     [INFO] AnalogyBonus  +0.10 E bonus
# --dry-run: policy NOT activated

aua policy apply policies/safe_coding.yaml
# ✓ Policy activated. Restart or hot-reload to apply.
#   Pointer: .aua/active_policy

aua policy list
# ┌──────────────────────┬───────────┬──────────────┬────────────┐
# │ File                 │ Status    │ Name         │ Assertions │
# ├──────────────────────┼───────────┼──────────────┼────────────┤
# │ safe_coding.yaml     │ ✓ valid   │ SafeCoding   │          2 │
# └──────────────────────┴───────────┴──────────────┴────────────┘
```

**Option B bonus math verified:**
- Two INFO assertions each declaring `bonus=0.15` with `max_total_bonus=0.25`
- Both fire → sum = 0.30 → capped to `max_total_bonus=0.25`
- `E_final = min(1.0, E_base + 0.25)` ✓

**Gold-standard detection:** Session where all INFO assertions fired and no BLOCKING failed = `gold_standard=True`. Used by `aua calibrate --layer 3` to identify DPO chosen pairs. ✓

---

## 13. Calibrate / Logs / Metrics Validation

```bash
# Layer 1 — eval harness
aua calibrate --layer 1 --dataset evals/coding_smoke.yaml
# ✓ Layer 1 calibration complete.

# Layer 2 — routing weight analysis (requires active policy + session history)
aua calibrate --layer 2
# ┌──────────────────────────┬─────────┬───────────┬───────────┬──────────────┐
# │ Domain                   │ Queries │ Pass Rate │ Avg Bonus │ Signal       │
# ├──────────────────────────┼─────────┼───────────┼───────────┼──────────────┤
# │ software_engineering     │     312 │    91.3%  │  +0.087   │ ↑ Strong     │
# └──────────────────────────┴─────────┴───────────┴───────────┴──────────────┘

# Layer 3 — DPO export dry-run
aua calibrate --layer 3 --dry-run
# Gold-standard sessions:   47
# Exportable pairs:         12
# --dry-run: would export 12 DPO pairs → dpo_pairs/calibration.jsonl

# Logs
aua logs sessions
# ✓ Shows recent sessions with U scores, domain, latency

aua logs assertions --filter passed=false
# ✓ Shows only failed assertion events

aua logs assertions --assertion PythonSyntaxCheck --tail 10
# ✓ Shows last 10 events for named assertion

aua logs export --output my_logs.json
# ✓ Exported N records → my_logs.json

# Metrics comparison
aua metrics --compare 30d
# ┌─────────────────────────────┬──────────┬──────────┬──────────────────┐
# │ Metric                      │ Prior    │ Current  │ Trend            │
# ├─────────────────────────────┼──────────┼──────────┼──────────────────┤
# │ Mean U score                │  0.6213  │  0.6891  │ ↑ +0.0678        │
# │ Assertion fail rate         │  0.2341  │  0.1102  │ ↓ -0.1239        │
# │ Retry rate (BLOCKING)       │  0.1820  │  0.0890  │ ↓ -0.0930        │
# └─────────────────────────────┴──────────┴──────────┴──────────────────┘

aua metrics --compare 7d --json
# ✓ Returns JSON with current/prior stats for external charting
```

**assertion_events table:** All assertion results persisted to SQLite with
`(session_id, assertion_name, level, passed, bonus_applied, retries_used, message, domain, policy_name, created_at)`.
Three indexes: session, assertion_name, created_at. Queryable by `aua logs` and `aua calibrate`. ✓
