# AUA Framework

> A production framework for self-correcting, multi-specialist LLM systems.

**Full site:** https://praneethtota.github.io/Adaptive-Utility-Agent

---

## What it does

AUA sits between your application and your language models. It routes prompts to specialist models, scores responses with a utility function, catches contradictions, injects prior verified corrections into future queries, enforces policies in real-time, and self-corrects across sessions.

The core idea: a model that makes a wrong answer on Tuesday should not make the same wrong answer on Thursday. AUA closes that loop without waiting for a new model release.

```bash
pip install adaptive-utility-agent
aua init my-project --preset coding --tier macbook
cd my-project && aua serve
```

---

## Sister project: AUA Veritas

**AUA Veritas** applies the framework ideas in a consumer-facing desktop app — compare multiple frontier models, remember corrections, return one answer with a confidence signal.

👉 [AUA Veritas](https://github.com/praneethtota/AUA-Veritas)

---

## Documentation

| Page | Audience | Link |
|---|---|---|
| **Landing page** | Everyone | [whitepaper.html](https://praneethtota.github.io/Adaptive-Utility-Agent/whitepaper.html) |
| **Tutorial** (20 How-tos) | ML engineers, builders | [tutorial.html](https://praneethtota.github.io/Adaptive-Utility-Agent/tutorial.html) |
| **Production architecture** | DevOps, platform engineers | [productionizing.html](https://praneethtota.github.io/Adaptive-Utility-Agent/productionizing.html) |
| **Whitepaper** (7 parts) | Researchers, theorists | [whitepaper_overview.html](https://praneethtota.github.io/Adaptive-Utility-Agent/whitepaper_overview.html) |
| **Roadmap** | Everyone | [aua_roadmap.html](https://praneethtota.github.io/Adaptive-Utility-Agent/aua_roadmap.html) |
| AI Data Centers | Inference infra, GPU cloud | [domain_ai_datacenters.html](https://praneethtota.github.io/Adaptive-Utility-Agent/domain_ai_datacenters.html) |
| Self-Driving Vehicles | AV engineers | [domain_self_driving.html](https://praneethtota.github.io/Adaptive-Utility-Agent/domain_self_driving.html) |
| Autonomous Systems | Robotics, safety engineering | [domain_autonomous_systems.html](https://praneethtota.github.io/Adaptive-Utility-Agent/domain_autonomous_systems.html) |
| Software Engineering | Coding agents, dev-tools | [domain_software_engineering.html](https://praneethtota.github.io/Adaptive-Utility-Agent/domain_software_engineering.html) |
| Dynamic Pricing | Pricing platforms | [domain_dynamic_pricing.html](https://praneethtota.github.io/Adaptive-Utility-Agent/domain_dynamic_pricing.html) |
| Energy Systems | Grid software, DER | [domain_energy_systems.html](https://praneethtota.github.io/Adaptive-Utility-Agent/domain_energy_systems.html) |
| Creative Systems | Generative media | [domain_creative_systems.html](https://praneethtota.github.io/Adaptive-Utility-Agent/domain_creative_systems.html) |
| Recommendation Engines | RecSys, personalization | [domain_recommendation_engines.html](https://praneethtota.github.io/Adaptive-Utility-Agent/domain_recommendation_engines.html) |

---

## Quickstart

### Install

```bash
pip install adaptive-utility-agent

# With GPU serving backend (Linux + CUDA)
pip install "adaptive-utility-agent[vllm]"

# With development tools
pip install "adaptive-utility-agent[dev]"
```

### Scaffold and serve

```bash
# Mac / Apple Silicon — uses Ollama (brew install ollama first)
aua init my-project --preset coding --tier macbook
cd my-project
aua doctor        # pre-flight check: config, deps, hardware, compat matrix
aua serve         # start specialists + router on :8000
```

### Send a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Write binary search in Python. State time complexity."}'
```

```python
from aua import Router
from aua.config import load_config

config = load_config("aua_config.yaml")
router = Router.from_config(config)
result = await router.query("Write bubble sort. What is its O complexity?")
print(result.response)
print(f"U={result.u_score:.3f}  mode={result.routing_mode}  degraded={result.degraded_mode}")
```

### Chat UI

```bash
# Terminal 1
aua serve --tier macbook

# Terminal 2
aua ui   # starts on http://localhost:3001 (admin / aua-admin)
```

---

## Hardware tiers

| Tier flag | Hardware | Backend | Notes |
|---|---|---|---|
| `macbook` | Apple M-series | Ollama | `brew install ollama` |
| `gaming-pc` | RTX 3080/4080 (10–16 GB) | Ollama | Windows/Linux dev |
| `single-4090` | 1× RTX 4090 24 GB | vLLM AWQ | |
| `quad-4090` | 4× RTX 4090 | vLLM AWQ | TP=2 per specialist |
| `a100-cluster` | 8× A100 80 GB | vLLM bf16 | TP=4 |
| `h100-cluster` | 8× H100 SXM5 NVLink | vLLM bf16 | TP=4, highest throughput |

Aliases: `gaming` → `gaming-pc`, `h100` → `h100-cluster`, `a100` → `a100-cluster`, `rtx4090` → `single-4090`.

Check compatibility before serving:

```bash
aua doctor --compat-matrix              # full model × hardware × backend matrix
aua doctor --compat-matrix-format json  # machine-readable
```

---

## What ships in v1.2

| Component | Detail |
|---|---|
| **REST API** | 50+ endpoints — query, stream, batch, corrections (full CRUD), config, deploy, blue-green, shadow mode, status, sessions, metrics, keyword search, analytics, context backups, domain ontology, batch jobs |
| **CLI** | 24 command groups — `aua init`, `aua serve`, `aua doctor`, `aua test`, `aua loadtest`, `aua eval`, `aua guard`, `aua policy`, `aua calibrate`, `aua models pin`, `aua token`, `aua certs`, and more |
| **Plugin system** | 15 Protocol interfaces, 13 fully wired (see below) |
| **Extended middleware** | `before_query` / `after_response` / `on_chunk` (SSE interception) / `before_batch` / `after_batch` / `on_error` |
| **Hooks** | 11 lifecycle hook points — `pre_query`, `post_route`, `pre_specialist_call`, `post_specialist_call`, `pre_arbiter`, `post_arbiter`, `on_correction`, `pre_response`, `post_response`, `on_promotion`, `on_rollback` |
| **Bearer token auth** | HMAC-SHA256, 15 scopes, revocation — activated via `security.auth_enabled: true` |
| **mTLS** | Server TLS and mutual TLS via `security.mtls.key_file / cert_file / ca_file` |
| **Retry + backoff** | Per-specialist transport retry, exponential backoff, ±25% jitter, configurable retryable status codes |
| **Circuit breaker** | Per-specialist CLOSED/OPEN/HALF_OPEN state machine; degraded-mode flag on responses when specialists are bypassed |
| **Multi-tenancy** | Per-tenant rate limits, field allowlists, model bindings, namespaced DB writes |
| **Shadow mode** | Silent GREEN evaluation on real traffic; fire-and-forget (zero latency impact) |
| **Regression gate** | Blocks promotion when GREEN regresses on an eval dataset |
| **Experiment tracking** | MLflow + W&B lazy integration — per-query metric logging |
| **Batch queue** | Persistent `/batch/jobs` REST API, priority lanes, partial results, restart recovery |
| **Model registry** | HF `@revision` / `@sha256` pinning, MLflow `models:/` URI resolution |
| **Compatibility matrix** | model format × hardware × backend — `aua doctor --compat-matrix` |
| **Arbiter pipeline** | ArbiterAgent (4-check: logical, mathematical, cross-session, empirical via SymPy/arXiv/PubMed) is the live default; simplified LLM path via `arbitration_mode: "llm"` |
| **Tau softmax routing** | `router.tau` — sharpens or softens the field classifier distribution before thresholds |
| **T_min gate** | Minimum shadow query count required before promotion is considered |
| **Test suite** | 759 tests, Python 3.10 / 3.11 / 3.12, CI green |

---

## Plugin system — 15 interfaces, 13 wired

Every major decision point is replaceable via a single YAML line. No forking required.

```yaml
plugins:
  routing_strategy:
    import_path: my_plugins:TenantRouter
  full_utility_scorer:
    import_path: my_plugins:SurgeryAwareScorer
  full_promotion_policy:
    import_path: my_plugins:CIGatePromoter
```

| YAML key | Wired | What it replaces |
|---|---|---|
| `field_classifier` | ✅ | Domain classifier |
| `utility_scorer` | ✅ | Final U score (adjustment mode — receives `prior_u`) |
| `full_utility_scorer` | ✅ | Entire U computation — bypasses `w_e·E + w_c·C + w_k·K`, enables quadratic/Cobb-Douglas/Rawlsian models |
| `arbiter_policy` | ✅ | LLM arbitration call in fanout routing |
| `promotion_policy` | ✅ | Promotion gate (pre-computed scalars) |
| `full_promotion_policy` | ✅ | Promotion gate with full context — shadow scores, std_delta, regression results |
| `contradiction_detector` | ✅ | Built-in code contradiction checker |
| `assertion_store` | ✅ | In-memory AssertionsStore |
| `routing_strategy` | ✅ | Post-classifier distribution — intercepts before single/fanout/arbiter decision |
| `scoring_component` | ✅ | One sub-score (E, C, or K) within the built-in pipeline |
| `correction_store` | ✅ | DPO pair / correction storage |
| `hook` | ✅ | 11 lifecycle points |
| `middleware` | ✅ | Request/response/streaming/batch pipeline |
| `model_backend` | ⏳ #74 | Per-specialist inference backend — validates at startup, not yet dispatched |
| `state_store` | ⏳ #75 | SQLite state store — validates at startup, not yet dispatched (init ordering) |

All plugins are validated against their Protocol interface at startup — a misconfigured plugin fails fast, never silently at query time. Every wired plugin has a safe fallback: an exception at query time logs at DEBUG and falls back to the built-in.

---

## The utility function

```
U = w_e(f) · E  +  w_c(f) · C  +  w_k(f) · K

E — Efficacy:    EMA-accumulated task performance                        [0, 1]
C — Confidence:  Kalman-filtered internal consistency after contradiction penalty  [0, 1]
K — Curiosity:   UCB-style exploration bonus (K_base + gap_bonus)       [0, 1]
f — field        (software_engineering, mathematics, surgery, law, ...)
```

The additive weighted structure is not a convenience — it is the unique functional form satisfying five behavioral axioms, proved via Debreu's representation theorem (Theorem B.1, [Appendix B](https://praneethtota.github.io/Adaptive-Utility-Agent/whitepaper_appendix_b.html)).

Replace it entirely with a `full_utility_scorer` plugin:

```python
class SurgeryAwareScorer:
    def score(self, response, field, prior_u, confidence, metadata):
        return prior_u  # fallback

    def score_full(self, field, efficacy, confidence, curiosity, weights, metadata):
        if field == "surgery":
            return min(1.0, efficacy * (confidence ** 2))  # non-linear — C is load-bearing
        return weights["w_e"]*efficacy + weights["w_c"]*confidence + weights["w_k"]*curiosity
```

---

## Policies — teaching the framework what good looks like

```python
from aua.guard import assertion, AssertionLevel
from aua.policy import Policy

@assertion(name="PythonSyntaxCheck", level=AssertionLevel.BLOCKING)
def validate_syntax(output: str, context: dict) -> tuple[bool, str | None]:
    import ast, re
    for block in re.findall(r"```python(.*?)```", output, re.DOTALL):
        try:
            ast.parse(block)
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}"
    return True, None

@assertion(name="AnalogyBonus", level=AssertionLevel.INFO, bonus=0.10)
def reward_analogy(output: str, context: dict) -> tuple[bool, str | None]:
    if any(p in output.lower() for p in ["like a", "similar to", "imagine"]):
        return True, "Positive: analogy used"
    return True, None

policy = Policy(name="SafeCoding", max_total_bonus=0.30)
policy.add(validate_syntax)
policy.add(reward_analogy)
```

Over time: BLOCKING assertions reduce failures → passing sessions become gold-standard DPO data → `aua calibrate --layer 3` exports them → fine-tune → repeat.

---

## Resilience — retry and circuit breaker

```yaml
router:
  retry:
    max_retries: 3          # 0 to disable
    base_delay_ms: 200      # doubles per attempt, capped at max_delay_ms
    max_delay_ms: 5000
    jitter: true            # ±25% — prevents thundering-herd
    retryable_status_codes: [429, 502, 503, 504]

  circuit_breaker:
    enabled: true
    failure_threshold: 5    # failures within window before opening
    failure_window_s: 60.0
    recovery_timeout_s: 30.0
    success_threshold: 2    # consecutive successes in HALF_OPEN → CLOSED
```

When a circuit is open, responses include `degraded_mode: true` and `degraded_specialists: ["mathematics"]`. The router continues serving via the arbiter or remaining healthy specialists — zero additional latency for end users once the circuit opens.

---

## Security

```yaml
security:
  auth_enabled: true
  token_secret_env: AUA_TOKEN_SECRET   # export AUA_TOKEN_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  token_expiry_days: 30
  mtls:
    key_file: certs/server.key
    cert_file: certs/server.crt
    ca_file: certs/ca.crt    # omit for server-TLS-only
```

```bash
aua token create --scope aua:query --expires 30d
curl -H "Authorization: Bearer aua_tk_..." http://localhost:8000/query ...
aua certs generate   # self-signed dev certs
```

---

## Project structure

```
aua/
├── router.py               # Request routing + 50+ REST endpoints
├── arbiter.py              # 4-check arbitration pipeline (logical, math, cross-session, empirical)
├── utility_scorer.py       # U = w_e·E + w_c·C + w_k·K
├── field_classifier.py     # Probabilistic domain routing
├── assertions_store.py     # Cross-session corrections with decay classes A–D
├── retry.py                # Transport-level retry with exponential backoff (#39)
├── circuit_breaker.py      # Per-specialist CLOSED/OPEN/HALF_OPEN state machine (#37)
├── middleware.py           # Extended pipeline: on_chunk, before/after_batch, on_error (#52)
├── auth.py                 # HMAC-SHA256 token auth, 15 scopes, revocation
├── auth_middleware.py      # FastAPI middleware wiring auth into the request path
├── shadow.py               # Shadow mode — real-traffic GREEN evaluation (#48)
├── experiment_tracker.py   # MLflow + W&B lazy integration (#47)
├── batch_queue.py          # Persistent batch queue, priority lanes (#56)
├── model_registry.py       # HF @revision pinning, MLflow models:/ resolution (#46)
├── compat.py               # Model × hardware × backend compatibility matrix (#55)
├── empirical.py            # SymPy / arXiv / PubMed cross-check for arbiter Stage 4 (#61)
├── keywords.py             # Async full-text keyword search index
├── tenancy.py              # Per-tenant contextvar isolation (#44)
├── loadtest.py             # aua loadtest engine (#50)
├── test_harness.py         # aua test built-in fixture suites (#54)
├── blue_green.py           # Utility-deviation-triggered promotion, T_min gate, tau routing
├── guard.py                # @assertion decorator, AssertionLevel, Policy.run()
├── policy.py               # Policy dataclass + YAML loader
├── hooks.py                # HookRunner — 11 lifecycle hook points
├── metrics.py              # 18 Prometheus metrics
├── otel.py                 # OpenTelemetry tracing
├── state.py                # SQLite state store (sessions, corrections, audit log)
├── cli.py                  # aua CLI — 24 command groups
├── config.py               # AUAConfig, RetryConfig, CircuitBreakerConfig, tier loader
└── plugins/
    ├── interfaces.py       # 15 Protocol interfaces
    ├── registry.py         # Plugin load + contract validation
    └── prebuilt/           # OpenAI, Anthropic, Google backends (wired when #74 ships)

apps/
└── aua_chat/               # Next.js 14 Chat UI

tests/                      # 759 tests across Python 3.10 / 3.11 / 3.12
```

---

## Validated results (v1.0 baseline, RTX 4090)

| Result | Value |
|---|---|
| Repeated error reduction | **69.6%** (14 vs 46 over 400 tasks) |
| Routing correctness gain (VCG) | **+43.3pp** vs no routing (p = 0.0003, d = 1.02) |
| Mismatched routing harm | −17.5% correctness, Brier 0.292 vs 0.160 |
| U ↔ correctness correlation | Pearson r = 0.461, p < 10⁻⁴⁰ |
| Brier calibration improvement | 14.3% overall, 29.5% by cycle 5 |
| Contradiction rate reduction | 22% → 6% over 10 cycles (73%) |

Full record: [`docs/v1_validation_report.md`](docs/v1_validation_report.md)

---

## Roadmap

Tracked in full at [aua_roadmap.html](https://praneethtota.github.io/Adaptive-Utility-Agent/aua_roadmap.html).

Recent completions (#37–#55 block):

| # | Feature | Status |
|---|---|---|
| #37 | Circuit breaker per specialist | ✅ v1.2 |
| #38 | Degraded-mode failover | ✅ v1.2 |
| #39 | Retry with exponential backoff | ✅ v1.2 |
| #44 | Multi-tenancy | ✅ v1.2 |
| #46 | Model registry + version pinning | ✅ v1.2 |
| #47 | Experiment tracking (MLflow, W&B) | ✅ v1.2 |
| #48 | Shadow mode | ✅ v1.2 |
| #49 | Regression gate | ✅ v1.2 |
| #50 | `aua loadtest` | ✅ v1.2 |
| #51 | Extended plugin system (4 new types) | ✅ v1.2 |
| #52 | Extended middleware (on_chunk, batch, error) | ✅ v1.2 |
| #53 | `full_utility_scorer` — non-linear utility | ✅ v1.2 |
| #54 | `aua test` — built-in suites | ✅ v1.2 |
| #55 | Compatibility matrix | ✅ v1.2 |
| #74 | Per-specialist `model_backend` dispatch | ⏳ planned |
| #75 | `state_store` plugin wiring | ⏳ planned |

---

## License

**Code:** GNU General Public License v3.0 — see `LICENSE`  
**Whitepaper:** Creative Commons Attribution 4.0 — see `LICENSE-CC-BY-4.0`

If you build on this work, please cite:
> Tota, P. (2026). *AUA Framework v1.2: A Production Framework for Self-Correcting Multi-Specialist AI Systems*. GitHub. https://github.com/praneethtota/Adaptive-Utility-Agent

---

📖 **Full documentation, tutorial, and domain deep-dives:**  
**https://praneethtota.github.io/Adaptive-Utility-Agent**
