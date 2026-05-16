# AUA Framework

> A production framework for self-correcting, multi-specialist LLM systems.

**Full site:** https://praneethtota.github.io/Adaptive-Utility-Agent

Recommended way to explore the project:** Start with the overview, then read the whitepaper, roadmap, and tutorial for implementation details.

---

## What it does

AUA sits between your application and your language models. It routes prompts to specialist models, scores responses with a utility function, catches contradictions, injects prior verified corrections into future queries, and enforces policies in real-time.

The core idea: a model that makes a wrong answer on Tuesday should not make the same wrong answer on Thursday. AUA closes that loop without waiting for a new model release.

```bash
pip install adaptive-utility-agent
aua init my-project --preset coding --tier macbook
cd my-project && aua serve
```

---
## Sister Project: AUA Veritas

I am currently building a standalone app based on this framework called **AUA Veritas**.

AUA Framework is intended for developers, MLEs, and AI infrastructure teams who want to build adaptive multi-model LLM systems with routing, utility scoring, arbitration, correction loops, observability, and deployment controls.

**AUA Veritas** applies those ideas in a consumer-facing desktop app. Instead of exposing framework internals, it gives everyday AI users a simple interface: ask a question, let Veritas compare multiple frontier models, remember prior corrections, and return one answer with a confidence signal.

The sister repo is here:

👉 [AUA Veritas](https://github.com/praneethtota/AUA-Veritas)


## Documentation


| Page | Audience | Link |
|---|---|---|
| **Landing page** | Everyone | [whitepaper.html](https://praneethtota.github.io/Adaptive-Utility-Agent/whitepaper.html) |
| **Tutorial** | ML engineers, builders | [tutorial.html](https://praneethtota.github.io/Adaptive-Utility-Agent/tutorial.html) |
| **Production architecture** | DevOps, platform engineers | [productionizing.html](https://praneethtota.github.io/Adaptive-Utility-Agent/productionizing.html) |
| **Whitepaper** (7 parts) | Researchers, theorists | [whitepaper_overview.html](https://praneethtota.github.io/Adaptive-Utility-Agent/whitepaper_overview.html) |
| **Roadmap & validation** | Everyone | [aua_roadmap.html](https://praneethtota.github.io/Adaptive-Utility-Agent/aua_roadmap.html) |
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
aua doctor       # pre-flight check
aua serve        # start specialists + router on :8000
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
print(f"U={result.u_score:.3f}  mode={result.routing_mode}")
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

| Tier | Hardware | Backend | Notes |
|---|---|---|---|
| `macbook` | Apple M-series / Intel | Ollama | Install: `brew install ollama` |
| `single-4090` | 1× RTX 4090 24 GB | vLLM AWQ | |
| `quad-4090` | 4× RTX 4090 | vLLM AWQ | Dedicated GPU per specialist |
| `a100-cluster` | 1× A100 80 GB | vLLM fp16 | No quantization |

Aliases: `rtx4090` → `single-4090`, `a100` → `a100-cluster`.

---

## What ships in v1.0

| Component | Detail |
|---|---|
| **REST API** | 23 endpoints — query, stream, batch, corrections, config, deploy, status, sessions, metrics |
| **CLI** | 22 command groups — `aua init`, `aua serve`, `aua doctor`, `aua status`, `aua eval`, `aua guard`, `aua policy`, `aua calibrate`, `aua logs`, `aua metrics`, and more |
| **Plugin interfaces** | 8 Protocol interfaces — FieldClassifier, UtilityScorer, ArbiterPolicy, PromotionPolicy, CorrectionStore, ModelBackend, StateStore, HookPlugin |
| **Hooks** | 11 lifecycle hook points — `pre_query`, `post_route`, `pre_specialist_call`, `post_specialist_call`, `pre_arbiter`, `post_arbiter`, `on_correction`, `pre_response`, `post_response`, `on_promotion`, `on_rollback` |
| **Middleware** | `AUAMiddleware` — `before_query` / `after_response` wraps every request |
| **Assertions + Policy** | `@assertion` decorator, `AssertionLevel` (BLOCKING/SOFT/INFO), `Policy` with YAML config, Option B E-bonus, gold-standard DPO session detection |
| **Calibration** | `aua calibrate --layer 1/2/3` — eval harness, routing weight analysis, DPO pair export |
| **Prometheus metrics** | 18 metrics including assertion fail rate, E-bonus histogram, retry counter |
| **Observability** | Structured JSON logs, Prometheus/Grafana, OpenTelemetry traces, ELK/Splunk-compatible |
| **Chat UI** | Next.js 14, three-panel layout: sidebar, chat, Framework Debugger |
| **Blue-green deployment** | Utility-deviation-triggered promotion, `aua rollback` |
| **Test suite** | 197 tests, Python 3.10 / 3.11 / 3.12 matrix |

---

## The utility function

```
U = w_e(f) · E  +  w_c(f) · C  +  w_k(f) · K

E — Efficacy:    how well the response serves the domain objective      [0, 1]
C — Confidence:  Kalman-filtered internal consistency                   [0, 1]
K — Curiosity:   UCB-style exploration bonus, capped at 50% of E+C
f — field        (software_engineering, mathematics, general, ...)
```

The additive weighted structure is not a convenience — it is the unique functional form satisfying five behavioral axioms, proved via Debreu's representation theorem (Theorem B.1, [Appendix B](https://praneethtota.github.io/Adaptive-Utility-Agent/whitepaper_appendix_b.html)).

---

## Policies — teaching the framework what good looks like

```python
from aua.guard import assertion, AssertionLevel
from aua.policy import Policy

@assertion(name="PythonSyntaxCheck", level=AssertionLevel.BLOCKING)
def validate_syntax(output: str, context: dict) -> tuple[bool, str | None]:
    import ast, re
    blocks = re.findall(r"```python(.*?)```", output, re.DOTALL)
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}"
    return True, None

@assertion(name="AnalogyBonus", level=AssertionLevel.INFO, bonus=0.10)
def reward_analogy(output: str, context: dict) -> tuple[bool, str | None]:
    if any(p in output.lower() for p in ["like a", "similar to", "imagine"]):
        return True, "Positive: analogy used"
    return True, None  # neutral — no bonus

policy = Policy(name="SafeCoding", max_total_bonus=0.30)
policy.add(validate_syntax)   # BLOCKING — retries on fail
policy.add(reward_analogy)    # INFO — boosts E score
```

YAML equivalent in `policies/safe_coding.yaml`:

```yaml
name: SafeCoding
version: "1.0"
max_retries: 3
max_total_bonus: 0.30
assertions:
  - import_path: mypackage.policies:validate_syntax
  - import_path: mypackage.policies:reward_analogy
    bonus: 0.10
utility_overrides:
  w_k: 0.30
```

```bash
aua policy validate policies/safe_coding.yaml
aua policy apply policies/safe_coding.yaml
```

Over time: BLOCKING assertions reduce failures → sessions that pass become gold-standard DPO data → `aua calibrate --layer 3` exports them → fine-tune → repeat.

---

## Hooks

```python
class SlackOnCorrection:
    async def __call__(self, event: dict) -> dict:
        # event["type"] == "on_correction"
        # event keys: subject, domain, claim, confidence, decay_class, source
        await notify_slack(f"New correction: {event['claim']}")
        return event
```

```yaml
hooks:
  on_correction:
    - import_path: plugins.hooks:SlackOnCorrection
      fail_closed: false
      timeout_s: 3.0
```

All 11 hook points, event dict schemas, and examples are in [tutorial.html Part 14](https://praneethtota.github.io/Adaptive-Utility-Agent/tutorial.html#part-14).

---

## Project structure

```
aua/                        # Core framework package
├── router.py               # Request routing + REST endpoints
├── arbiter.py              # Contradiction detection + 4-check arbitration
├── utility_scorer.py       # U = w_e·E + w_c·C + w_k·K
├── field_classifier.py     # Probabilistic domain routing
├── assertions_store.py     # Cross-session corrections with decay classes A–D
├── correction_loop.py      # DPO pair accumulation
├── blue_green.py           # Utility-deviation-triggered model promotion
├── rollback.py             # Model rollback with event log
├── guard.py                # @assertion decorator, AssertionLevel, Policy.run()
├── policy.py               # Policy dataclass + YAML loader
├── hooks.py                # HookRunner — 11 lifecycle hook points
├── auth.py                 # 15-scope token auth + mTLS
├── metrics.py              # 18 Prometheus metrics
├── otel.py                 # OpenTelemetry tracing
├── eval.py                 # Evaluation harness
├── chat.py                 # Chat session management
├── state.py                # SQLite state store (sessions, corrections, assertion_events)
├── cli.py                  # aua CLI — 22 command groups
├── config.py               # AUAConfig + tier loader
└── plugins/
    ├── interfaces.py       # 8 Protocol interfaces
    └── registry.py         # Plugin load + validation

apps/
└── aua_chat/               # Next.js 14 Chat UI (npm run dev or aua ui)

tests/                      # 197 tests across Python 3.10 / 3.11 / 3.12
docs/
├── v1_validation_report.md # Full validation record
└── archive/                # v0.5 pages (preserved)
```

---

## Validated results (v1.0, RTX 4090)

| Result | Value | Source |
|---|---|---|
| Repeated error reduction | **69.6%** (14 vs 46 over 400 tasks) | `agent/simulate_extended.py` |
| Routing correctness gain (VCG arbitration) | **+43.3pp** vs no routing (p = 0.0003, d = 1.02) | `agent/routing_experiment.py` |
| Mismatched routing harm | **−17.5%** correctness, Brier 0.292 vs 0.160 | Same |
| U ↔ correctness correlation | Pearson r = 0.461, p < 10⁻⁴⁰ | Extended simulation |
| Brier calibration improvement | 14.3% overall, 29.5% by cycle 5 | Extended simulation |
| Contradiction rate reduction | 22% → 6% over 10 cycles (73%) | Extended simulation |

Full validation record with all 197 test names, CLI reference, Docker Compose, Chat UI, security, and observability validation: [`docs/v1_validation_report.md`](docs/v1_validation_report.md).

---

## Roadmap

| Item | Status |
|---|---|
| Per-user correction scoping (multi-tenant isolation) | v1.1 |
| Full chosen+rejected DPO pair generation (auto populated) | v1.1 |
| Physical hardware comparison: 7B specialist graph vs 70B monolithic | Empirical priority |
| Safety-critical deployment validation (shadow-mode, abstention testing) | Planned |
| Regex + LLM-judge eval check types | v1.1 |
| Policy version history and rollback | v1.1 |
| Automatic fine-tuning pipeline (Axolotl/TRL integration) | v2.0 |

The v1.1 roadmap is also tracked in [aua_roadmap.html](https://praneethtota.github.io/Adaptive-Utility-Agent/aua_roadmap.html).

---

## The core mechanism: utility as a control law

The utility function governs behavior at every timescale:

- **At query time**: routes to the right specialist, scores the response, enforces assertions, injects prior corrections
- **Session-by-session**: specialists that consistently fail assertions don't get promoted via blue-green
- **Calibration cycles**: `aua calibrate --layer 3` exports gold-standard sessions (all INFO assertions fired, no BLOCKING failed) as DPO training pairs

The additive weighted structure is not a convenience — it is the unique functional form satisfying five behavioral axioms (monotonicity, continuity, separability, field invariance, linear scaling invariance), proved from first principles via Debreu's representation theorem and the Cauchy functional equation ([Theorem B.1](https://praneethtota.github.io/Adaptive-Utility-Agent/whitepaper_appendix_b.html)).

---

## License

**Code:** GNU General Public License v3.0 — see `LICENSE`  
**Whitepaper:** Creative Commons Attribution 4.0 — see `LICENSE-CC-BY-4.0`

If you build on this work, please cite:
> Tota, P. (2026). *AUA Framework v1.0: A Production Framework for Self-Correcting Multi-Specialist AI Systems*. GitHub. https://github.com/praneethtota/Adaptive-Utility-Agent

---

📖 **Full documentation, tutorial, and domain deep-dives:**  
**https://praneethtota.github.io/Adaptive-Utility-Agent**
