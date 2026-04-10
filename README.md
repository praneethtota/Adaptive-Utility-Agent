# Adaptive Utility Agents

> **The central failure mode of deployed language models is error repetition. This project builds AI agents that actively work against it — detecting errors, correcting behavior, and not repeating mistakes between model releases.**

---

## License

**Code:** GNU General Public License v3.0 — see `LICENSE`  
**Whitepaper:** Creative Commons Attribution 4.0 — see `LICENSE-CC-BY-4.0`

If you build on this work, please cite:
> Tota, P. (2026). *Adaptive Utility Agents: A Framework for Self-Optimizing AI Systems* (v0.5). GitHub. https://github.com/praneethtota/Adaptive-Utility-Agent

---

## The Problem

Deployed AI systems are static artifacts. A model that hallucinates today will hallucinate the same thing tomorrow, and every day until the next version ships — which may be months away. There is no feedback loop between detected errors and model behavior in the space between versions.

This project addresses that structural absence. The goal is **online learning and error non-repetition**: an agent that detects its own errors, adjusts behavior in response, and does not repeat those errors — continuously, between releases, without a new training cycle.

The work is grounded in multi-attribute utility theory from economics, extended by treating utility as a control signal in a feedback system rather than a static objective. It draws on mechanism design — specifically the Vickrey-Clarke-Groves (VCG) mechanism — for arbitration and incentive alignment across model components, and on Kalman filtering, Lyapunov stability analysis, and the Mann-Whitney dominance statistic for the formal foundations of each utility component.

---

## The Core Mechanism: Utility as a Control Law

```
U = w_e(f) · E + w_c(f) · C + w_k(f) · K

E — Efficacy:    performance relative to human baseline       [0, 1]
C — Confidence:  internal consistency, penalized by contradictions
K — Curiosity:   exploration bonus for high-upside uncertain domains
f — field (surgery, law, software, creative, ...)
```

The utility function is **not a monitoring metric**. It is the governing control law over the agent's behavior at every timescale:

- **At training time**: field penalty multipliers are DPO loss weights — a surgical contradiction is penalized 10× harder than a creative writing mistake at the weight-update level
- **During deployment**: utility deviation triggers behavioral corrections and controls whether a new model version is accepted
- **Across calibration cycles**: utility score determines which interactions generate DPO training pairs and how strongly each pair is weighted

The additive weighted structure is not a convenience — it is the unique functional form satisfying five behavioral axioms (monotonicity, continuity, separability, field invariance, linear scaling invariance). Proved from first principles via Debreu's representation theorem and the Cauchy functional equation, using continuity only — no differentiability required (Theorem B.1, Appendix B).

| Term | Name | Formal grounding |
|---|---|---|
| **E** | Efficacy | Mann-Whitney dominance probability under log-logistic model (Proposition B.3) |
| **C** | Confidence | Kalman-optimal EMA estimator for ρ = 0.05 noise ratio; geometric convergence with noise floor (Theorems B.4, B.5) |
| **K** | Curiosity | UCB-inspired exploration bonus; 50% cap enforces exploitation dominance (Proposition B.6) |

Field weights and minimum competence bounds are derived from existing societal licensing standards — medical malpractice thresholds, ICAO Annex 13 aviation certification, ISO 26262 safety classifications — making them principled rather than arbitrary.

---

## Architecture

### Monolithic Setting (Current)

Until the Micro-Expert Architecture is operational, the system wraps a monolithic base model. Three layers compensate for the constraints of a monolithic system:

```
Layer 1 — Per-session behavioral injection     (real-time, no weight change)
  Detected contradictions → corrective assertions → system prompt

Layer 2 — Calibration-cycle DPO fine-tuning   (several times daily)
  Utility-scored pairs → field-penalty-weighted DPO loss → LoRA update

Layer 3 — Release-level distillation          (monthly)
  Accumulated adapters → distilled into new base fine-tune
```

**Personality System (interim wrapper):** Between calibration cycles, a behavioral wrapper biases generation toward safer operating regimes. Formally: a log-linear tilt of the base model's output distribution parameterized by field-bounded trait scores (curiosity, caution, assertiveness, analytical_rigor, creativity). At the field-neutral point the wrapper is the identity — no effect on generation. Lyapunov-stable dynamics with half-life ≈ 34 cycles under zero drift (Theorem B.7). Resets on new model release; not instantiated in the Micro-Expert Architecture.

### Micro-Expert Architecture (Target)

The monolithic model is decomposed into independently deployable domain submodels — microservices architecture applied to model inference:

```
Router (Raft HA cluster, 150–300ms failover)
    ↓  probabilistic field classification + fan-out
Domain Submodels (surgery | law | software | creative | ...)
    ↓  independent weights, training, deployment
Arbiter Agent (§9.5) + VCG Mechanism (§9.6)
    ↓  cross-domain contradiction resolution
Blue-Green Deployment (§9.7)
    ↓  utility-deviation-triggered, softmax traffic routing
```

Updating surgery weights cannot affect software engineering weights. There are no shared parameters to interfere. Catastrophic forgetting is resolved architecturally. Graph depth is hardware-adaptive: high-VRAM GPUs run shallow graphs of large models; consumer GPUs run deeper graphs of smaller specialists at lower cost per query.

### Arbiter Agent (§9.5)

When two submodels produce conflicting outputs, a dedicated Arbiter Agent runs structured evidence checks:

| Check | Weight | What it tests |
|---|---|---|
| Logical | 0.30 | Does the output contradict its own premises? |
| Mathematical | 0.40 | Are complexity or numerical claims provably wrong? |
| Cross-session | 0.20 | Does it contradict prior verified assertions? |
| Empirical | 0.10 | Does it contradict verifiable external ground truth? |

Four verdict cases: A correct → correct B; B correct → correct A; both wrong → correct both + curiosity gap bonus; inconclusive → controlled external escalation under minimum-disclosure protocol. Corrections route internally as DPO signal. Nothing is disclosed externally beyond the verified answer, or a minimal hedge on inconclusive cases.

Arbiter calibration: 2–5% of verdicts independently verified against domain experts. Escalates adaptively to a 15% hard ceiling if correction volume rises above baseline.

### VCG Arbitration Mechanism (§9.6)

The hand-specified Arbiter check weights are an engineering approximation. The theoretically grounded alternative treats domain submodels as players in a cooperative game:

**Three theorems proved (§9.6):**

| Theorem | Statement |
|---|---|
| **S1 — Dominant Strategy Truthfulness** | Truthful reporting of $v_i$ is a weakly dominant strategy for every submodel, regardless of others' reports |
| **S2 — Social Optimum (POA = 1)** | Under dominant-strategy equilibrium the Arbiter selects the claim maximising $\sum_i v_i(a)$; Price of Anarchy = 1 exactly |
| **S3 — Individual Rationality** | Every submodel weakly prefers participation to abstention |

Clarke pivot transfers applied as DPO penalty weight adjustments make check weights endogenous and replace the periodic expert-sampling audit with a continuous self-correcting signal.

### Assertions Store (Evidence with Decay)

Verified facts persist across sessions with field-specific confidence decay:

| Class | Decay | Examples |
|---|---|---|
| A — No decay | Never | Mathematical proofs, physical laws, algorithm correctness |
| B — Slow (τ = 10yr) | Exponential | Mechanical engineering, classical physics |
| C — Moderate (τ = 3yr) | Exponential | Medical anatomy, legal common law |
| D — Fast (τ = 6mo) | Exponential | Clinical guidelines, security practices, ML benchmarks |

---

## The Consumer Hardware Argument (§9.9)

This is one of the more consequential implications of the Micro-Expert Architecture, and one the paper is careful to state with appropriate scope.

### The claim

The dominant assumption in AI deployment is that frontier capability requires frontier compute — specifically, the high-bandwidth GPU clusters subject to export controls. The Micro-Expert Architecture challenges this assumption in a specific and falsifiable way.

**The claim is not** that consumer GPUs match H100s on general workloads. They do not — H100s have 3× the memory bandwidth and NVLink interconnects that PCIe cannot approach.

**The claim is** that for inference on specialised domain queries — the highest-value AI use cases for most professional organisations — a graph of domain-specialist models on consumer hardware can match the output quality of a monolithic frontier model on enterprise hardware, at substantially lower cost per query. The routing and arbitration layer that makes this possible is what §9.9 formalises and partially validates.

### The cost arithmetic (from public hardware specs)

```
7B specialist on RTX 4090:  ~$0.00014 per 1K tokens
70B model on 2× H100:       ~$0.00083 per 1K tokens

Single-specialist query:     6× cheaper on consumer hardware
3-specialist fan-out:        2× cheaper even at maximum typical fan-out
```

### The routing experiment (§9.9.4)

A four-arm controlled study using the production agent codebase measured the contribution of the routing and arbitration layer to correctness, independently of model size or hardware. Quality parameters were derived from six published domain benchmarks (all cited in `routing_results.json`).

| Arm | Correctness | vs baseline | Brier | p-value |
|---|---|---|---|---|
| A — No routing (generic prompt) | 59.0% | — | 0.160 | — |
| B — Matched routing (oracle) | 71.5% | **+12.5%** | 0.106 | 0.009 |
| C — Mismatched routing (Regime 2) | 41.5% | **−17.5%** | 0.292 | <0.001 |
| D — VCG arbitration | 69.5% | **+10.5%** | 0.110 | 0.029 |

Three findings:

1. **Correct routing contributes +12.5% correctness** (p = 0.009) through prompt specialisation alone — before any weight-level fine-tuning. This is the routing layer's direct contribution, measurable independently of hardware.

2. **Mismatched routing is actively harmful** (−17.5%, p < 0.001) and dramatically worsens confidence calibration (Brier 0.292 vs 0.160). The model is not just wrong — it is confidently wrong. This quantifies the Regime 2 failure mode from §9.4.1 and makes the case for probabilistic routing and VCG arbitration concrete rather than theoretical.

3. **VCG arbitration captures 84% of the oracle matched-routing gain** (+10.5% vs +12.5%), statistically significant (p = 0.029), with near-matched Brier score. The 2.0pp gap to the oracle is not statistically significant (p = 0.66) — at 82% routing accuracy, VCG arbitration essentially closes on the oracle best case.

```bash
cd agent && python3 routing_experiment.py
# Outputs: routing_output/routing_results.json, routing_report.txt, plots/ (4 figures)
# Replace _generate_response() with live_generate_response() for Ollama inference
```

### The complete argument (stated scope)

The consumer hardware case combines three components with different evidential status:

| Component | Evidence | Source |
|---|---|---|
| Routing + arbitration adds +10.5% correctness | **Measured** (this work, statistically significant) | `routing_experiment.py` |
| Domain-specialist 7B models match general 70B on domain benchmarks | **Published** (independently replicated) | DeepSeek Coder, WizardMath, Med-PaLM citations |
| 2–6× lower cost per query on consumer hardware | **Analytical** (public hardware specs and cloud pricing) | Lambda Labs, RunPod, NVIDIA specs |

Together these form a complete argument. The third component — actual quality benchmarking of fine-tuned 7B specialists against Llama 3.1 70B on physical 4090 hardware — is the primary item of empirical future work and requires only consumer hardware to run.

---

## Implications for the AI Landscape

### The hardware moat is narrower than assumed for professional domains

Export controls on H100s, A100s, and their successors rest on a single architectural assumption: that frontier AI capability requires frontier compute. This assumption is well-founded for training and for general-purpose inference at scale. It is considerably weaker for the domain-specific professional inference use cases — medicine, law, engineering, software, mathematics — where AI has the clearest near-term value.

The published benchmark evidence is consistent and replicated across multiple independent groups: fine-tuned 7B–13B domain specialists routinely match or exceed general 70B models on their target domain benchmarks. This is not a marginal effect. WizardMath 7B achieves 54.9% on MATH versus 13.5% for Llama 2 70B. Med-PaLM 2 matches GPT-4 on MedQA despite being orders of magnitude smaller. DeepSeek Coder 7B matches GPT-3.5 175B on HumanEval.

The Micro-Expert Architecture makes this practically deployable: a router that activates the right specialist for each query, an Arbiter that resolves cross-domain conflicts, and a utility-weighted calibration loop that improves over time — running on consumer hardware, without export-controlled components.

### What this means for compute sovereignty

Countries and organisations operating without access to H100 clusters are not locked out of frontier AI capability in the domains that matter most for economic and scientific development. They face a different engineering challenge: building a routed graph of domain specialists rather than scaling a monolithic model. This paper is one piece of the technical foundation for that approach.

The critical caveat, stated explicitly throughout §9.9: general-purpose AI capability — the open-ended reasoning and knowledge breadth that frontier models provide on arbitrary queries — does retain a meaningful hardware advantage. The consumer hardware argument applies to the specialised slice, not the general case. That slice is, however, the commercially and professionally most important one.

### The routing failure modes matter as much as the architecture

The export control implication is only as strong as the routing is reliable. The Regime 2 result (−17.5% correctness, Brier 0.292) shows that wrong-domain routing is not merely suboptimal — it actively makes the system worse than no routing at all, and does so confidently. This is why the routing problem (§9.4.1) and its mitigations (probabilistic fan-out, VCG calibration, M1–M5) are central to the paper and not peripheral engineering details. A Micro-Expert system with poor routing is worse than a monolithic model. A Micro-Expert system with good routing and proper arbitration is competitive with a much larger model on domain tasks, on consumer hardware.

---

## Mathematical Foundations (Appendix B, v0.5)

All proofs use only continuity where differentiability is not assumed; all scope conditions are stated explicitly.

| Result | Content | Key note |
|---|---|---|
| **Theorem B.1** | Additive linear structure of U uniquely necessary from five axioms | Proved via Debreu + Cauchy functional equation; continuity only, no differentiability |
| **§B.2** | Field weights from error-cost proportionality, calibrated to liability standards | Design principle, not an optimality theorem |
| **Proposition B.3** | Efficacy sigmoid = Mann-Whitney dominance probability | Holds under log-logistic model with equal scale; distributional assumption stated |
| **Theorem B.4** | EMA with α = 0.2 is Kalman-optimal for ρ = 0.05 noise ratio | Reasoning direction clarified: α = 0.2 was chosen first, Kalman characterises the noise regime |
| **Theorem B.5** | Confidence convergence with noise-aware bound | $\mathbb{E}[|C_t - C^*|] \leq (1-\alpha)^t|C_0 - C^*| + \sigma_{\tilde{s}}\sqrt{\alpha/(2-\alpha)}$; requires $\lambda\mu(f) < 1$ |
| **Proposition B.6** | 50% curiosity cap enforces exploitation dominance | Proved exactly; regret analysis open |
| **Theorem B.7** | Personality Lyapunov stability | Part (iv) clarified: mean reversion β = 0.01 subsumed by field bounds at current parameters |

---

## Simulation Results

### Extended simulation (Appendix A) — 500-task two-arm + 10-cycle stability

```
Cycle  Agent U   Base U   Ag Brier  Bl Brier  Ag Rep↑  Bl Rep↑
─────  ────────  ───────  ────────  ────────  ───────  ───────
  1    0.5291    0.5333   0.3279    0.3502      0        0
  2    0.5441    0.5385   0.2177    0.2520      1        6
  3    0.5656    0.5604   0.2464    0.2860      4       10
  4    0.5828    0.5622   0.2149    0.2601      3       15
  5    0.5846    0.5765   0.1059    0.1501      6       15
```

**69.6% reduction in repeated errors** over uncalibrated baseline (14 vs 46, cycles 2–5).  
**14.3% Brier improvement** overall; 29.5% by cycle 5.  
**Pearson r = 0.461** (U vs correctness, p < 10⁻⁴⁰) — U is a statistically significant correctness predictor.

10-cycle stability: contradiction rate 22% → 6% (73% reduction); Brier reaches 0.049 by cycle 7.

### Routing experiment (§9.9) — four-arm study

| Arm | Correctness | Δ vs baseline | Brier | p-value |
|---|---|---|---|---|
| A — No routing | 59.0% | — | 0.160 | — |
| B — Matched (oracle) | 71.5% | +12.5% | 0.106 | 0.009 |
| C — Mismatched (Regime 2) | 41.5% | −17.5% | 0.292 | <0.001 |
| D — VCG arbitration | 69.5% | +10.5% | 0.110 | 0.029 |

---

## Validated Claims

| Claim | Result | Status |
|---|---|---|
| Agent reduces repeated errors vs uncalibrated baseline | 69.6% reduction (14 vs 46 over 400 tasks) | **Confirmed** |
| U correlates with ground-truth correctness | Pearson r = 0.461 (agent), p < 10⁻⁴⁰ | **Confirmed** |
| Confidence is better calibrated under agent vs baseline | Brier 0.2226 vs 0.2597 (14.3% improvement) | **Confirmed** |
| Personality converges stably (Theorem B.7) | Traits in field bounds throughout; dynamics match theorem | **Confirmed** |
| Contradiction rate falls with sustained calibration | 22% → 6% over 10 cycles (73% reduction) | **Confirmed** |
| Long-tail errors persist beyond five correction cycles | 8 patterns; root cause: surface-form variability in assertions store | **Confirmed — limitation identified** |
| Correct routing improves correctness vs no routing | +12.5% (p = 0.009, Cohen's d = 0.265) | **Confirmed** |
| Mismatched routing is actively harmful | −17.5% correctness, Brier 0.292 vs 0.160 (p < 0.001) | **Confirmed** |
| VCG arbitration captures most of the routing gain | +10.5% (84% of oracle), p = 0.029 | **Confirmed** |
| Consumer hardware cost advantage | 2–6× lower cost per token (analytical, from public specs) | **Analytical — empirical validation pending** |

---

## Project Structure

```
agent/
├── config.py                  # Field weights, bounds, penalty multipliers
├── field_classifier.py        # Field distribution: high-stakes floor, EMA drift, entropy fallback
├── contradiction_detector.py  # Logical, mathematical, cross-session detection
├── assertions_store.py        # Cross-session store with decay classes A–D
├── trust_manager.py           # Credential bootstrapping, tit-for-tat scoring
├── arbiter.py                 # 4-check pipeline, gap bonus, adaptive sampling
├── utility_scorer.py          # E (EMA), C, K (50% cap), difficulty routing
├── personality_manager.py     # Wrapper evolution, Lyapunov-stable dynamics
├── creative_efficacy.py       # Two-component creative efficacy model
├── agent.py                   # Main UtilityAgent — wires all components
├── harness.py                 # Live API harness (requires ANTHROPIC_API_KEY)
├── simulate.py                # Original 3-cycle / 8-problem simulation
├── simulate_extended.py       # Extended simulation: 500-task two-arm + 10-cycle stability
├── routing_experiment.py      # Four-arm routing quality study (§9.9)
├── requirements.txt
├── extended_output/
│   ├── extended_results.json  # Full raw data (task records, cycle stats, DPO pairs)
│   ├── report.txt
│   └── plots/                 # 10 publication figures (PNG, 150 dpi)
└── routing_output/
    ├── routing_results.json   # Four-arm results with benchmark citations
    ├── routing_report.txt
    └── plots/                 # 4 routing experiment figures (PNG, 150 dpi)
        ├── figR1_correctness.png
        ├── figR2_brier.png
        ├── figR3_domain_heatmap.png
        └── figR4_summary.png

whitepaper/
├── adaptive_utility_agents_v05_combined.html  # Full paper: HTML with KaTeX math + embedded figures
├── adaptive_utility_agents_v05_combined.md    # Markdown edition (figures omitted)
└── supplement_s1_vcg_arbitration.html         # Original VCG supplement (now integrated as §9.6)

docs/
└── to_do_in_version_v06_revised.md           # v0.6 backend design: privacy-first MVP spec
```

---

## Quick Start

```bash
# Original simulation — no API key needed
cd agent && python3 simulate.py

# Extended simulation — generates all results and plots
cd agent && python3 simulate_extended.py

# Routing quality experiment (§9.9)
cd agent && python3 routing_experiment.py
# For live Ollama inference: replace _generate_response() with live_generate_response()
# Instructions in routing_experiment.py module docstring

# Live harness — requires API key
pip install httpx
export ANTHROPIC_API_KEY=sk-ant-...
cd agent && python3 harness.py
```

Dependencies: `numpy`, `scipy`, `matplotlib` (standard scientific Python stack). No GPU required for any simulation.

---

## What's New in v0.5

### Theoretical additions

- **VCG arbitration mechanism (§9.6)**: Theorems S1–S3 prove dominant-strategy truthfulness, social optimality (POA = 1), and individual rationality. Clarke pivot transfers replace hand-specified check weights and the expert-sampling audit with a continuous self-correcting signal.

- **Appendix B — complete formal proofs (B.1–B.7)**: Key corrections: B.1 uses Cauchy functional equation (continuity only, no differentiability); B.5 noise-aware bound matches proof; B.7 Part (iv) clarified (β = 0.01 subsumed by field bounds); B.4 sensitivity table corrected.

- **§9.9 — Consumer hardware argument**: Analytical cost model (2–6× cheaper per token), routing quality experiment (+10.5% correctness from VCG arbitration, p = 0.029), and explicit scope statement distinguishing measured from analytical claims.

### Empirical additions

- **Extended simulation (Appendix A)**: 500-task two-arm comparison + 10-cycle stability run. 69.6% repeated-error reduction. Full data in `extended_results.json`.

- **Routing quality experiment (§9.9)**: Four-arm study quantifying the routing layer's contribution (+12.5% oracle, +10.5% VCG, −17.5% Regime 2). Quality model from published benchmarks; code structured for live Ollama drop-in. Data in `routing_results.json`.

### Structural additions

- Supplement S1 integrated as §9.6; sections renumbered to §§9.7–9.10
- References merged: Clarke, Groves, Harsanyi/Selten, Hurwicz, Nash, Vickrey added
- Validated claims table expanded from 6 to 10 claims

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Code generation MVP — single domain, validate U correlates with quality | Simulated ✓ |
| 2 | Multi-domain STEM — math proof verification (Lean/SymPy), field classifier | Planned |
| 3 | Personality system — trait weighting and evolution service | Simulated ✓ |
| 4 | Trust system — entity scoring and lenient tit-for-tat | Implemented |
| 5 | Creative fields — platform signal collection, two-component efficacy | Designed |
| 6 | Full continual learning — LoRA calibration in production, replay buffer | Planned |
| 7 | Feedback into training — distill adapters into base fine-tune | Planned |
| 7b | **Consumer hardware validation** — LoRA-adapted 7B specialists on 4× RTX 4090 vs Llama 3.1 70B on H100; latency and quality benchmarking under PCIe vs NVLink | **Next empirical priority** |
| **v0.6** | **Privacy-first backend MVP** — localhost correction memory, canonical query normalizer, domain-gated retry loop, context grammar, opt-in cross-user sharing | **In design** |

**Phase 7b** is the experiment that turns the consumer hardware argument from analytical to empirical. It requires only consumer hardware (4× RTX 4090, ~$1,600 on the used market or ~$1.60/hr on RunPod), domain-specific fine-tuning datasets (open source), and the existing routing codebase. The experimental design is fully specified in §9.9 of the whitepaper.

**v0.6 design** is in `docs/to_do_in_version_v06_revised.md`.

---

## Status

Active research project at v0.5. Three categories of claims are now validated at different evidential levels:

- **Measured** (this work): 69.6% repeated-error reduction, Brier calibration improvement, U↔correctness correlation, +10.5% correctness from VCG arbitration, −17.5% from Regime 2 routing failure
- **Analytical** (from public specs and published benchmarks): consumer hardware cost model, specialist quality gains
- **Pending empirical validation**: physical hardware comparison of 7B specialist graph vs 70B monolithic model

The gap between the second and third categories — turning the analytical consumer hardware claim into a measured one — is the clearest and most impactful next step, and one that requires only consumer hardware to close. Contributions and collaboration welcome.
