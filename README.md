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

- **At training time**: field penalty multipliers are DPO loss weights — a surgical contradiction penalized 10× harder than a creative writing mistake
- **During deployment**: utility deviation triggers behavioral corrections and controls when a new model version is accepted
- **Across calibration cycles**: utility score determines which interactions generate DPO training pairs and how strongly each is weighted

The additive weighted structure is not a convenience — it is the unique functional form satisfying five behavioral axioms (monotonicity, continuity, separability, field invariance, linear scaling invariance). Proved in Appendix B, Theorem B.1.

| Term | Name | Formal grounding |
|---|---|---|
| **E** | Efficacy | Mann-Whitney dominance probability under log-logistic model (Proposition B.3) |
| **C** | Confidence | Kalman-optimal EMA estimator; converges geometrically in expectation (Theorems B.4, B.5) |
| **K** | Curiosity | UCB-inspired exploration bonus; 50% cap enforces exploitation dominance (Proposition B.6) |

Field weights and minimum bounds are derived from existing societal licensing standards — medical malpractice thresholds, aviation certification, engineering safety — making them principled rather than arbitrary.

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

**Personality System (interim wrapper):** Between calibration cycles, a behavioral wrapper biases generation toward safer operating regimes. Formally: a log-linear tilt of the base model's output distribution parameterized by field-bounded trait scores (curiosity, caution, assertiveness, analytical_rigor, creativity). At the field-neutral point the wrapper is the identity — no effect. It resets on new model release and is not instantiated in the Micro-Expert Architecture.

### Micro-Expert Architecture (Target)

The monolithic model is decomposed into independently deployable domain submodels — microservices architecture applied to model inference:

```
Router (Raft HA cluster, 150–300ms failover)
    ↓  field classification + fan-out
Domain Submodels (surgery | law | software | creative | ...)
    ↓  independent weights, training, deployment
Arbiter Agent
    ↓  cross-domain contradiction resolution
Blue-Green Deployment
    ↓  utility-deviation-triggered, softmax traffic routing
```

Updating surgery weights cannot affect software engineering weights. There are no shared parameters to interfere. Catastrophic forgetting is resolved architecturally. Graph depth is hardware-adaptive: high-VRAM GPUs run shallow graphs of large models; consumer GPUs run deeper graphs of smaller specialists at lower cost per query.

### Arbiter Agent

When two submodels produce conflicting outputs:

| Check | Weight | What it tests |
|---|---|---|
| Logical | 0.30 | Does the output contradict its own premises? |
| Mathematical | 0.40 | Are complexity or numerical claims provably wrong? |
| Cross-session | 0.20 | Does it contradict prior verified assertions? |
| Empirical | 0.10 | Does it contradict verifiable external ground truth? |

Four verdict cases: A correct → correct B; B correct → correct A; both wrong → correct both + curiosity gap bonus on the knowledge gap; inconclusive → external escalation. Corrections route internally as DPO signal. Nothing is disclosed externally.

Arbiter calibration: 2–5% of verdicts independently verified against domain experts. Escalates to 15% hard ceiling if correction volume is elevated.

### Assertions Store (Evidence with Decay)

Verified facts persist across sessions with field-specific confidence decay:

| Class | Decay | Examples |
|---|---|---|
| A — No decay | Never | Mathematical proofs, physical laws |
| B — Slow (τ = 10yr) | Exponential | Mechanical engineering principles |
| C — Moderate (τ = 3yr) | Exponential | Medical anatomy, legal common law |
| D — Fast (τ = 6mo) | Exponential | Clinical guidelines, security practices, ML benchmarks |

Effective confidence at retrieval = `C_verified × exp(-Δt/τ)`. Stale evidence automatically loses weight.

---

## Mathematical Foundations (Appendix B, v0.5)

| Result | Content |
|---|---|
| **Theorem B.1** | Additive linear structure of U uniquely necessary from five axioms |
| **§B.2** | Field weights grounded in error-cost proportionality |
| **Proposition B.3** | Efficacy sigmoid = Mann-Whitney dominance probability |
| **Theorem B.4** | EMA with α = 0.2 is Kalman-optimal for ρ = 0.05 noise ratio |
| **Theorem B.5** | Confidence convergence in expectation; closed-form recovery time |
| **Proposition B.6** | 50% curiosity cap enforces exploitation dominance |
| **Theorem B.7** | Personality evolution: Lyapunov stable, half-life ≈ 34 cycles |

---

## Simulation Results (v0.4)

```
Cycle  Difficulty  avg U    avg E_ema  avg C    Contradictions
─────  ──────────  ──────   ─────────  ──────   ──────────────
  1    easy        0.5128   0.5124     0.6005        1
  2    medium      0.5921   0.5542     0.8128        0   (+0.0793)
  3    hard        0.6288   0.5740     0.8940        0   (+0.0367)

Overall U:      +0.116   (0.513 → 0.629)
Efficacy EMA:   +0.062   (accumulates across cycles)
Confidence:     +0.294   (0.601 → 0.894)
Contradictions: 1 → 0   (eliminated by cycle 2)
Routing:        easy → medium → hard (as confidence rises)
DPO pairs:      2        (from seeded O(n) claim on nested loop)
```

```bash
cd agent && python3 simulate.py
```

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
├── requirements.txt
└── simulate.py                # Self-contained simulation (no API needed)

whitepaper_v05.md              # Full theoretical writeup
whitepaper_v05.html            # HTML with rendered math (KaTeX) + charts
adaptive_utility_agents_arxiv.pdf
arxiv_submission.zip
simulation_results_v04.json
```

---

## Quick Start

```bash
# Simulation — no API key needed
cd agent && python3 simulate.py

# Live harness
pip install httpx
export ANTHROPIC_API_KEY=sk-ant-...
cd agent && python3 harness.py
```

---

## What's New in v0.5

- **Appendix B**: Complete formal proofs for all core components (B.1–B.7)
- **Personality as behavioral wrapper**: Exponential family tilt with W1–W4 properties; cleanly separates from utility function so all theorems hold unchanged
- **Revised framing**: Goal (error non-repetition), mechanism (utility as control law), monolithic vs. Micro-Expert distinction stated from the first paragraph
- **Theorem renumbering**: Sequential B.1–B.7, no more B.4′
- **Updated simulation**: EMA efficacy accumulation, difficulty routing, DPO pair generation

---

## Status

Active research project at v0.5. Empirical validation beyond simulation — live API experiments, ablation studies, baseline comparisons — is the next priority. Contributions and collaboration welcome.
