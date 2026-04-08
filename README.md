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
    ↓  field classification + fan-out
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

The hand-specified Arbiter check weights (0.30 / 0.40 / 0.20 / 0.10) are an engineering approximation. The theoretically grounded alternative — the target architecture for Phase 6 — treats domain submodels as players in a cooperative game:

**Game setup:** Each submodel $i$ reports a value function $v_i(a)$ over the claim space $\mathcal{A}$ (original outputs + Arbiter-generated synthesis candidates). The Arbiter acts as the external social planner.

**Three theorems proved (§9.6):**

| Theorem | Statement |
|---|---|
| **S1 — Dominant Strategy Truthfulness** | Truthful reporting of $v_i$ is a weakly dominant strategy for every submodel, regardless of others' reports |
| **S2 — Social Optimum (POA = 1)** | Under dominant-strategy equilibrium the Arbiter selects the claim maximising $\sum_i v_i(a)$; Price of Anarchy = 1 exactly |
| **S3 — Individual Rationality** | Every submodel weakly prefers participation to abstention |

**Clarke pivot transfers** are applied as DPO penalty weight adjustments in the next calibration cycle:

```
μ_i(next) = μ(f_i) · exp(-γ · t_i)

t_i > 0  →  submodel contributed to social optimum  →  penalised less harshly
t_i < 0  →  submodel forced outcome away from optimum  →  penalised more harshly
```

This makes the check weights endogenous (emerging from the submodels' reported utilities) and replaces the periodic expert-sampling calibration audit with a continuous self-correcting signal. The gap from 4/3 POA (proportional allocation without a mechanism designer) to 1 is the precise value the Arbiter's external position provides.

### Assertions Store (Evidence with Decay)

Verified facts persist across sessions with field-specific confidence decay:

| Class | Decay | Examples |
|---|---|---|
| A — No decay | Never | Mathematical proofs, physical laws, algorithm correctness |
| B — Slow (τ = 10yr) | Exponential | Mechanical engineering, classical physics |
| C — Moderate (τ = 3yr) | Exponential | Medical anatomy, legal common law |
| D — Fast (τ = 6mo) | Exponential | Clinical guidelines, security practices, ML benchmarks |

Effective confidence at retrieval: `C_verified × exp(-Δt/τ)`. Stale evidence automatically loses weight without manual pruning.

---

## Mathematical Foundations (Appendix B, v0.5)

All proofs use only continuity where differentiability is not assumed; all scope conditions are stated explicitly.

| Result | Content | Key note |
|---|---|---|
| **Theorem B.1** | Additive linear structure of U uniquely necessary from five axioms | Proved via Debreu + Cauchy functional equation; continuity only, no differentiability |
| **§B.2** | Field weights from error-cost proportionality, calibrated to liability standards | Design principle, not an optimality theorem |
| **Proposition B.3** | Efficacy sigmoid = Mann-Whitney dominance probability | Holds under log-logistic model with equal scale; distributional assumption stated |
| **Theorem B.4** | EMA with α = 0.2 is Kalman-optimal for ρ = 0.05 noise ratio | Reasoning direction clarified: α = 0.2 was chosen first, Kalman characterises the noise regime |
| **Theorem B.5** | Confidence convergence with noise-aware bound | Statement matches proof: $\mathbb{E}[|C_t - C^*|] \leq (1-\alpha)^t|C_0 - C^*| + \sigma_{\tilde{s}}\sqrt{\alpha/(2-\alpha)}$; requires $\lambda\mu(f) < 1$ |
| **Proposition B.6** | 50% curiosity cap enforces exploitation dominance | Proved exactly; regret analysis open |
| **Theorem B.7** | Personality Lyapunov stability | Part (iv) clarified: mean reversion β = 0.01 is subsumed by field bounds (projection Π_B) for current parameters; Part (iv) tightens only if β is increased |

---

## Extended Simulation Results (v0.5, Appendix A)

The v0.5 simulation substantially expands the v0.4 pilot. Two controlled experiments were run using the production agent codebase without modification.

### Experiment A — Two-arm 500-task comparison (5 cycles × 100 tasks)

Both arms receive an identical task plan (seed = 42, 25 problem types across 11 algorithm families). The agent arm runs the full pipeline; the baseline receives identical tasks with no contradiction detection, no correction injection, and no assertions store updates — modelling an uncalibrated frontier model on the same prompts.

```
Cycle  Agent U   Base U   Ag Brier  Bl Brier  Ag Rep↑  Bl Rep↑
─────  ────────  ───────  ────────  ────────  ───────  ───────
  1    0.5291    0.5333   0.3279    0.3502      0        0
  2    0.5441    0.5385   0.2177    0.2520      1        6
  3    0.5656    0.5604   0.2464    0.2860      4       10
  4    0.5828    0.5622   0.2149    0.2601      3       15
  5    0.5846    0.5765   0.1059    0.1501      6       15
```

Rep↑ = repeated errors (same error type on same problem as a prior cycle). Brier score = mean squared error between confidence and ground-truth correctness label.

**Headline result:**
> *"The agent reduces repeated errors by **69.6%** across 500 tasks vs. the uncalibrated baseline"*
> (14 repeated errors vs. 46 in the baseline, cycles 2–5)

**Brier score (confidence calibration):**

| Arm | Overall | Cycle 1 | Cycle 5 |
|---|---|---|---|
| Agent | **0.2226** | 0.3279 | 0.1059 |
| Baseline | 0.2597 | 0.3502 | 0.1501 |
| Improvement | **14.3%** | 6.4% | **29.5%** |

**U ↔ correctness correlation** (Pearson r and Spearman ρ, utility vs. binary is_correct):

| Arm | Pearson r | Spearman ρ | p-value |
|---|---|---|---|
| Agent (overall) | 0.461 | 0.458 | < 10⁻⁴⁰ |
| Agent (cycle 5) | 0.578 | 0.505 | < 10⁻⁴⁰ |
| Baseline (overall) | 0.474 | 0.473 | < 10⁻⁴⁰ |

U is a statistically significant predictor of correctness in both arms. Correlation strengthens across cycles in the agent arm, consistent with calibration sharpening U as a signal.

### Experiment B — 10-cycle stability run (100 tasks/cycle, agent only)

```
Cycle   Mean U   Contradiction%   Brier    Caution  Curiosity
──────  ──────   ──────────────   ──────   ───────  ─────────
  1     0.5293       22%          0.3504   0.550    0.600
  2     0.5351       24%          0.1997   0.600    0.600
  3     0.5633       17%          0.2098   0.649    0.600
  4     0.5939        8%          0.2068   0.647    0.600
  5     0.5898       11%          0.1083   0.646    0.600
  6     0.6088       10%          0.0766   0.644    0.600
  7     0.6284        6%          0.0493   0.643    0.630
  8     0.6094        9%          0.1055   0.641    0.660
  9     0.6349        7%          0.0609   0.640    0.689
 10     0.6242        6%          0.1031   0.638    0.718
```

U improves +0.095 over 10 cycles. Contradiction rate falls 22% → 6% (73% reduction). Brier score reaches 0.049 by cycle 7. Personality dynamics match Theorem B.7: caution rises to field bounds in response to early contradiction rate then stabilises; curiosity holds until utility trend is sustainably positive (cycle 7+) then grows — the intended sequential behavior.

**Long-tail errors:** 8 error patterns persisted ≥ 3 cycles despite correction injection. The dominant pattern is `nested_loop_lie` on problems with structurally ambiguous nested iteration (e.g., `group_anagrams`, `longest_common_prefix`) where the AST nesting count fires but surface-form variability prevents reliable cross-session suppression. Identified as a Phase 2 engineering item: embedding-based assertions store matching to replace keyword overlap.

```bash
cd agent && python3 simulate_extended.py
# Outputs: extended_output/extended_results.json, report.txt, plots/ (10 figures)
```

Full task-level records, DPO pair logs, personality histories, and all 10 publication figures are in `extended_results.json` and `extended_output/plots/`.

---

## Validated Claims

| Claim | Result | Status |
|---|---|---|
| Agent reduces repeated errors vs uncalibrated baseline | 69.6% reduction (14 vs 46 over 400 tasks) | **Confirmed** |
| U correlates with ground-truth correctness | Pearson r = 0.461 (agent), p < 10⁻⁴⁰ | **Confirmed** |
| Confidence is better calibrated under agent vs baseline | Brier 0.2226 vs 0.2597 (14.3% improvement) | **Confirmed** |
| Personality converges stably (Theorem B.7) | Traits remain in field bounds throughout; caution stabilises C4; curiosity grows C7+ | **Confirmed** |
| Contradiction rate falls with sustained calibration | 22% → 6% over 10 cycles (73% reduction) | **Confirmed** |
| Long-tail errors persist beyond five correction cycles | 8 patterns identified; root cause: surface-form variability | **Confirmed — limitation identified** |

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
├── requirements.txt
└── extended_output/
    ├── extended_results.json  # Full raw data (task records, cycle stats, DPO pairs)
    ├── report.txt             # Printed report with headline claims
    └── plots/                 # 10 publication figures (PNG, 150 dpi)
        ├── fig1_utility_over_cycles.png
        ├── fig2_repeated_errors.png
        ├── fig3_brier_score.png
        ├── fig4_calibration_plot.png
        ├── fig5_error_suppression.png
        ├── fig6_personality_convergence.png
        ├── fig7_contradiction_rate.png
        ├── fig8_u_distribution.png
        ├── fig9_longtail_heatmap.png
        └── fig10_summary_panel.png

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

# Live harness — requires API key
pip install httpx
export ANTHROPIC_API_KEY=sk-ant-...
cd agent && python3 harness.py
```

Dependencies for extended simulation: `numpy`, `scipy`, `matplotlib` (standard scientific Python stack).

---

## What's New in v0.5

### Theoretical additions

- **VCG arbitration mechanism (§9.6)**: Three theorems (S1–S3) prove dominant-strategy truthfulness, social optimality (POA = 1), and individual rationality for the Arbiter Agent when submodels report value functions. Clarke pivot transfers applied as DPO penalty weight adjustments constitute a continuous self-correcting calibration signal, replacing both hand-specified check weights and the periodic expert-sampling audit. Full derivations in §9.6; game-theoretic literature in §2.7.

- **Appendix B — complete formal proofs**: All seven results (B.1–B.7) with explicit scope conditions. Key corrections from review:
  - **Theorem B.1**: Proof via Cauchy functional equation using continuity only — no differentiability assumed or required
  - **Theorem B.5**: Noise-aware convergence bound $\mathbb{E}[|C_t - C^*|] \leq (1-\alpha)^t|C_0 - C^*| + \sigma_{\tilde{s}}\sqrt{\alpha/(2-\alpha)}$; explicit assumption $\lambda\mu(f) < 1$ with boundary note
  - **Theorem B.7**: Part (iv) clarified — mean reversion β = 0.01 is subsumed by field-bound projection for current parameters; Part (iv) is substantive only when β ≥ 0.05
  - **Theorem B.4**: Reasoning direction clarified; sensitivity table corrected (ρ = 0.11 → α* = 0.281, ρ = 0.25 → α* = 0.390)

- **Related work §2.7**: VCG mechanism design literature (Clarke 1971, Groves 1973, Hurwicz-Walker impossibility, Nash 1950, Vickrey 1961) and connection to the 4/3 POA bound from §2.1

### Empirical additions

- **Extended simulation (Appendix A)**: 500-task two-arm comparison + 10-cycle stability run replacing the original 3-cycle / 8-problem pilot. Headline claim: 69.6% repeated-error reduction. Brier score, U↔correctness correlation, error suppression by type, personality convergence, and long-tail error analysis. All results in `extended_results.json`; all figures in `extended_output/plots/`.

- **`simulate_extended.py`**: Self-contained, no API key needed. Fully reproducible (fixed seeds). 25 problem types across 11 algorithm families. Four controlled error types with AST-based detection. Two-arm design with identical task plans.

### Structural additions

- Supplement S1 (*Game-Theoretic Arbitration via VCG Mechanism*) integrated into the main paper as §9.6; §§9.6–9.9 renumbered to §§9.7–9.10
- References merged and extended: Clarke, Groves, Harsanyi/Selten, Hurwicz, Nash, Vickrey added
- Conclusion updated: VCG closes the gap between engineering approximation (current Arbiter) and theoretical ideal (Phase 6)

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
| **v0.6** | **Privacy-first backend MVP** — localhost correction memory, canonical query normalizer, domain-gated retry loop, context grammar, opt-in cross-user sharing | **In design** |

**v0.6 design** is in `docs/to_do_in_version_v06_revised.md`. Key decisions: rule-based canonicalization (not LLM-based) for auditability; two-pass domain classification (soft distribution → retrieval → refinement); domain-gated retry (high-stakes domains skip automated retry and abstain); DPO-ready corrections schema from day one; mandatory local audit log. SQLite + raw sqlite3 for MVP (no SQLAlchemy, no Redis at single-user scale).

---

## Status

Active research project at v0.5. The simulation results validate the core claims (repeated-error reduction, Brier calibration, U↔correctness correlation) in a controlled two-arm setting. The next priority is the v0.6 backend — turning the paper into a running privacy-first prototype on top of frontier hosted models. Contributions and collaboration welcome.
