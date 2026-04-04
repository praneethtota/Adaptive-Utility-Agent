# Adaptive Utility Agent

> **A self-optimizing AI agent framework that knows what it knows, knows what it doesn't, and actively corrects what it gets wrong — between model releases, not just at training time.**

---

## License

**Code:** GNU General Public License v3.0 — see `LICENSE`
**Whitepaper:** Creative Commons Attribution 4.0 — see `LICENSE-CC-BY-4.0`

If you build on this work, please cite:
> Tota, P. (2026). *Adaptive Utility Agents: A Framework for Self-Optimizing AI Systems* (v0.4). GitHub. https://github.com/praneethtota/Adaptive-Utility-Agent

---

## What This Is

Most deployed AI systems are static artifacts — optimized at training time and frozen on release. They apply the same confidence to a surgical recommendation as to a recipe suggestion. When they produce a contradiction today, they produce it again tomorrow.

This project proposes a different architecture: a **wrapper around a frontier language model** (Claude, GPT-4, etc.) with an adaptive utility layer that governs behavior through a mathematically grounded utility function. The agent does not just answer questions — it tracks how well it answers them, where its knowledge is contradictory or thin, and where the highest leverage for improvement lies.

Critically, the utility function is **not a monitoring metric**. It is the direct loss-weighting mechanism for a three-layer continual learning architecture that corrects contradictions and improves behavior between model releases, without waiting for a full retraining cycle.

Read the [whitepaper](whitepaper_v04.md) for the full theory, or the [arXiv preprint](adaptive_utility_agents_arxiv.pdf) for the academic version.

---

## Core Utility Function

```
U = w_e(field) · E + w_c(field) · C + w_k(field) · K

subject to:
    C ≥ C_min(field)          [field-specific confidence floor]
    E ≥ E_min(field)          [field-specific efficacy floor]
    w_k · K ≤ 0.5 · U_total  [curiosity cap — prevents gaming]
```

| Term | Name | What It Measures |
|------|------|-----------------|
| **E** | Efficacy | Performance relative to human baseline |
| **C** | Confidence | Internal consistency, penalized by detected contradictions |
| **K** | Curiosity | Pull toward high-upside unexplored domains |

Field weights and minimum bounds are derived from existing societal standards — medical licensing, aviation certification, bar passage, engineering requirements — making the thresholds principled rather than arbitrary.

A surgery contradiction is weighted **10× more harshly** than a creative writing mistake at training time. That is not a policy parameter — it falls directly out of the utility function.

---

## Architecture

The system has six main components beyond the utility function:

**1. Three-layer continual learning pipeline**
- *Per-session*: detected contradictions injected as corrective assertions into the system prompt (real-time, no weight change)
- *Calibration-cycle*: DPO fine-tuning with field-penalty-weighted loss, several times daily
- *Release-level*: LoRA adapter distillation into a new base fine-tune, monthly

**2. Personality system**
Trait weights (curiosity, caution, analytical rigor, assertiveness, creativity, conciseness) evolve with utility history under three-layer stability safeguards: field-specific hard bounds, drift rate caps, and mean reversion.

**3. Entity trust and reputation system**
Each interacting entity is scored on domain expertise (from verifiable credentials) and behavioral trust. Scores gate how external inputs are weighted and when external escalation is permitted.

**4. Distributed model graph**
The monolithic model is decomposed into independently deployable domain submodels communicating over structured APIs — analogous to microservices. This physically resolves catastrophic forgetting: updating one domain's weights cannot affect another. Deployment depth is hardware-adaptive (H100 → shallow graph of large models; consumer GPUs → deeper graph of small specialist models).

**5. Arbiter Agent**
Dedicated contradiction resolution across conflicting submodel outputs. Runs four structured checks: logical, mathematical, cross-session, empirical. Verified corrections feed back to the relevant submodels as DPO signal (internal only — never disclosed externally). When both models are wrong (Case 3), a curiosity gap bonus preferentially directs exploration toward the unresolved knowledge gap. When all checks fail (Case 4), a controlled external escalation queries verified domain experts through obfuscated, partialized queries.

**6. Blue-green deployment**
Submodel updates are triggered by utility deviation thresholds derived from power analysis (not arbitrary SLA targets), managed through self-regulating softmax traffic routing. No manual intervention required for traffic shifts.

---

## Project Structure

```
agent/
├── config.py                 # Field weights, bounds, penalty multipliers
├── field_classifier.py       # Field distribution with robustness mechanisms
│                             #   (high-stakes floor, EMA drift tracking,
│                             #    entropy-based conservative fallback)
├── contradiction_detector.py # Logical, mathematical, cross-session detection
├── utility_scorer.py         # E, C, K with growing curiosity + 50% cap
├── personality_manager.py    # Trait weights, evolution, three-layer stability
├── creative_efficacy.py      # Two-component creative efficacy model
├── agent.py                  # Main UtilityAgent class
├── harness.py                # LeetCode-style MVP test harness
└── simulate.py               # Self-contained simulation (no API needed)

whitepaper_v04.md             # Full theoretical writeup (v0.4)
whitepaper_v04.html           # HTML version with embedded simulation charts
adaptive_utility_agents_arxiv.pdf  # arXiv preprint
arxiv_submission.zip          # LaTeX source for arXiv submission
simulation_results.json       # Raw simulation output data
```

---

## Simulation Results

The core utility mechanism was validated via a self-contained simulation across 3 calibration cycles on 8 LeetCode-style problems. No live API required — runs locally.

```
Cycle    avg U    avg E    avg C    Contradictions
──────   ──────   ──────   ──────   ──────────────
  1      0.5949   0.5867   0.7628        1
  2      0.6588   0.5867   0.9602        0    (+0.0639)
  3      0.6704   0.5867   0.9933        0    (+0.0116)

Overall U improvement: +0.0755
Contradiction reduction: 1 → 0 (eliminated by cycle 2)
Confidence gain: +0.231 (0.763 → 0.993)
Every problem improved across all 3 cycles — no exceptions.
```

Run it yourself:

```bash
cd agent
python3 simulate.py
```

---

## Quick Start (Live API)

```bash
pip install httpx
cd agent
python harness.py
```

The harness runs LeetCode-style problems through the full pipeline:
1. Classifies the field → loads weights and bounds
2. Queries assertions store for relevant prior corrections
3. Builds system prompt with active corrections + personality traits
4. Calls Claude → gets solution
5. Runs contradiction detection (logical, mathematical, cross-session)
6. Scores U = w_e·E + w_c·C + w_k·K\_effective
7. Logs all components and triggers personality evolution every N interactions

---

## Key Design Decisions

- **Wrapper, not replacement** — builds on Claude or any frontier model; does not reinvent language modeling
- **Utility function as loss weighting** — field penalty multipliers are DPO training weights, not just labels
- **Societal standards as bounds** — C\_min and E\_min derived from real licensing requirements, not arbitrary thresholds
- **Conservative under ambiguity** — blended field bounds tighten toward the most conservative field present; high entropy increases caution
- **Abstain rather than fail** — agent refuses to act when confidence is below domain minimum; escalates rather than guesses
- **Physically isolated domain weights** — distributed graph means updating surgery cannot break CS; catastrophic forgetting resolved architecturally
- **Minimum disclosure** — internal state (evidence chains, arbiter verdicts, correction signals) never disclosed externally

---

## Roadmap

| Phase | Goal |
|-------|------|
| 1 — MVP | Live API, 1,000+ problems, first real calibration cycle |
| 2 — Multi-domain STEM | Lean/SymPy verification, field classifier in production |
| 3 — Personality | Trait evolution at scale, stability safeguards validated |
| 4 — Trust system | Entity scoring, lenient tit-for-tat, external escalation |
| 5 — Creative fields | Platform signal collection, two-component efficacy |
| 6 — Distributed graph | Physically separate submodels, Arbiter in production |
| 7 — Base model feedback | Distill adapters into base fine-tune |

---

## Status

This is a living research project. The mathematical formalization of the utility function and empirical validation beyond simulation are ongoing. The whitepaper is versioned — v0.4 is the current release.

Contributions, feedback, and collaboration welcome.
