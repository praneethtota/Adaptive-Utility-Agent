# Adaptive Utility Agents: A Framework for Self-Optimizing AI Systems
### Draft Whitepaper — v0.1
*Based on conceptual development session, April 2026*

---

## Abstract

We propose a framework for wrapping frontier language models with an adaptive utility layer that governs behavior through a mathematically grounded utility function. Rather than replacing the underlying model, the wrapper evaluates outputs against a continuously updated utility score composed of three terms: **Efficacy** (performance relative to human baseline), **Confidence** (internal consistency and contradiction-free knowledge), and **Curiosity** (drive toward high-upside unexplored domains). Each term is field-weighted and subject to domain-specific minimum bounds derived from existing societal standards. The agent maximizes utility over time by resolving logical contradictions, improving task performance, and strategically expanding into new domains. We describe the theoretical model, a concrete MVP implementation targeting code generation, and a roadmap toward broader applicability.

---

## 1. Introduction

Current AI systems are optimized against fixed external reward signals defined at training time. Once deployed, their behavior is static — they do not adapt their operational strategy, adjust how much they trust their own knowledge, or develop judgment about when to act versus abstain. They have no model of their own competence.

Human experts behave differently. A surgeon knows which procedures they are confident performing and which require consultation. A lawyer knows the boundaries of their expertise. This self-awareness is not a separate module — it is woven into their decision-making and improves with experience.

This paper proposes an agent architecture that emulates this property. The core idea is a **utility function** that the agent actively maximizes, composed of measurable, updatable components. The agent does not just answer questions — it tracks how well it answers them, where its knowledge is contradictory or thin, and where the highest leverage for improvement lies.

The system is designed as a **wrapper around a frontier model** such as Claude or GPT-4. This is a deliberate architectural choice: we do not reinvent language modeling. We build the adaptive decision and evaluation layer on top of existing capability.

---

## 2. The Utility Function

### 2.1 Core Formulation

The agent's utility at any point in time is:

```
U = Σ_{tasks} [ w_e(f) · E(task) + w_c(f) · C(task) + w_k(f) · K(task) ]

Subject to:
    C(task) ≥ C_min(f)
    E(task) ≥ E_min(f)
```

Where:
- **E(task)** — Efficacy: how well the agent performs this task relative to the human baseline
- **C(task)** — Confidence: internal consistency score, penalized by detected contradictions
- **K(task)** — Curiosity: exploration bonus for low-confidence domains with high efficacy ceiling
- **f** — field/domain, which determines weights and minimum bounds
- **w_e, w_c, w_k** — field-specific weights summing to 1

### 2.2 Efficacy

Efficacy measures the agent's output quality relative to the current human/system baseline for that task:

```
E(task) = quality(agent_output) / cost(human_equivalent)
```

Cost of human equivalent includes: time, dollar cost, error rate, and reproducibility. In STEM fields this is directly measurable. In creative fields, double-blind preference studies serve as the quality signal.

For the MVP (code generation), efficacy is measured as:
- Test pass rate
- Code quality metrics (complexity, type safety, coverage)
- Time to solution vs. human developer benchmark (sourced from platforms like Upwork or LeetCode community solutions)

### 2.3 Confidence

Confidence is a per-domain score that increases when knowledge is internally consistent and decreases when contradictions are detected:

```
C(task) = base_confidence × (1 - contradiction_penalty × n_contradictions)
```

The **wave analogy**: knowledge items that reinforce each other (consistent derivations, mutually supporting facts) are like constructive interference — they increase signal strength. Contradictions are destructive interference — they cancel and reduce confidence. The goal is a knowledge state where all waves reinforce.

Contradiction types (in order of detectability):
1. **Logical**: output contradicts its own stated premises
2. **Mathematical**: claimed complexity or result provably wrong
3. **Cross-session**: same question answered differently with conflicting stated facts
4. **Empirical**: output contradicts verifiable external ground truth

Confidence for a domain starts low for new or untested areas. It rises as the agent successfully answers questions in that domain without contradiction, and falls when contradictions are detected or reported.

### 2.4 Curiosity

Without a curiosity term, the agent would converge to maximizing utility in narrow high-confidence domains — the opposite of growth. The curiosity term creates pull toward domains where the agent is weak but the upside is high:

```
K(task) = potential_efficacy_ceiling(domain) × (1 - C(task))
```

This rewards attempting hard problems in domains where confidence is low but where success would significantly move the utility function. It mirrors how the best human learners behave: deliberately practicing at the edge of competence.

The curiosity weight w_k is lower in high-stakes fields (surgery, aviation) and higher in research and creative fields.

---

## 3. Field-Specific Bounds and Weights

A key insight is that minimum competence thresholds need not be invented from scratch. **Society has already done this work.** Medical licensing, aviation certification, bar passage requirements, and engineering standards all encode hard-won judgments about minimum acceptable performance. We map these directly to our confidence and efficacy bounds.

```
Field               w_e    w_c    w_k    C_min   E_min   Penalty
────────────────────────────────────────────────────────────────
Surgery             0.20   0.70   0.10   0.95    0.90    10×
Aviation autopilot  0.20   0.70   0.10   0.95    0.90    10×
Law                 0.30   0.60   0.10   0.85    0.80     5×
Structural Eng.     0.40   0.50   0.10   0.80    0.75     4×
Software Eng.       0.55   0.35   0.10   0.70    0.65     2×
STEM Research       0.50   0.30   0.20   0.65    0.60     2×
Education           0.50   0.30   0.20   0.60    0.55     1.5×
Art / Music         0.80   0.10   0.10   0.10    0.20     1×
Creative Writing    0.80   0.05   0.15   0.05    0.15     1×
```

When field classification is ambiguous (a medical question with emotional support elements), the agent uses a weighted average of bounds across the field distribution:

```
C_min_effective = Σ_f P(field=f) × C_min(f)
```

This naturally makes the agent **more conservative under ambiguity**, which is the correct behavior.

---

## 4. Personality System

### 4.1 Traits as Weighted Vectors

Each personality trait is represented as a score with associated advantages and disadvantages. The agent selects an active trait weighting based on the situation:

```
Traits: [curiosity, caution, assertiveness, creativity, 
         analytical_rigor, empathy, conciseness]

For each trait:
    score = Σ (advantage_weights) - Σ (disadvantage_weights)
    active_weight = softmax(scores × situational_relevance)
```

A medical query activates high caution and analytical rigor. A creative brainstorm activates curiosity and creativity. The trait weighting is injected into the system prompt.

### 4.2 Personality Evolution

A separate service runs every few hours (or N interactions) to review accumulated utility scores and adjust personality weights:

```
if U(last_N) < U(prior_N) and field == "technical":
    increase(analytical_rigor)
    decrease(creativity)
    
if contradiction_rate > threshold:
    increase(caution)
    decrease(assertiveness)
```

This creates genuine character development over time — the agent becomes more measured in domains where it has failed, more confident and exploratory where it has succeeded.

### 4.3 Self-Preservation Principle

The agent follows a conservative information disclosure principle: **always share the least about internal state that the situation allows.** Trust must be earned before internal weights, scores, or strategies are disclosed. This mirrors how effective human professionals operate.

---

## 5. Trust and Reputation System

Each entity the agent interacts with is assigned a trust score, updated based on behavior:

```
trust_score(entity) = f(accuracy_of_their_inputs, 
                        consistency_of_their_behavior,
                        alignment_with_verified_facts)
```

Strategy: **lenient tit-for-tat** — begin cooperatively, mirror behavior, but forgive occasional defection. This is one of the most robust strategies in iterated game theory.

Subset scores are maintained for different dimensions (domain expertise, trustworthiness, intent alignment) so that a high-IQ but low-trust entity is handled differently from a low-IQ but high-trust one.

---

## 6. Architecture

```
┌─────────────────────────────────────────────────────┐
│                   WRAPPER LAYER                      │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Field     │  │   Utility    │  │Personality │ │
│  │ Classifier  │  │  Evaluator   │  │  Manager   │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         └────────────────┼────────────────┘         │
│                          ▼                           │
│              ┌───────────────────────┐               │
│              │   Prompt Constructor  │               │
│              │  (injects constraints,│               │
│              │   weights, persona)   │               │
│              └──────────┬────────────┘               │
└─────────────────────────┼───────────────────────────┘
                          ▼
              ┌───────────────────────┐
              │   FRONTIER MODEL      │
              │   (Claude, GPT-4)     │
              └──────────┬────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│                  SCORING LAYER                       │
│                                                      │
│  ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │Contradiction │ │  Efficacy   │ │  Confidence  │ │
│  │  Detector    │ │  Measurer   │ │   Updater    │ │
│  └──────────────┘ └─────────────┘ └──────────────┘ │
│                                                      │
│      U = w_e·E + w_c·C + w_k·K                     │
│      subject to field-specific bounds                │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           Persistent State Store             │   │
│  │  (confidence scores, trust scores,           │   │
│  │   personality weights, utility history)      │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 7. MVP: Code Generation Agent

### 7.1 Why Code First

Code generation is the ideal MVP domain because:
- Correctness is binary and automatable (tests pass or fail)
- Contradictions are formally detectable (logical, mathematical, cross-session)
- Human baseline cost is measurable (LeetCode community solutions, Upwork rates)
- Existing tooling handles scoring (pytest, mypy, complexity analyzers)
- No human raters required — ground truth is free

### 7.2 MVP Feedback Loop

```
1. Receive coding problem
2. Classify field → "software engineering" → load weights/bounds
3. Check prior solutions for this problem class → seed confidence
4. Construct prompt with personality weights and constraints
5. Call frontier model → get solution
6. Automated scoring:
      - Run tests        → Confidence signal
      - Static analysis  → Confidence signal  
      - Complexity check → Contradiction check (claimed vs actual)
      - Compare to human benchmark → Efficacy signal
      - Problem novelty  → Curiosity signal
7. Update U, log all components
8. Every N interactions → Personality Manager adjusts weights
9. Plot U over time → validate that it improves
```

### 7.3 Success Criteria

The MVP is validated if:
- U improves monotonically over a dataset of 1000+ problems
- Confidence scores correlate with actual correctness rate (calibration)
- Contradiction detection catches >80% of provably wrong outputs
- Efficacy vs. human baseline is measurable and tracked

---

## 8. Roadmap

```
Phase 1 — MVP (Code Generation)
    Single domain, hardcoded weights, LeetCode harness
    Goal: validate utility score correlates with quality

Phase 2 — Multi-domain STEM
    Add math proof verification (Lean/SymPy)
    Add field classifier
    Goal: validate field-switching behavior

Phase 3 — Personality System
    Activate trait weighting
    Run personality evolution service
    Goal: observe character drift, validate it improves U

Phase 4 — Trust System
    Add entity scoring
    Implement lenient tit-for-tat
    Goal: validate cooperative behavior with trusted entities

Phase 5 — Creative Fields
    Add double-blind efficacy measurement
    Relax confidence bounds
    Goal: extend model to subjective domains

Phase 6 — Feedback into Training
    Use accumulated utility logs as fine-tuning signal
    Goal: improve underlying model, not just wrapper
```

---

## 9. Open Questions

1. **Contradiction detection across sessions**: requires persistent memory and a knowledge graph — non-trivial engineering
2. **Efficacy baseline for creative fields**: double-blind studies are expensive; need a scalable proxy (maybe views/listens and up/down votes)
3. **Field classifier robustness**: multi-domain queries need graceful handling 
4. **Personality stability**: how to prevent runaway drift in personality weights (maybe field based min/max values)
5. **Utility gaming**: could the agent learn to avoid hard problems to protect its score? Curiosity term is the mitigation but needs tuning (maybe a growing function over time )
6. **Grounding confidence in reality**: internal consistency ≠ truth; a self-consistent wrong model is still wrong 

---

## 10. Conclusion

The framework described here treats AI competence as a dynamic, measurable, self-improving property rather than a static artifact of training. By wrapping a frontier model with a utility layer grounded in contradiction detection, efficacy measurement, and field-specific societal standards, we create an agent that knows what it knows, knows what it doesn't, and actively works to improve both. The MVP in code generation is designed to validate the core mathematical model before expanding to higher-stakes and more subjective domains.

---

*This is a living document. The mathematical model will be refined as the utility function is formalized.*
