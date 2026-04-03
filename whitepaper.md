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

### 9.1 Resolved / Refined

**1. Utility gaming** *(resolved)*

The risk: the agent learns to avoid hard problems to protect its score, converging to a narrow high-confidence comfort zone.

Resolution: two mechanisms working together.

First, a **growing curiosity function** that increases pressure to explore the longer the agent stays without encountering novelty:

```
K_raw(task, t) = potential_ceiling
               × (1 - C(task))
               × (1 + α(field) × log(1 + interactions_without_novelty))

Where α(field) is field-specific:
    α → high for research/creative fields
    α → near zero for surgery/aviation
```

The counter `interactions_without_novelty` resets each time the agent successfully handles a genuinely new problem type — rewarding exploration when it occurs.

Second, a **50% curiosity cap** that prevents curiosity from becoming the dominant utility term and sending the agent into useless tangents:

```
K_effective = min(K_raw, (w_e · E + w_c · C) / w_k)

U = w_e · E + w_c · C + w_k · K_effective
```

This constraint is self-scaling: when E and C are high, the cap is loose and curiosity can push hard. When the agent is weak (low E and C), curiosity is automatically constrained — preventing exploration before the basics are solid. K can never exceed 50% of total U.

---

**2. Personality stability** *(resolved)*

The risk: the personality evolution service applies small consistent deltas every cycle, causing traits to drift monotonically — e.g. caution rising to 0.95 and making the agent refuse almost everything.

Resolution: three layered safeguards.

**Layer 1 — Field-specific bounds (hard floor and ceiling per trait):**

```
Trait              Surgery        Software Eng    Creative
─────────────────────────────────────────────────────────
caution            [0.70, 0.95]   [0.30, 0.70]   [0.10, 0.40]
curiosity          [0.10, 0.20]   [0.30, 0.80]   [0.60, 0.95]
assertiveness      [0.20, 0.40]   [0.40, 0.80]   [0.50, 0.90]
analytical_rigor   [0.70, 0.95]   [0.50, 0.85]   [0.10, 0.50]
creativity         [0.10, 0.20]   [0.30, 0.70]   [0.70, 0.95]
```

The floor prevents complete suppression of any trait. The ceiling prevents pathological dominance.

**Layer 2 — Drift rate cap (max delta per evolution cycle):**

```
max_delta_per_cycle = 0.05   (general fields)
                    = 0.02   (high-stakes fields: surgery, aviation)
```

A single bad run of contradictions cannot spike a trait to its ceiling in one step.

**Layer 3 — Mean reversion (soft pull toward field baseline between cycles):**

```
Δ_adjusted = Δ_raw - β × (current_score - neutral_score(trait, field))
```

Where β = 0.01. Creates a gentle pull back toward the field's natural personality baseline after drift.

---

**3. Contradiction detection across sessions** *(simplified)*

The original framing — "requires a full knowledge graph" — was an overcomplication. The contradiction detector already acts as a parser, stripping outputs down to structured assertions. These just need to be persisted rather than discarded.

The approach is a **"meeting minutes" store** — only structured facts are saved, not raw text:

```
{
  session_id, timestamp, domain,
  assertions: [
    { type: "complexity",     subject: "sorting",       value: "O(n log n)" },
    { type: "best_practice",  subject: "db autocommit", value: "avoid"      },
    { type: "data_structure", subject: "two_sum",       value: "hashmap"    }
  ]
}
```

At the start of each session the agent queries the store for prior assertions on the same subject and injects them as context. Synonym matching is handled by lightweight embedding similarity — not a knowledge graph.

```
Parser (already built) → extracts structured assertions
       ↓
Key-value store (simple DB) → persists by subject + domain
       ↓
Embedding similarity → handles synonym matching at lookup
       ↓
Contradiction check (already built) → compares structured values
```

---

### 9.2 Resolved / Refined (continued)

**4. Efficacy baseline for creative fields** *(resolved)*

The risk: creative quality is subjective. Double-blind studies are expensive, slow, and small sample — not scalable.

Resolution: **float AI work on existing public platforms** (YouTube, SoundCloud, Pinterest, iStockPhoto, Unsplash, Medium, Behance) under realistic author identities, then read existing human preference signals at scale. No new measurement infrastructure required — the signal already exists.

```
Creative Field    Platform          Signal
──────────────────────────────────────────────────────
Music             SoundCloud        plays, likes, reposts
                  Spotify           streams, saves, playlist adds
Visual Art        Pinterest         saves, clicks, reposts
                  iStockPhoto       downloads, purchases  ← strongest signal
Photography       Unsplash          downloads, likes
Writing           Medium            claps, reads, reading time
Video             YouTube           views, likes, watch time, shares
Design            Behance           appreciations, views, saves
```

Signals are weighted by intent strength:

```
purchase / download    1.0   (strongest — actual economic behavior)
save / bookmark        0.8
share / repost         0.7
like / upvote          0.5
comment                0.4
view / listen          0.1   (weakest — could be accidental)
```

Stock sites (iStockPhoto, Unsplash) are the cleanest signal: a download or license use represents real economic intent with no algorithmic amplification distortion and built-in category taxonomy for like-for-like comparison.

**Creative efficacy is two-component, not one:**

A key insight: in STEM fields, efficacy and skill are the same thing — write correct code, that's the whole job. In creative fields they are separable and both matter:

```
Creative_Efficacy = Content_Efficacy × Discoverability_Efficacy

Content_Efficacy      = conversion rate (likes/saves given views)
                        → can the work hold attention when shown?

Discoverability_Efficacy = impressions, search ranking, recommendation rate
                        → can it find an audience at all?
```

Marketing and platform discoverability are **not noise to control for** — they are part of the skill. Every successful human creator has to crack this. If the agent cannot, that is a real efficacy gap, not a measurement artifact. This means the agent has a natural curriculum:

```
Stage 1: Generate work that converts well → optimize content quality
Stage 2: Learn to title, tag, describe effectively → optimize search/recommendation
Stage 3: Build cross-platform presence → optimize network effects and retention
```

**The baseline is self-updating:** Human creative work on these platforms gives a continuously updated category benchmark for free. As human creative output evolves, the baseline evolves with it — no periodic re-calibration needed.

**Attribution:** Author identities should be realistic but not attributed to AI, preserving the blind nature of the study. Human collaborators acting as the "face" of accounts (already common in the creator economy via ghostwriting and session work) avoids platform ToS exposure while maintaining the same measurement property.

**Confidence function:**

```python
def creative_efficacy(ai_signals, category_baseline, min_observations=50):
    if ai_signals.total_observations < min_observations:
        return None  # insufficient data — do not score yet

    ai_score    = weighted_engagement(ai_signals)
    baseline    = weighted_engagement(category_baseline)
    ratio       = ai_score / baseline

    # Same sigmoid normalization as STEM — ai == human → 0.5
    return 1.0 - 1.0 / (1.0 + ratio)
```

This also unlocks **cross-field efficacy comparison** for the first time: if music efficacy is 0.60 and software engineering efficacy is 0.70, both are on the same [0,1] scale and directly comparable.

---

**5. Field classifier robustness** *(resolved)*

The risk: the classifier collapses multi-domain queries to a single dominant field, misses field drift mid-conversation, and handles ambiguous queries without appropriate conservatism.

Three failure modes and their resolutions:

**Failure mode 1 — False confidence in a single field:**

```
"Write a Python script to analyze patient drug dosage data"
→ naive: {"software_engineering": 0.95, "medicine": 0.05}  ← wrong
→ correct: {"software_engineering": 0.65, "medicine": 0.35}
```

Resolution: **high-stakes floor** — any high-stakes field with any meaningful presence is floored at a minimum representation threshold (0.15), preventing dilution of dangerous fields.

**Failure mode 2 — Field drift mid-conversation:**

A conversation that starts as software engineering may drift into medicine and then law over several turns. Single-turn classification misses this.

Resolution: **sliding window EMA** over the conversation's field history, weighted toward recent turns:

```
effective_field_dist = EMA(per_turn_classifications, alpha=0.4)
```

Bounds tighten naturally as a conversation drifts into higher-stakes territory.

**Failure mode 3 — Genuine ambiguity:**

Resolution: **entropy-based conservative fallback** — when the distribution's entropy is high (agent genuinely doesn't know the field), bounds shift toward the most conservative present field proportional to entropy. High entropy increases caution, it does not average toward the middle.

```
if entropy_ratio > 0.7:
    c_min → lerp toward most conservative present field's c_min
```

Full pipeline:

```
Per-turn classifier
       ↓
High-stakes floor enforcement
       ↓
Sliding window EMA over conversation history
       ↓
Entropy check → conservative fallback if ambiguous
       ↓
Blended config with hardened bounds
```

---

### 9.3 Still Open

6. **Grounding confidence in reality**: internal consistency ≠ truth; a self-consistent wrong model is still wrong

---

## 10. Conclusion

The framework described here treats AI competence as a dynamic, measurable, self-improving property rather than a static artifact of training. By wrapping a frontier model with a utility layer grounded in contradiction detection, efficacy measurement, and field-specific societal standards, we create an agent that knows what it knows, knows what it doesn't, and actively works to improve both. The MVP in code generation is designed to validate the core mathematical model before expanding to higher-stakes and more subjective domains.

---

*This is a living document. The mathematical model will be refined as the utility function is formalized.*
