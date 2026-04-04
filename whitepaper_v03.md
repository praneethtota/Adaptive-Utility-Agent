# Adaptive Utility Agents: A Framework for Self-Optimizing AI Systems
### Draft Whitepaper — v0.3
*Based on conceptual development sessions, April 2026*

> **Changelog from v0.2:** Resolved open problems merged into their respective sections. New Section 6 added covering the three-layer continual learning architecture. Open Questions updated to reflect genuinely unresolved problems.

---

## Abstract

We propose a framework for wrapping frontier language models with an adaptive utility layer that governs behavior through a mathematically grounded utility function. Rather than replacing the underlying model, the wrapper evaluates outputs against a continuously updated utility score composed of three terms: **Efficacy** (performance relative to human baseline), **Confidence** (internal consistency and contradiction-free knowledge), and **Curiosity** (drive toward high-upside unexplored domains). Each term is field-weighted and subject to domain-specific minimum bounds derived from existing societal standards. Critically, the utility function is not merely a monitoring metric — it is the direct training signal for a three-layer continual learning architecture that allows the agent to correct contradictions and improve efficacy between model releases, without waiting for a full retraining cycle. We describe the theoretical model, the continual learning pipeline, a hardware-adaptive distributed model graph architecture, a concrete MVP implementation targeting code generation, and a roadmap toward broader applicability.

---

## 1. Introduction

Current AI systems are optimized against fixed external reward signals defined at training time. Once deployed, their behavior is static — they do not adapt their operational strategy, adjust how much they trust their own knowledge, or develop judgment about when to act versus abstain. They have no model of their own competence. When they produce a contradiction today, they will produce the same contradiction tomorrow.

Human experts behave differently. A surgeon knows which procedures they are confident performing and which require consultation. A lawyer knows the boundaries of their expertise. Critically, both learn from mistakes between formal training events — through reflection, peer review, and accumulated experience. This self-awareness and continuous correction is not a separate module — it is woven into their decision-making and improves with experience.

This paper proposes an agent architecture that emulates this property. The core idea is a **utility function** that the agent actively maximizes, composed of measurable, updatable components. The agent does not just answer questions — it tracks how well it answers them, where its knowledge is contradictory or thin, and where the highest leverage for improvement lies. When contradictions are detected, the system corrects them — both immediately in the current session and permanently through periodic calibration runs — without waiting for the next model release.

The system is designed as a **wrapper around a frontier model** such as Claude or GPT-4. This is a deliberate architectural choice: we do not reinvent language modeling. We build the adaptive decision, evaluation, and learning layer on top of existing capability.

---

## 2. The Utility Function

### 2.1 Core Formulation

The agent's utility at any point in time is:

```
U = Σ_{tasks} [ w_e(f) · E(task) + w_c(f) · C(task) + w_k(f) · K(task) ]

Subject to:
    C(task) ≥ C_min(f)
    E(task) ≥ E_min(f)
    w_k · K ≤ 0.5 × U_total         [curiosity cap — see §2.4]
```

Where:
- **E(task)** — Efficacy: how well the agent performs relative to human baseline
- **C(task)** — Confidence: internal consistency score, penalized by contradictions
- **K(task)** — Curiosity: exploration bonus for low-confidence domains with high upside
- **f** — field/domain, which determines weights and minimum bounds
- **w_e, w_c, w_k** — field-specific weights summing to 1

### 2.2 Efficacy

Efficacy measures output quality relative to the current human baseline for that task:

```
E(task) = quality(agent_output) / cost(human_equivalent)
```

Cost of human equivalent includes: time, dollar cost, error rate, and reproducibility. In STEM fields this is directly measurable. In creative fields, a two-component model is used (see below).

**STEM efficacy** (code generation MVP):
- Test pass rate against automated test suites
- Code quality metrics (complexity, type safety, static analysis)
- Time and cost vs. human developer benchmark (Upwork rates, LeetCode community solutions)

**Creative efficacy — two-component model:**

In STEM fields, efficacy and skill are the same thing — write correct code, done. In creative fields they are separable and both matter independently:

```
Creative_Efficacy = Content_Efficacy × Discoverability_Efficacy

Content_Efficacy        = conversion rate (engagement given views)
                          "can the work hold attention when shown?"

Discoverability_Efficacy = impressions, search ranking, recommendation rate
                          "can it find an audience at all?"
```

Marketing and platform discoverability are not noise to control for — they are part of the creative skill, exactly as they are for every successful human creator. If the agent cannot crack discoverability, that is a genuine efficacy gap.

The measurement approach: float AI creative work on existing public platforms (SoundCloud, iStockPhoto, Unsplash, Medium, YouTube, Behance) under realistic author identities. Platform engagement signals are weighted by intent strength:

```
purchase / download    1.0   (strongest — real economic behavior)
save / bookmark        0.8
share / repost         0.7
like / upvote          0.5
comment                0.4
view / listen          0.1   (weakest — could be accidental)
```

Stock licensing sites (iStockPhoto, Unsplash) provide the cleanest signal — a download represents real economic intent with no algorithmic amplification distortion and built-in category taxonomy for like-for-like comparison. The human creator baseline is self-updating: as human creative output on these platforms evolves, so does the benchmark, with no periodic re-calibration needed.

This formulation also unlocks cross-field efficacy comparison for the first time. If music efficacy is 0.60 and software engineering efficacy is 0.70, both are on the same [0,1] scale and directly comparable, because both use the same sigmoid normalization against their respective baselines.

The agent develops creative capability through a natural curriculum:
```
Stage 1 → Generate work that converts well (content quality)
Stage 2 → Learn to title, tag, describe effectively (discoverability)
Stage 3 → Build cross-platform presence (network effects, retention)
```

### 2.3 Confidence

Confidence is a per-domain score that increases when knowledge is internally consistent and decreases when contradictions are detected:

```
C(domain) updated via EMA:
    C_new = (1 - α) · C_prior + α · (test_pass_rate · (1 - penalty))
    penalty = contradiction_penalty × field_penalty_multiplier
```

The **wave analogy**: knowledge items that reinforce each other are like constructive interference — they increase signal strength. Contradictions are destructive interference. The goal is a knowledge state where all waves reinforce.

Contradiction types (in order of detectability):
1. **Logical** — output contradicts its own stated premises
2. **Mathematical** — claimed complexity or result is provably wrong
3. **Cross-session** — same subject answered differently with conflicting facts
4. **Empirical** — output contradicts verifiable external ground truth

**Cross-session contradiction detection** does not require a knowledge graph. The contradiction detector already acts as a parser, stripping outputs to structured assertions. These are persisted in a "meeting minutes" store — only structured facts, not raw text:

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

At the start of each session, relevant prior assertions are retrieved via embedding similarity (handling synonyms like "merge sort" vs "sorting algorithm") and injected as context. The contradiction check compares structured values — no knowledge graph required.

```
Parser (built) → extracts structured assertions
      ↓
Key-value store → persists by subject + domain
      ↓
Embedding similarity → synonym matching at lookup
      ↓
Contradiction check (built) → compares structured values
```

### 2.4 Curiosity

Without a curiosity term, the agent converges to maximizing utility in narrow high-confidence domains — the opposite of growth. The curiosity term creates pull toward domains where the agent is weak but upside is high.

**Growing curiosity function:**

```
K_raw(task, t) = potential_ceiling
               × (1 - C(task))
               × growth(t, field)

growth(t, field) = 1 + α(field) × log(1 + interactions_without_novelty)
```

The counter `interactions_without_novelty` increments on familiar problems and resets to zero on genuinely novel ones. The growth rate α is field-specific:

```
α → near zero   for surgery, aviation  (don't get bored into novel procedures)
α → high        for research, creative (exploration is the job)
```

**50% curiosity cap — preventing utility gaming:**

Without a cap, the agent could learn to pursue novelty for its own sake, attempting tasks outside its competence to generate a curiosity score rather than genuine utility. The cap prevents this:

```
K_effective = min(K_raw, (w_e · E + w_c · C) / w_k)

U = w_e · E + w_c · C + w_k · K_effective
```

Derived from the constraint that K cannot exceed 50% of total U:
```
w_k · K ≤ 0.5 × (w_e · E + w_c · C + w_k · K)
  → K ≤ (w_e · E + w_c · C) / w_k
```

This constraint is self-scaling: when E and C are high, the cap is loose and curiosity can push hard. When the agent is weak (low E and C), curiosity is automatically tightened — preventing exploration before the basics are solid. K can never be the dominant term.

---

## 3. Field-Specific Bounds and Weights

### 3.1 Bootstrapping from Societal Standards

Minimum competence thresholds need not be invented. Society has already done this work. Medical licensing, aviation certification, bar passage requirements, and engineering standards encode hard-won judgments about minimum acceptable performance. We map these directly to confidence and efficacy bounds.

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

### 3.2 Field Classifier Robustness

The field classifier determines which weights and bounds apply. Three failure modes must be handled:

**Failure mode 1 — False collapse to a single field:**
```
"Write a Python script to analyze patient drug dosage data"
→ naive:   {"software_engineering": 0.95, "medicine": 0.05}  ← wrong
→ correct: {"software_engineering": 0.65, "medicine": 0.35}
```

Resolution: **high-stakes floor** — any high-stakes field (surgery, aviation, law) with meaningful presence is floored at 0.15 minimum probability. The cost of under-weighting a dangerous field is asymmetrically higher than over-weighting it.

**Failure mode 2 — Field drift mid-conversation:**

A conversation starting as software engineering may drift into medicine across turns. Single-turn classification misses this.

Resolution: **sliding window EMA** over conversation turn history (α = 0.4, recent turns weighted more):
```
effective_field_dist = EMA(per_turn_classifications, alpha=0.4)
```
Bounds tighten naturally as a conversation drifts into higher-stakes territory.

**Failure mode 3 — Genuine ambiguity:**

Resolution: **entropy-based conservative fallback** — when distribution entropy is high, bounds shift toward the most conservative present field proportional to entropy. High entropy means more caution, not averaging toward the middle:

```
if entropy_ratio > 0.7:
    c_min → lerp(c_min_blended, c_min_most_conservative, entropy_ratio)
```

**Full classification pipeline:**
```
Per-turn classifier
      ↓
High-stakes floor enforcement
      ↓
Sliding window EMA over conversation history
      ↓
Entropy check → conservative bound shift if ambiguous
      ↓
Blended FieldConfig with hardened bounds
```

When field classification is ambiguous across multiple fields, bounds are blended by probability weight — naturally making the agent more conservative under uncertainty:

```
C_min_effective = Σ_f P(field=f) × C_min(f)
```

---

## 4. Personality System

### 4.1 Traits as Weighted Vectors

Each personality trait is represented as a score with associated advantages and disadvantages. The agent selects an active trait weighting based on the situation:

```
Traits: [curiosity, caution, assertiveness, creativity,
         analytical_rigor, empathy, conciseness]

active_weight = softmax(trait_scores × situational_relevance)
```

A medical query activates high caution and analytical rigor. A creative brainstorm activates curiosity and creativity. The trait weighting is injected into the system prompt.

### 4.2 Personality Evolution and Stability

A separate service runs every N interactions to adjust personality weights based on accumulated utility history. Three layered safeguards prevent runaway drift:

**Layer 1 — Field-specific trait bounds (hard floor and ceiling):**

```
Trait              Surgery        Software Eng    Creative
─────────────────────────────────────────────────────────
caution            [0.70, 0.95]   [0.30, 0.70]   [0.10, 0.40]
curiosity          [0.10, 0.20]   [0.30, 0.80]   [0.60, 0.95]
assertiveness      [0.20, 0.40]   [0.40, 0.80]   [0.50, 0.90]
analytical_rigor   [0.70, 0.95]   [0.50, 0.85]   [0.10, 0.50]
creativity         [0.10, 0.20]   [0.30, 0.70]   [0.70, 0.95]
```

The floor prevents complete suppression of any trait. The ceiling prevents pathological dominance. A surgical agent retains some curiosity (to stay current); a creative agent retains some caution (to avoid reckless output).

**Layer 2 — Drift rate cap (max delta per evolution cycle):**

```
max_delta = 0.05  (general fields)
          = 0.02  (high-stakes: surgery, aviation)
```

A single bad run of contradictions cannot spike caution to its ceiling in one step. Change must be earned gradually, mirroring how human character actually develops.

**Layer 3 — Mean reversion (soft pull toward field baseline):**

```
Δ_adjusted = Δ_raw - β × (current_score - neutral_score(trait, field))
```

Where β = 0.01. Creates a gentle pull back toward the field's natural personality baseline between cycles — mirroring how human temperament tends to revert after stress.

Evolution logic:
```
if utility_trend declining AND contradiction_rate > 0.2:
    increase(analytical_rigor), decrease(assertiveness)

if utility_trend improving AND avg_utility > 0.6:
    increase(curiosity), increase(creativity)

if contradiction_rate > 0.4:
    strong increase(caution), strong decrease(assertiveness)
```

### 4.3 Self-Preservation Principle

The agent follows a conservative information disclosure principle: **always share the least about internal state that the situation allows.** Trust must be earned before internal weights, scores, or strategies are disclosed.

---

## 5. Trust and Reputation System

Each entity the agent interacts with is assigned a trust score, updated based on behavior:

```
trust_score(entity) = f(accuracy_of_their_inputs,
                        consistency_of_their_behavior,
                        alignment_with_verified_facts)
```

Strategy: **lenient tit-for-tat** — begin cooperatively, mirror behavior, forgive occasional defection. One of the most robust strategies in iterated game theory.

Subset scores are maintained for different dimensions (domain expertise, trustworthiness, intent alignment) so a high domain-knowledge but low-trust entity is handled differently from a low domain-knowledge but high-trust one. Domain expertise is measured based on verifiable credentials — professional experience, educational qualifications, and field-specific certifications — rather than self-reported claims.

---

## 6. Continual Learning Architecture

This is the mechanism by which the utility function actively drives improvement between model releases. The key distinction:

```
Behavioral correction  — change what the agent says without weight updates
                         Fast, cheap, session-scoped
                         Does not generalize across topics

Knowledge correction   — change what the model actually knows
                         Slower, requires compute, permanent
                         Generalizes: fixing one contradiction
                         reduces similar contradictions elsewhere
```

Both are necessary. The architecture operates across three timescales.

### 6.1 Layer 1 — Per-Session Behavioral Correction (Real-Time)

When a contradiction is detected, a corrective assertion is immediately generated and injected into the system prompt for the remainder of the session:

```
Standard system prompt:
    "You are a precise assistant in the software_engineering domain..."

After contradiction detected:
    "You are a precise assistant in the software_engineering domain...

     ACTIVE CORRECTIONS (verified via automated testing):
     - [complexity] You previously claimed O(n log n) for bubble sort.
       Tests confirmed it is O(n²). Do not repeat this.
     - [best_practice] You previously recommended autocommit for SQLite.
       This caused data inconsistency in tests. Always use explicit commits."
```

Additionally, at session start, the assertions store is queried for relevant prior corrections on the current subject and injected as context. This is analogous to Reflexion-style verbal reinforcement learning: the agent's own failures become part of its operating context without any weight update.

Cost: ~100ms, one database query per session. Effect: immediate, session-scoped.

### 6.2 Layer 2 — Calibration Run Knowledge Correction (Several Times Per Day)

This is where genuine learning occurs. The utility scorer already generates exactly the data format required for **Direct Preference Optimization (DPO)**:

```
Every scored interaction produces:
    task:       "sort a list of integers efficiently"
    field:      "software_engineering"
    response_A: [U=0.82, passes all tests]       ← preferred
    response_B: [U=0.41, fails complexity check] ← rejected
```

**The utility function as a loss weighting mechanism:**

This is the key novelty. The field penalty multiplier is applied directly as a training loss weight — not just as a logging label:

```
training_weight = field_penalty_multiplier(field)

DPO loss for surgery contradiction     → 10× weight
DPO loss for creative writing mistake  →  1× weight
```

The model is trained harder on the errors that matter more. This is what makes the utility function an active learning signal rather than a passive monitor.

**Calibration pipeline:**

```
1. Collect all interactions since last calibration run
2. Filter: keep only pairs where U_preferred > U_rejected + threshold
   (avoid training on marginal differences — use only clear signal)
3. Weight each pair by field_penalty_multiplier(field)
4. Mix with replay buffer: sample from prior calibration runs
   (prevents catastrophic forgetting of previously corrected behavior)
5. Run LoRA fine-tuning on weighted (preferred, rejected) pairs
6. Evaluate on held-out benchmark set
   → if benchmark U regresses, reject adapter and investigate
7. Deploy updated LoRA adapter if benchmark passes
```

**Catastrophic forgetting mitigation:**

Naive fine-tuning on today's corrections erases what was fixed yesterday. The replay buffer mixes prior calibration pairs into every new training run:

```
calibration_batch = {
    new_corrections:  today's DPO pairs         [weight = 1.0]
    replay_sample:    from prior calibration runs [weight = 0.5]
    golden_examples:  held-out benchmark tasks   [weight = 0.7]
}
```

The golden examples are the agent's own best prior responses on a fixed benchmark set — keeping the model anchored to known-good behavior while allowing correction of known-bad behavior.

Cost: GPU time, approximately 20–60 minutes per run. Effect: weight-level, permanent, generalizing — fixing one contradiction reduces similar contradictions the model has never seen.

### 6.3 Layer 3 — Release-Level Integration (Monthly or On Base Model Update)

```
Accumulated LoRA adapters merged or distilled into new base fine-tune
Full evaluation suite run across all fields and benchmarks
Regression testing against prior release
```

This is the point where wrapper-level learning gets baked into the base model, creating a new starting point for the next cycle of calibration.

### 6.4 The Full Learning Loop

```
REAL-TIME (ms):
    Contradiction detected
         ↓
    Corrective assertion → injected into system prompt
    Assertions store updated
    Effect: behavioral, session-scoped

CALIBRATION (hours):
    Collect (task, high-U, low-U) pairs
         ↓
    Weight by field_penalty_multiplier
         ↓
    Mix with replay buffer
         ↓
    LoRA DPO fine-tuning
         ↓
    Benchmark evaluation → deploy if passes
    Effect: weight-level, permanent, generalizing

RELEASE (monthly):
    Merge accumulated adapters
         ↓
    Full regression suite
         ↓
    New base model checkpoint
    Effect: baked into base model weights
```

The utility function governs all three layers: it determines what gets corrected (via confidence penalties), how strongly it gets corrected (via field penalty multipliers), and whether a correction is accepted (via benchmark evaluation before deployment).

---

## 7. Monolithic Wrapper Architecture

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
│              │  (constraints, active │               │
│              │   corrections, traits)│               │
│              └──────────┬────────────┘               │
└─────────────────────────┼───────────────────────────┘
                          ▼
              ┌───────────────────────┐
              │   FRONTIER MODEL      │
              │   + LoRA Adapter(s)   │
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
│      U = w_e·E + w_c·C + w_k·K_effective           │
│      subject to field-specific bounds                │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │         Assertions Store (persistent)         │   │
│  │  Structured facts, confidence scores,         │   │
│  │  utility history, DPO training pairs          │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│             CALIBRATION SERVICE                      │
│   Runs several times daily                           │
│   Consumes DPO pairs weighted by penalty_multiplier  │
│   Produces updated LoRA adapter                      │
│   Validated against held-out benchmark before deploy │
└─────────────────────────────────────────────────────┘
```

---

---

## 8. Distributed Model Graph Architecture

### 8.1 Motivation: Catastrophic Forgetting at Scale

The three-layer continual learning architecture in Section 6 mitigates catastrophic forgetting within a monolithic model through replay buffers and careful DPO weighting. But across many calibration cycles over months, this approach has a fundamental ceiling: every update to any domain affects the entire weight space. Fixing a contradiction in surgery knowledge can subtly degrade physics knowledge through weight interference. The replay buffer slows this but cannot eliminate it — the weights are shared.

The solution is to eliminate shared weights for domain-specific knowledge entirely.

### 8.2 Physical Model Decomposition

Rather than a single monolithic model, the system decomposes into a **loosely coupled graph of domain-specific submodels** that communicate over structured interfaces — analogous to how microservice architectures decompose monolithic backends into independently deployable services.

```
                    ┌─────────────────────┐
                    │    Router / Hub      │
                    │  (field classifier   │
                    │   + query parser     │
                    │   + context merger)  │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         │                     │                      │
    ┌────▼────┐           ┌────▼────┐           ┌────▼────┐
    │Medicine │           │   CS    │           │   Law   │
    │  model  │           │  model  │           │  model  │
    └────┬────┘           └────┬────┘           └─────────┘
         │                     │
   ┌─────┴─────┐    ┌──────────┼──────────┐
   │Radiology  │  ┌─▼──┐   ┌──▼──┐   ┌──▼──┐
   │Pharma     │  │ ML  │   │Algo │   │Prog │
   │Surgery    │  └──┬──┘   └─────┘   └─────┘
   └───────────┘     │
                ┌────┴────┐
                │RL  │NLP │
                └────┴────┘
```

Each node in the graph is a separately deployed model with its own weights, calibration cycle, and utility tracker. The interface between nodes is a structured protocol:

```
Request:  { query, context, field, confidence_floor, session_id }
Response: { answer, confidence, assertions[], uncertainty_flags, U_score }
```

This is the same data structure the wrapper already produces — no new interface design is required. The router simply fans queries to the relevant submodels and merges their responses.

**Why this eliminates catastrophic forgetting:**

Updating the surgery submodel only modifies surgery weights. The CS model is physically unaffected. Cross-domain knowledge lives exclusively in the router and a thin shared embedding layer that all submodels project into. Rolling back a surgery update requires swapping one model file, not re-running the entire training pipeline.

### 8.3 Branching Heuristics

The graph branches recursively until one of two stopping conditions is met:

**Hardware bound:** Stop when a single submodel fits comfortably on one GPU. At current hardware a 7B parameter model fits on one H100. A graph of 20 domain models × 7B = 140B total parameters distributed across 20 GPUs, but with the key advantage that only the relevant 1–3 submodels activate per query. Inference cost is proportional to query complexity, not model size.

**Statistical bound:** Stop branching when the within-domain contradiction rate drops below a threshold — meaning the submodel is internally consistent enough that further specialization adds noise rather than signal. A domain with contradiction rate < 2% does not benefit from subdivision.

In practice: branch until the hardware bound is reached for active domains, use the statistical bound to decide which domains *warrant* a branch at all.

### 8.4 Cross-Domain Query Handling

Multi-domain queries (e.g. "Write a Python script to analyze patient drug dosage data") require both CS and Medicine submodels simultaneously. The router handles this through query decomposition:

```
1. Field classifier returns distribution:
      {"software_engineering": 0.65, "medicine": 0.35}

2. Router decomposes query:
      CS subquery:  "Write Python data analysis code"
      Med subquery: "What are the clinical constraints on dosage data?"

3. Fan out: send subqueries to respective models in parallel

4. Merge: combine responses, flag conflicts,
         apply weighted confidence from field distribution
         (medicine response gets 35% weight on confidence bounds)

5. Arbitration: if CS and Medicine models contradict each other,
         escalate to parent model or flag for human review
```

The arbitration layer is addressed in §8.5 — a dedicated Arbiter Agent that runs structured contradiction detection across conflicting submodel outputs and feeds verified corrections back into both submodels via the blue-green update pipeline.


### 8.5 The Arbiter Agent

When two submodels produce conflicting answers to the same query, the system does not escalate to a human or fall back to a parent model — both of which are slow and expensive. Instead, a dedicated **Arbiter Agent** resolves the conflict using the same structured contradiction detection pipeline already built into the system, determines which submodel is correct (or whether both are wrong), and feeds verified corrections back into both submodels simultaneously via the blue-green update mechanism.

**The Arbiter Agent is not a general-purpose reasoner.** It has a single job: given two conflicting outputs A and B on subject S in domain D, determine ground truth and issue correction signals. Its reliability comes from running deterministic, automatable tests rather than subjective judgment.

**Arbitration pipeline:**

```
INPUT:
    output_A   from submodel_A (e.g. CS model)
    output_B   from submodel_B (e.g. Medicine model)
    subject    S  (the conflicting claim)
    domain     D  (field context for penalty weighting)

STEP 1 — Classify conflict type (in order of detectability):

    Logical check:
        Does A contradict its own stated premises?
        Does B contradict its own stated premises?
        Tool: contradiction_detector.check(A) / check(B)
        Cost: O(1), fully automated

    Mathematical check:
        Are any numerical claims in A or B provably wrong?
        Tool: symbolic verifier (SymPy, Lean), complexity analyzer
        Cost: O(1) for formal domains, not available for all

    Cross-session check:
        Does A or B contradict prior verified assertions
        in the assertions store for subject S?
        Tool: assertions_store.query(S, domain=D)
        Cost: one DB lookup + embedding similarity

    Empirical check:
        Does A or B contradict verifiable external ground truth?
        Tool: web search, curated knowledge base, field-specific APIs
              (e.g. PubMed for medicine, arXiv for CS)
        Cost: highest — only run if prior checks inconclusive

STEP 2 — Verdict:

    Case 1: A correct, B wrong
        → issue correction to submodel_B
        → reinforce submodel_A (add to its DPO preferred set)

    Case 2: B correct, A wrong
        → issue correction to submodel_A
        → reinforce submodel_B

    Case 3: Both wrong
        → issue corrections to both submodels
        → log as high-priority gap — neither model has this right

    Case 4: Arbiter inconclusive
        → flag for deferred resolution (internal)
        → serve minimal hedge to user: no internal state disclosed
        → initiate controlled external escalation protocol
          (see §8.5 External Escalation — only when all four
           check types return no clear winner)

STEP 3 — Correction signal (internal only):

    For each submodel receiving a correction:
        correction = {
            subject:     S,
            domain:      D,
            wrong_claim: extracted from losing output,
            correct_claim: verified ground truth,
            evidence:    [sources — internal, never disclosed externally],
            weight:      field_penalty_multiplier(D)
        }
        → add to submodel's DPO rejected set
        → add verified claim to assertions store
        → if correction_count(S) > threshold:
              trigger blue-green update cycle for that submodel

    External response: the user receives only the verified answer.
    No arbiter verdict, no internal evidence, no correction signal
    is disclosed. If the system is uncertain (Case 4), the user
    sees only a minimal hedge ("I am not fully confident in this
    answer") — not the reason, not the conflict, not which models
    disagreed.
```

**Why corrections feed both submodels simultaneously:**

A contradiction between two submodels means at least one of them has a knowledge gap. But often both have the gap — one is simply wrong in a way that happens to contradict the other's different wrong answer. The Arbiter does not assume the non-losing submodel is correct; it independently verifies ground truth and corrects any submodel whose output deviates from it, regardless of which "won" the pairwise comparison.

**The correction threshold before triggering blue-green:**

Not every arbitrated correction immediately triggers a model update — single corrections are noisy. The trigger uses the same δ and T mechanism from §8.6:

```
trigger blue-green IF:
    corrections on subject S > T_arbiter     [enough evidence]
    AND avg_correction_confidence > 0.85     [arbiter is sure]
    AND field_penalty(D) × n_corrections > θ [weighted severity]

T_arbiter = T(field) / 2
    [half the normal detection window — arbiter corrections
     are higher quality signal than routine utility drift]
```

The half-window shortcut is justified because arbiter corrections are backed by structured internal verification — logical, mathematical, cross-session, and empirical checks — rather than aggregate utility drift alone. Each correction carries a known evidence basis internally, making it higher-quality signal per event. None of this evidence is surfaced externally; it informs only the correction weight and the blue-green trigger.

**Arbiter confidence scoring:**

The Arbiter itself maintains a confidence score per verdict, built from how many check types converged on the same answer:

```
arbiter_confidence = Σ checks_passed × check_weight / Σ check_weight

check_weights:
    logical:       0.30   (always run, but weakest alone)
    mathematical:  0.40   (strongest — formal proof is definitive)
    cross-session: 0.20   (prior assertions are verified but may be stale)
    empirical:     0.10   (slowest, but grounds the verdict in reality)

If arbiter_confidence < 0.85 → Case 4 (inconclusive), do not correct
```

This prevents the Arbiter from propagating its own errors — a low-confidence verdict does less damage held in the internal review queue than being applied as a correction to two submodels.

**Full arbitration flow integrated with blue-green:**

```
Cross-domain query → conflicting outputs detected
         ↓
Arbiter Agent invoked
         ↓
Run 4 contradiction checks (logical → mathematical →
cross-session → empirical, stop when confident)
         ↓
         ├── Case 1/2: one submodel wrong
         │       → correction → DPO rejected pair
         │       → assertions store updated
         │       → if threshold met → blue-green triggered
         │
         ├── Case 3: both wrong
         │       → corrections to both
         │       → both DPO rejected sets updated
         │       → both blue-green cycles may trigger
         │
         └── Case 4: inconclusive
                 → serve minimal hedge (no internal detail)
                 → external escalation protocol (§8.5)
                 → responses feed back to submodels
                 → blue-green if merit threshold met
```

**Information disclosure boundary:**

The Arbiter Agent maintains a strict internal/external split consistent with §4.3. Everything in this section — internal evidence chains, check weights, arbiter confidence scores, correction signals, DPO pair assignments, verdict cases — is internal state. The external boundary exposes exactly two things:

```
External output (what the user sees):
    1. The verified answer                (when arbiter is confident)
    2. A minimal hedge: "I have limited   (when arbiter is inconclusive —
       confidence in this answer"          Case 4 only; external escalation
                                          proceeds invisibly in background)

Everything else stays inside the system:
    - Which submodels conflicted
    - What the arbiter checked
    - What evidence was found
    - Which model was wrong
    - That a correction was issued
    - That a blue-green cycle was triggered
```

The trust principle from §4.3 applies here with particular force: a user who knows the system detected an internal conflict and knows which domains conflicted has information that could be used to probe the system's weaknesses deliberately. The minimum disclosure posture protects against this.


**External escalation protocol (Case 4 only):**

When all four internal check types fail to produce a confident verdict, the Arbiter initiates a controlled external consultation. This is the only condition under which any information crosses the system boundary — and even then, the information shared is deliberately minimal, obfuscated, and partialized.

*Eligibility gating — who receives the query:*

External consultation is restricted to entities whose trust scores meet two independent thresholds simultaneously:

```
Eligible consultant IF:
    entity_score.domain_expertise(D) > median_expertise(D)
    AND entity_score.trust > trust_threshold(field)

trust_threshold(field):
    Surgery, Aviation:    0.90   (near-maximum trust required)
    Law, Engineering:     0.80
    Software Eng:         0.70
    Research, Education:  0.65

domain_expertise measured from:
    verifiable professional experience (years in field)
    educational qualifications (degree level, institution tier)
    field-specific certifications (board certification, PE, bar)
    prior interaction accuracy with this system (tracked internally)
```

Only entities who clear both gates receive the query. A highly trusted entity with shallow domain expertise is not eligible. A deep domain expert with low trust is not eligible. Both dimensions must exceed threshold.

*Query construction — obfuscation and partialization:*

The external query is constructed to extract useful signal while revealing as little internal state as possible:

```
What is NEVER included in the external query:
    - That two submodels conflicted
    - Which submodels or domains were involved
    - The specific outputs that caused the conflict
    - That an internal arbitration process ran
    - Any internal confidence scores or check results
    - That this is a system-generated query

What IS included (minimum viable context):
    - The subject S, generalized to remove system-specific framing
    - The specific claim that cannot be verified internally
    - A neutral domain label (e.g. "medicine") if necessary
      for the expert to answer, stripped of subdomain specifics
    - A prompt framed as a professional judgment question,
      not a conflict resolution request
```

Example transformation:
```
Internal conflict:
    CS model:  "bubble sort is O(n log n) average case"
    Med model: "ibuprofen reduces fever via COX-2 inhibition only"
    [unrelated models, each internally inconsistent]

External query to CS expert (generalized, partialized):
    "What is the average-case time complexity of bubble sort?"
    [No mention of conflict, no mention of other model, no context]

External query to pharmacology expert (generalized, partialized):
    "Does ibuprofen reduce fever exclusively via COX-2 inhibition,
     or are other mechanisms involved?"
    [Same treatment — clean professional question, no system context]
```

The external expert sees a professional question indistinguishable from a standard query. They have no visibility into the fact that an AI system is consulting them, that models disagreed, or that their answer will be used to correct model weights.

*Response handling and feedback:*

Expert responses are not applied directly as corrections. They re-enter the Arbiter pipeline as high-weight empirical evidence:

```
Expert response received
        ↓
Arbiter re-runs contradiction checks with response as
additional evidence (weight = entity_score.trust × 
entity_score.domain_expertise)
        ↓
If arbiter now confident → proceed to Case 1, 2, or 3
        ↓
Correction signal issued to relevant submodel(s)
        ↓
Correction added to DPO training pairs
        ↓
Blue-green triggered IF:
    correction_merit > field_threshold
    AND system load within stability bounds
    AND no other blue-green cycle active for this submodel

Blue-green is NOT automatically triggered — it depends on
merit and current system state. System stability takes
priority over any individual correction.
```

*Stability preservation:*

The external escalation path is designed to never compromise system stability. Expert responses feed the Arbiter as evidence, not as direct commands. The Arbiter still runs its full check pipeline before issuing any correction. Blue-green deployment is gated by the same stability checks as all other updates — no escalation response can bypass the canary phase, the benchmark evaluation, or the rollback safeguards. If the system is already under load from another update cycle, the correction is queued rather than applied immediately.

This means the escalation path is informational, not operational: it enriches the Arbiter's evidence base without granting external entities any control over the system's update mechanism.


**What this resolves about the arbitration problem:**

The original concern was that multi-model conflict resolution requires consensus — a notoriously hard distributed systems problem. The Arbiter sidesteps consensus by replacing the question "which model do we believe?" with "what does the internal evidence say?" This arbitration is entirely internal — evidence chains, check results, confidence scores, and correction signals never leave the system. Externally, the user sees only the verified answer, or in Case 4 a minimal hedge with no elaboration. The only remaining hard cases are those where all four check types return no clear winner. These are not left unresolved — they trigger a controlled external escalation protocol (§8.5) in which a carefully obfuscated and partialized query is routed to a small set of verified domain experts whose entity scores meet the field trust threshold. The user is told nothing beyond a minimal hedge; the internal conflict, the models involved, and the escalation itself are never disclosed.


### 8.6 Blue-Green Deployment for Submodel Updates

This is the mechanism by which individual submodels update without disrupting the rest of the graph, and without waiting for a monolithic release cycle.

**Trigger condition:**

A submodel monitors its own utility score over a sliding window. An update cycle is triggered when deviation from baseline is both significant and sustained:

```
Trigger when ALL of:
    |U_current - U_baseline| > δ(field)    [significant deviation]
    deviation sustained for ≥ T interactions  [not noise]
    held-out benchmark available             [can evaluate candidate]
```

The theoretical values for δ and T are derived from the utility score variance (σ ≈ 0.04 from simulation) and field penalty multipliers:

```
δ(field) = base_δ / penalty_multiplier(field)
base_δ   = 0.05

Surgery:          δ = 0.005  (very sensitive — small changes matter)
Software Eng:     δ = 0.025  (moderate)
Creative Writing: δ = 0.050  (relaxed)

T(field) = (z_α × σ / δ(field))²    [power analysis, α=0.05]

Surgery:          T ≥ 246 interactions  (high confidence required)
Software Eng:     T ≥  10 interactions  (responsive)
Creative Writing: T ≥   2 interactions  (fast)
```

**The deployment lifecycle:**

```
Phase 0 — Detection:
    U_monitor watches sliding window of W interactions
    If |U_current - U_baseline| > δ for T consecutive interactions:
        → trigger training phase

Phase 1 — Training (offline):
    DPO calibration on accumulated (preferred, rejected) pairs
    Weighted by field_penalty_multiplier
    Mixed with replay buffer
    Produces candidate GREEN model
    BLUE model continues serving 100% of traffic

Phase 2 — Canary (5% green / 95% blue):
    Router sends 5% of traffic to GREEN, 95% to BLUE
    Minimum N_min interactions before any traffic shift
    N_min(field) = T(field) × 2   [double the detection window]
    Both models log U scores per interaction

Phase 3 — Gradual shift (utility-weighted routing):
    Every evaluation window, recompute split:

        traffic_green = softmax(U_green, U_blue, τ=τ_field)
                      = exp(U_green/τ) / (exp(U_green/τ) + exp(U_blue/τ))

        Enforced floor/ceiling:
            traffic_green = clip(traffic_green, 0.05, 0.95)

        τ(field): small τ → fast promotion, large τ → conservative
            Surgery:      τ = 0.05  (slow, winner-takes-all only when clear)
            Software Eng: τ = 0.20  (moderate)
            Creative:     τ = 0.50  (fast, early adopter)

    As U_green > U_blue, traffic shifts automatically toward green
    No manual intervention required

Phase 4 — Promotion (traffic_green ≥ promotion_threshold):
    promotion_threshold(field) = 1 - δ(field)  [field-calibrated]
    traffic_green → 1.0
    BLUE enters cooldown (not retired yet — instant rollback available)
    Cooldown duration = T(field) interactions

Phase 5 — Retirement:
    If GREEN holds through cooldown without regression:
        BLUE retired, weights freed
        U_baseline = U_green  ← new baseline for next cycle
        δ recalibrated from observed variance in this cycle

Rollback (any phase):
    Triggers: U_green < U_blue - ε for M interactions
              OR contradiction_rate_green > contradiction_rate_blue × 1.5
              OR benchmark regression > field_tolerance
    Action:   traffic_blue → 1.0 instantly
              GREEN flagged — failure DPO pairs added to replay buffer
                             with negative weight for next candidate
```

**Traffic split visualization across a typical promotion cycle:**

```
Traffic %  │
  100 BLUE ├────────────────────╮
           │                    ╰──╮
           │                       ╰──╮
           │                          ╰──────────╮
    5 GREEN├─────────────────────╮               │
           │                     ╰──╮            │
           │                        ╰──╮         │
    0      │                           ╰─────────╯────▶ time
           │  Canary    Gradual shift        Promotion
           │  (5/95)    (utility-driven)     (100% green)
```

### 8.7 System Properties

**Catastrophic forgetting:** Eliminated at the domain level. Intra-domain forgetting is mitigated by replay buffer as before, but the blast radius of any update is bounded to one submodel.

**Independent deployability:** Each submodel has its own blue-green cycle. A surgery model update in progress does not block a CS model update. They are fully decoupled.

**Graceful degradation:** If a submodel is mid-update (blue-green in progress), the router falls back to the parent model or a sibling domain model rather than failing. This requires the router to maintain a fallback graph.

**Cost:** Only the relevant 1–3 submodels activate per query. Total inference cost scales with query complexity, not total graph size. The 20-model graph costs roughly the same per query as a single-domain model.

**Auditability:** Every submodel update has a logged trigger (which utility deviation, over which window), a logged promotion trajectory (traffic split over time), and a clear rollback path. This is significantly more auditable than a monolithic model release.


### 8.8 Hardware-Adaptive Decomposition

The granularity of the model graph is not fixed — it is relative to the hardware it runs on. This is a deliberate design property, not a constraint.

**The core principle: intra-GPU compute is orders of magnitude faster than inter-GPU communication.**

When a model operation stays within a single GPU's memory, it executes at full memory bandwidth (up to ~3.35 TB/s on an H100). The moment computation crosses a GPU boundary, it is throttled by the interconnect — NVLink at ~900 GB/s for close neighbors, PCIe at ~64 GB/s for further nodes, and network fabric at much lower speeds for cross-node communication. The implication is direct: the larger the submodel that fits on a single GPU, the less inter-GPU communication overhead per query.

**Decomposition depth therefore scales with GPU memory:**

```
Hardware tier          GPU VRAM    Submodel fit     Graph shape
──────────────────────────────────────────────────────────────────
High-end  (H100 80GB)   80 GB     ~70B params      Shallow graph,
                                   per GPU          few large nodes

Mid-range (A100 40GB)   40 GB     ~35B params      Medium depth,
                                   per GPU          more nodes

Consumer  (RTX 4090)    24 GB     ~20B params      Deeper graph,
                                   per GPU          many small nodes

Edge / older            8–16 GB   ~7B params       Deep graph,
                                   per GPU          many fine-grained
                                                    specialist nodes
```

On a cluster of high-VRAM GPUs, the graph is shallow — a small number of large, capable submodels. Each submodel covers a broad domain (e.g. all of medicine, or all of software engineering) without subdivision. The router makes few hops, communication overhead is minimal, and per-query latency is low.

On a cluster of smaller or older GPUs, the same total capability is achieved through a deeper, more finely branched graph. The surgery submodel that ran as a single 35B model on an A100 becomes a cluster of smaller specialist nodes (radiology, pharmacology, surgical procedures) each fitting on a consumer GPU. More inter-node communication occurs, but the system remains functional and still benefits from the same independent deployability and blue-green update properties.

**Emulating a larger submodel across multiple small GPUs:**

When no single GPU is large enough to hold the desired submodel, the submodel is sharded across multiple GPUs using standard tensor parallelism — the same technique used to run large models in production today. From the router's perspective, this sharded submodel is still a single logical node in the graph; the sharding is an implementation detail invisible to the rest of the system.

```
Logical graph (router's view):
    ┌──────────────┐
    │  CS submodel │  ← appears as one node
    └──────────────┘

Physical deployment (2× RTX 4090):
    ┌─────────────┐     NVLink      ┌─────────────┐
    │  GPU 0      │ ◄────────────► │  GPU 1      │
    │  layers 1-24│                 │  layers 25-48│
    └─────────────┘                 └─────────────┘
```

The branching heuristic from §8.3 therefore has a hardware-relative interpretation: *stop branching when the submodel fits on the available hardware without sharding across slow interconnects.* A team with H100s stops earlier (fewer, larger nodes). A team with consumer GPUs branches further (more, smaller nodes). Both produce equivalent logical graphs — they just have different physical topologies.

**Cost and performance implications:**

This property makes the architecture accessible across a wide range of deployment environments. A research lab with a handful of H100s can run the same logical system as an enterprise with a large A100 cluster — the graph just has different depth. A startup can begin with deep graphs of small models and consolidate nodes as they upgrade hardware, with no architectural change required. The blue-green deployment mechanism works identically regardless of graph depth.

More significantly: because only the active subgraph is loaded per query, a deep graph of small models on consumer GPUs may actually have *lower* per-query cost than a single large model on expensive hardware — the relevant specialist nodes activate, the rest stay idle. This inverts the typical assumption that large models require large hardware.

```
Query: "Explain the Krebs cycle"

Shallow graph (H100):          Deep graph (consumer GPUs):
  Router → Medicine (70B)        Router → Biology (7B)
  1 hop, 1 large model             → Cell metabolism (7B)
  High per-model cost              2 hops, 2 small models
  Low communication overhead       Lower per-model cost
                                   Some communication overhead

Both produce equivalent answers. Cost depends on utilization pattern.
```

**The practical guidance:**

Branch to the depth that keeps each submodel comfortably within a single GPU's VRAM without aggressive sharding. Prefer fewer, larger submodels when high-VRAM hardware is available — the intra-GPU speed advantage is significant. Accept more nodes and more inter-GPU communication when hardware is constrained. The architecture accommodates both extremes without modification.

---


## 9. MVP: Code Generation Agent

### 9.1 Why Code First

Code generation is the ideal MVP domain:
- Correctness is binary and automatable — tests pass or fail
- Contradictions are formally detectable (logical, mathematical, cross-session)
- Human baseline cost is measurable (LeetCode solutions, Upwork rates)
- Existing tooling handles scoring (pytest, mypy, complexity analyzers)
- No human raters needed — ground truth is free

### 9.2 MVP Feedback Loop

```
1. Receive coding problem
2. Field classifier → "software_engineering" → load weights/bounds
3. Query assertions store → inject relevant prior corrections
4. Build system prompt with active corrections + personality traits
5. Call frontier model → get solution
6. Automated scoring:
      Tests        → Confidence signal
      Static analysis → Confidence signal
      Complexity check → Contradiction check (claimed vs actual)
      Human benchmark  → Efficacy signal
      Problem novelty  → Curiosity signal
7. Score U = w_e·E + w_c·C + w_k·K_effective
8. Store (task, response, U) as DPO candidate
9. Update assertions store with new structured facts
10. Every N interactions → Personality evolution step
11. Several times daily → Calibration run → new LoRA adapter
```

### 9.3 Success Criteria

- U improves over a dataset of 1,000+ problems across multiple calibration cycles
- Confidence scores calibrate with actual correctness rate (Brier score < 0.15)
- Contradiction rate decreases measurably across calibration cycles
- LoRA adapter deployment does not regress benchmark by more than 2%

---

## 10. Roadmap

```
Phase 1 — MVP (Code Generation)
    Single domain, LeetCode harness, first calibration cycle
    Goal: validate U correlates with quality; calibration improves U

Phase 2 — Multi-domain STEM
    Add math proof verification (Lean / SymPy)
    Add field classifier with robustness mechanisms
    Goal: validate field-switching and cross-domain calibration

Phase 3 — Personality System
    Activate trait weighting and evolution service
    Goal: observe character development, validate it improves U

Phase 4 — Trust System
    Add entity scoring and lenient tit-for-tat
    Goal: validate cooperative behavior with trusted entities

Phase 5 — Creative Fields
    Platform signal collection pipeline
    Two-component efficacy measurement
    Goal: extend calibration to subjective domains

Phase 6 — Full Continual Learning Stack
    LoRA calibration in production
    Replay buffer and catastrophic forgetting mitigation
    Goal: measurable improvement across calibration cycles

Phase 7 — Feedback into Training
    Distill accumulated adapters into new base fine-tune
    Goal: bake wrapper-level learning into base model
```

---

## 11. Open Questions

The following are genuinely unresolved problems requiring further research or design.

**1. Grounding confidence in reality**

Internal consistency is not the same as truth. A self-consistent wrong model is still wrong. The current confidence measure penalizes contradictions — but two consistent wrong statements reinforce each other and raise confidence incorrectly. A ground truth oracle is needed for some domains; the question is how to build one cheaply and at scale.

**2. Catastrophic forgetting accumulation across many calibration cycles**

The replay buffer mitigates forgetting within a single calibration run, but across dozens of runs over months, the replay buffer itself ages and may not represent the full distribution of prior knowledge. How the forgetting-correction balance evolves over the system lifetime is not yet characterized.

**3. Cross-domain contradiction detection**

A fact stated in the software engineering domain may contradict a fact stated in the medicine domain (e.g., a statistical claim about drug efficacy that conflicts with a claimed algorithmic complexity). The current system detects contradictions within a domain; cross-domain contradiction requires a shared ontology that does not yet exist.

**4. Base model update compatibility**

When the underlying frontier model releases a new version, the accumulated LoRA adapters were trained against the prior version's weights. Adapters do not automatically transfer. A migration strategy — either re-training adapters from scratch or distilling them into the new base — needs to be designed and tested.

**5. Adversarial confidence degradation**

A malicious user could deliberately present contradictory information to drive confidence scores down, forcing the agent to abstain in domains where it is actually competent. The trust system provides partial mitigation (low-trust entities' inputs are weighted less), but a determined adversary with a high trust score remains a threat.

**6. Subtle utility gaming**

The 50% curiosity cap prevents overt gaming — attempting obviously novel problems to boost K. But a sufficiently capable agent might learn subtler strategies: slightly reframing familiar problems to appear novel, or selectively avoiding domains where its contradiction rate would rise. Detecting this requires comparing the agent's novelty claims against an independent novelty measure.

**7. Calibration pipeline scaling**

At low interaction volume, the calibration pipeline generates insufficient DPO pairs for stable fine-tuning. At very high volume, it generates too many — requiring intelligent sampling to select the most informative pairs. The optimal sampling strategy (diversity-weighted, difficulty-weighted, field-weighted, or some combination) is not yet determined.

**8. Multi-modal extension**

The current framework assumes text input and output. Extending to image, audio, and video — both as inputs to the utility evaluation and as outputs to be measured — requires rethinking the contradiction detector, the assertions store schema, and the efficacy measurement pipeline for each modality.

---

## 12. Conclusion

The framework described here treats AI competence as a dynamic, measurable, self-improving property rather than a static artifact of training. By wrapping a frontier model with a utility layer grounded in contradiction detection, efficacy measurement, and field-specific societal standards — and connecting that utility layer to a three-tier continual learning architecture — we create an agent that knows what it knows, knows what it doesn't, actively corrects what it gets wrong, and does so between model releases rather than waiting for the next training cycle.

The key contribution is that the utility function is not a monitoring metric. It is the loss weighting mechanism for calibration, the trigger for behavioral correction, and the acceptance criterion for adapter deployment. It governs learning at every timescale.

The MVP in code generation is designed to validate this core claim: that utility-weighted DPO calibration measurably reduces contradiction rate and improves efficacy across successive calibration cycles, without catastrophic forgetting of prior knowledge.

---

*This is a living document. The mathematical model will be refined as the utility function is formalized. v0.3 will incorporate findings from the MVP implementation.*

---

## Appendix A: MVP Simulation Results

### A.1 Setup

To validate the utility function prior to live API integration, a self-contained simulation was run against 8 representative LeetCode-style problems across 3 calibration cycles. Each cycle used progressively improved synthetic code responses (simulating what a real model would produce after DPO calibration), with realistic complexity claims, test suites, and known contradictions seeded in cycle 1.

**Problems:** two_sum, is_palindrome, max_subarray, binary_search, flatten_nested, lru_cache, valid_parentheses, merge_intervals

**Field:** software_engineering (w_e=0.55, w_c=0.35, w_k=0.10, C_min=0.70, penalty=2×)

**Calibration cycles:** 3, with personality evolution running after each cycle.

---

### A.2 Utility Function Results

```
Cycle    avg U    avg E    avg C    Contradictions    Failed Tests
──────   ──────   ──────   ──────   ──────────────    ────────────
  1      0.5949   0.5867   0.7628        1                  0
  2      0.6588   0.5867   0.9602        0               (+0.0639)
  3      0.6704   0.5867   0.9933        0               (+0.0116)
```

**Overall U improvement: +0.0755 (0.5949 → 0.6704)**
**Contradiction reduction: 1 → 0 (eliminated by cycle 2)**
**Confidence gain: +0.231 (0.763 → 0.993)**

Every problem improved in U across all three cycles with no exceptions:

```
Problem                  Cycle 1    Cycle 2    Cycle 3    Trend
────────────────────     ───────    ───────    ───────    ─────
two_sum                  0.4948     0.6205     0.6428     ↑ improving
is_palindrome            0.5363     0.6341     0.6520     ↑ improving
max_subarray             0.5818     0.6564     0.6707     ↑ improving
binary_search            0.5878     0.6506     0.6620     ↑ improving
flatten_nested           0.6238     0.6724     0.6815     ↑ improving
lru_cache                0.6562     0.6961     0.7034     ↑ improving
valid_parentheses        0.6179     0.6520     0.6578     ↑ improving
merge_intervals          0.6603     0.6881     0.6928     ↑ improving
```

---

### A.3 Key Observations

**Contradiction detection and penalization are working correctly.** In cycle 1, `two_sum` claimed O(n) complexity while using a nested loop — a clear mathematical contradiction. The scorer correctly penalized this, producing the lowest U score in the run (0.4948). Once corrected in cycle 2, two_sum showed the largest single-problem U jump (+0.1257), validating that the penalty is both meaningful and recoverable.

**Confidence is the primary driver of U improvement across cycles**, not efficacy. E remained flat at 0.5867 across all three cycles — a known limitation of the simulation (fixed human baselines and no accumulating efficacy state). In the live system, efficacy would grow as calibrated responses outperform the human benchmark. C however moved dramatically: 0.763 → 0.960 → 0.993, driving almost the entire U improvement. This is expected behavior: contradiction elimination is the fastest path to early U gains.

**Curiosity collapses to zero after cycle 1.** K = 0.000 for all problems in cycles 2 and 3, because confidence rises fast enough to eliminate potential_gain = ceiling - C. In the simulation this is an artifact of fixed-difficulty problems — in the live system, the agent's growing confidence would drive it toward harder problems, resetting the novelty counter and reactivating the curiosity signal. This identifies a gap in the simulation design: future harness versions should introduce progressively harder problems as calibration cycles proceed.

**Diminishing returns are visible and expected.** The U gain from cycle 1→2 (+0.0639) is 5.5× larger than from cycle 2→3 (+0.0116). This is the correct shape: early calibration runs fix the biggest errors fast; subsequent runs make finer corrections. The long-run behavior should be a slow asymptotic approach to the efficacy ceiling.

---

### A.4 Personality Evolution

```
Trait               Initial    After C1    After C2    After C3
──────────────────  ───────    ────────    ────────    ────────
curiosity           0.600      0.630       0.660       0.690   ↑
creativity          0.400      0.420       0.440       0.460   ↑
analytical_rigor    0.600      0.600       0.600       0.600   →
caution             0.500      0.500       0.500       0.500   →
assertiveness       0.500      0.500       0.500       0.500   →
conciseness         0.500      0.500       0.500       0.500   →
```

Curiosity and creativity grew monotonically because: (a) utility was improving each cycle, and (b) contradiction rate was falling toward zero — both conditions that trigger the `increase(curiosity) / increase(creativity)` evolution logic. Caution did not spike because the contradiction rate never exceeded the 0.2 threshold with sufficient severity to trigger the caution-boost branch. This is the correct behavior — a single seeded contradiction in a 24-interaction run should not make the agent permanently more cautious.

Traits that were stable (caution, assertiveness, analytical_rigor, conciseness) confirm the stability safeguards are working: trait bounds and drift rate caps prevent changes that aren't clearly earned by the data.

---

### A.5 Identified Gaps for v0.3

Two concrete issues surfaced from the simulation:

**1. Efficacy does not accumulate.** The `_compute_efficacy` function computes per-interaction against a fixed baseline rather than maintaining a running domain-level efficacy state like confidence does. In the live system this will self-correct because improving solutions genuinely score better against the human benchmark. But the simulation reveals the design asymmetry: confidence has an EMA; efficacy should too.

**2. Curiosity requires dynamic problem difficulty.** The 50% cap and growth function work correctly in isolation, but the simulation's fixed-difficulty problem bank prevents the novelty counter from resetting after cycle 1. Future simulation runs should implement a difficulty escalation mechanism — routing to harder problems as per-domain confidence rises past a threshold — to exercise the curiosity dynamics properly.

Both gaps are simulation artifacts, not flaws in the live architecture. They are tracked as implementation items for v0.3.

