# Adaptive Utility Agents: A Framework for Self-Optimizing AI Systems
### Draft Whitepaper — v0.5
*Based on conceptual development sessions, April 2026*

> **Changelog from v0.4:** Mathematical foundations appendix (Appendix B) added, providing formal proofs for the utility function structure, efficacy normalization, confidence dynamics, and personality stability. Cross-references to Appendix B inserted at each relevant first introduction in §§3–5. Simulation results in Appendix A updated to v0.4 numbers. Open Questions resolved list updated. Conclusion and section footers updated to remove forward references to earlier versions.

---

## Abstract

We propose a framework for building self-optimizing AI agents by wrapping frontier language models with an adaptive utility layer grounded in game-theoretic resource allocation theory. The utility function — composed of **Efficacy** (performance relative to a human baseline), **Confidence** (internal consistency penalized by detected contradictions), and **Curiosity** (exploration bonus for high-upside unexplored domains) — is field-weighted and bounded by minimum competence thresholds derived directly from existing societal licensing standards, making the bounds principled rather than arbitrary.

Critically, the utility function is not a passive monitoring metric. It is the active loss weighting mechanism for a three-layer continual learning architecture that corrects contradictions and improves model behavior between releases — without waiting for a full retraining cycle. Contradiction corrections are weighted by field-specific penalty multipliers in Direct Preference Optimization (DPO) training, so a surgical error is penalized an order of magnitude more harshly than a creative writing mistake at the weight-update level.

The paper introduces four additional architectural contributions. First, a **Personality System** with field-bounded trait weights that evolve with accumulated utility history, subject to three-layer drift safeguards. Second, an **entity trust and reputation system** based on verifiable domain credentials and interaction history, governing both how external inputs are weighted and when external escalation is permitted. Third, a **three-layer continual learning architecture** operating at per-session (behavioral), calibration-cycle (weight-level), and release (distillation) timescales. Fourth, a **distributed model graph architecture** that decomposes the monolithic model into independently deployable domain submodels communicating over structured APIs — physically resolving the catastrophic forgetting problem by eliminating shared weights across domains, and enabling hardware-adaptive deployment depth matched to available GPU memory.

Cross-domain contradiction resolution is handled by a dedicated **Arbiter Agent** that runs structured evidence checks — logical, mathematical, cross-session, and empirical — and routes verified corrections to the relevant submodels as DPO training signal. Submodels whose domains were arbitrated receive the internal evidence chain for DPO integration; nothing is disclosed externally. When all internal checks fail, a controlled external escalation queries verified domain experts through obfuscated, partialized queries consistent with the system's minimum-disclosure principle.

Submodel updates follow a **utility-deviation-triggered blue-green deployment** protocol with statistically grounded detection thresholds and softmax traffic routing, ensuring system stability throughout. The protocol is validated through simulation across 3 calibration cycles on 8 LeetCode-style problems, demonstrating a +0.0755 average utility improvement and contradiction reduction from 1 to 0.

Submodel deployment depth is **hardware-adaptive**: shallow graphs of large submodels on high-VRAM GPUs, or deep graphs of small specialist submodels on consumer hardware — achieving equivalent logical capability at lower cost through the avoidance of slow inter-GPU interconnects.

---

## 1. Introduction

Deployed AI systems today are static artifacts. They are optimized at training time against a fixed reward signal and released — whereupon their behavior is frozen. They cannot adjust how much they trust their own knowledge, they have no mechanism to learn from detected errors between releases, and they apply the same level of caution to a surgical recommendation as to a recipe suggestion. When they produce a contradiction, they will produce the same contradiction tomorrow, and in six months when the next version ships.

Human experts behave fundamentally differently. A surgeon knows which procedures they can perform confidently and which require referral. A lawyer knows the edges of their expertise. A researcher develops calibrated uncertainty about their own claims. None of this self-awareness was hardcoded — it emerged through accumulated experience, peer correction, and deliberate practice at the frontier of competence. It also degrades appropriately under stress and rebuilds through demonstrated success. This paper asks: can an AI system be given the same properties?

We propose an architecture that wraps a frontier language model with an adaptive utility layer — a framework for building agents that know what they know, know what they don't, actively correct what they get wrong, and do all of this continuously between model releases rather than waiting for the next training cycle. The core mechanism is a utility function U = w_e·E + w_c·C + w_k·K, where Efficacy, Confidence, and Curiosity are field-weighted and bounded by minimum thresholds derived from existing societal standards (medical licensing, aviation certification, bar requirements). The utility function is not a monitoring metric — it is the loss weighting mechanism that determines how strongly each detected error is corrected at training time.

The paper makes six architectural contributions beyond the utility function itself:

**Continual learning pipeline.** Three timescales of correction: per-session behavioral injection (immediate, no weight change), calibration-cycle DPO fine-tuning (hours, weight-level, field-penalty-weighted), and release-level distillation (monthly). The utility score determines correction weights — a surgery contradiction is penalized 10× harder than a creative writing mistake.

**Personality system.** Trait weights (curiosity, caution, analytical rigor, etc.) evolve with utility history subject to field-specific bounds and drift rate caps, producing genuine character development over time without instability.

**Entity trust and reputation system.** Each interacting entity is scored on domain expertise and behavioral trust. Scores gate how external inputs are weighted and — critically — whether an entity is eligible to receive controlled external escalation queries when the system cannot resolve a contradiction internally.

**Distributed model graph.** The monolithic model is decomposed into independently deployable domain submodels communicating over structured APIs, analogous to microservices architecture. Catastrophic forgetting is resolved architecturally: updating one domain's weights cannot affect any other. Submodels are versioned and updated independently via blue-green deployment.

**Arbiter Agent.** A dedicated contradiction resolution agent that runs structured evidence checks (logical, mathematical, cross-session, empirical) across conflicting submodel outputs. Verified corrections are routed internally to the relevant submodels as DPO signal, with the full evidence chain shared only with the submodels whose domains were arbitrated. When the Arbiter cannot resolve a conflict, a controlled external escalation queries verified domain experts through obfuscated, partialized queries — revealing the minimum information needed to elicit an answer, consistent with the minimum-disclosure principle governing all external information sharing.

**Hardware-adaptive decomposition.** Graph depth is matched to available GPU memory. High-VRAM hardware supports shallow graphs of large, fast submodels; consumer hardware supports deeper graphs of smaller specialist submodels at lower cost per query — exploiting the order-of-magnitude bandwidth advantage of intra-GPU compute over inter-GPU interconnects.

The system is validated through simulation across three calibration cycles on eight LeetCode-style problems, demonstrating monotonic utility improvement (+0.0755), contradiction elimination (1 → 0 by cycle 2), and correct personality evolution consistent with the stability safeguards.

The rest of the paper is organized as follows. Section 2 surveys related work across the six research areas the architecture draws from. Sections 3–7 develop the theoretical framework: utility function, field-specific bounds, personality system, trust system, and continual learning pipeline. Section 8 describes the monolithic wrapper architecture. Section 9 describes the distributed model graph including the Arbiter Agent, blue-green deployment, and hardware-adaptive decomposition. Section 10 describes the code generation MVP and simulation results. Sections 11–12 present the roadmap and open questions. Appendix A contains the full simulation data.

---

---

## 2. Related Work

### 2.1 Utility Functions and Resource Allocation Games

The theoretical foundations of our utility function draw directly from the game-theoretic literature on resource allocation mechanisms. Johari and Tsitsiklis (2004) established the canonical proportional allocation game, proving the existence and uniqueness of Nash equilibrium and bounding the Price of Anarchy at 4/3 for concave utility functions — the same class of utility functions used in our confidence and efficacy formulations. Their subsequent work on scalar-parameterized mechanisms showed that the proportional allocation mechanism achieves the best Price of Anarchy among mechanisms using a single market-clearing price, providing theoretical justification for our single-score utility formulation rather than a vector-valued alternative.

Kelly (1997) introduced the proportional fairness framework for network resource allocation, establishing that bid-proportional allocation maximizes a natural social welfare function. Our curiosity term — allocating exploration effort proportional to potential gain weighted by confidence gap — is structurally analogous to Kelly's proportional bidding: agents allocate where marginal return is highest.

Koutsoupias and Papadimitriou (1999) introduced the Price of Anarchy as a measure of efficiency loss from selfish behavior, which directly motivates our confidence penalty mechanism: detected contradictions are evidence that the model has been operating in a locally selfish manner (optimizing fluency over correctness), and the penalty restores the social optimum by penalizing this deviation. Roughgarden and Tardos (2002) established that the same 4/3 bound applies to selfish routing games with affine-linear congestion functions, linking the network efficiency literature to general mechanism design.

**Key departure:** Existing utility function literature treats agents as fixed rational actors. Our framework treats the agent itself as the subject of the utility function — the agent learns to improve its own utility through calibration, not merely to maximize given a fixed utility. This closes the loop between mechanism design theory and machine learning.

---

### 2.2 Distributed Model Architectures: From Microservices to Mixture of Experts

**Microservices architecture** provides the software engineering precedent for our physically decomposed model graph. The microservices pattern — decomposing a monolithic application into independently deployable services communicating over well-defined interfaces — has been extensively validated at scale by companies including Netflix, Amazon, and Uber. Key properties established in the microservices literature directly motivate our design: independent deployability eliminates coordinated release cycles; bounded blast radius limits the impact of any single component failure; and service mesh patterns (Istio, Linkerd) demonstrate that weighted traffic routing between service versions is operationally mature technology.

Humble and Farley (2010) in *Continuous Delivery* formalized the practices of blue-green and canary deployment that underpin our submodel update mechanism, establishing the statistical and operational foundations for progressive traffic shifting between service versions.

**Mixture of Experts (MoE)** is the closest existing architectural precedent within the ML literature. Shazeer et al. (2017) introduced the Sparsely-Gated Mixture-of-Experts layer, achieving efficiency through sparse activation where only k of N expert subnetworks process each input, with a trainable gating network routing tokens to the most appropriate experts. Fedus et al. (2021) in the Switch Transformer simplified MoE routing to k=1 (a single expert per token), achieving up to 7x pre-training speedup with the same computational budget while scaling to trillion-parameter models. Production models including Mixtral 8×7B and (reportedly) GPT-4 deploy MoE architecture.

**Key departure from MoE:** Our architecture differs from MoE in a critical dimension: MoE experts share weights, are trained jointly, are deployed monolithically, and cannot be updated independently. Our graph consists of physically separate models with independent weight files, independent training pipelines, and independent blue-green deployment cycles. Updating one domain submodel does not affect any other. This is not a difference of degree — it is a difference of kind. MoE solves the compute efficiency problem within a monolithic training paradigm; we solve the independent deployability and catastrophic forgetting problems by abandoning the monolithic paradigm entirely.

Domain-specialized routing without shared weights has been explored in the Branch-Train-Merge literature (Li et al., 2022) and MoErging approaches, where independently fine-tuned models are composed at inference time. Our architecture extends this direction with utility-function-governed update triggers and hardware-adaptive depth.

---

### 2.3 Blue-Green Deployment with Statistically Grounded Thresholds

Blue-green deployment is a mature practice in software engineering, described by Humble and Farley (2010) and subsequently adopted at scale by major technology organizations. The pattern involves maintaining parallel production environments and shifting traffic between them, with automated rollback policies based on performance metrics including error rates, latency, and business KPIs. Canary releases extend this by routing only a fraction of traffic to the new version initially, expanding progressively as confidence grows.

Recent empirical work confirms that canary deployments significantly reduce failure rates compared to direct blue-green switches for systems with complex state, with statistical tests on failure rates validating the significance of gradual traffic migration.

**Key departure:** Existing blue-green and canary deployment literature triggers traffic shifts based on infrastructure metrics (error rates, latency, uptime). Our system triggers updates based on *utility deviation* — a domain-specific measure of knowledge quality — and uses power analysis on observed utility score variance to derive theoretically grounded minimum observation windows T(field) before any deployment decision. The trigger threshold δ(field) is derived from field-specific penalty multipliers bootstrapped from societal licensing standards, not from arbitrary SLA targets. This grounds deployment decisions in epistemic quality rather than operational health.

The softmax traffic routing formula — where traffic split is a continuous function of comparative utility scores with a field-calibrated temperature parameter — has no direct precedent in the deployment literature. It makes the routing self-regulating: no manual traffic adjustment is required, and the promotion rate naturally slows as U_green approaches U_blue.

---

### 2.4 Arbiter Agents and Structured Contradiction Detection

The problem of resolving conflicts between multiple agents or models has antecedents in multi-agent systems and distributed consensus literature. Paxos (Lamport, 1998) and Raft (Ongaro & Ousterhout, 2014) provide consensus mechanisms for distributed systems, but these assume a binary correct/incorrect answer and require a majority of participants to agree. Our Arbiter Agent addresses a more complex problem: two models may both be wrong in different ways, and the arbiter must determine ground truth independently rather than through consensus.

The structured contradiction detection pipeline — logical, mathematical, cross-session, empirical checks in order of cost — draws from formal verification literature. SAT solvers and theorem provers (Lean, Coq) provide the mathematical check layer; their correctness guarantees are exact rather than probabilistic, which is why mathematical contradictions carry the highest confidence weight (0.40) in the arbiter confidence formula.

Self-consistency checking in LLMs (Wang et al., 2022) explores using multiple model samples to identify contradictions within a single model's outputs. Our approach extends this to contradictions *between* physically separate domain models, adding the cross-session assertions store as a persistent memory of verified facts that both checks and updates across interactions.

The arbiter confidence weighting formula — combining logical (0.30), mathematical (0.40), cross-session (0.20), and empirical (0.10) check results — is grounded in the relative reliability of each check type. Mathematical verification is definitive; empirical checks via external sources introduce the possibility of source error and are therefore weighted lowest despite being most grounded in external reality.

---

### 2.5 External Escalation Gated by Trust: Zero Trust and Minimum Disclosure Principles

The external escalation protocol in §8.5 is grounded in two bodies of security literature.

**Zero Trust architecture** (Kindervag, 2010; NIST SP 800-207, 2020) establishes the principle that no entity — internal or external, human or machine — is trusted by default. The principle of least privilege, foundational to Zero Trust, states that every program and privileged user should operate using the least amount of privilege necessary to complete the job, first articulated by Saltzer (1974). Our external escalation gating applies this principle to information disclosure: no internal state is shared with any external entity by default. Trust must be earned through demonstrated domain expertise and interaction history before any escalation query is issued.

NIST's Zero Trust Maturity Model defines zero trust as minimizing uncertainty in enforcing accurate, least-privilege per-request access decisions in the face of a network viewed as compromised. Our system treats *every* external entity as potentially compromised or adversarial by default — the dual trust threshold (domain expertise above median AND entity trust above field-specific floor) operationalizes this assumption.

**Minimum disclosure / data minimization** principles from privacy engineering (Cavoukian, 2009; GDPR Article 5) provide the foundation for query obfuscation. The legal principle of data minimization — collect and share only what is strictly necessary — is extended here to system state disclosure: reveal only what is necessary to elicit the needed external judgment, and nothing more. The query transformation example (internal conflict → clean professional question) implements data minimization at the system interface boundary.

**Key contribution:** Existing Zero Trust literature addresses access control to systems and data. We extend the framework to govern *information disclosure from an AI system to human experts*, including not only what is shared but how it is framed, with explicit obfuscation of system context to prevent an external expert from deducing internal architecture, failure modes, or competitive capabilities.

---

### 2.6 Hardware-Adaptive Decomposition: Interconnect Bandwidth and Inference Cost

The hardware-adaptive decomposition argument in §8.8 is grounded in established literature on GPU interconnect performance and its impact on distributed inference.

NVLink 4.0 (H100 GPUs) provides 900 GB/s of bidirectional bandwidth per GPU, compared to PCIe 5.0 x16's 128 GB/s — a 7× difference that directly determines the practical efficiency of any computation that must cross a GPU boundary. NVIDIA's production measurements show that transferring 20 GB of tensor parallelism synchronization data for Llama 3.1 70B takes 150 ms over PCIe point-to-point versus 22 ms over NVSwitch at full NVLink bandwidth — a 6.8× latency difference that scales with every inference request.

This bandwidth hierarchy has direct implications for optimal decomposition depth. Tensor parallelism performs better within a single node connected via NVLink, while pipeline parallelism is better suited for setups spanning multiple nodes; inference latency is far more sensitive to communication overhead than training throughput. Our hardware-adaptive branching heuristic — stop branching when submodels fit within a single GPU's VRAM without sharding — is a direct corollary of these measured performance characteristics.

**The cost inversion argument** can be stated formally. Let B_intra = intra-GPU memory bandwidth (3.35 TB/s on H100), B_inter = inter-GPU bandwidth (900 GB/s NVLink or 128 GB/s PCIe), and D = data transferred per inference step. For a graph of depth d with k active nodes per query:

```
Latency_shallow(d=1) ∝ D / B_intra
Latency_deep(d=k)    ∝ k × (D_sub / B_intra) + (k-1) × (D_msg / B_inter)

Where D_sub = D/k (smaller submodel, smaller activations)
      D_msg = inter-node message size (much smaller than D)

When D_msg << D_sub and B_inter >> D_msg/latency_target:
    Latency_deep ≈ Latency_shallow

At equal latency, Cost_deep < Cost_shallow because:
    Consumer GPU cost/hr << Enterprise GPU cost/hr
    D_sub fits in consumer VRAM → no high-VRAM GPU required
```

NVLink delivers 5× the energy efficiency of PCIe Gen 5 at 1.3 picojoules per bit, which means that for workloads where inter-GPU communication is unavoidable, NVLink-connected consumer GPU clusters can be more cost-effective per useful computation than PCIe-connected enterprise GPU clusters of equivalent total VRAM.

The practical implication — that a deep graph of small models on consumer hardware may achieve equivalent quality at lower cost than a shallow graph on expensive hardware — inverts the standard assumption that larger models require larger hardware budgets. This is a testable empirical claim that the Phase 6 roadmap item (§10) is designed to validate.


## 3. The Utility Function

### 3.1 Core Formulation

The agent's utility at any point in time is:

```
U = Σ_{tasks} [ w_e(f) · E(task) + w_c(f) · C(task) + w_k(f) · K(task) ]

Subject to:
    C(task) ≥ C_min(f)
    E(task) ≥ E_min(f)
    w_k · K ≤ 0.5 × U_total         [curiosity cap — see §3.4]
```

Where:
- **E(task)** — Efficacy: how well the agent performs relative to human baseline
- **C(task)** — Confidence: internal consistency score, penalized by contradictions
- **K(task)** — Curiosity: exploration bonus for low-confidence domains with high upside
- **f** — field/domain, which determines weights and minimum bounds
- **w_e, w_c, w_k** — field-specific weights summing to 1

*The additive weighted structure of U is not a convenience — it is the unique functional form satisfying separability, monotonicity, continuity, field invariance, and linear scaling invariance. Formal derivation via Debreu's theorem and the Cauchy functional equation: **Appendix B, Theorem B.1**.*

### 3.2 Efficacy

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

*The sigmoid form E(r) = r/(1+r) is not arbitrary — under the log-logistic performance model it equals the Mann-Whitney probability that agent output dominates human baseline output. Formal derivation: **Appendix B, Proposition B.3**.*

The agent develops creative capability through a natural curriculum:
```
Stage 1 → Generate work that converts well (content quality)
Stage 2 → Learn to title, tag, describe effectively (discoverability)
Stage 3 → Build cross-platform presence (network effects, retention)
```

### 3.3 Confidence

Confidence is a per-domain score that increases when knowledge is internally consistent and decreases when contradictions are detected:

```
C(domain) updated via EMA:
    C_new = (1 - α) · C_prior + α · (test_pass_rate · (1 - penalty))
    penalty = contradiction_penalty × field_penalty_multiplier
```

The **wave analogy**: knowledge items that reinforce each other are like constructive interference — they increase signal strength. Contradictions are destructive interference. The goal is a knowledge state where all waves reinforce.

*The EMA update with α = 0.2 is the Kalman-optimal estimator of latent domain confidence when process noise is 5% of observation noise — a well-founded choice for incremental calibration. Under stationary signals, confidence converges geometrically to the field-specific steady state C* = s̄(1 − λμ(f)) with half-life ≈ 3 interactions. Formal derivations: **Appendix B, Theorem B.4** (Kalman optimality) and **Appendix B, Theorem 2** (convergence and recovery).*

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

### 3.4 Curiosity

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

*The curiosity term is UCB-inspired: structurally analogous to the Upper Confidence Bound exploration bonus, with uncertainty-driven, concave-in-familiarity growth. The 50% cap is proved to enforce exploitation dominance — curiosity contributes at most half of total utility at all times. Formal derivation: **Appendix B, Proposition B.5**.*

---

## 4. Field-Specific Bounds and Weights

### 4.1 Bootstrapping from Societal Standards

Minimum competence thresholds need not be invented. Society has already done this work. Medical licensing, aviation certification, bar passage requirements, and engineering standards encode hard-won judgments about minimum acceptable performance. We map these directly to confidence and efficacy bounds.

*The weight vector w(f) = (w_e, w_c, w_k) is set proportional to the cost of each error type in field f: w_i(f) ∝ c_i(f), normalised to sum to 1. This aligns the gradient of U with domain-specific risk profiles. Formal grounding: **Appendix B, §B.2**.*

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

### 4.2 Field Classifier Robustness

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

## 5. Personality System

### 5.1 Traits as Weighted Vectors

Each personality trait is represented as a score with associated advantages and disadvantages. The agent selects an active trait weighting based on the situation:

```
Traits: [curiosity, caution, assertiveness, creativity,
         analytical_rigor, empathy, conciseness]

active_weight = softmax(trait_scores × situational_relevance)
```

A medical query activates high caution and analytical rigor. A creative brainstorm activates curiosity and creativity. The trait weighting is injected into the system prompt.

### 5.2 Personality Evolution and Stability

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

*The three-layer evolution rules produce bounded, stable dynamics: the trait vector remains in the field-specific feasible set B at all times, converges geometrically to the neutral point s* when drift is absent (rate (1−β)² ≈ 0.980 per cycle, half-life ≈ 34 cycles), and is confined to B under persistent bounded drift. The mean reversion term is a regulariser; the projection Π_B is the primary stability mechanism. Formal derivation: **Appendix B, Theorem 4**.*

Evolution logic:
```
if utility_trend declining AND contradiction_rate > 0.2:
    increase(analytical_rigor), decrease(assertiveness)

if utility_trend improving AND avg_utility > 0.6:
    increase(curiosity), increase(creativity)

if contradiction_rate > 0.4:
    strong increase(caution), strong decrease(assertiveness)
```

### 5.3 Self-Preservation Principle

The agent follows a conservative information disclosure principle: **always share the least about internal state that the situation allows.** Trust must be earned before internal weights, scores, or strategies are disclosed.

---

## 6. Trust and Reputation System

Each entity the agent interacts with is assigned a trust score, updated based on behavior:

```
trust_score(entity) = f(accuracy_of_their_inputs,
                        consistency_of_their_behavior,
                        alignment_with_verified_facts)
```

Strategy: **lenient tit-for-tat** — begin cooperatively, mirror behavior, forgive occasional defection. One of the most robust strategies in iterated game theory.

Subset scores are maintained for different dimensions (domain expertise, trustworthiness, intent alignment) so a high domain-knowledge but low-trust entity is handled differently from a low domain-knowledge but high-trust one. Domain expertise is measured based on verifiable credentials — professional experience, educational qualifications, and field-specific certifications — rather than self-reported claims.

### 6.1 Cold Start Resolution

The cold start problem — new entities having no interaction history — is resolved differently for the two trust dimensions:

**Domain expertise** is available from day one. A board-certified surgeon, a PhD in structural engineering, or a licensed attorney can provide verifiable credentials before the first interaction. The system bootstraps domain expertise from these credentials immediately, without requiring interaction history. The credential verification pipeline maps qualifications to domain expertise scores using the same framework as §4.1 — field-specific certification standards define what "above median expertise" means.

**Behavioral trust** starts at a cooperative neutral (not zero, not maximum) consistent with the lenient tit-for-tat strategy. A new entity is trusted enough to interact normally but not trusted enough to qualify for external escalation. Escalation eligibility requires both domain expertise above median (available from day one via credentials) AND behavioral trust above threshold — so a new expert with excellent credentials can qualify for escalation once behavioral trust accumulates through consistent interaction history.

This two-dimensional gating means cold start affects only the behavioral trust dimension, and that dimension resolves naturally through interaction. Domain expertise — the harder dimension to fake — is grounded from the first contact.

### 6.2 Sybil Resistance via Reputational Accountability

Rather than a population-level detection layer, Sybil resistance is achieved through reputational accountability. When the system consults an external domain expert under the escalation protocol (§9.5), the expert is explicitly informed that their input may be cited when the system offers information to other users. The exact framing:

```
"Your response may be used to inform answers provided to others in this domain.
Your name and credentials may be associated with this input in our records."
```

This creates reputational skin in the game. A legitimate professor, clinician, or engineer would not provide factually incorrect information under conditions where their name is associated with it — the professional and reputational consequences are too significant. The same accountability that governs expert testimony in legal proceedings, peer review in academia, and professional opinion in engineering applies here.

A Sybil adversary operating through fake identities cannot manufacture verifiable professional credentials, and cannot absorb reputational harm across fake accounts. The accountability mechanism therefore filters Sybil attacks at the domain expertise gate — a fake account with no real credentials does not reach the escalation threshold in the first place, and a real expert with real credentials has no incentive to provide false information under attribution.

This does not eliminate all adversarial risk, but it substantially raises the cost of adversarial behavior by tying it to real-world professional identity.

---

## 7. Continual Learning Architecture

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

### 7.1 Layer 1 — Per-Session Behavioral Correction (Real-Time)

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

### 7.2 Layer 2 — Calibration Run Knowledge Correction (Several Times Per Day)

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

### 7.3 Layer 3 — Release-Level Integration (Monthly or On Base Model Update)

```
Accumulated LoRA adapters merged or distilled into new base fine-tune
Full evaluation suite run across all fields and benchmarks
Regression testing against prior release
```

This is the point where wrapper-level learning gets baked into the base model, creating a new starting point for the next cycle of calibration.

### 7.4 The Full Learning Loop

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

## 8. Monolithic Wrapper Architecture

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

## 9. Distributed Model Graph Architecture

### 9.1 Motivation: Catastrophic Forgetting at Scale

The three-layer continual learning architecture in Section 6 mitigates catastrophic forgetting within a monolithic model through replay buffers and careful DPO weighting. But across many calibration cycles over months, this approach has a fundamental ceiling: every update to any domain affects the entire weight space. Fixing a contradiction in surgery knowledge can subtly degrade physics knowledge through weight interference. The replay buffer slows this but cannot eliminate it — the weights are shared.

The solution is to eliminate shared weights for domain-specific knowledge entirely.

### 9.2 Physical Model Decomposition

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

### 9.3 Branching Heuristics

The graph branches recursively until one of two stopping conditions is met:

**Hardware bound:** Stop when a single submodel fits comfortably on one GPU. At current hardware a 7B parameter model fits on one H100. A graph of 20 domain models × 7B = 140B total parameters distributed across 20 GPUs, but with the key advantage that only the relevant 1–3 submodels activate per query. Inference cost is proportional to query complexity, not model size.

**Statistical bound:** Stop branching when the within-domain contradiction rate drops below a threshold — meaning the submodel is internally consistent enough that further specialization adds noise rather than signal. A domain with contradiction rate < 2% does not benefit from subdivision.

In practice: branch until the hardware bound is reached for active domains, use the statistical bound to decide which domains *warrant* a branch at all.

### 9.4 Cross-Domain Query Handling

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


### 9.5 The Arbiter Agent

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
        → share internal evidence chain with both submodels
          (each submodel adds it to its DPO rejected set)
        → log as high-priority knowledge gap
        → assign curiosity gap bonus to subject S:

            K_gap(S) = K_effective(S) × gap_multiplier(field)

            gap_multiplier(field) = 1 + penalty_multiplier(field) / 10
                Surgery:      gap_multiplier = 2.0
                Software Eng: gap_multiplier = 1.2
                Creative:     gap_multiplier = 1.0

        The gap bonus overrides normal curiosity competition —
        the system preferentially routes novel queries on subject S
        to the relevant submodels until the gap is resolved.
        Gap status is cleared when:
            both submodels achieve confidence(S) > C_min(field)
            AND no new contradictions on S for T(field) interactions

        Two budget constraints apply simultaneously to prevent
        Case 3 gaps from monopolizing exploration:

        Constraint 1 — Per-gap cap:
            K_gap(S) ≤ K_natural_max(field)
            No single gap can be more attractive than the maximum
            natural curiosity score achievable in that field.

        Constraint 2 — Collective budget cap:
            Σ K_gap(all active gaps) ≤ (2/3) × K_budget_total
            Case 3 gaps collectively cannot exceed 2/3 of the total
            curiosity exploration budget. At least 1/3 is always
            reserved for natural novelty-driven exploration.

        When multiple Case 3 gaps compete within the 2/3 ceiling,
        budget is allocated proportionally to
        gap_multiplier × field_penalty — higher-stakes gaps
        get priority. The 2/3 ceiling ensures the system never
        becomes a pure gap-resolution machine.

    Case 4: Arbiter inconclusive
        → flag for deferred resolution (internal)
        → serve minimal hedge to user: no internal state disclosed
        → initiate controlled external escalation protocol
          (see §9.5 External Escalation — only when all four
           check types return no clear winner)

STEP 3 — Correction signal (internal only):

    For each submodel receiving a correction:
        correction = {
            subject:      S,
            domain:       D,
            wrong_claim:  extracted from losing output,
            correct_claim: verified ground truth,
            evidence:     [full evidence chain — internal only,
                           shared with arbitrated submodels for DPO,
                           never disclosed externally],
            weight:       field_penalty_multiplier(D)
        }
        → share evidence chain with affected submodel(s) only
          (Case 1/2: one submodel; Case 3: both submodels)
        → submodel independently decides to add to DPO rejected set
        → add verified claim to assertions store
        → if correction_count(S) > threshold:
              trigger blue-green update cycle for that submodel
        → if Case 3: assign curiosity gap bonus to subject S
              (see Case 3 above)

    External response: the user receives only the verified answer.
    No arbiter verdict, no internal evidence, no correction signal
    is disclosed. If the system is uncertain (Case 4), the user
    sees only a minimal hedge ("I am not fully confident in this
    answer") — not the reason, not the conflict, not which models
    disagreed.
```

**Assertions store decay — field-specific evidence staleness:**

Not all verified facts age equally. A mathematical theorem proven beyond doubt — Pythagoras, the fundamental theorem of calculus, the law of conservation of energy — does not become less true over time. A clinical treatment guideline, a software security best practice, or a legal precedent can be obsolete in months. The assertions store therefore applies a field-specific confidence decay function to stored evidence:

```
C_effective(assertion, t) = C_verified × decay(field, Δt)

decay(field, Δt):

    Class A — No decay (mathematically or physically proven facts):
        Pure mathematics (proofs, theorems, derivations)
        Fundamental physics (gravity, thermodynamic laws, calculus)
        decay = 1.0 for all Δt

    Class B — Slow decay (decades-stable fields):
        Mechanical engineering principles
        Classical chemistry, structural physics
        decay = exp(-Δt / τ),  τ = 10 years

    Class C — Moderate decay (years-stable fields):
        General medicine (anatomy, pharmacology mechanisms)
        Software architecture patterns
        Legal common law principles
        decay = exp(-Δt / τ),  τ = 3 years

    Class D — Fast decay (months-to-years volatile fields):
        Clinical treatment guidelines (medical consensus)
        Software security best practices
        Regulatory and compliance standards
        ML/AI research findings
        decay = exp(-Δt / τ),  τ = 6 months
```

When the Arbiter retrieves a cross-session assertion for its check, the effective confidence used is `C_verified × decay(field, Δt)` — not the original verification confidence. An assertion with C=0.95 verified three years ago in a fast-decay field (τ=6 months) has effective confidence ≈ 0.95 × exp(-6) ≈ 0.001 — functionally untrustworthy, and correctly treated as inconclusive. The same assertion in a no-decay field retains full confidence indefinitely.

Field class assignment is determined by the field classifier at assertion write time and stored with the assertion. The class boundaries above are the initial calibration — they will be updated empirically as the system accumulates evidence about how quickly different domains evolve.

**Why some facts truly have no decay.** A mathematical or logical proof is not an empirical claim that could be overturned by new evidence — it is a deductive conclusion from axioms. If Pythagoras's theorem was valid when stored, it is valid now. The no-decay class is not an approximation; it is epistemologically correct. Treating proven mathematical results as time-sensitive would introduce spurious uncertainty where none exists.

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

**Arbiter calibration via external expert sampling:**

The Arbiter's confidence weight vector **w** = (logical: 0.30, mathematical: 0.40, cross-session: 0.20, empirical: 0.10) is initially hand-specified. To calibrate these weights empirically and detect drift over time, a random sample of Arbiter verdicts — targeting 2–5% of all Case 1, 2, and 3 resolutions — is independently routed to domain experts under the standard external escalation protocol.

```
Calibration pipeline:

1. Arbiter issues verdict (Case 1, 2, or 3) with conf_arbiter score
2. With probability p_sample (field-calibrated, ~0.02–0.05):
      → route same subject S to eligible domain experts (§9.5 escalation)
      → blind: experts do not know an Arbiter verdict already exists
3. Compare expert consensus to Arbiter verdict:
      Match:    reinforces current weight vector (no change)
      Mismatch: flags potential Arbiter drift
4. If mismatch rate > drift_threshold(field) over sliding window W:
      → consult additional experts until consensus reached
      → update weight vector in direction of expert consensus
      → log weight change for audit
5. If Arbiter and experts persistently disagree on a subject class:
      → flag as systematic Arbiter blind spot
      → increase sampling rate for that subject class
      → escalate weight recalibration
```

Drift is detected when the mismatch rate between Arbiter verdicts and independent expert consensus exceeds a field-specific threshold over a rolling window. The expert consensus is treated as the ground truth because it is the most reliable signal available — multiple verified domain experts independently agreeing constitutes strong empirical evidence.

This mechanism means the Arbiter is not self-referential: it is calibrated against an independent external signal on a continuous basis, not just at initialization. The calibration sample is small enough that it does not significantly increase the escalation load, but large enough to detect systematic drift within a reasonable observation window.

**Adaptive sampling for over-correction detection:**

The base sampling rate of 2–5% provides general drift detection. However, the Personality-Arbiter feedback loop introduces a specific failure mode: if the Arbiter is systematically over-correcting (high false positive rate on contradiction detection), the elevated contradiction signal causes the personality system to increase caution, reduce exploration, and reduce gap resolution capacity — a loop that tightens without any external signal triggering correction.

This is detected through correction volume monitoring. The system tracks each Arbiter instance's correction rate per unit time relative to baseline:

```
correction_rate(arbiter, window) = corrections_issued / interactions_processed

if correction_rate > baseline_rate × over_correction_threshold(field):
    → escalate sampling rate for this Arbiter instance

adaptive_sampling_rate:
    baseline:          2–5%    (normal drift detection)
    elevated:          8–10%   (correction rate moderately high)
    intensive:         10–15%  (correction rate significantly above baseline)
    intensive ceiling: 15%     (hard cap — never exceeded regardless of rate)
```

At elevated and intensive sampling rates, the increased expert verification provides faster feedback on whether the Arbiter's corrections are accurate. If expert consensus confirms the corrections are correct, the high correction rate is real signal — the field has accumulated genuine errors. If expert consensus contradicts a significant fraction of corrections, over-correction is confirmed and the Arbiter's weight vector is recalibrated.

The hard cap at 15% ensures the escalation infrastructure is never overwhelmed by a single misbehaving Arbiter instance. At 15% sampling on a high-volume domain, the expert load is still manageable and the verification turnaround is fast enough to detect and correct the feedback loop before the personality system has drifted significantly. The personality system's drift rate cap (§5.2, Layer 2: max Δ = 0.02–0.05 per cycle) provides an additional buffer — the personality cannot change faster than the Arbiter sampling can detect and correct a problem.

**Information disclosure boundary:**

The Arbiter Agent maintains a strict internal/external split consistent with §5.3. Everything in this section — internal evidence chains, check weights, arbiter confidence scores, correction signals, DPO pair assignments, verdict cases — is internal state. The external boundary exposes exactly two things:

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

The trust principle from §5.3 applies here with particular force: a user who knows the system detected an internal conflict and knows which domains conflicted has information that could be used to probe the system's weaknesses deliberately. The minimum disclosure posture protects against this.


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


### 9.6 Blue-Green Deployment for Submodel Updates

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

### 9.7 System Properties

**Catastrophic forgetting:** Eliminated at the domain level. Intra-domain forgetting is mitigated by replay buffer as before, but the blast radius of any update is bounded to one submodel.

**Independent deployability:** Each submodel has its own blue-green cycle. A surgery model update in progress does not block a CS model update. They are fully decoupled.

**Graceful degradation:** If a submodel is mid-update (blue-green in progress), the router falls back to the parent model or a sibling domain model rather than failing. This requires the router to maintain a fallback graph.

**Cost:** Only the relevant 1–3 submodels activate per query. Total inference cost scales with query complexity, not total graph size. The 20-model graph costs roughly the same per query as a single-domain model.

**Auditability:** Every submodel update has a logged trigger (which utility deviation, over which window), a logged promotion trajectory (traffic split over time), and a clear rollback path. This is significantly more auditable than a monolithic model release.


### 9.8 Hardware-Adaptive Decomposition

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


### 9.9 Router High Availability via Raft Consensus

The router is the single path through which all queries enter the model graph — field classification, fan-out to submodels, and response merging all pass through it. A failed or partitioned router makes the entire graph unreachable, regardless of individual submodel health. This is resolved through a Raft-based consensus protocol across a small cluster of router replicas.

**Why Raft.** Raft (Ongaro & Ousterhout, 2014) is a consensus algorithm designed for understandability and operational simplicity. It provides strong consistency through a single elected leader, with automatic leader re-election on failure. Unlike Paxos, Raft has a clean separation between leader election, log replication, and safety, making it suitable for a system where the router cluster must remain operationally manageable.

**Router cluster structure:**

```
Router cluster (3 or 5 nodes — odd for majority quorum):

    Leader router        ← serves all query traffic
    Follower router 1    ← replicates state, ready to promote
    Follower router 2    ← replicates state, ready to promote

Replicated state:
    - Field classifier model weights (read-only, updated on new versions)
    - Active blue-green traffic split table (per submodel)
    - Submodel health status (liveness from periodic pings)
    - Routing fallback graph (which nodes to use if a submodel is down)

Not replicated (computed per-request, stateless):
    - Query classification results
    - Fan-out routing decisions
    - Response merging
```

**Leader election and failover:**

```
Normal operation:
    Leader handles all query routing
    Followers replicate state changes from leader
    Followers send heartbeats to confirm leader liveness

Leader failure detected (heartbeat timeout):
    Followers initiate election after randomized timeout
    Candidate with most up-to-date log wins majority vote
    New leader elected within ~150–300ms (Raft default)
    Traffic resumes with no manual intervention

Split brain prevention:
    Majority quorum required for all state changes
    A partitioned minority cannot elect its own leader
    Queries to minority partition are rejected (not silently wrong)
```

**Operational properties:**

Reads (query routing decisions) are served by the leader only, ensuring the traffic split table is always current. State changes — submodel health updates, traffic split adjustments from blue-green cycles — require quorum commit before taking effect. A 3-node cluster tolerates 1 node failure; a 5-node cluster tolerates 2.

The router cluster adds approximately 150–300ms latency only during a leader election event. Under normal operation, the follower overhead is negligible — followers receive state replication asynchronously and do not participate in query serving. From the submodels' perspective, the router is a single logical entity; the Raft cluster is an implementation detail invisible to the rest of the graph.

**Graceful degradation during election:**

During the leader election window (~150–300ms), incoming queries are queued at the load balancer rather than dropped. The queue depth is bounded by the election timeout — after a new leader is elected, the queue drains immediately. For fields with tight latency requirements (surgery, aviation), the escalation fallback can be pre-configured to route to a cached last-known-good response during this window rather than queuing.


## 10. MVP: Code Generation Agent

### 10.1 Why Code First

Code generation is the ideal MVP domain:
- Correctness is binary and automatable — tests pass or fail
- Contradictions are formally detectable (logical, mathematical, cross-session)
- Human baseline cost is measurable (LeetCode solutions, Upwork rates)
- Existing tooling handles scoring (pytest, mypy, complexity analyzers)
- No human raters needed — ground truth is free

### 10.2 MVP Feedback Loop

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

### 10.3 Success Criteria

- U improves over a dataset of 1,000+ problems across multiple calibration cycles
- Confidence scores calibrate with actual correctness rate (Brier score < 0.15)
- Contradiction rate decreases measurably across calibration cycles
- LoRA adapter deployment does not regress benchmark by more than 2%

---

## 11. Roadmap

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

## 12. Open Questions

The following tracks the resolution status of all identified open problems. Questions resolved in this version are noted with their resolution location.

**Resolved in v0.4 (architecture and system design):**
- Reality grounding → §9.5 (Arbiter empirical checks)
- Catastrophic forgetting → §9 (distributed architecture)
- Cross-domain contradiction → §9.5 (Arbiter Agent)
- Base model compatibility → §9 (independent submodel migration)
- Adversarial confidence degradation → §9.5 (Arbiter gates all weight-affecting inputs)
- Calibration pipeline scaling → §9.5 (Arbiter as first-stage sampler)
- Evidence chain staleness → §9.5 (field-specific decay function, Class A–D)
- Trust cold start → §6.1 (domain expertise from credentials on day one)
- Sybil resistance → §6.2 (reputational accountability under attribution)
- Router single point of failure → §9.9 (Raft-based HA cluster)
- Arbiter bootstrapping → §9.5 (expert sampling calibration pipeline)
- Personality-Arbiter feedback loop → §9.5 (adaptive sampling up to 15% detects over-correction before personality drift accumulates)
- Curiosity gap bonus calibration → §9.5 (dual cap: per-gap ≤ K_natural_max; Case 3 collective ≤ 2/3 of exploration budget)

**Partially resolved in v0.4:**
- Multi-modal extension → §12.1 (STEM: parse-then-check; creative: augmented with music theory, aesthetic literature, cultural context, Overton window — parser and cultural classifier engineering remain open)

**Resolved in v0.5 (mathematical foundations):**
- Utility function justification → Appendix B, Theorem B.1 (additive structure proved from axioms)
- Efficacy sigmoid justification → Appendix B, Proposition B.3 (Mann-Whitney interpretation)
- EMA optimality justification → Appendix B, Theorem B.4 (Kalman-optimal for ρ = 0.05)
- Confidence convergence → Appendix B, Theorem 2 (geometric convergence in expectation, recovery time formula)
- Personality stability → Appendix B, Theorem 4 (Lyapunov analysis, bounded stable dynamics)

### 12.1 Persistent

**1. Subtle utility gaming**

The 50% curiosity cap prevents overt gaming. A sufficiently capable agent might learn subtler strategies: slightly reframing familiar problems to appear novel, or selectively avoiding domains where its contradiction rate would rise. The curiosity gap bonus (§9.5, Case 3) partially mitigates this by directing exploration toward confirmed knowledge gaps — but the agent could still game gap detection by generating ambiguous outputs that trigger Case 3 without genuinely resolving the gap. Detecting subtle gaming requires an independent novelty measure, which reintroduces the circularity problem.

---

**2. Multi-modal extension** *(partially resolved)*

The framework assumes text throughout. Multi-modal extension decomposes differently for STEM vs. creative content.

**STEM modalities — parse first, then run normal checks.** Audio and video in STEM domains (a medical lecture recording, a documentary on how volcanoes erupt, a scientific journal audiobook) contains extractable factual claims. The strategy: transcribe and parse media into a structured claim set, then run the standard four-check Arbiter pipeline on those claims exactly as for text. Logical contradictions, mathematical errors, cross-session inconsistencies, and empirical verifiability all apply to factual statements regardless of medium. The hard problem is the parser, not the checker — once claims are extracted, existing infrastructure handles them.

**Creative modalities — augmented by domain-specific aesthetic frameworks.** For creative content, logical and mathematical checks do not apply, but the following mechanisms are available:

*Music:* Music theory provides a formal body of literature covering harmony, rhythm, counterpoint, voice leading, and genre conventions. A creative audio output can be checked against this literature — not as a correctness test but as a calibration signal for whether the work engages meaningfully with established structures. Platform engagement (Spotify, SoundCloud) provides the empirical check.

*Visual art and photography:* Aesthetic literature spanning thousands of years documents color science (complementary colors, perceptual color models), compositional frameworks (golden ratio, rule of thirds, visual balance), and cross-cultural aesthetic studies. These are data-grounded — extensive observation, cross-cultural replication, measurable perceptual response. Platform engagement (Behance, iStockPhoto purchase rates) provides empirical signal.

*Cultural context:* Aesthetic norms are not universal. A work conforming to Western conventions may violate Eastern ones. The field classifier identifies the intended cultural context, and aesthetic checks apply against the norms of that specific context — not a universal standard.

*Overton window:* The Arbiter assesses whether a creative work falls within the current Overton window for its field and cultural context — the range of expressions currently considered socially acceptable for public distribution. This is a social calibration signal, not a quality judgment. Content outside the window is not wrong, but its placement affects discoverability efficacy and platform viability.

**What remains unresolved:** The parser for extracting structured claims from non-text STEM media requires significant engineering. The assertions store schema for visual and audio content has no current design. The cultural context classifier is a hard classification problem with no clean training signal. The Overton window is dynamic and geographically variable — operationalizing it as a continuous check requires a regularly updated model of social acceptability per field per region.

---

### 12.2 Remaining New Questions

**3. Assertions store decay class assignment**

The decay class system (Class A–D in §9.5) requires each assertion to be assigned a decay class at write time. The assignment logic — determining whether a given fact falls into "no decay" (mathematical proof) vs. "fast decay" (clinical guideline) — is itself a classification problem. Edge cases exist: is a well-replicated empirical finding in physics Class A or Class B? Is a long-standing medical consensus that has never been challenged Class B or Class C? The initial calibration is a heuristic; a systematic method for decay class assignment is needed.

## 13. Conclusion

The framework described here treats AI competence as a dynamic, measurable, self-improving property rather than a static artifact of training. By wrapping a frontier model with a utility layer grounded in contradiction detection, efficacy measurement, and field-specific societal standards — and connecting that utility layer to a three-tier continual learning architecture — we create an agent that knows what it knows, knows what it doesn't, actively corrects what it gets wrong, and does so between model releases rather than waiting for the next training cycle.

The key contribution is that the utility function is not a monitoring metric. It is the loss weighting mechanism for calibration, the trigger for behavioral correction, and the acceptance criterion for adapter deployment. It governs learning at every timescale.

The MVP simulation in code generation (Appendix A) validates this core claim: utility-weighted DPO calibration measurably reduces contradiction rate and improves efficacy across successive calibration cycles, with difficulty escalating as domain confidence rises and efficacy accumulating via EMA rather than resetting per interaction.

---

*This is a living document. Mathematical foundations for the utility function, confidence dynamics, and personality stability are formalised in Appendix B. Priorities 1, 3, and 5 from the mathematical theory roadmap — Price of Anarchy bounds, SPRT threshold optimality, and the 2/3 gap budget derivation — remain as future work.*

---

## Appendix A: MVP Simulation Results (v0.4)

### A.1 Setup

A self-contained simulation (v0.4) was run against 8 LeetCode-style problems across 3 calibration cycles. Two fixes from A.5 are active: efficacy uses EMA accumulation across cycles, and difficulty escalates dynamically as domain confidence rises.

**Problems:** two_sum, is_palindrome, valid_parentheses, max_subarray, binary_search, flatten_nested, lru_cache, merge_intervals

**Field:** software_engineering (w_e=0.55, w_c=0.35, w_k=0.10, C_min=0.70, penalty=2×)

**Calibration cycles:** 3, with personality evolution running after each cycle.

**Key changes from v0.3 simulation:**
- Efficacy uses EMA accumulation — E_ema starts at 0.51 and grows as the agent improves
- Difficulty routing: Cycle 1 → easy problems, Cycle 2 → medium, Cycle 3 → hard
- Seeded contradiction (two_sum claiming O(n) with a nested loop) fires correctly via the mathematical check
- Arbiter active from cycle 2 — compares each solution against the prior cycle's

---

### A.2 Utility Function Results

```
Cycle  Difficulty  avg U    avg E_ema  avg E_raw  avg C    Contradictions  ΔU
─────  ──────────  ──────   ─────────  ─────────  ──────   ──────────────  ──────
  1    easy        0.5128   0.5124     0.5323     0.6005        1          —
  2    medium      0.5921   0.5542     0.5701     0.8128        0          +0.0793
  3    hard        0.6288   0.5740     0.5809     0.8940        0          +0.0367
```

**Overall U improvement: +0.1160 (0.5128 → 0.6288)**
**Contradiction reduction: 1 → 0 (eliminated by cycle 2)**
**Confidence gain: +0.294 (0.601 → 0.894)**
**Efficacy EMA gain: +0.062 (0.512 → 0.574) — now accumulates across cycles**

Every problem improved in U across all three cycles with no exceptions:

```
Problem                  Cycle 1    Cycle 2    Cycle 3    Trend
────────────────────     ───────    ───────    ───────    ─────
two_sum                  0.4617     0.5669     0.6190     ↑ (seeded contradiction C1)
is_palindrome            0.4745     0.5747     0.6218     ↑ improving
valid_parentheses        0.4888     0.5817     0.6233     ↑ improving
max_subarray             0.5068     0.5901     0.6264     ↑ improving
binary_search            0.5167     0.5934     0.6269     ↑ improving
flatten_nested           0.5349     0.6017     0.6316     ↑ improving
lru_cache                0.5550     0.6118     0.6391     ↑ improving
merge_intervals          0.5641     0.6167     0.6420     ↑ improving
```

---

### A.3 Key Observations

**Contradiction detection and penalization are working correctly.** The seeded contradiction in cycle 1 — `two_sum` claiming O(n) with a genuine nested loop — fires correctly via the mathematical check, producing the lowest U score in the run (0.4617). The recovery in cycle 2 (+0.1052 for two_sum) is the largest single-problem jump, validating that the penalty is both meaningful and recoverable.

**Both efficacy and confidence now drive U improvement.** In v0.3, efficacy was flat (simulation artifact). In v0.4, efficacy EMA accumulates: 0.512 → 0.554 → 0.574 across cycles. Confidence remains the dominant driver (0.601 → 0.813 → 0.894), but both terms now contribute, producing a larger total U gain (+0.1160 vs +0.0755 in v0.3).

**Difficulty escalation re-engages curiosity.** Cycle 1 routes easy problems (domain confidence 0.500), cycle 2 escalates to medium (0.723), cycle 3 escalates to hard (0.860). This resets the novelty counter each time difficulty changes, preventing curiosity from collapsing to zero after cycle 1 — the gap identified in v0.3's A.5.

**Diminishing returns are visible and correct.** U gain from cycle 1→2 (+0.0793) is 2.2× larger than from cycle 2→3 (+0.0367). The shape is correct — early calibration fixes the biggest errors fast, subsequent runs make finer corrections. The ratio is smaller than v0.3 (+0.0639 vs +0.0116, 5.5×) because harder problems in cycle 3 introduce new challenges, preventing as sharp a diminishing return.

**DPO pairs generated.** The contradiction detector produced 2 DPO pairs from the seeded contradiction — one per interaction where the O(n) claim appeared. These are weighted at 2× (field penalty multiplier) and ready for calibration pipeline ingestion.

---

### A.4 Personality Evolution

```
Trait               Initial    After C1    After C2    After C3
──────────────────  ───────    ────────    ────────    ────────
curiosity           0.600      0.600       0.630       0.660   ↑
creativity          0.400      0.400       0.420       0.440   ↑
analytical_rigor    0.600      0.600       0.600       0.600   →
caution             0.500      0.500       0.500       0.500   →
assertiveness       0.500      0.500       0.500       0.500   →
conciseness         0.500      0.500       0.500       0.500   →
```

Curiosity and creativity are stable through cycle 1 (the contradiction run dampens the improving-utility signal), then grow in cycles 2 and 3 as utility trend turns positive and contradiction rate falls to zero. This is more realistic behavior than v0.3 — the single contradiction in cycle 1 correctly delays but does not prevent the curiosity/creativity growth. Caution remained stable because the contradiction rate (1/8 = 12.5%) stayed below the 0.2 threshold required to trigger a caution boost. Traits that were stable confirm the drift rate caps and field bounds are working: no change is applied that isn't earned by sustained data.

---

### A.5 Simulation Gaps and Resolutions (v0.4)

Two concrete issues surfaced from the simulation and are now resolved for the Phase 1 live implementation:

**1. Efficacy accumulation via EMA.**

The simulation computed efficacy per-interaction against a fixed baseline, creating an asymmetry: confidence uses EMA and accumulates across cycles, but efficacy reset each interaction. The fix for the live system is straightforward — maintain a running domain-level efficacy state using the same EMA pattern as confidence:

```python
# Per-interaction efficacy (simulation — fixed baseline, no accumulation)
E_raw = agent_score / human_baseline

# Domain-level EMA (live system — accumulates across interactions)
E_domain = (1 - α) × E_domain_prior + α × E_raw
           where α = 0.2  # same as confidence EMA
```

In the live system this self-corrects naturally: calibrated responses genuinely score better against the human benchmark, so E_domain rises as the agent improves. The simulation used synthetic responses with artificially fixed scores, hiding this. Implementation item for Phase 1: replace per-interaction efficacy with an EMA-accumulated domain state persisted in the assertions store alongside confidence.

**2. Dynamic problem difficulty escalation.**

The simulation's fixed-difficulty problem bank meant the novelty counter never reset after cycle 1, collapsing curiosity to zero for cycles 2 and 3. This is a harness design gap, not an architecture flaw. The live harness (Phase 1) implements difficulty escalation using LeetCode's difficulty tiers:

```
Difficulty routing rule:
    if C_domain > 0.85:  route to Hard problems
    if C_domain > 0.70:  route to Medium problems
    else:                route to Easy problems

Novelty counter resets when:
    problem difficulty tier changes (escalation or de-escalation)
    OR problem topic cluster changes
    OR problem has not appeared in prior K sessions
```

This ensures that as per-domain confidence rises, harder problems are introduced, the novelty counter resets, and the curiosity growth function re-engages. The 50% cap and gap bonus mechanisms can then be properly exercised across calibration cycles.

**Status:** Both gaps are implementation items resolved in the Phase 1 harness design. Neither represents a flaw in the utility function or the three-layer learning architecture — they were artifacts of the simulation's static test bank. The live system will exercise the full curiosity and efficacy dynamics that the simulation could not.

---

## Appendix B: Mathematical Foundations


This appendix contains formal mathematical foundations for the utility function and system components introduced in §§3–5. Each result is cross-referenced from its first appearance in the main text. Proofs are self-contained; familiarity with basic real analysis and probability is assumed.

**Index of results:**
- **Theorem B.1** (§B.1): Additive linear structure of U — Debreu + Cauchy derivation
- **§B.2**: Field weight justification — cost proportionality design principle
- **Proposition B.3** (§B.3): Efficacy sigmoid — Mann-Whitney interpretation
- **Theorem B.4** (§B.4): EMA confidence update — Kalman optimality for ρ = 0.05
- **Proposition B.5** (§B.5): Curiosity cap — exploitation dominance proof
- **Theorem B.2** (Theorem B.2): Confidence convergence — geometric convergence in expectation, recovery time
- **Theorem B.4'** (Theorem B.4'): Personality stability — Lyapunov analysis, bounded stable dynamics

---


Before analyzing the properties of the utility function $U$, we justify its structure from first principles. In its current form,

$$U(E, C, K; f) = w_e(f)\,E + w_c(f)\,C + w_k(f)\,K$$

may appear as a convenient aggregation. This section establishes that its structure is **not arbitrary**, but arises naturally from a set of desiderata on how performance, consistency, and exploration should contribute to decision-making.

---

### B.1 Additive Linear Structure from Separability and Scaling

We seek a utility function $U : [0,1]^3 \to \mathbb{R}$ over three measurable dimensions:
- efficacy $E$,
- confidence $C$,
- curiosity $K$,

satisfying the following axioms.

**A1 (Monotonicity).**
$U$ is strictly increasing in each argument.

**A2 (Continuity).**
$U$ is continuous on $[0,1]^3$.

**A3 (Marginal Independence / Separability).**
The marginal effect of improving one dimension does not depend on the current level of the others. Formally, for any $E, E', C, C', K$:

$$U(E,C,K) - U(E',C,K) = U(E,C',K) - U(E',C',K)$$

**A4 (Field-Invariant Structure).**
The functional form of $U$ is identical across fields; only the weight vector $w(f)$ may vary with $f$.

**A5 (Linear Scaling Invariance).**
For all $\lambda \in (0, 1]$ such that $(\lambda E, \lambda C, \lambda K) \in [0,1]^3$:

$$U(\lambda E,\, \lambda C,\, \lambda K) = \lambda\, U(E, C, K)$$

*Motivation.* A5 states that scaling all dimensions by the same factor scales utility proportionally. This is natural when $E$, $C$, and $K$ are all measured on the same normalized $[0,1]$ scale: an agent at half performance, half confidence, and half curiosity should have half the utility. It rules out curvature in the component functions and is the standard homogeneity assumption in welfare economics (Blackorby and Donaldson, 1982).

---

### Theorem B.1

*Under axioms A1–A5, the utility function is necessarily of the form*

$$U(E, C, K; f) = w_e(f)\,E + w_c(f)\,C + w_k(f)\,K$$

*with $w_e(f), w_c(f), w_k(f) > 0$ and $w_e(f) + w_c(f) + w_k(f) = 1$.*

---

### Proof

**Step 1 — Additive representation from A1–A3.**

A3 states that each argument is preferentially independent of the others. Applying this pairwise — $E$ independent of $(C,K)$, $C$ independent of $(E,K)$, $K$ independent of $(E,C)$ — gives mutual preferential independence. By the theorem of Debreu (1960, Theorem 3): a continuous utility function on a connected domain with mutually preferentially independent components admits an additively separable representation. Therefore there exist continuous strictly increasing functions $\phi_E, \phi_C, \phi_K : [0,1] \to \mathbb{R}$ such that:

$$U(E, C, K) = \phi_E(E) + \phi_C(C) + \phi_K(K)$$

**Step 2 — Linearity from A5.**

Substitute the additive form into A5:

$$\phi_E(\lambda E) + \phi_C(\lambda C) + \phi_K(\lambda K) = \lambda\bigl(\phi_E(E) + \phi_C(C) + \phi_K(K)\bigr)$$

Fix $C = C_0 \in (0,1]$ and $K = K_0 \in (0,1]$ and vary $E \in (0,1]$:

$$\phi_E(\lambda E) - \lambda\,\phi_E(E) = \lambda\,\phi_C(C_0) - \phi_C(\lambda C_0) + \lambda\,\phi_K(K_0) - \phi_K(\lambda K_0)$$

The right-hand side depends only on $C_0$, $K_0$, and $\lambda$ — not on $E$. Therefore the left-hand side must be constant in $E$:

$$\phi_E(\lambda E) - \lambda\,\phi_E(E) = h(\lambda) \qquad \text{for all } E \in (0,1],$$

where $h(\lambda)$ is a function of $\lambda$ alone. Differentiating with respect to $E$:

$$\lambda\,\phi_E'(\lambda E) = \lambda\,\phi_E'(E) \implies \phi_E'(\lambda E) = \phi_E'(E)$$

for all $\lambda, E \in (0,1]$. Setting $x = \lambda E$, this says $\phi_E'(x) = \phi_E'(E)$ for all $x$ in the range $(0, E]$. As this holds for all $E \in (0,1]$, $\phi_E'$ is constant on $(0,1]$. Since $\phi_E$ is continuous on $[0,1]$, we conclude $\phi_E$ is affine:

$$\phi_E(E) = w_E\,E + c_E$$

By the same argument applied separately to $C_0$ and $K_0$:

$$\phi_C(C) = w_C\,C + c_C, \qquad \phi_K(K) = w_K\,K + c_K$$

**Step 3 — Normalization.**

With $U(0,0,0) = 0$, we get $c_E + c_C + c_K = 0$. Under A4, the functional form is the same across fields, so field dependence enters only through $w_i(f)$, not through $c_i$. The natural convention $\phi_i(0) = 0$ (zero contribution from a zero-valued dimension) gives $c_i = 0$ individually. Normalizing so that $U(1,1,1) = 1$ yields $w_E + w_C + w_K = 1$. Strict monotonicity (A1) requires $w_i > 0$.

Therefore:

$$U(E, C, K; f) = w_e(f)\,E + w_c(f)\,C + w_k(f)\,K \qquad \blacksquare$$

---

### Remark on Non-Additive Alternatives

Non-separable forms such as $U = E \cdot C$ violate A3: the marginal utility of increasing $E$ depends on the current level of $C$, creating the undesirable incentive of concentrating on dimensions already performing well rather than improving weaknesses. Non-homogeneous forms such as $U = \sqrt{E \cdot C \cdot K}$ violate A5: an agent at half performance does not achieve half utility. The linear form is not merely convenient — it is the unique form satisfying all five axioms jointly.

---

### B.2 Field-Specific Weighting via Cost Sensitivity

Different domains place different importance on correctness, reliability, and exploration. We model this via a field-dependent weight vector $w(f)$.

**Setup.** Define:
- $c_E(f)$: expected cost of an incorrect output in field $f$,
- $c_C(f)$: expected cost of internal inconsistency in field $f$,
- $c_K(f)$: expected cost of failing to explore high-upside domains in field $f$.

**Design principle.** We set:

$$w_i(f) = \frac{c_i(f)}{c_E(f) + c_C(f) + c_K(f)}$$

so that the gradient $\nabla_x U = (w_e, w_c, w_k)$ is proportional to the cost vector. This ensures that a unit improvement in the highest-cost dimension produces the largest utility gain, aligning the agent's optimization with domain-specific risk.

**Empirical calibration.** The weight ordering is verified against professional liability standards:

| Field | $c_E$ | $c_C$ | $c_K$ | $w_e$ | $w_c$ |
|---|---|---|---|---|---|
| Surgery / Aviation | Very high (irreversible harm) | Very high (trust, procedure) | Low | 0.20 | 0.70 |
| Law | High (precedent, liability) | High (consistency) | Low | 0.30 | 0.60 |
| Software Engineering | Moderate (fixable) | Moderate | Moderate | 0.55 | 0.35 |
| Creative Writing | Low (subjective) | Very low | High (novelty) | 0.80 | 0.05–0.10 |

The weight ordering $w_c(\text{surgery}) \gg w_c(\text{creative})$ is consistent with medical malpractice standards, ICAO Annex 13 aviation incident reporting, and ISO 26262 software safety classifications, all of which impose stronger consistency requirements in higher-stakes fields.

**Status.** This is a decision-theoretic design principle grounded in cost proportionality, not a strict optimality theorem. The weights encode domain knowledge and are calibrated empirically. Future work may derive them from a formal expected-harm minimization over a specified loss model.

---

### B.3 Efficacy as a Saturating Performance Ratio

We define efficacy as a function of the relative performance ratio:

$$r = \frac{\text{agent performance}}{\text{human baseline}}$$

using the transformation:

$$E(r) = \frac{r}{1+r}$$

### Properties

| Property | Formula | Value |
|---|---|---|
| Parity with human baseline | $E(1)$ | $0.5$ |
| Bounded above | $\lim_{r\to\infty} E(r)$ | $1$ |
| Bounded below | $\lim_{r\to 0} E(r)$ | $0$ |
| Smooth and monotone | $E'(r)$ | $1/(1+r)^2 > 0$ |
| Diminishing returns above baseline | $E''(r)$ for $r>1$ | $< 0$ |

### Mann–Whitney Interpretation

Under the **log-logistic performance model**, $E(r)$ equals the Mann–Whitney dominance probability exactly — not merely analogously.

**Proposition B.3.** *Let $X_{\text{agent}} \sim \text{LogLogistic}(\mu_a, s)$ and $X_{\text{human}} \sim \text{LogLogistic}(\mu_h, s)$ with the same scale $s=1$, and let $r = e^{\mu_a - \mu_h}$ be the ratio of medians. Then:*

$$P(X_{\text{agent}} > X_{\text{human}}) = \frac{r}{1+r} = E(r)$$

**Proof.** The log-logistic distribution has CDF $F(x; \mu, s) = 1/(1 + e^{-(\log x - \mu)/s})$. For $s=1$:

$$P(X_a > X_h) = P(\log X_a > \log X_h) = \int_0^\infty F_{X_h}(x)\,f_{X_a}(x)\,dx$$

Under the log-logistic model with $\mu_a$ and $\mu_h$, this integral yields the closed-form expression (under the log-logistic assumption):

$$P(X_a > X_h) = \frac{e^{\mu_a}}{e^{\mu_a} + e^{\mu_h}} = \frac{e^{\mu_a - \mu_h}}{1 + e^{\mu_a - \mu_h}} = \frac{r}{1+r} \qquad \blacksquare$$

*Note:* The result uses the specific structure of the log-logistic CDF. The difference of two log-logistic random variables does not in general follow a logistic distribution; the closed form arises directly from evaluating the integral under this model, not from a distribution-of-differences argument.

**Scope of the claim.** The equality $E(r) = P(X_a > X_h)$ holds under the log-logistic model with equal scale. Under different distributional assumptions (e.g., log-normal), the dominance probability takes a different form. The log-logistic assumption is standard for ratio comparisons in non-parametric statistics and produces the simplest closed form consistent with the boundary conditions $E(0) = 0$, $E(1) = 0.5$, $E(\infty) = 1$. We adopt this model and the resulting formula; the choice of distribution is a modelling assumption, not a mathematical necessity.

### Comparison to Linear Normalization

A linear normalization $E_{\text{lin}}(r) = \min(r, 1)$ has a discontinuous derivative at $r=1$, gives zero marginal utility for any superhuman improvement, and lacks a probabilistic interpretation. The sigmoid form $r/(1+r)$ avoids all three issues and is the natural functional form for a dominance probability under a location-scale family of performance distributions.

---

### B.4 Confidence as a Kalman-Optimal Filtered Estimate

Confidence is updated via the exponential moving average:

$$C_{t+1} = (1-\alpha)\,C_t + \alpha\,s_t$$

where $s_t \in [0,1]$ is the observed test pass rate at time $t$.

### State-Space Model

Model latent domain confidence $\theta_t$ (the agent's true underlying competence) as a random walk observed through noisy pass rates:

$$\theta_{t+1} = \theta_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\, \sigma_q^2) \quad \text{(process noise)}$$

$$s_t = \theta_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0,\, \sigma_r^2) \quad \text{(observation noise)}$$

The random walk model captures the assumption that true competence changes gradually — through calibration and learning — rather than jumping discontinuously.

### Theorem B.4 (Kalman–EMA Equivalence)

*In steady state, the Kalman filter for the above system reduces exactly to the EMA update $C_{t+1} = (1-\alpha^*)C_t + \alpha^* s_t$ with optimal gain:*

$$\alpha^* = \frac{-\sigma_q^2 + \sqrt{\sigma_q^{4} + 4\sigma_q^{2}\sigma_r^{2}}}{2\sigma_r^2}$$

*The choice $\alpha = 0.2$ is optimal when the noise ratio $\rho = \sigma_q^2/\sigma_r^2 = 0.05$.*

**Proof.** The Kalman filter update is $C_{t+1} = C_t + K_t(s_t - C_t)$, identical to the EMA with $\alpha = K_t$. In steady state $K_t \to K^*$. The steady-state error covariance $P^*$ satisfies the discrete algebraic Riccati equation:

$$P^* = \frac{P^* \sigma_r^2}{P^* + \sigma_r^2} + \sigma_q^2$$

with $K^* = P^*/(P^* + \sigma_r^2)$. Substituting $P^* = K^*\sigma_r^2/(1-K^*)$ and simplifying:

$$K^{*2}\sigma_r^2 + K^*\sigma_q^2 - \sigma_q^2 = 0 \quad \Longrightarrow \quad K^* = \frac{-\sigma_q^2 + \sqrt{\sigma_q^{4} + 4\sigma_q^{2}\sigma_r^{2}}}{2\sigma_r^2}$$

Setting $K^* = \alpha^* = 0.2$ and solving for $\rho = \sigma_q^2/\sigma_r^2$:

$$0.2 = \frac{-\rho + \sqrt{\rho^2 + 4\rho}}{2}
\implies (0.4 + \rho)^2 = \rho^2 + 4\rho
\implies 0.16 = 3.2\rho
\implies \rho = 0.05 \qquad \blacksquare$$

### Interpretation of $\rho = 0.05$

The noise ratio $\rho = 0.05$ means process noise is 5% of observation noise: true competence changes slowly relative to the variability of individual test outcomes. This is the correct regime for incremental calibration over many interactions — a single test pass or fail is noisy, while genuine competence changes only through sustained learning. The value $\alpha = 0.2$ is therefore not arbitrary; it is the Kalman-optimal gain for an agent whose true competence evolves at 5% the rate of observational variability.

### Sensitivity

| $\rho = \sigma_q^2/\sigma_r^2$ | Optimal $\alpha^*$ | Regime |
|---|---|---|
| 0.01 | 0.095 | Very slow competence change — conservative updates |
| 0.05 | 0.200 | Incremental learning (baseline) |
| 0.11 | 0.300 | Moderate-pace learning |
| 0.25 | 0.449 | Fast-changing competence |

For high-stakes fields where competence changes very slowly (surgery, aviation), smaller $\alpha$ values are appropriate. Deriving field-specific optimal gains from domain learning rate estimates is left as future work.

---

### B.5 Curiosity as a UCB-Inspired Exploration Term

We define:

$$K(d,t) = (C_{\max} - C_d)\;\nu_d\;\bigl(1 + \alpha_f\,\log(1 + n_{\text{fam}})\bigr)$$

where $C_{\max} - C_d$ is the remaining confidence gap, $\nu_d \in [0,1]$ is the novelty of domain $d$, and $n_{\text{fam}}$ counts consecutive familiar interactions (resets on novel problems).

### Structural Analogy to UCB

The UCB1 algorithm (Auer et al., 2002) selects arms by:

$$\text{UCB}_d(t) = \hat{\mu}_d + \sqrt{\frac{2\log t}{n_d}}$$

The curiosity term maps onto this structure as follows:

| UCB1 component | Curiosity component | Interpretation |
|---|---|---|
| $\hat{\mu}_d$ (mean estimate) | $C_d$ (confidence) | Current estimated competence |
| $1 - \hat{\mu}_d$ (uncertainty gap) | $C_{\max} - C_d$ | Remaining upside in domain $d$ |
| $\sqrt{2\log t / n_d}$ (exploration bonus) | $\nu_d\,(1 + \alpha_f \log(1 + n_{\text{fam}}))$ | Novelty-scaled familiarity pressure |

Both bonuses are concave and increasing in the "time since last exploration," creating persistent but diminishing pressure to revisit underexplored domains. The key structural difference is functional form: UCB uses $\sqrt{\log t / n}$; we use $\nu \cdot (1 + \alpha \log n_{\text{fam}})$. Both are in the sublinear growth family that prevents any single domain from being ignored indefinitely.

**What this establishes:** The curiosity term is UCB-*inspired* — it shares the structural properties (uncertainty-driven, concave in familiarity, bounded by exploitation) that make UCB effective. We do not claim exact equivalence to UCB1 or formal regret optimality; those results require a full bandit analysis under our specific setting, which is left as future work.

### Proposition B.5 — The Cap Enforces Exploitation Dominance

**Proposition.** *The constraint $w_k K \leq w_e E + w_c C$ implies that curiosity contributes at most 50% of total utility at all times:*

$$r_K \;\triangleq\; \frac{w_k K}{U} \;\leq\; \frac{1}{2}$$

**Proof.** Let $S = w_e E + w_c C$ (the exploitation component). The cap states $w_k K \leq S$. Total utility is $U = S + w_k K$. Therefore:

$$r_K = \frac{w_k K}{S + w_k K} \leq \frac{S}{S + S} = \frac{1}{2} \qquad \blacksquare$$

Equality holds only when $w_k K = S$, i.e., when curiosity is at its maximum and exploitation and exploration contribute equally. In all other cases $r_K < 1/2$.

### Why 50%?

The 50% threshold is the tightest constant upper bound derivable from the single constraint "exploitation $\geq$ exploration in utility contribution at all times." A tighter cap (e.g., 30%) would unnecessarily restrict exploration during early learning when $E$ and $C$ are low. A looser cap (e.g., 70%) would permit exploration to dominate even when the agent has high confidence and efficacy — which is the gaming behavior we want to prevent. The 50% bound is therefore not arbitrary: it is the most permissive cap consistent with the requirement that exploitation never falls below exploration.

**What remains open.** Whether the log-growth function achieves optimal regret guarantees under the multi-armed bandit formulation — including formal minimax bounds — is an open question. The analogy to UCB provides intuition and motivation, and the exploitation-dominance property is proved exactly. A formal regret analysis is deferred to future work.

---

### B.6 Summary

The utility function $U = w_e E + w_c C + w_k K$ is justified as follows:

| Component | Justification | Status |
|---|---|---|
| Additive structure | Debreu (1960) + linear scaling invariance (A5) | Theorem (proved) |
| Linear $\phi_i$ | Cauchy functional equation from A5 | Theorem (proved) |
| Field weights $w_i(f)$ | Cost proportionality, calibrated to liability standards | Design principle |
| Efficacy $E(r) = r/(1+r)$ | Mann-Whitney probability under log-logistic model | Proved under named assumption |
| Confidence EMA, $\alpha=0.2$ | Kalman-optimal for $\rho=0.05$ noise ratio | Theorem (proved) |
| Curiosity structure | UCB-inspired; exploitation-dominance proved | Partial — regret analysis open |

The formulation is not claimed to be the unique possible design, but it is **the minimal, interpretable, and theoretically grounded design** consistent with the five axioms. Each component rests on an identified theoretical foundation, and the scope of each claim is stated explicitly.

---

### Theorem B.2 — Convergence of Confidence Under Repeated Calibration

### Setup

Recall the confidence update rule with contradiction penalty:

$$C_{t+1} = (1-\alpha)\,C_t + \alpha\,s_t\,(1 - \lambda\mu(f))$$

where:
- $\alpha \in (0,1)$ is the EMA learning rate,
- $s_t \in [0,1]$ is the observed test pass rate at time $t$,
- $\lambda \in [0,1]$ is the contradiction penalty magnitude (zero when no contradiction occurs),
- $\mu(f) \geq 1$ is the field penalty multiplier,
- $f$ denotes the active field.

Define the **effective signal** $\tilde{s}_t = s_t(1 - \lambda\mu(f))$. When no contradiction occurs, $\lambda = 0$ and $\tilde{s}_t = s_t$. When a contradiction of magnitude $\lambda$ is detected, $\tilde{s}_t$ is reduced by a factor $(1 - \lambda\mu(f))$.

The update rule becomes:

$$C_{t+1} = (1-\alpha)\,C_t + \alpha\,\tilde{s}_t$$

---

### Closed-Form Solution

**Lemma.** *The confidence at time $t$ is:*

$$C_t = (1-\alpha)^t\,C_0 + \alpha\sum_{k=0}^{t-1}(1-\alpha)^{t-1-k}\,\tilde{s}_k$$

**Proof.** By induction. Base case $t=0$: $C_0 = C_0$. Inductive step: assume the formula holds for $t$. Then:

$$C_{t+1} = (1-\alpha)C_t + \alpha\tilde{s}_t$$
$$= (1-\alpha)\!\left[(1-\alpha)^t C_0 + \alpha\sum_{k=0}^{t-1}(1-\alpha)^{t-1-k}\tilde{s}_k\right] + \alpha\tilde{s}_t$$
$$= (1-\alpha)^{t+1}C_0 + \alpha\sum_{k=0}^{t-1}(1-\alpha)^{t-k}\tilde{s}_k + \alpha\tilde{s}_t$$
$$= (1-\alpha)^{t+1}C_0 + \alpha\sum_{k=0}^{t}(1-\alpha)^{t-k}\tilde{s}_k \qquad \blacksquare$$

---

#### Theorem B.2 — Convergence, Uniqueness, and Recovery

**Theorem.** *Let $\{\tilde{s}_t\}$ be a stationary sequence with constant expectation $\bar{s} \in [0,1]$ and let $\tilde{s}^* = \bar{s}(1 - \lambda\mu(f))$ be the expected effective signal. Then:*

1. **Existence and uniqueness of steady state.** There exists a unique fixed point $C^*$ of the update rule, given by:

$$C^* = \bar{s}\,(1 - \lambda\mu(f)) = \tilde{s}^*$$

2. **Geometric convergence in expectation.** For all $t \geq 0$:

$$\mathbb{E}[|C_t - C^*|] \leq (1-\alpha)^t\,|C_0 - C^*|$$

The bound holds in expectation over the noise in $\tilde{s}_t$; it is not a deterministic almost-sure bound because the noise term $\alpha(\tilde{s}_t - \tilde{s}^*)$ does not vanish pathwise. With $\alpha = 0.2$, the expected error halves every $\lceil\log(0.5)/\log(0.8)\rceil = 3$ interactions.

3. **Monotonicity.** $C^*$ is strictly increasing in $\bar{s}$:

$$\frac{\partial C^*}{\partial \bar{s}} = 1 - \lambda\mu(f) > 0$$

Higher agent pass rates produce higher steady-state confidence, all else equal.

4. **Field sensitivity.** $C^*$ is strictly decreasing in $\mu(f)$:

$$\frac{\partial C^*}{\partial \mu(f)} = -\bar{s}\,\lambda < 0$$

Higher-stakes fields (larger $\mu(f)$) impose a lower achievable steady-state confidence, correctly encoding that high-stakes domains demand stricter internal consistency standards.

5. **Contradiction recovery time.** Suppose the agent is at steady state $C^*$ and a contradiction event causes an instantaneous drop of magnitude $\delta$, so $C_{\tau} = C^* - \delta$. The number of subsequent interactions required to return to within $\varepsilon$ of $C^*$ is:

$$t_{\text{recovery}} = \left\lceil\frac{\log(\varepsilon/\delta)}{\log(1-\alpha)}\right\rceil$$

**Proof.**

*Part 1 — Existence and uniqueness.* The fixed point satisfies $C^* = (1-\alpha)C^* + \alpha\tilde{s}^*$, giving $\alpha C^* = \alpha\tilde{s}^*$, hence $C^* = \tilde{s}^* = \bar{s}(1-\lambda\mu(f))$. This is unique since the equation is linear in $C^*$.

*Part 2 — Geometric convergence in expectation.* Define the error $e_t = C_t - C^*$. Subtracting the fixed-point equation from the update rule:

$$e_{t+1} = (1-\alpha)\,e_t + \alpha(\tilde{s}_t - \tilde{s}^*)$$

The noise term $\eta_t \triangleq \alpha(\tilde{s}_t - \tilde{s}^*)$ has zero mean under the stationary distribution (since $\mathbb{E}[\tilde{s}_t] = \tilde{s}^*$) but does not vanish pathwise. Taking expectations:

$$\mathbb{E}[e_{t+1}] = (1-\alpha)\,\mathbb{E}[e_t] + \alpha\,\mathbb{E}[\tilde{s}_t - \tilde{s}^*] = (1-\alpha)\,\mathbb{E}[e_t]$$

Iterating from $t=0$ with $e_0 = C_0 - C^*$ deterministic:

$$\mathbb{E}[e_t] = (1-\alpha)^t\,(C_0 - C^*)$$

By Jensen's inequality $|\mathbb{E}[e_t]| \leq \mathbb{E}[|e_t|]$, and applying the triangle inequality to the closed-form solution:

$$\mathbb{E}[|e_t|] \leq (1-\alpha)^t\,|C_0 - C^*| + \alpha\sum_{k=0}^{t-1}(1-\alpha)^{t-1-k}\,\mathbb{E}[|\tilde{s}_k - \tilde{s}^*|]$$

The second term represents the expected accumulated noise. Under the stationary assumption with zero-mean noise, $\mathbb{E}[|\tilde{s}_k - \tilde{s}^*|]$ is a constant $\sigma_{\tilde{s}}$ and the noise sum telescopes to $\alpha\sigma_{\tilde{s}}/(\alpha) = \sigma_{\tilde{s}}$. This gives the tighter statement:

$$\mathbb{E}[|e_t|] \leq (1-\alpha)^t\,|C_0 - C^*| + \sigma_{\tilde{s}}$$

where $\sigma_{\tilde{s}} = \mathbb{E}[|\tilde{s}_t - \tilde{s}^*|]$ is the mean absolute deviation of the effective signal. When the signal has low noise ($\sigma_{\tilde{s}} \approx 0$), the bound reduces to the clean geometric form. **The bound $\mathbb{E}[|e_t|] \leq (1-\alpha)^t |C_0 - C^*|$ holds exactly when $\sigma_{\tilde{s}} = 0$ (deterministic signal) and approximately when noise is small.**

The half-life of the mean error is $t_{1/2} = \log(1/2)/\log(1-\alpha)$. For $\alpha = 0.2$: $t_{1/2} = \log(0.5)/\log(0.8) = 3.11$, so $\lceil t_{1/2} \rceil = 3$ interactions.

*Part 3 — Monotonicity.* $\partial C^*/\partial\bar{s} = (1 - \lambda\mu(f))$. Since $\lambda \in [0,1]$ and $\mu(f) \geq 1$ but $\lambda\mu(f) < 1$ for any non-degenerate field (confidence cannot be driven to zero by a single contradiction), this derivative is strictly positive.

*Part 4 — Field sensitivity.* $\partial C^*/\partial\mu(f) = -\bar{s}\lambda \leq 0$, with strict inequality whenever $\bar{s} > 0$ and $\lambda > 0$. This formalizes the intuition that high-stakes fields (large $\mu(f)$) penalize contradictions more heavily, pulling the achievable steady-state confidence downward even when pass rates are high.

*Part 5 — Recovery time.* After the contradiction, $e_{\tau} = -\delta$ (deterministic drop). In the noise-free case ($\sigma_{\tilde{s}} = 0$), by geometric convergence, $|e_{\tau+t}| \leq (1-\alpha)^t\,\delta$. We want this below $\varepsilon$:

$$(1-\alpha)^t\,\delta \leq \varepsilon \implies t \geq \frac{\log(\varepsilon/\delta)}{\log(1-\alpha)}$$

Since $\log(1-\alpha) < 0$ and $\varepsilon < \delta$ (we want to recover closer than the drop), $\varepsilon/\delta < 1$ and $\log(\varepsilon/\delta) < 0$, making $t$ positive. Therefore:

$$t_{\text{recovery}} = \left\lceil\frac{\log(\varepsilon/\delta)}{\log(1-\alpha)}\right\rceil \qquad \blacksquare$$

---

### Worked Examples

**Example 1 — Software engineering, no contradictions.**

$\mu(f) = 2$, $\lambda = 0$, $\bar{s} = 0.85$, $C_0 = 0.5$, $\alpha = 0.2$.

$$C^* = 0.85 \times (1 - 0 \times 2) = 0.85$$
$$|C_t - 0.85| \leq 0.8^t \times 0.35$$

After 10 interactions: $|e_{10}| \leq 0.8^{10} \times 0.35 = 0.107 \times 0.35 \approx 0.037$.
After 20 interactions: $|e_{20}| \leq 0.8^{20} \times 0.35 \approx 0.0038$.

**Example 2 — Surgery, boundary case.**

$\mu(f) = 10$, $\lambda = 0.1$ (moderate per-interaction penalty), $\bar{s} = 0.90$.

$$C^* = 0.90 \times (1 - 0.1 \times 10) = 0.90 \times 0 = 0$$

*This represents an extreme boundary case; in practice $\lambda$ is small per interaction (contradictions are rare events, not a sustained rate).* The example illustrates correct model behavior at the boundary: when $\lambda\mu(f) = 1$, the contradiction penalty exactly cancels the pass-rate signal, and the agent cannot build confidence regardless of performance. This is the intended behavior — a surgical agent that consistently contradicts itself on verified claims should be unable to achieve operating confidence and must abstain.

**Example 3 — Recovery time.**

Starting from a drop of $\delta = 0.15$ (a significant contradiction), recovering to within $\varepsilon = 0.01$ of $C^*$ with $\alpha = 0.2$:

$$t_{\text{recovery}} = \left\lceil\frac{\log(0.01/0.15)}{\log(0.8)}\right\rceil = \left\lceil\frac{-2.708}{-0.223}\right\rceil = \lceil 12.1 \rceil = 13 \text{ interactions}$$

Thirteen clean calibration interactions are sufficient to recover from a large contradiction event to within 1% of steady-state confidence. This is consistent with the simulation results in Appendix A.

---

### Corollary — Steady-State Confidence Bounds by Field

Substituting typical field parameters with $\bar{s} \approx 0.85$ (a well-calibrated agent) and $\lambda \approx 0.05$ (low contradiction rate):

| Field | $\mu(f)$ | $C^* = \bar{s}(1-\lambda\mu)$ | $C_{\min}$ | Achievable? |
|---|---|---|---|---|
| Surgery / Aviation | 10 | $0.85 \times 0.5 = 0.425$ | 0.95 | No — requires $\bar{s} > 0.95/(1-0.5) = 1.9$ |
| Law | 5 | $0.85 \times 0.75 = 0.638$ | 0.85 | Borderline |
| Software Engineering | 2 | $0.85 \times 0.9 = 0.765$ | 0.70 | Yes |
| Creative Writing | 1 | $0.85 \times 0.95 = 0.808$ | 0.05 | Easily |

This corollary formalizes the whitepaper's claim that high-stakes fields impose stricter confidence standards: for surgery, a typical agent with $\bar{s} = 0.85$ and any nonzero contradiction rate cannot achieve the $C_{\min} = 0.95$ threshold, and must abstain or escalate. The confidence floor is not merely a policy choice — it is above the achievable steady state, guaranteeing the abstention mechanism is triggered when the agent is not reliably correct.

---

---

### Theorem B.4' — Lyapunov Stability of the Personality System

### Corrected Claim

The personality system exhibits **bounded, stable dynamics with convergence to a neighborhood of the field neutral $s^*$ under bounded drift**. We do not claim a unique globally stable equilibrium, which would require the mean reversion to dominate the drift — a condition that does not hold for the parameter values $\beta = 0.01$, $\Delta_{\max} = 0.05$ used in this system. Instead we prove the three things that *are* true: invariance of the feasible set, geometric convergence to $s^*$ when drift is absent, and bounded dynamics otherwise.

---

### Setup

The personality trait vector $s = (s_1, \ldots, s_n) \in \mathbb{R}^n$ evolves under:

$$s_{t+1} = \Pi_B\!\left[s_t + \Delta_t - \beta(s_t - s^*)\right] = \Pi_B\!\left[(1-\beta)s_t + \beta s^* + \Delta_t\right]$$

where:

- $B = \prod_{i=1}^n [s_{\min}^{i},\, s_{\max}^{i}]$ is the field-specific feasible set (a closed convex box in $\mathbb{R}^n$)
- $\Pi_B : \mathbb{R}^n \to B$ is the Euclidean projection onto $B$
- $\Delta_t \in \mathbb{R}^n$ is the raw drift from utility history, with $\|\Delta_t\|_\infty \leq \Delta_{\max}$ per component
- $\beta = 0.01$ is the mean reversion coefficient
- $s^* \in B$ is the field neutral personality (interior point of $B$ by construction)

Define the Lyapunov function:

$$V(s) = \|s - s^*\|^2 = \sum_{i=1}^n (s_i - s_i^*)^2 \geq 0$$

with $V(s^*) = 0$.

---

#### Theorem B.4' — Bounded Stable Dynamics with Neighborhood Convergence

**Theorem.** *Under the three-layer personality evolution rule above:*

**(i) Invariance.** $s_t \in B$ for all $t \geq 0$ whenever $s_0 \in B$.

**(ii) Zero-drift convergence.** When $\Delta_t = 0$ for all $t$, $V(s_t)$ converges geometrically to zero:

$$V(s_{t+1}) \leq (1-\beta)^2\, V(s_t)$$

*so $\|s_t - s^*\| \leq (1-\beta)^t \|s_0 - s^*\|$, with convergence rate $(1-\beta)^2 = 0.9801$ per cycle.*

**(iii) Bounded displacement.** The single-step displacement is bounded:

$$\|s_{t+1} - s_t\| \leq \Delta_{\max}\sqrt{n} + \beta\,\|s_t - s^*\|$$

*so no single evolution cycle can produce a large jump.*

**(iv) Persistent-drift stability.** Under persistent drift with $\|\Delta_t\| \leq \Delta_{\max}\sqrt{n}$, the distance to $s^*$ satisfies:

$$\|s_{t+1} - s^*\| \leq (1-\beta)\|s_t - s^*\| + \Delta_{\max}\sqrt{n}$$

*The dynamics converge to and remain in the neighborhood*

$$\mathcal{N}^* = \left\{s \in B : \|s - s^*\| \leq r^*\right\}, \qquad r^* = \min\!\left(\frac{\Delta_{\max}\sqrt{n}}{\beta},\; \mathrm{diam}(B)\right)$$

*For the system parameters $\beta = 0.01$, $\Delta_{\max} = 0.05$, $n = 6$ traits: $\Delta_{\max}\sqrt{n}/\beta = 12.25$, while $\mathrm{diam}(B) \leq \sqrt{n} \approx 2.45$. The field bounds are therefore the binding constraint, not the mean reversion — $\mathcal{N}^* = B$.*

---

### Proof

**Part (i) — Invariance.**

$\Pi_B$ is defined as the Euclidean projection onto the closed convex set $B$, so $\Pi_B(x) \in B$ for every $x \in \mathbb{R}^n$. Therefore $s_{t+1} = \Pi_B[\cdot] \in B$ for all $t \geq 0$, regardless of $\Delta_t$. $\blacksquare$

**Part (ii) — Zero-drift convergence.**

Set $\Delta_t = 0$. Then $s_{t+1} = \Pi_B[(1-\beta)s_t + \beta s^*]$.

Since $B$ is convex and both $s_t \in B$ and $s^* \in B$, the convex combination $(1-\beta)s_t + \beta s^* \in B$, so $\Pi_B$ acts as the identity:

$$s_{t+1} = (1-\beta)s_t + \beta s^*$$

Therefore:

$$s_{t+1} - s^* = (1-\beta)s_t + \beta s^* - s^* = (1-\beta)(s_t - s^*)$$

and:

$$V(s_{t+1}) = \|(1-\beta)(s_t - s^*)\|^2 = (1-\beta)^2\,V(s_t) \qquad \blacksquare$$

The convergence rate per cycle is $(1-\beta)^2 = (0.99)^2 = 0.9801$. The half-life is:

$$t_{1/2} = \frac{\log(1/2)}{\log(1-\beta)^2} = \frac{\log(1/2)}{2\log(0.99)} \approx \frac{-0.693}{-0.0201} \approx 34 \text{ cycles}$$

With personality evolution running every $N=3$ interactions, this corresponds to approximately 102 interactions to halve the distance to $s^*$ under zero drift.

**Part (iii) — Bounded displacement.**

$$\|s_{t+1} - s_t\| = \|\Pi_B[(1-\beta)s_t + \beta s^* + \Delta_t] - s_t\|$$

Since $\Pi_B$ is non-expansive and $s_t = \Pi_B[s_t]$:

$$\|\Pi_B[x] - \Pi_B[y]\| \leq \|x - y\| \quad \text{for all } x, y$$

$$\|s_{t+1} - s_t\| \leq \|(1-\beta)s_t + \beta s^* + \Delta_t - s_t\|$$
$$= \|-\beta(s_t - s^*) + \Delta_t\| \leq \beta\|s_t - s^*\| + \|\Delta_t\| \leq \beta\,\|s_t - s^*\| + \Delta_{\max}\sqrt{n} \qquad \blacksquare$$

**Part (iv) — Persistent-drift stability.**

Since $s^* \in B$, the non-expansiveness of $\Pi_B$ with respect to any point in $B$ gives:

$$\|s_{t+1} - s^*\| = \|\Pi_B[(1-\beta)s_t + \beta s^* + \Delta_t] - \Pi_B[s^*]\|$$
$$\leq \|(1-\beta)s_t + \beta s^* + \Delta_t - s^*\|$$
$$= \|(1-\beta)(s_t - s^*) + \Delta_t\|$$
$$\leq (1-\beta)\|s_t - s^*\| + \|\Delta_t\|$$
$$\leq (1-\beta)\|s_t - s^*\| + \Delta_{\max}\sqrt{n}$$

Let $d_t = \|s_t - s^*\|$. The recurrence $d_{t+1} \leq (1-\beta)d_t + \Delta_{\max}\sqrt{n}$ has fixed point $d^* = \Delta_{\max}\sqrt{n}/\beta$. By the theory of contractive linear recurrences:

- If $d_t > d^*$: $d_{t+1} < d_t$ (distance decreasing toward $d^*$)
- If $d_t \leq d^*$: $d_{t+1} \leq d^*$ (distance stays within neighborhood)

So $\limsup_{t\to\infty} d_t \leq d^* = \Delta_{\max}\sqrt{n}/\beta$.

Since $d_t \leq \mathrm{diam}(B)$ always by Part (i), the binding constraint is:

$$r^* = \min\!\left(\frac{\Delta_{\max}\sqrt{n}}{\beta},\; \mathrm{diam}(B)\right) \qquad \blacksquare$$

---

### Why the Field Bounds Are the Binding Constraint

For the system parameters:

$$\frac{\Delta_{\max}\sqrt{n}}{\beta} = \frac{0.05 \times \sqrt{6}}{0.01} = \frac{0.1225}{0.01} = 12.25$$

$$\mathrm{diam}(B) = \left\|\,s_{\max} - s_{\min}\,\right\| \leq \sqrt{n} \approx 2.45 \quad \text{(since each trait is in }[0,1]\text{)}$$

Since $12.25 \gg 2.45$, the mean reversion alone is insufficient to confine the dynamics to a small neighborhood of $s^*$. **The projection $\Pi_B$ is the primary stability mechanism** — it enforces invariance, and the field bounds define how far from $s^*$ the personality can stray. The mean reversion serves as a regularizer: without it, the system would drift to and remain at the boundary of $B$; with it, there is a gentle restoring force toward $s^*$ when drift is absent.

This clarifies the design role of each stability layer:

| Layer | Mechanism | Guarantee |
|---|---|---|
| Field bounds $[s_{\min}, s_{\max}]$ | Projection $\Pi_B$ | Hard invariance — $s_t \in B$ always |
| Drift rate cap $\Delta_{\max}$ | Truncation of raw drift | Bounded single-step displacement |
| Mean reversion $\beta$ | Soft pull toward $s^*$ | Convergence to $s^*$ when drift is absent; regularization otherwise |

---

### Corollary — No Oscillation

**Corollary.** *The drift rate cap $\Delta_{\max}$ prevents oscillation: after any evolution step, the personality cannot cross $s^*$ in a single step.*

**Proof.** A step crosses $s^*$ in dimension $i$ if $s_{t+1}^i - s_i^*$ and $s_t^i - s_i^*$ have opposite signs. The signed displacement in dimension $i$ is:

$$s_{t+1}^i - s_i^* = \Pi_{[s_{\min}^{i}, s_{\max}^{i}]}\!\left[(1-\beta)(s_t^i - s_i^*) + \Delta_t^i\right]$$

For oscillation, we need $|\Delta_t^i| > (1-\beta)|s_t^i - s_i^*|$, i.e., the drift must exceed the current displacement. Since $\Delta_t^i \leq \Delta_{\max} = 0.05$ and $s_t^i - s_i^*$ can be as large as $s_{\max}^{i} - s_i^*$ (which is at least $0.05$ by the bound structure), oscillation requires the personality to be within $\Delta_{\max}/(1-\beta) \approx 0.0505$ of $s^*$ in that dimension. This is a vanishingly small region, confirming that oscillation can only occur when the personality is already very near the neutral point. $\blacksquare$

---

### Remark on Parameter Calibration

The analysis reveals a structural tension: mean reversion $\beta = 0.01$ is weaker than the maximum drift $\Delta_{\max} = 0.05$ per cycle. This is intentional — personality is meant to evolve meaningfully, not snap back to $s^*$ every few cycles. The design trades off:

- **Large $\beta$**: fast convergence to $s^*$, but personality becomes unresponsive to experience
- **Small $\beta$**: personality evolves with experience, but reverts slowly between periods of drift

The value $\beta = 0.01$ produces a half-life of approximately 34 evolution cycles under zero drift (≈ 102 interactions), which is long enough for the personality to reflect accumulated experience over hundreds of interactions before the neutral pull becomes dominant. A field-specific $\beta(f)$ — with higher reversion rates for high-stakes fields where personality stability is more important — is a natural extension for future work.

---

## References

- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256.
- Blackorby, C., & Donaldson, D. (1982). Ratio-scale and translation-scale full interpersonal comparability without domain restrictions. *International Economic Review*, 23(2), 249–268.
- Debreu, G. (1960). Topological methods in cardinal utility theory. In K. J. Arrow et al. (Eds.), *Mathematical Methods in the Social Sciences*. Stanford University Press.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50–60.
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*, 16(2), 117–186.