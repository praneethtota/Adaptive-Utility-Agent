# To Do in Version v0.6 — Revised Design

> **Revision note.** This document incorporates a full design review of the original v0.6 spec.
> Additions and changes are marked **[REVISED]** or **[ADDED]**; unchanged sections are left as-is.
> The structure, scope, and two-phase privacy model from the original spec are preserved.
> Changes address: canonicalization approach, retry logic, prompt storage privacy,
> context grammar schema, domain sequencing, shared correction versioning,
> DPO-readiness, domain-gated retry policy, real-time utility proxy, and must-have scope.

---

## Goal

Build a privacy-first MVP backend that wraps a frontier LLM (for example GPT or Claude) and reduces
repeated hallucinations by learning from prior failures at the **single-user level first**, then
optionally extending to **cross-user correction sharing**.

This document describes:

- MVP backend design
- services
- tables
- request flow
- localhost-first privacy model
- opt-in cross-user correction sharing
- a note on canonical query memory as context grammar for personalization across chats

---

## Core Design Principle

The system should not try to change the base model weights directly. Instead, it should introduce a
**control layer** around the model:

1. normalize and classify the query
2. retrieve relevant prior corrections
3. call the model
4. validate the output
5. retry or correct when needed
6. store failure/correction memory
7. reuse validated corrections on similar future queries

This is the practical MVP form of the utility-governed control system described in the whitepaper (§7–8).

---

## Scope for v0.6

### Phase 1: Single-user local memory
- database runs on localhost only
- no raw user data leaves the machine
- correction memory is private to the user
- system learns from repeated errors across that user's chats

### Phase 2: Optional cross-user correction sharing
- extension to Phase 1
- canonical form + domain + correction results may be uploaded to a central server
- this must be **opt-in**
- the user must be able to see exactly what is being sent before upload
- no raw chat history should be uploaded by default

---

## Privacy Requirement

For the single-user MVP, the database should stay on **localhost** for privacy.

Recommended local deployment options:
- SQLite for the fastest MVP
- PostgreSQL on localhost for a more production-like local setup

Default rule — the following should remain **local only** unless the user explicitly opts in:
- raw queries
- raw model answers
- user-specific notes
- personal context
- uploaded files
- private corrections
- assembled prompts with injections **[REVISED — see model_runs table note]**

---

## High-Level Architecture

```text
User
  ->
API / App Layer
  ->
Query Normalizer
  ->
Soft Domain Classifier (initial pass)     [REVISED — soft pass first, see §4]
  ->
Local Memory Retrieval (correction-informed domain refinement)
  ->
Domain Finalizer                          [ADDED — refines domain from retrieved corrections]
  ->
Prompt Builder / Context Grammar Layer
  ->
LLM Gateway
  ->
Validator / Contradiction Detector
  ->
Domain-Gated Retry Controller             [REVISED — high-stakes domains skip retry]
  ->
Local Memory Store
```

Optional extension:

```text
Local Correction Memory
  ->
Share Review Screen
  ->
Opt-In Uploader
  ->
Central Shared Correction Service
  ->
Shared Correction Retrieval
  ->
Local Injection into future prompts
```

---

## Services for v0.6

## 1. API / App Service

Responsibilities:
- receives user query
- authenticates local user session
- orchestrates request lifecycle
- returns final answer plus optional explanation of what corrections were applied

Possible stack:
- FastAPI
- Flask
- Node backend if preferred, but Python is simpler for this MVP

**[REVISED — stack guidance]** For the single-user localhost MVP: use FastAPI with raw `sqlite3`
(not SQLAlchemy) and no Redis. SQLAlchemy and Redis add dependencies without meaningful benefit at
single-user scale. Introduce SQLAlchemy only when moving to a multi-user PostgreSQL deployment.

---

## 2. Query Normalizer Service

Responsibilities:
- convert raw user query into a simplified canonical form
- infer domain (soft probability distribution — not a hard single label)
- extract constraints
- identify whether query resembles a previously seen problem

Example:

Raw query:
- "Why is my SQLite DB losing writes?"
- "My inserts vanish in sqlite unless I restart"
- "why aren't sqlite changes persisting?"

Canonical form:
- `sqlite write persistence transaction commit issue`

Outputs:
- canonical query
- domain probability distribution (soft — not a single label)
- optional subdomain
- extracted entities / constraints

**[ADDED — canonicalization approach]**

Canonicalization is the hardest and most failure-prone component of the system. The choice of
approach determines correction retrieval accuracy across the entire pipeline. Three options exist:

| Approach | Speed | Auditability | Risk |
|---|---|---|---|
| LLM-based | Slow (extra round trip) | Low (black box output) | Circular dependency on model being corrected |
| Embedding similarity | Fast | Low (no interpretable key) | Unpredictable collision rate |
| Rule-based NLP pipeline | Fast | High (deterministic output) | Brittle on domain-crossing queries |

**Recommended for MVP:** rule-based hybrid. Use spaCy or a lightweight fine-tuned model for entity
extraction and domain classification, then construct the canonical form deterministically from
extracted slots:

```
canonical_form = domain_slug + "_" + error_category_slug + "_" + entity_slugs
```

Example: `software_engineering_transaction_semantics_sqlite_autocommit`

This approach is:
- fast (no extra LLM call)
- auditable (the canonical form is human-readable and deterministic)
- independent of the model being wrapped
- directly usable as a database key for exact and fuzzy matching

Important: the canonical form should be compact, stripped of personal details where possible,
and reusable as a key for correction matching.

Upgrade path: once enough correction data has accumulated, replace or augment string matching
with embedding-based retrieval (cosine similarity over canonical form embeddings). Do not start
there — string matching on canonical forms is sufficient for MVP and far easier to debug.

---

## 3. Domain Classifier

**[REVISED — soft pass, then refinement]**

The domain classifier runs in two passes, not one.

### Pass 1 — Soft domain distribution (before retrieval)

Assign a probability distribution over candidate domains from the canonical query alone. Do not
commit to a single domain yet.

Example domains:
- software_engineering
- math
- sql
- medicine
- legal
- creative_writing
- general_factual

The domain distribution governs:
- which correction records are retrieved (weighted by domain probability)
- validation method applied
- correction rules
- confidence threshold
- whether cross-user sharing is allowed by default

### Pass 2 — Domain refinement (after retrieval)

After local memory retrieval returns matching corrections, refine the domain estimate. If
retrieved corrections strongly cluster in a domain that differs from the initial soft estimate,
update the distribution before building the prompt. This prevents ambiguous queries (e.g., "why is
my query returning duplicate rows?" — could be SQL, software_engineering, or distributed systems)
from being misrouted when the user's correction history provides a clearer signal.

Effective domain is then:

```
effective_domain = argmax(P(domain | canonical_query, retrieved_corrections))
```

Soft-domain weighting logic mirrors the field classifier in the whitepaper (§4.2): use a
high-stakes floor — any high-stakes domain (medicine, legal, finance) with meaningful probability
mass is floored at 0.15 minimum before retrieval to avoid under-weighting dangerous domains.

---

## 4. Local Memory Retrieval Service

Responsibilities:
- fetch prior corrections for similar canonical queries
- fetch domain-relevant historical patterns
- retrieve prior user preferences / context grammar
- return only relevant memory, not full history

**[REVISED — retrieval informs domain, not the reverse]**

Retrieval runs against the soft domain distribution from Pass 1 (above), not against a finalized
single domain. This allows the retrieved corrections to inform domain finalization before the
prompt is built. The service returns corrections weighted by:

1. canonical form similarity (exact match first, fuzzy match second)
2. domain probability weight
3. correction confidence
4. recency (staleness-penalized)

This service is the main engine for reducing repeated errors.

---

## 5. Prompt Builder / Context Grammar Layer

Responsibilities:
- build final prompt for the LLM
- inject active corrections (retrieved from local memory)
- inject user-specific context (from context grammar table)
- inject domain-specific guardrails
- inject personalization hints from prior local memory

Example injected block:

```text
ACTIVE CORRECTIONS:
- For SQLite write persistence issues, prefer explicit transactions/commit semantics.
- Do not recommend relying on autocommit without clarifying engine behavior.

USER CONTEXT:
- User prefers concise backend-focused answers.
- User frequently works in Python, SQL, and distributed systems.
```

---

## 6. LLM Gateway

Responsibilities:
- call GPT / Claude / other hosted model
- support configurable model selection
- track request metadata
- store request/response references locally

The LLM gateway should be thin and replaceable.

---

## 7. Validator / Contradiction Detector

Responsibilities:
- check model output for likely failure
- validate against domain-specific rules
- compare against prior validated corrections
- flag contradictions or repeated hallucinations

Validation examples:
- code → tests, lint, static checks (AST-based, as in the existing `contradiction_detector.py`)
- math → symbolic / numeric verification
- SQL → parse + rules + optional execution in sandbox
- factual → retrieval or grounded lookup
- structured tasks → rule-based consistency checks

Outputs:
- pass / fail / uncertain
- error type
- confidence impact
- correction candidate

**[ADDED — correction candidate generation on first failure]**

On first failure, the system does not yet have a stored correction to inject. The validator must
generate a correction candidate from its own output. Three mechanisms, applied in order:

1. **Structured validator output → correction template.** If the validator is deterministic
   (e.g., AST-based complexity check, SQL parser), it can produce a specific structured error
   description that maps to a pre-defined correction template. Example:
   ```
   validator detects: nested_loop_depth=2, claimed_complexity="O(n)"
   → correction_template: "Code has nested loops inconsistent with claimed O(n).
     Use a hash-based approach or correct the complexity claim."
   ```
   This is cheap, reliable, and produces auditable correction text.

2. **LLM self-correction prompt.** If the validator flags a failure but cannot produce a
   structured explanation (e.g., factual hallucination), ask the model to self-correct:
   ```
   "Your previous answer had the following problem: {validator_output}.
    Please try again, correcting that specific issue."
   ```
   Use this as a fallback, not a primary path. The same model that failed may fail again the
   same way, but it will often self-correct surface-level errors.

3. **Retrieval from prior corrections.** From cycle 2 onward, when correction memory has
   accumulated, retrieve the best matching prior correction for this canonical query and
   error type instead of generating a new one. This is the steady-state mechanism and the
   primary path described in the whitepaper (§7.1 — Layer 1 behavioral correction).

The correction candidate generated by mechanism 1 or 2 on first failure is stored to the
`corrections` table after the retry completes, so it becomes available for mechanism 3 in
future cycles.

---

## 8. Retry Controller

**[REVISED — domain-gated retry policy]**

Responsibilities:
- decide whether to:
  - accept answer
  - retry once with correction injection
  - hedge / abstain
  - escalate
- prevent infinite retry loops
- update local confidence / utility state

### Domain-gated retry policy

The retry behaviour depends on domain risk level. High-stakes domains (medicine, legal, finance,
aviation) **skip the automated LLM retry entirely**. Asking the same model that produced a wrong
medical or legal answer to self-correct is actively dangerous. Instead, high-stakes failures go
directly to abstention and escalation.

```
if domain.penalty_multiplier >= 5.0:        # medicine, legal, aviation
    → skip retry
    → return answer with explicit uncertainty hedge:
      "I have limited confidence in this answer. Please verify with a domain expert."
    → log to correction memory for future suppression
    → flag for human review queue (if implemented)
else:
    → retry once with correction candidate injection
    → if retry passes: store as validated correction
    → if retry fails: return best answer with uncertainty, log both attempts
```

The `penalty_multiplier >= 5.0` threshold maps directly to the field configs in `config.py`
(medicine = 10×, legal = 5×). No new configuration needed.

### Basic MVP retry flow (non-high-stakes)

```
first failure
  → generate correction candidate (see §7 above)
  → retry once with correction injection
  → if pass: store validated correction, return corrected answer
  → if fail: return best attempt with uncertainty note, log both
```

---

## 9. Local Memory Store

Responsibilities:
- persist all user-local correction memory
- persist user-local context grammar
- remain on localhost by default

This is the main privacy-preserving state layer.

---

## 10. Optional Shared Correction Service

Responsibilities:
- receive opt-in correction uploads
- store canonical form + domain + validated correction result
- provide shared correction retrieval for similar future queries

This service should never receive private raw user data by default.

---

## Local Database Design (Single User)

### Recommended local DB

**[REVISED]** SQLite only for MVP. Do not introduce SQLAlchemy or PostgreSQL at this stage.
Use raw `sqlite3` with explicit schema migrations. Upgrade to PostgreSQL when moving to
multi-user or when local DB exceeds 1 GB.

---

## users

For a local-first single-user MVP, this can still exist for future-proofing.

Columns:
- user_id
- display_name
- created_at
- sharing_opt_in (boolean)
- settings_json

---

## conversations

Columns:
- conversation_id
- user_id
- title
- created_at
- updated_at

---

## messages

Columns:
- message_id
- conversation_id
- role
- raw_text
- created_at

Notes:
- raw_text stays local
- never uploaded unless explicitly exported by the user

---

## normalized_queries

Columns:
- normalized_query_id
- user_id
- conversation_id
- source_message_id
- raw_query_hash
- canonical_query
- domain
- subdomain
- entities_json
- constraints_json
- created_at

Notes:
- `canonical_query` is the main retrieval key
- `raw_query_hash` can help deduplicate locally

---

## model_runs

Columns:
- run_id
- normalized_query_id
- model_name
- prompt_version
- corrections_applied_ids    **[REVISED — replaces prompt_with_injections]**
- context_keys_used          **[REVISED — replaces full context dump]**
- raw_model_output
- final_output
- utility_score_proxy        **[REVISED — see utility proxy note below]**
- confidence_score
- efficacy_score
- curiosity_score
- validation_status
- retry_count
- created_at

**[REVISED — do not store prompt_with_injections]**

The original spec stored the full assembled prompt (including injected corrections and user
context) in `model_runs`. This contradicts the privacy model: the assembled prompt is the union
of the user's query, their personal context, and their correction history — a more sensitive
artifact than any individual field. It should not be stored at rest.

Instead:
- `corrections_applied_ids`: array of FK references to `corrections.correction_id` — the
  corrections that were injected into this prompt
- `context_keys_used`: array of `user_context_grammar.key` values that were activated

The full prompt can always be reconstructed from these references if needed for debugging, without
holding the assembled artifact at rest.

**[ADDED — real-time utility proxy]**

The whitepaper's full utility function U = w_e·E + w_c·C + w_k·K requires test pass rates,
cross-session EMA, and novelty tracking — none of which can be computed synchronously inside a
live request without blocking. Use a real-time proxy for `utility_score_proxy`:

```
utility_score_proxy = w_e × validator_pass_fraction
                    + w_c × local_confidence_estimate
                    + w_k × novelty_flag

where:
  validator_pass_fraction   = fraction of validator checks that passed (0.0–1.0)
  local_confidence_estimate = EMA of recent validation pass rates for this canonical query
                              (from confidence_state table, α = 0.2)
  novelty_flag              = 1.0 if no prior correction exists for this canonical query,
                              0.0 otherwise
  w_e, w_c, w_k            = field weights from config (software_engineering: 0.55, 0.35, 0.10)
```

This is computable in milliseconds. Document it explicitly as a real-time proxy. The full U
formulation (whitepaper §3) applies at calibration-cycle level (daily/weekly batch over stored
model_runs), not per-request.

Notes:
- raw_model_output and final_output stay local
- prompt and output stay local by default

---

## corrections

**[REVISED — added DPO-ready columns]**

Columns:
- correction_id
- user_id
- normalized_query_id
- canonical_query
- domain
- error_type
- bad_pattern_summary
- correction_text
- rejected_output_ref       **[ADDED — FK to model_runs.run_id: the bad answer]**
- chosen_output             **[ADDED — the corrected answer text]**
- evidence_type
- evidence_summary
- confidence
- scope_json
- created_at
- last_used_at
- use_count

Examples:
- error_type = complexity_claim_wrong
- error_type = transaction_semantics_error
- error_type = fabricated_fact
- error_type = missing_edge_case

**[ADDED — DPO-readiness note]**

The whitepaper's Phase 7 distills accumulated corrections into a base model fine-tune via Direct
Preference Optimization (DPO). DPO training requires (rejected_output, chosen_output) pairs.
The two new columns make every correction record directly exportable as a DPO training pair
without a schema migration. Cost: two extra columns. Benefit: the path from wrapper → fine-tune
stays open from day one.

Export query for DPO pairs:

```sql
SELECT
  canonical_query          AS prompt,
  mr.raw_model_output      AS rejected,
  c.chosen_output          AS chosen,
  c.domain,
  c.error_type,
  c.confidence
FROM corrections c
JOIN model_runs mr ON c.rejected_output_ref = mr.run_id
WHERE c.confidence >= 0.7
  AND c.chosen_output IS NOT NULL;
```

---

## correction_applications

Columns:
- application_id
- correction_id
- run_id
- applied (boolean)
- helped (boolean)
- outcome_notes
- created_at

This table helps measure whether corrections are actually useful.

---

## user_context_grammar

**[REVISED — added last_confirmed_at and decay note]**

Columns:
- context_id
- user_id
- domain          (nullable — null means global preference)
- key
- value
- source          **[REVISED — values: 'user_explicit' | 'inferred' | 'correction_derived']**
- confidence
- last_confirmed_at  **[ADDED — tracks staleness]**
- created_at
- updated_at

Examples:
- key = preferred_answer_style,  value = concise_backend_focused
- key = known_stack,             value = python_sql_distributed_systems
- key = prefers_backend_first,   value = true
- key = project_context,         value = event-store api with duplicate detection

**[ADDED — staleness and source hierarchy]**

The `source` column governs persistence:
- `user_explicit`: highest persistence — decays only when the user explicitly updates it
- `inferred`: medium persistence — decays if not confirmed within 90 days
- `correction_derived`: lowest persistence — derived from correction patterns, treated as a hint

Staleness is tracked via `last_confirmed_at`. When building the prompt, down-weight context
grammar entries whose `last_confirmed_at` is older than the source-specific decay threshold:

```
user_explicit:        no decay (confirmed by definition when last updated)
inferred:             confidence × exp(−days_since_confirmed / 90)
correction_derived:   confidence × exp(−days_since_confirmed / 30)
```

This mirrors the assertions store decay class system from the whitepaper (§9.5, Class A–D)
applied to user-level context rather than domain facts.

This is the table that supports personalization across chats.

---

## confidence_state

Columns:
- state_id
- user_id
- domain
- confidence_value
- contradiction_count
- successful_corrections
- failed_corrections
- updated_at

---

## local_audit_log

**[REVISED — moved to must-have]**

Columns:
- audit_id
- user_id
- event_type
- event_payload
- created_at

Used for:
- privacy audit
- debugging
- proving what was stored locally vs shared

**[ADDED — must-have rationale]**

The audit log is the mechanism by which users can verify the privacy model is working. It should
be built from day one, not added later. Without it, the localhost-first privacy claim is
unauditable. Every correction stored, every shared upload initiated, and every context grammar
update should produce an audit event. This is what lets a user answer the question: "What does
this system actually know about me?"

---

## Single-User Request Flow

**[REVISED — reflects two-pass domain classification and domain-gated retry]**

```text
1.  User submits query
2.  Query normalizer creates canonical query
3.  Soft domain classifier produces domain probability distribution (Pass 1)
4.  Local memory retrieval fetches matching prior corrections + context grammar
    (weighted by soft domain distribution — not yet finalized)
5.  Domain finalizer refines domain estimate from retrieved corrections (Pass 2)
6.  Prompt builder injects:
    - active corrections
    - user context grammar
    - domain guardrails
7.  LLM gateway calls model
8.  Validator checks answer → produces pass/fail + error_type + correction_candidate
9.  If pass:
      store result (corrections_applied_ids, context_keys_used, utility_score_proxy)
      update confidence_state (EMA)
      return answer
10. If fail:
      check domain risk level
      if HIGH STAKES (penalty_multiplier >= 5.0):
        return hedge + uncertainty note
        log failure for correction memory
        skip retry
      else:
        inject correction_candidate
        retry once
        if retry passes:
          store validated correction (with rejected_output_ref + chosen_output)
          return corrected answer
        if retry fails:
          return best attempt with uncertainty note
          log both attempts
11. Write audit_log entry for every store or share event
```

---

## Why Localhost Matters

For v0.6 single-user MVP, localhost storage is important because it enables:
- private correction memory
- private user-specific context across chats
- reuse of prior mistakes without leaking chat history
- auditable local state
- easier trust adoption for early users

This is especially important if the system stores:
- debugging questions
- code snippets
- personal project details
- confidential work notes

---

## Multi-User Extension (Opt-In Only)

The multi-user system should be an **extension** of the single-user model, not a replacement.

Default remains:
- local-only

Only if the user opts in:
- selected correction records can be uploaded to a central service

---

## What gets uploaded

Only the following should be candidates for upload:
- canonical_query
- domain
- error_type
- correction_text
- evidence_type
- evidence_summary
- confidence
- scope metadata
- correction success/failure stats

Not uploaded by default:
- raw query
- raw answer
- chat history
- user identity
- uploaded files
- local context grammar
- personal notes
- assembled prompts **[ADDED]**
- rejected_output / chosen_output full text **[ADDED — only stats, not content]**

---

## User Visibility Requirement

Before anything is uploaded, the app must show the user exactly what will be sent.

Example review panel:

```text
You are about to share the following correction:

Canonical query:
sqlite write persistence transaction commit issue

Domain:
software_engineering

Error type:
transaction_semantics_error

Correction:
Prefer explicit transaction handling and commit semantics in SQLite answers.

Evidence:
Validated by local tests.

Include this in shared correction memory?
[Approve] [Reject] [Edit]
```

This should be mandatory.

---

## Opt-In Policy

Cross-user sharing must be:
- disabled by default
- enabled explicitly by the user
- revocable
- inspectable

The user should be able to:
- see shared records
- delete future sharing permission
- optionally remove past submitted shared records if the central service supports deletion

---

## Shared Correction Tables (Central Server)

## shared_corrections

**[REVISED — added versioning, status, and supercession]**

Columns:
- shared_correction_id
- canonical_query
- domain
- error_type
- correction_text
- evidence_type
- evidence_summary
- confidence
- scope_json
- freshness_ttl
- status              **[ADDED — 'pending' | 'validated' | 'deprecated' | 'contested']**
- version             **[ADDED — integer, incremented on update]**
- superceded_by       **[ADDED — FK to newer shared_correction_id, nullable]**
- created_at
- updated_at

**[ADDED — status lifecycle note]**

A correction that turns out to be wrong should not require deletion to stop being used. Instead:
- set `status = 'deprecated'` to stop new retrieval while preserving the record for audit
- set `status = 'contested'` when incoming vote data contradicts the correction above a threshold
- set `superceded_by` when a better correction for the same canonical query is validated

This prevents bad corrections from spreading while maintaining an auditable history of what was
shared and when.

**[ADDED — freshness_ttl guidance]**

`freshness_ttl` should map to the decay class system from the whitepaper (§9.5):
- mathematical facts (complexity proofs, algorithm correctness): no decay
- software engineering best practices: 6-month TTL (Class D — fast-changing)
- SQL semantics: 1-year TTL (Class C — moderate)
- factual claims: domain-dependent, default 3-month TTL

---

## shared_correction_votes

**[REVISED — removed local_client_id for privacy]**

The original spec included `local_client_id` on vote records. Even a pseudonymous client ID
creates a linkability risk: a user who consistently votes on corrections in a specific domain
reveals their interest profile to the central server. Unless per-client accountability is a
hard requirement, this column should be dropped.

Columns:
- vote_id
- shared_correction_id
- helped (boolean)
- confidence_delta
- created_at

Per-client accountability, if needed, should be implemented at the upload authentication layer
(TLS client certificates or anonymous credential schemes), not embedded in the correction data.

---

## shared_correction_stats

Columns:
- shared_correction_id
- retrieval_count
- apply_count
- success_count
- failure_count
- updated_at

---

## Multi-User Request Flow

```text
1.  User submits query
2.  Query normalizer creates canonical query + soft domain distribution
3.  Local memory retrieval runs first
4.  Domain finalizer refines domain from retrieved corrections
5.  If user has opted in to shared corrections:
      query central shared correction service
      merge: local corrections take priority, shared corrections are secondary
6.  Prompt builder injects corrections
7.  LLM gateway calls model
8.  Validator checks answer
9.  Save results locally (corrections_applied_ids, context_keys_used, utility_score_proxy)
10. If a new validated correction is created:
      show user the exact shareable payload (canonical form, domain, error_type, correction_text)
11. Upload only if user approves
```

---

## Conflict Resolution Rule

If a local correction and a shared correction disagree:
- local correction wins
- user-local privacy and direct evidence take precedence

If multiple shared corrections disagree:
- prefer highest confidence
- prefer freshest (lowest staleness relative to freshness_ttl)
- prefer `status = 'validated'` over `status = 'pending'`  **[ADDED]**
- optionally require multiple confirmations before use

---

## Security Notes for Multi-User Extension

### 1. Local-first by default
Raw interactions stay on localhost.

### 2. Central uploads are minimized
Only canonicalized correction records should be shared.

### 3. Transport security
Use TLS for all uploads.

### 4. At-rest protection
If central service exists, shared correction DB should be encrypted at rest.

### 5. Local auditability
User can see:
- what was shared
- when it was shared
- why it was shared

### 6. Sensitive domains
For medicine, legal, finance, or private enterprise contexts:
- disable cross-user sharing by default
- require stronger review before enabling
- **[ADDED]** automated retry is skipped for these domains regardless of sharing setting

---

## Context Grammar for Personalization Across Chats

### Keynote

The **canonical form + domain stored in localhost** can also serve as a **context grammar** that
helps the model personalize responses and retain useful user-provided data over multiple chats.

This matters because many current chat systems behave as though:
- each new chat is its own sandbox
- information provided in one chat or project does not reliably propagate to others

A local context grammar layer can partially solve that.

### What context grammar means here

Instead of storing raw transcripts and replaying them, store compact structured memory such as:
- preferred explanation style
- recurring technical domains
- known project context
- previously corrected misconceptions
- user-specific terminology
- known constraints or preferences

Examples:
- `prefers_backend_first = true`
- `known_stack = python, sql, distributed systems`
- `wants_blunt_feedback = true`
- `project_context = event-store api with duplicate detection`
- `avoid_repeating_correction = sqlite requires explicit commit guidance`

Then future chats can begin with:
- relevant local context grammar
- relevant correction memory
- relevant domain hints

This gives cross-chat continuity without sending full user history to a remote server.

### Important boundary

This is still not true base-model memory. It is:
- local memory retrieval
- structured personalization
- wrapper-level continuity

But for many practical use cases, this is enough to make the system feel significantly more
personalized and less stateless.

---

## Proposed v0.6 Deliverables

### Must-have **[REVISED]**

- localhost SQLite DB (raw sqlite3 — no SQLAlchemy)
- canonical query normalizer (rule-based NLP pipeline, not LLM-based)
- two-pass domain classifier (soft distribution → retrieval → refinement)
- correction memory with DPO-ready schema (rejected_output_ref + chosen_output)
- domain-gated one-retry correction loop (high-stakes domains skip retry)
- local context grammar with staleness tracking
- coding/SQL validator for MVP (AST-based + sandbox execution)
- **local_audit_log** (moved up from should-have — required for privacy claim)

### Should-have

- per-domain confidence state (EMA-updated from validation outcomes)
- correction application metrics
- real-time utility score proxy (stored in model_runs)
- share-review screen for opt-in uploads

### Later extension

- central shared correction service
- shared correction ranking and status lifecycle
- deletion / revocation support
- stronger domain-specific validators (math symbolic, SQL sandbox, factual retrieval)
- multi-tenant security hardening
- embedding-based correction retrieval (upgrade from string matching on canonical forms)
- DPO export pipeline (batch job over corrections table)

---

## Suggested Tech Stack

### Local-first MVP **[REVISED]**

- FastAPI
- SQLite (raw sqlite3 — not SQLAlchemy at this stage)
- spaCy or lightweight fine-tuned classifier for canonicalization
- OpenAI / Anthropic API gateway
- string-similarity matching on canonical forms for correction retrieval (no vector DB yet)

### Optional central service

- FastAPI
- PostgreSQL
- SQLAlchemy (appropriate at this scale)
- object-safe API contracts
- explicit review + upload endpoint
- admin dashboard for shared correction inspection

---

## Suggested First MVP Domain

Start with:
- software engineering
- SQL
- code debugging

Reasons:
- validation is easier (AST-based, test execution, SQL parsing — deterministic)
- repeated error patterns are common and well-defined
- privacy risks are lower than medicine/legal
- success is easier to measure
- the existing `contradiction_detector.py` and `config.py` from the simulation codebase
  can be reused directly **[ADDED — references existing agent code]**

---

## Final Positioning for v0.6

Version v0.6 should turn the paper into a practical prototype by implementing:

- a privacy-first single-user correction memory on localhost
- a canonical-query + domain retrieval layer (rule-based, auditable, no LLM dependency)
- a domain-gated correction-injection retry loop (high-stakes domains abstain, not retry)
- a local context grammar for personalization across chats (with staleness decay)
- a DPO-ready corrections table (rejected/chosen pairs from day one)
- a mandatory local audit log (privacy claim must be verifiable, not just asserted)
- an optional, transparent, opt-in multi-user correction sharing layer
  (with versioned shared corrections and no per-client vote linkability)

This is the most realistic wrapper-based path toward reducing repeated hallucinations on top of
frontier hosted models, building directly on the architecture and simulation results from the
whitepaper's §7–8 and Appendix A.
