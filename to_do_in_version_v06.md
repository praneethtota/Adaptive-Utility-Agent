# To Do in Version v.06.html

## Goal

Build a privacy-first MVP backend that wraps a frontier LLM (for example GPT or Claude) and reduces repeated hallucinations by learning from prior failures at the **single-user level first**, then optionally extending to **cross-user correction sharing**.

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

The system should not try to change the base model weights directly. Instead, it should introduce a **control layer** around the model:

1. normalize and classify the query
2. retrieve relevant prior corrections
3. call the model
4. validate the output
5. retry or correct when needed
6. store failure/correction memory
7. reuse validated corrections on similar future queries

This is the practical MVP form of the utility-governed control system.

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

Default rule:
- raw queries
- raw model answers
- user-specific notes
- personal context
- uploaded files
- private corrections

should remain **local only** unless the user explicitly opts in to sharing a stripped-down correction record.

---

## High-Level Architecture

```text
User
  ->
API / App Layer
  ->
Query Normalizer + Domain Classifier
  ->
Local Memory Retrieval
  ->
Prompt Builder / Context Grammar Layer
  ->
LLM Gateway
  ->
Validator / Contradiction Detector
  ->
Retry Controller
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

---

## 2. Query Normalizer Service
Responsibilities:
- convert raw user query into a simplified canonical form
- infer domain
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
- domain
- optional subdomain
- extracted entities / constraints

Important:
This canonical form should be:
- compact
- stripped of personal details where possible
- reusable as a key for correction matching

---

## 3. Domain Classifier
Responsibilities:
- assign domain probabilities
- pick effective domain
- apply conservative logic for high-stakes domains if needed

Example domains:
- software_engineering
- math
- sql
- medicine
- legal
- creative_writing
- general_factual

The domain will govern:
- validation method
- correction rules
- confidence threshold
- whether cross-user sharing is allowed by default

---

## 4. Local Memory Retrieval Service
Responsibilities:
- fetch prior corrections for similar canonical queries
- fetch domain-relevant historical patterns
- retrieve prior user preferences / context grammar
- return only relevant memory, not full history

This service is the main engine for reducing repeated errors.

---

## 5. Prompt Builder / Context Grammar Layer
Responsibilities:
- build final prompt for the LLM
- inject active corrections
- inject user-specific context
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
- code -> tests, lint, static checks
- math -> symbolic / numeric verification
- SQL -> parse + rules + optional execution in sandbox
- factual -> retrieval or grounded lookup
- structured tasks -> rule-based consistency checks

Outputs:
- pass / fail / uncertain
- error type
- confidence impact
- correction candidate

---

## 8. Retry Controller
Responsibilities:
- decide whether to:
  - accept answer
  - retry once with correction injection
  - hedge / abstain
  - escalate
- prevent infinite retry loops
- update local confidence / utility state

Basic MVP retry policy:
- first failure -> generate correction + retry once
- second failure -> return best answer with uncertainty or abstain
- log for future correction memory

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
- SQLite first
- move to PostgreSQL localhost later if needed

### Tables

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
- prompt_with_injections
- raw_model_output
- final_output
- utility_score
- confidence_score
- efficacy_score
- curiosity_score
- validation_status
- retry_count
- created_at

Notes:
- prompt and output stay local by default

---

## corrections
Columns:
- correction_id
- user_id
- normalized_query_id
- canonical_query
- domain
- error_type
- bad_pattern_summary
- correction_text
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
Columns:
- context_id
- user_id
- domain
- key
- value
- source
- confidence
- created_at
- updated_at

Examples:
- key = preferred_answer_style
- value = concise_backend_focused
- key = known_stack
- value = python_sql_distributed_systems

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

---

## Single-User Request Flow

```text
1. User submits query
2. Query normalizer creates canonical query + domain
3. Local memory retrieval fetches matching prior corrections + context grammar
4. Prompt builder injects:
   - active corrections
   - user context grammar
   - domain guardrails
5. LLM gateway calls model
6. Validator checks answer
7. If pass:
      save result + update local confidence
      return answer
8. If fail:
      create correction candidate
      retry once with correction injection
9. Save correction memory locally
10. Return corrected answer or abstention
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
- created_at
- updated_at

---

## shared_correction_votes
Optional if you want trust weighting later.

Columns:
- vote_id
- shared_correction_id
- local_client_id
- helped (boolean)
- confidence_delta
- created_at

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
1. User submits query
2. Query normalizer creates canonical query + domain
3. Local memory retrieval runs first
4. If user has opted in to shared corrections:
      query central shared correction service
5. Merge:
      local corrections take priority
      shared corrections are secondary
6. Prompt builder injects corrections
7. LLM gateway calls model
8. Validator checks answer
9. Save results locally
10. If a new validated correction is created:
      show user the exact shareable payload
11. Upload only if user approves
```

---

## Conflict Resolution Rule

If a local correction and a shared correction disagree:
- local correction wins
- user-local privacy and direct evidence take precedence

If multiple shared corrections disagree:
- prefer highest-confidence
- prefer freshest
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

---

## Context Grammar for Personalization Across Chats

### Keynote

The **canonical form + domain stored in localhost** can also serve as a **context grammar** that helps the model personalize responses and retain useful user-provided data over multiple chats.

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
This is still not true base-model memory.
It is:
- local memory retrieval
- structured personalization
- wrapper-level continuity

But for many practical use cases, this is enough to make the system feel significantly more personalized and less stateless.

---

## Proposed v0.6 Deliverables

## Must-have
- localhost DB
- canonical query normalizer
- domain classifier
- correction memory
- one-retry correction loop
- local context grammar
- coding/domain validator for MVP

## Should-have
- per-domain confidence state
- correction application metrics
- privacy audit log
- share-review screen for opt-in uploads

## Later extension
- central shared correction service
- shared correction ranking
- deletion / revocation support
- stronger domain-specific validators
- multi-tenant security hardening

---

## Suggested Tech Stack

### Local-first MVP
- FastAPI
- SQLite
- SQLAlchemy
- Redis optional for short-lived cache
- OpenAI / Anthropic API gateway
- local embeddings for correction retrieval if possible

### Optional central service
- FastAPI
- PostgreSQL
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
- validation is easier
- repeated error patterns are common
- privacy risks are lower than medicine/legal
- success is easier to measure

---

## Final Positioning for v0.6

Version v0.6 should turn the paper into a practical prototype by implementing:

- a privacy-first single-user correction memory on localhost
- a canonical-query + domain retrieval layer
- a correction-injection retry loop
- a local context grammar for personalization across chats
- an optional, transparent, opt-in multi-user correction sharing layer

This is the most realistic wrapper-based path toward reducing repeated hallucinations on top of frontier hosted models.
