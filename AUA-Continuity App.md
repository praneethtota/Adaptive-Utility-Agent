# AUA Continuity App

AUA Continuity is a local-first AI memory agent that watches connected AI sessions, stores
only durable corrections and instructions, and generates compact restart prompts so users
can continue projects without re-explaining context or repeating the same model mistakes.


## 1. Executive Summary

**AUA Continuity** is a local-first memory and correction-capture app for users of frontier
AI tools such as ChatGPT, Claude, Gemini, Cursor, Copilot, and other LLM-based assistants.

The app continuously observes AI conversations that the user explicitly routes through or
connects to the app, but it does **not** persist full raw conversations by default. Instead,
it detects durable correction signals, persistent instructions, project decisions, user
preferences, and known model failure patterns. It then distills those moments into
structured memory fragments stored locally in a database.

Those memories are later reconstituted into compact restart prompts, guardrail prompts, and
project-continuity prompts so users can resume work without starting from a blank screen and
without repeating the same model mistakes.

The core philosophy is:

```text Observe broadly. Store narrowly. Reuse precisely. ```

AUA Continuity is separate from the AUA Framework.

- **AUA Framework** is for developers, MLEs, AI platform engineers, and SWEs who want to
bind LLM specialists, route requests, score utility, arbitrate outputs, and build controlled
multi-model systems. - **AUA Continuity** is for everyday AI users who want persistent
project continuity, durable instructions, correction memory, and reduced repeated
hallucinations or repeated context loss across AI sessions.

---

## 2. Product Definition

AUA Continuity is a local-first memory agent that monitors AI conversations the user
connects, extracts durable instructions, corrections, project decisions, and known model
failure patterns, stores them as structured memory, and uses them to generate restart and
guardrail prompts for future AI sessions.

It is not primarily a chatbot.

It is not a general note-taking app.

It is not a full conversation archive.

It is a **correction distillation and continuity layer** for frontier AI usage.

---

## 3. Problem Statement

Current frontier AI tools have several persistent user-facing problems:

1. **Session loss** - Users leave a project for days or weeks. - When they return, the model
has no reliable memory of prior decisions. - Users must repeatedly explain context.

2. **Crash or reset recovery** - Browser tabs crash. - IDE sessions reset. - Long
conversations become unusable. - Users need a clean restart prompt.

3. **Repeated hallucinations** - The model repeats mistakes the user already corrected. -
Example: proposing SQLite after the user already decided production storage should be
Postgres.

4. **Persistent instruction loss** - Users say things like: - “Going forward, add comments
to generated Python code.” - “Do not merge the framework and app concepts.” - “Use
industry-standard structure from now on.” - Existing tools often fail to reliably preserve
these instructions across sessions or projects.

5. **High user friction** - Manual transcript copying, prompt bookkeeping, and note-taking
create extra work. - A good app should reduce effort, not add chores.

AUA Continuity addresses these issues by automatically detecting memory-worthy moments and
saving only distilled, useful memory fragments.

---

## 4. Target Users

AUA Continuity is intended for a broad AI-user audience.

Primary users:

- Software engineers using ChatGPT, Claude, Cursor, Copilot, or Gemini. - Students and
researchers using AI for long-running work. - Writers and product managers managing
persistent context. - Founders or builders working across multiple AI sessions. - Anyone who
frequently corrects AI tools and wants those corrections remembered.

Unlike the AUA Framework, this app is not only for MLEs or infrastructure engineers.

---

## 5. Product Boundary

AUA Continuity should not compete with the AUA Framework.

They are sibling products.

### AUA Framework

Audience:

```text MLEs AI infra engineers backend engineers AI platform teams advanced developers ```

Purpose:

```text Bind specialist models Route prompts Score outputs with utility functions Arbitrate
multi-model responses Manage correction loops Support model deployment workflows ```

### AUA Continuity

Audience:

```text All AI users developers students writers researchers product people ```

Purpose:

```text Capture durable corrections Store project instructions Preserve user preferences
Generate restart prompts Reduce repeated model mistakes Help resume long-running AI-assisted
work ```

---

## 6. Core Product Philosophy

The app should minimize user effort.

Bad approach:

```text Ask the user to manually paste transcripts. Ask the user to curate memory
constantly. Store every conversation. Dump all prior context into every prompt. ```

Good approach:

```text Monitor connected conversations. Keep only an ephemeral buffer. Detect durable
correction/instruction signals. Extract structured memory. Auto-save obvious high-confidence
memories. Ask only when ambiguous. Use only relevant memories in future prompts. ```

The app should follow this operating rule:

```text Observe broadly. Store narrowly. Reuse precisely. ```

---

## 7. Main Modes of Operation

Three modes were discussed.

### 7.1 Mode 3 — API Proxy / Local AI Gateway

This is the preferred long-term mode.

Architecture:

```text User / IDE / app / browser ↓ AUA Continuity local gateway ↓ OpenAI / Claude / Gemini
/ local model APIs ```

The user routes AI calls through AUA Continuity. The app sees prompts, model responses, and
user follow-ups. It can therefore detect corrections and persistent instructions accurately.

Advantages:

- Cleanest data capture. - Works naturally for developer tools and API-based workflows. -
Full control over request/response lifecycle. - Easier to implement reliable ephemeral
buffers. - Easier to avoid storing raw conversations.

Disadvantages:

- Requires users or tools to route traffic through the app. - Does not automatically cover
normal ChatGPT/Claude web UI unless integrated separately.

### 7.2 Mode 2 — Browser Extension

Fallback / second priority.

A browser extension observes ChatGPT, Claude, Gemini, or other AI web conversations and
sends relevant events to the local AUA Continuity app.

Advantages:

- Better for everyday users. - Works with common web-based AI tools. - Lower workflow change
for users.

Disadvantages:

- Browser extension complexity. - DOM changes on AI websites can break capture. -
Permission/privacy UX must be handled carefully. - Chrome Web Store / extension distribution
overhead.

### 7.3 Mode 1 — Manual Paste / Import

This is useful only for onboarding and backup import.

It should not be the main product workflow.

Manual paste is too high-friction for long-term use. Most users will not reliably do it.

Use cases where manual import is still useful:

- Importing old transcripts. - Debugging extraction behavior. - Initial testing before
extension/gateway support. - One-time migration of project context.

---

## 8. What the App Should Store

The app should not store full conversations by default.

It should store distilled memory fragments extracted from conversation events.

Memory-worthy categories include:

### 8.1 Persistent Instructions

Examples:

```text "Going forward, add comments to make Python code more readable." "Always continue
from where we left off." "Use concise but complete explanations." ```

### 8.2 Project-Level Decisions

Examples:

```text "AUA Framework and AUA Continuity are separate products." "For the production app,
use Postgres with JSONB." "Mode 3 API proxy is preferred over manual paste." ```

### 8.3 Corrections

Examples:

```text "No, that's incorrect. This app should use Postgres, not SQLite." "No, the framework
and the app are separate non-competing entities." "That's wrong. The app should monitor
conversations but not store all raw conversations." ```

### 8.4 Failure Patterns

Examples:

```text The model keeps collapsing the framework and app into one product. The model keeps
recommending manual transcript paste as the main workflow. The model keeps treating SQLite
as the target production database. ```

### 8.5 Preferences

Examples:

```text "Prefer industry-standard Python structure." "Add comments where logic is
non-obvious." "Use Postgres first unless I explicitly say prototype." ```

### 8.6 Current Tasks

These may be tracked temporarily but should usually not become durable memory.

Example:

```text "Rewrite the current Python files with comments." ```

This is a current task, not a long-term instruction.

If the same user message includes both a durable instruction and an immediate task, the app
should split them.

Example:

```text "Add comments to make the code more readable going forward and also rewrite current
Python files." ```

Extract:

```json [ { "type": "persistent_instruction", "scope": "project_or_user", "content": "When
generating or rewriting Python code, include useful comments that improve readability.",
"should_store": true }, { "type": "current_task", "scope": "session", "content": "Rewrite
the current Python files with readability comments.", "should_store": false } ] ```

---

## 9. What the App Should Not Store by Default

AUA Continuity should avoid storing:

- Entire raw conversations. - Every user message. - Every assistant response. - Sensitive
personal details unless explicitly saved by the user. - Low-value transient tasks. - Vague
instructions with no clear future value. - Temporary prompts like “rewrite this paragraph.”
- Repetitive model outputs that do not contain user correction.

The app can keep a short-lived local buffer for context extraction, but this should expire
quickly.

---

## 10. Ephemeral Buffer

To understand corrections like “No, that is incorrect,” the app needs recent context.

However, that context does not need to be persisted.

The app should maintain an ephemeral local buffer such as:

```text last N user messages last N assistant messages current task current files touched
current project recent model output ```

Suggested retention:

```text 5–30 minutes or until session closes or last 20–50 turns ```

Default behavior:

```text Persisted: extracted memory fragments only

Not persisted: full raw conversation ```

This gives the app enough context to understand corrections while preserving privacy and
reducing storage noise.

---

## 11. Trigger Detection

The app should monitor for phrases and semantic signals indicating durable memory.

High-signal phrases include:

```text no, that's incorrect that's wrong not what I asked you misunderstood we already
decided going forward from now on always never remember don't do that again use X instead
prefer X avoid X for this project for this repo next time whenever generating code ```

Keyword detection alone is not enough.

The app must also detect semantic corrections.

Example:

```text "We are not merging the framework and app. They are separate." ```

This should be detected as a correction even though it does not use the word “incorrect.”

---

## 12. Memory Capture Pipeline

The capture pipeline should be:

```text Conversation stream ↓ Ephemeral local buffer ↓ Trigger detector ↓ Correction/context
extractor ↓ Memory classifier ↓ Scope resolver ↓ Utility scorer ↓ Conflict detector ↓ Memory
candidate ↓ Auto-save or passive review ↓ Postgres durable memory ↓ Future prompt
reconstitution ```

Detailed stages:

### 12.1 Conversation Stream

Input from:

- API proxy. - Browser extension. - IDE plugin. - Manual import fallback.

### 12.2 Ephemeral Local Buffer

Short-lived memory used only for extraction.

### 12.3 Trigger Detector

Detects whether the latest user message likely contains:

- correction - persistent instruction - preference - decision - project constraint -
anti-pattern

### 12.4 Correction / Context Extractor

Looks backward into bounded recent context.

Inputs:

```text previous assistant response previous user instruction current user correction active
project memory ```

Outputs structured candidate memory.

### 12.5 Memory Classifier

Classifies candidate as:

```text correction instruction preference decision constraint failure_pattern current_task
```

### 12.6 Scope Resolver

Determines whether memory scope is:

```text global project repo file session ```

### 12.7 Utility Scorer

Computes whether memory is worth storing and later retrieving.

### 12.8 Conflict Detector

Detects if the new memory contradicts or narrows an older memory.

### 12.9 Persistence Policy

Decides:

```text auto-save ask/passive review ignore ```

### 12.10 Prompt Reconstitution

Builds compact restart and guardrail prompts from active relevant memory.

---

## 13. Utility Scoring

Utility scoring should be used for two core decisions:

1. Should this correction/instruction be stored? 2. Should this memory be included in a
future prompt?

This borrows the AUA utility concept, but applies it to user-facing memory capture and
prompt reconstitution.

### 13.1 Store Utility

Suggested formula:

```text store_utility = 0.30 * correction_strength + 0.25 * future_reuse_probability + 0.20
* project_relevance + 0.15 * user_explicitness + 0.10 * severity - 0.20 * ambiguity - 0.20 *
sensitivity_risk ```

Factors:

```text correction_strength: Did the user clearly say the model was wrong?

future_reuse_probability: Is this likely to matter again?

project_relevance: Does it map to the current project/product/codebase?

user_explicitness: Did the user say "going forward", "always", "remember", etc.?

severity: Would repeating this mistake materially hurt output quality?

ambiguity: Is the correction vague?

sensitivity_risk: Is this personal, private, or sensitive? ```

### 13.2 Include Utility

Suggested formula for future prompt inclusion:

```text include_utility = 0.30 * relevance_to_current_task + 0.25 * failure_prevention_value
+ 0.20 * importance + 0.10 * recency + 0.10 * confidence + 0.05 * pinned_boost - 0.20 *
staleness - 0.15 * token_cost ```

This prevents the prompt from becoming a memory dump.

---

## 14. Auto-Save vs User Review

The app should minimize friction.

It should not ask the user to approve every memory.

Suggested policy:

```text store_utility >= 0.85 and confidence >= 0.85: auto-save with undo/edit toast

0.60 <= store_utility < 0.85: show passive review card

store_utility < 0.60: do not save ```

Example UI:

```text Saved project memory: "Use Postgres with JSONB for the AUA Continuity app."

Undo | Edit ```

For ambiguous memories:

```text Save this project rule?

"Avoid global mutable runtime state in the FastAPI backend unless explicitly protected or
scoped."

Save | Edit | Ignore ```

The user should not become a database curator.

---

## 15. Example Capture Flow

Conversation:

```text User: Build the backend for the memory app.

AI: Use SQLite for local storage.

User: No, that's incorrect. This app should use Postgres with JSONB because memories need
structured fragments. ```

Trigger detected:

```text "No, that's incorrect" ```

Backward context:

```text Previous assistant proposed SQLite. User corrected to Postgres with JSONB. ```

Extracted memory:

```json { "memory_type": "correction", "scope": "project", "content": "For the AUA
Continuity app, use Postgres with JSONB as the primary memory store.", "avoid_instruction":
"Do not propose SQLite as the production database for this app.", "positive_instruction":
"Use Postgres with JSONB for structured memory fragments.", "confidence_score": 0.96,
"store_utility": 0.91 } ```

Future restart prompt should include:

```text Known decisions: - The AUA Continuity app uses Postgres with JSONB as the primary
memory store.

Avoid: - Do not propose SQLite as the production DB for this app unless explicitly
discussing a throwaway local prototype. ```

---

## 16. Conflict Handling

Memory evolves.

Example later correction:

```text "Actually, for the desktop-only version, SQLite is fine." ```

This conflicts with:

```text "Use Postgres with JSONB for the app." ```

The app should not blindly delete the old memory. It should narrow the scope.

Updated memory:

```text Use Postgres with JSONB for the production app. SQLite is acceptable only for
desktop-only local prototypes. ```

Memory revision record:

```json { "old": "Use Postgres with JSONB for the app.", "new": "Use Postgres with JSONB for
the production app. SQLite is acceptable only for desktop-only local prototypes.", "reason":
"User narrowed previous rule." } ```

---

## 17. Prompt Reconstitution

When generating a restart prompt, the app should pull memory in layers.

Suggested order:

```text 1. Global user instructions 2. Project instructions 3. Project decisions 4. Active
constraints 5. Known failure patterns 6. Recent corrections 7. Open tasks 8. Recent session
summary ```

Generated prompt structure:

```text You are continuing an existing project. Do not restart from scratch.

Project: <project name>

Current objective: <objective>

Persistent user instructions: - ...

Known project decisions: - ...

Architecture / domain context: - ...

Known mistakes to avoid: - ...

Corrections that must be treated as true: - ...

Current open tasks: - ...

When responding: - Continue from the current state. - Ask at most one clarifying question
only if blocked. - Prefer concrete edits over general advice. ```

Example compact memory prompt:

```text Before answering, respect these project memories:

1. AUA Framework and AUA Continuity are separate products. 2. AUA Continuity uses Postgres +
JSONB for production memory storage. 3. Do not store full raw conversations by default; only
persist extracted memory events. 4. Monitor continuously through API proxy or browser
extension; do not require manual paste/import. 5. When generating Python code, include
comments where they clarify non-obvious logic. ```

---

## 18. Data Store Recommendation

Use:

```text Postgres + JSONB ```

Reason:

- The product has relational entities: - users - projects - sessions - memory events -
memory fragments - revisions - prompt runs - It also needs flexible structured fields. -
JSONB gives document flexibility without giving up relational integrity. - Later semantic
retrieval can be added with `pgvector`.

MongoDB is possible, but Postgres + JSONB is the better default because the app has strong
relational structure and future audit/security needs.

SQLite is acceptable only for early local throwaway prototypes, not for the serious
production app.

---

## 19. Proposed Database Schema

### 19.1 `users`

```sql CREATE TABLE users ( user_id UUID PRIMARY KEY, email TEXT, display_name TEXT,
created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT
now(), metadata JSONB NOT NULL DEFAULT '{}' ); ```

### 19.2 `projects`

```sql CREATE TABLE projects ( project_id UUID PRIMARY KEY, user_id UUID NOT NULL REFERENCES
users(user_id), name TEXT NOT NULL, description TEXT, created_at TIMESTAMPTZ NOT NULL
DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), last_active_at TIMESTAMPTZ,
status TEXT NOT NULL DEFAULT 'active', metadata JSONB NOT NULL DEFAULT '{}' ); ```

### 19.3 `conversation_sources`

```sql CREATE TABLE conversation_sources ( source_id UUID PRIMARY KEY, user_id UUID NOT NULL
REFERENCES users(user_id), source_type TEXT NOT NULL, display_name TEXT, created_at
TIMESTAMPTZ NOT NULL DEFAULT now(), metadata JSONB NOT NULL DEFAULT '{}' ); ```

Source types:

```text api_proxy browser_extension ide_plugin manual_import ```

### 19.4 `conversations`

```sql CREATE TABLE conversations ( conversation_id UUID PRIMARY KEY, user_id UUID NOT NULL
REFERENCES users(user_id), project_id UUID REFERENCES projects(project_id), source_id UUID
REFERENCES conversation_sources(source_id), title TEXT, started_at TIMESTAMPTZ NOT NULL
DEFAULT now(), last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(), metadata JSONB NOT NULL
DEFAULT '{}' ); ```

Note: this table can store session metadata without storing full raw conversation contents.

### 19.5 `memory_events`

A durable event created only after a trigger.

```sql CREATE TABLE memory_events ( event_id UUID PRIMARY KEY, user_id UUID NOT NULL
REFERENCES users(user_id), project_id UUID REFERENCES projects(project_id), conversation_id
UUID REFERENCES conversations(conversation_id),

    source_type TEXT NOT NULL, event_type TEXT NOT NULL,

    trigger_text TEXT NOT NULL, extracted_bad_behavior TEXT, extracted_correction TEXT NOT
    NULL,

    scope TEXT NOT NULL, confidence_score DOUBLE PRECISION NOT NULL, store_utility DOUBLE
    PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), metadata JSONB NOT NULL DEFAULT '{}' );
    ```

Event types:

```text correction instruction preference decision constraint failure_pattern ```

### 19.6 `memory_fragments`

The active memories used for prompt generation.

```sql CREATE TABLE memory_fragments ( memory_id UUID PRIMARY KEY, user_id UUID NOT NULL
REFERENCES users(user_id), project_id UUID REFERENCES projects(project_id),

    memory_type TEXT NOT NULL, scope TEXT NOT NULL,

    content TEXT NOT NULL, avoid_instruction TEXT, positive_instruction TEXT,

    importance_score DOUBLE PRECISION NOT NULL DEFAULT 0.5, confidence_score DOUBLE
    PRECISION NOT NULL DEFAULT 0.5, include_utility DOUBLE PRECISION NOT NULL DEFAULT 0.5,

    active BOOLEAN NOT NULL DEFAULT true, pinned BOOLEAN NOT NULL DEFAULT false,

    source_event_id UUID REFERENCES memory_events(event_id), created_at TIMESTAMPTZ NOT NULL
    DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), expires_at TIMESTAMPTZ,

    metadata JSONB NOT NULL DEFAULT '{}' ); ```

### 19.7 `memory_revisions`

```sql CREATE TABLE memory_revisions ( revision_id UUID PRIMARY KEY, memory_id UUID NOT NULL
REFERENCES memory_fragments(memory_id), old_content TEXT, new_content TEXT NOT NULL,
revision_reason TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), metadata JSONB NOT NULL
DEFAULT '{}' ); ```

### 19.8 `prompt_runs`

Records generated restart/guardrail prompts.

```sql CREATE TABLE prompt_runs ( prompt_run_id UUID PRIMARY KEY, user_id UUID NOT NULL
REFERENCES users(user_id), project_id UUID REFERENCES projects(project_id),

    prompt_type TEXT NOT NULL, generated_prompt TEXT NOT NULL, memory_ids UUID[] NOT NULL
    DEFAULT '{}',

    token_estimate INTEGER, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), metadata JSONB
    NOT NULL DEFAULT '{}' ); ```

Prompt types:

```text restart guardrail handoff debugging coding_session ```

### 19.9 `memory_conflicts`

```sql CREATE TABLE memory_conflicts ( conflict_id UUID PRIMARY KEY, user_id UUID NOT NULL
REFERENCES users(user_id), project_id UUID REFERENCES projects(project_id),

    old_memory_id UUID NOT NULL REFERENCES memory_fragments(memory_id), new_event_id UUID
    NOT NULL REFERENCES memory_events(event_id),

    conflict_type TEXT NOT NULL, resolution_status TEXT NOT NULL DEFAULT 'unresolved',

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), resolved_at TIMESTAMPTZ, metadata JSONB
    NOT NULL DEFAULT '{}' ); ```

---

## 20. Application Architecture

Suggested high-level structure:

```text apps/ aua_continuity/ backend/ app/ api/ conversations.py memory.py restart.py
proxy.py projects.py core/ config.py scoring.py prompt_builder.py normalizer.py services/
conversation_gateway.py conversation_ingestor.py trigger_detector.py memory_extractor.py
memory_classifier.py scope_resolver.py utility_scorer.py conflict_detector.py
prompt_reconstitutor.py audit_service.py db/ models.py repositories.py migrations/ frontend/
components/ ConversationMonitor.tsx MemoryInbox.tsx MemoryReviewPanel.tsx
RestartPromptBuilder.tsx ProjectDashboard.tsx ```

---

## 21. Backend Services

### 21.1 Conversation Gateway

Responsible for Mode 3 API proxy.

Duties:

- Accept user/model requests. - Forward requests to configured AI provider. - Stream
responses back to caller. - Feed turns into ephemeral buffer. - Trigger memory detection
after user corrections.

### 21.2 Trigger Detector

Determines whether a user message contains memory-worthy signals.

Inputs:

```text latest user message recent assistant response project context ```

Output:

```json { "triggered": true, "trigger_type": "correction", "confidence": 0.93 } ```

### 21.3 Memory Extractor

Looks backward into bounded context and extracts structured memory.

Output example:

```json { "event_type": "correction", "scope": "project", "bad_behavior": "The assistant
proposed SQLite as the production store.", "corrective_instruction": "Use Postgres with
JSONB as the production memory store.", "confidence": 0.96 } ```

### 21.4 Scope Resolver

Determines if memory is:

```text global project repo file session ```

### 21.5 Utility Scorer

Computes:

```text store_utility include_utility ```

### 21.6 Conflict Detector

Checks new memories against active memories.

Examples:

- contradiction - scope narrowing - preference update - duplicate memory - stale memory

### 21.7 Prompt Reconstitutor

Builds restart and guardrail prompts from active relevant memory.

---

## 22. Frontend UX

The frontend should not require users to manually curate everything.

Core screens:

### 22.1 Project Dashboard

Shows:

- active projects - last active time - number of captured memories - restart prompt button -
recent corrections - known failure patterns

### 22.2 Memory Inbox

Shows recently captured memory candidates.

Possible states:

```text auto-saved needs review ignored conflict detected ```

Actions:

```text Keep Edit Reject Pin Archive Change scope ```

### 22.3 Passive Save Toast

For obvious high-confidence captures:

```text Saved project memory: "Use Postgres with JSONB for the AUA Continuity app."

Undo | Edit ```

### 22.4 Restart Prompt Builder

Shows generated prompt with editable sections:

```text Project context Persistent instructions Decisions Mistakes to avoid Open tasks ```

Actions:

```text Copy Send through gateway Regenerate Edit Save snapshot ```

### 22.5 Memory Review Panel

For ambiguous captures:

```text Save this project rule?

"Avoid global mutable runtime state in the FastAPI backend unless explicitly protected or
scoped."

Save | Edit | Ignore ```

---

## 23. Privacy and Storage Policy

AUA Continuity should be local-first.

Default guarantees:

```text Full raw conversations are not persisted by default. Only extracted memory
events/fragments are stored. Ephemeral buffers expire. All stored memories are visible to
the user. Users can edit, delete, archive, export, and import memory. No cloud sync unless
explicitly enabled. ```

Important product language:

```text AUA Continuity watches conversations you explicitly connect or route through the
app. ```

Avoid saying:

```text AUA Continuity silently monitors everything you do. ```

The latter creates privacy concerns.

---

## 24. API Surface

Initial backend API endpoints:

```text POST /proxy/chat POST /projects GET  /projects GET  /projects/{project_id} POST
/memory/events GET  /memory/fragments PATCH /memory/fragments/{memory_id} POST
/restart/generate GET  /prompt-runs POST /import/transcript GET  /health ```

Later:

```text POST /browser-extension/events POST /ide/events POST
/memory/conflicts/{conflict_id}/resolve POST /sync/export POST /sync/import ```

---

## 25. MVP Requirements

The first serious MVP should include:

```text 1. Local backend with FastAPI. 2. Postgres + JSONB persistence. 3. Project creation
and selection. 4. Mode 3 API proxy for at least one provider. 5. Ephemeral conversation
buffer. 6. Trigger detector for correction/instruction phrases. 7. Memory extractor. 8.
Scope resolver. 9. Utility scorer. 10. Durable memory event storage. 11. Active memory
fragment storage. 12. Restart prompt generator. 13. Simple frontend dashboard. 14. Memory
inbox/review panel. 15. Copy generated prompt to clipboard. ```

Mode 1 manual transcript import can exist, but should not be treated as the primary
workflow.

---

## 26. Suggested Build Phases

### Phase 1 — Local Gateway MVP

Goal:

```text Prove Mode 3 works. ```

Build:

```text FastAPI backend Postgres schema local API proxy for OpenAI-compatible provider
ephemeral buffer trigger detector memory extractor memory fragment persistence restart
prompt builder basic UI ```

### Phase 2 — Better Memory Hygiene

Build:

```text conflict detection memory revisions scope narrowing include utility scoring
pinned/archived memory prompt snapshots ```

### Phase 3 — IDE Integration

Build:

```text VS Code / Cursor extension repo-level memory file-level memory coding-specific
correction capture ```

### Phase 4 — Browser Extension

Build:

```text ChatGPT / Claude web capture local-only extension bridge trigger-only persistence
memory review panel ```

### Phase 5 — Advanced Retrieval

Build:

```text pgvector semantic retrieval memory clustering staleness detection summary
compression multi-project memory isolation ```

---

## 27. Non-Goals for Initial Version

Do not build these first:

```text Full raw conversation archive Cloud sync Team/shared memory Enterprise admin console
Multi-tenant SaaS Full browser extension before gateway MVP Automatic capture of every
website Full semantic vector memory from day one Complex symbolic validators DPO export
pipeline ```

These can come later.

---

## 28. Product Risks

### 28.1 Memory Noise

If the app saves too much, prompt quality degrades.

Mitigation:

```text store narrowly score utility ask only when ambiguous archive stale memory
deduplicate aggressively ```

### 28.2 Privacy Concerns

Continuous monitoring can sound invasive.

Mitigation:

```text local-first explicit connection no raw conversation persistence by default visible
memory inbox delete/export controls ```

### 28.3 Bad Extraction

The app may misunderstand a correction.

Mitigation:

```text confidence scoring passive review undo/edit source snippets revision history ```

### 28.4 Prompt Bloat

Too many memories can overwhelm the model.

Mitigation:

```text include utility scoring token budget project-specific retrieval staleness penalty
summary compression ```

### 28.5 Workflow Adoption

Users may not want to change how they use AI tools.

Mitigation:

```text start with API proxy for developer/power users then add browser extension then add
IDE integration manual import only as fallback ```

---

## 29. Success Criteria

The product is working if users can say:

```text I returned to a project after a week and AUA Continuity gave me the right restart
prompt. The AI stopped repeating a mistake I had corrected before. I did not have to
manually paste and organize every conversation. I can see and edit what the app remembered.
The app helps me continue from where I left off. ```

Measurable signals:

```text number of useful memories captured percentage of auto-saved memories not undone
restart prompt usage rate user edits per generated prompt repeat correction reduction time
saved during project restart ```

---

## 30. Open Design Questions

These still need to be decided:

1. Should the first gateway target OpenAI-compatible APIs only, or include Claude
immediately? 2. Should the app be desktop-first, web-localhost-first, or
browser-extension-first? 3. How much raw context should the ephemeral buffer retain? 4.
Should high-confidence memories auto-save by default, or should users opt into auto-save? 5.
Should project detection be manual first or inferred from working directory/browser tab? 6.
Should memory extraction use a local model, hosted model, or rule-based + model hybrid? 7.
How should sensitive data detection work locally? 8. Should Postgres run via Docker for
local users? 9. Should there be an encrypted local mode? 10. How should restart prompts be
sent back to ChatGPT/Claude/Cursor?

---

## 31. Recommended First Implementation Slice

Build this first:

```text AUA Continuity Local Gateway MVP ```

Must include:

```text FastAPI backend Postgres via Docker Compose one provider proxy project selection
ephemeral buffer trigger phrase detector LLM-based memory extractor store_utility scoring
memory_events table memory_fragments table restart prompt generator minimal React UI
copy-to-clipboard ```

Do not build browser extension first unless Mode 3 becomes blocked.

---

## 32. One-Sentence Product Description

AUA Continuity is a local-first AI memory agent that watches connected AI sessions, stores
only durable corrections and instructions, and generates compact restart prompts so users
can continue projects without re-explaining context or repeating the same model mistakes.

---

## 33. Short Product Tagline Options

```text Continue where you left off. Memory for your AI work. Stop correcting the same AI
mistake twice. A local-first continuity layer for ChatGPT, Claude, and Cursor. Project
memory and correction capture for frontier AI tools. ```

Preferred:

```text Continue where you left off. ```

Secondary:

```text Stop correcting the same AI mistake twice. ```
