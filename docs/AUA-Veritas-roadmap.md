# AUA-Veritas — Product Roadmap

**Status:** Design complete, pre-build
**Date:** 2026-05-14
**Repo:** `AUA-Veritas` (separate GitHub repo, GPL 3.0)
**Relationship to AUA:** Separate consumer product. Copies and adapts from AUA framework
but never modifies AUA source. Model plugins developed here are contributed back to the
AUA repo as prebuilt plugins once complete.

---

## What it is

AUA-Veritas is a standalone AI assistant desktop app for non-technical users. The user
picks which frontier AI models they have API keys for, and the app silently runs AUA's
control logic underneath: routing, VCG welfare maximization, peer-review contradiction
detection, and correction memory that builds up over time.

The user never sees terms like "VCG", "U score", or "arbiter". They see a familiar chat
interface, a plain-language confidence level, and amber callouts when the system does
something interesting (corrected an answer, cross-checked with another model, applied a
past correction).

**Core value proposition:** A more truthful answer than any single model gives directly.
The higher the accuracy setting, the more models check each other's work before an answer
reaches the user.

---

## All design decisions (locked)

| Decision | Choice | Reason |
|---|---|---|
| Distribution | Electron desktop app | Zero CLI, installs like Chrome, true native window |
| Data storage | SQLite, local only | Privacy-first, no login, no data leaves machine |
| API key storage | OS keychain (`keyring` library) | Secure, never in plaintext |
| Correction sharing | None in v1 — local only | Trust and simplicity; opt-in sharing in v2 |
| License | GPL 3.0 | Consistent with AUA framework |
| Pricing | Free, user brings own API keys | Maximum adoption |
| First OS | macOS, then Windows before launch | |
| Starting models | All 10 from day 1 | 8/10 are OpenAI-compatible, low marginal effort |
| Mode picker | Single accuracy slider (4 levels) | Simpler mental model than named modes |
| Peer review judge | Cheapest available model (GPT-4o mini or Haiku) | 60-70% cost reduction, same accuracy |
| Phase 2 scope | All 4 accuracy levels ship together | App makes no sense without them |

---

## The 10 model plugins

All plugins implemented as `ModelBackendPlugin` (copied from `aua/plugins/interfaces.py`).
Completed plugins contributed back to AUA repo as prebuilt plugins, documented in AUA tutorial §13.

| # | Model | Provider | API type | Role |
|---|---|---|---|---|
| 1 | GPT-4o | OpenAI | OpenAI SDK | Primary |
| 2 | GPT-4o mini | OpenAI | OpenAI SDK | Peer review judge (cheap) |
| 3 | Claude Sonnet 4.5 | Anthropic | Anthropic SDK | Primary |
| 4 | Claude Haiku 4.5 | Anthropic | Anthropic SDK | Peer review judge (cheap) |
| 5 | Gemini 1.5 Pro | Google | Google GenAI SDK | Primary |
| 6 | Gemini 2.0 Flash | Google | Google GenAI SDK | Fast/cheap option |
| 7 | Grok-2 | xAI | OpenAI-compatible | Differentiated on current events |
| 8 | Mistral Large | Mistral | OpenAI-compatible | European data residency |
| 9 | Llama 3.3 70B (via Groq) | Groq | OpenAI-compatible | Fast open-weights option |
| 10 | DeepSeek-V3 | DeepSeek | OpenAI-compatible | Strong at code, lowest cost |

**Implementation breakdown:** 3 distinct SDK implementations (OpenAI, Anthropic, Google GenAI).
The remaining 6 are OpenAI-compatible — same plugin pattern, different base URL and API key.

---

## Accuracy levels

Single slider in the left panel. Four positions. Plain-language labels only — no technical
terms visible to the user.

### Fast
- Route to one model based on query type (code → GPT-4o or Claude, factual → Gemini, etc.)
- Inject correction store into prompt before calling
- Deterministic validation (AST check for code, consistency check against corrections)
- **API calls:** 1
- **Cost:** Lowest

### Balanced
- Two models answer in parallel
- Deterministic validation on both responses
- VCG welfare maximization selects the better response
- **API calls:** 2
- **Cost:** Low

### High
- All user-selected models answer in parallel
- VCG selects the best response by W_i = P(domain) × confidence × prior_mean_U
- Arbiter stores contradictions as corrections for future queries
- **API calls:** N (number of connected models)
- **Cost:** Medium

### Maximum
- **Round 1:** All selected models answer in parallel (same as High)
- VCG selects provisional winner
- **Round 2 — Peer review:** All other models review the winner's response
  - Structured prompt: "Here is a question and an answer. Is this correct? If not, what is wrong?"
  - Uses cheapest available model (GPT-4o mini or Claude Haiku) as reviewer — not primary model
  - Cost saving: 60-70% vs using primary models for review
- If reviewers agree with winner → confirmed, high confidence
- If reviewers disagree → show user which model disagreed and why; store as correction
- **API calls:** N + (N-1)×2, e.g. 3 models = 7 calls
- **Cost:** Higher — clearly labeled with estimated call count and cost before user enables

**Live cost estimate:** Left panel shows estimated calls and USD cost per message, updating
as user changes accuracy level and checks/unchecks models.
Example: *"~7 API calls · est. $0.04 per message"*

**High-stakes domains (medical, legal, finance):** At all accuracy levels, queries
classified as high-stakes always get an amber callout: *"This topic requires professional
advice. Please verify with a qualified expert."* Automated retry is skipped on high-stakes
failures — same domain-gated policy as AUA v1.0.

---

## UI layout

Three-panel layout. No technical jargon visible anywhere.

```
┌──────────────────────────────────────────────────────────────────────┐
│  AUA-Veritas                                              [− □ ×]   │
├──────────────────┬─────────────────────────────┬────────────────────┤
│                  │                             │                    │
│  Past chats      │  ╭─────────────────────╮   │  Response quality  │
│  ──────────────  │  │ User message        │   │  ────────────────  │
│  Today           │  ╰─────────────────────╯   │                    │
│  · Chat 1        │                             │  Confidence        │
│  · Chat 2        │  AI response card           │  ● High            │
│  ──────────────  │  (white/light card)         │                    │
│  Yesterday       │                             │  Models            │
│  · Chat 3        │  ┌─────────────────────┐   │  · GPT-4o ✓        │
│                  │  │ Amber callout       │   │  · Claude ✓        │
│                  │  │ "I corrected my     │   │                    │
│                  │  │  first answer"      │   │  What happened     │
│                  │  └─────────────────────┘   │  Both models       │
│                  │                             │  agreed ✓          │
│  ──────────────  │  ╭─────────────────────╮   │                    │
│  Models          │  │ User message        │   │                    │
│  ☑ ChatGPT       │  ╰─────────────────────╯   │                    │
│  ☑ Claude        │                             │                    │
│  ☑ Gemini        │  AI response card           │                    │
│  ☐ Grok          │                             │                    │
│  ☐ DeepSeek      ├─────────────────────────────┤                    │
│                  │  [Ask anything...]  [Send]  │                    │
│  ──────────────  │                             │                    │
│  Accuracy        │                             │                    │
│  Fast ○──●───○   │                             │                    │
│     Balanced     │                             │                    │
│  ~2 calls·$0.01  │                             │                    │
└──────────────────┴─────────────────────────────┴────────────────────┘
```

### Three message colors (centre panel)

| Message type | Visual treatment |
|---|---|
| User message | Dark rounded box, right-aligned (same as Claude.ai) |
| AI response | White/light card, left-aligned, model name small above |
| System callout | Amber/warm card — used when the system does something notable |

### Callout examples (amber cards)

- *"I corrected my first answer before sending it to you."*
- *"I cross-checked this with a second model. Both agreed."*
- *"Models disagreed. I used the more reliable answer and explained why."*
- *"Applied a past correction — I've seen a similar question before."*
- *"Medical topic — please verify with a qualified professional."*

### Right panel (plain language)

- **Confidence:** High / Medium / Uncertain (single indicator, not a number)
- **Models:** Which ones answered this query (checkmarks)
- **What happened:** One sentence, plain English, no technical terms

---

## What gets copied from AUA (never modifying originals)

Copy to new repo → modify the copy.

| AUA source | Veritas copy | Changes |
|---|---|---|
| `aua/router.py` | `core/router.py` | Remove vLLM/Ollama plumbing, wire frontier backends |
| `aua/arbiter.py` | `core/arbiter.py` | Keep 4-check logic, add peer-review round |
| `aua/assertions_store.py` | `core/memory.py` | Per-user scoping from day 1 (not global) |
| `aua/contradiction_detector.py` | `core/validator.py` | Unchanged — already model-agnostic |
| `aua/guard.py` + `aua/policy.py` | `core/policy.py` | Simplified — user never configures assertions |
| `aua/state.py` | `core/state.py` | Add `user_context_grammar`, remove blue-green tables |
| `aua/plugins/interfaces.py` | `core/interfaces.py` | `ModelBackendPlugin` protocol unchanged |

### Not copied (not needed)

- vLLM/Ollama serving code — only frontier APIs in Veritas
- `aua/cli.py` — replaced by Electron startup, no terminal
- Blue-green deployment — irrelevant for single-user app
- Prometheus metrics — internal telemetry, user doesn't need
- mTLS / token auth — single user, local, no need

---

## New components (not in AUA)

| Component | Description |
|---|---|
| 10 frontier model plugins | `OpenAIBackend`, `AnthropicBackend`, `GoogleGenAIBackend`, + 7 OpenAI-compatible variants |
| OS keychain integration | `keyring` library — API keys stored securely in Mac Keychain / Windows Credential Store |
| Electron shell | Python process lifecycle management, app window, system tray daemon |
| Consumer UI | React, completely new design — 3-panel, 3-color, accuracy slider, no technical debugger |
| First-run setup | API key entry per model, test connection, onboarding walkthrough |
| Peer review orchestrator | Round 2 logic — routes winner's response to cheap judge model |
| Live cost estimator | Calculates and displays estimated calls and USD per message |
| `user_context_grammar` | Personalization layer from AUA v0.6 spec — learns preferences across chats |

---

## Phase roadmap

### Phase 0 — Foundation (2 weeks)
**Goal:** Repo set up, backend running headless, 2 models working, API keys stored securely.

- Create `AUA-Veritas` GitHub repo (GPL 3.0)
- Copy and simplify AUA backend modules (router, arbiter, memory, validator, state)
- SQLite schema: `users`, `conversations`, `messages`, `corrections`, `user_context_grammar`, `audit_log`
- OS keychain integration for API key storage
- `OpenAIBackend` plugin (GPT-4o + GPT-4o mini)
- `AnthropicBackend` plugin (Claude Sonnet + Haiku)
- Fast and Balanced accuracy levels working
- FastAPI backend starts headless (no terminal — called by Electron)
- Basic Electron shell: opens window, starts Python server on launch

**Deliverable:** App opens, user can chat with GPT-4o or Claude, basic routing works.

---

### Phase 1 — All 10 model plugins (1 week)
**Goal:** All supported models callable. Plugins contributed back to AUA simultaneously.

- `GoogleGenAIBackend` (Gemini 1.5 Pro + 2.0 Flash)
- `XAIBackend` (Grok-2 — OpenAI-compatible)
- `MistralBackend` (OpenAI-compatible)
- `GroqBackend` (Llama 3.3 70B — OpenAI-compatible)
- `DeepSeekBackend` (OpenAI-compatible)
- All plugins tested end-to-end
- All 10 contributed to AUA repo with tutorial update

**Deliverable:** All 10 models selectable and working in the app.

---

### Phase 2 — Consumer UI (3–4 weeks)
**Goal:** Full UI working inside Electron. Looks and feels like a real product.

- Full 3-panel layout
- Left panel: conversation list, model checkboxes, accuracy slider with live cost estimate
- Centre panel: 3-color message system (user box, AI response, amber callouts)
- Right panel: confidence indicator, models used, plain-language "what happened"
- All 4 accuracy levels: Fast, Balanced, High, Maximum (including peer review round)
- First-run setup screen: API key entry, test connection button, model picker
- Settings page: manage API keys, add/remove models
- High-stakes domain callout (medical, legal, finance)

**Deliverable:** Full working app. All features functional. Ready for internal testing.

---

### Phase 3 — Packaging and distribution (2 weeks)
**Goal:** One-click install on Mac and Windows. No terminal required.

- macOS: `.dmg` installer — PyInstaller bundles Python, Electron packages UI, code-signed
- Windows: `.exe` NSIS installer — same approach
- Linux: `.AppImage` (best-effort, not required for launch)
- System tray / menu bar icon — app runs in background, click to open
- Startup-on-login option
- Auto-update via Electron's built-in updater

**Deliverable:** Download link → double-click installer → app appears in dock/taskbar.
No terminal. No commands. Works like Chrome.

---

### Phase 4 — Polish for launch (2 weeks)
**Goal:** Smooth enough for a non-technical friend to use without help.

- Onboarding walkthrough (explains what the 4 accuracy levels do in plain language)
- Error states: graceful handling ("ChatGPT is unavailable, using Claude instead")
- Conversation search
- Keyboard shortcuts (Cmd+K new chat, Enter to send)
- Expandable callout explanations ("Why did I get this note?")
- Usage/cost page: queries per model, estimated monthly spend
- Basic correction transparency: user can view what the app has learned about a topic

**Deliverable:** v1.0 of AUA-Veritas. Ready for real users.

---

### Phase 5 — Post-launch (ongoing)
- More model plugins as new frontier models release
- Opt-in cross-user correction sharing (v2 feature, detailed in AUA v0.6 spec)
- Local model support via Ollama (fully offline option)
- Context grammar personalization (learns preferred style, known domains, project context)
- DPO pair export for power users
- Mobile consideration (much later, different scope)

---

## Timeline summary

| Phase | Duration | End state |
|---|---|---|
| 0 — Foundation | 2 weeks | 2 models working, Electron opens, local storage |
| 1 — All 10 plugins | 1 week | All models callable, contributed to AUA |
| 2 — Consumer UI | 3–4 weeks | Full app, all 4 accuracy levels, all features |
| 3 — Packaging | 2 weeks | Mac .dmg and Windows .exe installers |
| 4 — Polish | 2 weeks | Launch-ready |
| **Total** | **~10–11 weeks** | v1.0 AUA-Veritas |

---

## Repo structure (planned)

```
AUA-Veritas/
├── core/               # Python backend (copied + modified from AUA)
│   ├── router.py
│   ├── arbiter.py
│   ├── memory.py
│   ├── validator.py
│   ├── state.py
│   ├── policy.py
│   ├── interfaces.py
│   └── plugins/
│       ├── openai_backend.py
│       ├── anthropic_backend.py
│       ├── google_backend.py
│       ├── xai_backend.py
│       ├── mistral_backend.py
│       ├── groq_backend.py
│       └── deepseek_backend.py
├── api/                # FastAPI layer (headless, called by Electron)
│   └── main.py
├── electron/           # Electron shell
│   ├── main.js         # Process management, window creation
│   ├── preload.js
│   └── tray.js         # System tray / menu bar icon
├── ui/                 # React consumer UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.tsx        # Conversations + models + slider
│   │   │   ├── ChatPanel.tsx      # 3-color messages
│   │   │   ├── QualityPanel.tsx   # Right panel (plain language)
│   │   │   ├── AccuracySlider.tsx # Fast → Maximum with cost estimate
│   │   │   ├── ModelPicker.tsx    # Checkboxes + API key status
│   │   │   └── Callout.tsx        # Amber system callout card
│   │   └── ...
├── db/                 # SQLite schema and migrations
│   └── schema.sql
├── docs/
│   └── ...
├── tests/
├── LICENSE             # GPL 3.0
└── README.md
```

---

## What the user sees on first launch

```
Welcome to AUA-Veritas

Connect your AI models. You'll need API keys from each service.
Your keys are stored securely on this device and never shared.

  ChatGPT (OpenAI)     [sk-proj-•••••••••]  ✓ Connected
  Claude (Anthropic)   [sk-ant-•••••••••]  ✓ Connected
  Gemini (Google)      [________________________]  + Connect
  Grok (xAI)           [________________________]  + Connect

  [Start chatting →]

Need help getting API keys? → How to get your API keys
```

Then a chat window opens. That's it.

---

## Connection to AUA

AUA-Veritas is the consumer face of the AUA framework. AUA solves the engineering problem
(how do you build a utility-governed multi-model system). AUA-Veritas solves the adoption
problem (how do you put that in the hands of people who will never read a whitepaper).

The two projects reinforce each other:
- Plugins developed for Veritas feed back into the AUA plugin library
- Veritas demonstrates AUA's value proposition in a concrete, downloadable product
- AUA's whitepaper results (+43.3pp correctness) are the evidence that Veritas actually works
- Veritas users who want to go deeper have a path to the full AUA framework

