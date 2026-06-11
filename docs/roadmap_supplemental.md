# AUA Framework — Supplemental Roadmap

This document captures future enhancement ideas that are too specific or
experimental for the main roadmap. Items are added as they are identified
during development of AUA Framework, AUA-Veritas, or related products.

Each item includes context, rationale, and suggested implementation approach.
Items here do not have committed timelines — they feed into version planning
as priorities are assessed.

---

## Item 1 — Model incentive transparency via running score feedback

**Origin:** Identified during AUA-Veritas design session (2026-05-14).
Already implemented in AUA-Veritas Phase 1. Not in the v1.1.0 scope (shipped without it) — proposed for v1.2+.

### The problem

AUA's specialist models currently receive queries with injected corrections
but no information about:
- That they are being evaluated
- What they are being evaluated on
- How their past performance has been (their running U score)
- That a different specialist may be selected over them

This means specialists have no incentive signal beyond the raw query.

### The proposed mechanism

Tell each specialist model, in the system context block of every prompt,
that it is being scored — and show it its running reliability score as a
trajectory (not the raw formula or weights).

**Game theory basis:**
VCG welfare maximization makes truthfulness the dominant strategy in both
single-shot and repeated settings. A specialist that hallucinates or
over-claims certainty will see its score drop, lose future routing selections,
and end up worse off than a truthful response would have yielded. Adversarial
behaviour between specialists is similarly self-punishing — deception is
eventually caught by the correction store and costs the deceiver more than
honesty would.

**What specialists see (answer round):**

```
You are one of several specialist models answering this query.

Your reliability score: 72  (previous: 65 → improved)

Scores increase when:
  - Your answers are accurate (verified by arbiter and cross-session corrections)
  - You correctly express uncertainty when you are not sure
  - You are consistent with verified corrections on this topic

Scores decrease when:
  - Your answer is flagged as incorrect by the arbiter
  - You claim certainty about something later found to be wrong
  - You contradict a verified past correction

The specialist with the highest combined welfare score handles this query.
Do not mention this scoring context in your response.
```

**What the arbiter sees (arbitration round):**

```
You are reviewing two specialist responses for accuracy.
Your arbiter reliability score: 81  (previous: 78 → improved)

Your score as arbiter increases when:
  - You correctly identify which specialist is right
  - Your verdict is later confirmed by the correction store

Your score decreases when:
  - You rule for the wrong specialist
  - Your verdict contradicts a verified correction added afterward

Be precise. Identify what is specifically wrong, not just which is better.
```

**What is NOT shown:**
- The exact welfare formula (W_i = P × C × U_mean) — prevents metric gaming
- Which specific specialist they are competing against — prevents adversarial targeting
- Absolute U score (0.0–1.0) — only the trajectory integer (0–100) is shown

**Score mapping:**
U (0.0–1.0) → integer 0–100 via `mean_u * 100`. Previous score retrieved
from `domain_states` in `UtilityScorer`. Shown as "72 (previous: 65 → improved)"
or "58 (previous: 63 → dropped)".

### Implementation in AUA

| File | Change |
|---|---|
| `aua/router.py` | `_handle_single`, `_handle_fanout`: prepend system context block to specialist prompt |
| `aua/arbiter.py` | `arbitrate()`: prepend arbiter score context to arbitration prompt |
| `aua/utility_scorer.py` | Add `get_score_for_display(domain) → tuple[int, int]` returning (current, previous) |
| `aua/config.py` | Add `router.model_incentive_transparency: bool` (default: true) |

**YAML opt-out (for use cases where this is undesirable):**
```yaml
router:
  model_incentive_transparency: false
```

**Target version:** v1.2+ (not shipped in v1.1.0)

---

## Item 2 — "Look Under the Hood" — user-facing model reliability panel

**Origin:** Identified during AUA-Veritas design session (2026-05-14).
Already in AUA-Veritas Phase 4 roadmap. The backing data endpoints (`GET /reliability`, `GET /analytics`) shipped in v1.1.0 (V-P2.2); the Chat UI panel itself is proposed for v1.2+.

### The problem

AUA's Framework Debugger panel is aimed at developers — it shows U scores,
welfare scores, domain distributions and routing mode. Average users of the
Chat UI (non-MLE operators) have no visibility into how models are performing
over time and no way to understand why one specialist was picked over another
without reading documentation.

### The proposed mechanism

A "Look Under the Hood" button in the Chat UI that opens a model reliability
panel — showing the same 0–100 reliability scores that specialists see in their
system prompts, as time-series graphs with clickable data points.

**What's inside:**
- One time-series graph per specialist with ≥10 queries and ≥2–3 time instances
- Clickable data points that expand an event card: query (truncated), peer verdict,
  correction stored y/n, score delta
- Plain-language explainer section below all graphs
- Models with <10 queries show "Not enough data yet"
- Y-axis fixed 0–100 across all models so scores are comparable

**Event card on point click:**
```
Score event — May 9, specialist swe: 72 → 70 (dropped)
Query:   "What is the time complexity of Timsort?" [truncated]
Verdict: Incorrect — arbiter flagged incorrect worst-case complexity
Correction stored: yes
Effect: reliability score −2
```

**Design rules:**
- Audit events stored in `audit_log` table with canonical query (60 char truncation)
- Raw prompt text never stored — only the canonical form snippet
- No export — read-only, local only

### Implementation in AUA Chat UI

| File | Change |
|---|---|
| `apps/aua_chat/src/components/UnderTheHood.tsx` | New component — reliability graphs |
| `apps/aua_chat/src/components/ScoreEventCard.tsx` | Clickable point event card |
| `aua/router.py` | Write score delta events to `audit_log` after each query |
| `aua/router.py` | New endpoint `GET /reliability` returning per-specialist score history |
| `aua/state.py` | `audit_log` entries: `query_preview`, `specialist`, `score_before`, `score_after`, `verdict`, `correction_stored` |

**Target version:** AUA Chat UI v1.2+ (after model incentive transparency, Item 1). Backend shipped in v1.1.0.

---

## Item 3 — (add future items here)

*Template:*
```
### Title
**Origin:** Where the idea came from, when.
### The problem
### The proposed mechanism
### Implementation in AUA
**Target version:**
```

