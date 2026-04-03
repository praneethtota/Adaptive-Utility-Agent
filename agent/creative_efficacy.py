"""
Creative efficacy tracker: measures AI creative output performance
against human baselines using existing platform engagement signals.

Two-component model:
    Creative_Efficacy = Content_Efficacy × Discoverability_Efficacy

Content_Efficacy      — conversion rate (engagement given views)
                        "can the work hold attention when shown?"

Discoverability_Efficacy — impressions, search ranking, reach
                        "can it find an audience at all?"

Marketing and discoverability are NOT noise to control for.
They are part of the creative skill, just as they are for human creators.

Platform signal weights by intent strength:
    purchase / download    1.0   (strongest — real economic behavior)
    save / bookmark        0.8
    share / repost         0.7
    like / upvote          0.5
    comment                0.4
    view / listen          0.1   (weakest — could be accidental)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


# Minimum observations before a score is considered meaningful
MIN_OBSERVATIONS = 50

# Signal weights by intent strength
SIGNAL_WEIGHTS = {
    "purchase":    1.0,
    "download":    1.0,
    "save":        0.8,
    "bookmark":    0.8,
    "share":       0.7,
    "repost":      0.7,
    "like":        0.5,
    "upvote":      0.5,
    "comment":     0.4,
    "view":        0.1,
    "listen":      0.1,
    "impression":  0.05,
}

# Platform → recommended signals to collect
PLATFORM_SIGNALS = {
    "soundcloud":   ["listens", "likes", "reposts", "downloads"],
    "spotify":      ["streams", "saves", "playlist_adds"],
    "pinterest":    ["saves", "clicks", "reposts"],
    "istockphoto":  ["downloads", "purchases"],
    "unsplash":     ["downloads", "likes"],
    "medium":       ["reads", "claps", "saves"],
    "youtube":      ["views", "likes", "shares", "saves"],
    "behance":      ["views", "appreciations", "saves"],
}


@dataclass
class EngagementSignals:
    """Raw engagement signals collected from a platform."""
    platform: str
    category: str                      # e.g. "ambient_electronic", "landscape_photography"
    signals: Dict[str, int]            # e.g. {"views": 1200, "likes": 80, "saves": 30}
    observation_window_days: int = 30
    collected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def total_observations(self) -> int:
        """Views or listens are the base observation count."""
        return self.signals.get("view", 0) + self.signals.get("listen", 0) + \
               self.signals.get("stream", 0) + self.signals.get("read", 0) + \
               self.signals.get("impression", 0)

    def weighted_score(self) -> float:
        """Compute weighted engagement score."""
        score = 0.0
        for signal_name, count in self.signals.items():
            # Match signal name to weight key (flexible naming)
            weight = _resolve_weight(signal_name)
            score += weight * count
        return score


@dataclass
class CreativeEfficacyScore:
    work_id: str
    platform: str
    category: str
    content_efficacy: float            # conversion quality given views
    discoverability_efficacy: float    # reach relative to baseline
    combined_efficacy: float           # geometric mean of both
    sufficient_data: bool
    observation_count: int
    notes: str = ""


class CreativeEfficacyTracker:
    """
    Tracks creative efficacy across platforms and categories.
    Maintains category baselines from human creator benchmarks.

    Efficacy is on [0, 1] with 0.5 = matches human average,
    consistent with the STEM efficacy scale.
    """

    def __init__(self):
        # category → list of human baseline EngagementSignals
        self.baselines: Dict[str, List[EngagementSignals]] = {}
        # work_id → list of EngagementSignals over time
        self.work_history: Dict[str, List[EngagementSignals]] = {}
        # scored results
        self.scores: List[CreativeEfficacyScore] = []

    def add_baseline(self, signals: EngagementSignals):
        """
        Register a human creator's engagement signals as a baseline sample.
        Call with many human works to build a robust category average.
        """
        key = f"{signals.platform}:{signals.category}"
        self.baselines.setdefault(key, []).append(signals)

    def update_work(self, work_id: str, signals: EngagementSignals):
        """Record a new observation snapshot for an AI work."""
        self.work_history.setdefault(work_id, []).append(signals)

    def score_work(self, work_id: str) -> Optional[CreativeEfficacyScore]:
        """
        Compute the current efficacy score for a work.
        Returns None if insufficient data.
        """
        history = self.work_history.get(work_id)
        if not history:
            return None

        # Use most recent observation
        latest = history[-1]
        baseline_key = f"{latest.platform}:{latest.category}"
        baselines = self.baselines.get(baseline_key, [])

        if latest.total_observations < MIN_OBSERVATIONS:
            return CreativeEfficacyScore(
                work_id=work_id,
                platform=latest.platform,
                category=latest.category,
                content_efficacy=0.0,
                discoverability_efficacy=0.0,
                combined_efficacy=0.0,
                sufficient_data=False,
                observation_count=latest.total_observations,
                notes=f"Insufficient data: {latest.total_observations} < {MIN_OBSERVATIONS} observations"
            )

        # ── Content Efficacy ──────────────────────────────────────────────────
        # Conversion rate: weighted engagement per view
        # Measures quality given an audience — independent of reach
        ai_conversion = _conversion_rate(latest)
        baseline_conversion = _average_conversion_rate(baselines) if baselines else ai_conversion
        content_efficacy = _sigmoid_efficacy(ai_conversion, baseline_conversion)

        # ── Discoverability Efficacy ──────────────────────────────────────────
        # Raw reach relative to baseline accounts at same account age/size
        # Measures the marketing/platform skill component
        ai_reach = latest.total_observations
        baseline_reach = _average_reach(baselines) if baselines else ai_reach
        discoverability_efficacy = _sigmoid_efficacy(ai_reach, baseline_reach)

        # ── Combined Efficacy ─────────────────────────────────────────────────
        # Geometric mean: both components must be strong for high combined score
        # A viral but low-quality work scores low; high quality but no reach scores low
        combined = math.sqrt(content_efficacy * discoverability_efficacy)

        score = CreativeEfficacyScore(
            work_id=work_id,
            platform=latest.platform,
            category=latest.category,
            content_efficacy=round(content_efficacy, 4),
            discoverability_efficacy=round(discoverability_efficacy, 4),
            combined_efficacy=round(combined, 4),
            sufficient_data=True,
            observation_count=latest.total_observations,
            notes=self._build_notes(content_efficacy, discoverability_efficacy)
        )

        self.scores.append(score)
        return score

    def category_summary(self, platform: str, category: str) -> dict:
        """Summary of all scored works in a category."""
        key = f"{platform}:{category}"
        relevant = [s for s in self.scores if f"{s.platform}:{s.category}" == key]
        if not relevant:
            return {}
        return {
            "platform": platform,
            "category": category,
            "works_scored": len(relevant),
            "avg_content_efficacy": round(sum(s.content_efficacy for s in relevant) / len(relevant), 4),
            "avg_discoverability": round(sum(s.discoverability_efficacy for s in relevant) / len(relevant), 4),
            "avg_combined": round(sum(s.combined_efficacy for s in relevant) / len(relevant), 4),
            "baseline_samples": len(self.baselines.get(key, [])),
        }

    def _build_notes(self, content: float, discoverability: float) -> str:
        notes = []
        if content > 0.65:
            notes.append("strong content quality")
        elif content < 0.35:
            notes.append("weak content quality — work not converting when shown")
        if discoverability > 0.65:
            notes.append("strong reach")
        elif discoverability < 0.35:
            notes.append("weak discoverability — improve tagging/titling/cross-promotion")
        return "; ".join(notes) if notes else "within baseline range"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_weight(signal_name: str) -> float:
    """Flexible signal name → weight mapping."""
    name = signal_name.lower().rstrip("s")  # plurals
    for key, weight in SIGNAL_WEIGHTS.items():
        if key in name or name in key:
            return weight
    return 0.1  # unknown signal → treat as weak


def _conversion_rate(signals: EngagementSignals) -> float:
    """Weighted engagement score per view — quality given an audience."""
    views = max(signals.total_observations, 1)
    # Exclude views/listens from numerator — we want engagement RATE
    engagement = sum(
        _resolve_weight(k) * v
        for k, v in signals.signals.items()
        if not any(w in k.lower() for w in ["view", "listen", "stream", "read", "impression"])
    )
    return engagement / views


def _average_conversion_rate(baselines: List[EngagementSignals]) -> float:
    if not baselines:
        return 0.01
    rates = [_conversion_rate(b) for b in baselines if b.total_observations >= MIN_OBSERVATIONS]
    return sum(rates) / len(rates) if rates else 0.01


def _average_reach(baselines: List[EngagementSignals]) -> float:
    if not baselines:
        return 100.0
    reaches = [b.total_observations for b in baselines if b.total_observations >= MIN_OBSERVATIONS]
    return sum(reaches) / len(reaches) if reaches else 100.0


def _sigmoid_efficacy(ai_value: float, baseline_value: float) -> float:
    """
    Sigmoid-normalized efficacy — same scale as STEM fields.
    ai == baseline → 0.5
    ai > baseline  → > 0.5
    ai < baseline  → < 0.5
    """
    if baseline_value == 0:
        return 0.5
    ratio = ai_value / baseline_value
    return 1.0 - 1.0 / (1.0 + ratio)
