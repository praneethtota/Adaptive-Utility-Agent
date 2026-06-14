"""
tests/test_routing_and_utility.py — Comprehensive routing + utility dynamics suite.

Four areas:
  1. Routing correctness — queries reach the right specialist or right set.
  2. Utility EMA dynamics — confidence & efficacy increments slow down with
     query count (geometric decay, ratio ≈ 1−α); decrements behave symmetrically.
  3. Fanout re-evaluation — every specialist used in a fanout has its utility /
     welfare re-evaluated per the §10.6.7.1 formula.
  4. Extended VCG mechanism checks — allocation, individual rationality,
     monotonicity, shrinkage convergence.

Routing tests drive a controlled domain distribution by patching the classifier
and intercept `_call` to record which specialist endpoints were hit — no live
GPU or model required.
"""

import asyncio

import pytest

from aua.config import FIELD_CONFIGS, AUAConfig, load_config
from aua.router import Router
from aua.utility_scorer import UtilityScorer

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures + helpers
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def routing_router(tmp_path) -> Router:
    """Two-specialist router with an isolated temp state DB."""
    cfg: AUAConfig = load_config("tests/fixtures/aua_config_two_specialists.yaml")
    cfg.state.path = str(tmp_path / "routing_test.db")
    return Router.from_config(cfg)


def _patch_distribution(monkeypatch, router: Router, dist: dict[str, float]) -> None:
    """Force the classifier to return a fixed domain distribution."""
    monkeypatch.setattr(
        router._classifier,
        "classify",
        lambda q, update_history=True: dict(dist),
    )


def _patch_call_recorder(monkeypatch, router: Router) -> list[tuple[str, str]]:
    """
    Replace _call with a recorder. Returns a list that will be populated with
    (url, domain) tuples for every specialist/arbiter call made.
    """
    called: list[tuple[str, str]] = []

    async def fake_call(
        url, query, domain, history=None, system_prompt=None, model_name="default_model"
    ):
        called.append((url, domain))
        # Return a plausible response + a "stop" confidence
        if domain == "arbiter":
            return ("VERDICT: A\nREASON: ok\nCORRECTION: none", 0.75)
        return (f"answer from {domain}", 0.75)

    monkeypatch.setattr(router, "_call", fake_call)
    return called


# ──────────────────────────────────────────────────────────────────────────────
# 1. ROUTING CORRECTNESS
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_domain_routes_to_correct_specialist(routing_router, monkeypatch):
    """A query with one dominant domain (≥ single_domain_threshold) routes to that specialist only."""
    r = routing_router
    _patch_distribution(monkeypatch, r, {"software_engineering": 0.95, "mathematics": 0.05})
    called = _patch_call_recorder(monkeypatch, r)

    resp = await r.query("write a quicksort function")

    assert resp.routing_mode == "single"
    assert resp.primary_domain == "software_engineering"
    # exactly one specialist call, to the SWE domain
    assert len(called) == 1
    assert called[0][1] == "software_engineering"
    # the URL must be the SWE specialist's endpoint
    assert called[0][0] == r._field_to_url["software_engineering"]


@pytest.mark.asyncio
async def test_single_domain_routes_to_math_specialist(routing_router, monkeypatch):
    """Dominant math distribution routes to the math specialist, not SWE."""
    r = routing_router
    _patch_distribution(monkeypatch, r, {"mathematics": 0.90, "software_engineering": 0.10})
    called = _patch_call_recorder(monkeypatch, r)

    resp = await r.query("prove the pythagorean theorem")

    assert resp.routing_mode == "single"
    assert resp.primary_domain == "mathematics"
    assert len(called) == 1
    assert called[0][1] == "mathematics"


@pytest.mark.asyncio
async def test_fanout_activates_both_specialists(routing_router, monkeypatch):
    """
    A cross-domain query (both domains ≥ fanout_threshold) fans out to BOTH
    specialists. In pairwise mode the ArbiterAgent (or LLM) is also called.
    """
    from unittest.mock import MagicMock

    from aua.arbiter import ArbiterVerdict, VerdictCase

    r = routing_router
    r._arbitration_mode = "pairwise"
    _patch_distribution(monkeypatch, r, {"software_engineering": 0.55, "mathematics": 0.45})
    called = _patch_call_recorder(monkeypatch, r)

    # Mock ArbiterAgent so it returns a deterministic verdict without network calls
    mock_agent = MagicMock()
    mock_verdict = ArbiterVerdict(
        subject="test",
        domain="software_engineering",
        case=VerdictCase.CASE_1,
        arbiter_confidence=0.90,
    )
    mock_verdict.correct_B = True
    mock_verdict.evidence_summary = "A is better"
    mock_agent.arbitrate.return_value = mock_verdict
    r._arbiter_agent = mock_agent

    resp = await r.query("write code to compute eigenvalues and explain the math")

    assert resp.routing_mode == "fanout"
    domains_called = {d for _, d in called}
    # both specialists must have been called
    assert "software_engineering" in domains_called
    assert "mathematics" in domains_called
    # ArbiterAgent was invoked (pairwise mode)
    assert mock_agent.arbitrate.called


@pytest.mark.asyncio
async def test_active_set_excludes_below_threshold(routing_router, monkeypatch):
    """
    A specialist whose probability is below fanout_threshold must NOT be in the
    active set. swe=0.70 (< 0.75 single, ≥ 0.30 fanout-eligible alone), math=0.20
    (< 0.30) → only swe active → arbiter fallback (single not met, <2 active).
    """
    r = routing_router
    _patch_distribution(monkeypatch, r, {"software_engineering": 0.70, "mathematics": 0.20})
    called = _patch_call_recorder(monkeypatch, r)

    resp = await r.query("ambiguous query")

    # math is below fanout_threshold (0.30) → not active; swe alone (1) and
    # top_prob 0.70 < single_threshold 0.75 → arbiter fallback
    assert resp.routing_mode == "arbiter"
    # the SWE specialist must NOT be called; the fallback hits the arbiter URL
    # (the arbiter answers as a generalist, so domain is recorded as "general")
    assert r._field_to_url["software_engineering"] not in {u for u, _ in called}
    assert all(u == r._arbiter_url for u, _ in called)


@pytest.mark.asyncio
async def test_arbiter_fallback_on_low_confidence(routing_router, monkeypatch):
    """No domain reaches single_domain_threshold and <2 active → arbiter fallback."""
    r = routing_router
    _patch_distribution(monkeypatch, r, {"software_engineering": 0.50, "mathematics": 0.10})
    called = _patch_call_recorder(monkeypatch, r)

    resp = await r.query("vague question")

    assert resp.routing_mode == "arbiter"
    assert resp.primary_domain == "general"
    assert len(called) == 1
    # arbiter fallback routes to the arbiter URL, answering as a generalist
    assert called[0][0] == r._arbiter_url
    assert called[0][1] == "general"


@pytest.mark.asyncio
async def test_force_domain_overrides_classifier(routing_router, monkeypatch):
    """force_domain pins the distribution to {domain: 1.0} and routes there."""
    r = routing_router
    called = _patch_call_recorder(monkeypatch, r)
    # Do NOT patch the classifier — force_domain should bypass it entirely

    resp = await r.query("anything", force_domain="mathematics")

    assert resp.routing_mode == "single"
    assert resp.primary_domain == "mathematics"
    assert called[0][1] == "mathematics"


@pytest.mark.asyncio
async def test_fanout_threshold_boundary_exact(routing_router, monkeypatch):
    """A specialist exactly at fanout_threshold (0.30) IS active (>= comparison)."""
    r = routing_router
    r._arbitration_mode = "pairwise"
    _patch_distribution(monkeypatch, r, {"software_engineering": 0.70, "mathematics": 0.30})
    called = _patch_call_recorder(monkeypatch, r)

    resp = await r.query("boundary query")

    # math at exactly 0.30 → active → 2 active → fanout
    assert resp.routing_mode == "fanout"
    domains_called = {d for _, d in called}
    assert "mathematics" in domains_called


# ──────────────────────────────────────────────────────────────────────────────
# 2. UTILITY EMA DYNAMICS — increment slows down as query count grows
# ──────────────────────────────────────────────────────────────────────────────


def _confidence_series(scorer: UtilityScorer, domain_cfg, signal: float, n: int) -> list[float]:
    """Run n scoring steps with a constant test_pass_rate; return full-precision confidences."""
    out = []
    for i in range(n):
        scorer.score(
            task_id=f"t{i}",
            field_config=domain_cfg,
            test_pass_rate=signal,
            human_baseline_score=0.65,
            contradiction_penalty=0.0,
            problem_novelty=0.3,
        )
        out.append(scorer.domain_states[domain_cfg.name].confidence)
    return out


def test_confidence_increment_is_positive_and_slows_down():
    """
    With a constant high signal, confidence rises but each increment is smaller
    than the last (EMA geometric decay). This is the 'slows down as queries
    increase' property.
    """
    scorer = UtilityScorer()
    cfg = FIELD_CONFIGS["software_engineering"]
    conf = _confidence_series(scorer, cfg, signal=0.95, n=10)

    increments = [conf[i + 1] - conf[i] for i in range(len(conf) - 1)]

    # All increments positive (rising toward 0.95 from 0.5 start)
    assert all(d > 0 for d in increments), increments
    # Strictly decreasing magnitude
    for i in range(len(increments) - 1):
        assert increments[i + 1] < increments[i], (i, increments)


def test_confidence_increment_ratio_matches_one_minus_alpha():
    """
    Consecutive EMA increments form a geometric sequence with ratio (1−α).
    UtilityScorer.CONFIDENCE_ALPHA = 0.2 → ratio ≈ 0.8.
    """
    scorer = UtilityScorer()
    cfg = FIELD_CONFIGS["software_engineering"]
    conf = _confidence_series(scorer, cfg, signal=0.95, n=8)

    increments = [conf[i + 1] - conf[i] for i in range(len(conf) - 1)]
    ratios = [increments[i + 1] / increments[i] for i in range(len(increments) - 1)]

    expected = 1 - UtilityScorer.CONFIDENCE_ALPHA
    assert all(abs(r - expected) < 1e-6 for r in ratios), ratios


def test_confidence_converges_to_signal():
    """After many steps the confidence EMA approaches the constant signal."""
    scorer = UtilityScorer()
    cfg = FIELD_CONFIGS["software_engineering"]
    conf = _confidence_series(scorer, cfg, signal=0.90, n=60)
    assert abs(conf[-1] - 0.90) < 0.01


def test_confidence_decrement_slows_down():
    """
    After building confidence high, a sustained low signal makes confidence
    fall — and each decrement shrinks in magnitude (symmetric EMA decay).
    """
    scorer = UtilityScorer()
    cfg = FIELD_CONFIGS["software_engineering"]
    # Build up high confidence first
    _confidence_series(scorer, cfg, signal=0.95, n=40)
    # Now feed a sustained low signal
    low = _confidence_series(scorer, cfg, signal=0.20, n=10)

    decrements = [low[i + 1] - low[i] for i in range(len(low) - 1)]
    # All decrements negative (falling toward 0.20)
    assert all(d < 0 for d in decrements), decrements
    # Magnitude strictly shrinks
    mags = [abs(d) for d in decrements]
    for i in range(len(mags) - 1):
        assert mags[i + 1] < mags[i], (i, mags)


def test_confidence_decrement_ratio_matches_one_minus_alpha():
    """Decrement magnitudes also decay with ratio (1−α) = 0.8."""
    scorer = UtilityScorer()
    cfg = FIELD_CONFIGS["software_engineering"]
    _confidence_series(scorer, cfg, signal=0.95, n=40)
    low = _confidence_series(scorer, cfg, signal=0.10, n=8)

    decrements = [low[i + 1] - low[i] for i in range(len(low) - 1)]
    ratios = [decrements[i + 1] / decrements[i] for i in range(len(decrements) - 1)]
    expected = 1 - UtilityScorer.CONFIDENCE_ALPHA
    assert all(abs(r - expected) < 1e-6 for r in ratios), ratios


def test_efficacy_ema_increment_slows_down():
    """Efficacy EMA increments also shrink geometrically (ratio 1−α=0.8)."""
    scorer = UtilityScorer()
    cfg = FIELD_CONFIGS["software_engineering"]
    effs = []
    for i in range(8):
        scorer.score(
            task_id=f"t{i}",
            field_config=cfg,
            test_pass_rate=0.95,
            human_baseline_score=0.65,
            contradiction_penalty=0.0,
            problem_novelty=0.3,
        )
        effs.append(scorer.domain_states[cfg.name].efficacy_ema)

    increments = [effs[i + 1] - effs[i] for i in range(len(effs) - 1)]
    assert all(d > 0 for d in increments), increments
    ratios = [increments[i + 1] / increments[i] for i in range(len(increments) - 1)]
    expected = 1 - UtilityScorer.EFFICACY_ALPHA
    assert all(abs(r - expected) < 1e-6 for r in ratios), ratios


def test_utility_increases_monotonically_under_good_signal():
    """U rises across queries under a sustained high signal (E and C both climbing)."""
    scorer = UtilityScorer()
    cfg = FIELD_CONFIGS["software_engineering"]
    utils = []
    for i in range(12):
        ts = scorer.score(
            task_id=f"t{i}",
            field_config=cfg,
            test_pass_rate=0.95,
            human_baseline_score=0.65,
            contradiction_penalty=0.0,
            problem_novelty=0.3,
        )
        utils.append(ts.utility)
    # Non-decreasing trend (allow tiny curiosity wobble via >= with epsilon)
    for i in range(len(utils) - 1):
        assert utils[i + 1] >= utils[i] - 1e-3, (i, utils)
    # Net rise from first to last
    assert utils[-1] > utils[0]


def test_contradiction_penalty_lowers_confidence():
    """A contradiction penalty drives confidence below the no-penalty trajectory."""
    clean = UtilityScorer()
    penalised = UtilityScorer()
    cfg = FIELD_CONFIGS["software_engineering"]
    for i in range(10):
        clean.score(
            task_id=f"c{i}",
            field_config=cfg,
            test_pass_rate=0.90,
            human_baseline_score=0.65,
            contradiction_penalty=0.0,
            problem_novelty=0.3,
        )
        penalised.score(
            task_id=f"p{i}",
            field_config=cfg,
            test_pass_rate=0.90,
            human_baseline_score=0.65,
            contradiction_penalty=0.15,
            problem_novelty=0.3,
        )
    c_clean = clean.domain_states[cfg.name].confidence
    c_pen = penalised.domain_states[cfg.name].confidence
    assert c_pen < c_clean


# ──────────────────────────────────────────────────────────────────────────────
# 3. FANOUT RE-EVALUATION — every used specialist's utility is recomputed
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vcg_fanout_reevaluates_all_specialists(routing_router, monkeypatch):
    """
    In a VCG fanout, welfare must be computed for EVERY specialist that
    responded — not just the winner. welfare_scores in the response carries
    one entry per active specialist.
    """
    r = routing_router
    r._arbitration_mode = "vcg"
    _patch_distribution(monkeypatch, r, {"software_engineering": 0.55, "mathematics": 0.45})
    _patch_call_recorder(monkeypatch, r)

    resp = await r.query("cross-domain query")

    assert resp.routing_mode == "vcg"
    assert resp.welfare_scores is not None
    # both specialists must appear in the welfare scores
    assert set(resp.welfare_scores.keys()) == {"swe", "math"}
    # all welfare values are positive (individual rationality precondition)
    assert all(v > 0 for v in resp.welfare_scores.values())


@pytest.mark.asyncio
async def test_vcg_fanout_welfare_matches_formula(routing_router, monkeypatch):
    """
    Each welfare score in the response must EXACTLY equal the §10.6.7.1 formula
    W_i(q) = Σ_j p(j|q)·effective_u(i,j), recomputed independently here.
    """
    r = routing_router
    r._arbitration_mode = "vcg"
    dist = {"software_engineering": 0.60, "mathematics": 0.40}
    _patch_distribution(monkeypatch, r, dist)
    _patch_call_recorder(monkeypatch, r)

    resp = await r.query("cross-domain query")

    # Independently recompute each specialist's welfare from the same formula
    for spec in r._config.specialists:
        expected_w, _ = r._vcg_welfare(spec.name, dist)
        assert spec.name in resp.welfare_scores
        assert abs(resp.welfare_scores[spec.name] - expected_w) < 1e-6


@pytest.mark.asyncio
async def test_vcg_fanout_persists_one_run_per_specialist(routing_router, monkeypatch):
    """
    After a VCG fanout, model_runs must contain one 'answer' row per specialist
    with vcg_winner / vcg_welfare_score populated — the data /analytics and
    /reliability read.
    """
    r = routing_router
    r._arbitration_mode = "vcg"
    _patch_distribution(monkeypatch, r, {"software_engineering": 0.55, "mathematics": 0.45})
    _patch_call_recorder(monkeypatch, r)

    await r.query("cross-domain query", session_id="conv-xyz")
    # Let fire_and_forget background tasks flush
    await asyncio.sleep(0.1)

    runs = r._state_store.query("model_runs", limit=100)
    answer_runs = [r_ for r_ in runs if r_.get("round") == "answer"]
    specialists = {r_["specialist"] for r_ in answer_runs}
    assert {"swe", "math"} <= specialists

    # exactly one winner among the two
    winners = [r_ for r_ in answer_runs if r_.get("vcg_winner")]
    assert len(winners) == 1
    # winner has a welfare score recorded
    assert winners[0].get("vcg_welfare_score") is not None


@pytest.mark.asyncio
async def test_vcg_winner_has_highest_persisted_welfare(routing_router, monkeypatch):
    """The persisted winner row must hold the max vcg_welfare_score."""
    r = routing_router
    r._arbitration_mode = "vcg"
    _patch_distribution(monkeypatch, r, {"software_engineering": 0.70, "mathematics": 0.30})
    _patch_call_recorder(monkeypatch, r)

    await r.query("query", session_id="conv-1")
    await asyncio.sleep(0.1)

    runs = [x for x in r._state_store.query("model_runs", limit=100) if x.get("round") == "answer"]
    winner = [x for x in runs if x.get("vcg_winner")][0]
    max_welfare = max(x["vcg_welfare_score"] for x in runs)
    assert winner["vcg_welfare_score"] == max_welfare


# ──────────────────────────────────────────────────────────────────────────────
# 4. EXTENDED VCG MECHANISM CHECKS
# ──────────────────────────────────────────────────────────────────────────────


class _StubRouter:
    """Stub exposing the VCG methods with a controllable fake model_runs DB."""

    _VCG_N_CLIFF = 10
    _VCG_GLOBAL_PRIOR = 0.65

    def __init__(self, run_db=None):
        from unittest.mock import MagicMock

        self._run_db = run_db or {}
        self._state_store = MagicMock()
        self._state_store.query.side_effect = self._fake_query

    def _fake_query(self, table, filters=None, limit=500):
        if table != "model_runs":
            return []
        key = ((filters or {}).get("specialist", ""), (filters or {}).get("domain", ""))
        return self._run_db.get(key, [])

    def _vcg_effective_u(self, name, domain):
        return Router._vcg_effective_u(self, name, domain)

    def _vcg_welfare(self, name, dist):
        return Router._vcg_welfare(self, name, dist)

    def _vcg_select(self, responses, dist):
        return Router._vcg_select(self, responses, dist)


def _runs(wins, total, domain, spec):
    return [
        {
            "specialist": spec,
            "domain": domain,
            "round": "answer",
            "vcg_winner": 1 if i < wins else 0,
        }
        for i in range(total)
    ]


def _make_spec(name, field):
    from unittest.mock import MagicMock

    s = MagicMock()
    s.name = name
    s.field = field
    return s


def test_vcg_allocation_selects_social_welfare_max():
    """
    Theorem S2: VCG selects argmax_i W_i. The specialist with the strongest
    domain track record under the query distribution wins.
    """
    db = {
        ("swe", "software_engineering"): _runs(18, 20, "software_engineering", "swe"),
        ("math", "mathematics"): _runs(4, 20, "mathematics", "math"),
    }
    router = _StubRouter(db)
    swe = _make_spec("swe", "software_engineering")
    math_s = _make_spec("math", "mathematics")
    responses = [(swe, "a", 0.7), (math_s, "b", 0.7)]
    dist = {"software_engineering": 0.7, "mathematics": 0.3}

    idx, welfare = router._vcg_select(responses, dist)
    assert idx == 0
    assert welfare["swe"] == max(welfare.values())


def test_vcg_individual_rationality_winner_positive():
    """Theorem S3: the winner's welfare is strictly > 0 (π_i > 0)."""
    db = {("swe", "software_engineering"): _runs(10, 15, "software_engineering", "swe")}
    router = _StubRouter(db)
    swe = _make_spec("swe", "software_engineering")
    responses = [(swe, "a", 0.8)]
    dist = {"software_engineering": 1.0}
    idx, welfare = router._vcg_select(responses, dist)
    assert welfare[responses[idx][0].name] > 0.0


def test_vcg_monotonicity_more_wins_flips_winner():
    """
    Increasing a specialist's win history raises its effective_u and can flip
    the VCG winner (Proposition B.8.3 P2 + allocation rule).
    """
    swe = _make_spec("swe", "software_engineering")
    math_s = _make_spec("math", "software_engineering")  # same domain, head-to-head
    dist = {"software_engineering": 1.0}

    # Scenario A: math weak → swe wins
    db_a = {
        ("swe", "software_engineering"): _runs(15, 20, "software_engineering", "swe"),
        ("math", "software_engineering"): _runs(2, 20, "software_engineering", "math"),
    }
    idx_a, _ = _StubRouter(db_a)._vcg_select([(swe, "a", 0.7), (math_s, "b", 0.7)], dist)
    assert idx_a == 0  # swe

    # Scenario B: math now dominates → math wins
    db_b = {
        ("swe", "software_engineering"): _runs(2, 20, "software_engineering", "swe"),
        ("math", "software_engineering"): _runs(19, 20, "software_engineering", "math"),
    }
    idx_b, _ = _StubRouter(db_b)._vcg_select([(swe, "a", 0.7), (math_s, "b", 0.7)], dist)
    assert idx_b == 1  # math


def test_vcg_effective_u_change_slows_as_n_grows():
    """
    Shrinkage convergence (Lemma B.8.1): with a constant win-rate, the change
    in effective_u from adding a fixed block of observations shrinks as the
    sample size grows.
    """
    win_rate = 0.8

    def eu_at(n):
        wins = round(win_rate * n)
        db = {("swe", "software_engineering"): _runs(wins, n, "software_engineering", "swe")}
        return _StubRouter(db)._vcg_effective_u("swe", "software_engineering")

    # change in eu when going 5→10, 50→55, 200→205 (same +5 block)
    d_small_n = abs(eu_at(10) - eu_at(5))
    d_mid_n = abs(eu_at(55) - eu_at(50))
    d_large_n = abs(eu_at(205) - eu_at(200))

    assert d_small_n > d_mid_n > d_large_n


def test_vcg_welfare_bounded_by_per_domain_utilities():
    """
    Proposition B.8.3 P3: W_i is a convex combination, so it lies within the
    range of the per-domain effective_u values.
    """
    db = {
        ("swe", "software_engineering"): _runs(18, 20, "software_engineering", "swe"),
        ("swe", "mathematics"): _runs(4, 20, "mathematics", "swe"),
    }
    router = _StubRouter(db)
    dist = {"software_engineering": 0.6, "mathematics": 0.4}
    w, breakdown = router._vcg_welfare("swe", dist)
    assert min(breakdown.values()) <= w <= max(breakdown.values())


def test_vcg_no_data_all_specialists_tie_on_prior():
    """With no history, every specialist's effective_u equals the global prior."""
    router = _StubRouter({})
    eu1 = router._vcg_effective_u("a", "software_engineering")
    eu2 = router._vcg_effective_u("b", "mathematics")
    assert eu1 == eu2 == router._VCG_GLOBAL_PRIOR
