"""Tests for VCG welfare maximization arbitration (§10.6.7.1, Appendix B §B.8)."""

from unittest.mock import MagicMock

from aua.config import RouterConfig

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_spec(name: str, field: str):
    spec = MagicMock()
    spec.name = name
    spec.field = field
    return spec


def _make_runs(wins: int, total: int, domain: str, spec: str) -> list[dict]:
    """Build fake model_runs rows: `wins` winners out of `total`."""
    rows = []
    for i in range(total):
        rows.append(
            {
                "specialist": spec,
                "domain": domain,
                "round": "answer",
                "vcg_winner": 1 if i < wins else 0,
            }
        )
    return rows


class _VCGRouter:
    """
    Minimal stub exposing _vcg_select, _vcg_welfare, and _vcg_effective_u
    for unit testing without a live Router instance.
    """

    _VCG_N_CLIFF = 10
    _VCG_GLOBAL_PRIOR = 0.65

    def __init__(self, run_db: dict[tuple, list] | None = None):
        """
        run_db: {(specialist_name, domain): [model_run_rows, ...]}
        Used to fake _state_store.query() responses.
        """
        self._run_db = run_db or {}
        self._state_store = MagicMock()
        self._state_store.query.side_effect = self._fake_query

    def _fake_query(self, table, filters=None, limit=500):
        if table != "model_runs":
            return []
        spec = (filters or {}).get("specialist", "")
        domain = (filters or {}).get("domain", "")
        return self._run_db.get((spec, domain), [])

    def _vcg_effective_u(self, specialist_name, domain):
        from aua.router import Router

        return Router._vcg_effective_u(self, specialist_name, domain)

    def _vcg_welfare(self, spec_name, distribution):
        from aua.router import Router

        return Router._vcg_welfare(self, spec_name, distribution)

    def _vcg_select(self, responses, distribution):
        from aua.router import Router

        return Router._vcg_select(self, responses, distribution)


# ── RouterConfig ──────────────────────────────────────────────────────────────


def test_router_config_default_arbitration_mode():
    cfg = RouterConfig()
    assert cfg.arbitration_mode == "pairwise"


def test_router_config_accepts_vcg():
    cfg = RouterConfig(arbitration_mode="vcg")
    assert cfg.arbitration_mode == "vcg"


# ── _vcg_effective_u — shrinkage formula ──────────────────────────────────────


def test_effective_u_no_history_returns_prior():
    """With no observations, effective_u falls back to the global prior."""
    router = _VCGRouter(run_db={})
    eu = router._vcg_effective_u("swe", "software_engineering")
    assert abs(eu - 0.65) < 1e-6


def test_effective_u_shrinkage_toward_prior_low_n():
    """
    With n=2 wins out of 4 queries (raw=0.5), effective_u should be pulled
    toward prior 0.65.  Formula: (4*0.5 + 10*0.65) / (4+10) = 8.5/14 ≈ 0.607.
    """
    runs = _make_runs(wins=2, total=4, domain="software_engineering", spec="swe")
    router = _VCGRouter(run_db={("swe", "software_engineering"): runs})
    eu = router._vcg_effective_u("swe", "software_engineering")
    expected = (4 * 0.5 + 10 * 0.65) / (4 + 10)
    assert abs(eu - expected) < 1e-6


def test_effective_u_large_n_converges_to_raw():
    """
    With n=100 wins out of 100 (raw=1.0), effective_u ≈ 1.0 (prior washes out).
    Formula: (100*1.0 + 10*0.65) / (100+10) = 106.5/110 ≈ 0.9682.
    """
    runs = _make_runs(wins=100, total=100, domain="software_engineering", spec="swe")
    router = _VCGRouter(run_db={("swe", "software_engineering"): runs})
    eu = router._vcg_effective_u("swe", "software_engineering")
    expected = (100 * 1.0 + 10 * 0.65) / (100 + 10)
    assert abs(eu - expected) < 1e-6


def test_effective_u_positivity():
    """effective_u must always be > 0 (Proposition B.8.3 P1)."""
    runs = _make_runs(wins=0, total=20, domain="mathematics", spec="math")
    router = _VCGRouter(run_db={("math", "mathematics"): runs})
    eu = router._vcg_effective_u("math", "mathematics")
    assert eu > 0.0


def test_effective_u_skill_monotonicity():
    """
    Specialist A with more wins than B in the same domain should have a
    higher effective_u (Proposition B.8.3 P2).
    """
    runs_a = _make_runs(wins=15, total=20, domain="mathematics", spec="a")
    runs_b = _make_runs(wins=5, total=20, domain="mathematics", spec="b")
    router_a = _VCGRouter(run_db={("a", "mathematics"): runs_a})
    router_b = _VCGRouter(run_db={("b", "mathematics"): runs_b})
    eu_a = router_a._vcg_effective_u("a", "mathematics")
    eu_b = router_b._vcg_effective_u("b", "mathematics")
    assert eu_a > eu_b


def test_effective_u_state_store_error_returns_prior():
    """DB lookup failures should silently fall back to the global prior."""
    router = _VCGRouter()
    router._state_store.query.side_effect = Exception("DB error")
    eu = router._vcg_effective_u("swe", "software_engineering")
    assert abs(eu - 0.65) < 1e-6


# ── _vcg_welfare — multi-domain formula ───────────────────────────────────────


def test_vcg_welfare_single_domain_equals_effective_u():
    """With one domain at p=1.0, welfare equals effective_u for that domain."""
    runs = _make_runs(wins=8, total=10, domain="software_engineering", spec="swe")
    router = _VCGRouter(run_db={("swe", "software_engineering"): runs})
    w, _ = router._vcg_welfare("swe", {"software_engineering": 1.0})
    eu = router._vcg_effective_u("swe", "software_engineering")
    assert abs(w - eu) < 1e-5


def test_vcg_welfare_multi_domain_is_convex_combination():
    """
    W_i = p1*eu1 + p2*eu2 (additive separability, Proposition B.8.3 P3).
    """
    runs_swe = _make_runs(wins=8, total=10, domain="software_engineering", spec="swe")
    runs_math = _make_runs(wins=3, total=10, domain="mathematics", spec="swe")
    router = _VCGRouter(
        run_db={
            ("swe", "software_engineering"): runs_swe,
            ("swe", "mathematics"): runs_math,
        }
    )
    dist = {"software_engineering": 0.70, "mathematics": 0.30}
    w, breakdown = router._vcg_welfare("swe", dist)

    eu_swe = router._vcg_effective_u("swe", "software_engineering")
    eu_math = router._vcg_effective_u("swe", "mathematics")
    expected = (0.70 * eu_swe + 0.30 * eu_math) / 1.0
    assert abs(w - round(expected, 6)) < 1e-5
    assert set(breakdown.keys()) == {"software_engineering", "mathematics"}


def test_vcg_welfare_ignores_low_probability_domains():
    """Domains with p < 0.05 should be excluded from welfare calculation."""
    runs = _make_runs(wins=9, total=10, domain="software_engineering", spec="swe")
    router = _VCGRouter(run_db={("swe", "software_engineering"): runs})
    dist = {"software_engineering": 0.97, "law": 0.03}  # law < 0.05 threshold
    w, breakdown = router._vcg_welfare("swe", dist)
    assert "law" not in breakdown


def test_vcg_welfare_non_negative():
    """Welfare is always ≥ 0 (Proposition B.8.3 P1)."""
    router = _VCGRouter(run_db={})  # all-prior scenario
    w, _ = router._vcg_welfare("swe", {"software_engineering": 1.0})
    assert w >= 0.0


# ── _vcg_select — allocation rule ─────────────────────────────────────────────


def test_vcg_select_winner_has_highest_welfare():
    """Specialist with highest welfare score wins."""
    # swe: 15/20 wins → eu ≈ 0.793; math: 3/20 wins → eu ≈ 0.502
    # W_swe = 0.80 * eu_swe; W_math = 0.20 * eu_math → swe wins
    runs_swe = _make_runs(wins=15, total=20, domain="software_engineering", spec="swe")
    runs_math = _make_runs(wins=3, total=20, domain="mathematics", spec="math")
    router = _VCGRouter(
        run_db={
            ("swe", "software_engineering"): runs_swe,
            ("math", "mathematics"): runs_math,
        }
    )
    swe = make_spec("swe", "software_engineering")
    math_spec = make_spec("math", "mathematics")
    responses = [(swe, "swe text", 0.75), (math_spec, "math text", 0.90)]
    distribution = {"software_engineering": 0.80, "mathematics": 0.20}

    winner_idx, welfare = router._vcg_select(responses, distribution)

    assert winner_idx == 0
    assert welfare["swe"] > welfare["math"]


def test_vcg_select_welfare_dict_contains_all_specialists():
    """Welfare dict must have one key per specialist regardless of n."""
    router = _VCGRouter(run_db={})
    swe = make_spec("swe", "software_engineering")
    math_spec = make_spec("math", "mathematics")
    law = make_spec("law", "law")

    responses = [(swe, "a", 0.80), (math_spec, "b", 0.70), (law, "c", 0.60)]
    distribution = {"software_engineering": 0.50, "mathematics": 0.30, "law": 0.20}

    _, welfare = router._vcg_select(responses, distribution)

    assert set(welfare.keys()) == {"swe", "math", "law"}


def test_vcg_select_n_ge_3_correct_winner():
    """With n=3 specialists, the argmax is selected correctly."""
    # No history → all fall back to prior 0.65; winner determined by P(domain)
    router = _VCGRouter(run_db={})
    a = make_spec("a", "software_engineering")
    b = make_spec("b", "mathematics")
    c = make_spec("c", "law")

    responses = [(a, "ta", 0.70), (b, "tb", 0.70), (c, "tc", 0.70)]
    distribution = {"software_engineering": 0.60, "mathematics": 0.25, "law": 0.15}

    winner_idx, welfare = router._vcg_select(responses, distribution)
    # W_a dominates: p=0.60 > 0.25 > 0.15, all same effective_u (prior=0.65)
    # W_a = 0.65, W_b = 0.65, W_c = 0.65 (all single-domain → W_i = eu = prior)
    # tie-break by P(domain): a wins
    assert winner_idx == 0


def test_vcg_select_tie_broken_by_confidence():
    """When welfare scores are equal, higher confidence wins."""
    # Equal runs → equal effective_u; equal distribution → equal welfare
    runs = _make_runs(wins=5, total=10, domain="software_engineering", spec="a")
    runs2 = _make_runs(wins=5, total=10, domain="mathematics", spec="b")
    router = _VCGRouter(
        run_db={
            ("a", "software_engineering"): runs,
            ("b", "mathematics"): runs2,
        }
    )
    a = make_spec("a", "software_engineering")
    b = make_spec("b", "mathematics")
    # Both p=0.50 and same wins/total → same effective_u → same welfare
    # Tie broken by confidence: b has higher conf
    responses = [(a, "ta", 0.60), (b, "tb", 0.90)]
    distribution = {"software_engineering": 0.50, "mathematics": 0.50}

    winner_idx, _ = router._vcg_select(responses, distribution)
    assert winner_idx == 1  # b wins on confidence tie-break


def test_vcg_select_welfare_scores_non_negative():
    """All welfare scores must be ≥ 0 (Proposition B.8.3 P1)."""
    router = _VCGRouter(run_db={})
    swe = make_spec("swe", "software_engineering")
    math_spec = make_spec("math", "mathematics")

    responses = [(swe, "a", 0.50), (math_spec, "b", 0.30)]
    distribution = {"software_engineering": 0.70, "mathematics": 0.30}

    _, welfare = router._vcg_select(responses, distribution)
    assert all(v >= 0.0 for v in welfare.values())


def test_vcg_select_single_specialist():
    """Single specialist trivially wins."""
    router = _VCGRouter(run_db={})
    spec = make_spec("swe", "software_engineering")
    responses = [(spec, "text", 0.80)]
    distribution = {"software_engineering": 0.95}

    winner_idx, welfare = router._vcg_select(responses, distribution)

    assert winner_idx == 0
    assert len(welfare) == 1


def test_vcg_select_no_history_uses_prior():
    """With no run history, effective_u = global prior = 0.65 for all specialists."""
    router = _VCGRouter(run_db={})
    a = make_spec("a", "software_engineering")
    b = make_spec("b", "mathematics")

    responses = [(a, "ta", 0.80), (b, "tb", 0.80)]
    distribution = {"software_engineering": 0.70, "mathematics": 0.30}

    winner_idx, welfare = router._vcg_select(responses, distribution)

    # Each specialist has a single domain at full weight → W_i = effective_u = prior = 0.65
    # Winner chosen by tie-break on P(domain): a has p=0.70 > b's p=0.30
    assert winner_idx == 0
    assert abs(welfare["a"] - 0.65) < 0.01
    assert abs(welfare["b"] - 0.65) < 0.01


# ── Version ───────────────────────────────────────────────────────────────────


def test_version_matches_source():
    import aua
    from aua.version import __version__ as _v

    assert aua.__version__ == _v
