"""Tests for VCG welfare maximization arbitration."""

from unittest.mock import MagicMock

from aua.config import RouterConfig

# ── Helpers ──────────────────────────────────────────────────────────────────


def make_spec(name: str, field: str):
    spec = MagicMock()
    spec.name = name
    spec.field = field
    return spec


class _VCGRouter:
    """Minimal stub exposing only _vcg_select for unit testing."""

    def __init__(self):
        self._arbitration_mode = "vcg"
        self._scorer = MagicMock()
        self._scorer.history = []

    def _vcg_select(self, responses, distribution):
        from aua.router import Router

        return Router._vcg_select(self, responses, distribution)


def make_vcg_router():
    return _VCGRouter()


# ── RouterConfig tests ────────────────────────────────────────────────────────


def test_router_config_default_arbitration_mode():
    cfg = RouterConfig()
    assert cfg.arbitration_mode == "pairwise"


def test_router_config_accepts_vcg():
    cfg = RouterConfig(arbitration_mode="vcg")
    assert cfg.arbitration_mode == "vcg"


# ── _vcg_select tests ─────────────────────────────────────────────────────────


def test_vcg_select_winner_has_highest_welfare():
    """Specialist with highest P × C × U_mean wins."""
    router = make_vcg_router()
    swe = make_spec("swe", "software_engineering")
    math = make_spec("math", "mathematics")

    # W_swe = 0.80 * 0.85 * 1.0 = 0.680  (prior_u=1.0 — no history)
    # W_math = 0.20 * 0.90 * 1.0 = 0.180
    responses = [(swe, "swe text", 0.85), (math, "math text", 0.90)]
    distribution = {"software_engineering": 0.80, "mathematics": 0.20}

    winner_idx, welfare = router._vcg_select(responses, distribution)

    assert winner_idx == 0
    assert welfare["swe"] > welfare["math"]


def test_vcg_select_welfare_dict_contains_all_specialists():
    """Welfare dict must have one entry per specialist in responses."""
    router = make_vcg_router()
    swe = make_spec("swe", "software_engineering")
    math = make_spec("math", "mathematics")
    law = make_spec("law", "law")

    responses = [(swe, "a", 0.80), (math, "b", 0.70), (law, "c", 0.60)]
    distribution = {"software_engineering": 0.50, "mathematics": 0.30, "law": 0.20}

    winner_idx, welfare = router._vcg_select(responses, distribution)

    assert set(welfare.keys()) == {"swe", "math", "law"}
    assert winner_idx == 0  # W_swe=0.40, W_math=0.21, W_law=0.12


def test_vcg_select_n2_correct_winner():
    """With n=2, lower-probability but higher-confidence specialist can win."""
    router = make_vcg_router()
    a = make_spec("a", "software_engineering")
    b = make_spec("b", "mathematics")

    # W_a = 0.60 * 0.40 * 1.0 = 0.24
    # W_b = 0.40 * 0.95 * 1.0 = 0.38  ← b wins despite lower P(domain)
    responses = [(a, "a text", 0.40), (b, "b text", 0.95)]
    distribution = {"software_engineering": 0.60, "mathematics": 0.40}

    winner_idx, welfare = router._vcg_select(responses, distribution)

    assert winner_idx == 1
    assert welfare["b"] > welfare["a"]


def test_vcg_select_tie_broken_by_confidence():
    """Equal welfare → tie broken by confidence."""
    router = make_vcg_router()
    a = make_spec("a", "software_engineering")
    b = make_spec("b", "mathematics")

    # P equal → W_a = 0.50*0.70=0.35, W_b = 0.50*0.90=0.45 → b wins
    responses = [(a, "a text", 0.70), (b, "b text", 0.90)]
    distribution = {"software_engineering": 0.50, "mathematics": 0.50}

    winner_idx, welfare = router._vcg_select(responses, distribution)

    assert winner_idx == 1


def test_vcg_select_no_history_defaults_prior_u_to_1():
    """No prior session history → prior_u = 1.0 so W = P × confidence."""
    router = make_vcg_router()
    spec = make_spec("swe", "software_engineering")
    responses = [(spec, "text", 0.75)]
    distribution = {"software_engineering": 0.90}

    winner_idx, welfare = router._vcg_select(responses, distribution)

    expected_w = round(0.90 * 0.75 * 1.0, 6)
    assert winner_idx == 0
    assert abs(welfare["swe"] - expected_w) < 0.001


def test_vcg_select_with_prior_history():
    """Prior mean U from scorer history is used in welfare calculation."""
    router = make_vcg_router()

    # Inject fake history
    swe_rec = MagicMock()
    swe_rec.field = "software_engineering"
    swe_rec.utility = 0.60

    router._scorer.history = [swe_rec]

    spec = make_spec("swe", "software_engineering")
    responses = [(spec, "text", 0.80)]
    distribution = {"software_engineering": 1.00}

    _, welfare = router._vcg_select(responses, distribution)

    expected = round(1.00 * 0.80 * 0.60, 6)
    assert abs(welfare["swe"] - expected) < 0.001


def test_vcg_welfare_scores_are_non_negative():
    """All welfare scores must be ≥ 0."""
    router = make_vcg_router()
    swe = make_spec("swe", "software_engineering")
    math = make_spec("math", "mathematics")

    responses = [(swe, "a", 0.50), (math, "b", 0.30)]
    distribution = {"software_engineering": 0.70, "mathematics": 0.30}

    _, welfare = router._vcg_select(responses, distribution)

    assert all(v >= 0.0 for v in welfare.values())


def test_vcg_select_single_specialist():
    """Single specialist in responses — trivially wins."""
    router = make_vcg_router()
    spec = make_spec("swe", "software_engineering")
    responses = [(spec, "text", 0.80)]
    distribution = {"software_engineering": 0.95}

    winner_idx, welfare = router._vcg_select(responses, distribution)

    assert winner_idx == 0
    assert len(welfare) == 1


# ── Version ───────────────────────────────────────────────────────────────────


def test_version_is_102():
    import aua

    assert aua.__version__ == "1.0.2"
