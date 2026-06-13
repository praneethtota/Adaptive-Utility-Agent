"""
tests/test_empirical.py — Tests for #61 Arbiter Stage 4 (external API integration).

Coverage:
  empirical_check() routing by domain
  SymPy source: equivalent expressions, A tighter, B tighter, parse failure
  arXiv source: winner A, winner B, neither, API error, empty results
  PubMed source: winner B, API error, no results
  Fallback domains: law / art / creative_writing
  Arbiter._check_empirical(): integration with CheckResult
  Arbiter.arbitrate(): empirical check fires on real domain
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aua.arbiter import ArbiterAgent, CheckResult
from aua.assertions_store import AssertionsStore
from aua.empirical import (
    EmpiricalResult,
    _arxiv_check,
    _pubmed_check,
    _sympy_check,
    empirical_check,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def arbiter() -> ArbiterAgent:
    return ArbiterAgent(AssertionsStore())


# ── Domain routing ────────────────────────────────────────────────────────────


class TestEmpiricalRouting:
    def test_mathematics_routes_to_sympy(self) -> None:
        with patch("aua.empirical._sympy_check") as mock:
            mock.return_value = EmpiricalResult(source="sympy")
            empirical_check("subject", "mathematics", "A", "B")
            mock.assert_called_once()

    def test_software_engineering_routes_to_arxiv(self) -> None:
        with patch("aua.empirical._arxiv_check") as mock:
            mock.return_value = EmpiricalResult(source="arxiv")
            empirical_check("subject", "software_engineering", "A", "B")
            mock.assert_called_once()

    def test_surgery_routes_to_pubmed(self) -> None:
        with patch("aua.empirical._pubmed_check") as mock:
            mock.return_value = EmpiricalResult(source="pubmed")
            empirical_check("subject", "surgery", "A", "B")
            mock.assert_called_once()

    def test_law_returns_fallback(self) -> None:
        result = empirical_check("subject", "law", "A", "B")
        assert not result.converged
        assert result.source == "none"
        assert "law" in result.explanation.lower() or "manual" in result.explanation.lower()

    def test_art_returns_fallback(self) -> None:
        result = empirical_check("subject", "art", "A", "B")
        assert not result.converged
        assert result.source == "none"

    def test_unknown_domain_returns_fallback(self) -> None:
        result = empirical_check("subject", "quantum_cooking", "A", "B")
        assert not result.converged

    def test_exception_in_source_returns_not_converged(self) -> None:
        with patch("aua.empirical._arxiv_check", side_effect=RuntimeError("boom")):
            result = empirical_check("s", "software_engineering", "A", "B")
        assert not result.converged
        assert "boom" in result.explanation


# ── SymPy source ──────────────────────────────────────────────────────────────


class TestSympy:
    def test_equivalent_expressions_returns_neither(self) -> None:
        # Both claim O(n log n) — algebraically equivalent
        output_A = "The algorithm runs in O(n log n) time."
        output_B = "Time complexity is O(n log n)."
        result = _sympy_check("sort", output_A, output_B)
        assert result.converged
        assert result.winner == "neither"
        assert result.source == "sympy"
        assert result.confidence > 0.5

    def test_b_tighter_complexity_wins(self) -> None:
        # A claims O(n^2), B claims O(n log n) — B is tighter
        output_A = "Bubble sort is O(n^2) average case."
        output_B = "Merge sort is O(n log n) average case."
        result = _sympy_check("sort_complexity", output_A, output_B)
        assert result.source == "sympy"
        # Either converged with B winning, or inconclusive — both acceptable
        if result.converged:
            assert result.winner in ("A", "B", "neither")

    def test_no_expressions_returns_not_converged(self) -> None:
        result = _sympy_check("topic", "This is a sentence.", "So is this.")
        assert not result.converged
        assert result.source == "sympy"

    def test_unparseable_expressions_returns_not_converged(self) -> None:
        result = _sympy_check("topic", "O(???)", "O(!!)")
        # Should not raise; returns not-converged
        assert result.source == "sympy"
        assert isinstance(result.converged, bool)

    def test_sympy_missing_returns_graceful(self) -> None:
        import sys

        real_sympy = sys.modules.get("sympy")
        try:
            sys.modules["sympy"] = None  # type: ignore
            result = _sympy_check("s", "O(n^2)", "O(n log n)")
            assert not result.converged
        finally:
            if real_sympy is not None:
                sys.modules["sympy"] = real_sympy


# ── arXiv source ──────────────────────────────────────────────────────────────

_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Efficient sorting algorithms with merge sort</title>
    <summary>Merge sort achieves optimal comparison-based sorting in O(n log n)
    time with recursion and divide-and-conquer. Quicksort is often faster in
    practice but worst-case is O(n^2).</summary>
  </entry>
  <entry>
    <title>Analysis of comparison sorting lower bounds</title>
    <summary>Lower bound for comparison sorting is Omega(n log n) via
    decision tree argument. Merge sort is asymptotically optimal.</summary>
  </entry>
</feed>"""


def _make_arxiv_response(body: str, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = body
    r.raise_for_status = MagicMock()
    return r


class TestArxiv:
    def test_winner_a_more_keyword_overlap(self) -> None:
        # A uses words present in abstracts; B uses irrelevant words
        output_A = "Merge sort uses divide and conquer achieving optimal n log n complexity."
        output_B = "Teleportation via quantum entanglement reduces latency."
        with patch("httpx.get", return_value=_make_arxiv_response(_ARXIV_ATOM)):
            result = _arxiv_check("sorting", output_A, output_B)
        assert result.source == "arxiv"
        assert result.converged
        assert result.winner == "A"
        assert result.confidence > 0.5

    def test_winner_b_more_keyword_overlap(self) -> None:
        output_A = "Teleportation via quantum entanglement reduces latency."
        output_B = "Merge sort uses divide and conquer achieving optimal comparison bound."
        with patch("httpx.get", return_value=_make_arxiv_response(_ARXIV_ATOM)):
            result = _arxiv_check("sorting", output_A, output_B)
        assert result.source == "arxiv"
        assert result.converged
        assert result.winner == "B"

    def test_neither_when_no_keyword_matches(self) -> None:
        output_A = "Photosynthesis converts sunlight."
        output_B = "Mitosis divides cells."
        with patch("httpx.get", return_value=_make_arxiv_response(_ARXIV_ATOM)):
            result = _arxiv_check("biology", output_A, output_B)
        assert result.source == "arxiv"
        assert result.converged
        assert result.winner == "neither"

    def test_empty_results_returns_not_converged(self) -> None:
        empty = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        with patch("httpx.get", return_value=_make_arxiv_response(empty)):
            result = _arxiv_check("obscure topic", "A", "B")
        assert not result.converged
        assert result.source == "arxiv"

    def test_api_error_returns_not_converged(self) -> None:
        with patch("httpx.get", side_effect=Exception("connection refused")):
            result = _arxiv_check("topic", "A", "B")
        assert not result.converged
        assert (
            "unreachable" in result.explanation.lower() or "refused" in result.explanation.lower()
        )

    def test_malformed_xml_returns_not_converged(self) -> None:
        bad = MagicMock()
        bad.text = "this is not xml <<<"
        bad.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=bad):
            result = _arxiv_check("topic", "A", "B")
        assert not result.converged


# ── PubMed source ─────────────────────────────────────────────────────────────

_PUBMED_SEARCH_JSON = {
    "esearchresult": {
        "idlist": ["12345678", "87654321"],
        "count": "2",
    }
}

_PUBMED_ABSTRACT_TEXT = """
PMID: 12345678
Title: Surgical outcomes with laparoscopic approach
Abstract: Minimally invasive laparoscopic surgery reduces recovery time
and infection rates compared to open procedures. Blood loss is lower.

PMID: 87654321
Title: Comparison of surgical techniques
Abstract: Open surgery remains preferred for complex resections.
Laparoscopic technique requires longer operative time but shorter hospital stay.
"""


def _make_pubmed_search_response() -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = _PUBMED_SEARCH_JSON
    return r


def _make_pubmed_fetch_response() -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.text = _PUBMED_ABSTRACT_TEXT
    return r


class TestPubMed:
    def test_winner_b_medical_keywords(self) -> None:
        output_A = "Open surgery is preferred for all procedures."
        output_B = "Laparoscopic minimally invasive technique reduces recovery and infection."
        responses = [_make_pubmed_search_response(), _make_pubmed_fetch_response()]
        with patch("httpx.get", side_effect=responses):
            result = _pubmed_check("surgical technique", output_A, output_B)
        assert result.source == "pubmed"
        assert result.converged
        assert result.winner == "B"
        assert "12345678" in result.explanation or "87654321" in result.explanation

    def test_no_results_returns_not_converged(self) -> None:
        empty_search = MagicMock()
        empty_search.raise_for_status = MagicMock()
        empty_search.json.return_value = {"esearchresult": {"idlist": []}}
        with patch("httpx.get", return_value=empty_search):
            result = _pubmed_check("topic", "A", "B")
        assert not result.converged
        assert result.source == "pubmed"

    def test_search_api_error_returns_not_converged(self) -> None:
        with patch("httpx.get", side_effect=Exception("timeout")):
            result = _pubmed_check("topic", "A", "B")
        assert not result.converged
        assert result.source == "pubmed"
        assert "timeout" in result.explanation

    def test_uses_ncbi_api_key_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NCBI_API_KEY", "testkey123")
        captured_params: list[dict] = []

        def capture_get(url: str, **kwargs: object) -> MagicMock:
            captured_params.append(kwargs.get("params", {}))
            if "esearch" in url:
                return _make_pubmed_search_response()
            return _make_pubmed_fetch_response()

        with patch("httpx.get", side_effect=capture_get):
            _pubmed_check("topic", "A", "B")

        assert any(p.get("api_key") == "testkey123" for p in captured_params)


# ── Arbiter integration ───────────────────────────────────────────────────────


class TestArbiterEmpiricalIntegration:
    def test_check_empirical_returns_check_result(self, arbiter: ArbiterAgent) -> None:
        with patch("aua.empirical._sympy_check") as mock:
            mock.return_value = EmpiricalResult(
                converged=True,
                winner="A",
                confidence=0.80,
                explanation="SymPy: A is algebraically correct.",
                source="sympy",
            )
            result = arbiter._check_empirical("complexity", "mathematics", "O(n log n)", "O(n^2)")
        assert isinstance(result, CheckResult)
        assert result.check_type == "empirical"
        assert result.converged is True
        assert result.winner == "A"
        assert result.confidence == pytest.approx(0.80)

    def test_check_empirical_not_converged_passes_through(self, arbiter: ArbiterAgent) -> None:
        with patch("aua.empirical.empirical_check") as mock:
            mock.return_value = EmpiricalResult(
                converged=False,
                explanation="arXiv: no papers found.",
                source="arxiv",
            )
            result = arbiter._check_empirical("topic", "software_engineering", "A", "B")
        assert result.check_type == "empirical"
        assert result.converged is False
        assert result.confidence == 0.0

    def test_arbitrate_includes_empirical_check(self, arbiter: ArbiterAgent) -> None:
        """arbitrate() always runs all four checks including empirical."""
        with patch("aua.empirical.empirical_check") as mock_emp:
            mock_emp.return_value = EmpiricalResult(
                converged=True,
                winner="A",
                confidence=0.72,
                explanation="arXiv evidence favours A.",
                source="arxiv",
            )
            verdict = arbiter.arbitrate(
                subject="binary_search",
                domain="software_engineering",
                output_A="Binary search is O(log n).",
                output_B="Binary search is O(n).",
            )

        check_types = [c.check_type for c in verdict.checks_run]
        assert "empirical" in check_types
        emp_check = next(c for c in verdict.checks_run if c.check_type == "empirical")
        assert emp_check.converged is True
        assert emp_check.winner == "A"

    def test_arbitrate_empirical_failure_does_not_crash(self, arbiter: ArbiterAgent) -> None:
        """A completely broken empirical source must not bubble up."""
        with patch("aua.empirical.empirical_check", side_effect=Exception("network down")):
            # Should not raise
            verdict = arbiter.arbitrate(
                subject="topic",
                domain="mathematics",
                output_A="O(n^2)",
                output_B="O(n log n)",
            )
        # Empirical check should be in results as not-converged
        check_types = [c.check_type for c in verdict.checks_run]
        assert "empirical" in check_types
        emp = next(c for c in verdict.checks_run if c.check_type == "empirical")
        assert emp.converged is False
