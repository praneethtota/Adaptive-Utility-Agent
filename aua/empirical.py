"""
aua/empirical.py — External ground-truth integrations for Arbiter Stage 4 (#61).

Three sources, selected by domain:

  SymPy     — mathematics / structural_engineering
              Extracts and evaluates symbolic expressions from both outputs.
              Uses sympy.simplify() to test algebraic equivalence, and
              sympy.limit() / sympy.solve() for calculus and equation claims.

  arXiv     — software_engineering / stem_research / education / general
              Keyword search via the arXiv Atom API (no key required).
              Extracts technical claims from outputs and checks whether
              retrieved abstracts corroborate one side over the other.

  PubMed    — surgery / aviation / medicine
              NCBI E-utilities (no key required for low-volume use; set
              NCBI_API_KEY env var for >3 req/s).
              Searches by MeSH-style terms extracted from the outputs and
              compares abstract evidence.

  Fallback  — law / art / creative_writing / unknown domains
              Returns not-converged with a clear explanation. Law has no
              reliable free structured API; creative domains have no ground
              truth concept.

Architecture
────────────
All sources share the same interface:

    result: EmpiricalResult = source.check(subject, output_A, output_B)

EmpiricalResult carries:
    converged     — bool: did we find usable evidence?
    winner        — "A" | "B" | "neither" | None
    confidence    — 0.0–1.0: how strongly evidence favours the winner
    explanation   — human-readable summary of what was found
    source        — "sympy" | "arxiv" | "pubmed" | "none"

Timeout policy: every HTTP call is bounded by EXTERNAL_TIMEOUT_S (5 s).
On any network or parse error the source returns not-converged (never raises),
so a flaky external API never breaks the arbitration pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

EXTERNAL_TIMEOUT_S = 5.0  # hard cap on any single external call

# ── Domain → source routing ───────────────────────────────────────────────────

_DOMAIN_SOURCE: dict[str, str] = {
    "mathematics": "sympy",
    "structural_engineering": "sympy",
    "software_engineering": "arxiv",
    "stem_research": "arxiv",
    "education": "arxiv",
    "general": "arxiv",
    "surgery": "pubmed",
    "aviation": "pubmed",
    "medicine": "pubmed",
    # law / art / creative_writing → fallback (no reliable free source)
}


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class EmpiricalResult:
    converged: bool = False
    winner: str | None = None  # "A" | "B" | "neither" | None
    confidence: float = 0.0
    explanation: str = ""
    source: str = "none"
    raw: dict[str, Any] = field(default_factory=dict)


# ── Public entry point ────────────────────────────────────────────────────────


def empirical_check(
    subject: str,
    domain: str,
    output_A: str,
    output_B: str,
) -> EmpiricalResult:
    """
    Run the appropriate external ground-truth check for this domain.

    Never raises — all errors are caught and returned as not-converged.
    """
    source_name = _DOMAIN_SOURCE.get(domain, "none")
    try:
        if source_name == "sympy":
            return _sympy_check(subject, output_A, output_B)
        if source_name == "arxiv":
            return _arxiv_check(subject, output_A, output_B)
        if source_name == "pubmed":
            return _pubmed_check(subject, output_A, output_B)
    except Exception as e:
        log.debug("empirical_check error domain=%s source=%s: %s", domain, source_name, e)
        return EmpiricalResult(
            explanation=f"External check failed ({source_name}): {e}",
            source=source_name,
        )

    # Fallback domains — law, art, creative_writing
    domain_notes = {
        "law": "No reliable free structured API for legal ground truth. Manual review recommended.",
        "art": "No ground truth concept applies to artistic domains.",
        "creative_writing": "No ground truth concept applies to creative writing domains.",
    }
    note = domain_notes.get(domain, f"No external source configured for domain '{domain}'.")
    return EmpiricalResult(explanation=note, source="none")


# ── SymPy source ──────────────────────────────────────────────────────────────

# Patterns for extracting expressions from text
_COMPLEXITY_RE = re.compile(r"O\s*\(\s*([^)]+)\s*\)", re.IGNORECASE)
_EQUATION_RE = re.compile(r"([a-zA-Z0-9_^*/+\-\s]+)\s*=\s*([a-zA-Z0-9_^*/+\-\s]+)")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _extract_sympy_exprs(text: str) -> list[str]:
    """Extract candidate symbolic expressions from a text output."""
    exprs: list[str] = []
    # Big-O complexity claims
    for m in _COMPLEXITY_RE.finditer(text):
        exprs.append(m.group(1).strip())
    # Equations
    for m in _EQUATION_RE.finditer(text):
        exprs.append(f"{m.group(1).strip()} - ({m.group(2).strip()})")
    return exprs[:5]  # limit to avoid runaway


def _sympy_check(subject: str, output_A: str, output_B: str) -> EmpiricalResult:
    """
    Use SymPy to verify symbolic claims.

    Strategy:
      1. Extract complexity expressions from both outputs (e.g. "n log n", "n^2")
      2. Parse with sympy.sympify()
      3. Use sympy.simplify(A_expr - B_expr) == 0 to test equivalence
      4. If they differ, use sympy's ordering to determine which is smaller
         (lower complexity = correct for typical algorithmic claims)
    """
    try:
        import sympy
    except ImportError:
        return EmpiricalResult(
            explanation="sympy not installed — run: pip install sympy",
            source="sympy",
        )

    exprs_A = _extract_sympy_exprs(output_A)
    exprs_B = _extract_sympy_exprs(output_B)

    if not exprs_A and not exprs_B:
        return EmpiricalResult(
            explanation="No symbolic expressions found in either output.",
            source="sympy",
        )

    n = sympy.Symbol("n", positive=True)
    parsed_A: list[Any] = []
    parsed_B: list[Any] = []

    for expr_str in exprs_A:
        try:
            # normalise: "n log n" → "n*log(n)", "n^2" → "n**2"
            cleaned = expr_str.replace("^", "**").replace(" log ", "*log")
            parsed_A.append(sympy.sympify(cleaned, locals={"n": n, "log": sympy.log}))
        except Exception:
            pass

    for expr_str in exprs_B:
        try:
            cleaned = expr_str.replace("^", "**").replace(" log ", "*log")
            parsed_B.append(sympy.sympify(cleaned, locals={"n": n, "log": sympy.log}))
        except Exception:
            pass

    if not parsed_A or not parsed_B:
        return EmpiricalResult(
            explanation="Could not parse symbolic expressions from one or both outputs.",
            source="sympy",
        )

    # Compare first valid expressions from each
    expr_A = parsed_A[0]
    expr_B = parsed_B[0]

    try:
        diff = sympy.simplify(expr_A - expr_B)
        if diff == 0:
            return EmpiricalResult(
                converged=True,
                winner="neither",
                confidence=0.85,
                explanation=f"SymPy: both expressions are algebraically equivalent ({expr_A}).",
                source="sympy",
            )
        # Try to determine which is asymptotically smaller (for complexity claims)
        try:
            limit_diff = sympy.limit(diff / sympy.log(n), n, sympy.oo)
            if limit_diff > 0:
                # A grows faster than B → B is the tighter (correct) bound
                winner = "B"
                explanation = (
                    f"SymPy: A claims {expr_A}, B claims {expr_B}. "
                    f"B is asymptotically tighter (limit of difference / log(n) → {limit_diff})."
                )
            elif limit_diff < 0:
                winner = "A"
                explanation = (
                    f"SymPy: A claims {expr_A}, B claims {expr_B}. " f"A is asymptotically tighter."
                )
            else:
                winner = "neither"
                explanation = (
                    f"SymPy: expressions differ ({expr_A} vs {expr_B}) "
                    "but asymptotic ordering is inconclusive."
                )
            return EmpiricalResult(
                converged=True,
                winner=winner,
                confidence=0.80,
                explanation=explanation,
                source="sympy",
            )
        except Exception:
            return EmpiricalResult(
                converged=True,
                winner="neither",
                confidence=0.60,
                explanation=(
                    f"SymPy: expressions differ ({expr_A} vs {expr_B}) "
                    "but ordering could not be determined."
                ),
                source="sympy",
            )
    except Exception as e:
        return EmpiricalResult(
            explanation=f"SymPy simplification failed: {e}",
            source="sympy",
        )


# ── arXiv source ──────────────────────────────────────────────────────────────

_ARXIV_API = "https://export.arxiv.org/api/query"
_MAX_ARXIV_RESULTS = 3


def _extract_keywords(text: str, n: int = 6) -> list[str]:
    """Extract distinctive technical keywords from output text."""
    # Strip common stop words, keep tokens ≥ 4 chars that aren't pure numbers
    stop = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "been",
        "will",
        "also",
        "they",
        "their",
        "then",
        "than",
        "when",
        "which",
        "would",
        "could",
        "should",
        "there",
        "these",
        "those",
        "some",
        "into",
        "about",
        "more",
        "most",
        "other",
        "such",
        "each",
        "both",
        "only",
        "very",
        "just",
        "even",
        "after",
        "before",
        "because",
        "through",
        "where",
        "output",
        "answer",
        "result",
        "response",
        "correct",
        "incorrect",
    }
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for t in tokens:
        if t not in stop and t not in seen:
            seen.add(t)
            keywords.append(t)
        if len(keywords) >= n:
            break
    return keywords


def _arxiv_check(subject: str, output_A: str, output_B: str) -> EmpiricalResult:
    """
    Search arXiv for papers relevant to the subject, compare abstract evidence.

    Strategy:
      1. Build a query from subject + keywords from both outputs
      2. Fetch top-3 abstracts via the Atom API
      3. Count keyword overlap between each abstract and each output
      4. Winner is whichever output's keywords appear more in the abstracts
    """
    kw_A = set(_extract_keywords(output_A))
    kw_B = set(_extract_keywords(output_B))
    subject_kw = re.sub(r"[^a-z0-9 ]", " ", subject.lower()).split()[:4]

    query_terms = subject_kw + list(kw_A | kw_B)[:8]
    query = " AND ".join(f"ti:{t}" for t in query_terms[:6]) if query_terms else subject

    try:
        resp = httpx.get(
            _ARXIV_API,
            params={
                "search_query": query,
                "max_results": _MAX_ARXIV_RESULTS,
                "sortBy": "relevance",
            },
            timeout=EXTERNAL_TIMEOUT_S,
        )
        resp.raise_for_status()
    except Exception as e:
        return EmpiricalResult(
            explanation=f"arXiv API unreachable: {e}",
            source="arxiv",
        )

    try:
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
    except ET.ParseError as e:
        return EmpiricalResult(explanation=f"arXiv response parse error: {e}", source="arxiv")

    if not entries:
        return EmpiricalResult(
            explanation=f"arXiv: no papers found for query '{query}'.",
            source="arxiv",
        )

    abstracts: list[str] = []
    titles: list[str] = []
    for entry in entries:
        summary = entry.find("atom:summary", ns)
        title = entry.find("atom:title", ns)
        if summary is not None and summary.text:
            abstracts.append(summary.text.lower())
        if title is not None and title.text:
            titles.append(title.text.strip())

    combined = " ".join(abstracts)

    # Score by keyword overlap
    score_A = sum(1 for kw in kw_A if kw in combined)
    score_B = sum(1 for kw in kw_B if kw in combined)

    if score_A == 0 and score_B == 0:
        return EmpiricalResult(
            converged=True,
            winner="neither",
            confidence=0.40,
            explanation=(
                f"arXiv: found {len(abstracts)} paper(s) but neither output's keywords "
                f"matched abstracts. Titles: {'; '.join(titles[:2])}."
            ),
            source="arxiv",
            raw={"titles": titles, "score_A": score_A, "score_B": score_B},
        )

    total = score_A + score_B
    if score_A > score_B:
        winner = "A"
        confidence = min(0.75, 0.5 + 0.25 * (score_A - score_B) / total)
    elif score_B > score_A:
        winner = "B"
        confidence = min(0.75, 0.5 + 0.25 * (score_B - score_A) / total)
    else:
        winner = "neither"
        confidence = 0.45

    return EmpiricalResult(
        converged=True,
        winner=winner,
        confidence=confidence,
        explanation=(
            f"arXiv: {len(abstracts)} paper(s) retrieved. "
            f"Keyword overlap — A: {score_A}, B: {score_B}. "
            f"Papers: {'; '.join(titles[:2])}."
        ),
        source="arxiv",
        raw={"titles": titles, "score_A": score_A, "score_B": score_B},
    )


# ── PubMed source ─────────────────────────────────────────────────────────────

_PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_MAX_PUBMED_RESULTS = 3


def _pubmed_check(subject: str, output_A: str, output_B: str) -> EmpiricalResult:
    """
    Search PubMed for clinical/medical evidence relevant to the subject.

    Uses NCBI E-utilities (no key needed for ≤3 req/s; set NCBI_API_KEY for more).
    Strategy mirrors arXiv: keyword overlap between abstracts and each output.
    """
    api_key = os.environ.get("NCBI_API_KEY", "")
    kw_A = set(_extract_keywords(output_A))
    kw_B = set(_extract_keywords(output_B))
    subject_terms = re.sub(r"[^a-z0-9 ]", " ", subject.lower()).split()[:4]
    query = " AND ".join(subject_terms + list(kw_A | kw_B)[:4])

    params: dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": _MAX_PUBMED_RESULTS,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    # Step 1: search for IDs
    try:
        search_resp = httpx.get(_PUBMED_SEARCH, params=params, timeout=EXTERNAL_TIMEOUT_S)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        ids = search_data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        return EmpiricalResult(explanation=f"PubMed search failed: {e}", source="pubmed")

    if not ids:
        return EmpiricalResult(
            explanation=f"PubMed: no articles found for '{query}'.",
            source="pubmed",
        )

    # Step 2: fetch abstracts
    fetch_params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(ids),
        "rettype": "abstract",
        "retmode": "text",
    }
    if api_key:
        fetch_params["api_key"] = api_key

    try:
        fetch_resp = httpx.get(_PUBMED_FETCH, params=fetch_params, timeout=EXTERNAL_TIMEOUT_S)
        fetch_resp.raise_for_status()
        combined = fetch_resp.text.lower()
    except Exception as e:
        return EmpiricalResult(explanation=f"PubMed fetch failed: {e}", source="pubmed")

    score_A = sum(1 for kw in kw_A if kw in combined)
    score_B = sum(1 for kw in kw_B if kw in combined)

    if score_A == 0 and score_B == 0:
        return EmpiricalResult(
            converged=True,
            winner="neither",
            confidence=0.40,
            explanation=f"PubMed: {len(ids)} article(s) found but no keyword matches. IDs: {ids}.",
            source="pubmed",
            raw={"pmids": ids, "score_A": score_A, "score_B": score_B},
        )

    total = score_A + score_B
    if score_A > score_B:
        winner = "A"
        confidence = min(0.75, 0.5 + 0.25 * (score_A - score_B) / total)
    elif score_B > score_A:
        winner = "B"
        confidence = min(0.75, 0.5 + 0.25 * (score_B - score_A) / total)
    else:
        winner = "neither"
        confidence = 0.45

    return EmpiricalResult(
        converged=True,
        winner=winner,
        confidence=confidence,
        explanation=(
            f"PubMed: {len(ids)} article(s). "
            f"Keyword overlap — A: {score_A}, B: {score_B}. PMIDs: {ids}."
        ),
        source="pubmed",
        raw={"pmids": ids, "score_A": score_A, "score_B": score_B},
    )
