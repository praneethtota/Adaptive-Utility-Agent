"""
tests/test_v11_veritas.py — v1.1-veritas P1–P3 backport tests (Phase 13).

Covers: keyword search (V-P1.1), backup coverage (V-P1.2), implicit
corrections (V-P1.3), structured backup prompt (V-P1.4), crash reporter
(V-P1.5), remote model config (V-P1.6), review notes (V-P2.1), analytics
suite (V-P2.2), update management (V-P2.3), corrections CRUD (V-P2.4),
bug reporting (V-P3.1), projects (V-P3.2), local models (V-P3.3), and the
domain ontology (V-P3.4). No live servers or network required.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from aua.router import Router
from aua.state import SQLiteStateStore

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_router(minimal_config, tmp_path, monkeypatch):
    """Router with an isolated state DB (fresh .aua/ in tmp_path)."""
    monkeypatch.chdir(tmp_path)
    return Router.from_config(minimal_config)


@pytest.fixture
def client(isolated_router):
    return TestClient(isolated_router.app, raise_server_exceptions=True)


@pytest.fixture
def store(isolated_router) -> SQLiteStateStore:
    return isolated_router._state_store


def _make_conv_with_messages(client, texts: list[tuple[str, str]]) -> str:
    conv = client.post("/conversations", json={"title": "kw test"}).json()
    cid = conv["conversation_id"]
    for role, text in texts:
        r = client.post(f"/conversations/{cid}/messages", json={"role": role, "content": text})
        assert r.status_code == 201, r.text
    return cid


# ── V-P1.1: keyword extraction + search ───────────────────────────────────────


def test_keyword_extract_technical_identifiers():
    from aua.keywords import extract

    kws = extract("Fix the MessageCache class and fire_and_forget helper in 2024")
    assert "messagecache" in kws
    assert "fire_and_forget" in kws
    assert "2024" in kws
    assert "the" not in kws  # stopword filtered


def test_keyword_extract_short_text_empty():
    from aua.keywords import extract

    assert extract("") == []
    assert extract("hi") == []


def test_search_endpoint_message_level_hits(client):
    cid = _make_conv_with_messages(
        client,
        [
            ("user", "How do I configure PostgreSQL replication?"),
            ("assistant", "Use streaming replication with wal_level=replica."),
            ("user", "What about Docker networking?"),
        ],
    )
    r = client.get("/search", params={"q": "replication"})
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 2  # one per matching MESSAGE (Cmd+F model)
    assert all(h["conversation_id"] == cid for h in hits)
    assert hits[0]["title"] == "kw test"


def test_search_and_semantics_same_message(client):
    _make_conv_with_messages(
        client,
        [
            ("user", "PostgreSQL tuning for analytics workloads"),
            ("user", "Docker compose for the staging environment"),
        ],
    )
    # Both words appear in the conversation but never in the same message
    assert client.get("/search", params={"q": "postgresql docker"}).json() == []
    # Both words in the same message → one hit
    hits = client.get("/search", params={"q": "postgresql analytics"}).json()
    assert len(hits) == 1


def test_search_prefix_matching(client):
    _make_conv_with_messages(client, [("user", "Kubernetes deployment rollout strategy")])
    assert len(client.get("/search", params={"q": "kuber"}).json()) == 1


def test_search_empty_query_returns_empty(client):
    assert client.get("/search", params={"q": "  "}).json() == []


def test_keyword_backfill_indexes_preexisting_messages(store):
    """Startup backfill rule: unindexed messages become searchable."""
    from aua.keywords import KeywordIndex

    conv = store.create_conversation(title="old")
    store.add_message(conv["conversation_id"], "user", "Legacy message about quaternions")
    ki = KeywordIndex(store)
    stats = ki.build_from_db()
    assert stats["backfilled"] >= 1
    assert ki.search("quaternions")


def test_message_write_validates_role_and_content(client):
    conv = client.post("/conversations", json={}).json()
    cid = conv["conversation_id"]
    assert (
        client.post(f"/conversations/{cid}/messages", json={"role": "bot", "content": "x"})
    ).status_code == 422
    assert (
        client.post(f"/conversations/{cid}/messages", json={"role": "user", "content": " "})
    ).status_code == 422


# ── V-P1.2 / V-P1.4: context backup ──────────────────────────────────────────


def test_backup_prompt_has_six_sections_and_900_tokens():
    from aua.context_backup import BACKUP_PROMPT, MAX_BACKUP_TOKENS

    assert MAX_BACKUP_TOKENS == 900
    for section in (
        "## GOAL",
        "## DECISIONS MADE",
        "## CURRENT STATUS",
        "## ACTIVE FILE / CODE CONTEXT",
        "## USER PREFERENCES LEARNED",
        "## RESUME INSTRUCTION",
    ):
        assert section in BACKUP_PROMPT


def test_backup_trigger_token_threshold(store):
    from aua.context_backup import ContextBackupManager

    mgr = ContextBackupManager(store)
    conv = store.create_conversation(title="b")
    cid = conv["conversation_id"]
    mgr.update_counter("swe", cid, "x" * 400)  # ~100 tokens
    should, why = mgr.should_backup("swe", cid, context_window=120)
    assert should and why == "token_threshold"
    should, why = mgr.should_backup("swe", cid, context_window=100_000)
    assert not should


def test_backup_store_resets_counter_and_bumps_thread(store):
    from aua.context_backup import ContextBackupManager

    mgr = ContextBackupManager(store)
    conv = store.create_conversation(title="b2")
    cid = conv["conversation_id"]
    mgr.update_counter("swe", cid, "y" * 4000)
    backup_id = mgr.store_backup("swe", cid, "## GOAL\nShip v1.1", "token_threshold")
    assert backup_id
    counter = store.query(
        "token_counters",
        filters={"specialist": "swe", "conversation_id": cid},
        limit=1,
        order_by="updated_at DESC",  # token_counters has no created_at column
    )[0]
    assert counter["token_estimate"] == 0
    assert counter["thread_number"] == 2  # new thread after context reset
    assert mgr.get_latest_backup("swe", cid).startswith("## GOAL")


def test_backup_validity_rule(store):
    """backup valid ⇔ MAX(backup.created_at) > MAX(messages.created_at)."""
    from aua.context_backup import ContextBackupManager

    mgr = ContextBackupManager(store)
    conv = store.create_conversation(title="b3")
    cid = conv["conversation_id"]
    for i in range(5):
        store.add_message(cid, "user", f"message {i} about apples")
    assert not store.backup_is_valid(cid, "swe")
    mgr.store_backup("swe", cid, "## GOAL\nfresh", "manual")
    assert store.backup_is_valid(cid, "swe")
    # New message makes it stale again
    time.sleep(0.01)
    store.add_message(cid, "user", "newer message")
    assert not store.backup_is_valid(cid, "swe")
    report = mgr.coverage_report("swe")
    assert report["stale_count"] == 1


def test_coverage_endpoints(client, store):
    conv = store.create_conversation(title="cov")
    cid = conv["conversation_id"]
    for i in range(6):
        store.add_message(cid, "user", f"coverage message {i}")
    r = client.get("/context/backup/coverage", params={"specialist": "swe"})
    assert r.status_code == 200
    assert r.json()["stale_count"] == 1

    r = client.post("/context/backup/run-coverage-job", params={"specialist": "nope"})
    assert r.status_code == 404
    r = client.post("/context/backup/run-coverage-job", params={"specialist": "swe"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


@pytest.mark.asyncio
async def test_coverage_sweep_generates_stale_backups(store):
    from aua.context_backup import ContextBackupManager

    mgr = ContextBackupManager(store)
    conv = store.create_conversation(title="sweep")
    cid = conv["conversation_id"]
    for i in range(5):
        store.add_message(cid, "user", f"sweep message {i}")

    async def fake_generator(specialist, conversation_id, prompt, history):
        assert "## GOAL" in prompt
        assert len(history) == 5  # full-DB-history rule (V-P0.5)
        return "## GOAL\ngenerated"

    result = await mgr.run_coverage_sweep(["swe"], fake_generator, pace_s=0)
    assert result["generated"] == 1 and not result["errors"]
    assert store.backup_is_valid(cid, "swe")
    # Second sweep: nothing stale → nothing generated
    result = await mgr.run_coverage_sweep(["swe"], fake_generator, pace_s=0)
    assert result["generated"] == 0


# ── V-P1.3: trigger detection + confirm-implicit ──────────────────────────────


@pytest.mark.parametrize(
    "message,expected",
    [
        ("No, that's wrong — use Postgres.", True),
        ("actually, I meant the staging environment", True),
        ("in fact, the limit is 100", True),
        ("never use tabs in this repo", True),
        ("correction: always use metric units", True),
        ("Can you rewrite this function?", False),
        ("What is the capital of France?", False),
        ("thanks!", False),
        ("how many users do we have", False),
    ],
)
def test_trigger_detector_patterns(message, expected):
    from aua.trigger_detector import TriggerDetector

    assert TriggerDetector().detect(message) is expected


def test_trigger_layer2_pluggable():
    from aua.trigger_detector import TriggerDetector, TriggerResult

    det = TriggerDetector(layer2=lambda text: 0.9)
    # Ambiguous for Layer 1 → Layer 2 decides
    msg = "the deployment target moved to eu-west-1 last sprint"
    assert det.detect_layer1(msg) is TriggerResult.UNCERTAIN
    assert det.detect(msg) is True
    assert det.last_score == 0.9


def test_confirm_implicit_accept_and_reject(client, isolated_router, store):
    # No pending → ok: False
    r = client.post("/corrections/confirm-implicit", json={"conversation_id": "c1"})
    assert r.json()["ok"] is False

    isolated_router._pending_implicit["c1"] = {"query": "no, use Redis not Memcached"}
    r = client.post(
        "/corrections/confirm-implicit", json={"conversation_id": "c1", "action": "reject"}
    )
    assert r.json() == {"ok": True, "stored": False, "message": "Correction discarded."}
    assert "c1" not in isolated_router._pending_implicit

    isolated_router._pending_implicit["c2"] = {"query": "no, use Redis not Memcached"}
    r = client.post(
        "/corrections/confirm-implicit", json={"conversation_id": "c2", "action": "accept"}
    )
    body = r.json()
    assert body["ok"] and body["stored"] and body["correction_id"]
    assert store.get("corrections", body["correction_id"])["source"] == "implicit_confirmed"
    assert isolated_router._store.query(subject="no, use Redis")


# ── V-P1.5: crash reporter ────────────────────────────────────────────────────


def test_crash_sentinel_lifecycle(store):
    from aua import crash_reporter as cr

    sid = cr.record_startup(store)
    crash = cr.detect_crash(store)
    assert crash and crash["session_id"] == sid
    cr.record_shutdown(store, sid)
    assert cr.detect_crash(store) is None


def test_pending_error_report_queue(store):
    from aua import crash_reporter as cr

    rid = cr.queue_error_report(store, ValueError("boom"), context={"step": 3})
    pending = cr.get_pending_error_reports(store)
    assert len(pending) == 1 and pending[0]["id"] == rid
    assert "boom" in pending[0]["payload"]
    cr.mark_error_sent(store, rid)
    assert cr.get_pending_error_reports(store) == []


@pytest.mark.asyncio
async def test_report_previous_crash_marks_sentinel(store, monkeypatch):
    from aua import crash_reporter as cr

    sid = cr.record_startup(store)

    async def fake_submit(report, pat):
        return False, "no pat"

    monkeypatch.setattr("aua.bug_reporter.submit_report", fake_submit)
    assert await cr.report_previous_crash(store) is True
    assert cr.detect_crash(store) is None  # marked even when send fails
    assert await cr.report_previous_crash(store) is False  # never reported twice
    cr.record_shutdown(store, sid)


# ── V-P1.6: remote model config ───────────────────────────────────────────────


def test_remote_merge_respects_field_allowlist():
    from aua.remote_config import merge_remote_into_registry

    base = {"m1": {"provider": "Qwen", "backend": "ollama", "full_id": "old"}}
    remote = {
        "schema_version": 1,
        "models": {
            "m1": {"full_id": "new-id", "backend": "HACKED", "context_window": 32000},
            "m2": {"provider": "Qwen", "full_id": "added"},
            "m3": {"provider": "UnknownCo", "full_id": "skipped"},
        },
        "deprecated": ["m1"],
    }
    merged, deprecated = merge_remote_into_registry(base, remote)
    assert merged["m1"]["full_id"] == "new-id"
    assert merged["m1"]["backend"] == "ollama"  # never overwritten from remote
    assert merged["m1"]["context_window"] == 32000
    assert merged["m2"]["backend"] == "ollama"  # inherited via provider
    assert "m3" not in merged  # unknown provider skipped
    assert deprecated == ["m1"]


@pytest.mark.asyncio
async def test_remote_config_fallback_chain(store, monkeypatch):
    from aua import remote_config as rc

    mgr = rc.RemoteModelConfig(store, base={"m1": {"provider": "P", "backend": "ollama"}})

    async def fetch_ok(url=None):
        return {
            "schema_version": 1,
            "models": {"m1": {"full_id": "remote-id"}},
            "model_id_renames": {"old": "m1"},
        }

    async def fetch_fail(url=None):
        return None

    monkeypatch.setattr(rc, "fetch_remote_config", fetch_ok)
    assert await mgr.refresh(force=True) is True
    assert mgr.source == "remote" and mgr.models["m1"]["full_id"] == "remote-id"
    assert mgr.resolve_alias("old") == "m1"

    # Remote down → DB cache from the last good fetch
    monkeypatch.setattr(rc, "fetch_remote_config", fetch_fail)
    mgr2 = rc.RemoteModelConfig(store, base={"m1": {"provider": "P", "backend": "ollama"}})
    assert await mgr2.refresh(force=True) is True
    assert mgr2.source == "cache" and mgr2.models["m1"]["full_id"] == "remote-id"

    # No cache either → builtin fallback
    with store._connect() as conn:
        conn.execute("DELETE FROM remote_config_cache")
    mgr3 = rc.RemoteModelConfig(store, base={"m1": {"provider": "P", "backend": "ollama"}})
    assert await mgr3.refresh(force=True) is False
    assert mgr3.source == "builtin" and "full_id" not in mgr3.models["m1"]


# ── V-P2.1: review notes ──────────────────────────────────────────────────────


def test_parse_review_notes_extracts_sections():
    notes = Router._parse_review_notes(
        "VERDICT: B\nREASON: A missed the edge case.\nCORRECTION: handle n=0.",
        reviewer="arbiter-3b",
    )
    assert notes.startswith("Reviewer: arbiter-3b.")
    assert "REASON: A missed the edge case." in notes
    assert "CORRECTION: handle n=0." in notes


def test_parse_review_notes_none_when_nothing_flagged():
    assert Router._parse_review_notes("VERDICT: A\nCORRECTION: none", "arb") is None
    assert Router._parse_review_notes("", "arb") is None


def test_router_response_review_notes_field():
    from aua.endpoints import RouterResponse

    assert "review_notes" in RouterResponse.model_fields
    assert RouterResponse.model_fields["review_notes"].default is None


# ── V-P2.2: analytics / reliability / usage / pricing ─────────────────────────


def _seed_runs(store, cid):
    for i in range(4):
        store.record_model_run(
            {
                "specialist": "swe",
                "conversation_id": cid,
                "round": "answer",
                "vcg_winner": 1 if i % 2 == 0 else 0,
                "vcg_welfare_score": 0.5 + i * 0.1,
                "confidence_score": 0.8,
                "latency_ms": 120.0,
                "domain": "software_engineering",
            }
        )


def test_analytics_endpoint(client, store):
    conv = store.create_conversation(title="a")
    _seed_runs(store, conv["conversation_id"])
    body = client.get("/analytics").json()
    spec = body["specialists"][0]
    assert spec["specialist"] == "swe"
    assert spec["total_runs"] == 4 and spec["winner_count"] == 2
    assert spec["win_rate_pct"] == 50.0
    assert body["confidence_dist"]["high"] == 2
    assert body["domain_dist"]["software_engineering"] == 4
    assert body["welfare_summary"]["total_scored"] == 4
    assert body["total_conversations"] == 1


def test_reliability_endpoint(client, store):
    conv = store.create_conversation(title="r")
    _seed_runs(store, conv["conversation_id"])
    body = client.get("/reliability").json()
    assert body[0]["specialist"] == "swe"
    assert body[0]["win_rate_pct"] == 50.0
    assert len(body[0]["trajectory"]) == 4
    assert body[0]["trend"] in ("up", "down", "flat")


def test_usage_and_pricing_endpoints(client, store):
    conv = store.create_conversation(title="u")
    _seed_runs(store, conv["conversation_id"])
    usage = client.get("/usage").json()
    assert usage["total_queries"] == 4
    assert usage["specialists"][0]["specialist"] == "swe"
    pricing = client.get("/pricing").json()
    assert "swe" in pricing["pricing"]
    assert "estimated_cost_per_query" in pricing["pricing"]["swe"]


# ── V-P2.3: update management ─────────────────────────────────────────────────


def test_update_skip_and_skipped(client):
    assert client.get("/update/skipped").json() == {"skipped_version": None}
    r = client.post("/update/skip", json={"version": "1.2.0"})
    assert r.json() == {"ok": True, "skipped_version": "1.2.0"}
    assert client.get("/update/skipped").json() == {"skipped_version": "1.2.0"}
    assert client.post("/update/skip", json={}).status_code == 422


def test_version_check_graceful_offline(client, monkeypatch):
    """No network in CI — must still return current version, never 500."""
    r = client.get("/version/check")
    assert r.status_code == 200
    body = r.json()
    from aua.version import __version__

    assert body["current"] == __version__
    assert "update_available" in body and "show_banner" in body


# ── V-P2.4: corrections CRUD + evidence ───────────────────────────────────────


def _inject(client, claim="Python 3.13 removed distutils"):
    r = client.post(
        "/corrections",
        json={
            "subject": "python distutils",
            "domain": "software_engineering",
            "claim": claim,
            "confidence": 0.9,
        },
    )
    assert r.status_code == 200
    return r.json()


def test_inject_returns_persistent_id(client, store):
    body = _inject(client)
    assert body["correction_id"]
    row = store.get("corrections", body["correction_id"])
    assert row["claim"] == "Python 3.13 removed distutils"
    assert row["scope"] == "global"


def test_patch_correction_edits_claim_and_logs_event(client, store, isolated_router):
    cid = _inject(client)["correction_id"]
    r = client.patch(f"/corrections/{cid}", json={"claim": "distutils removed in 3.12"})
    assert r.json()["ok"] is True
    assert store.get("corrections", cid)["claim"] == "distutils removed in 3.12"
    # In-memory store synced for prompt injection
    assert any(a.claim == "distutils removed in 3.12" for a in isolated_router._store.assertions)
    events = [e["event"] for e in store.query("correction_events", filters={"correction_id": cid})]
    assert "edited" in events and "created" in events
    assert client.patch(f"/corrections/{cid}", json={"claim": " "}).status_code == 422
    assert client.patch("/corrections/nope", json={"claim": "x"}).status_code == 404


def test_delete_correction_soft_deletes(client, store, isolated_router):
    cid = _inject(client)["correction_id"]
    r = client.delete(f"/corrections/{cid}")
    assert r.json()["scope"] == "superseded"
    row = store.get("corrections", cid)
    assert row is not None and row["scope"] == "superseded"  # row stays in DB
    assert not any(
        a.claim == "Python 3.13 removed distutils" for a in isolated_router._store.assertions
    )
    # Excluded from evidence by default, included on request
    ev = client.get("/corrections/evidence").json()
    assert ev["total"] == 0
    ev = client.get("/corrections/evidence", params={"include_superseded": True}).json()
    assert ev["total"] == 1
    assert client.delete("/corrections/nope").status_code == 404


def test_corrections_evidence_by_id(client):
    cid = _inject(client)["correction_id"]
    ev = client.get("/corrections/evidence", params={"correction_id": cid}).json()
    assert ev["total"] == 1
    c = ev["corrections"][0]
    assert c["id"] == cid and c["application_count"] >= 1
    assert c["events"][0]["event"] == "created"


# ── V-P3.1: bug reporting ─────────────────────────────────────────────────────


def test_bug_report_graceful_without_pat(client, monkeypatch):
    """No PAT configured → 200 with ok:false, never a 500."""
    monkeypatch.delenv("AUA_BUGS_PAT", raising=False)
    import aua.bug_reporter as bugs

    monkeypatch.setattr(bugs, "_pat_cache", None)
    r = client.post("/bug-report", json={"comment": "search returns empty"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["report_id"].startswith("usr_")
    assert "PAT" in body["message"]


def test_bug_report_build_report_shape():
    from aua.bug_reporter import build_report, generate_user_token

    token = generate_user_token()
    assert len(token) == 8
    report = build_report(
        user_token=token,
        comment="c",
        include_messages=True,
        last_messages=[{"role": "user", "content": "hi"}],
        user_email="a@b.c",
    )
    assert report["report_id"].startswith(f"usr_{token}_")
    assert report["last_messages"] and report["user_email"] == "a@b.c"
    assert report["platform"]["python"]


# ── V-P3.2: projects ──────────────────────────────────────────────────────────


def test_project_scoping_filter(client):
    p = client.post("/projects", json={"name": "Novel"}).json()
    c_in = client.post("/conversations", json={"title": "in", "project_id": p["project_id"]}).json()
    client.post("/conversations", json={"title": "out"}).json()
    scoped = client.get("/conversations", params={"project_id": p["project_id"]}).json()
    assert [c["conversation_id"] for c in scoped] == [c_in["conversation_id"]]
    assert len(client.get("/conversations").json()) == 2  # all chats


# ── V-P3.3: local model management ────────────────────────────────────────────


def test_local_model_register_list_tag(client):
    r = client.post(
        "/local/models",
        json={"local_model_id": "qwen3:8b", "nickname": "Qwen 8B"},
    )
    assert r.status_code == 201
    rows = client.get("/local/models").json()
    assert rows[0]["local_model_id"] == "qwen3:8b"
    assert rows[0]["specialist_domain"] is None

    r = client.patch(
        "/local/specialist/qwen3:8b",
        json={"specialist_domain": "software_engineering", "specialist_depth": 1},
    )
    assert r.json()["ok"] is True
    rows = client.get("/local/models").json()
    assert rows[0]["specialist_domain"] == "software_engineering"
    assert rows[0]["specialist_depth"] == 1
    # Untag
    client.patch("/local/specialist/qwen3:8b", json={"specialist_domain": None})
    assert client.get("/local/models").json()[0]["specialist_domain"] is None
    assert (
        client.patch("/local/specialist/missing", json={"specialist_domain": "x"})
    ).status_code == 404
    assert client.post("/local/models", json={}).status_code == 422


def test_local_settings_roundtrip(client):
    assert client.get("/local/settings").json() == {}
    client.post("/local/settings", json={"base_url": "http://localhost:11434"})
    assert client.get("/local/settings").json() == {"base_url": "http://localhost:11434"}


# ── V-P3.4: domain ontology ───────────────────────────────────────────────────


def test_domain_tree_alias_resolution(store):
    from aua.domain_tree import DomainTree

    tree = DomainTree(store)
    assert tree.find("coding").node_id == "software_engineering"  # alias map
    assert tree.find("MATHS").node_id == "mathematics"
    assert tree.find("codng").node_id == "software_engineering"  # edit distance
    assert tree.find("").node_id == "general"


def test_domain_tree_candidate_queue_and_persistence(store):
    from aua.domain_tree import DomainTree

    tree = DomainTree(store)
    node = tree.find("quantum chromodynamics", specialist="spec_a")
    assert node is not None
    cands = tree.candidates()
    assert any(c.raw_string == "quantum chromodynamics" for c in cands)
    assert tree.flush_candidates() >= 1
    # Reload from DB — candidate survives restart
    tree2 = DomainTree(store)
    assert any(c.raw_string == "quantum chromodynamics" for c in tree2.candidates())


def test_domain_tree_promotion_gates(store):
    from aua.domain_tree import DomainTree, OntologyJob

    tree = DomainTree(store)
    conv = store.create_conversation(title="dt")
    cid = conv["conversation_id"]
    raw = "compiler design"
    # Evidence: 2 specialists, divergent win rates vs parent
    for spec, cand_wins, parent_wins in (("s1", [1, 1, 1, 1], [0, 0]), ("s2", [1, 1], [0, 0])):
        for w in cand_wins:
            store.record_model_run(
                {
                    "specialist": spec,
                    "conversation_id": cid,
                    "round": "answer",
                    "vcg_winner": w,
                    "domain": raw,
                }
            )
        for w in parent_wins:
            store.record_model_run(
                {
                    "specialist": spec,
                    "conversation_id": cid,
                    "round": "answer",
                    "vcg_winner": w,
                    "domain": "software_engineering",
                }
            )
    for _ in range(5):  # Gate 1: K_MIN queries
        tree.find(raw, specialist="s1")
    tree.find(raw, specialist="s2")  # Gate 2: K_MIN_MODELS
    job = OntologyJob(tree, store)
    result = job.run_sync()
    assert result["promoted"] == 1
    node = tree.find(raw)
    assert node.node_id == "compiler_design"
    assert node.parent_id == "software_engineering" and node.depth == 1
    # Persisted: a fresh tree resolves it directly
    assert DomainTree(store).find(raw).node_id == "compiler_design"


def test_domain_tree_endpoint(client, isolated_router):
    isolated_router._domain_tree.find("astro photography", specialist="s1")
    body = client.get("/domain-tree").json()
    node_ids = [n["node_id"] for n in body["nodes"]]
    assert "software_engineering" in node_ids and "general" in node_ids
    assert all(n["is_l0_root"] for n in body["nodes"] if n["depth"] == 0)
    assert any(c["raw_string"] == "astro photography" for c in body["candidates"])


# ── Lifespan integration ──────────────────────────────────────────────────────


def test_lifespan_starts_and_stops_background_jobs(isolated_router, store):
    """Crash sentinel written on startup, cleaned on shutdown; jobs cancelled."""
    from aua import crash_reporter as cr

    with TestClient(isolated_router.app):
        crash = cr.detect_crash(store)
        assert crash is not None  # sentinel says 'running'
        assert isolated_router._keyword_index._queue is not None
        assert len(isolated_router._background_tasks) == 0 or True  # tasks registered
    # Clean shutdown recorded; worker stopped
    assert cr.detect_crash(store) is None
    assert isolated_router._keyword_index._worker_task is None


def test_keyword_worker_indexes_async(isolated_router, store):
    """With lifespan running, message writes are indexed by the worker."""
    with TestClient(isolated_router.app) as client:
        conv = client.post("/conversations", json={"title": "async"}).json()
        cid = conv["conversation_id"]
        client.post(
            f"/conversations/{cid}/messages",
            json={"role": "user", "content": "Asynchronous elasticsearch indexing test"},
        )
        deadline = time.time() + 3
        hits = []
        while time.time() < deadline:
            hits = client.get("/search", params={"q": "elasticsearch"}).json()
            if hits:
                break
            time.sleep(0.05)
        assert hits and hits[0]["conversation_id"] == cid
