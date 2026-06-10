"""
tests/test_yaml_extensions.py — F-09/F-10/F-11: the YAML expert path.

Simulates exactly what tutorial How-to 13/14 tells a newcomer to do: write a
plugin file in the project directory, register it in aua_config.yaml, start
the router, and see it take effect — no AUA source edits.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml
from fastapi.testclient import TestClient

from aua.config import load_config
from aua.router import Router

PLUGIN_FILE = textwrap.dedent('''
    """Project-local extensions — created by following tutorial How-to 13/14."""


    class KeywordClassifier:
        """FieldClassifierPlugin: route 'integral' queries to mathematics."""

        def __init__(self, confidence_boost: float = 1.0):
            self.boost = float(confidence_boost)

        def classify(self, query: str) -> dict[str, float]:
            if "integral" in query.lower():
                return {"mathematics": min(1.0, 0.95 * self.boost)}
            return {"software_engineering": min(1.0, 0.90 * self.boost)}


    class HalfScorer:
        """UtilityScorerPlugin: halve the built-in U score."""

        def score(self, response, field, prior_u, confidence, metadata) -> float:
            return prior_u * 0.5


    class TagHook:
        """HookPlugin: record every pre_query event it sees."""

        seen: list[dict] = []

        async def __call__(self, event: dict) -> dict:
            TagHook.seen.append(dict(event))
            return event


    class ShoutMiddleware:
        """AUAMiddleware: uppercase the query in, tag the response out."""

        async def before_query(self, request: dict) -> dict:
            request["query"] = request["query"].upper()
            return request

        async def after_response(self, response: dict) -> dict:
            response["response"] = "[mw] " + response["response"]
            return response
    ''')


@pytest.fixture
def project_dir(tmp_path, fixtures_dir, monkeypatch):
    """A newcomer's project: config + plugins/ package, nothing on sys.path."""
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "__init__.py").write_text("")
    (tmp_path / "plugins" / "mine.py").write_text(PLUGIN_FILE)
    monkeypatch.chdir(tmp_path)
    base = yaml.safe_load((fixtures_dir / "aua_config_minimal.yaml").read_text())
    return tmp_path, base


def _write_and_load(tmp_path, raw):
    cfg_path = tmp_path / "aua_config.yaml"
    cfg_path.write_text(yaml.dump(raw))
    cfg = load_config(cfg_path)
    return Router.from_config(cfg, config_path=str(cfg_path))


def test_tutorial_config_blocks_now_load(project_dir):
    """The blocks documented in Parts 2/13/14/15 pass strict validation."""
    tmp_path, raw = project_dir
    raw["state"] = {"backend": "sqlite", "path": ".aua/state/aua.db"}
    raw["security"] = {"cors_origins": ["http://localhost:3001"]}
    raw["plugins"] = {"field_classifier": {"import_path": "plugins.mine:KeywordClassifier"}}
    raw["hooks"] = [{"hook_point": "pre_query", "import_path": "plugins.mine:TagHook"}]
    raw["middleware"] = ["plugins.mine:ShoutMiddleware"]
    router = _write_and_load(tmp_path, raw)
    assert router is not None


def test_field_classifier_plugin_routes_queries(project_dir, fake_swe_server):
    tmp_path, raw = project_dir
    raw["plugins"] = {
        "field_classifier": {
            "import_path": "plugins.mine:KeywordClassifier",
            "config": {"confidence_boost": 1.0},
        }
    }
    router = _write_and_load(tmp_path, raw)
    client = TestClient(router.app, raise_server_exceptions=True)
    body = client.post("/query", json={"query": "refactor this function"}).json()
    # The custom classifier sends everything non-math to software_engineering
    assert body["primary_domain"] == "software_engineering"
    assert body["domain_distribution"] == {"software_engineering": 0.9}


def test_utility_scorer_plugin_owns_final_u(project_dir, fake_swe_server):
    tmp_path, raw = project_dir
    router_plain = _write_and_load(tmp_path, dict(raw))
    client = TestClient(router_plain.app, raise_server_exceptions=True)
    u_builtin = client.post("/query", json={"query": "Write binary search"}).json()["u_score"]

    raw["plugins"] = {"utility_scorer": {"import_path": "plugins.mine:HalfScorer"}}
    router_custom = _write_and_load(tmp_path, raw)
    client2 = TestClient(router_custom.app, raise_server_exceptions=True)
    u_custom = client2.post("/query", json={"query": "Write binary search"}).json()["u_score"]
    assert u_custom == pytest.approx(u_builtin * 0.5, abs=0.01)


def test_hook_from_yaml_fires_with_ids(project_dir, fake_swe_server):
    from aua.hooks import reset_hook_runner

    reset_hook_runner()
    tmp_path, raw = project_dir
    raw["hooks"] = [{"hook_point": "pre_query", "import_path": "plugins.mine:TagHook"}]
    router = _write_and_load(tmp_path, raw)
    import plugins.mine as mine  # type: ignore[import-not-found]

    mine.TagHook.seen.clear()
    client = TestClient(router.app, raise_server_exceptions=True)
    client.post("/query", json={"query": "Write binary search", "session_id": "hooked"})
    assert mine.TagHook.seen, "pre_query hook fired"
    evt = mine.TagHook.seen[0]
    assert evt["session_id"] == "hooked" and evt["trace_id"] and evt["request_id"]
    reset_hook_runner()


def test_middleware_pipeline_rewrites_request_and_response(project_dir, fake_swe_server):
    tmp_path, raw = project_dir
    raw["middleware"] = ["plugins.mine:ShoutMiddleware"]
    router = _write_and_load(tmp_path, raw)
    client = TestClient(router.app, raise_server_exceptions=True)
    body = client.post("/query", json={"query": "write binary search"}).json()
    assert body["query"] == "WRITE BINARY SEARCH"  # before_query rewrote it
    assert body["response"].startswith("[mw] ")  # after_response tagged it


def test_security_cors_overrides_router_cors(project_dir):
    tmp_path, raw = project_dir
    raw["security"] = {"cors_origins": ["https://only-me.example"]}
    router = _write_and_load(tmp_path, raw)
    client = TestClient(router.app)
    r = client.options(
        "/query",
        headers={
            "Origin": "https://only-me.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "https://only-me.example"
    r2 = client.options(
        "/query",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert r2.headers.get("access-control-allow-origin") != "https://evil.example"


def test_state_block_controls_db_path(project_dir):
    tmp_path, raw = project_dir
    raw["state"] = {"backend": "sqlite", "path": "custom/dir/my.db"}
    router = _write_and_load(tmp_path, raw)
    assert (tmp_path / "custom" / "dir" / "my.db").exists()
    assert str(router._state_store._db_path).endswith("custom/dir/my.db")


def test_bad_import_path_fails_fast_with_clear_error(project_dir):
    from aua.plugins.registry import PluginLoadError

    tmp_path, raw = project_dir
    raw["plugins"] = {"field_classifier": {"import_path": "plugins.mine:Nope"}}
    with pytest.raises(PluginLoadError, match="Nope"):
        _write_and_load(tmp_path, raw)


def test_config_validation_rejects_typos(project_dir):
    tmp_path, raw = project_dir
    raw["plugins"] = {"feild_classifier": {"import_path": "plugins.mine:KeywordClassifier"}}
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.dump(raw))
    with pytest.raises(ValueError, match="feild_classifier"):
        load_config(cfg_path)
    raw2 = {k: v for k, v in project_dir[1].items() if k != "plugins"}
    raw2["hooks"] = [{"hook_point": "pre_qeury", "import_path": "plugins.mine:TagHook"}]
    cfg_path.write_text(yaml.dump(raw2))
    with pytest.raises(ValueError, match="pre_qeury"):
        load_config(cfg_path)
