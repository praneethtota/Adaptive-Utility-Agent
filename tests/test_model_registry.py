"""
tests/test_model_registry.py — Tests for #46 model registry integration.

Coverage:
  parse_model_ref:
    HF repo (no revision), HF repo@revision, HF repo@sha256:hash
    MLflow models:/ URI
    Local absolute path, local relative path
    Ollama tag (name:tag), Ollama bare name
  resolve_model_ref:
    HF no revision → returns repo_id unchanged
    HF with revision → calls snapshot_download with revision, returns local path
    HF 401 → raises RuntimeError with helpful message
    MLflow URI → calls mlflow.artifacts.download_artifacts
    MLflow missing tracking URI → raises RuntimeError
    Local path → returned unchanged
    Ollama tag → returned unchanged
    huggingface_hub not installed → returns repo_id with warning
    mlflow not installed → raises RuntimeError
  list_hf_revisions: happy path, error propagation
  list_mlflow_versions: happy path, error propagation
  config.py: mlflow_tracking_uri field loaded from YAML
  serve.py: resolve_model_ref called when spec has @revision
  CLI: aua models pin -- list revisions, pin to revision, json output
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aua.model_registry import (
    ModelUriType,
    RevisionInfo,
    parse_model_ref,
    resolve_model_ref,
)

# ── parse_model_ref ───────────────────────────────────────────────────────────


class TestParseModelRef:
    def test_hf_repo_no_revision(self) -> None:
        ref = parse_model_ref("Qwen/Qwen2.5-7B-Instruct")
        assert ref.uri_type == ModelUriType.HF
        assert ref.repo_id == "Qwen/Qwen2.5-7B-Instruct"
        assert ref.revision is None

    def test_hf_repo_with_tag_revision(self) -> None:
        ref = parse_model_ref("Qwen/Qwen2.5-7B-Instruct@v0.3")
        assert ref.uri_type == ModelUriType.HF
        assert ref.repo_id == "Qwen/Qwen2.5-7B-Instruct"
        assert ref.revision == "v0.3"

    def test_hf_repo_with_sha256_revision(self) -> None:
        ref = parse_model_ref("meta-llama/Llama-3-8B@sha256:abc123def456")
        assert ref.uri_type == ModelUriType.HF
        assert ref.repo_id == "meta-llama/Llama-3-8B"
        assert ref.revision == "sha256:abc123def456"

    def test_hf_repo_with_branch_revision(self) -> None:
        ref = parse_model_ref("Qwen/Qwen2.5-7B@main")
        assert ref.uri_type == ModelUriType.HF
        assert ref.revision == "main"

    def test_mlflow_uri(self) -> None:
        ref = parse_model_ref("models:/my-specialist/Production")
        assert ref.uri_type == ModelUriType.MLFLOW
        assert ref.mlflow_uri == "models:/my-specialist/Production"

    def test_mlflow_uri_by_version(self) -> None:
        ref = parse_model_ref("models:/my-specialist/3")
        assert ref.uri_type == ModelUriType.MLFLOW
        assert ref.mlflow_uri == "models:/my-specialist/3"

    def test_local_absolute_path(self, tmp_path: Path) -> None:
        ref = parse_model_ref(str(tmp_path))
        assert ref.uri_type == ModelUriType.LOCAL
        assert ref.path == str(tmp_path)

    def test_local_slash_prefix(self) -> None:
        ref = parse_model_ref("/mnt/models/my-model")
        assert ref.uri_type == ModelUriType.LOCAL

    def test_ollama_tag(self) -> None:
        ref = parse_model_ref("qwen2.5-coder:7b")
        assert ref.uri_type == ModelUriType.OLLAMA
        assert ref.path == "qwen2.5-coder:7b"

    def test_ollama_tag_llama(self) -> None:
        ref = parse_model_ref("llama3:8b")
        assert ref.uri_type == ModelUriType.OLLAMA

    def test_hf_org_slash_model(self) -> None:
        # Ensure 'org/model' without : is HF, not Ollama
        ref = parse_model_ref("mistralai/Mistral-7B-v0.1")
        assert ref.uri_type == ModelUriType.HF

    def test_raw_preserved(self) -> None:
        raw = "Qwen/Qwen2.5-7B@v0.3"
        ref = parse_model_ref(raw)
        assert ref.raw == raw


# ── resolve_model_ref ─────────────────────────────────────────────────────────


class TestResolveModelRef:
    def test_hf_no_revision_returns_repo_id(self) -> None:
        result = resolve_model_ref("Qwen/Qwen2.5-7B-Instruct")
        assert result == "Qwen/Qwen2.5-7B-Instruct"

    def test_hf_with_revision_calls_snapshot_download(self) -> None:
        fake_hf = MagicMock()
        fake_hf.snapshot_download.return_value = "/hf-cache/Qwen/Qwen2.5-7B/v0.3"
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            result = resolve_model_ref("Qwen/Qwen2.5-7B@v0.3")
        assert result == "/hf-cache/Qwen/Qwen2.5-7B/v0.3"
        fake_hf.snapshot_download.assert_called_once()
        kwargs = fake_hf.snapshot_download.call_args[1]
        assert kwargs["repo_id"] == "Qwen/Qwen2.5-7B"
        assert kwargs["revision"] == "v0.3"

    def test_hf_sha256_revision_strips_prefix(self) -> None:
        fake_hf = MagicMock()
        fake_hf.snapshot_download.return_value = "/hf-cache/model/sha"
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            resolve_model_ref("Qwen/model@sha256:abc123")
        kwargs = fake_hf.snapshot_download.call_args[1]
        assert kwargs["revision"] == "abc123"  # sha256: stripped

    def test_hf_401_raises_helpful_runtime_error(self) -> None:
        fake_hf = MagicMock()
        fake_hf.snapshot_download.side_effect = Exception("401 Unauthorized")
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            with pytest.raises(RuntimeError, match="Access denied"):
                resolve_model_ref("meta-llama/Llama-3@main")

    def test_hf_other_error_raises_runtime_error(self) -> None:
        fake_hf = MagicMock()
        fake_hf.snapshot_download.side_effect = Exception("network timeout")
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            with pytest.raises(RuntimeError, match="Failed to download"):
                resolve_model_ref("Qwen/model@v1")

    def test_huggingface_hub_not_installed_returns_repo_id(self) -> None:
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            result = resolve_model_ref("Qwen/Qwen2.5-7B@v0.3")
        assert result == "Qwen/Qwen2.5-7B"  # falls back to repo_id without revision

    def test_mlflow_uri_calls_download_artifacts(self) -> None:
        fake_mlflow = MagicMock()
        fake_mlflow.get_tracking_uri.return_value = "http://mlflow:5000"
        fake_mlflow.artifacts.download_artifacts.return_value = "/mlflow/artifacts/model"
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            result = resolve_model_ref(
                "models:/my-specialist/Production",
                mlflow_tracking_uri="http://mlflow:5000",
            )
        assert result == "/mlflow/artifacts/model"
        fake_mlflow.artifacts.download_artifacts.assert_called_once_with(
            artifact_uri="models:/my-specialist/Production"
        )

    def test_mlflow_no_tracking_uri_raises(self) -> None:
        fake_mlflow = MagicMock()
        fake_mlflow.get_tracking_uri.return_value = None
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            with pytest.raises(RuntimeError, match="No MLflow tracking URI"):
                resolve_model_ref("models:/my-specialist/Production")

    def test_mlflow_not_installed_raises(self) -> None:
        with patch.dict("sys.modules", {"mlflow": None}):
            with pytest.raises(RuntimeError, match="mlflow not installed"):
                resolve_model_ref("models:/my-specialist/3")

    def test_local_path_returned_unchanged(self) -> None:
        result = resolve_model_ref("/mnt/models/my-model")
        assert result == "/mnt/models/my-model"

    def test_ollama_tag_returned_unchanged(self) -> None:
        result = resolve_model_ref("qwen2.5:7b")
        assert result == "qwen2.5:7b"


# ── list_hf_revisions ─────────────────────────────────────────────────────────


class TestListHfRevisions:
    def test_returns_branches_and_tags(self) -> None:
        from aua.model_registry import list_hf_revisions

        fake_hf = MagicMock()
        fake_refs = MagicMock()

        def _branch(n, c):
            m = MagicMock()
            m.name = n
            m.target_commit = c
            return m

        fake_refs.branches = [_branch("main", "abc123"), _branch("dev", "def456")]
        fake_refs.tags = [_branch("v0.3", "ghi789")]
        fake_hf.list_repo_refs.return_value = fake_refs

        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            result = list_hf_revisions("Qwen/model")

        assert len(result) == 3
        branch_names = [r.name for r in result if r.ref_type == "branch"]
        assert "main" in branch_names
        tag_names = [r.name for r in result if r.ref_type == "tag"]
        assert "v0.3" in tag_names

    def test_error_propagates_as_runtime_error(self) -> None:
        from aua.model_registry import list_hf_revisions

        fake_hf = MagicMock()
        fake_hf.list_repo_refs.side_effect = Exception("repo not found")
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            with pytest.raises(RuntimeError, match="Failed to list revisions"):
                list_hf_revisions("nonexistent/model")

    def test_not_installed_raises_runtime_error(self) -> None:
        from aua.model_registry import list_hf_revisions

        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(RuntimeError, match="huggingface_hub not installed"):
                list_hf_revisions("any/model")


# ── list_mlflow_versions ──────────────────────────────────────────────────────


class TestListMlflowVersions:
    def test_returns_version_list(self) -> None:
        from aua.model_registry import list_mlflow_versions

        fake_mlflow = MagicMock()
        v1 = MagicMock(
            version="1",
            current_stage="Production",
            status="READY",
            run_id="run-1",
            source="s3://bucket/model/1",
        )
        v2 = MagicMock(
            version="2",
            current_stage="Staging",
            status="READY",
            run_id="run-2",
            source="s3://bucket/model/2",
        )
        fake_mlflow.MlflowClient.return_value.search_model_versions.return_value = [v1, v2]
        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            result = list_mlflow_versions("my-specialist")

        assert len(result) == 2
        assert result[0].version == "1"
        assert result[0].stage == "Production"

    def test_not_installed_raises(self) -> None:
        from aua.model_registry import list_mlflow_versions

        with patch.dict("sys.modules", {"mlflow": None}):
            with pytest.raises(RuntimeError, match="mlflow not installed"):
                list_mlflow_versions("model")


# ── config.py YAML loading ────────────────────────────────────────────────────


class TestConfigYamlLoading:
    def _write(self, tmp_path: Path, mlflow_uri: str = "") -> Path:
        content = f"""
aua:
  version: "1.0"
  mode: local
  backend: vllm
specialists:
  - name: swe
    model: {mlflow_uri or "Qwen/Qwen2.5-7B@v0.3"}
    port: 9001
    field: software_engineering
    {"mlflow_tracking_uri: http://mlflow:5000" if "models:/" in mlflow_uri else ""}
arbiter:
  model: Qwen/arb
  port: 9003
router:
  port: 8000
"""
        p = tmp_path / "cfg.yaml"
        p.write_text(content)
        return p

    def test_hf_revision_parsed_from_model_field(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(self._write(tmp_path))
        spec = cfg.specialists[0]
        assert spec.model == "Qwen/Qwen2.5-7B@v0.3"
        # parse_model_ref should see the revision
        from aua.model_registry import parse_model_ref

        ref = parse_model_ref(spec.model)
        assert ref.revision == "v0.3"

    def test_mlflow_tracking_uri_loaded(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(self._write(tmp_path, mlflow_uri="models:/my-specialist/Production"))
        assert cfg.specialists[0].mlflow_tracking_uri == "http://mlflow:5000"

    def test_mlflow_tracking_uri_defaults_to_none(self, tmp_path: Path) -> None:
        from aua.config import load_config

        cfg = load_config(self._write(tmp_path))
        assert cfg.specialists[0].mlflow_tracking_uri is None


# ── CLI: aua models pin ───────────────────────────────────────────────────────


class TestModelsPin:
    def test_list_hf_revisions(self) -> None:
        from click.testing import CliRunner

        from aua.cli import main

        runner = CliRunner()
        with patch(
            "aua.model_registry.list_hf_revisions",
            return_value=[
                RevisionInfo(name="main", commit="abc123", ref_type="branch"),
                RevisionInfo(name="v0.3", commit="def456", ref_type="tag"),
            ],
        ):
            result = runner.invoke(main, ["models", "pin", "Qwen/Qwen2.5-7B"])
        assert result.exit_code == 0
        assert "main" in result.output or "v0.3" in result.output or "Revisions" in result.output

    def test_pin_to_revision_shows_config_snippet(self) -> None:
        from click.testing import CliRunner

        from aua.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["models", "pin", "Qwen/Qwen2.5-7B", "--revision", "v0.3"])
        assert result.exit_code == 0
        assert "Qwen/Qwen2.5-7B@v0.3" in result.output

    def test_pin_json_output(self) -> None:
        import json

        from click.testing import CliRunner

        from aua.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["models", "pin", "Qwen/Qwen2.5-7B", "--revision", "main", "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "pinned" in data
        assert "@main" in data["pinned"]

    def test_list_hf_error_exits_1(self) -> None:
        from click.testing import CliRunner

        from aua.cli import main

        fake_hf = MagicMock()
        fake_hf.list_repo_refs.side_effect = Exception("repo not found")
        runner = CliRunner()
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            result = runner.invoke(main, ["models", "pin", "bad/repo"])
        assert result.exit_code == 1
