"""
aua/model_registry.py — Model registry integration (#46).

Resolves model references in specialist config to local paths before
vLLM or Ollama starts. Supports:

  HuggingFace Hub with version pinning
  ────────────────────────────────────
  Syntax in aua_config.yaml:
      model: Qwen/Qwen2.5-7B-Instruct              # latest (existing behaviour)
      model: Qwen/Qwen2.5-7B-Instruct@v0.3         # branch/tag
      model: Qwen/Qwen2.5-7B-Instruct@sha256:abc123 # exact commit (recommended for prod)

  The '@' suffix is passed as the `revision` argument to
  huggingface_hub.snapshot_download(). If the model is already cached at
  that revision, download is skipped.

  MLflow Model Registry
  ─────────────────────
  Syntax in aua_config.yaml:
      model: models:/my-specialist/Production      # by stage
      model: models:/my-specialist/3               # by version number
      model: models:/my-specialist/latest          # alias (MLflow ≥ 2.9)
      mlflow_tracking_uri: http://mlflow:5000      # required on specialist

  The URI is resolved via mlflow.artifacts.download_artifacts() to a local
  directory containing the model artifacts. That local path is then used as
  the vLLM --model argument.

  Local paths and Ollama tags are passed through unchanged.

Public API
──────────
  parse_model_ref(model_str) → ModelRef
    Parses a model string into (repo_id, revision, uri_type) without any
    network calls.

  resolve_model_ref(model_str, mlflow_tracking_uri, cache_dir) → str
    Returns a local path suitable for vLLM's --model flag. Downloads if
    needed. Used in serve.py before _start_specialist/_start_arbiter.

  list_hf_revisions(repo_id, token) → list[RevisionInfo]
    Lists available branches/tags for a HF repo (for `aua models pin`).

  list_mlflow_versions(model_name, tracking_uri) → list[MLflowVersionInfo]
    Lists registered model versions (for `aua models pin`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)

# ── Model reference parsing ───────────────────────────────────────────────────


class ModelUriType(str, Enum):
    HF = "hf"  # HuggingFace Hub repo (with optional @revision)
    MLFLOW = "mlflow"  # models:/ URI
    LOCAL = "local"  # absolute or relative path
    OLLAMA = "ollama"  # tag format: name:tag (no slash prefix)


@dataclass
class ModelRef:
    raw: str  # original string from config
    uri_type: ModelUriType
    # HF fields
    repo_id: str | None = None  # e.g. "Qwen/Qwen2.5-7B-Instruct"
    revision: str | None = None  # e.g. "v0.3" or "abc1234" (None → default branch)
    # MLflow fields
    mlflow_uri: str | None = None  # e.g. "models:/my-model/Production"
    # Local / Ollama
    path: str | None = None  # local path or Ollama tag


def parse_model_ref(model_str: str) -> ModelRef:
    """
    Parse a model string from aua_config.yaml into a ModelRef.

    Rules (checked in order):
      1. Starts with 'models:/'   → MLflow
      2. Starts with '/' or './'  → Local path
      3. Contains '/' and no ':'  → HF repo (with optional @revision)
      4. Otherwise                → Ollama tag (e.g. qwen2.5:7b)

    HF @revision: anything after the first '@' is the revision.
      Qwen/Qwen2.5-7B@v0.3
      Qwen/Qwen2.5-7B@sha256:abc123   (note: sha256: is part of the revision)
    """
    s = model_str.strip()

    # MLflow model registry URI
    if s.startswith("models:/"):
        return ModelRef(raw=s, uri_type=ModelUriType.MLFLOW, mlflow_uri=s)

    # Local path
    if s.startswith("/") or s.startswith("./") or s.startswith("../") or os.path.exists(s):
        return ModelRef(raw=s, uri_type=ModelUriType.LOCAL, path=s)

    # HuggingFace: contains a slash and is not an Ollama tag (which also uses ':')
    # An Ollama tag is namespace:tag without a slash, or name:tag — no '/' before ':'
    slash_pos = s.find("/")
    colon_pos = s.find(":")
    if slash_pos > 0 and (colon_pos < 0 or slash_pos < colon_pos):
        # HF repo with optional @revision
        if "@" in s:
            repo_id, revision = s.split("@", 1)
            return ModelRef(
                raw=s,
                uri_type=ModelUriType.HF,
                repo_id=repo_id,
                revision=revision,
            )
        return ModelRef(raw=s, uri_type=ModelUriType.HF, repo_id=s, revision=None)

    # Ollama tag (qwen2.5:7b, llama3:8b, etc.) or bare name
    return ModelRef(raw=s, uri_type=ModelUriType.OLLAMA, path=s)


# ── Resolution ────────────────────────────────────────────────────────────────


def resolve_model_ref(
    model_str: str,
    mlflow_tracking_uri: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """
    Resolve a model string to a local path for vLLM --model.

    - HF without revision: returns repo_id unchanged (vLLM pulls it)
    - HF with revision: downloads the specific revision via huggingface_hub
      and returns the local cache path
    - MLflow URI: downloads artifacts and returns local path
    - Local path / Ollama tag: returned unchanged

    Args:
        model_str:           The model field from aua_config.yaml
        mlflow_tracking_uri: MLflow tracking server URL (required for MLflow URIs)
        cache_dir:           Optional HF cache directory override

    Returns:
        A string suitable as vLLM's --model argument.
    """
    ref = parse_model_ref(model_str)

    if ref.uri_type == ModelUriType.LOCAL:
        return ref.path  # type: ignore[return-value]

    if ref.uri_type == ModelUriType.OLLAMA:
        return ref.path  # type: ignore[return-value]

    if ref.uri_type == ModelUriType.HF:
        if ref.revision is None:
            # No pinning — vLLM handles download via HF cache normally
            return ref.repo_id  # type: ignore[return-value]
        return _resolve_hf_revision(ref.repo_id, ref.revision, cache_dir)  # type: ignore[arg-type]

    if ref.uri_type == ModelUriType.MLFLOW:
        return _resolve_mlflow_uri(ref.mlflow_uri, mlflow_tracking_uri)  # type: ignore[arg-type]

    return model_str  # unreachable but satisfies mypy


def _resolve_hf_revision(repo_id: str, revision: str, cache_dir: str | None) -> str:
    """
    Download a specific revision from HuggingFace Hub and return the local path.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.warning(
            "model_registry: huggingface_hub not installed — "
            "using repo_id without pinned revision. "
            "Install with: pip install huggingface_hub"
        )
        return repo_id

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    # Handle sha256: prefix in revision strings
    hf_revision = revision.replace("sha256:", "") if revision.startswith("sha256:") else revision

    log.info("model_registry: pinning %s to revision %s", repo_id, revision)
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=hf_revision,
            token=token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            **({"cache_dir": cache_dir} if cache_dir else {}),
        )
        log.info("model_registry: %s@%s → %s", repo_id, revision, local_dir)
        return local_dir
    except Exception as e:
        if "401" in str(e) or "403" in str(e):
            raise RuntimeError(
                f"Access denied for {repo_id}@{revision}. "
                f"Set HF_TOKEN env var and accept the model's license at "
                f"https://huggingface.co/{repo_id}"
            ) from e
        raise RuntimeError(f"Failed to download {repo_id}@{revision}: {e}") from e


def _resolve_mlflow_uri(mlflow_uri: str, tracking_uri: str | None) -> str:
    """
    Resolve a models:/ URI to a local artifact directory via MLflow.
    """
    try:
        import mlflow
    except ImportError:
        raise RuntimeError(
            f"mlflow not installed — cannot resolve '{mlflow_uri}'. "
            "Install with: pip install mlflow"
        )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif not mlflow.get_tracking_uri():
        raise RuntimeError(
            f"No MLflow tracking URI set for model '{mlflow_uri}'. "
            "Add mlflow_tracking_uri to the specialist config or set "
            "MLFLOW_TRACKING_URI env var."
        )

    log.info("model_registry: resolving MLflow URI %s", mlflow_uri)
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=mlflow_uri)
        log.info("model_registry: %s → %s", mlflow_uri, local_path)
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to resolve MLflow URI '{mlflow_uri}': {e}") from e


# ── Discovery helpers (used by aua models pin) ────────────────────────────────


@dataclass
class RevisionInfo:
    name: str  # branch/tag name
    commit: str  # commit hash
    ref_type: str  # "branch" | "tag"


def list_hf_revisions(repo_id: str, token: str | None = None) -> list[RevisionInfo]:
    """
    List available branches and tags for a HuggingFace repo.
    Requires huggingface_hub installed.
    """
    try:
        from huggingface_hub import list_repo_refs
    except ImportError:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")

    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        refs = list_repo_refs(repo_id, token=token)
    except Exception as e:
        raise RuntimeError(f"Failed to list revisions for {repo_id}: {e}") from e

    result: list[RevisionInfo] = []
    for branch in refs.branches:
        result.append(
            RevisionInfo(
                name=branch.name,
                commit=branch.target_commit,
                ref_type="branch",
            )
        )
    for tag in refs.tags:
        result.append(
            RevisionInfo(
                name=tag.name,
                commit=tag.target_commit,
                ref_type="tag",
            )
        )
    return result


@dataclass
class MLflowVersionInfo:
    version: str
    stage: str  # "Production" | "Staging" | "Archived" | "None"
    status: str
    run_id: str
    source: str  # artifact URI of the model source


def list_mlflow_versions(
    model_name: str,
    tracking_uri: str | None = None,
) -> list[MLflowVersionInfo]:
    """
    List registered versions of an MLflow model.
    """
    try:
        from mlflow import MlflowClient
    except ImportError:
        raise RuntimeError("mlflow not installed. Run: pip install mlflow")

    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        raise RuntimeError(f"Failed to list MLflow versions for {model_name}: {e}") from e

    return [
        MLflowVersionInfo(
            version=v.version,
            stage=v.current_stage,
            status=v.status,
            run_id=v.run_id,
            source=v.source,
        )
        for v in versions
    ]
