"""
tests/test_compat_matrix.py — Tests for #55 compatibility matrix.

Covers:
  Matrix completeness — every (format, hardware, backend) lookup returns an entry
    or at least a deterministic None (not a KeyError)
  Status invariants — known-good combinations are "supported"
  Known-bad combinations are "unsupported"
  vLLM requires CUDA — all non-CUDA hardware with vLLM is unsupported
  MLX is Apple Silicon only — mlx_lm and mlx format unsupported elsewhere
  GGUF is universal on Ollama — supported on all hardware tiers
  infer_model_format — correct format inferred from model name strings
  lookup() — works with tier aliases and backend aliases
  check_config() — returns (status, notes) without raising
  to_markdown() — produces valid Markdown with all backends and formats
  doctor _check_compat — correct check statuses for known combinations
"""

from __future__ import annotations

import pytest

from aua.compat import (
    BACKENDS,
    HARDWARE_TIERS,
    MATRIX,
    MODEL_FORMATS,
    check_config,
    infer_model_format,
    lookup,
    to_markdown,
)

# ── Matrix structure invariants ───────────────────────────────────────────────


class TestMatrixInvariants:
    def test_all_entries_have_valid_status(self) -> None:
        valid = {"supported", "untested", "unsupported"}
        for key, entry in MATRIX.items():
            assert entry.status in valid, f"{key}: invalid status {entry.status!r}"

    def test_all_entries_have_non_empty_notes(self) -> None:
        for key, entry in MATRIX.items():
            assert entry.notes, f"{key}: empty notes"

    def test_supported_entries_have_vram_for_cuda(self) -> None:
        """Supported CUDA entries should document VRAM requirements."""
        for (fmt, hw, be), entry in MATRIX.items():
            if "cuda" in hw and entry.status == "supported" and be in ("vllm", "ollama"):
                if fmt not in ("gguf", "awq", "gptq", "bnb4", "bnb8"):
                    # Full precision formats need VRAM annotation
                    assert entry.min_vram_gb is not None or fmt in (
                        "gguf",
                    ), f"({fmt}, {hw}, {be}): supported but no min_vram_gb"


# ── Known-good combinations ───────────────────────────────────────────────────


class TestKnownGoodCombinations:
    def test_bf16_cuda_sm86_vllm_supported(self) -> None:
        e = MATRIX.get(("bf16", "cuda_sm86+", "vllm"))
        assert e is not None
        assert e.status == "supported"

    def test_awq_cuda_consumer_vllm_supported(self) -> None:
        e = MATRIX.get(("awq", "cuda_consumer", "vllm"))
        assert e is not None
        assert e.status == "supported"

    def test_gguf_apple_silicon_ollama_supported(self) -> None:
        e = MATRIX.get(("gguf", "apple_silicon", "ollama"))
        assert e is not None
        assert e.status == "supported"

    def test_mlx_apple_silicon_mlx_lm_supported(self) -> None:
        e = MATRIX.get(("mlx", "apple_silicon", "mlx_lm"))
        assert e is not None
        assert e.status == "supported"

    def test_gguf_cpu_avx2_llamacpp_supported(self) -> None:
        e = MATRIX.get(("gguf", "cpu_avx2", "llamacpp"))
        assert e is not None
        assert e.status == "supported"

    def test_gguf_cuda_consumer_ollama_supported(self) -> None:
        e = MATRIX.get(("gguf", "cuda_consumer", "ollama"))
        assert e is not None
        assert e.status == "supported"


# ── Known-bad combinations ────────────────────────────────────────────────────


class TestKnownBadCombinations:
    def test_vllm_apple_silicon_always_unsupported(self) -> None:
        """vLLM does not support Apple Silicon for any format."""
        for fmt in MODEL_FORMATS:
            e = MATRIX.get((fmt, "apple_silicon", "vllm"))
            if e is not None:
                assert (
                    e.status == "unsupported"
                ), f"({fmt}, apple_silicon, vllm) should be unsupported, got {e.status}"

    def test_vllm_cpu_always_unsupported(self) -> None:
        """vLLM requires CUDA — CPU is unsupported."""
        for fmt in MODEL_FORMATS:
            for hw in ("cpu_avx2", "cpu_arm"):
                e = MATRIX.get((fmt, hw, "vllm"))
                if e is not None:
                    assert e.status == "unsupported", f"({fmt}, {hw}, vllm) should be unsupported"

    def test_mlx_format_on_non_apple_hardware_unsupported(self) -> None:
        """MLX format only works on Apple Silicon."""
        cuda_hw = [h for h in HARDWARE_TIERS if "cuda" in h]
        for hw in cuda_hw + ["cpu_avx2", "cpu_arm"]:
            for be in BACKENDS:
                e = MATRIX.get(("mlx", hw, be))
                if e is not None:
                    assert e.status == "unsupported", f"(mlx, {hw}, {be}) should be unsupported"

    def test_mlx_lm_on_non_apple_unsupported(self) -> None:
        """mlx_lm backend only works on Apple Silicon."""
        for fmt in MODEL_FORMATS:
            for hw in HARDWARE_TIERS:
                if hw == "apple_silicon":
                    continue
                e = MATRIX.get((fmt, hw, "mlx_lm"))
                if e is not None:
                    assert e.status == "unsupported", f"({fmt}, {hw}, mlx_lm) should be unsupported"

    def test_awq_unsupported_on_apple_silicon(self) -> None:
        """AWQ is CUDA-specific."""
        for be in BACKENDS:
            e = MATRIX.get(("awq", "apple_silicon", be))
            if e is not None:
                assert e.status in (
                    "unsupported",
                    "untested",
                ), f"(awq, apple_silicon, {be}) should not be supported"

    def test_bitsandbytes_unsupported_in_ollama(self) -> None:
        """bitsandbytes is not supported in Ollama."""
        for hw in HARDWARE_TIERS:
            for fmt in ("bnb4", "bnb8"):
                e = MATRIX.get((fmt, hw, "ollama"))
                if e is not None:
                    assert (
                        e.status == "unsupported"
                    ), f"({fmt}, {hw}, ollama) bitsandbytes not in Ollama"


# ── lookup() and check_config() ──────────────────────────────────────────────


class TestLookup:
    def test_lookup_with_tier_alias(self) -> None:
        """Tier aliases (e.g. 'gaming-pc') resolve to hardware tiers."""
        e = lookup("gguf", "gaming-pc", "ollama")
        assert e is not None
        assert e.status == "supported"

    def test_lookup_with_backend_alias(self) -> None:
        """Backend aliases (e.g. 'llama.cpp') resolve to canonical backends."""
        e = lookup("gguf", "cpu_avx2", "llama.cpp")
        assert e is not None
        assert e.status == "supported"

    def test_lookup_unknown_combination_returns_none(self) -> None:
        e = lookup("bf16", "unknown_hw_tier", "vllm")
        assert e is None

    def test_check_config_returns_tuple(self) -> None:
        status, notes = check_config("awq", "cuda_sm86+", "vllm")
        assert status in ("supported", "untested", "unsupported", "unknown")
        assert isinstance(notes, str)
        assert len(notes) > 0

    def test_check_config_unknown_returns_unknown(self) -> None:
        status, notes = check_config("bf16", "quantum_compute", "vllm")
        assert status == "unknown"
        assert "not in compatibility matrix" in notes

    def test_check_config_h100_alias(self) -> None:
        status, _ = check_config("bf16", "h100-cluster", "vllm")
        assert status == "supported"

    def test_lookup_macbook_tier(self) -> None:
        e = lookup("gguf", "macbook", "ollama")
        assert e is not None
        assert e.status == "supported"


# ── infer_model_format ────────────────────────────────────────────────────────


class TestInferModelFormat:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Qwen/Qwen2.5-7B-Instruct-AWQ", "awq"),
            ("model-gptq-4bit", "gptq"),
            ("llama-7b.Q4_K_M.gguf", "gguf"),
            ("mlx-community/Qwen2.5-7B-mlx", "mlx"),
            ("model-nf4", "bnb4"),
            ("model-int8", "bnb8"),
            ("Qwen/Qwen2.5-72B-Instruct-fp16", "fp16"),
            ("model-bf16", "bf16"),
            ("Qwen/Qwen2.5-7B-Instruct", "unknown"),  # no suffix
            ("meta-llama/Llama-3-8B", "unknown"),
        ],
    )
    def test_infer_format(self, name: str, expected: str) -> None:
        assert infer_model_format(name) == expected

    def test_gguf_file_extension(self) -> None:
        assert infer_model_format("mistral-7b-q4_k_m.gguf") == "gguf"

    def test_awq_case_insensitive(self) -> None:
        assert infer_model_format("model-AWQ") == "awq"

    def test_unknown_hf_id(self) -> None:
        assert infer_model_format("meta-llama/Meta-Llama-3-8B-Instruct") == "unknown"

    # ── Backend-aware inference (v1.2 audit fix) ─────────────────────────────

    def test_ollama_tag_infers_gguf(self) -> None:
        """Ollama model tags without a suffix should resolve to gguf, not unknown."""
        assert infer_model_format("qwen2.5-coder:7b", backend="ollama") == "gguf"
        assert infer_model_format("qwen2.5:3b", backend="ollama") == "gguf"

    def test_llamacpp_backend_infers_gguf(self) -> None:
        assert infer_model_format("some-model", backend="llamacpp") == "gguf"
        assert infer_model_format("some-model", backend="llama.cpp") == "gguf"

    def test_mlx_lm_backend_infers_mlx(self) -> None:
        assert infer_model_format("qwen2.5-7b", backend="mlx_lm") == "mlx"

    def test_vllm_backend_no_suffix_still_unknown(self) -> None:
        """vLLM has no single native format, so no suffix → unknown."""
        assert infer_model_format("Qwen/Qwen2.5-7B-Instruct", backend="vllm") == "unknown"

    def test_explicit_suffix_overrides_backend(self) -> None:
        """An explicit format suffix wins over the backend hint."""
        assert infer_model_format("model-AWQ", backend="ollama") == "awq"

    def test_no_backend_preserves_unknown(self) -> None:
        """Without a backend hint, a suffix-less name is still unknown (back-compat)."""
        assert infer_model_format("qwen2.5-coder:7b") == "unknown"


# ── to_markdown ───────────────────────────────────────────────────────────────


class TestToMarkdown:
    def test_markdown_contains_all_backends(self) -> None:
        md = to_markdown()
        for backend in BACKENDS:
            assert f"## {backend}" in md

    def test_markdown_contains_all_formats(self) -> None:
        md = to_markdown()
        for fmt in MODEL_FORMATS:
            assert f"`{fmt}`" in md

    def test_markdown_contains_legend(self) -> None:
        md = to_markdown()
        assert "Supported" in md
        assert "Untested" in md
        assert "Unsupported" in md

    def test_markdown_is_string(self) -> None:
        md = to_markdown()
        assert isinstance(md, str)
        assert len(md) > 500  # substantial content


# ── Doctor _check_compat integration ─────────────────────────────────────────


class TestDoctorCompatCheck:
    def _make_fake_cfg(self, model: str, backend: str = "vllm"):
        """Create a minimal fake config object with one specialist."""

        class FakeSpec:
            def __init__(self, name, mdl):
                self.name = name
                self.model = mdl

        class FakeCfg:
            def __init__(self, be, mdl):
                self.backend = be
                self.specialists = [FakeSpec("swe", mdl)]

        return FakeCfg(backend, model)

    def test_known_good_combo_passes(self) -> None:
        from aua.doctor import _check_compat

        # Use a combination that's valid on CPU (gguf + ollama) to be env-independent
        cfg = self._make_fake_cfg("llama-7b.Q4_K_M.gguf", backend="ollama")
        checks = _check_compat(cfg)
        # Should produce at least one check (pass or warn — not unsupported fail)
        gguf_check = next((c for c in checks if "gguf" in c.name), None)
        if gguf_check:  # if format was inferred
            assert gguf_check.status in (
                "pass",
                "warn",
                "info",
            ), f"gguf+ollama should be supported, got {gguf_check.status}: {gguf_check.detail}"

    def test_unsupported_combo_fails(self) -> None:

        # vLLM + apple_silicon is unsupported — but we can't guarantee the test
        # machine is Apple Silicon. Test the logic via a direct matrix check instead.
        status, _ = check_config("mlx", "apple_silicon", "vllm")
        assert status == "unsupported"

    def test_unknown_format_produces_info(self) -> None:
        from aua.doctor import _check_compat

        cfg = self._make_fake_cfg("meta-llama/Llama-3-8B-Instruct", backend="vllm")
        checks = _check_compat(cfg)
        info_check = next(
            (c for c in checks if "unknown" in c.name.lower() or c.status == "info"), None
        )
        assert info_check is not None

    def test_no_specialists_returns_skip(self) -> None:
        from dataclasses import dataclass

        from aua.doctor import _check_compat

        @dataclass
        class EmptyCfg:
            backend: str = "vllm"
            specialists: list = None

            def __post_init__(self):
                if self.specialists is None:
                    self.specialists = []

        checks = _check_compat(EmptyCfg())
        assert any(c.status == "skip" for c in checks)

    def test_none_config_returns_skip(self) -> None:
        from aua.doctor import _check_compat

        checks = _check_compat(None)
        assert any(c.status == "skip" for c in checks)
