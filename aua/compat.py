"""
aua/compat.py — Model × hardware × backend compatibility matrix.

Implements roadmap #55: Extended compatibility matrix.

The matrix is the authoritative source for which (model_format, hardware, backend)
combinations are:
  supported  — tested and confirmed working
  untested   — not tested by the AUA team; may work, use at your own risk
  unsupported — known to fail or architecturally impossible

Used by:
  aua doctor     — check group 6 reports compatibility status for every specialist
  aua doctor --compat-matrix  — dump the full matrix as JSON or Markdown
  CI             — tests/test_compat_matrix.py asserts invariants

Design:
  Model formats:  bf16 | fp16 | awq | gptq | gguf | mlx | bnb4 | bnb8
  Hardware tiers: cuda_sm86+  (A100/H100/4090)
                  cuda_sm80   (A100 older, RTX 3090)
                  cuda_sm75   (T4, RTX 20xx)
                  cuda_consumer  (RTX 30xx/40xx < A100 class)
                  apple_silicon  (M1/M2/M3/M4 via Metal)
                  cpu_avx2       (x86-64 with AVX2, no GPU)
                  cpu_arm        (ARM64 without Metal, e.g. Raspberry Pi)
  Backends:       vllm | ollama | llamacpp | mlx_lm | transformers

Each entry is a CompatEntry with:
  status:  "supported" | "untested" | "unsupported"
  notes:   short human-readable explanation (shown by aua doctor)
  min_vram_gb: approximate minimum VRAM for a 7B model in this format (None = N/A)
  tier_aliases: which aua --tier names cover this hardware class
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CompatEntry:
    """One cell in the compatibility matrix."""

    status: str  # "supported" | "untested" | "unsupported"
    notes: str
    min_vram_gb: float | None = None  # for 7B model; scale proportionally for larger
    tier_aliases: list[str] = field(default_factory=list)


# ── Status constants ──────────────────────────────────────────────────────────
OK = "supported"
UNTESTED = "untested"
NO = "unsupported"

# ── The matrix: (model_format, hardware, backend) → CompatEntry ──────────────
# fmt: off
MATRIX: dict[tuple[str, str, str], CompatEntry] = {

    # ── vLLM backend ─────────────────────────────────────────────────────────
    # vLLM requires CUDA; MPS and CPU are not supported by vLLM itself.

    ("bf16",  "cuda_sm86+",    "vllm"):  CompatEntry(OK,       "Native BF16 on H100/A100 — best throughput, highest quality", 14.0, ["h100-cluster", "a100-cluster"]),
    ("bf16",  "cuda_sm80",     "vllm"):  CompatEntry(OK,       "BF16 supported on A100 PCIe / older SXM", 14.0, ["a100-cluster"]),
    ("bf16",  "cuda_sm75",     "vllm"):  CompatEntry(OK,       "BF16 supported on T4/RTX 20xx but slower (no BF16 tensor cores)", 14.0, []),
    ("bf16",  "cuda_consumer", "vllm"):  CompatEntry(OK,       "BF16 on RTX 3090/4090; check model fits in VRAM", 14.0, ["single-4090", "quad-4090"]),
    ("bf16",  "apple_silicon", "vllm"):  CompatEntry(NO,       "vLLM does not support Apple Silicon / MPS. Use ollama or mlx_lm instead.", None, []),
    ("bf16",  "cpu_avx2",      "vllm"):  CompatEntry(NO,       "vLLM requires CUDA. Use llamacpp or transformers for CPU inference.", None, []),
    ("bf16",  "cpu_arm",       "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),

    ("fp16",  "cuda_sm86+",    "vllm"):  CompatEntry(OK,       "FP16 works on all CUDA tiers; slightly lower throughput than BF16 on H100", 14.0, ["h100-cluster", "a100-cluster"]),
    ("fp16",  "cuda_sm80",     "vllm"):  CompatEntry(OK,       "FP16 fully supported", 14.0, ["a100-cluster"]),
    ("fp16",  "cuda_sm75",     "vllm"):  CompatEntry(OK,       "FP16 tensor cores available on Turing GPUs", 14.0, []),
    ("fp16",  "cuda_consumer", "vllm"):  CompatEntry(OK,       "Standard configuration for consumer CUDA GPUs", 14.0, ["single-4090", "quad-4090"]),
    ("fp16",  "apple_silicon", "vllm"):  CompatEntry(NO,       "vLLM does not support MPS.", None, []),
    ("fp16",  "cpu_avx2",      "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),
    ("fp16",  "cpu_arm",       "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),

    ("awq",   "cuda_sm86+",    "vllm"):  CompatEntry(OK,       "AWQ 4-bit quantisation — best throughput/quality tradeoff on CUDA. Requires vLLM >= 0.3.0", 5.5, ["h100-cluster", "a100-cluster", "single-4090", "quad-4090"]),
    ("awq",   "cuda_sm80",     "vllm"):  CompatEntry(OK,       "AWQ supported on A100 and newer", 5.5, ["a100-cluster"]),
    ("awq",   "cuda_sm75",     "vllm"):  CompatEntry(UNTESTED, "AWQ may work on sm75 but is untested; quantisation kernels optimised for sm80+", 5.5, []),
    ("awq",   "cuda_consumer", "vllm"):  CompatEntry(OK,       "AWQ is the recommended format for RTX 30xx/40xx with vLLM", 5.5, ["single-4090", "quad-4090"]),
    ("awq",   "apple_silicon", "vllm"):  CompatEntry(NO,       "vLLM not supported on Apple Silicon.", None, []),
    ("awq",   "cpu_avx2",      "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),
    ("awq",   "cpu_arm",       "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),

    ("gptq",  "cuda_sm86+",    "vllm"):  CompatEntry(OK,       "GPTQ 4-bit supported via AutoGPTQ integration in vLLM. Slightly lower throughput than AWQ.", 5.5, ["h100-cluster", "a100-cluster"]),
    ("gptq",  "cuda_sm80",     "vllm"):  CompatEntry(OK,       "GPTQ fully supported", 5.5, []),
    ("gptq",  "cuda_sm75",     "vllm"):  CompatEntry(OK,       "GPTQ on sm75 is slower but works", 5.5, []),
    ("gptq",  "cuda_consumer", "vllm"):  CompatEntry(OK,       "GPTQ works on consumer CUDA GPUs", 5.5, ["single-4090"]),
    ("gptq",  "apple_silicon", "vllm"):  CompatEntry(NO,       "vLLM not supported on Apple Silicon.", None, []),
    ("gptq",  "cpu_avx2",      "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),
    ("gptq",  "cpu_arm",       "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),

    ("gguf",  "cuda_sm86+",    "vllm"):  CompatEntry(UNTESTED, "vLLM has experimental GGUF support since 0.4.x. Prefer AWQ/GPTQ for CUDA.", 5.5, []),
    ("gguf",  "cuda_sm80",     "vllm"):  CompatEntry(UNTESTED, "Experimental GGUF support in vLLM; not recommended for production.", None, []),
    ("gguf",  "cuda_sm75",     "vllm"):  CompatEntry(UNTESTED, "Experimental.", None, []),
    ("gguf",  "cuda_consumer", "vllm"):  CompatEntry(UNTESTED, "Experimental GGUF in vLLM; prefer Ollama for GGUF on consumer GPUs.", None, []),
    ("gguf",  "apple_silicon", "vllm"):  CompatEntry(NO,       "vLLM not supported on Apple Silicon. Use ollama for GGUF.", None, []),
    ("gguf",  "cpu_avx2",      "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),
    ("gguf",  "cpu_arm",       "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),

    ("mlx",   "cuda_sm86+",    "vllm"):  CompatEntry(NO,       "MLX format is Apple Silicon only. Convert to AWQ/GPTQ for CUDA.", None, []),
    ("mlx",   "cuda_sm80",     "vllm"):  CompatEntry(NO,       "MLX not supported on CUDA.", None, []),
    ("mlx",   "cuda_sm75",     "vllm"):  CompatEntry(NO,       "MLX not supported on CUDA.", None, []),
    ("mlx",   "cuda_consumer", "vllm"):  CompatEntry(NO,       "MLX not supported on CUDA.", None, []),
    ("mlx",   "apple_silicon", "vllm"):  CompatEntry(NO,       "vLLM not supported on Apple Silicon. Use mlx_lm for MLX models.", None, []),
    ("mlx",   "cpu_avx2",      "vllm"):  CompatEntry(NO,       "MLX is Apple Silicon only.", None, []),
    ("mlx",   "cpu_arm",       "vllm"):  CompatEntry(NO,       "MLX is Apple Silicon only.", None, []),

    ("bnb4",  "cuda_sm86+",    "vllm"):  CompatEntry(UNTESTED, "bitsandbytes 4-bit (NF4) has limited vLLM support. Prefer AWQ for production.", 4.5, []),
    ("bnb4",  "cuda_sm80",     "vllm"):  CompatEntry(UNTESTED, "bitsandbytes NF4 via vLLM — experimental.", 4.5, []),
    ("bnb4",  "cuda_sm75",     "vllm"):  CompatEntry(NO,       "bitsandbytes NF4 not well-supported on older CUDA generations in vLLM.", None, []),
    ("bnb4",  "cuda_consumer", "vllm"):  CompatEntry(UNTESTED, "bitsandbytes NF4 — experimental in vLLM; prefer AWQ/GPTQ.", 4.5, []),
    ("bnb4",  "apple_silicon", "vllm"):  CompatEntry(NO,       "vLLM not supported on Apple Silicon.", None, []),
    ("bnb4",  "cpu_avx2",      "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),
    ("bnb4",  "cpu_arm",       "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),

    ("bnb8",  "cuda_sm86+",    "vllm"):  CompatEntry(UNTESTED, "bitsandbytes INT8 — higher quality than NF4 but more VRAM. Untested in AUA CI.", 8.0, []),
    ("bnb8",  "cuda_sm80",     "vllm"):  CompatEntry(UNTESTED, "bitsandbytes INT8 via vLLM — experimental.", 8.0, []),
    ("bnb8",  "cuda_sm75",     "vllm"):  CompatEntry(NO,       "bitsandbytes INT8 not reliable on sm75 with vLLM.", None, []),
    ("bnb8",  "cuda_consumer", "vllm"):  CompatEntry(UNTESTED, "bitsandbytes INT8 — prefer AWQ/GPTQ for consumer GPUs.", 8.0, []),
    ("bnb8",  "apple_silicon", "vllm"):  CompatEntry(NO,       "vLLM not supported on Apple Silicon.", None, []),
    ("bnb8",  "cpu_avx2",      "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),
    ("bnb8",  "cpu_arm",       "vllm"):  CompatEntry(NO,       "vLLM requires CUDA.", None, []),

    # ── Ollama backend ────────────────────────────────────────────────────────
    # Ollama uses llama.cpp under the hood. Native GGUF. BF16/FP16 converted.
    # Excellent MPS support on Apple Silicon. CUDA acceleration via CUBLAS.

    ("bf16",  "cuda_sm86+",    "ollama"):  CompatEntry(UNTESTED, "Ollama converts to FP16 internally on CUDA. vLLM is preferred for CUDA+BF16.", 14.0, []),
    ("bf16",  "cuda_sm80",     "ollama"):  CompatEntry(UNTESTED, "Use vLLM for CUDA BF16 production workloads.", 14.0, []),
    ("bf16",  "cuda_sm75",     "ollama"):  CompatEntry(UNTESTED, "Ollama CUDA on sm75 — works but slower than vLLM.", 14.0, []),
    ("bf16",  "cuda_consumer", "ollama"):  CompatEntry(OK,       "Ollama on RTX 3080/4080 — slower than vLLM for large throughput but easier setup", 14.0, ["gaming-pc"]),
    ("bf16",  "apple_silicon", "ollama"):  CompatEntry(OK,       "Ollama on Apple Silicon (Metal) — works well for 7B models; 70B needs unified memory ≥ 64 GB", 14.0, ["macbook"]),
    ("bf16",  "cpu_avx2",      "ollama"):  CompatEntry(OK,       "CPU inference via llama.cpp AVX2 — very slow but works. Only for testing.", None, []),
    ("bf16",  "cpu_arm",       "ollama"):  CompatEntry(OK,       "CPU inference on ARM64 via llama.cpp — limited performance", None, []),

    ("fp16",  "cuda_sm86+",    "ollama"):  CompatEntry(OK,       "FP16 on CUDA via Ollama — GPU-accelerated through CUBLAS", 14.0, []),
    ("fp16",  "cuda_sm80",     "ollama"):  CompatEntry(OK,       "FP16 on A100 via Ollama", 14.0, []),
    ("fp16",  "cuda_sm75",     "ollama"):  CompatEntry(OK,       "FP16 via Ollama on sm75", 14.0, []),
    ("fp16",  "cuda_consumer", "ollama"):  CompatEntry(OK,       "FP16 via Ollama on consumer CUDA", 14.0, ["gaming-pc"]),
    ("fp16",  "apple_silicon", "ollama"):  CompatEntry(OK,       "FP16 via Metal on Apple Silicon", 14.0, ["macbook"]),
    ("fp16",  "cpu_avx2",      "ollama"):  CompatEntry(OK,       "CPU FP16 via llama.cpp — slow", None, []),
    ("fp16",  "cpu_arm",       "ollama"):  CompatEntry(OK,       "CPU FP16 on ARM64 via llama.cpp", None, []),

    ("awq",   "cuda_sm86+",    "ollama"):  CompatEntry(UNTESTED, "Ollama has limited AWQ support; prefer vLLM for AWQ on CUDA.", None, []),
    ("awq",   "cuda_sm80",     "ollama"):  CompatEntry(UNTESTED, "Limited AWQ support in Ollama.", None, []),
    ("awq",   "cuda_sm75",     "ollama"):  CompatEntry(NO,       "AWQ not reliably supported via Ollama on older CUDA.", None, []),
    ("awq",   "cuda_consumer", "ollama"):  CompatEntry(UNTESTED, "AWQ via Ollama on consumer CUDA — prefer vLLM.", None, []),
    ("awq",   "apple_silicon", "ollama"):  CompatEntry(NO,       "AWQ is CUDA-specific. Use GGUF Q4_K_M for Apple Silicon.", None, []),
    ("awq",   "cpu_avx2",      "ollama"):  CompatEntry(NO,       "AWQ requires CUDA-capable GPU.", None, []),
    ("awq",   "cpu_arm",       "ollama"):  CompatEntry(NO,       "AWQ requires CUDA-capable GPU.", None, []),

    ("gptq",  "cuda_sm86+",    "ollama"):  CompatEntry(UNTESTED, "Ollama has limited GPTQ support; prefer vLLM for GPTQ on CUDA.", None, []),
    ("gptq",  "cuda_sm80",     "ollama"):  CompatEntry(UNTESTED, "Limited GPTQ support in Ollama.", None, []),
    ("gptq",  "cuda_sm75",     "ollama"):  CompatEntry(UNTESTED, "Limited GPTQ support via Ollama.", None, []),
    ("gptq",  "cuda_consumer", "ollama"):  CompatEntry(UNTESTED, "GPTQ via Ollama — prefer vLLM or GGUF.", None, []),
    ("gptq",  "apple_silicon", "ollama"):  CompatEntry(NO,       "GPTQ is CUDA-specific. Use GGUF for Apple Silicon.", None, []),
    ("gptq",  "cpu_avx2",      "ollama"):  CompatEntry(NO,       "GPTQ requires CUDA.", None, []),
    ("gptq",  "cpu_arm",       "ollama"):  CompatEntry(NO,       "GPTQ requires CUDA.", None, []),

    ("gguf",  "cuda_sm86+",    "ollama"):  CompatEntry(OK,       "GGUF Q4_K_M on CUDA via Ollama — good balance of quality and speed. Recommended format for Ollama+CUDA.", 5.5, ["gaming-pc"]),
    ("gguf",  "cuda_sm80",     "ollama"):  CompatEntry(OK,       "GGUF on A100 via Ollama — GPU accelerated via CUBLAS", 5.5, []),
    ("gguf",  "cuda_sm75",     "ollama"):  CompatEntry(OK,       "GGUF on sm75 via Ollama — CUBLAS acceleration", 5.5, []),
    ("gguf",  "cuda_consumer", "ollama"):  CompatEntry(OK,       "GGUF Q4_K_M is the recommended format for gaming-tier CUDA via Ollama", 5.5, ["gaming-pc"]),
    ("gguf",  "apple_silicon", "ollama"):  CompatEntry(OK,       "GGUF Q4_K_M on Apple Silicon — best native format via Metal. Fully supported.", 0.0, ["macbook"]),
    ("gguf",  "cpu_avx2",      "ollama"):  CompatEntry(OK,       "GGUF is the recommended CPU format. Q4_K_M balances quality and speed.", None, []),
    ("gguf",  "cpu_arm",       "ollama"):  CompatEntry(OK,       "GGUF on ARM64 — works well, lower performance than Apple Silicon Metal", None, []),

    ("mlx",   "cuda_sm86+",    "ollama"):  CompatEntry(NO,       "MLX is Apple Silicon only.", None, []),
    ("mlx",   "cuda_sm80",     "ollama"):  CompatEntry(NO,       "MLX is Apple Silicon only.", None, []),
    ("mlx",   "cuda_sm75",     "ollama"):  CompatEntry(NO,       "MLX is Apple Silicon only.", None, []),
    ("mlx",   "cuda_consumer", "ollama"):  CompatEntry(NO,       "MLX is Apple Silicon only.", None, []),
    ("mlx",   "apple_silicon", "ollama"):  CompatEntry(NO,       "Ollama does not use MLX directly. Use mlx_lm backend for MLX models.", None, []),
    ("mlx",   "cpu_avx2",      "ollama"):  CompatEntry(NO,       "MLX is Apple Silicon only.", None, []),
    ("mlx",   "cpu_arm",       "ollama"):  CompatEntry(NO,       "MLX is Apple Silicon only.", None, []),

    ("bnb4",  "cuda_sm86+",    "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama. Use GGUF Q4_K_M instead.", None, []),
    ("bnb4",  "cuda_sm80",     "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),
    ("bnb4",  "cuda_sm75",     "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),
    ("bnb4",  "cuda_consumer", "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama. Use GGUF.", None, []),
    ("bnb4",  "apple_silicon", "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama or on Apple Silicon.", None, []),
    ("bnb4",  "cpu_avx2",      "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),
    ("bnb4",  "cpu_arm",       "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),

    ("bnb8",  "cuda_sm86+",    "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),
    ("bnb8",  "cuda_sm80",     "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),
    ("bnb8",  "cuda_sm75",     "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),
    ("bnb8",  "cuda_consumer", "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),
    ("bnb8",  "apple_silicon", "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),
    ("bnb8",  "cpu_avx2",      "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),
    ("bnb8",  "cpu_arm",       "ollama"):  CompatEntry(NO,       "bitsandbytes not supported in Ollama.", None, []),

    # ── mlx_lm backend ───────────────────────────────────────────────────────
    # mlx_lm is Apple Silicon only. Fastest inference on M-series hardware.

    ("bf16",  "apple_silicon", "mlx_lm"):  CompatEntry(OK,       "BF16 via mlx_lm on Apple Silicon — fastest option on M-series hardware", 0.0, ["macbook"]),
    ("fp16",  "apple_silicon", "mlx_lm"):  CompatEntry(OK,       "FP16 via mlx_lm on Apple Silicon", 0.0, ["macbook"]),
    ("mlx",   "apple_silicon", "mlx_lm"):  CompatEntry(OK,       "Native MLX quantised format — best performance on Apple Silicon", 0.0, ["macbook"]),
    ("gguf",  "apple_silicon", "mlx_lm"):  CompatEntry(UNTESTED, "mlx_lm can convert GGUF but prefer native MLX format", None, []),
    ("awq",   "apple_silicon", "mlx_lm"):  CompatEntry(NO,       "AWQ is CUDA-specific. Use MLX or GGUF for Apple Silicon.", None, []),
    ("gptq",  "apple_silicon", "mlx_lm"):  CompatEntry(NO,       "GPTQ is CUDA-specific. Use MLX or GGUF for Apple Silicon.", None, []),
    ("bnb4",  "apple_silicon", "mlx_lm"):  CompatEntry(NO,       "bitsandbytes requires CUDA. Not supported on Apple Silicon.", None, []),
    ("bnb8",  "apple_silicon", "mlx_lm"):  CompatEntry(NO,       "bitsandbytes requires CUDA.", None, []),

    # mlx_lm does not run on non-Apple-Silicon hardware
    **{
        (fmt, hw, "mlx_lm"): CompatEntry(NO, "mlx_lm is Apple Silicon only.", None, [])
        for fmt in ("bf16", "fp16", "awq", "gptq", "gguf", "mlx", "bnb4", "bnb8")
        for hw in ("cuda_sm86+", "cuda_sm80", "cuda_sm75", "cuda_consumer", "cpu_avx2", "cpu_arm")
    },

    # ── llamacpp backend ──────────────────────────────────────────────────────
    # llamacpp (direct, without Ollama wrapper) — CPU and CUDA via cuBLAS.

    ("gguf",  "cuda_sm86+",    "llamacpp"):  CompatEntry(OK,       "GGUF + CUDA via cuBLAS in llama.cpp — fast on H100/4090", 5.5, []),
    ("gguf",  "cuda_sm80",     "llamacpp"):  CompatEntry(OK,       "GGUF + cuBLAS on A100", 5.5, []),
    ("gguf",  "cuda_sm75",     "llamacpp"):  CompatEntry(OK,       "GGUF + cuBLAS on T4/sm75", 5.5, []),
    ("gguf",  "cuda_consumer", "llamacpp"):  CompatEntry(OK,       "GGUF + cuBLAS on consumer CUDA GPUs", 5.5, []),
    ("gguf",  "apple_silicon", "llamacpp"):  CompatEntry(OK,       "GGUF + Metal in llama.cpp — similar to Ollama under the hood", 0.0, ["macbook"]),
    ("gguf",  "cpu_avx2",      "llamacpp"):  CompatEntry(OK,       "GGUF CPU AVX2 — recommended for CPU-only inference", None, []),
    ("gguf",  "cpu_arm",       "llamacpp"):  CompatEntry(OK,       "GGUF ARM64 via llama.cpp NEON", None, []),

    ("bf16",  "cuda_sm86+",    "llamacpp"):  CompatEntry(UNTESTED, "llama.cpp BF16 on CUDA — possible but untested in AUA CI", 14.0, []),
    ("bf16",  "cpu_avx2",      "llamacpp"):  CompatEntry(OK,       "llama.cpp BF16 on CPU AVX2 — slow but accurate", None, []),
    ("bf16",  "apple_silicon", "llamacpp"):  CompatEntry(UNTESTED, "BF16 via Metal in llama.cpp — untested; prefer GGUF", None, []),

    ("fp16",  "cuda_sm86+",    "llamacpp"):  CompatEntry(OK,       "FP16 via llama.cpp cuBLAS on H100/A100", 14.0, []),
    ("fp16",  "cpu_avx2",      "llamacpp"):  CompatEntry(OK,       "FP16 on CPU via llama.cpp — very slow; prefer Q4_K_M GGUF", None, []),
    ("fp16",  "apple_silicon", "llamacpp"):  CompatEntry(OK,       "FP16 via Metal in llama.cpp", 0.0, []),

    # Other formats on llamacpp default to unsupported/untested
    **{
        (fmt, hw, "llamacpp"): CompatEntry(NO, f"{fmt} not natively supported in llama.cpp. Use GGUF instead.", None, [])
        for fmt in ("awq", "gptq", "mlx", "bnb4", "bnb8")
        for hw in ("cuda_sm86+", "cuda_sm80", "cuda_sm75", "cuda_consumer", "apple_silicon", "cpu_avx2", "cpu_arm")
    },
    **{
        (fmt, hw, "llamacpp"): CompatEntry(NO, f"{fmt} not natively supported in llama.cpp. Use GGUF instead.", None, [])
        for fmt in ("bf16", "fp16")
        for hw in ("cuda_sm75", "cuda_sm80", "cuda_consumer")
    },

    # ── transformers backend ──────────────────────────────────────────────────
    # HuggingFace transformers — slower but most format-flexible for prototyping.

    ("bf16",  "cuda_sm86+",    "transformers"):  CompatEntry(OK,       "transformers BF16 on H100/A100 — great for prototyping; prefer vLLM for production throughput", 14.0, []),
    ("bf16",  "cuda_sm80",     "transformers"):  CompatEntry(OK,       "transformers BF16 on A100", 14.0, []),
    ("bf16",  "cuda_consumer", "transformers"):  CompatEntry(OK,       "transformers BF16 on consumer CUDA", 14.0, []),
    ("bf16",  "apple_silicon", "transformers"):  CompatEntry(OK,       "transformers BF16 via MPS on Apple Silicon — slow vs mlx_lm/ollama but works", None, ["macbook"]),
    ("bf16",  "cpu_avx2",      "transformers"):  CompatEntry(OK,       "transformers BF16 CPU — very slow; only for testing", None, []),
    ("fp16",  "cuda_sm86+",    "transformers"):  CompatEntry(OK,       "transformers FP16 on CUDA", 14.0, []),
    ("fp16",  "apple_silicon", "transformers"):  CompatEntry(OK,       "transformers FP16 via MPS", None, []),
    ("awq",   "cuda_sm86+",    "transformers"):  CompatEntry(OK,       "transformers + AutoAWQ — install with pip install autoawq", 5.5, []),
    ("gptq",  "cuda_sm86+",    "transformers"):  CompatEntry(OK,       "transformers + AutoGPTQ — install with pip install auto-gptq", 5.5, []),
    ("bnb4",  "cuda_sm86+",    "transformers"):  CompatEntry(OK,       "transformers + bitsandbytes NF4 — install with pip install bitsandbytes", 4.5, []),
    ("bnb8",  "cuda_sm86+",    "transformers"):  CompatEntry(OK,       "transformers + bitsandbytes INT8", 8.0, []),
    ("bnb4",  "cuda_consumer", "transformers"):  CompatEntry(OK,       "bitsandbytes NF4 on consumer CUDA — useful for 13B models on 24 GB", 4.5, ["single-4090"]),
    ("bnb8",  "cuda_consumer", "transformers"):  CompatEntry(OK,       "bitsandbytes INT8 on consumer CUDA", 8.0, ["single-4090"]),
}
# fmt: on


# ── Public API ────────────────────────────────────────────────────────────────

MODEL_FORMATS = ("bf16", "fp16", "awq", "gptq", "gguf", "mlx", "bnb4", "bnb8")
HARDWARE_TIERS = (
    "cuda_sm86+",
    "cuda_sm80",
    "cuda_sm75",
    "cuda_consumer",
    "apple_silicon",
    "cpu_avx2",
    "cpu_arm",
)
BACKENDS = ("vllm", "ollama", "mlx_lm", "llamacpp", "transformers")

# Map aua --tier aliases to hardware tier identifiers
TIER_TO_HARDWARE: dict[str, str] = {
    "h100-cluster": "cuda_sm86+",
    "a100-cluster": "cuda_sm80",
    "a100": "cuda_sm80",
    "single-4090": "cuda_consumer",
    "quad-4090": "cuda_consumer",
    "gaming-pc": "cuda_consumer",
    "gaming": "cuda_consumer",
    "macbook": "apple_silicon",
}

# Map backend string from AUAConfig to canonical backend identifier
BACKEND_ALIASES: dict[str, str] = {
    "vllm": "vllm",
    "ollama": "ollama",
    "mlx_lm": "mlx_lm",
    "mlx-lm": "mlx_lm",
    "llamacpp": "llamacpp",
    "llama.cpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "transformers": "transformers",
    "hf": "transformers",
}


def lookup(
    model_format: str,
    hardware: str,
    backend: str,
) -> CompatEntry | None:
    """
    Look up a compatibility entry.

    Args:
        model_format: one of MODEL_FORMATS
        hardware:     one of HARDWARE_TIERS or a tier alias (e.g. "gaming-pc")
        backend:      one of BACKENDS or a backend alias

    Returns:
        CompatEntry or None if the combination is not in the matrix.
    """
    hw = TIER_TO_HARDWARE.get(hardware, hardware)
    be = BACKEND_ALIASES.get(backend, backend)
    return MATRIX.get((model_format, hw, be))


def check_config(model_format: str, hardware: str, backend: str) -> tuple[str, str]:
    """
    Check a single (model_format, hardware, backend) combination.

    Returns:
        (status, message) where status is "supported" | "untested" | "unsupported" | "unknown"
    """
    entry = lookup(model_format, hardware, backend)
    if entry is None:
        return "unknown", (
            f"Combination ({model_format}, {hardware}, {backend}) not in compatibility matrix. "
            "Treat as untested."
        )
    return entry.status, entry.notes


def infer_model_format(model_name_or_path: str, backend: str | None = None) -> str:
    """
    Infer model format from a model name/path string.

    Uses naming conventions common on HuggingFace Hub. When an explicit
    quantisation suffix is absent, the backend (if given) determines the
    native format: Ollama and llama.cpp serve GGUF; mlx_lm serves MLX.

    Args:
        model_name_or_path: model tag, HF id, or path
        backend:            optional backend hint (ollama/llamacpp/mlx_lm/vllm)

    Returns "unknown" only when no suffix matches and the backend is
    unknown or doesn't imply a native format (e.g. vllm/transformers).
    """
    name = model_name_or_path.lower()
    if ".gguf" in name or "gguf" in name:
        return "gguf"
    if "-awq" in name or ".awq" in name or "awq" in name:
        return "awq"
    if "-gptq" in name or ".gptq" in name or "gptq" in name:
        return "gptq"
    if "-mlx" in name or ".mlx" in name or "mlx" in name:
        return "mlx"
    if "bnb4" in name or "nf4" in name or "int4" in name:
        return "bnb4"
    if "bnb8" in name or "int8" in name:
        return "bnb8"
    if "fp16" in name:
        return "fp16"
    if "bf16" in name:
        return "bf16"
    # No quantisation suffix → infer from the backend's native format
    be = BACKEND_ALIASES.get((backend or "").lower(), (backend or "").lower())
    if be in ("ollama", "llamacpp"):
        return "gguf"  # Ollama and llama.cpp always serve GGUF under the hood
    if be == "mlx_lm":
        return "mlx"
    return "unknown"


def to_markdown() -> str:
    """
    Render the matrix as a Markdown table grouped by backend.
    Used by aua doctor --compat-matrix and published to docs.
    """
    lines = ["# AUA Compatibility Matrix", ""]
    icon = {"supported": "✅", "untested": "⚠️", "unsupported": "❌"}

    for backend in BACKENDS:
        lines += [f"## {backend}", ""]
        lines += ["| Format | " + " | ".join(HARDWARE_TIERS) + " |"]
        lines += ["|--------|" + "|".join(["-----"] * len(HARDWARE_TIERS)) + "|"]
        for fmt in MODEL_FORMATS:
            row = [f"`{fmt}`"]
            for hw in HARDWARE_TIERS:
                entry = MATRIX.get((fmt, hw, backend))
                if entry is None:
                    row.append("—")
                else:
                    row.append(icon.get(entry.status, "?"))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines += [
        "**Legend:** ✅ Supported  ⚠️ Untested  ❌ Unsupported",
        "",
        "_VRAM estimates are for a 7B parameter model. Scale proportionally for larger models._",
    ]
    return "\n".join(lines)
