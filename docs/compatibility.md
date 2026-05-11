# AUA Framework v1 — Compatibility Matrix

**Version:** 1.0.0

---

## Python

| Version | Status | Notes |
|---|---|---|
| 3.10 | ✓ Supported | Minimum version |
| 3.11 | ✓ Supported | Recommended |
| 3.12 | ✓ Supported | Tested in CI |
| 3.9 | ✗ Not supported | f-string syntax incompatible |
| 3.13 | ⚠ Experimental | Not yet in CI matrix |

---

## Operating Systems

| OS | Backend | Notes |
|---|---|---|
| macOS (Apple Silicon M1/M2/M3/M4) | Ollama only | vLLM has no macOS support |
| macOS (Intel) | Ollama only | CPU inference only |
| Ubuntu 20.04+ | vLLM + Ollama | Recommended for production |
| Debian 11+ | vLLM + Ollama | |
| RHEL/Rocky 8+ | vLLM + Ollama | |
| Windows | Not tested | Use WSL2 + Ubuntu |

---

## GPU & VRAM

| GPU | VRAM | Tier | Max simultaneous specialists |
|---|---|---|---|
| Apple M-series (unified) | 16–128 GB | macbook | 3 (via Ollama, sequential) |
| NVIDIA RTX 4090 | 24 GB | single-4090 | 3 (AWQ, concurrent) |
| NVIDIA RTX 3090/4080 | 24 GB / 16 GB | single-4090 | 2–3 (may need lower util) |
| 4× NVIDIA RTX 4090 | 96 GB total | quad-4090 | 6–8 |
| NVIDIA A100 80 GB | 80 GB | a100-cluster | 4–6 (fp16) |
| NVIDIA H100 80 GB | 80 GB | a100-cluster | 4–6 (fp16) |

**VRAM estimates (AWQ 4-bit):**
- 3B model: ~2.5 GB
- 7B model: ~5 GB
- 14B model: ~9 GB
- 32B model: ~20 GB

---

## LLM Backends

### Ollama

| Version | Status | Notes |
|---|---|---|
| 0.3.x | ✓ Supported | |
| 0.4.x | ✓ Supported | Recommended |
| 0.5.x+ | ✓ Supported | |

**Supported model formats via Ollama:** GGUF (Q4, Q5, Q8), fp16

### vLLM

| Version | Status | Notes |
|---|---|---|
| 0.4.x | ✓ Supported | |
| 0.5.x | ✓ Supported | Recommended |
| 0.6.x+ | ✓ Supported | |

**Supported model formats via vLLM:** AWQ, GPTQ, fp16, bf16

---

## Model Formats

| Format | Ollama | vLLM | Notes |
|---|---|---|---|
| GGUF (Q4_K_M) | ✓ | ✗ | Ollama default |
| GGUF (Q5_K_M) | ✓ | ✗ | Higher quality |
| AWQ | ✗ | ✓ | Fastest on GPU |
| GPTQ | ✗ | ✓ | |
| fp16 | ✓ (via Ollama) | ✓ | Full precision |
| bf16 | ✗ | ✓ | A100/H100 only |

---

## CUDA

| CUDA Version | Status | Notes |
|---|---|---|
| 11.8 | ✓ Supported | |
| 12.0 | ✓ Supported | |
| 12.1 | ✓ Supported | Recommended |
| 12.2+ | ✓ Supported | |

Requires: `nvidia-driver >= 520`

---

## State Store

| Backend | Status | Notes |
|---|---|---|
| SQLite (WAL) | ✓ Default | All deployments |
| Files (JSONL) | ✓ Legacy | v0.7 compatibility |
| PostgreSQL 14+ | ✓ Supported | Team/Enterprise profiles |
| PostgreSQL 13 | ⚠ Partial | No JSON operators |
| MySQL/MariaDB | ✗ Not supported | |

---

## Observability

| Tool | Version | Status |
|---|---|---|
| Prometheus | 2.x / 3.x | ✓ Supported |
| Grafana | 9.x / 10.x / 13.x | ✓ Supported |
| OpenTelemetry Collector | 0.80+ | ✓ Supported |
| Datadog | Any (via OTEL) | ✓ Supported |
| Jaeger | 1.x | ✓ Supported |

---

## Docker

| Tool | Version | Status |
|---|---|---|
| Docker Engine | 24+ | ✓ Supported |
| Docker Desktop (Mac) | 4.x | ✓ Supported |
| Docker Compose | v2.x | ✓ Required |
| Podman | 4+ | ⚠ Experimental |

---

## Chat UI

| Browser | Status |
|---|---|
| Chrome / Chromium 110+ | ✓ Supported |
| Firefox 110+ | ✓ Supported |
| Safari 16+ | ✓ Supported |
| Edge 110+ | ✓ Supported |

**Runtime:** Node.js 18+ required for `aua ui` / `aua serve --with-ui`

---

## Python Dependencies (key packages)

| Package | Min version | Notes |
|---|---|---|
| fastapi | 0.100+ | |
| uvicorn | 0.20+ | |
| httpx | 0.25+ | |
| pydantic | 2.0+ | v1 not supported |
| click | 8.0+ | |
| rich | 13.0+ | |
| pyyaml | 6.0+ | |
| cryptography | 41.0+ | Optional — certs + encryption |
| prometheus-client | 0.17+ | Optional — metrics |
| opentelemetry-sdk | 1.20+ | Optional — aua[otel] |

---

## Tested Hardware (v1.0)

| Hardware | OS | Backend | Status |
|---|---|---|---|
| MacBook Pro M1 Max (32 GB) | macOS 14 | Ollama | ✓ Primary dev platform |
| MacBook Pro M2 (16 GB) | macOS 14 | Ollama | ✓ |
| Desktop RTX 4090 | Ubuntu 22.04 | vLLM | ✓ |
| RunPod RTX 4090 (24 GB) | Ubuntu 22.04 | vLLM | ✓ CI validation |
