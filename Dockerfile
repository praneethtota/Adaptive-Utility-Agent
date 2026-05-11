# ── AUA Framework — multi-stage Docker image ──────────────────────────────────
#
# Stages:
#   base      Python 3.11 slim + runtime deps only
#   dev       base + dev/test deps (for CI)
#   prod      base + wheel install (minimal image for deployment)
#
# Build:
#   docker build --target prod -t aua:0.7.0b0 .
#   docker build --target dev  -t aua-dev:latest .
#
# Run (CPU/Ollama mode):
#   docker run -p 8000:8000 -v $(pwd)/aua_config.yaml:/app/aua_config.yaml aua:0.7.0b0
#
# Note: GPU (vLLM) deployment requires nvidia-docker runtime and --gpus all flag.
#       See docker-compose.gpu.yml for the full GPU stack.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

LABEL org.opencontainers.image.title="AUA Framework"
LABEL org.opencontainers.image.description="Adaptive Utility Agents — self-optimizing AI specialist framework"
LABEL org.opencontainers.image.source="https://github.com/praneethtota/Adaptive-Utility-Agent"
LABEL org.opencontainers.image.licenses="GPL-3.0"

# System deps required at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    lsof \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime deps first (cache layer — rarely changes)
COPY pyproject.toml README.md ./
COPY aua/version.py aua/version.py
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e "." \
    && rm -rf /root/.cache/pip

# ── Dev stage (CI + local development) ────────────────────────────────────────
FROM base AS dev

RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["pytest", "-q"]

# ── Prod stage (minimal deployment image) ─────────────────────────────────────
FROM base AS prod

# Copy full source (wheel already installed in base via -e .)
COPY aua/ aua/
COPY aua_config.yaml ./

# Runtime directories
RUN mkdir -p .aua/logs .aua/pids .aua/state .aua/checkpoints \
             models dpo_pairs results logs

# Non-root user for security
RUN groupadd -r aua && useradd -r -g aua -d /app -s /bin/false aua \
    && chown -R aua:aua /app

USER aua

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Default: start router only (specialists run separately or via Ollama)
CMD ["aua", "serve", "--router-only"]
