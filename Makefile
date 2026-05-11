# ── AUA Framework — developer targets ─────────────────────────────────────────
#
# make install      install in editable mode with dev deps
# make check        run all checks (lint + typecheck + test) — CI equivalent
# make test         run pytest only
# make lint         ruff + black check (no fix)
# make format       ruff fix + black + isort (modifies files)
# make typecheck    mypy
# make doctor       aua doctor against the default config
# make serve-dry    print startup commands without executing
# make version      show current framework version
# make build        build wheel
# make clean        remove build artifacts
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: install check test lint format typecheck doctor serve-dry version build clean help

# Default: show help
.DEFAULT_GOAL := help

install:
	pip install -e ".[dev]"

# Full CI equivalent: runs everything in the same order as .github/workflows/ci.yml
check: lint typecheck test
	@echo ""
	@echo "✓ All checks passed"

test:
	pytest -q

lint:
	ruff check aua tests
	black --check --line-length 100 aua tests
	isort --check-only --profile black --line-length 100 aua tests

format:
	ruff check --fix --line-length 100 aua tests
	black --line-length 100 aua tests
	isort --profile black --line-length 100 aua tests

typecheck:
	mypy aua --ignore-missing-imports

doctor:
	aua doctor

serve-dry:
	aua serve --dry-run

version:
	@python3 -c "from aua.version import __version__; print(__version__)"

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "AUA Framework — available targets:"
	@echo ""
	@echo "  make install     install in editable mode with dev deps"
	@echo "  make check       full CI check (lint + typecheck + test)"
	@echo "  make test        run pytest only"
	@echo "  make lint        ruff + black check"
	@echo "  make format      ruff fix + black + isort (modifies files)"
	@echo "  make typecheck   mypy"
	@echo "  make doctor      aua doctor"
	@echo "  make serve-dry   print startup commands"
	@echo "  make version     show current version"
	@echo "  make build       build wheel"
	@echo "  make clean       remove build artifacts"
