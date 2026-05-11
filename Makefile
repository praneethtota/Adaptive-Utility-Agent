.PHONY: install test lint format typecheck doctor serve-dry build clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check aua tests
	black --check aua tests

format:
	ruff check --fix aua tests
	black --line-length 100 aua tests
	isort --profile black --line-length 100 aua tests

typecheck:
	mypy aua

doctor:
	aua doctor

serve-dry:
	aua serve --dry-run

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info __pycache__ .mypy_cache .ruff_cache
