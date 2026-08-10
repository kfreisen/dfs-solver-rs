# Thin wrappers over uv. Nothing here is required to build, test or publish;
# CI calls the underlying commands directly.

# Pinned to match .pre-commit-config.yaml. An unpinned `uvx ruff` picks the latest
# release and the formatter's output changes between versions, so `make lint`
# passing while the pre-commit hook fails in CI is not hypothetical.
RUFF := ruff@0.14.5

.PHONY: help test lint fmt typecheck bench bench-quick examples docs docs-build lock clean

help:
	@echo "make test         run the tests with coverage"
	@echo "make lint         ruff check + format check"
	@echo "make fmt          ruff format + autofix"
	@echo "make typecheck    mypy"
	@echo "make bench        full benchmark run -> benchmarks/results/"
	@echo "make bench-quick  smoke-run the benchmarks, no timings recorded"
	@echo "make examples     execute every marimo example"
	@echo "make docs         serve the docs site on :8000"
	@echo "make lock         re-lock dependencies"

test:
	uv run pytest --cov --cov-report=term-missing

lint:
	uvx $(RUFF) check .
	uvx $(RUFF) format --check .

fmt:
	uvx $(RUFF) format .
	uvx $(RUFF) check --fix .

typecheck:
	uv run mypy

bench:
	uv run --group bench pytest benchmarks/ --benchmark-json=.benchmark.json --benchmark-only
	uv run --no-project --python 3.12 --with psutil tools/bench_report.py .benchmark.json --write

bench-quick:
	uv run --group bench pytest benchmarks/ -q --benchmark-disable

examples:
	@for nb in examples/*.py; do echo "--- $$nb"; uv run marimo export html "$$nb" -o /dev/null || exit 1; done

docs:
	uv run --project docs mkdocs serve

docs-build:
	uv run --project docs mkdocs build --strict

lock:
	uv lock

clean:
	find . -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf site .benchmark.json rust/target
