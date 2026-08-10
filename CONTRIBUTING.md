# Contributing

## Setup

You need [`uv`](https://docs.astral.sh/uv/) — it manages Python versions itself, so
nothing else is required.

Building the extension also needs a Rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

```bash
uv sync --all-extras
uv run pytest
```

Install the hooks once:

```bash
uvx pre-commit install
```

**Run `pre-commit`, not bare `ruff`.** The hooks pin a ruff version; a bare
`uvx ruff` resolves to the latest release, and the formatter's output changes
between versions. `make lint` uses the pinned version for the same reason.

## The bar for a change

- **Tests.** The coverage threshold is set in `[tool.coverage.report] fail_under`.
  It is not negotiable downward; if code is hard to cover, that is usually a design
  signal.
- **Types.** `mypy --strict` over the source. The package ships `py.typed`.
- **Docstrings.** Google convention, enforced by ruff's `D` rules. mkdocstrings
  renders these directly into the published docs, so they are user-facing text.

## Benchmarks

`benchmarks/baselines/` holds the *previous* implementation as real, imported,
tested code. This is the point of the exercise: a speedup claim only means
something if the slow version still runs and still produces the same answer.

Two rules follow:

1. **`tests/test_parity.py` is the most important test here.** It runs the baseline
   and the current implementation on the same inputs and asserts they agree. If you
   change the fast path, this is what catches you having changed its behavior rather
   than its speed.
2. **CI never asserts on wall-clock time.** Shared runners vary by roughly 2×, and a
   perf gate that flakes is a perf gate everyone learns to ignore. CI runs the
   benchmark suite with timing disabled, to prove the benchmark code still executes.

Real numbers are produced on a known machine and committed by hand:

```bash
make bench
```

That writes `benchmarks/results/<hardware-id>/<date>-<sha>.json`, including a
hardware block. The docs site renders those files at build time, so published tables
cannot drift from the committed data. Include the hardware you ran on in the PR.

## Examples

Examples are [marimo](https://marimo.io) notebooks stored as plain Python under
`examples/`. CI executes them via `marimo export html`, which fails on any
exception — so an example must run in under a minute, must not touch the network,
and must read only from small committed fixtures in `examples/data/`.

## Releasing

Tag `v<version>`, e.g. `v0.2.0`. The release workflow verifies the tag matches
`project.version`, builds, and publishes to PyPI via Trusted Publishing. The `pypi`
GitHub Environment requires a manual approval, so a stray tag push cannot publish on
its own.
