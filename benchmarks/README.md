# mlb_dfs_solver benchmarks

Not pytest. These are long by design — the contest-scale scenario alone is ten thousand
sequential CP-SAT solves — and a test runner was the wrong home for them twice over: CI executed
the suite on every push, and the portfolio size was pinned at 25 lineups because that was the
largest a solver could finish inside a test session. A harness limit had become the published
experiment.

```bash
task bench                      # every scenario, writes results/
python benchmarks/run.py --help # scenario selection, --no-solver, --smoke
```

`run.py` writes `results/<hardware-id>/<date>-<sha>.json`, which is committed. `task bench` then
runs `tools/bench_report.py --write`, which renders it into the README. The docs site renders the
same file through `docs/hooks/bench_tables.py`. Nobody types a number into a document.

## What is measured

Two things: **how long generation takes**, and **what it generates** — player coverage, exposure
concentration, the spread of projections against a proven optimum.

Nothing is scored against a simulated contest. That needs a simulator and a field model, this
package supplies neither, and a benchmark shipping its own would be grading its own fixture.

## The scenarios

Each is a configuration somebody actually plays, and they are cumulative — every row adds one
thing a player turns on. `scenarios.py` defines them and documents what is *not* expressible,
which is currently the secondary stack every real MME player wants.

Sizes come from the game: 150 entries is a DraftKings MLB classic maximum, cash is played a few
entries deep, and the contest-scale row asks for 10,000 because that is what a field simulation
or a candidate pool needs.

## The baselines

`baselines/` holds the MILP formulations and a pure-Python transcription of the greedy algorithm.
They are imported, executed on every run, and covered by tests — `tests/test_parity.py` asserts
the transcription and the kernel agree, which is what makes a speedup number mean anything.

## CI

CI runs `run.py --smoke`: one scenario, five lineups, nothing written. It proves the harness
still executes and produces no number. Shared runners vary by roughly 2×, and a flaky performance
gate is a gate people learn to ignore.
