# mlb_dfs_solver

Fast roster optimization: randomized greedy lineup construction and lazy-greedy submodular
portfolio selection, implemented in Rust.

> **Status: implemented, not yet published.** The code, tests, benchmarks and docs
> are complete; `0.0.1.dev0` is the placeholder version until the first PyPI
> release.

## The problem

Two different problems, usually conflated:

1. **Build one valid roster.** Pick players under a salary cap, filling positional slots, subject
   to group limits ("at most 6 from one team"). This is an integer program, and an ILP solver
   answers it exactly.

   Caps, minimums on how many distinct key values a lineup uses ("players from at least two
   different games"), and stacks ("at least four hitters from one team") are all expressible.
2. **Build a *portfolio* of rosters.** Pick 150 lineups that collectively do well across
   simulated outcomes. Optimality per lineup is close to worthless here — 150 optimal lineups are
   150 nearly identical lineups. What matters is diverse coverage of the outcome space.

For (2), asking an ILP for 150 solutions is both slow and the wrong objective. `mlb_dfs_solver`
generates a large randomized-greedy candidate pool and then selects from it by **lazy-greedy
submodular maximization** — Minoux's lazy evaluation, stochastic greedy for large pools, and a
CVaR-upside objective that scores a lineup by how much it improves the portfolio's *tail*, not
its mean.

## What's in it

- Randomized greedy construction with salary-repair backtracking, over an arbitrary roster
  specification — slots, position eligibility as bitmasks, salary cap and floor, and generic
  group constraints.
- Per-slot score and salary multipliers, so showdown / single-game formats (a captain worth
  1.5× and costing 1.5×) are the same code path as a classic roster.
- Optional pairwise conflicts — "no hitters against my starting pitcher", expressed as a join
  between two player keys. Off unless you ask: it is a strategy, not a contest rule, and a
  contrarian deliberately wants that correlation.
- Locks (players forced into every lineup, matched to slots properly rather than greedily) and
  per-player exposure caps across the portfolio.
- Stacking, with the stacked team drawn per attempt so a portfolio spreads across teams instead
  of piling onto one.
- Lazy-greedy submodular selection with a CVaR-upside objective, ownership/leverage discounting,
  and a diversity penalty.
- Sport presets (`mlb_dfs_solver.presets`) shipped as data, not hardcoded branches.
- Runtime AVX2 dispatch. Wheels are built portably; `mlb_dfs_solver.active_isa()` reports which path
  your machine took.

## Benchmarks

Measured against MILP formulations of the same problem (PuLP/CBC and OR-Tools), which live in
[`benchmarks/baselines/`](https://github.com/kfreisen/mlb-dfs-solver/benchmarks/baselines) as real, tested, importable code — along with a
pure-Python transcription of the greedy algorithm that serves as the parity oracle.

Numbers and the hardware they were measured on: <https://kfreisen.github.io/mlb-dfs-solver/benchmarks/>

## Install

```bash
pip install mlb_dfs_solver
```

Binary wheels are published for Linux (x86-64, aarch64), macOS (arm64, x86-64), and Windows
(x86-64). Installing from source requires a Rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## License

Apache-2.0.
