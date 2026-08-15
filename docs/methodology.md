# How benchmarks work

Every package here claims to be faster than something. This page describes what those claims
rest on, because a speedup number with no stated method is decoration.

## The baseline is real code

This repository has a `benchmarks/baselines/` directory containing the implementation it replaced —
or, where the fast path was written first, a deliberately straightforward implementation of the
same algorithm. These are not archived text. They are imported, executed on every benchmark run,
and covered by tests.

Keeping them alive costs something, and it buys the only thing that makes the comparison
meaningful.

## Parity is asserted before speed is measured

`tests/test_parity.py` runs the baseline and the optimized implementation on the same inputs and
asserts they agree. It runs on every change, in ordinary CI, on small inputs.

This is the load-bearing test here. A fast path that has quietly changed its behavior
will produce an excellent benchmark number, and only parity catches it. Where exact equality is
not the right assertion — a blocking scheme is allowed to be a search heuristic — the parity test
measures and bounds the difference instead, and the benchmark reports it alongside the timing.
A matcher that is 100× faster and loses 4% of true matches is a regression, so where that
applies the benchmark reports quality alongside speed.

## CI never times anything

Shared CI runners vary by roughly 2× run to run, depending on what else is on the host. A
performance assertion in that environment fails for reasons unrelated to the code, and a check
that fails for unrelated reasons is a check people learn to ignore.

So CI does two things instead:

1. Runs the parity tests, which is where the correctness of the claim actually lives.
2. Runs the benchmark suite with timing disabled, to prove the benchmark code still executes.

Neither asserts a duration.

## Numbers come from named hardware

Real measurements are taken on a known machine:

```bash
task bench
```

That writes `benchmarks/results/<hardware-id>/<date>-<sha>.json`, which is
committed. Each file records the CPU model, core count, RAM, OS, Python version, package version,
and the commit it was measured against. Results are grouped by machine because a speedup is a
claim about a machine, not a universal constant.

Timings use [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/), which handles warmup
and round calibration, rather than a hand-rolled `perf_counter` loop.

The tables on the benchmark page are rendered from those JSON files when the site is
built. Nobody types a number into a document, so no document can disagree with the data.

## What is measured, and what is not

These benchmarks measure **how long generation takes** and **what it generates**.
Nothing more.

They do not report what a portfolio would have won, cashed, or returned. Those
numbers require a simulator and a model of the field, this package supplies
neither, and a benchmark that wrote its own would be measuring that fixture. The
in-the-money figures this page used to define were exactly that: a quantile of a
synthetic field, scored against Gaussian player outcomes, both written here. They
moved when the fixture was rewritten, which is the tell.

What survives is enough to tell the two approaches apart, and every figure can be
checked by reading the lineups.

## What the columns mean

### The optimum

The single highest-projection roster satisfying every constraint, proved by
CP-SAT with no cuts and no time pressure. One lineup, not a portfolio. It is the
yardstick the ratios divide by.

It is projection arithmetic on the inputs both implementations were given, so it
measures how well each searched the space — not whether the projections were any
good.

### `players used`

How many distinct players appear anywhere in the draw, out of the slate. A method
returning 150 rosters built from 33 players is exploring one corner of the slate;
one returning 150 from 117 is not.

The single most legible difference between the two approaches, and the cheapest
to verify.

### `projection, min → max`

The lowest and highest entry in the draw, each as a fraction of the optimum. A
solver enumerating by projection produces a very narrow band — its worst entry is
close to its best — because "second best" means "the best one with a player
swapped". Randomized construction produces a wide one.

### `top player's share`

The fraction of lineups containing the most-used player. Says how concentrated
the draw is on a single name, which is what an exposure cap exists to control.

Read this rather than the cap you requested: the cap is computed against lineups
*requested*, so when yield falls short the realized share runs higher than asked.

### `returned` / yield

How many of the requested lineups came back, **reported for both
implementations**. A solver is complete and returns all of them. Randomized
construction is not: when constraints bite it runs out of legal rosters it has
not already found.

Both are shown because a speedup dividing our time for 14 lineups by a solver's
time for 25 is not a ratio.

### `no-good cuts`

How the solver is made to produce a different lineup each time. After it returns a
roster, a constraint is added saying at least one of those players must be dropped
next time. Solve again for the second-best roster, then the third.

It is the standard way to enumerate solutions in order, and it is why the solver's
draw looks the way it does. It is not the only way to get diversity out of a
solver — an overlap constraint bounding how many players a new lineup may share
with each earlier one is the other common technique, and it is a fairer
comparison at 150 entries. It is also markedly slower, which is why the
contest-scale table exists.

### Contest scale

Both implementations are run to a full 10,000-lineup draw. Nothing in that table
is extrapolated.

An earlier version measured the solver on a 25-lineup prefix and multiplied,
assuming per-lineup cost is flat in the count. It is not: every solve carries one
more no-good cut than the last, so the rate degrades across ten thousand of them
and the extrapolation understated the real figure. The per-lineup column exists
so that degradation is visible rather than inferred — compare it against the
solver's per-lineup cost at 25 lineups on the constraint ladder.

This is by a wide margin the slowest case in the suite. It is worth the wall
clock, because the number it replaced was wrong.

## What a benchmark here does not tell you

The cases are the ones these packages were built for, at the sizes they were built for. They are
described in each result file's `params`, and they are the honest scope of the claim. A different
workload can invert any of these results, and if it does, that is a bug report worth filing.
