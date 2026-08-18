# How benchmarks work

This package claims to be faster than something. This page describes what that claim
rests on, because a speedup number with no stated method is decoration.

## The baseline is real code

This repository has a `benchmarks/baselines/` directory containing the implementation it replaced —
or, where the fast path was written first, a deliberately straightforward implementation of the
same algorithm. These are not archived text. They are imported, executed on every benchmark run,
and covered by tests.

Keeping them alive costs something, and it buys the only thing that makes the comparison
meaningful.

The solver baseline is OR-Tools' CP-SAT: open source, installs everywhere, and markedly
faster on this problem shape than the bundled-CBC route PuLP offers — the baseline carried a
CBC formulation until it was benchmarked, and beating the slower of two free solvers was the
weaker claim anyway. Gurobi would be faster still and is absent on purpose: it needs a
commercial license, so a benchmark nobody can reproduce.

## Parity is asserted before speed is measured

`tests/test_parity.py` runs the baseline and the optimized implementation on the same inputs and
asserts they agree. It runs on every change, in ordinary CI, on small inputs.

This is the load-bearing test here. A fast path that has quietly changed its behavior
will produce an excellent benchmark number, and only parity catches it. Parity does not mean
byte-identical output — that would require reimplementing the kernel's RNG in Python. It means
the properties a caller relies on: every lineup valid under an independently written validator,
comparable yield and spread, and the same response to constraints tightening. The precise claim
is in `tests/test_parity.py`'s docstring, and the benchmark reports quality alongside speed so a
faster path that produces worse lineups reads as the regression it is.

## CI never times anything

Shared CI runners vary by roughly 2× run to run, depending on what else is on the host. A
performance assertion in that environment fails for reasons unrelated to the code, and a check
that fails for unrelated reasons is a check people learn to ignore.

So CI does two things instead:

1. Runs the parity tests, which is where the correctness of the claim actually lives.
2. Runs `benchmarks/run.py --smoke` — one scenario at five lineups — to prove the harness
   still imports and executes.

Neither asserts a duration, and the smoke run writes nothing.

## Numbers come from named hardware

Real measurements are taken on a known machine:

```bash
task bench
```

That writes `benchmarks/results/<hardware-id>/<date>-<sha>.json`, which is
committed. Each file records the CPU model, core count, RAM, OS, Python version, package version,
and the commit it was measured against. Results are grouped by machine because a speedup is a
claim about a machine, not a universal constant.

Benchmarks are not pytest. They are long by design — the contest-scale row is ten thousand
sequential CP-SAT solves — and a test runner is the wrong home for that: it put them in CI on
every push, and it pinned the portfolio size to 25 lineups, a number nobody plays, because that
was the largest a solver could finish inside a test session. `benchmarks/run.py` is a plain
script. Each case is timed with `perf_counter` and repeated while it is short enough for
repetition to mean anything, reporting the median rather than the minimum.

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

### `in >50% of entries`

How many players appear in more than half the draw. A concentration measure that
discriminates where "share of the most-used player" often does not: a single
must-play ace can push the top share high for every implementation while the
rest of the draws differ completely. See `max_exposure` in the per-scenario
detail for the raw figure.

Where an exposure cap is set, read the realized figure rather than the cap
requested. The limit is computed against lineups *requested*, so when yield falls
short the realized share runs higher than asked.

### `returned` / yield

How many of the requested lineups came back, **reported for both
implementations**. A solver is complete and returns all of them. Randomized
construction is not: when constraints bite it runs out of legal rosters it has
not already found.

Both are shown because a speedup dividing one side's time for a partial draw by
the other's time for a full one is not a ratio. Where the yields differ, the
tables dash the speedup cell and let the yield columns carry the comparison.

### `no-good cuts`

How the solver is made to produce a different lineup each time. After it returns a
roster, a constraint is added saying at least one of those players must be dropped
next time. Solve again for the second-best roster, then the third.

It is the standard way to enumerate solutions in order, and it is why the solver's
draw looks the way it does. It is not the only way to get diversity out of a
solver — an overlap constraint bounding how many players a new lineup may share
with each earlier one is the other common technique, and a fairer comparison at
150 entries. Both formulations are measured: the `mme` scenario carries a second
solver row built with the overlap constraint, which spreads across the slate
properly and pays for it in the time column. Comparing only against no-good cuts
would understate what a solver can do; only against the overlap formulation
would understate its speed.

### Contest scale

Nothing in that table is extrapolated. The kernel is run to the full
10,000-lineup draw. The solver runs under a wall-clock budget — 10,000 no-good-cut
solves have no natural upper bound — and its row reports how many lineups the
budget bought, which is the honest form of the claim.

An earlier version measured the solver on a 25-lineup prefix and multiplied,
assuming per-lineup cost is flat in the count. It is not: every solve carries one
more no-good cut than the last, so the rate degrades across ten thousand of them
and the extrapolation understated the real figure. The per-lineup column exists
so that degradation is visible rather than inferred — compare it against the
solver's per-lineup cost on the 150-entry scenarios above.

This is by a wide margin the slowest case in the suite. It is worth the wall
clock, because the number it replaced was wrong.

## What a benchmark here does not tell you

The cases are the ones these packages were built for, at the sizes they were built for. They are
described in each result file's `params`, and they are the honest scope of the claim. A different
workload can invert any of these results, and if it does, that is a bug report worth filing.
