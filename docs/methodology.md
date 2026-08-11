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

## What the columns mean

Every measure in the benchmark tables is defined here, because several are easy to
read as something more (or less) impressive than they are.

### The optimum

The single highest-projection roster that satisfies every constraint, proved by
CP-SAT with no cuts and no time pressure. It is one lineup, not a portfolio, and
it is the yardstick the two ratios below divide by.

### `median vs optimum` and `best vs optimum`

Add up each entry's projected points, take the median (or the maximum) across the
portfolio, and divide by the optimum. "94%" means the middle entry of the 150
projects to 94% of what the single best legal roster projects to.

This is the measure a solver wins by construction. Asked for 150 lineups it
returns the 150 highest-projection rosters, so its median *is* the optimum and no
sampling method can match that. It is reported because it is the honest half of
the comparison — and because it is not the half a contest pays on.

Note what it is not: a prediction of points scored. It is projection arithmetic on
the same inputs both approaches were given, so it measures how well each searched
the space, not whether the projections were any good.

### `overlap`

The average fraction of players two entries in the same portfolio share, over
every pair. Two ten-player lineups differing by one player overlap 0.9.

A portfolio of near-identical entries wins and loses as a block, which is the
central objection to asking a solver for 150 lineups: "second best" means
"the best one with a player swapped". Overlap is how that shows up as a number.

Computed over the first sixty entries, because it is quadratic in the count and
the figure is stable long before that.

### `in the money`

The measure that decides the argument, and the one with the most machinery behind
it. In full:

1. A **simulated universe** gives every player a score in each of a thousand
   possible outcomes, correlated within a team so that a lineup stacking one team
   moves together.
2. A **field** of about a hundred thousand entries is built to stand in for
   everyone else in the contest — chalk-seeking profiles, no ownership fade, and
   built independently of the portfolio being judged.
3. For each outcome separately, the **payout line** is the score that would finish
   in the top 0.1% of that field *in that outcome*. It is a different number in
   every outcome, because a high-scoring slate lifts everyone.
4. An outcome counts if **any single entry** in the portfolio reaches that
   outcome's line. `in the money = the fraction of outcomes that count.`

So "84%" means: in 84% of the simulated ways the slate could break, at least one
of the 150 entries would have finished in the top 0.1% of the field.

Three things to hold onto. It is a **portfolio** measure — one entry cashing is
enough, which is what makes covering different outcomes worth more than being
individually excellent. The line is **per outcome**; against a fixed line the
measure would mostly report whether the slate was high-scoring, which is no edge
because every rival entry scored more in those worlds too. And the payout tier is
deliberately harsh: at the top 20% or top 1% both approaches succeed in nearly
every outcome and the number stops discriminating.

### `Returned` / yield

How many of the requested lineups came back. A solver is complete and always
returns all of them. Randomized construction is not: when constraints bite it can
run out of legal rosters it has not already found, and returning fewer is the
honest answer. A method that keeps its throughput up by handing back half the
portfolio has not been fast, and only this column shows it.

### `no-good cuts`

How the solver is made to produce a *different* lineup each time. After it returns
a roster, a constraint is added saying that at least one of those players must be
dropped next time — the previously found set may not all appear together. Solve
again and you get the second-best roster, then the third, and so on.

It is the standard way to enumerate solutions in order and it is why the solver's
portfolio looks the way it does: each entry is the best remaining roster after
forbidding the last, which usually means the last one with a single player
changed.

## What a benchmark here does not tell you

The cases are the ones these packages were built for, at the sizes they were built for. They are
described in each result file's `params`, and they are the honest scope of the claim. A different
workload can invert any of these results, and if it does, that is a bug report worth filing.
