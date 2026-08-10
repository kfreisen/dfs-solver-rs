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

## What a benchmark here does not tell you

The cases are the ones these packages were built for, at the sizes they were built for. They are
described in each result file's `params`, and they are the honest scope of the claim. A different
workload can invert any of these results, and if it does, that is a bug report worth filing.
