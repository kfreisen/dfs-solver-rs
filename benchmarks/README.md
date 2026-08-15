# mlb_dfs_solver benchmarks

`baselines/` holds the implementation this package replaced, kept working and kept
tested. It is the reason a speedup number here means anything: `tests/test_parity.py`
asserts the baseline and the current implementation produce the same output, so the
comparison is between two things that do the same job.

Run the full suite and record a result:

```bash
task bench
```

That writes `results/<hardware-id>/<date>-<sha>.json`, which is committed. The docs
site renders those files directly, so a published table cannot drift from the data.

CI never times anything — shared runners vary by roughly 2×, and a flaky performance
gate is a gate people learn to ignore. CI runs this suite with `--benchmark-disable`
purely to prove the benchmark code still executes.

It also passes `-m "not slow"`. Disabling timing still executes each body once, and
once for the contest-scale draw is ten thousand sequential CP-SAT solves. Run it
yourself with `task bench`; `task bench-quick` skips it.
