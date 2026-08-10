# mlb_dfs_solver benchmarks

`baselines/` holds the implementation this package replaced, kept working and kept
tested. It is the reason a speedup number here means anything: `tests/test_parity.py`
asserts the baseline and the current implementation produce the same output, so the
comparison is between two things that do the same job.

Run the full suite and record a result:

```bash
make bench          # from the repository root
```

That writes `results/<hardware-id>/<date>-<sha>.json`, which is committed. The docs
site renders those files directly, so a published table cannot drift from the data.

CI never times anything — shared runners vary by roughly 2×, and a flaky performance
gate is a gate people learn to ignore. CI runs this suite in `--quick` mode purely to
prove the benchmark code still executes.
