"""Reference implementations, for comparison only.

Nothing in this package is published: it is excluded from the wheel and the
sdist, it is not a supported API, and it is deliberately slow. It exists so the
benchmarks have live baselines to run against and so `tests/test_parity.py` has
an oracle to hold the kernel to. If you want the algorithm, use
`dfs_solver`; if you want a MILP solver, vendor `milp.py` from the
repository.
"""
