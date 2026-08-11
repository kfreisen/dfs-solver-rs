"""Benchmark: how construction scales with the size of the pool asked for.

The other files answer different questions. `bench_constraints.py` measures what
each rule costs, `bench_pipeline.py` compares the whole job against the whole
alternative, and `bench_quality.py` asks whether the output is any good. This one
does the narrow thing its name says: how does building `n` lineups scale in `n`,
for the kernel and for the pure-Python transcription that serves as its oracle.

That scaling is the reason the pipeline works at all. Selection needs a pool far
larger than the portfolio — tens of thousands of candidates to choose 150 from —
so the cost of the ten-thousandth lineup matters much more than the cost of the
first.

**A correction, recorded because the old numbers were published.** This file used
to run on a slate of its own with only twelve distinct `(salary, projection)`
pairs across ninety players — eleven exact clones of everybody. It reported CBC
at "roughly 15-20 seconds per lineup" and noted that twenty lineups would not
finish in fifteen minutes. Both were true of that slate and neither is true of
the problem: the ties sent branch-and-bound hunting through interchangeable
optima. On the shared realistic slate CBC produces 150 lineups in about eighteen
seconds, some 119 ms each — roughly 676 times faster per lineup than the figure
this file used to publish.

So the solver is not slow at making a hundred and fifty lineups, and this package
should not claim it is. What the solver cannot do is make *twenty thousand*: at
119 ms each that is forty minutes, and they would be the top twenty thousand by
projection, which is the most redundant set of lineups obtainable. The argument
for randomized construction is throughput at candidate-pool scale and the
portfolio that selection then builds — see `bench_pipeline.py` — not that a
solver struggles with a portfolio.

Run with:

    task bench
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.milp import solve_portfolio_pulp
from baselines.reference import build_lineups_reference
from mlb_dfs_solver import build_lineups
from slates import make_slate, rung_by_name

# PuLP 3.x warns that constructing LpVariable directly is deprecated in favour of a
# 4.0 API that does not exist in 3.x. The package pins `pulp<4` precisely because
# 4.0 removes PULP_CBC_CMD, so there is nothing to migrate to yet and the warning
# is noise. Scoped to this module rather than relaxed globally.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pulp.*")

# The kernel is measured out to a real candidate-pool size, because that is the
# regime it exists for. The oracle stops earlier only because it is ~30x slower
# and adds nothing past the point where the shape of the curve is clear.
KERNEL_SIZES = [10, 150, 1_000, 20_000]
REFERENCE_SIZES = [10, 150, 1_000]
# CBC is measured where it is genuinely usable, which — now that the slate is not
# degenerate — includes a full 150-entry portfolio. The largest kernel size is
# deliberately absent: 20_000 solves is about forty minutes, and the result would
# be twenty thousand near-identical lineups.
MILP_SIZES = [10, 150]


def _record(benchmark, num_lineups: int, impl: str, pool_size: int, produced: int) -> None:
    benchmark.extra_info["case"] = f"build/{num_lineups}"
    benchmark.extra_info["impl"] = impl
    benchmark.extra_info["params"] = {
        "num_lineups": num_lineups,
        "pool": pool_size,
        "produced": produced,
        "requested": num_lineups,
    }


@pytest.fixture(scope="module")
def slate():
    """The shared slate, so this table can be read next to the others."""
    pool = make_slate()
    return pool, rung_by_name(pool, "floor").spec


@pytest.mark.parametrize("num_lineups", KERNEL_SIZES)
def test_kernel(benchmark, slate, num_lineups: int) -> None:
    pool, spec = slate
    result = benchmark(
        build_lineups, pool, spec, num_lineups=num_lineups, seed=1, attempts_per_lineup=5
    )
    _record(benchmark, num_lineups, "slatekit_rust", len(pool), len(result))
    assert len(result) > 0


@pytest.mark.parametrize("num_lineups", REFERENCE_SIZES)
def test_reference_python(benchmark, slate, num_lineups: int) -> None:
    pool, spec = slate
    result = benchmark(
        build_lineups_reference, pool, spec, num_lineups=num_lineups, seed=1, attempts_per_lineup=5
    )
    _record(benchmark, num_lineups, "reference_python", len(pool), len(result))
    assert len(result) > 0


@pytest.mark.parametrize("num_lineups", MILP_SIZES)
def test_milp_pulp(benchmark, slate, num_lineups: int) -> None:
    pool, spec = slate
    # Few rounds: each is a full sequence of solves, and pytest-benchmark's
    # calibration would otherwise spend a long time here for no extra precision.
    result = benchmark.pedantic(
        solve_portfolio_pulp,
        args=(pool, spec),
        kwargs={"num_lineups": num_lineups, "time_limit_s": 60},
        rounds=2,
        iterations=1,
    )
    _record(benchmark, num_lineups, "milp_pulp_cbc", len(pool), len(result))
    assert len(result) > 0
