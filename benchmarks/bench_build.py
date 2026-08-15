"""Benchmark: how construction scales with the size of the pool asked for.

The narrow thing the name says: how does building `n` lineups scale in `n`, for
the kernel and for the pure-Python transcription that serves as its oracle.
`bench_constraints.py` measures what each rule costs; `bench_generated.py`
describes what comes out.

That scaling is the reason selection is practical. It needs a pool far larger
than the portfolio — tens of thousands of candidates to choose 150 from — so the
cost of the ten-thousandth lineup matters more than the cost of the first.

A solver is not slow at 150 lineups: on the shared slate CBC produces them in
about eighteen seconds, roughly 119 ms each. It is at candidate-pool scale that
the approaches separate, which is what `bench_generated.py` measures directly.

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
