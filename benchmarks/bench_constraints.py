"""Benchmark: what each constraint costs, kernel against solver.

`bench_build.py` answers "how fast is it?" on one problem. That number says
nothing about a *feature*: every constraint added is work for the kernel and work
for the solver, and they do not scale alike. A cap is nearly free for both. A
stack is a linear constraint the solver searches over and an existential the
greedy has to guess at. A conflict prunes the solver's tree and makes the greedy
paint itself into corners. Which of those hurts more is not something to reason
about from the code.

So this walks the ladder in `slates.py`, adding one constraint at a time and
measuring both implementations on each rung. The MILP side models every rule the
specification can express — see `baselines/milp.py` — so a rung is a comparison
rather than a handicap.

Two numbers matter per rung and both are recorded:

* **Throughput**, as lineups per second. The headline.
* **Yield**, the fraction of the requested lineups actually returned. The solver
  is complete and returns all of them; randomized construction is not, and on the
  hardest rungs it gives back fewer. A
  constraint that halves throughput while still filling the portfolio is cheap; a
  constraint that keeps throughput up by giving back sixty lineups instead of a
  hundred and fifty is not, and a table with only the first number would call
  those the same. Yield below 1.0 is a real answer — the slate ran out of legal
  lineups — but it has to be visible.

Run with:

    task bench
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.milp import solve_portfolio_ortools
from baselines.reference import build_lineups_reference
from mlb_dfs_solver import build_lineups
from slates import RUNG_NAMES, make_slate, rung_by_name

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pulp.*")

# The portfolio size every rung is asked for — the *same* for all three
# implementations, which is what makes the published ratio a ratio. Asking the
# kernel for 150 and the solver for 5 because that is what each can manage
# produces two wall-clock numbers that cannot be divided by each other, and a
# table that quietly does it anyway is worse than no table. Twenty-five is the
# largest size CP-SAT finishes the whole ladder at.
LINEUPS = 25
# Attempts allowed per requested lineup, the same for both so the comparison is
# "how fast with this budget" rather than "who was given more tries". The default
# of 3 is tuned for an unconstrained slate; the upper rungs need more, and giving
# only the hard rungs extra would make the ladder incomparable along its length.
ATTEMPTS = 10


def _record(benchmark, rung, impl: str, requested: int, produced: int) -> None:
    """Attach the case metadata the report generator reads."""
    benchmark.extra_info["case"] = f"constraints/{rung.name}"
    benchmark.extra_info["impl"] = impl
    benchmark.extra_info["detail"] = rung.detail
    benchmark.extra_info["params"] = {
        "rung": rung.name,
        "requested": requested,
        "produced": produced,
        "yield": round(produced / requested, 3) if requested else 0.0,
    }


@pytest.mark.parametrize("rung_name", RUNG_NAMES)
def test_kernel(benchmark, rung_name: str) -> None:
    pool = make_slate()
    rung = rung_by_name(pool, rung_name)
    kwargs = {
        "num_lineups": LINEUPS,
        "seed": 1,
        "attempts_per_lineup": ATTEMPTS,
        "locks": list(rung.locks) or None,
        "max_exposure": rung.max_exposure,
    }

    result = benchmark(build_lineups, pool, rung.spec, **kwargs)
    _record(benchmark, rung, "slatekit_rust", LINEUPS, len(result))
    assert len(result) > 0, f"rung {rung.name} produced nothing to measure"


@pytest.mark.parametrize("rung_name", RUNG_NAMES)
def test_reference_python(benchmark, rung_name: str) -> None:
    pool = make_slate()
    rung = rung_by_name(pool, rung_name)
    kwargs = {
        "num_lineups": LINEUPS,
        "seed": 1,
        "attempts_per_lineup": ATTEMPTS,
        "locks": list(rung.locks) or None,
        "max_exposure": rung.max_exposure,
    }

    result = benchmark(build_lineups_reference, pool, rung.spec, **kwargs)
    _record(benchmark, rung, "reference_python", LINEUPS, len(result))
    assert len(result) > 0


@pytest.mark.parametrize("rung_name", RUNG_NAMES)
def test_milp_ortools(benchmark, rung_name: str) -> None:
    pool = make_slate()
    rung = rung_by_name(pool, rung_name)

    # Few rounds: each is a full sequence of solves, and pytest-benchmark's
    # calibration would otherwise spend the afternoon here for no extra precision.
    result = benchmark.pedantic(
        solve_portfolio_ortools,
        args=(pool, rung.spec),
        kwargs={
            "num_lineups": LINEUPS,
            "time_limit_s": 60,
            "locks": list(rung.locks) or None,
            "max_exposure": rung.max_exposure,
        },
        rounds=2,
        iterations=1,
    )
    _record(benchmark, rung, "milp_ortools_cpsat", LINEUPS, len(result))
    assert len(result) > 0
