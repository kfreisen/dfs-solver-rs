"""Benchmark: what each implementation actually generates.

The comparison this package rests on is between two sets of lineups, so this
measures the lineups. Not what they would have scored -- scoring needs a
simulator and a field, both of which are fixtures written for the benchmark, and
a number that moves when the fixture is rewritten is a fact about the fixture.

Everything below is a property of the output and of the inputs the caller
supplied: how much of the slate each method uses, how concentrated it is on
individual players, how far its rosters spread in projection, and what the
rosters look like. All of it is checkable by reading the lineups.

Two questions, and they want different scales.

**At portfolio size (150 entries)** the two are alternatives, and the question is
how their output differs. Both are asked for the same count on the same config.

**At contest scale (10,000 lineups)** they are not alternatives. This is the draw
a field simulation or a candidate pool needs. Both implementations are run to
completion — the solver's figure is a measurement, not a per-lineup rate
multiplied out, because its rate is not flat: each solve carries one more no-good
cut than the last. That makes this the slowest case in the suite by a wide margin
and it is worth what it costs, since the extrapolation it replaced was wrong.

Neither table says which output is better. That depends on the contest and on
projections this package does not supply.

Run with:

    task bench
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines.milp import solve_milp_ortools, solve_portfolio_ortools
from mlb_dfs_solver import CONTRARIAN, JitterProfile, build_lineups
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.spec import RosterSpec
from slates import (
    CONTEST_EXPOSURE_CAP,
    CONTEST_PER_POSITION,
    CONTEST_RUNG,
    describe_lineups,
    make_slate,
    rung_by_name,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pulp.*")

# The configs worth describing: the plain contest, and the fully constrained one.
# The middle rungs interpolate.
DESCRIBE_RUNGS = ["floor", "stack"]
PORTFOLIO = 150
# Attempts per requested lineup, the same for both so the comparison is "how does
# the output differ at this budget" rather than "who was given more tries".
ATTEMPTS = 5

# Contest scale. See `slates.CONTEST_*` for why the slate is larger here.
CONTEST_DRAW = 10_000
SOLVER_LIMIT_S = 30.0
# The reference distribution percentile rank is measured against.
REFERENCE_POPULATION = 2_000


def random_population(pool: PlayerPool, spec: RosterSpec) -> np.ndarray:
    """Valid lineups built with the objective all but switched off.

    Noise is turned far past the projections and the ceiling and leverage draws
    are pinned flat, so selection is close to arbitrary among *legal* lineups.
    That is the population a caller implicitly compares against when they ask
    whether a lineup is any good.

    Not a uniform sample of the lineup space -- nobody can produce one of those
    for a problem this size. It is a "what does an arbitrary legal lineup look
    like?" baseline and nothing more.
    """
    flat = JitterProfile(ceiling=(0.0, 0.0), leverage=(0.0, 0.0))
    return build_lineups(
        pool,
        spec,
        num_lineups=REFERENCE_POPULATION,
        seed=99,
        noise=500.0,
        attempts_per_lineup=6,
        profiles=[flat],
    )


@pytest.fixture(scope="module")
def portfolio_slate():
    """The shared 288-player slate, so these tables read next to the others."""
    return make_slate()


@pytest.fixture(scope="module")
def contest_slate():
    """A larger slate, sized so a 10,000-lineup draw is not measuring a ceiling."""
    return make_slate(CONTEST_PER_POSITION)


@pytest.mark.parametrize("rung_name", DESCRIBE_RUNGS)
@pytest.mark.parametrize("impl", ["mlb_dfs_solver_rust", "milp_ortools_cpsat"])
def test_generated(benchmark, portfolio_slate, rung_name: str, impl: str) -> None:
    """Describe a 150-lineup draw from each implementation on one config."""
    pool = portfolio_slate
    rung = rung_by_name(pool, rung_name)
    spec = rung.spec

    best = solve_milp_ortools(pool, spec, time_limit_s=120)
    assert best is not None, f"no optimum for rung {rung.name}"
    optimum = float(pool.projection_of(np.array([best]), spec)[0])

    if impl == "mlb_dfs_solver_rust":
        lineups = benchmark(
            build_lineups, pool, spec, num_lineups=PORTFOLIO, seed=1, attempts_per_lineup=ATTEMPTS
        )
    else:
        # Few rounds: each is 150 sequential solves.
        lineups = np.array(
            benchmark.pedantic(
                solve_portfolio_ortools,
                args=(pool, spec),
                kwargs={"num_lineups": PORTFOLIO, "time_limit_s": SOLVER_LIMIT_S},
                rounds=2,
                iterations=1,
            )
        )

    assert len(lineups) > 0

    # Where these rosters sit among arbitrary legal ones. Cheap to compute and
    # the one figure that says "good" without needing an outcome: a median entry
    # at the 97th percentile of legal lineups is drawn from the good end of the
    # space, whatever a contest would have paid it.
    baseline = np.sort(pool.projection_of(random_population(pool, spec), spec))
    percentiles = (
        np.searchsorted(baseline, pool.projection_of(lineups, spec)) / len(baseline) * 100.0
    )

    benchmark.extra_info["case"] = f"generated/{rung.name}"
    benchmark.extra_info["impl"] = impl
    benchmark.extra_info["detail"] = rung.detail
    benchmark.extra_info["params"] = {"requested": PORTFOLIO, "produced": int(len(lineups))}
    benchmark.extra_info["quality"] = {
        "optimum": round(optimum, 2),
        "median_percentile": round(float(np.median(percentiles)), 2),
        "p10_percentile": round(float(np.percentile(percentiles, 10)), 2),
        "reference_population": int(len(baseline)),
        **describe_lineups(pool, spec, lineups, optimum),
    }


def test_generated_differs(benchmark, portfolio_slate) -> None:
    """The two draws must be visibly different sets of lineups.

    Asserted rather than left to a table. If a solver enumerating by projection
    and a randomized greedy started returning comparable spreads of players, the
    reason to prefer either would have gone, and a table alone would not say so
    loudly enough.
    """
    pool = portfolio_slate
    spec = rung_by_name(pool, "floor").spec

    ours = build_lineups(pool, spec, num_lineups=PORTFOLIO, seed=1, attempts_per_lineup=ATTEMPTS)
    solver = np.array(
        solve_portfolio_ortools(pool, spec, num_lineups=PORTFOLIO, time_limit_s=SOLVER_LIMIT_S)
    )

    ours_d = describe_lineups(pool, spec, ours)
    solver_d = describe_lineups(pool, spec, solver)

    benchmark.extra_info["case"] = "generated/head-to-head"
    benchmark.extra_info["impl"] = "mlb_dfs_solver_rust"
    benchmark.extra_info["metric"] = "quality"
    benchmark.extra_info["quality"] = {
        **{f"ours_{k}": v for k, v in ours_d.items()},
        **{f"solver_{k}": v for k, v in solver_d.items()},
    }
    benchmark.pedantic(lambda: None, rounds=1, iterations=1)

    assert ours_d["distinct_players"] > solver_d["distinct_players"], (
        "randomized construction should touch more of the slate than a solver "
        "enumerating by projection; if it does not, the reason to prefer it has gone"
    )
    # The solver returns the top rosters by projection, so its entries cluster
    # near the optimum by construction. That is its half of the comparison.
    assert solver_d["projection_median"] > ours_d["projection_median"]


def test_contrarian_profile_shifts_ownership(benchmark, portfolio_slate) -> None:
    """The contrarian profile must actually fade popular players.

    Ownership is an input the caller supplied, so this compares two draws on a
    column the caller owns rather than on a simulated result. A profile that
    changed nothing measurable would be documentation rather than behaviour.
    """
    pool = portfolio_slate
    spec = rung_by_name(pool, "floor").spec
    standard = build_lineups(pool, spec, num_lineups=PORTFOLIO, seed=2, profiles=None)
    contrarian = build_lineups(pool, spec, num_lineups=PORTFOLIO, seed=2, profiles=[CONTRARIAN])

    def mean_ownership(lineups: np.ndarray) -> float:
        return float(np.mean(pool.ownership[lineups]))

    benchmark.extra_info["case"] = "generated/profiles"
    benchmark.extra_info["impl"] = "mlb_dfs_solver_rust"
    benchmark.extra_info["metric"] = "quality"
    benchmark.extra_info["quality"] = {
        "standard_ownership": round(mean_ownership(standard), 4),
        "contrarian_ownership": round(mean_ownership(contrarian), 4),
    }
    benchmark.pedantic(lambda: None, rounds=1, iterations=1)
    assert mean_ownership(contrarian) < mean_ownership(standard)


@pytest.mark.parametrize("impl", ["mlb_dfs_solver_rust", "milp_ortools_cpsat"])
def test_contest_scale(benchmark, contest_slate, impl: str) -> None:
    """A field-sized draw: 10,000 lineups under a stacked, exposure-capped config.

    This is the scale at which the two stop being alternatives. It is what a
    field simulation needs, and what a candidate pool for selection needs.
    """
    pool = contest_slate
    rung = rung_by_name(pool, CONTEST_RUNG)
    spec = rung.spec
    caps = dict.fromkeys(range(len(pool)), CONTEST_EXPOSURE_CAP)

    if impl == "mlb_dfs_solver_rust":
        lineups = benchmark(
            build_lineups,
            pool,
            spec,
            num_lineups=CONTEST_DRAW,
            seed=1,
            attempts_per_lineup=15,
            max_exposure=caps,
        )
    else:
        # The full draw, actually run. An earlier version measured a 25-lineup
        # prefix and multiplied, on the assumption that per-lineup cost is flat
        # in the count. It is not: every solve carries one more no-good cut than
        # the last, so the rate degrades over ten thousand of them and the
        # extrapolation understated the real figure. The solver's exposure cap is
        # applied between solves, the same greedy rule the kernel uses at merge.
        lineups = np.array(
            benchmark.pedantic(
                solve_portfolio_ortools,
                args=(pool, spec),
                kwargs={
                    "num_lineups": CONTEST_DRAW,
                    "time_limit_s": SOLVER_LIMIT_S,
                    "max_exposure": caps,
                },
                rounds=1,
                iterations=1,
            )
        )

    assert len(lineups) > 0
    benchmark.extra_info["case"] = "contest/10k draw"
    benchmark.extra_info["impl"] = impl
    benchmark.extra_info["detail"] = (
        f"{CONTEST_DRAW:,} lineups, {len(pool)}-player slate, "
        f"stacked, {CONTEST_EXPOSURE_CAP:.0%} exposure cap"
    )
    benchmark.extra_info["params"] = {
        "requested": CONTEST_DRAW,
        "produced": int(len(lineups)),
    }
    benchmark.extra_info["quality"] = {
        "requested_exposure_cap": CONTEST_EXPOSURE_CAP,
        **describe_lineups(pool, spec, lineups),
    }
