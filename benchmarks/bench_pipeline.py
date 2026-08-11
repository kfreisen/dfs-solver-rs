"""Benchmark: the whole pipeline against the whole alternative.

Every other benchmark here measures a part. `bench_build.py` and
`bench_constraints.py` time *construction*, and compare it against a solver
producing a portfolio — which is not the same job, and flatters nobody
consistently. `bench_quality.py` scores a portfolio without timing the work that
produced it.

This is the comparison a reader actually wants, and until now the repository did
not have it. Two ways to arrive at 150 contest entries:

* **This package** — generate a large candidate pool, score it against simulated
  outcomes, select the portfolio that covers the most ways of winning.
* **A solver** — ask for the best lineup, add a no-good cut, ask again, 150 times.

Both are timed end to end and both are judged on the same thing: how often the
portfolio puts an entry in the money against an independent field.

The result is the thesis of the package stated as two numbers. The solver's
lineups are individually better — they are the top 150 by projection, so their
median *is* the optimum. The portfolio they form is worse, because lineups that
differ by one player win and lose together, and a contest pays for covering ways
the slate can break rather than for being right on average.

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
from mlb_dfs_solver import build_lineups, score_lineups, select_portfolio
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.spec import RosterSpec
from slates import build_field, lineup_overlap, make_slate, payout_line, rung_by_name

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pulp.*")

ENTRIES = 150
CANDIDATES = 20_000
SIMULATIONS = 1_000
FIELD_ENTRIES = 200_000
# A top-heavy tournament. The wider payout tiers saturate — both approaches put
# something in the top 20% in essentially every outcome — so they say nothing
# about which is better.
PAYOUT_FRACTION = 0.001
# Per solve. The solver needs a ceiling or one hard instance stalls the suite;
# it is generous enough that none of these hit it.
SOLVER_LIMIT_S = 30.0


def simulated_universe(pool: PlayerPool, seed: int = 7) -> np.ndarray:
    """Correlated player outcomes. See `bench_quality.simulated_universe`."""
    rng = np.random.default_rng(seed)
    teams = pool.keys["team"]
    shock = rng.standard_normal((int(teams.max()) + 1, SIMULATIONS)) * 0.55
    noise = rng.standard_normal((len(pool), SIMULATIONS))
    return pool.projections[:, None] + pool.stddevs[:, None] * (0.8 * noise + shock[teams])


def our_pipeline(pool: PlayerPool, spec: RosterSpec, universe: np.ndarray, line: np.ndarray):
    """Generate, score, select — the thing the package is for."""
    candidates = build_lineups(pool, spec, num_lineups=CANDIDATES, seed=1, attempts_per_lineup=5)
    sim = score_lineups(pool, spec, candidates, universe)
    chosen = select_portfolio(sim, mode="gpp", line=line, n_select=ENTRIES)
    return candidates[chosen]


def mean_overlap(lineups: np.ndarray, limit: int = 60) -> float:
    """Average share of players two entries have in common."""
    sample = [list(x) for x in lineups[:limit]]
    if len(sample) < 2:
        return 1.0
    return float(
        np.mean([lineup_overlap(a, b) for i, a in enumerate(sample) for b in sample[i + 1 :]])
    )


def _profile(
    pool: PlayerPool,
    spec: RosterSpec,
    universe: np.ndarray,
    line: np.ndarray,
    optimum: float,
    lineups: np.ndarray,
) -> dict[str, float]:
    """How good a portfolio is, on both axes that matter."""
    projections = pool.projection_of(lineups, spec)
    scored = score_lineups(pool, spec, lineups, universe)
    return {
        "entries": int(len(lineups)),
        # Per-lineup quality, where the solver is unbeatable by construction.
        "median_ratio": round(float(np.median(projections)) / optimum, 4),
        "best_ratio": round(float(projections.max()) / optimum, 4),
        # Portfolio quality, which is what a contest pays.
        "p_in_the_money": round(float((scored.max(axis=0) >= line).mean()), 4),
        "overlap": round(mean_overlap(lineups), 4),
    }


@pytest.mark.parametrize("impl", ["slatekit_rust", "milp_ortools_cpsat"])
def test_pipeline(benchmark, impl: str) -> None:
    pool = make_slate()
    spec = rung_by_name(pool, "floor").spec
    universe = simulated_universe(pool)

    best = solve_milp_ortools(pool, spec, time_limit_s=120)
    assert best is not None
    optimum = float(pool.projection_of(np.array([best]), spec)[0])

    field = build_field(pool, spec, n_entries=FIELD_ENTRIES)
    line = payout_line(score_lineups(pool, spec, field, universe), PAYOUT_FRACTION)

    if impl == "slatekit_rust":
        lineups = benchmark(our_pipeline, pool, spec, universe, line)
    else:
        # Few rounds: each is 150 sequential solves.
        lineups = np.array(
            benchmark.pedantic(
                solve_portfolio_ortools,
                args=(pool, spec),
                kwargs={"num_lineups": ENTRIES, "time_limit_s": SOLVER_LIMIT_S},
                rounds=2,
                iterations=1,
            )
        )

    benchmark.extra_info["case"] = "pipeline/150-entry tournament"
    benchmark.extra_info["impl"] = impl
    benchmark.extra_info["detail"] = (
        f"150 entries against a {len(field):,}-entry field, paying the top {PAYOUT_FRACTION:.1%}"
    )
    benchmark.extra_info["params"] = {"requested": ENTRIES, "produced": int(len(lineups))}
    benchmark.extra_info["quality"] = {
        "optimum": round(optimum, 2),
        "field_entries": int(len(field)),
        **_profile(pool, spec, universe, line, optimum, lineups),
    }
    assert len(lineups) > 0


def test_portfolio_beats_the_solver_where_it_counts(benchmark) -> None:
    """The claim the package exists to make, asserted rather than implied.

    The solver's entries are individually better and its portfolio is worse. If
    that ever stops being true the argument for randomized construction plus
    submodular selection has gone with it, and a benchmark table alone would not
    say so loudly enough.
    """
    pool = make_slate()
    spec = rung_by_name(pool, "floor").spec
    universe = simulated_universe(pool)

    best = solve_milp_ortools(pool, spec, time_limit_s=120)
    assert best is not None
    optimum = float(pool.projection_of(np.array([best]), spec)[0])

    field = build_field(pool, spec, n_entries=FIELD_ENTRIES)
    line = payout_line(score_lineups(pool, spec, field, universe), PAYOUT_FRACTION)

    ours = _profile(pool, spec, universe, line, optimum, our_pipeline(pool, spec, universe, line))
    solver = _profile(
        pool,
        spec,
        universe,
        line,
        optimum,
        np.array(
            solve_portfolio_ortools(pool, spec, num_lineups=ENTRIES, time_limit_s=SOLVER_LIMIT_S)
        ),
    )

    benchmark.extra_info["case"] = "pipeline/head-to-head"
    benchmark.extra_info["impl"] = "slatekit_rust"
    benchmark.extra_info["metric"] = "quality"
    benchmark.extra_info["quality"] = {
        **{f"ours_{k}": v for k, v in ours.items()},
        **{f"solver_{k}": v for k, v in solver.items()},
    }
    benchmark.pedantic(lambda: None, rounds=1, iterations=1)

    assert solver["median_ratio"] > ours["median_ratio"], (
        "the solver should win per-lineup — it returns the top entries by "
        "projection — and if it does not, the comparison is not measuring what "
        "it claims"
    )
    assert ours["p_in_the_money"] > solver["p_in_the_money"], (
        "the portfolio should win where a contest pays; without this the package has no argument"
    )
    assert ours["overlap"] < solver["overlap"]
