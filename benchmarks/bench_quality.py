"""Benchmark: are the fast lineups any good, and are they actually different?

Speed alone would be a misleading thing to publish. A generator that returns a
hundred and fifty near-identical lineups very quickly is worthless for the job
this package exists to do, and so is one that returns wildly diverse rubbish. The
claim being made is narrower and has to be measured as two things at once:

**Still optimized.** Every greedy lineup is scored against the true optimum, which
CP-SAT provides exactly. The number to look at is the *ratio* — a portfolio whose
median lineup scores 96% of the best possible lineup is doing real work.

**Not the same lineups.** The solver's own portfolio is the comparison. Asked for
K lineups it returns the best, then the second best, then the third, and those
differ from each other by a player or two. Measured as mean pairwise overlap
within each portfolio, and as overlap between the two.

The third measure is the one that ties them together: **percentile rank**. Each
greedy lineup is placed against a reference population of randomly constructed
valid lineups, so "the median lineup sits at the 97th percentile" says the
portfolio is drawn from the good end of the space rather than from all of it. Note
what that population is and is not — it is randomly *constructed* valid lineups,
not a uniform sample of the space, which nobody can produce for a problem this
size. It is a defensible "what does an arbitrary legal lineup look like?" baseline
and nothing more.

Nothing here is a wall-clock measurement. It is a table of quality numbers, run
under pytest-benchmark so it lands in the same report, and it is why
`bench_build.py` can claim a speedup without that claim being cheap.

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
from mlb_dfs_solver import (
    CONTRARIAN,
    JitterProfile,
    build_lineups,
    field_line,
    score_lineups,
    select_portfolio,
)
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.spec import RosterSpec
from slates import lineup_overlap, make_slate, rung_by_name

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pulp.*")

# The rungs worth measuring quality on: the plain contest, and the fully
# constrained one. Quality on the middle rungs interpolates and the solver time
# does not.
QUALITY_RUNGS = ["floor", "stack"]
PORTFOLIO = 150
# How many lineups the solver is asked for. Each is a fresh solve with one more
# no-good cut, so this is the practical ceiling rather than a chosen sample size.
SOLVER_PORTFOLIO = 10
REFERENCE_POPULATION = 2_000
# How many candidates selection gets to choose from. Generation is cheap and
# selection is the point: the ratio is what the table below measures.
CANDIDATES = 20_000
SIMULATIONS = 1_500


def random_population(pool: PlayerPool, spec: RosterSpec, **kwargs: object) -> np.ndarray:
    """Valid lineups built with the objective all but switched off.

    The reference distribution percentile rank is measured against. Noise is
    turned up far past the projections and the ceiling and leverage draws are
    pinned flat, so selection is close to arbitrary among *legal* lineups — which
    is the population a caller implicitly compares against when they ask whether a
    lineup is any good.

    Not a uniform sample of the lineup space. Nobody can produce one of those for
    a problem this size, and claiming otherwise would be the sort of quiet
    overstatement this file exists to avoid.
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
        **kwargs,  # type: ignore[arg-type]
    )


def mean_pairwise_overlap(lineups: list[list[int]], limit: int = 60) -> float:
    """Average fraction of players shared between two lineups of a portfolio.

    Capped at `limit` lineups because this is quadratic and the answer is stable
    long before the cap.
    """
    sample = lineups[:limit]
    if len(sample) < 2:
        return 1.0
    totals = [lineup_overlap(a, b) for i, a in enumerate(sample) for b in sample[i + 1 :]]
    return float(np.mean(totals))


@pytest.mark.parametrize("rung_name", QUALITY_RUNGS)
def test_quality(benchmark, rung_name: str) -> None:
    """Score, spread and overlap for one rung. Not a timing."""
    pool = make_slate()
    rung = rung_by_name(pool, rung_name)
    spec = rung.spec
    extra = {"locks": list(rung.locks) or None, "max_exposure": rung.max_exposure}

    best = solve_milp_ortools(pool, spec, locks=list(rung.locks) or None, time_limit_s=120)
    assert best is not None, f"no optimum for rung {rung.name}"
    optimum = float(pool.projection_of(np.array([best]), spec)[0])

    ours = build_lineups(pool, spec, num_lineups=PORTFOLIO, seed=1, **extra)
    assert len(ours) > 0
    scores = pool.projection_of(ours, spec)

    population = random_population(pool, spec, **extra)
    baseline = np.sort(pool.projection_of(population, spec))
    percentiles = np.searchsorted(baseline, scores) / len(baseline) * 100.0

    solver_portfolio = solve_portfolio_ortools(
        pool,
        spec,
        num_lineups=SOLVER_PORTFOLIO,
        locks=list(rung.locks) or None,
        time_limit_s=120,
    )
    ours_lists = ours.tolist()
    cross = (
        float(np.mean([lineup_overlap(a, b) for a in ours_lists[:60] for b in solver_portfolio]))
        if solver_portfolio
        else float("nan")
    )
    identical = sum(1 for a in ours_lists if any(set(a) == set(b) for b in solver_portfolio))

    # Overlap is compared at matched portfolio size. A hundred and fifty lineups
    # have more chances to resemble each other than ten do, so reading the two
    # self-overlap figures against each other at different sizes would flatter
    # whichever portfolio was smaller — here, always the solver's.
    matched = min(len(ours_lists), len(solver_portfolio))
    ours_matched = mean_pairwise_overlap(ours_lists[:matched])
    solver_matched = mean_pairwise_overlap(solver_portfolio[:matched])

    benchmark.extra_info["case"] = f"quality/{rung.name}"
    benchmark.extra_info["impl"] = "slatekit_rust"
    benchmark.extra_info["detail"] = rung.detail
    # Not a timing. The report tool keeps this row out of the speedup table.
    benchmark.extra_info["metric"] = "quality"
    benchmark.extra_info["quality"] = {
        "lineups": int(len(ours)),
        "optimum": round(optimum, 2),
        # How close the portfolio gets to the single best possible lineup.
        "best_ratio": round(float(scores.max()) / optimum, 4),
        "median_ratio": round(float(np.median(scores)) / optimum, 4),
        "worst_ratio": round(float(scores.min()) / optimum, 4),
        # Where those lineups sit among arbitrary legal ones.
        "median_percentile": round(float(np.median(percentiles)), 2),
        "p10_percentile": round(float(np.percentile(percentiles, 10)), 2),
        "population": int(len(population)),
        # Diversity, which is the whole reason for not just asking the solver.
        "self_overlap": round(mean_pairwise_overlap(ours_lists), 4),
        "self_overlap_matched": round(ours_matched, 4),
        "solver_self_overlap": round(solver_matched, 4),
        "matched_size": matched,
        "overlap_with_solver": round(cross, 4),
        "identical_to_solver": identical,
    }

    # Run once through the harness so the row exists in the report. The timing is
    # meaningless and is not what anyone should read off this table.
    benchmark.pedantic(lambda: None, rounds=1, iterations=1)

    # The claims this file exists to defend, asserted so they cannot rot quietly.
    # The thresholds are set below what is measured today, not at some aspiration:
    # they are regression guards, and a guard nobody can pass is just a red build.
    assert float(scores.max()) <= optimum * 1.0001, "scored above the proven optimum"
    assert float(scores.max()) > optimum * 0.78, "the best candidate has drifted from the optimum"
    assert float(np.median(scores)) > optimum * 0.70, "the pool has stopped tracking the optimum"
    assert ours_matched < solver_matched, (
        f"at {matched} lineups each, the greedy portfolio overlaps "
        f"{ours_matched:.3f} against the solver's {solver_matched:.3f} — it is no "
        f"more diverse than re-solving with no-good cuts, which is the entire "
        f"argument for it"
    )


def test_contrarian_profile_shifts_ownership(benchmark) -> None:
    """The contrarian profile must actually fade popular players.

    A profile that changed nothing measurable would be documentation rather than
    behaviour, and the diversity argument leans on it.
    """
    pool = make_slate()
    spec = rung_by_name(pool, "floor").spec
    standard = build_lineups(pool, spec, num_lineups=PORTFOLIO, seed=2, profiles=None)
    contrarian = build_lineups(pool, spec, num_lineups=PORTFOLIO, seed=2, profiles=[CONTRARIAN])

    def mean_ownership(lineups: np.ndarray) -> float:
        return float(np.mean(pool.ownership[lineups]))

    benchmark.extra_info["case"] = "quality/profiles"
    benchmark.extra_info["impl"] = "slatekit_rust"
    benchmark.extra_info["metric"] = "quality"
    benchmark.extra_info["quality"] = {
        "standard_ownership": round(mean_ownership(standard), 4),
        "contrarian_ownership": round(mean_ownership(contrarian), 4),
    }
    benchmark.pedantic(lambda: None, rounds=1, iterations=1)
    assert mean_ownership(contrarian) < mean_ownership(standard)


def simulated_universe(pool: PlayerPool, seed: int = 7) -> np.ndarray:
    """Correlated player outcomes, standing in for a real simulator.

    A team-level shock plus idiosyncratic noise. Crude next to a copula fit on
    real distributions, and deliberately so — this library does not simulate, and
    a benchmark that shipped a serious simulator would be testing the simulator.
    What it needs is a matrix with *some* correlation structure, so that lineups
    sharing a stack move together and coverage means something.
    """
    rng = np.random.default_rng(seed)
    teams = pool.keys["team"]
    shock = rng.standard_normal((int(teams.max()) + 1, SIMULATIONS)) * 0.55
    noise = rng.standard_normal((len(pool), SIMULATIONS))
    return pool.projections[:, None] + pool.stddevs[:, None] * (0.8 * noise + shock[teams])


@pytest.mark.parametrize("rung_name", QUALITY_RUNGS)
def test_selection(benchmark, rung_name: str) -> None:
    """What selection buys over entering the pool as it comes.

    The claim being measured: construction answers which rosters are legal, and
    on a realistically priced slate the median of what it returns sits well below
    the best of them. Selecting from a much larger pool closes that gap. It does
    not close all of it — the pool's *best* is a property of generation, and no
    amount of selecting raises it.
    """
    pool = make_slate()
    rung = rung_by_name(pool, rung_name)
    spec = rung.spec
    extra = {"locks": list(rung.locks) or None, "max_exposure": rung.max_exposure}

    best = solve_milp_ortools(pool, spec, locks=list(rung.locks) or None, time_limit_s=120)
    assert best is not None
    optimum = float(pool.projection_of(np.array([best]), spec)[0])

    unselected = build_lineups(pool, spec, num_lineups=PORTFOLIO, seed=1, **extra)
    candidates = build_lineups(pool, spec, num_lineups=CANDIDATES, seed=1, **extra)
    universe = simulated_universe(pool)
    sim = score_lineups(pool, spec, candidates, universe)

    # Per outcome, not a constant: the field's score swings far more between
    # outcomes than lineups do within one, so a fixed bar would mostly measure
    # whether the slate was high-scoring.
    win_line = field_line(sim, 0.99)
    chosen = select_portfolio(sim, mode="gpp", line=win_line, n_select=PORTFOLIO)
    selected = candidates[chosen]

    def profile(lineups: np.ndarray) -> dict[str, float]:
        projections = pool.projection_of(lineups, spec)
        scored = score_lineups(pool, spec, lineups, universe)
        return {
            "n": int(len(lineups)),
            "median_ratio": round(float(np.median(projections)) / optimum, 4),
            "worst_ratio": round(float(projections.min()) / optimum, 4),
            "best_ratio": round(float(projections.max()) / optimum, 4),
            "p_any_wins": round(float((scored.max(axis=0) >= win_line).mean()), 4),
            "overlap": round(mean_pairwise_overlap(lineups.tolist()), 4),
        }

    before = profile(unselected)
    after = profile(selected)

    benchmark.extra_info["case"] = f"selection/{rung.name}"
    benchmark.extra_info["impl"] = "slatekit_rust"
    benchmark.extra_info["detail"] = rung.detail
    benchmark.extra_info["metric"] = "quality"
    benchmark.extra_info["quality"] = {
        "candidates": int(len(candidates)),
        "optimum": round(optimum, 2),
        **{f"unselected_{k}": v for k, v in before.items()},
        **{f"selected_{k}": v for k, v in after.items()},
    }
    benchmark.pedantic(lambda: None, rounds=1, iterations=1)

    # Selection has to earn its place on both axes, or it is machinery for its
    # own sake.
    assert after["median_ratio"] > before["median_ratio"], "selection did not lift the median"
    assert after["p_any_wins"] >= before["p_any_wins"], "selection did not improve the tail"


@pytest.mark.parametrize("rung_name", ["floor"])
def test_contest_modes_disagree(benchmark, rung_name: str) -> None:
    """Cash and tournament portfolios must each win on their own measure.

    If they tied, the two modes would have collapsed into one objective and one
    of them would be decoration.
    """
    pool = make_slate()
    rung = rung_by_name(pool, rung_name)
    spec = rung.spec
    candidates = build_lineups(pool, spec, num_lineups=CANDIDATES, seed=1)
    universe = simulated_universe(pool)
    sim = score_lineups(pool, spec, candidates, universe)

    cash_line = field_line(sim, 0.5)
    win_line = field_line(sim, 0.99)
    cash_idx = select_portfolio(sim, mode="cash", line=cash_line, n_select=PORTFOLIO)
    gpp_idx = select_portfolio(sim, mode="gpp", line=win_line, n_select=PORTFOLIO)

    # Matched size: tournament mode stops when no candidate covers a new outcome,
    # and comparing a mean over 150 entries with one over 40 compares portfolio
    # sizes rather than objectives.
    n = min(len(cash_idx), len(gpp_idx))
    cash, gpp = sim[cash_idx[:n]], sim[gpp_idx[:n]]

    cash_rate = (float((cash >= cash_line).mean()), float((gpp >= cash_line).mean()))
    win_rate = (
        float((cash.max(axis=0) >= win_line).mean()),
        float((gpp.max(axis=0) >= win_line).mean()),
    )

    benchmark.extra_info["case"] = "selection/modes"
    benchmark.extra_info["impl"] = "slatekit_rust"
    benchmark.extra_info["metric"] = "quality"
    benchmark.extra_info["quality"] = {
        "matched_size": int(n),
        "median_cash_line": round(float(np.median(cash_line)), 2),
        "median_win_line": round(float(np.median(win_line)), 2),
        # How far the bar moves between outcomes, which is the reason it is a
        # vector rather than a number.
        "win_line_spread": round(float(win_line.max() - win_line.min()), 2),
        "cash_mode_entry_cash_rate": round(cash_rate[0], 4),
        "gpp_mode_entry_cash_rate": round(cash_rate[1], 4),
        "cash_mode_p_any_wins": round(win_rate[0], 4),
        "gpp_mode_p_any_wins": round(win_rate[1], 4),
    }
    benchmark.pedantic(lambda: None, rounds=1, iterations=1)

    assert cash_rate[0] > cash_rate[1], "cash mode lost on cash rate"
    assert win_rate[1] > win_rate[0], "tournament mode lost on win probability"
