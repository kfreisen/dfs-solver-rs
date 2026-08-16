"""The whole job a tournament player runs, timed stage by stage.

The scenario tables time one call. A sharp's actual session is a pipeline —
build a candidate pool, build a field to draw the line from, simulate, score
everything, select a tournament portfolio and a cash portfolio — and the honest
speed claim about this package is the wall clock on that whole job, not on its
fastest piece.

Every reported figure is a stage timing or a description of the lineups
(spread, exposure, overlap with the field). **Nothing here scores a contest.**
An in-the-money rate computed inside this harness would be a quantile of a
synthetic field, scored by a synthetic simulator, both written a few files away
— a number that moves when the fixture is rewritten and describes nothing but
the fixture. Those measures were built once and removed; do not reinstate them.

The candidate build goes through `mlb_dfs_solver.recipes`, which doubles as a
live check that the recipe layer really is the plain calls it documents. The
field build calls `build_lineups` directly, because a field model wants the
machinery knobs (`noise`, `profiles`) the recipes deliberately leave out.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
from mlb_dfs_solver import JitterProfile, build_lineups, recipes, score_lineups, select_portfolio
from scenarios import scenario_by_name
from simulate import simulate
from slates import describe_lineups

__all__ = ["run_workflow"]

# Full-run sizes: a candidate pool big enough to select 150 from, a field the
# size of a large contest, and enough outcomes for the lines to be stable.
CANDIDATES = 20_000
FIELD = 100_000
OUTCOMES = 1_000
GPP_ENTRIES = 150
CASH_ENTRIES = 20

# The crowd builds under looser, more chalk-shaped settings than our candidates,
# mirroring examples/02_pipeline.py: no ownership fade, wider noise.
_CROWD = (
    JitterProfile(ceiling=(0.1, 0.6), leverage=(0.0, 0.1)),
    JitterProfile(ceiling=(0.3, 1.2), leverage=(0.1, 0.5)),
)


def run_workflow(*, smoke: bool = False) -> dict[str, Any]:
    """Run the pipeline once, timing each stage with a plain `perf_counter`.

    Stages are seconds long and run once — repetition statistics belong to the
    single-call scenarios. Returns the record stored under the result file's
    `workflow` key.
    """
    scale = 100 if smoke else 1
    n_candidates = max(CANDIDATES // scale, 100)
    n_field = max(FIELD // scale, 200)
    n_outcomes = max(OUTCOMES // scale, 50)
    n_gpp = max(GPP_ENTRIES // scale, 5)
    n_cash = max(CASH_ENTRIES // scale, 2)

    # The contest-scale slate and specs: 432 players, and a strategy spec that
    # carries the floor, the 4-hitter stack and the opposing-pitcher conflict.
    # The classic 288-player slate tops out around six thousand distinct lineups
    # under a stack, which would cap the candidate pool rather than measure it.
    strategy_scenario = scenario_by_name("contest-scale")
    pool, strategy = strategy_scenario.pool, strategy_scenario.spec
    # The crowd obeys the contest, not our strategy: cap, team limits, distinct
    # games — no floor, no stack, no conflict rule. Forcing the field under our
    # strategy shrinks it several-fold and draws the line from too few entries.
    contest_rules = scenario_by_name("rules-only").spec

    stages: list[dict[str, Any]] = []

    def timed(name: str, fn: Any) -> Any:
        start = time.perf_counter()
        result = fn()
        stages.append({"name": name, "seconds": time.perf_counter() - start})
        print(f"  workflow: {name:18s} {stages[-1]['seconds']:8.2f} s", flush=True)
        return result

    candidates = timed(
        "build candidates",
        lambda: recipes.candidate_pool(strategy, seed=1, entries=n_candidates).build(pool),
    )
    field = timed(
        "build field",
        lambda: build_lineups(
            pool,
            replace(contest_rules, salary_floor=0),
            num_lineups=n_field,
            seed=99,
            noise=0.45,
            profiles=_CROWD,
        ),
    )

    universe = timed("simulate", lambda: simulate(pool, n_outcomes))
    field_scores = timed("score field", lambda: score_lineups(pool, strategy, field, universe))
    scores = timed("score candidates", lambda: score_lineups(pool, strategy, candidates, universe))

    def lines() -> tuple[np.ndarray, np.ndarray]:
        win = np.quantile(field_scores, 0.999, axis=0).astype(np.float32)
        cash = np.quantile(field_scores, 0.50, axis=0).astype(np.float32)
        return win, cash

    win_line, cash_line = timed("draw lines", lines)
    gpp_idx = timed(
        "select gpp",
        lambda: select_portfolio(scores, mode="gpp", line=win_line, n_select=n_gpp),
    )
    cash_idx = timed(
        "select cash",
        lambda: select_portfolio(scores, mode="cash", line=cash_line, n_select=n_cash),
    )

    gpp_lineups, cash_lineups = candidates[gpp_idx], candidates[cash_idx]

    # How the tournament portfolio sits against the crowd, descriptively: how
    # much it looks like a field entry, and how far its heaviest player runs
    # above that player's field ownership.
    field_exposure = np.bincount(field.ravel(), minlength=len(pool)) / max(len(field), 1)
    gpp_exposure = np.bincount(gpp_lineups.ravel(), minlength=len(pool)) / max(len(gpp_lineups), 1)

    return {
        "sizes": {
            "candidates": int(len(candidates)),
            "field": int(len(field)),
            "outcomes": int(n_outcomes),
            "pool": int(len(pool)),
        },
        "stages": stages,
        "total_seconds": float(sum(s["seconds"] for s in stages)),
        "gpp": {
            **describe_lineups(pool, strategy, gpp_lineups),
            "field_overlap": round(_cross_overlap(gpp_lineups, field), 4),
            "top_leverage": round(float((gpp_exposure - field_exposure).max()), 4),
        },
        "cash": {
            **describe_lineups(pool, strategy, cash_lineups),
            "field_overlap": round(_cross_overlap(cash_lineups, field), 4),
        },
    }


def _cross_overlap(portfolio: np.ndarray, field: np.ndarray) -> float:
    """Mean players shared between a portfolio entry and a field entry.

    The same membership-matrix product as `slates._mean_overlap`, across the two
    sets, on a seeded field sample. High means the portfolio looks like the
    crowd; low means it is differentiated. Neither is declared better — that
    depends on the contest — which is why this is reported and not scored.
    """
    if not len(portfolio) or not len(field):
        return 0.0
    sample = field
    if len(sample) > 500:
        picks = np.random.default_rng(0).choice(len(sample), size=500, replace=False)
        sample = sample[picks]
    pool_size = int(max(portfolio.max(), sample.max())) + 1
    ours = np.zeros((len(portfolio), pool_size), dtype=np.float32)
    ours[np.arange(len(portfolio))[:, None], portfolio] = 1.0
    theirs = np.zeros((len(sample), pool_size), dtype=np.float32)
    theirs[np.arange(len(sample))[:, None], sample] = 1.0
    return float((ours @ theirs.T).mean() / portfolio.shape[1])
