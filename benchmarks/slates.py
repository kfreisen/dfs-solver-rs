"""The slate every benchmark is measured on, and the vocabulary for describing a draw.

Shared so every scenario refers to the same problem. If each built its own slate,
a timing table and a description of the output could not be read together. The
scenarios themselves live in `scenarios.py`.

`describe_lineups` is the other half: player coverage, exposure, the spread of
projections. Deliberately no simulation, no field model and no payout — those
need inputs this package does not supply, so a benchmark computing them would be
reporting on a fixture written here rather than on either implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.presets import DK_MLB_CLASSIC, DK_MLB_SHOWDOWN
from mlb_dfs_solver.spec import RosterSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "HITTER_SLOTS",
    "SALARY_FLOOR",
    "describe_lineups",
    "lineup_overlap",
    "make_showdown_slate",
    "make_slate",
]

HITTER_SLOTS = ("C", "SS", "2B", "3B", "1B", "OF")

# What the scenarios that impose a floor use. DraftKings imposes none; spending
# nearly all the cap is a strategy, so it belongs here rather than in the preset.
SALARY_FLOOR = 49_000

# Every third hitter carries a second position, which is roughly what a
# DraftKings MLB slate looks like. Nothing in this repository had a
# multi-position player before, which left the one formulation detail the MILP
# baseline exists to handle -- a binary per (player, slot) rather than per player
# -- unexercised by every published number.
_SECOND_POSITION = {
    "C": "1B",
    "1B": "OF",
    "2B": "SS",
    "3B": "1B",
    "SS": "2B",
    "OF": "1B",
}
_MULTI_POSITION_EVERY = 3

# Thirty-two per position, outfield tripled, so 288 players — the size of a real
# DraftKings MLB main slate. Sized up twice, both times because a thin slate was
# measuring the wrong thing. At 90 players the later constraint rungs ran out of
# legal lineups. At 144 the candidate pool topped out around 6,000 distinct
# lineups, which starved selection and made its quality numbers pessimistic: the
# same measurement at 288 puts the median entry at 0.90 of the optimum rather
# than 0.83, because there were simply better candidates to choose from.
_PER_POSITION = 32


def make_slate(per_position: int = _PER_POSITION) -> PlayerPool:
    """A realistic-shaped DraftKings MLB slate, with games as well as teams.

    Deterministic: a benchmark whose input changes between runs cannot be compared
    against a committed result.

    Ten teams paired into five games, so `opponent` and `game` are both real keys
    rather than decoration — the conflict rule needs the first and the
    distinct-games rule needs the second.

    Team assignment is deliberately *not* `k % 10`. Salary and projection are both
    driven by `k`, so that mapping correlates team with price at 0.63 — every good
    player on one team, every cheap one on another. Every team-shaped constraint
    then measures that artefact instead of itself: the conflict rung collapsed to
    two lineups out of 150 before this was fixed, not because conflicts are hard
    but because excluding a team excluded a price bracket. Mixing by `7k + 3p`
    keeps the correlation at 0.09 with teams still evenly sized.

    Projections are distinct per player, which the obvious `4.0 + (k % 12) * 1.2`
    is not: it gives a 144-player slate twelve distinct `(salary, projection)`
    pairs and eleven exact clones of everybody. That wrecks every quality
    measurement made on it. The optimum stops being a single lineup, and the
    solver's no-good cuts produce "different" lineups by swapping interchangeable
    players — which made the solver look *more* diverse than randomized
    construction, reversing the one comparison this package rests on. Salary stays
    tiered, because real slates are priced in tiers.

    Projection is also deliberately *not* a monotone function of salary. Pricing
    players strictly by projection makes the optimization degenerate: every legal
    lineup that spends the cap scores about the same, so the salary floor alone
    picks a near-optimal roster and there is no edge to find. It measured as a
    0.997 correlation, and the visible symptom was a median lineup landing at the
    70th percentile of *arbitrary* legal lineups rather than the high nineties —
    not because construction was poor but because the slate had nothing to
    discriminate. Mispriced players are the entire reason this problem is worth
    solving; the term below puts the correlation at 0.91, which is about what a
    real slate looks like, and keeps every projection positive.

    Every third hitter is eligible at a second position. Real slates are full of
    them, and they are the reason the MILP baseline assigns a binary per
    `(player, slot)` rather than per player. Measured, they cost CP-SAT more than
    they cost the kernel — 63% against 25% — so their absence was understating
    the gap rather than inflating it.
    """
    records: list[dict[str, object]] = []
    for position_index, position in enumerate(("P", "C", "1B", "2B", "3B", "SS", "OF")):
        count = per_position * 3 if position == "OF" else per_position
        for k in range(count):
            team = (7 * k + 3 * position_index) % 10
            opponent = team ^ 1
            positions = (position,)
            if position in _SECOND_POSITION and k % _MULTI_POSITION_EVERY == 0:
                positions = (position, _SECOND_POSITION[position])
            records.append(
                {
                    "name": f"{position}-{k}",
                    "positions": positions,
                    "salary": 2500 + (k % 12) * 750,
                    "projection": (
                        4.0
                        + (k % 12) * 1.2
                        + (k // 12) * 0.29
                        + position_index * 0.037
                        # Value, decoupled from price. See below.
                        + (((k * 13 + position_index * 5) % 7) - 3) * 0.9
                    ),
                    "stddev": 3.0 + (k % 4),
                    "ownership": ((k * 7) % 30) / 100.0,
                    "team": f"TM{team}",
                    "opponent": f"TM{opponent}",
                    "game": f"G{min(team, opponent)}",
                }
            )
    return PlayerPool.from_records(records, DK_MLB_CLASSIC, key_fields=["team", "opponent", "game"])


def make_showdown_slate(per_team: int = 20) -> PlayerPool:
    """One game, two teams — the format where slot multipliers are load bearing.

    A captain scores 1.5x and costs 1.5x, so the same player is a different
    proposition depending on where they are rostered. Both implementations price
    per placement rather than per player to handle it, and until this slate
    existed nothing measured that path.

    Built against `DK_MLB_SHOWDOWN`, whose slots take every position, so position
    eligibility does no work here and the multipliers do all of it.
    """
    records: list[dict[str, object]] = []
    for team_index in range(2):
        for k in range(per_team):
            records.append(
                {
                    "name": f"TM{team_index}-{k}",
                    # Showdown slots accept anyone; the label is kept so the pool
                    # still reads like a baseball roster.
                    "positions": ("P",) if k == 0 else ("OF",),
                    "salary": 2500 + (k % 12) * 750,
                    "projection": (
                        4.0
                        + (k % 12) * 1.2
                        + team_index * 0.31
                        + (((k * 13 + team_index * 5) % 7) - 3) * 0.9
                    ),
                    "stddev": 3.0 + (k % 4),
                    "ownership": ((k * 7) % 30) / 100.0,
                    "team": f"TM{team_index}",
                }
            )
    return PlayerPool.from_records(records, DK_MLB_SHOWDOWN, key_fields=["team"])


def describe_lineups(
    pool: PlayerPool, spec: RosterSpec, lineups: np.ndarray, optimum: float | None = None
) -> dict[str, float | int]:
    """What a set of generated lineups looks like.

    Every figure is a property of the lineups and the inputs the caller supplied.
    Nothing here simulates an outcome, models a field, or scores a contest: those
    numbers describe a fixture rather than a package, and two implementations
    cannot be separated on them without the reader taking the fixture on trust.

    What is left is still the whole comparison. A method that returns 150 rosters
    built from 33 players, all projecting within a tenth of a point of each other,
    is doing something visibly different from one that returns 150 built from 117
    -- and both halves of that are checkable by reading the lineups.

    Args:
        pool: The players the lineups index into.
        spec: Supplies slot multipliers for the salary and projection totals.
        lineups: An `(n, roster_size)` index array.
        optimum: The proven best single roster's projection, if known. Adds the
            three `*_ratio` fields.

    Returns:
        A flat mapping, ready to be recorded as a benchmark's `quality` payload.
    """
    n = len(lineups)
    if n == 0:
        return {"lineups": 0}

    projections = pool.projection_of(lineups, spec)
    salaries = pool.salary_of(lineups, spec)
    counts = np.bincount(np.asarray(lineups).ravel(), minlength=len(pool))
    exposure = counts / n
    used = exposure[exposure > 0]

    teams = pool.keys["team"]
    blocks = [
        max(np.bincount(teams[row][teams[row] >= 0]).tolist() or [0]) for row in np.asarray(lineups)
    ]

    described: dict[str, float | int] = {
        "lineups": int(n),
        # Coverage: how much of the slate the method is willing to touch.
        "distinct_players": int((exposure > 0).sum()),
        "pool_size": int(len(pool)),
        "players_over_50pct": int((exposure > 0.5).sum()),
        "players_over_20pct": int((exposure > 0.2).sum()),
        "max_exposure": round(float(exposure.max()), 4),
        "median_exposure_used": round(float(np.median(used)), 4),
        # Spread: a portfolio whose entries all project alike is one entry.
        "projection_min": round(float(projections.min()), 2),
        "projection_median": round(float(np.median(projections)), 2),
        "projection_max": round(float(projections.max()), 2),
        "salary_min": int(salaries.min()),
        "salary_median": int(np.median(salaries)),
        "salary_max": int(salaries.max()),
        "team_block_median": int(np.median(blocks)),
        "team_block_max": int(max(blocks)),
    }
    if optimum:
        described["best_ratio"] = round(float(projections.max()) / optimum, 4)
        described["median_ratio"] = round(float(np.median(projections)) / optimum, 4)
        described["worst_ratio"] = round(float(projections.min()) / optimum, 4)
    return described


def lineup_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    """Fraction of players two lineups share.

    The diversity measure the portfolio argument rests on. Two lineups differing
    by one player out of ten overlap 0.9, and a portfolio of those is not a
    portfolio.
    """
    if not a:
        return 0.0
    return len(set(a) & set(b)) / len(set(a))
