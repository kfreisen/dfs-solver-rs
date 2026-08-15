"""The slate, the constraint ladder, and the vocabulary for describing a draw.

Shared so every benchmark refers to the same problem. If each built its own
slate, a timing table and a description of the output could not be read together.

Two things live here.

**The ladder.** A single "is it fast?" number says nothing about a feature: every
constraint added is work for the kernel and work for the solver, and they do not
scale alike. Walking the rungs one at a time shows where each approach struggles.

**`describe_lineups`.** The measures a draw is characterized by — player
coverage, exposure, the spread of projections. Deliberately no simulation, no
field model, and no payout: those need inputs this package does not supply, so a
benchmark computing them would be reporting on a fixture written here rather than
on either implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.presets import DK_MLB_CLASSIC
from mlb_dfs_solver.spec import ConflictRule, GroupConstraint, RosterSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CONTEST_EXPOSURE_CAP",
    "CONTEST_PER_POSITION",
    "CONTEST_RUNG",
    "Rung",
    "build_ladder",
    "describe_lineups",
    "lineup_overlap",
    "make_slate",
    "rung_by_name",
]

HITTER_SLOTS = ("C", "SS", "2B", "3B", "1B", "OF")

# The floor the ladder's `floor` rung and everything above it build under.
SALARY_FLOOR = 49_000

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
    """
    records: list[dict[str, object]] = []
    for position_index, position in enumerate(("P", "C", "1B", "2B", "3B", "SS", "OF")):
        count = per_position * 3 if position == "OF" else per_position
        for k in range(count):
            team = (7 * k + 3 * position_index) % 10
            opponent = team ^ 1
            records.append(
                {
                    "name": f"{position}-{k}",
                    "positions": (position,),
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


@dataclass(frozen=True)
class Rung:
    """One step of the constraint ladder.

    Attributes:
        name: Short label used as the benchmark case id.
        detail: What this rung adds, for the published table.
        spec: The specification to build against.
        locks: Players forced into every lineup.
        max_exposure: Portfolio-level exposure cap, if any.
    """

    name: str
    detail: str
    spec: RosterSpec
    locks: tuple[int, ...] = ()
    max_exposure: dict[int, float] | None = None


def _pick_locks(pool: PlayerPool, spec: RosterSpec) -> tuple[int, ...]:
    """The ace, and the best value bat that does not clash with them.

    Derived from the pool rather than written down as indices. Hardcoded indices
    point at a different player the moment the slate changes size, and the two
    that looked fine on a 90-player slate turned out to be a pitcher and a hitter
    on his opposing team at 144 — which the conflict rung correctly refuses,
    yielding zero lineups and a benchmark measuring nothing.

    The bat is chosen by projection *per dollar* rather than by projection. Taking
    the best of both leaves 21_500 of a 50_000 cap in two players, and against a
    49_000 floor and a four-hitter stack that starves the rung: 5 lineups of 150,
    which measures a slate with nothing left rather than the cost of locking. By
    value it yields 88, which is the number worth publishing. It is also what
    anyone actually does — you lock an ace and a bargain, not two max-priced
    players.
    """
    pitcher_mask = spec.mask_for(("P",))
    pitchers = [i for i in range(len(pool)) if int(pool.positions[i]) & pitcher_mask]
    ace = max(pitchers, key=lambda i: (float(pool.projections[i]), -i))

    forbidden = {
        int(b) for a, b in zip(*pool.conflict_pairs(spec).tolist(), strict=True) if int(a) == ace
    }
    bats = [
        i
        for i in range(len(pool))
        if not int(pool.positions[i]) & pitcher_mask and i not in forbidden and i != ace
    ]
    bat = max(bats, key=lambda i: (float(pool.projections[i]) / int(pool.salaries[i]), -i))
    return (ace, bat)


def build_ladder(pool: PlayerPool) -> list[Rung]:
    """Build the rungs, each adding one constraint to the one before it.

    Deliberately cumulative rather than one-at-a-time. A caller does not turn on
    stacking *instead of* a salary floor, they turn it on as well, and the cost of
    a constraint depends on what it is layered onto — a stack over an already
    tight cap is a different problem from a stack alone.
    """
    # Rung 0 strips DK_MLB_CLASSIC back to the bare assignment problem, so the
    # ladder starts somewhere both implementations find easy.
    bare = replace(DK_MLB_CLASSIC, salary_floor=0, groups=())
    caps = replace(bare, groups=DK_MLB_CLASSIC.groups)
    # A floor is strategy, not a DraftKings rule, so the preset no longer carries
    # one and the ladder states it. 49_000 of a 50_000 cap is a tight but
    # ordinary setting: leaving salary unspent is usually a mistake.
    floor = replace(caps, salary_floor=SALARY_FLOOR)
    games = replace(
        floor,
        groups=(*floor.groups, GroupConstraint(key="game", min_distinct=2)),
    )
    conflicts = replace(
        games,
        conflicts=(
            ConflictRule(
                left_key="opponent",
                right_key="team",
                left_positions=("P",),
                right_positions=HITTER_SLOTS,
            ),
        ),
    )
    stacked = replace(
        conflicts,
        groups=(
            *conflicts.groups,
            GroupConstraint(key="team", min_stack=4, slots=HITTER_SLOTS),
        ),
    )
    locks = _pick_locks(pool, stacked)
    # Capping every player at 40% would contradict the locks, which are in 100%
    # of lineups by construction, and the library rejects that rather than
    # silently honouring one of the two. So the cap names everyone else — which
    # is what a caller actually wants anyway: lock your core, spread the rest.
    exposure = {i: 0.4 for i in range(len(pool)) if i not in locks}
    return [
        Rung("bare", "salary cap and position slots only", bare),
        Rung("caps", "+ team caps (6 total, 5 hitters)", caps),
        Rung("floor", "+ salary floor", floor),
        Rung("games", "+ players from 2 distinct games", games),
        Rung("conflicts", "+ no hitters against the rostered pitcher", conflicts),
        Rung("stack", "+ a 4-hitter team stack", stacked),
        Rung("locks", "+ two locked players", stacked, locks=locks),
        Rung(
            "exposure",
            "+ 40% exposure cap on every unlocked player",
            stacked,
            locks=locks,
            max_exposure=exposure,
        ),
    ]


RUNG_NAMES = (
    "bare",
    "caps",
    "floor",
    "games",
    "conflicts",
    "stack",
    "locks",
    "exposure",
)
"""Every rung, in increasing order of constraint.

Named separately from `build_ladder` so a benchmark can parametrize over them
without building a pool at collection time.
"""


def rung_by_name(pool: PlayerPool, name: str) -> Rung:
    """Look up one rung of the ladder built against `pool`."""
    for rung in build_ladder(pool):
        if rung.name == name:
            return rung
    known = ", ".join(RUNG_NAMES)
    msg = f"unknown rung {name!r}; the ladder has: {known}"
    raise KeyError(msg)


# --- Contest scale -----------------------------------------------------------
#
# A field-sized draw needs a config that can actually supply one. The 288-player
# slate tops out at 6,094 distinct lineups under the stack rung, so asking it for
# 10,000 measures the ceiling rather than throughput. 48 per position reaches
# 10,000 with room over, and 432 players is a realistic size for a large MLB main
# slate.
CONTEST_PER_POSITION = 48
# Stacked, because a real field stacks. This is the one place a stack is not our
# strategy: it is a description of what the crowd does.
CONTEST_RUNG = "stack"
# Without a cap one player reaches 82.9% of the draw, which no real field shows.
#
# Read the realized exposure this produces, not the number requested. The limit
# is `floor(cap x lineups *requested*)`, and a cap lowers yield, so the fraction
# of what actually comes back is higher than the cap asked for -- at 60% here,
# 8,650 lineups come back and the top player holds 69.4% of them.
CONTEST_EXPOSURE_CAP = 0.6


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
