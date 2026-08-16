"""The configurations benchmarked, each one a way somebody actually plays.

The previous version of this file measured a "constraint ladder" at 25 lineups.
Twenty-five was not a number anyone plays; it was the largest portfolio CP-SAT
could finish inside a pytest run. A harness limit had become the published
experiment.

These are sized to the game instead. A DraftKings MLB classic contest takes 150
entries, so mass-multi-entry scenarios ask for 150. Cash games are played a
handful of entries deep, so that one asks for 20. The contest-scale row asks for
10,000, which is what a field simulation or a candidate pool needs and what
nobody generates with a solver.

The scenarios are cumulative: each adds one thing a player turns on, in roughly
the order they turn it on, so the cost of each is isolated and the last one is
the whole configuration rather than a toy.

| Scenario | What it adds | Entries |
| --- | --- | ---: |
| `rules-only` | what DraftKings enforces, nothing else | 150 |
| `single-entry` | one lineup — the case a solver wins | 1 |
| `cash` | a salary floor, few entries | 20 |
| `stack` | a four-hitter team stack | 150 |
| `conflict` | no hitters against the rostered pitcher | 150 |
| `locks` | an ace and a value bat in every lineup | 150 |
| `mme` | exposure caps on everyone else | 150 |
| `mme-diverse` | fading players the portfolio already used | 150 |
| `showdown` | single game, captain at 1.5x score and salary | 150 |
| `contest-scale` | the `mme` configuration at field size | 10,000 |

`single-entry` is where the solver legitimately wins: on one roster CP-SAT
proves the optimum and randomized construction returns something slightly
worse. The row exists because a comparison that only shows the cases this
package wins is an advertisement.

`mme-diverse` has no solver row. `diversity_weight` is a preference with no
CP-SAT analog, and its comparator is the `mme` row directly above it — the same
configuration with the preference off.

## What is not expressible

A real mass-multi-entry player usually wants a **secondary stack** — a 4-2 or
5-3 structure, where a second team supplies two or three more hitters. That
cannot be stated here. `GroupConstraint(min_stack=...)` is existential over *one*
key value: it asks that some team reach the size, and there is no way to ask that
a *different* team also reach a second size. Nor is "every team used must supply
at least two", which is the same gap from the other side.

This is a real limitation on how faithfully these scenarios mimic MME, and it is
recorded here rather than worked around, because working around it would mean
generating with a stack and filtering afterwards — which measures a filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from mlb_dfs_solver.presets import DK_MLB_CLASSIC, DK_MLB_SHOWDOWN
from mlb_dfs_solver.spec import ConflictRule, GroupConstraint, RosterSpec
from slates import HITTER_SLOTS, SALARY_FLOOR, make_showdown_slate, make_slate

if TYPE_CHECKING:
    from mlb_dfs_solver.pool import PlayerPool

__all__ = ["SCENARIO_NAMES", "Scenario", "build_scenarios", "scenario_by_name"]

# What a DraftKings MLB classic contest actually allows one entrant.
MME_ENTRIES = 150
# Cash games are played a few entries deep, not a hundred and fifty.
CASH_ENTRIES = 20
# Field size for the scale row. See `_contest_slate` for why the slate is larger.
CONTEST_ENTRIES = 10_000
# Exposure ceiling for the MME scenarios. Note this is a fraction of entries
# *requested*; a cap lowers yield, so the realized share runs higher. Both are
# recorded, which is the point.
MME_EXPOSURE_CAP = 0.6


@dataclass(frozen=True)
class Scenario:
    """One benchmarked configuration.

    Attributes:
        name: Short id, used as the case name in results.
        detail: One line describing what this adds, for the published table.
        pool: The slate to build against.
        spec: What makes a lineup legal.
        entries: How many lineups to ask both implementations for.
        attempts: Attempts per requested lineup, the same for both so the
            comparison is "how does the output differ at this budget".
        locks: Players forced into every lineup.
        max_exposure: Per-player ceiling, applied by both implementations.
        solver: Whether to run the MILP baseline. Off where it cannot finish in
            a sane wall clock, or where the thing measured has no solver analog.
        solver_budget_s: Wall-clock budget for the solver's whole portfolio, or
            None for no budget. The contest-scale row sets one because 10,000
            no-good-cut solves have no natural upper bound; the report then says
            how many lineups the budget bought, which is the honest form of the
            claim.
        build_extra: Extra keyword arguments for `build_lineups`, merged last.
            How a scenario reaches a knob — `diversity_weight` — without this
            dataclass growing a field per parameter.
    """

    name: str
    detail: str
    pool: PlayerPool
    spec: RosterSpec
    entries: int
    attempts: int = 5
    locks: tuple[int, ...] = ()
    max_exposure: dict[int, float] | None = None
    solver: bool = True
    solver_budget_s: float | None = None
    build_extra: dict[str, Any] = field(default_factory=dict)

    @property
    def build_kwargs(self) -> dict[str, Any]:
        """Arguments for `build_lineups`."""
        return {
            "num_lineups": self.entries,
            "seed": 1,
            "attempts_per_lineup": self.attempts,
            "locks": list(self.locks) or None,
            "max_exposure": self.max_exposure,
            **self.build_extra,
        }

    @property
    def solver_kwargs(self) -> dict[str, Any]:
        """Arguments for `solve_portfolio_ortools`."""
        return {
            "num_lineups": self.entries,
            "locks": list(self.locks) or None,
            "max_exposure": self.max_exposure,
            **({"budget_s": self.solver_budget_s} if self.solver_budget_s else {}),
        }


def _pick_locks(pool: PlayerPool, spec: RosterSpec) -> tuple[int, ...]:
    """The ace, and the best value bat that does not clash with them.

    Derived from the pool rather than written down as indices, which point at a
    different player the moment the slate changes shape.

    The bat is chosen by projection *per dollar* rather than by projection.
    Locking the best of both leaves 21,500 of a 50,000 cap in two players, which
    against a 49,000 floor and a four-hitter stack starves the scenario — it
    measures a slate with nothing left rather than the cost of locking. It is
    also what anyone actually does: you lock an ace and a bargain, not two
    max-priced players.
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


# The classic slate tops out at roughly six thousand distinct lineups under a
# stack, so asking it for ten thousand would measure that ceiling rather than
# throughput. Forty-eight per position reaches ten thousand with room over, and
# 432 players is a realistic size for a large MLB main slate.
_CONTEST_PER_POSITION = 48


def build_scenarios() -> list[Scenario]:
    """Every scenario, in the order a player turns each thing on."""
    pool = make_slate()

    # What DraftKings enforces and nothing else: cap, slots, the two team caps,
    # and players from at least two games.
    rules = replace(
        DK_MLB_CLASSIC,
        groups=(*DK_MLB_CLASSIC.groups, GroupConstraint(key="game", min_distinct=2)),
    )
    floor = replace(rules, salary_floor=SALARY_FLOOR)
    stacked = replace(
        floor,
        groups=(*floor.groups, GroupConstraint(key="team", min_stack=4, slots=HITTER_SLOTS)),
    )
    conflicted = replace(
        stacked,
        conflicts=(
            ConflictRule(
                left_key="opponent",
                right_key="team",
                left_positions=("P",),
                right_positions=HITTER_SLOTS,
            ),
        ),
    )
    locks = _pick_locks(pool, conflicted)
    # Capping every player would contradict the locks, which are in 100% of
    # lineups by construction, and the library rejects that rather than silently
    # honouring one. So the cap names everyone else — which is what a caller
    # wants anyway: lock your core, spread the rest.
    caps = {i: MME_EXPOSURE_CAP for i in range(len(pool)) if i not in locks}

    showdown_pool = make_showdown_slate()
    showdown = replace(
        DK_MLB_SHOWDOWN,
        groups=(GroupConstraint(key="team", min_distinct=2),),
    )

    contest_pool = make_slate(_CONTEST_PER_POSITION)
    contest_locks = _pick_locks(contest_pool, conflicted)
    contest_caps = {i: MME_EXPOSURE_CAP for i in range(len(contest_pool)) if i not in contest_locks}

    return [
        Scenario(
            name="rules-only",
            detail="what DraftKings enforces: cap, slots, team caps, 2 distinct games",
            pool=pool,
            spec=rules,
            entries=MME_ENTRIES,
        ),
        Scenario(
            name="single-entry",
            detail="one lineup under cash rules — the case a solver wins",
            pool=pool,
            spec=floor,
            entries=1,
            # A one-lineup build is microseconds; a real budget is what a player
            # asking for one roster would give it.
            attempts=200,
        ),
        Scenario(
            name="cash",
            detail="+ a salary floor, played a few entries deep",
            pool=pool,
            spec=floor,
            entries=CASH_ENTRIES,
        ),
        Scenario(
            name="stack",
            detail="+ a 4-hitter team stack",
            pool=pool,
            spec=stacked,
            entries=MME_ENTRIES,
            attempts=10,
        ),
        Scenario(
            name="conflict",
            detail="+ no hitters against the rostered pitcher",
            pool=pool,
            spec=conflicted,
            entries=MME_ENTRIES,
            attempts=10,
        ),
        Scenario(
            name="locks",
            detail="+ an ace and a value bat in every lineup",
            pool=pool,
            spec=conflicted,
            entries=MME_ENTRIES,
            attempts=10,
            locks=locks,
        ),
        Scenario(
            name="mme",
            detail=f"+ {MME_EXPOSURE_CAP:.0%} exposure cap on every unlocked player",
            pool=pool,
            spec=conflicted,
            entries=MME_ENTRIES,
            attempts=10,
            locks=locks,
            max_exposure=caps,
        ),
        Scenario(
            name="mme-diverse",
            detail="+ diversity_weight=0.6, fading players the portfolio already used",
            pool=pool,
            spec=conflicted,
            entries=MME_ENTRIES,
            attempts=10,
            locks=locks,
            max_exposure=caps,
            # No CP-SAT analog: this is a preference, not a constraint. The
            # comparator is the `mme` row, which is this row with the weight off.
            solver=False,
            build_extra={"diversity_weight": 0.6},
        ),
        Scenario(
            name="showdown",
            detail="single game, captain at 1.5x score and salary, both teams required",
            pool=showdown_pool,
            spec=showdown,
            entries=MME_ENTRIES,
            attempts=10,
        ),
        Scenario(
            name="contest-scale",
            detail=(f"the mme configuration at field size, {len(contest_pool)}-player slate"),
            pool=contest_pool,
            spec=conflicted,
            entries=CONTEST_ENTRIES,
            attempts=15,
            locks=contest_locks,
            max_exposure=contest_caps,
            # 10,000 no-good-cut solves have no natural upper bound — the last
            # attempt at an unbudgeted run was killed mid-way. Four hours buys a
            # measured per-lineup cost and an honest "N of 10,000 in 4 h" row.
            solver_budget_s=4 * 3600.0,
        ),
    ]


SCENARIO_NAMES: tuple[str, ...] = (
    "rules-only",
    "single-entry",
    "cash",
    "stack",
    "conflict",
    "locks",
    "mme",
    "mme-diverse",
    "showdown",
    "contest-scale",
)
"""Every scenario name, in order. Listed separately so a caller can name one
without paying to construct the slates."""


def scenario_by_name(name: str) -> Scenario:
    """Look up one scenario by name."""
    for scenario in build_scenarios():
        if scenario.name == name:
            return scenario
    known = ", ".join(SCENARIO_NAMES)
    msg = f"unknown scenario {name!r}; known: {known}"
    raise KeyError(msg)
