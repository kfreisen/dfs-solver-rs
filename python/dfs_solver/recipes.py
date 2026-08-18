"""Intent-named starting points over `build_lineups` and `select_portfolio`.

[`build_lineups`][dfs_solver.greedy.build_lineups] takes twelve parameters
and most of them interact. The interactions are documented, but a caller who
just wants to enter a cash game should not have to read them first. A
[`Recipe`][dfs_solver.recipes.Recipe] is a named bundle of those same
parameters, chosen coherently for one way of playing: print it to see every
choice, override any field with `dataclasses.replace`, and call `.build()` /
`.select()` to run the ordinary API with them.

```python
import dataclasses
from dfs_solver import recipes

r = recipes.gpp(spec, seed=1)
r = dataclasses.replace(r, diversity_weight=1.0)  # override, typed and checked
lineups = r.build(pool)
entries = lineups[r.select(scores, line=win_line, lineups=lineups)]
```

Recipes sit beside [`presets`][dfs_solver.presets], not inside them, because
they are the thing presets promise never to carry: strategy. A preset says what
the operator enforces; a recipe says how one kind of player approaches it.

What no recipe sets, and why:

- **The line.** It is data about the contest, not intent — an argument to
  `.select()`, never a field.
- **`chunks`, `noise`, `profiles`.** Machinery whose defaults are right. A
  caller who needs them has outgrown recipes and should call `build_lineups`
  directly; every recipe's `.build()` is a plain call to it with the fields you
  can read off the recipe.
- **Stacks and conflicts.** A `GroupConstraint` or `ConflictRule` needs slot and
  key names this module cannot know. The recipes page in the docs shows the two
  `dataclasses.replace` lines per sport.
- **`seed`.** Every factory requires it, with no default. Reproducibility is the
  caller's contract to keep, and a hidden seed would be a hidden claim that two
  runs agree by accident.

A recipe does choose the selection objective — `cash()` returning
`mode=Mode.CASH` is not a hidden default, it is what choosing the recipe means.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from dfs_solver.greedy import build_lineups
from dfs_solver.select import Mode, select_portfolio
from dfs_solver.spec import RosterSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dfs_solver.pool import PlayerPool

__all__ = [
    "Mode",
    "Recipe",
    "candidate_pool",
    "cash",
    "gpp",
    "showdown",
    "single_entry",
]


@dataclasses.dataclass(frozen=True)
class Recipe:
    """One way of playing, as explicit fields.

    Nothing here is a new code path: `.build()` calls
    [`build_lineups`][dfs_solver.greedy.build_lineups] and `.select()` calls
    [`select_portfolio`][dfs_solver.select.select_portfolio], each with
    exactly the fields on this object and the functions' own defaults for
    everything else. Print the recipe to see what was chosen; change a choice
    with `dataclasses.replace`.

    Attributes:
        spec: What makes a lineup legal — a preset, possibly adjusted (the cash
            recipe adds a salary floor; every other factory passes yours through
            untouched).
        entries: How many lineups `.build()` asks for.
        seed: Fixes `.build()`'s output exactly. Always the caller's.
        attempts_per_lineup: Attempts made per requested lineup before giving up.
        value_weight: How strongly salary is priced into a player's rank.
        diversity_weight: How strongly to fade players the portfolio already
            used.
        mode: The selection objective, or `None` when selection is not part of
            the intent — a candidate pool is built to be scored and selected
            against a simulator, not entered as-is.
        n_select: How many entries `.select()` keeps.
        max_exposure: Ceiling on how many chosen entries may contain any one
            player, applied at selection — where a cap skips candidates from a
            pool already built — rather than at construction, where it discards
            finished lineups and costs yield.
        locks: Players forced into every built lineup, as pool indices — or a
            mapping from index to slot name to pin one (`{7: "CPT"}`). A locked
            player is in 100% of entries by construction, so when this recipe
            also caps exposure, `.select()` exempts the locked players from the
            cap — the same lock-your-core-cap-the-rest rule `build_lineups`
            enforces by raising on the contradiction.
    """

    spec: RosterSpec
    entries: int
    seed: int
    attempts_per_lineup: int = 3
    value_weight: float = 0.75
    diversity_weight: float = 0.0
    mode: Mode | None = None
    n_select: int | None = None
    max_exposure: float | None = None
    locks: Sequence[int] | Mapping[int, str | None] | None = None

    def build(self, pool: PlayerPool) -> np.ndarray:
        """Build the candidate lineups this recipe describes.

        Returns:
            An `(n, roster_size)` index array from
            [`build_lineups`][dfs_solver.greedy.build_lineups]; `n` may be
            smaller than `entries` when the pool cannot support more.
        """
        return build_lineups(
            pool,
            self.spec,
            num_lineups=self.entries,
            seed=self.seed,
            attempts_per_lineup=self.attempts_per_lineup,
            value_weight=self.value_weight,
            diversity_weight=self.diversity_weight,
            locks=self.locks,
        )

    def select(
        self,
        sim_scores: np.ndarray,
        *,
        line: float | np.ndarray,
        lineups: np.ndarray | None = None,
    ) -> np.ndarray:
        """Choose which candidates to enter, under this recipe's objective.

        Args:
            sim_scores: An `(n_candidates, n_outcomes)` array from
                [`score_lineups`][dfs_solver.select.score_lineups].
            line: The score to beat — data about the contest, which is why it is
                an argument here and not a field.
            lineups: The candidate rosters, required only when the recipe caps
                exposure.

        Returns:
            Indices into `sim_scores`, as
            [`select_portfolio`][dfs_solver.select.select_portfolio] returns
            them.

        Raises:
            ValueError: If this recipe has no selection half (`mode is None`).
        """
        if self.mode is None or self.n_select is None:
            msg = (
                "this recipe has no selection half; it builds candidates for you "
                "to score and select against your own simulator"
            )
            raise ValueError(msg)
        max_exposure: float | dict[int, float] | None = self.max_exposure
        if self.max_exposure is not None and self.locks:
            # A locked player is in every candidate; a blanket cap would stop
            # the whole portfolio at the cap. Exempt the locks — the same rule
            # build_lineups enforces by raising on the contradiction.
            cap = self.max_exposure
            locked = {int(i) for i in self.locks}
            size = int(np.asarray(lineups).max()) + 1 if lineups is not None else 0
            max_exposure = {i: cap for i in range(size) if i not in locked}
        return select_portfolio(
            sim_scores,
            mode=self.mode,
            line=line,
            n_select=self.n_select,
            lineups=lineups,
            max_exposure=max_exposure,
        )


def cash(
    spec: RosterSpec,
    *,
    seed: int,
    entries: int = 20,
    salary_floor: int | None = None,
) -> Recipe:
    """A cash game: a handful of entries, each judged alone.

    Diversity is actively wrong here — a flat payout for clearing a line wants
    the most probable roster, then the next. So `value_weight` goes to `1.0`
    (measured to find the proven optimum on the slate it was tested against),
    nothing fades repetition, and selection judges each entry by its own chance
    of cashing.

    A salary floor is set because leaving money unspent is almost always a
    projection given away. It defaults to `salary_cap - 1_000`; pass
    `salary_floor` to move it, including `0` to remove it.
    """
    floor = salary_floor if salary_floor is not None else spec.salary_cap - 1_000
    return Recipe(
        spec=dataclasses.replace(spec, salary_floor=floor),
        entries=entries,
        seed=seed,
        value_weight=1.0,
        mode=Mode.CASH,
        n_select=entries,
    )


def single_entry(spec: RosterSpec, *, seed: int, candidates: int = 300) -> Recipe:
    """One entry, chosen from a small pool of candidates.

    Be clear about what this is: on a single roster, a MILP solver wins. It
    proves the optimum; this builds `candidates` good lineups and enters the one
    that scores best against your simulation. Use it when you are already in
    this pipeline — same pool, same scoring, same objects — and the difference
    between the provable best lineup and the best of three hundred is smaller
    than your projection error. When you are not, use a solver.
    """
    return Recipe(
        spec=spec,
        entries=candidates,
        seed=seed,
        attempts_per_lineup=10,
        value_weight=1.0,
        mode=Mode.CASH,
        n_select=1,
    )


def gpp(
    spec: RosterSpec,
    *,
    seed: int,
    entries: int = 150,
    diversity_weight: float = 0.6,
    locks: Sequence[int] | Mapping[int, str | None] | None = None,
) -> Recipe:
    """A tournament: many entries that win in different outcomes.

    Two entries that win in the same worlds are largely wasted, so construction
    fades players the portfolio already used (`diversity_weight`) and selection
    optimizes the chance that *some* entry clears the line (`Mode.GPP`).

    Your spec passes through untouched. Stacks and the opposing-pitcher conflict
    are strategy this module cannot write for you — they need your slot and key
    names. The recipes page shows the two `dataclasses.replace` lines.

    `locks` forces players into every built lineup; add a `max_exposure` with
    `dataclasses.replace` to cap everyone else at selection.
    """
    return Recipe(
        spec=spec,
        entries=entries,
        seed=seed,
        attempts_per_lineup=10,
        diversity_weight=diversity_weight,
        mode=Mode.GPP,
        n_select=entries,
        locks=locks,
    )


def candidate_pool(
    spec: RosterSpec,
    *,
    seed: int,
    entries: int = 20_000,
    diversity_weight: float = 1.0,
) -> Recipe:
    """A large, spread pool of candidates — or a model of the field.

    This is the build that feeds scoring and selection: far more lineups than
    will be entered, spread wide (`diversity_weight=1.0` was measured to nearly
    triple distinct-player coverage for 3% of the median entry's projection).
    There is no selection half; what to keep depends on the contest, which is
    the next recipe's job.
    """
    return Recipe(
        spec=spec,
        entries=entries,
        seed=seed,
        attempts_per_lineup=5,
        diversity_weight=diversity_weight,
    )


def showdown(spec: RosterSpec, *, seed: int, entries: int = 150) -> Recipe:
    """A single-game contest, where the captain slot does the work.

    Pass a showdown spec — `DK_MLB_SHOWDOWN` or `DK_NFL_SHOWDOWN` from
    [`presets`][dfs_solver.presets] — with the
    both-teams rule added once your records carry a `team` key:
    `replace(spec, groups=(*spec.groups, GroupConstraint(key="team", min_distinct=2)))`.
    The slot multipliers live on the spec, so nothing else changes: the same
    build, scoring, and selection treat the captain as 1.5x everywhere.
    """
    return Recipe(
        spec=spec,
        entries=entries,
        seed=seed,
        attempts_per_lineup=10,
        mode=Mode.GPP,
        n_select=entries,
    )
