"""Ready-made roster specifications for common contest formats.

These are data, not code paths — every one is an ordinary
[`RosterSpec`][mlb_dfs_solver.spec.RosterSpec] you can copy and modify. There is no
branch anywhere in this package that asks which sport it is looking at.

Operators change these rules, sometimes mid-season. Treat a preset as a starting
point that was correct when written, and check it against the contest you are
actually entering.

No preset declares a [`ConflictRule`][mlb_dfs_solver.spec.ConflictRule] or a
`min_stack`. Rules here are the ones the operator enforces; "no hitters against my
pitcher" and "stack four bats" are strategies, and a preset that quietly imposed
either would be wrong for anyone playing differently. Add them with
`dataclasses.replace`.

Nor does any preset carry the "players from at least two different games" rule
these contests do impose. It needs a `game` key on every player, which a pool
built for a preset that never asked for one would not have, and a constraint
naming a missing key fails loudly at build time. Add it yourself once your
records carry the key:

```python
from dataclasses import replace
from mlb_dfs_solver.presets import DK_NFL_CLASSIC
from mlb_dfs_solver.spec import GroupConstraint

spec = replace(
    DK_NFL_CLASSIC,
    groups=(*DK_NFL_CLASSIC.groups, GroupConstraint(key="game", min_distinct=2)),
)
```
"""

from __future__ import annotations

from mlb_dfs_solver.spec import GroupConstraint, RosterSpec, Slot

__all__ = ["DK_MLB_CLASSIC", "DK_NFL_CLASSIC", "DK_NFL_SHOWDOWN", "PRESETS"]


# Slots are listed scarce-first, which is the order construction fills them in.
# Catcher leads because it is the thinnest position on a typical slate; pitchers
# trail because they are the most expensive and benefit from being priced against
# whatever budget is left.
DK_MLB_CLASSIC = RosterSpec(
    positions=("P", "C", "1B", "2B", "3B", "SS", "OF"),
    slots=(
        Slot("C", ("C",)),
        Slot("SS", ("SS",)),
        Slot("2B", ("2B",)),
        Slot("3B", ("3B",)),
        Slot("1B", ("1B",)),
        Slot("OF", ("OF",), count=3),
        Slot("P", ("P",), count=2),
    ),
    salary_cap=50_000,
    # DraftKings imposes no floor, so neither does this. Leaving salary unspent is
    # usually a mistake, but that is a strategy and belongs with the caller's
    # other strategy — `replace(DK_MLB_CLASSIC, salary_floor=49_000)` sets one.
    salary_floor=0,
    groups=(
        GroupConstraint(key="team", max_count=6),
        # Five hitters plus that team's starting pitcher is a common and legal
        # shape, so the hitter cap has to be separate from the total cap rather
        # than one being derived from the other.
        GroupConstraint(
            key="team",
            max_count=5,
            slots=("C", "SS", "2B", "3B", "1B", "OF"),
        ),
    ),
)


DK_NFL_CLASSIC = RosterSpec(
    positions=("QB", "RB", "WR", "TE", "DST"),
    slots=(
        Slot("QB", ("QB",)),
        Slot("TE", ("TE",)),
        Slot("DST", ("DST",)),
        Slot("RB", ("RB",), count=2),
        Slot("WR", ("WR",), count=3),
        # The flex is why slots carry their own eligibility mask rather than a
        # single position name.
        Slot("FLEX", ("RB", "WR", "TE")),
    ),
    salary_cap=50_000,
    salary_floor=0,
    # No team cap: NFL classic does not impose one, and inventing a constraint the
    # contest does not have would silently shrink the search space.
    groups=(),
)


# A single-game contest, which is the format that makes slot multipliers load
# bearing rather than decorative: the captain scores 1.5x and costs 1.5x, so the
# same player is a different proposition depending on where they are rostered.
#
# Every position fills every slot here, which means slot *order* is doing all the
# work — the captain is listed first so the fill spends its premium on the best
# available player rather than on whoever is left.
DK_NFL_SHOWDOWN = RosterSpec(
    positions=("QB", "RB", "WR", "TE", "K", "DST"),
    slots=(
        Slot(
            "CPT",
            ("QB", "RB", "WR", "TE", "K", "DST"),
            score_multiplier=1.5,
            salary_multiplier=1.5,
        ),
        Slot("FLEX", ("QB", "RB", "WR", "TE", "K", "DST"), count=5),
    ),
    salary_cap=50_000,
    salary_floor=0,
    # DraftKings also requires players from both teams. That is expressible —
    # `GroupConstraint(key="team", min_distinct=2)` — but it is left off here for
    # the same reason as the game rule above: it needs a key on every record, and
    # a preset cannot know the caller supplied one. Add it when yours does.
    groups=(),
)


PRESETS: dict[str, RosterSpec] = {
    "dk_mlb_classic": DK_MLB_CLASSIC,
    "dk_nfl_classic": DK_NFL_CLASSIC,
    "dk_nfl_showdown": DK_NFL_SHOWDOWN,
}
"""Every preset, by name."""
