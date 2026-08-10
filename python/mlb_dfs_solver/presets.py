"""Ready-made roster specifications for common contest formats.

These are data, not code paths — every one is an ordinary
[`RosterSpec`][mlb_dfs_solver.spec.RosterSpec] you can copy and modify. There is no
branch anywhere in this package that asks which sport it is looking at.

Operators change these rules, sometimes mid-season. Treat a preset as a starting
point that was correct when written, and check it against the contest you are
actually entering.
"""

from __future__ import annotations

from mlb_dfs_solver.spec import GroupConstraint, RosterSpec, Slot

__all__ = ["DK_MLB_CLASSIC", "DK_NFL_CLASSIC", "PRESETS"]


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
    # A floor is unusual — DraftKings does not impose one. It is here because
    # leaving salary unspent is almost always a mistake in a large-field contest,
    # and making it a constraint is cheaper than filtering afterwards. Set it to 0
    # to disable.
    salary_floor=49_000,
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


PRESETS: dict[str, RosterSpec] = {
    "dk_mlb_classic": DK_MLB_CLASSIC,
    "dk_nfl_classic": DK_NFL_CLASSIC,
}
"""Every preset, by name."""
