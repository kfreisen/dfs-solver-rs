"""Legality re-derived from the specification, against the DK MLB preset.

Every lineup the kernel builds must pass; every way of breaking a rule must be
named in the violations, and named together when several are broken at once.

The hand-built lineups below are worked out against the record generator: player
`X-k` is on team `TM{k % 10}`, in game `G{k % 10 // 2}` (so TM2 and TM3 share a
game), faces `TM{k % 10 ^ 1}`, and costs `2500 + (k % 12) * 750`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from dfs_solver import build_lineups
from dfs_solver.backtest.legality import IllegalLineupError, assert_legal, check_lineup
from dfs_solver.pool import PlayerPool
from dfs_solver.presets import DK_MLB_CLASSIC, DK_MLB_SHOWDOWN
from dfs_solver.spec import ConflictRule, GroupConstraint, RosterSpec, Slot

HITTER_SLOTS = ("C", "SS", "2B", "3B", "1B", "OF")
HITTER_POSITIONS = ("C", "1B", "2B", "3B", "SS", "OF")


def records_with_games() -> list[dict[str, object]]:
    """The conftest MLB slate, with each team paired into a game and given an opponent."""
    records: list[dict[str, object]] = []
    for position in ("P", "C", "1B", "2B", "3B", "SS", "OF"):
        count = 30 if position == "OF" else 10
        for k in range(count):
            team = k % 10
            records.append(
                {
                    "name": f"{position}-{k}",
                    "positions": (position,),
                    "salary": 2500 + (k % 12) * 750,
                    "projection": 4.0 + (k % 12) * 1.2,
                    "team": f"TM{team}",
                    "opponent": f"TM{team ^ 1}",
                    "game": f"G{team // 2}",
                }
            )
    return records


GAMES = replace(
    DK_MLB_CLASSIC,
    groups=(*DK_MLB_CLASSIC.groups, GroupConstraint(key="game", min_distinct=2)),
)
RULED = replace(
    GAMES,
    conflicts=(ConflictRule(left_key="opponent", right_key="team", left_positions=("P",)),),
)


@pytest.fixture
def pool() -> PlayerPool:
    return PlayerPool.from_records(records_with_games(), RULED)


def idx(pool: PlayerPool, *names: str) -> list[int]:
    assert pool.names is not None
    return [pool.names.index(name) for name in names]


def legal_lineup(pool: PlayerPool) -> list[int]:
    """Four TM3 bats, four singletons, two pitchers facing nobody rostered.

    Columns in `DK_MLB_CLASSIC` slot order; 47,500 in salary; teams from five
    games; no pitcher faces a rostered hitter or the other pitcher (the rule
    with `right_positions=None` forbids opposing pitchers too).
    """
    return idx(pool, "C-3", "SS-3", "2B-3", "3B-3", "1B-0", "OF-15", "OF-16", "OF-27", "P-0", "P-8")


# --- what the kernel builds passes -----------------------------------------


def test_every_built_lineup_is_legal(pool: PlayerPool) -> None:
    lineups = build_lineups(pool, GAMES, num_lineups=200, seed=3)
    assert len(lineups) == 200
    for lineup in lineups:
        assert check_lineup(pool, GAMES, lineup) == []
        assert_legal(pool, GAMES, lineup)


def test_built_lineups_under_a_conflict_rule_are_legal(pool: PlayerPool) -> None:
    # The pitcher-versus-hitter rule; the yield is low on this synthetic slate,
    # and every lineup that does come back must satisfy it.
    spec = replace(
        GAMES,
        conflicts=(
            ConflictRule(
                "opponent", "team", left_positions=("P",), right_positions=HITTER_POSITIONS
            ),
        ),
    )
    lineups = build_lineups(pool, spec, num_lineups=50, seed=3, attempts_per_lineup=20)
    assert len(lineups) > 0
    for lineup in lineups:
        assert check_lineup(pool, spec, lineup) == []


def test_built_showdown_lineups_are_legal_under_multipliers(mlb_pool: PlayerPool) -> None:
    lineups = build_lineups(mlb_pool, DK_MLB_SHOWDOWN, num_lineups=100, seed=1)
    assert len(lineups) > 0
    for lineup in lineups:
        assert check_lineup(mlb_pool, DK_MLB_SHOWDOWN, lineup) == []


def test_the_hand_built_lineup_is_legal(pool: PlayerPool) -> None:
    assert check_lineup(pool, RULED, legal_lineup(pool)) == []


# --- shape ------------------------------------------------------------------


def test_wrong_roster_size_and_duplicates_are_reported_together(pool: PlayerPool) -> None:
    problems = check_lineup(pool, RULED, [0, 0, 1])
    assert any("3 players; the roster holds 10" in p for p in problems)
    assert any("duplicate player(s): [0]" in p for p in problems)


def test_out_of_range_index_is_reported(pool: PlayerPool) -> None:
    problems = check_lineup(pool, RULED, [*range(9), len(pool)])
    assert problems == [f"player index(es) [{len(pool)}] outside a pool of {len(pool)}"]


# --- slots --------------------------------------------------------------------


def test_no_slot_assignment_is_a_violation(pool: PlayerPool) -> None:
    # Two second basemen and no shortstop: no matching exists.
    lineup = legal_lineup(pool)
    lineup[1] = idx(pool, "2B-4")[0]
    problems = check_lineup(pool, RULED, lineup)
    assert any(p.startswith("no complete slot assignment") for p in problems)


def test_columns_out_of_slot_order_are_matched_rather_than_rejected(pool: PlayerPool) -> None:
    # Pitchers first, then the bats: eligible under some assignment, so legal.
    lineup = legal_lineup(pool)
    assert check_lineup(pool, RULED, lineup[8:] + lineup[:8]) == []


def test_slot_restricted_groups_use_the_matched_assignment(pool: PlayerPool) -> None:
    # Six from TM3 is legal only because one of them is the pitcher. With the
    # pitcher listed in a hitter's column the matching must put P-3 back in a
    # pitcher slot, so the hitter cap counts five, not six.
    lineup = idx(
        pool, "C-3", "SS-3", "2B-3", "3B-3", "1B-3", "OF-5", "OF-16", "OF-27", "P-3", "P-0"
    )
    assert check_lineup(pool, RULED, lineup) == []
    assert check_lineup(pool, RULED, lineup[8:] + lineup[:8]) == []


# --- salary -------------------------------------------------------------------


def test_over_cap_is_reported(pool: PlayerPool) -> None:
    lineup = idx(
        pool, "C-9", "SS-9", "2B-9", "3B-9", "1B-9", "OF-11", "OF-23", "OF-10", "P-9", "P-8"
    )
    problems = check_lineup(pool, RULED, lineup)
    assert any("exceeds cap 50000" in p for p in problems)


def test_under_floor_is_reported(pool: PlayerPool) -> None:
    floored = replace(RULED, salary_floor=49_000)
    problems = check_lineup(pool, floored, legal_lineup(pool))
    assert problems == ["salary 47500 below floor 49000"]


def test_showdown_captain_is_priced_at_the_multiplier() -> None:
    # Six players at 8,000 fit a 50,000 cap at face value (48,000) but not with
    # the captain at 1.5x (52,000). The kernel prices it that way; so must this.
    records = [
        {"name": f"p{i}", "positions": ("OF",), "salary": 8_000, "projection": 1.0}
        for i in range(6)
    ]
    pool = PlayerPool.from_records(records, DK_MLB_SHOWDOWN)
    problems = check_lineup(pool, DK_MLB_SHOWDOWN, list(range(6)))
    assert problems == ["salary 52000 exceeds cap 50000"]


def test_salary_is_still_checked_when_no_assignment_exists(pool: PlayerPool) -> None:
    # Ten pitchers: no assignment, and over the cap at face value. Both named.
    lineup = idx(pool, *[f"P-{k}" for k in range(10)])
    problems = check_lineup(pool, replace(RULED, salary_cap=30_000), lineup)
    assert any("no complete slot assignment" in p for p in problems)
    assert any("exceeds cap 30000" in p for p in problems)


# --- groups -------------------------------------------------------------------


def nine_from_one_team(pool: PlayerPool) -> list[int]:
    """Eight TM3 bats and TM3's pitcher: over both team caps, nothing else wrong."""
    return idx(pool, "C-3", "SS-3", "2B-3", "3B-3", "1B-3", "OF-3", "OF-13", "OF-23", "P-3", "P-0")


def test_team_caps_are_reported_with_the_offending_count(pool: PlayerPool) -> None:
    lineup = nine_from_one_team(pool)
    team = int(pool.keys["team"][lineup[0]])
    assert check_lineup(pool, RULED, lineup) == [
        f"'team': {{{team}: 9}} exceeds max_count 6",
        f"'team' in slots ['C', 'SS', '2B', '3B', '1B', 'OF']: {{{team}: 8}} exceeds max_count 5",
    ]


def test_min_distinct_is_checked(pool: PlayerPool) -> None:
    # TM2 and TM3 share a game; five TM3 bats, three TM2 bats, both pitchers.
    # Under the conflict rule each pitcher also faces the other team's bats.
    lineup = idx(
        pool, "C-3", "SS-3", "2B-3", "3B-3", "1B-3", "OF-2", "OF-12", "OF-22", "P-3", "P-2"
    )
    assert check_lineup(pool, replace(RULED, conflicts=()), lineup) == [
        "'game': 1 distinct value(s), min_distinct is 2"
    ]
    with_conflicts = check_lineup(pool, RULED, lineup)
    assert with_conflicts[0] == "'game': 1 distinct value(s), min_distinct is 2"
    # Each pitcher faces the other team's bats, and the two pitchers face each
    # other — reported from both sides.
    assert sum("conflict" in p for p in with_conflicts) == 10


def test_min_stack_and_secondary_are_checked(pool: PlayerPool) -> None:
    stacked = replace(
        DK_MLB_CLASSIC,
        groups=(
            *DK_MLB_CLASSIC.groups,
            GroupConstraint(key="team", min_stack=4, secondary_min_stack=2, slots=HITTER_SLOTS),
        ),
    )
    where = "'team' in slots ['C', 'SS', '2B', '3B', '1B', 'OF']"
    # Every hitter from a different team: largest stack is one.
    spread = idx(
        pool, "C-0", "SS-1", "2B-2", "3B-3", "1B-4", "OF-5", "OF-16", "OF-27", "P-0", "P-1"
    )
    assert check_lineup(pool, stacked, spread) == [f"{where}: largest stack is 1, min_stack is 4"]
    # Four from TM3 and singletons elsewhere: primary met, secondary not.
    assert check_lineup(pool, stacked, legal_lineup(pool)) == [
        f"{where}: second stack is 1, secondary_min_stack is 2"
    ]
    # Four from TM3 and two from TM5: the 4-2.
    four_two = legal_lineup(pool)
    four_two[7] = idx(pool, "OF-25")[0]
    assert check_lineup(pool, stacked, four_two) == []


def test_a_negative_key_belongs_to_no_group() -> None:
    spec = RosterSpec(
        positions=("X",),
        slots=(Slot("X", ("X",), count=3),),
        salary_cap=100,
        groups=(GroupConstraint(key="team", max_count=1, min_distinct=2),),
    )
    records = [
        {"positions": ("X",), "salary": 1, "projection": 1.0, "team": None} for _ in range(3)
    ]
    pool = PlayerPool.from_records(records, spec)
    # Uncapped — but three nobodies are zero distinct teams, not three.
    assert check_lineup(pool, spec, [0, 1, 2]) == ["'team': 0 distinct value(s), min_distinct is 2"]


def test_a_missing_key_column_is_a_wiring_error(mlb_pool: PlayerPool) -> None:
    with pytest.raises(KeyError, match="constrains 'game' but the pool has no such key"):
        check_lineup(mlb_pool, RULED, list(range(10)))


# --- conflicts ----------------------------------------------------------------


def test_pitcher_against_own_hitter_is_a_conflict(pool: PlayerPool) -> None:
    # P-3 (TM3) faces TM2; OF-2 is the one TM2 bat. P-8 faces TM9: nobody.
    lineup = idx(
        pool, "C-0", "SS-1", "2B-4", "3B-5", "1B-6", "OF-2", "OF-13", "OF-24", "P-3", "P-8"
    )
    p3, of2 = idx(pool, "P-3", "OF-2")
    assert check_lineup(pool, RULED, lineup) == [
        f"players {p3} and {of2} conflict: 'opponent' of {p3} matches 'team' of {of2}"
    ]


def test_conflict_rule_needs_its_keys(mlb_pool: PlayerPool) -> None:
    spec = replace(DK_MLB_CLASSIC, conflicts=RULED.conflicts)
    with pytest.raises(KeyError, match="reads 'opponent' but the pool has no such key"):
        check_lineup(mlb_pool, spec, list(range(10)))


def test_extra_conflict_pairs_are_checked(pool: PlayerPool) -> None:
    lineup = legal_lineup(pool)
    a, b = lineup[0], lineup[1]
    problems = check_lineup(pool, RULED, lineup, conflict_pairs=[(b, a), (a, a), (a, 999)])
    assert problems == [f"players {b} and {a} are a forbidden pair"]


# --- assert_legal ---------------------------------------------------------------


def test_assert_legal_lists_every_violation(pool: PlayerPool) -> None:
    with pytest.raises(IllegalLineupError) as caught:
        assert_legal(pool, RULED, np.asarray(nine_from_one_team(pool)))
    assert len(caught.value.violations) == 2
    assert "max_count 6" in str(caught.value)
    assert "max_count 5" in str(caught.value)
    assert isinstance(caught.value, ValueError)
    assert_legal(pool, RULED, legal_lineup(pool))


def test_floor_is_still_checked_when_no_assignment_exists(pool: PlayerPool) -> None:
    lineup = idx(pool, *[f"P-{k}" for k in range(10)])
    # Ten pitchers cost 58,750; the floor must sit above that and under the cap.
    spec = replace(RULED, salary_cap=70_000, salary_floor=60_000)
    problems = check_lineup(pool, spec, lineup)
    assert any("no complete slot assignment" in p for p in problems)
    assert "salary 58750 below floor 60000" in problems


def test_a_satisfied_primary_stack_without_a_secondary_is_legal(pool: PlayerPool) -> None:
    stacked = replace(
        DK_MLB_CLASSIC,
        groups=(
            *DK_MLB_CLASSIC.groups,
            GroupConstraint(key="team", min_stack=4, slots=HITTER_SLOTS),
        ),
    )
    assert check_lineup(pool, stacked, legal_lineup(pool)) == []
