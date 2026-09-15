"""Checking a finished lineup against a specification.

[`build_lineups`][dfs_solver.greedy.build_lineups] only produces legal rosters,
but a backtest should not take that on trust — and lineups arrive from other
places: a file, a previous run, a hand edit. This module re-derives legality
from the [`RosterSpec`][dfs_solver.spec.RosterSpec] alone, in plain Python, so
a lineup the kernel produced and a lineup someone typed are held to exactly the
same rules. Nothing here knows what sport it is looking at.

The checks are the specification's fields, one each: roster size and distinct
players; a complete slot assignment; salary within the floor and cap, priced by
the slot each player fills; every
[`GroupConstraint`][dfs_solver.spec.GroupConstraint] — the cap, the distinct
minimum, and both stacks; and every
[`ConflictRule`][dfs_solver.spec.ConflictRule].

[`check_lineup`][dfs_solver.backtest.legality.check_lineup] reports every
violation rather than the first, because a lineup that is over the cap *and*
short a catcher is two mistakes, and fixing one to discover the other is a
tedious way to learn about the second.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from dfs_solver.greedy import assign_locks, index_pairs
from dfs_solver.spec import scaled_salary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dfs_solver.pool import PlayerPool
    from dfs_solver.spec import RosterSpec

__all__ = ["IllegalLineupError", "assert_legal", "check_lineup"]


class IllegalLineupError(ValueError):
    """A lineup breaks the specification.

    Attributes:
        violations: Every rule broken, one message each, in the order checked.
    """

    def __init__(self, violations: Sequence[str]) -> None:
        self.violations: tuple[str, ...] = tuple(violations)
        super().__init__("; ".join(self.violations))


def _slot_assignment(
    pool: PlayerPool, spec: RosterSpec, players: list[int]
) -> tuple[list[int] | None, str | None]:
    """Which slot group holds each player, in lineup order.

    The lineup's own column order is taken first: that is what
    [`build_lineups`][dfs_solver.greedy.build_lineups] emits, and with showdown
    multipliers it is the order that decides what the lineup cost. Only when a
    column holds a player ineligible for its slot is a matching searched, and
    the lineup is illegal only when no matching exists.
    """
    by_column = [group for group, slot in enumerate(spec.slots) for _ in range(slot.count)]
    masks = [spec.mask_for(slot.eligible) for slot in spec.slots]
    if all(int(pool.positions[p]) & masks[g] for p, g in zip(players, by_column, strict=True)):
        return by_column, None
    try:
        pairs = assign_locks(pool, spec, players)
    except ValueError as exc:
        return None, f"no complete slot assignment: {exc}"
    group_of = dict(pairs)
    return [group_of[p] for p in players], None


def _check_groups(
    pool: PlayerPool, spec: RosterSpec, players: list[int], groups: list[int]
) -> list[str]:
    """Every group constraint, over the players its slots count."""
    problems: list[str] = []
    for constraint in spec.groups:
        if constraint.key not in pool.keys:
            available = ", ".join(sorted(pool.keys)) or "none"
            msg = (
                f"the specification constrains {constraint.key!r} but the pool has no "
                f"such key column (available: {available})"
            )
            raise KeyError(msg)
        counted_groups = (
            None
            if constraint.slots is None
            else {i for i, slot in enumerate(spec.slots) if slot.name in constraint.slots}
        )
        values = pool.keys[constraint.key]
        counts: dict[int, int] = {}
        for player, group in zip(players, groups, strict=True):
            if counted_groups is not None and group not in counted_groups:
                continue
            value = int(values[player])
            # A negative key means "belongs to no group": never capped, and not
            # a distinct value either.
            if value < 0:
                continue
            counts[value] = counts.get(value, 0) + 1
        where = "" if constraint.slots is None else f" in slots {list(constraint.slots)}"
        ranked = sorted(counts.values(), reverse=True)

        if constraint.max_count is not None:
            over = {value: n for value, n in counts.items() if n > constraint.max_count}
            if over:
                problems.append(
                    f"{constraint.key!r}{where}: {over} exceeds max_count {constraint.max_count}"
                )
        if constraint.min_distinct and len(counts) < constraint.min_distinct:
            problems.append(
                f"{constraint.key!r}{where}: {len(counts)} distinct value(s), "
                f"min_distinct is {constraint.min_distinct}"
            )
        if constraint.min_stack:
            largest = ranked[0] if ranked else 0
            if largest < constraint.min_stack:
                problems.append(
                    f"{constraint.key!r}{where}: largest stack is {largest}, "
                    f"min_stack is {constraint.min_stack}"
                )
            elif constraint.secondary_min_stack:
                second = ranked[1] if len(ranked) > 1 else 0
                if second < constraint.secondary_min_stack:
                    problems.append(
                        f"{constraint.key!r}{where}: second stack is {second}, "
                        f"secondary_min_stack is {constraint.secondary_min_stack}"
                    )
    return problems


def _check_conflicts(pool: PlayerPool, spec: RosterSpec, players: list[int]) -> list[str]:
    """Every conflict rule, over every ordered pair of rostered players."""
    problems: list[str] = []
    for rule in spec.conflicts:
        for key in (rule.left_key, rule.right_key):
            if key not in pool.keys:
                available = ", ".join(sorted(pool.keys)) or "none"
                msg = (
                    f"a conflict rule reads {key!r} but the pool has no such key "
                    f"column (available: {available})"
                )
                raise KeyError(msg)
        left_mask = spec.mask_for(rule.left_positions) if rule.left_positions else None
        right_mask = spec.mask_for(rule.right_positions) if rule.right_positions else None
        left_values = pool.keys[rule.left_key]
        right_values = pool.keys[rule.right_key]
        for i in players:
            left = int(left_values[i])
            if left < 0 or (left_mask is not None and not int(pool.positions[i]) & left_mask):
                continue
            for j in players:
                if i == j:
                    continue
                if right_mask is not None and not int(pool.positions[j]) & right_mask:
                    continue
                if int(right_values[j]) == left:
                    problems.append(
                        f"players {i} and {j} conflict: {rule.left_key!r} of {i} matches "
                        f"{rule.right_key!r} of {j}"
                    )
    return problems


def check_lineup(
    pool: PlayerPool,
    spec: RosterSpec,
    lineup: np.ndarray | Sequence[int],
    *,
    conflict_pairs: Sequence[tuple[int, int]] | np.ndarray | None = None,
) -> list[str]:
    """Every way a lineup breaks the specification.

    Args:
        pool: The players the lineup indexes into.
        spec: The rules.
        lineup: Pool indices, one per roster position. Column order is taken as
            slot order when every column's player is eligible for that slot —
            which is how [`build_lineups`][dfs_solver.greedy.build_lineups]
            lays them out — and otherwise a matching is searched for.
        conflict_pairs: Extra `(i, j)` pairs forbidden from sharing a lineup,
            the same argument `build_lineups` takes, for a pool-specific rule
            that is not on the specification.

    Returns:
        One message per violation. Empty means the lineup is legal.

    Raises:
        KeyError: If the specification reads a key the pool does not carry.
            That is a wiring error, not a property of the lineup.
    """
    players = [int(i) for i in np.asarray(lineup).ravel()]
    problems: list[str] = []

    if len(players) != spec.roster_size:
        problems.append(f"{len(players)} players; the roster holds {spec.roster_size}")
    duplicated = sorted(p for p, n in Counter(players).items() if n > 1)
    if duplicated:
        problems.append(f"duplicate player(s): {duplicated}")
    out_of_range = sorted({p for p in players if not 0 <= p < len(pool)})
    if out_of_range:
        problems.append(f"player index(es) {out_of_range} outside a pool of {len(pool)}")
    if problems:
        # Nothing below is meaningful for a roster of the wrong shape, and the
        # slot matching would raise on the out-of-range index anyway.
        return problems

    groups, unassignable = _slot_assignment(pool, spec, players)
    if unassignable is not None:
        problems.append(unassignable)

    if groups is not None:
        salary = sum(
            scaled_salary(int(pool.salaries[p]), spec.slots[g].salary_multiplier)
            for p, g in zip(players, groups, strict=True)
        )
        if salary > spec.salary_cap:
            problems.append(f"salary {salary} exceeds cap {spec.salary_cap}")
        if salary < spec.salary_floor:
            problems.append(f"salary {salary} below floor {spec.salary_floor}")
        problems.extend(_check_groups(pool, spec, players, groups))
    else:
        # Without an assignment, slot-restricted groups cannot be evaluated;
        # the ones counting every slot still can, and so can the salary at
        # face value. Reporting those beats reporting only the matching.
        salary = int(pool.salaries[players].sum())
        if salary > spec.salary_cap:
            problems.append(f"salary {salary} exceeds cap {spec.salary_cap}")
        if salary < spec.salary_floor:
            problems.append(f"salary {salary} below floor {spec.salary_floor}")
        problems.extend(_check_groups(pool, spec, players, [-1] * len(players)))

    problems.extend(_check_conflicts(pool, spec, players))

    if conflict_pairs is not None:
        extra = index_pairs(conflict_pairs, None)
        rostered = set(players)
        for i, j in extra.tolist():
            if i in rostered and j in rostered and i != j:
                problems.append(f"players {i} and {j} are a forbidden pair")

    return problems


def assert_legal(
    pool: PlayerPool,
    spec: RosterSpec,
    lineup: np.ndarray | Sequence[int],
    *,
    conflict_pairs: Sequence[tuple[int, int]] | np.ndarray | None = None,
) -> None:
    """Raise unless a lineup satisfies the specification.

    The same checks as [`check_lineup`][dfs_solver.backtest.legality.check_lineup];
    this is the form for a loop that should stop.

    Raises:
        IllegalLineupError: Listing every violation.
        KeyError: If the specification reads a key the pool does not carry.
    """
    problems = check_lineup(pool, spec, lineup, conflict_pairs=conflict_pairs)
    if problems:
        raise IllegalLineupError(problems)
