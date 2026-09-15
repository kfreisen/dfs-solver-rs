"""`conflict_pairs` takes `(n_pairs, 2)` and rejects the `(2, n_pairs)` layout `PlayerPool.conflict_pairs` returns.

Found 2026-09-15 from the mlb-dfs NHL harness: a forward-line stack plus a goalie-vs-opposing-skaters rule built
zero lineups. The harness passed `pool.conflict_pairs(spec)` (shape `(2, n)`) as `conflict_pairs`, and
`build_lineups` flattened and re-paired it with `reshape(-1, 2)`. The left row and the right row were zipped into
neighbouring indices: skater pairs on the same line were forbidden, so every line stack was unsatisfiable. The
Rust kernel was never wrong.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from dfs_solver import ConflictRule, GroupConstraint, build_lineups
from dfs_solver.backtest.legality import check_lineup
from dfs_solver.pool import PlayerPool
from dfs_solver.presets import DK_NHL_CLASSIC


def _records() -> list[dict[str, object]]:
    records = []
    teams = [f"T{k}" for k in range(8)]
    for gi in range(4):
        home, away = teams[2 * gi], teams[2 * gi + 1]
        for team, opp in ((home, away), (away, home)):
            for line in range(1, 5):
                for pos in ("C", "W", "W"):
                    records.append(
                        {
                            "name": f"{team}{pos}{line}{len(records)}",
                            "positions": (pos,),
                            "salary": 3000 + 600 * (4 - line),
                            "projection": 6.0 + (4 - line),
                            "team": team,
                            "opponent": opp,
                            "line": f"{team}_F{line}",
                        }
                    )
            for k in range(6):
                records.append(
                    {
                        "name": f"{team}D{k}",
                        "positions": ("D",),
                        "salary": 3500,
                        "projection": 5.0,
                        "team": team,
                        "opponent": opp,
                        "line": None,
                    }
                )
            records.append(
                {
                    "name": f"{team}G",
                    "positions": ("G",),
                    "salary": 8000,
                    "projection": 12.0,
                    "team": team,
                    "opponent": opp,
                    "line": None,
                }
            )
    return records


def _spec_and_pool():
    stack = GroupConstraint(key="line", min_stack=3, slots=("C", "W", "UTIL"))
    conflict = ConflictRule(
        left_key="opponent",
        right_key="team",
        left_positions=("G",),
        right_positions=("C", "W", "D"),
    )
    spec = replace(DK_NHL_CLASSIC, groups=(stack,), conflicts=(conflict,))
    pool = PlayerPool.from_records(_records(), spec, key_fields=("team", "opponent", "line"))
    return spec, pool


def test_line_stack_with_goalie_conflict_builds_and_holds() -> None:
    spec, pool = _spec_and_pool()
    lineups = build_lineups(pool, spec, num_lineups=20, seed=1)
    assert len(lineups) > 0
    pairs = {tuple(p) for p in pool.conflict_pairs(spec).T.tolist()}
    for lineup in lineups.tolist():
        assert not any((i, j) in pairs for i in lineup for j in lineup), lineup
        assert check_lineup(pool, spec, lineup) == []


def test_kernel_layout_is_rejected_with_the_fix_named() -> None:
    spec, pool = _spec_and_pool()
    with pytest.raises(ValueError, match=r"\(2, n_pairs\) layout"):
        build_lineups(pool, spec, num_lineups=5, seed=1, conflict_pairs=pool.conflict_pairs(spec))
    with pytest.raises(ValueError, match=r"\(2, n_pairs\) layout"):
        check_lineup(pool, spec, list(range(9)), conflict_pairs=pool.conflict_pairs(spec))


def test_transposed_kernel_pairs_are_accepted_and_change_nothing() -> None:
    spec, pool = _spec_and_pool()
    alone = build_lineups(pool, spec, num_lineups=20, seed=1)
    doubled = build_lineups(
        pool, spec, num_lineups=20, seed=1, conflict_pairs=pool.conflict_pairs(spec).T
    )
    assert np.array_equal(alone, doubled)


@pytest.mark.parametrize("bad", [[0, 1, 2, 3], [[0, 1, 2]], np.zeros((3, 3), dtype=np.int64)])
def test_other_shapes_are_rejected(bad) -> None:
    spec, pool = _spec_and_pool()
    with pytest.raises(ValueError, match="must be \\(n_pairs, 2\\)"):
        build_lineups(pool, spec, num_lineups=5, seed=1, conflict_pairs=bad)


def test_empty_pairs_are_inert() -> None:
    spec, pool = _spec_and_pool()
    base = build_lineups(pool, spec, num_lineups=10, seed=3)
    assert np.array_equal(
        base, build_lineups(pool, spec, num_lineups=10, seed=3, conflict_pairs=[])
    )
    assert np.array_equal(
        base,
        build_lineups(
            pool, spec, num_lineups=10, seed=3, conflict_pairs=np.empty((0, 2), dtype=np.int64)
        ),
    )


def test_line_stack_without_conflict_builds() -> None:
    stack = GroupConstraint(key="line", min_stack=3, slots=("C", "W", "UTIL"))
    spec = replace(DK_NHL_CLASSIC, groups=(stack,))
    pool = PlayerPool.from_records(_records(), spec, key_fields=("team", "opponent", "line"))
    assert len(build_lineups(pool, spec, num_lineups=20, seed=1)) > 0
