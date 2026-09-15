"""Known defect (found 2026-09-15 from mlb-dfs NHL harness): a stack on a key other than the conflict's keys
plus a non-empty conflict graph builds nothing, although the same spec builds with the conflict graph empty
and a team-key stack builds with the conflicts.

Repro shape: NHL-like pool (C/W/D/G), GroupConstraint(key="line", min_stack=3, slots=C/W/UTIL) where `line`
is team+forward-line (null for D/G), ConflictRule(opponent->team, left G, right skaters). Expected: lineups.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from dfs_solver import ConflictRule, GroupConstraint, build_lineups
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


@pytest.mark.xfail(
    reason="stack on a non-conflict key + conflict pairs builds no lineups (open defect)",
    strict=True,
)
def test_line_stack_with_goalie_conflict_builds() -> None:
    stack = GroupConstraint(key="line", min_stack=3, slots=("C", "W", "UTIL"))
    conflict = ConflictRule(
        left_key="opponent",
        right_key="team",
        left_positions=("G",),
        right_positions=("C", "W", "D"),
    )
    spec = replace(DK_NHL_CLASSIC, groups=(stack,), conflicts=(conflict,))
    pool = PlayerPool.from_records(_records(), spec, key_fields=("team", "opponent", "line"))
    assert (
        len(
            build_lineups(
                pool, spec, num_lineups=20, seed=1, conflict_pairs=pool.conflict_pairs(spec)
            )
        )
        > 0
    )


def test_line_stack_without_conflict_builds() -> None:
    stack = GroupConstraint(key="line", min_stack=3, slots=("C", "W", "UTIL"))
    spec = replace(DK_NHL_CLASSIC, groups=(stack,))
    pool = PlayerPool.from_records(_records(), spec, key_fields=("team", "opponent", "line"))
    assert len(build_lineups(pool, spec, num_lineups=20, seed=1)) > 0
