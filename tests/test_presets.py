"""Every preset builds legal lineups from a pool shaped for it, and the NHL classic slots match DraftKings'."""

from __future__ import annotations

from collections import Counter

import pytest
from dfs_solver import build_lineups
from dfs_solver.greedy import assign_locks
from dfs_solver.pool import PlayerPool
from dfs_solver.presets import DK_NHL_CLASSIC, PRESETS
from dfs_solver.spec import RosterSpec


def _records(spec: RosterSpec, per_position: int = 12) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for position in spec.positions:
        for k in range(per_position):
            records.append(
                {
                    "name": f"{position}{k}",
                    "positions": (position,),
                    "salary": 3000 + k * 450,
                    "projection": 4.0 + k * 1.2,
                    "stddev": 2.0 + (k % 3),
                    "ownership": (k % 5) / 10.0,
                    "team": f"T{k % 6}",
                }
            )
    return records


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_builds_legal_lineups(name: str) -> None:
    spec = PRESETS[name]
    pool = PlayerPool.from_records(_records(spec), spec)
    lineups = build_lineups(pool, spec, num_lineups=50, seed=7)
    assert len(lineups) > 0
    for lineup in lineups:
        players = [int(i) for i in lineup]
        assert len(set(players)) == spec.roster_size
        assign_locks(pool, spec, players)  # raises if no complete slot assignment exists
    assert (pool.salary_of(lineups, spec) <= spec.salary_cap).all()


def test_nhl_classic_is_two_c_three_w_two_d_one_g_one_util() -> None:
    counts = Counter()
    for slot in DK_NHL_CLASSIC.slots:
        counts[slot.name] += slot.count
    assert counts == {"C": 2, "W": 3, "D": 2, "G": 1, "UTIL": 1}
    assert DK_NHL_CLASSIC.roster_size == 9
    assert DK_NHL_CLASSIC.salary_cap == 50_000
    util = next(s for s in DK_NHL_CLASSIC.slots if s.name == "UTIL")
    assert set(util.eligible) == {"C", "W", "D"}
