"""The CP-SAT baseline's two diversity formulations do what they claim.

The baseline is benchmark code, but the published comparison rests on it, so its
one nontrivial mechanism — how a solver is made to produce *different* lineups —
is pinned here: no-good cuts give distinct lineups, and `max_shared` bounds how
many players any two share.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import pytest
from baselines.milp import solve_milp_ortools, solve_portfolio_ortools
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.spec import GroupConstraint, RosterSpec

pytest.importorskip("ortools")


def test_no_good_cuts_produce_distinct_lineups(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    found = solve_portfolio_ortools(tiny_pool, tiny_spec, num_lineups=6)
    assert len(found) == 6
    keys = {frozenset(lineup) for lineup in found}
    assert len(keys) == 6


def test_max_shared_bounds_every_pair(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    limit = 2
    found = solve_portfolio_ortools(tiny_pool, tiny_spec, num_lineups=5, max_shared=limit)
    assert len(found) >= 2, "need at least a pair to test the bound"
    for a, b in combinations(found, 2):
        assert len(set(a) & set(b)) <= limit


def test_the_solver_honors_a_stack_pair(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # Two distinct teams must supply two players each — the 4-2 shape the
    # kernel builds, formulated as two indicator families forced apart.
    spec = replace(
        tiny_spec, groups=(GroupConstraint(key="team", min_stack=2, secondary_min_stack=2),)
    )
    lineup = solve_milp_ortools(tiny_pool, spec)
    assert lineup is not None
    counts: dict[int, int] = {}
    for p in lineup:
        key = int(tiny_pool.keys["team"][p])
        counts[key] = counts.get(key, 0) + 1
    sizes = sorted(counts.values(), reverse=True)
    assert sizes[0] >= 2, lineup
    assert sizes[1] >= 2, lineup


def test_max_shared_binds_where_cuts_do_not(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # No-good cuts only require distinctness, so consecutive optima share all
    # but one player; the overlap constraint is what actually bounds the pair.
    def worst_pair(found: list[list[int]]) -> int:
        return max(len(set(a) & set(b)) for a, b in combinations(found, 2))

    cuts = solve_portfolio_ortools(tiny_pool, tiny_spec, num_lineups=5)
    overlap = solve_portfolio_ortools(tiny_pool, tiny_spec, num_lineups=5, max_shared=2)
    assert worst_pair(cuts) > 2
    assert worst_pair(overlap) <= 2
