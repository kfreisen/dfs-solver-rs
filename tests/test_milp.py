"""The CP-SAT baseline's two diversity formulations do what they claim.

The baseline is benchmark code, but the published comparison rests on it, so its
one nontrivial mechanism — how a solver is made to produce *different* lineups —
is pinned here: no-good cuts give distinct lineups, and `max_shared` bounds how
many players any two share.
"""

from __future__ import annotations

from itertools import combinations

import pytest
from baselines.milp import solve_portfolio_ortools
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.spec import RosterSpec

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


def test_max_shared_binds_where_cuts_do_not(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # No-good cuts only require distinctness, so consecutive optima share all
    # but one player; the overlap constraint is what actually bounds the pair.
    def worst_pair(found: list[list[int]]) -> int:
        return max(len(set(a) & set(b)) for a, b in combinations(found, 2))

    cuts = solve_portfolio_ortools(tiny_pool, tiny_spec, num_lineups=5)
    overlap = solve_portfolio_ortools(tiny_pool, tiny_spec, num_lineups=5, max_shared=2)
    assert worst_pair(cuts) > 2
    assert worst_pair(overlap) <= 2
