"""Recipes: the typed bundles are exactly the calls they stand for.

The load-bearing tests are parity: a recipe's `.build()` and `.select()` must be
byte-identical to the hand-written `build_lineups` / `select_portfolio` calls
with the same parameters, written out independently here. If they ever drift,
the recipe is documenting one thing and doing another.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from mlb_dfs_solver import build_lineups, recipes, score_lineups, select_portfolio, tail_line
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.select import Mode
from mlb_dfs_solver.spec import RosterSpec


@pytest.fixture
def universe(tiny_pool: PlayerPool, rng: np.random.Generator) -> np.ndarray:
    """Simulated scores for every player, correlated within a team."""
    n_sims = 300
    teams = tiny_pool.keys["team"]
    shock = rng.standard_normal((int(teams.max()) + 1, n_sims)) * 0.6
    noise = rng.standard_normal((len(tiny_pool), n_sims))
    return tiny_pool.projections[:, None] + tiny_pool.stddevs[:, None] * (noise + shock[teams])


# --- Build parity ---------------------------------------------------------


def test_cash_build_matches_the_handwritten_call(
    mlb_pool: PlayerPool, mlb_spec: RosterSpec
) -> None:
    r = recipes.cash(mlb_spec, seed=3)
    expected = build_lineups(
        mlb_pool,
        dataclasses.replace(mlb_spec, salary_floor=mlb_spec.salary_cap - 1_000),
        num_lineups=20,
        seed=3,
        value_weight=1.0,
    )
    np.testing.assert_array_equal(r.build(mlb_pool), expected)


def test_single_entry_build_matches_the_handwritten_call(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    r = recipes.single_entry(tiny_spec, seed=3)
    expected = build_lineups(
        tiny_pool,
        tiny_spec,
        num_lineups=300,
        seed=3,
        attempts_per_lineup=10,
        value_weight=1.0,
    )
    np.testing.assert_array_equal(r.build(tiny_pool), expected)


def test_gpp_build_matches_the_handwritten_call(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    r = recipes.gpp(tiny_spec, seed=3)
    expected = build_lineups(
        tiny_pool,
        tiny_spec,
        num_lineups=150,
        seed=3,
        attempts_per_lineup=10,
        diversity_weight=0.6,
    )
    np.testing.assert_array_equal(r.build(tiny_pool), expected)


def test_candidate_pool_build_matches_the_handwritten_call(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    r = recipes.candidate_pool(tiny_spec, seed=3, entries=500)
    expected = build_lineups(
        tiny_pool,
        tiny_spec,
        num_lineups=500,
        seed=3,
        attempts_per_lineup=5,
        diversity_weight=1.0,
    )
    np.testing.assert_array_equal(r.build(tiny_pool), expected)


def test_showdown_build_matches_the_handwritten_call(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    # The showdown factory works on any spec; the multipliers live on the spec,
    # not in the recipe, so the tiny fixture serves for parity.
    r = recipes.showdown(tiny_spec, seed=3)
    expected = build_lineups(tiny_pool, tiny_spec, num_lineups=150, seed=3, attempts_per_lineup=10)
    np.testing.assert_array_equal(r.build(tiny_pool), expected)


# --- Select parity --------------------------------------------------------


def test_gpp_select_matches_the_handwritten_call(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, universe: np.ndarray
) -> None:
    r = recipes.gpp(tiny_spec, seed=3, entries=30)
    lineups = r.build(tiny_pool)
    sim = score_lineups(tiny_pool, tiny_spec, lineups, universe)
    line = tail_line(sim, 0.99)
    expected = select_portfolio(sim, mode="gpp", line=line, n_select=30)
    np.testing.assert_array_equal(r.select(sim, line=line), expected)


def test_a_capped_recipe_passes_the_cap_through(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, universe: np.ndarray
) -> None:
    r = dataclasses.replace(recipes.gpp(tiny_spec, seed=3, entries=30), max_exposure=0.5)
    lineups = r.build(tiny_pool)
    sim = score_lineups(tiny_pool, tiny_spec, lineups, universe)
    line = tail_line(sim, 0.9)
    expected = select_portfolio(
        sim, mode="gpp", line=line, n_select=30, lineups=lineups, max_exposure=0.5
    )
    np.testing.assert_array_equal(r.select(sim, line=line, lineups=lineups), expected)


def test_a_locked_recipe_builds_the_handwritten_call(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    r = recipes.gpp(tiny_spec, seed=3, entries=30, locks=[0, 8])
    expected = build_lineups(
        tiny_pool,
        tiny_spec,
        num_lineups=30,
        seed=3,
        attempts_per_lineup=10,
        diversity_weight=0.6,
        locks=[0, 8],
    )
    np.testing.assert_array_equal(r.build(tiny_pool), expected)


def test_a_locked_and_capped_recipe_exempts_the_locks_from_the_cap(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, universe: np.ndarray
) -> None:
    # A locked player is in every candidate; a blanket selection cap would stop
    # the whole portfolio at the cap. The recipe caps everyone else instead.
    r = dataclasses.replace(recipes.gpp(tiny_spec, seed=3, entries=30, locks=[0]), max_exposure=0.5)
    lineups = r.build(tiny_pool)
    assert (lineups == 0).any(axis=1).all(), "lock must be in every candidate"
    sim = score_lineups(tiny_pool, tiny_spec, lineups, universe)
    line = tail_line(sim, 0.9)
    caps = {i: 0.5 for i in range(int(lineups.max()) + 1) if i != 0}
    expected = select_portfolio(
        sim, mode="gpp", line=line, n_select=30, lineups=lineups, max_exposure=caps
    )
    chosen = r.select(sim, line=line, lineups=lineups)
    np.testing.assert_array_equal(chosen, expected)


# --- The choices themselves ----------------------------------------------


def test_cash_sets_a_floor_without_touching_the_callers_spec(mlb_spec: RosterSpec) -> None:
    r = recipes.cash(mlb_spec, seed=1)
    assert r.spec.salary_floor == mlb_spec.salary_cap - 1_000
    assert mlb_spec.salary_floor == 0


def test_cash_accepts_an_explicit_floor_including_none_at_all(mlb_spec: RosterSpec) -> None:
    assert recipes.cash(mlb_spec, seed=1, salary_floor=47_500).spec.salary_floor == 47_500
    assert recipes.cash(mlb_spec, seed=1, salary_floor=0).spec.salary_floor == 0


def test_gpp_passes_the_spec_through_untouched(mlb_spec: RosterSpec) -> None:
    # Stacks and conflicts are the caller's strategy; a factory that quietly
    # added any would be wrong for anyone playing differently.
    assert recipes.gpp(mlb_spec, seed=1).spec is mlb_spec


def test_replace_is_the_override_idiom(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    r = dataclasses.replace(recipes.gpp(tiny_spec, seed=3), entries=5)
    assert len(r.build(tiny_pool)) <= 5


def test_a_build_only_recipe_refuses_to_select(tiny_spec: RosterSpec) -> None:
    r = recipes.candidate_pool(tiny_spec, seed=1)
    with pytest.raises(ValueError, match="no selection half"):
        r.select(np.zeros((4, 8), dtype=np.float32), line=0.0)


def test_select_portfolio_accepts_the_mode_enum(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, universe: np.ndarray
) -> None:
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=50, seed=2)
    sim = score_lineups(tiny_pool, tiny_spec, lineups, universe)
    line = tail_line(sim, 0.9)
    np.testing.assert_array_equal(
        select_portfolio(sim, mode=Mode.GPP, line=line, n_select=10),
        select_portfolio(sim, mode="gpp", line=line, n_select=10),
    )
