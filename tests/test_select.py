"""Portfolio selection: scoring against a universe, and the contest modes.

The algorithm itself is covered by the Rust unit tests, including the claim that
lazy evaluation returns exactly what the naive loop does. What is tested here is
the Python surface — argument validation, the encoding, and that the two contest
modes really do pull in different directions rather than being one objective with
a knob on it.
"""

from __future__ import annotations

import numpy as np
import pytest
from mlb_dfs_solver import (
    build_lineups,
    field_line,
    portfolio_value,
    score_lineups,
    select_portfolio,
    tail_line,
)
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.spec import RosterSpec, Slot


@pytest.fixture
def universe(tiny_pool: PlayerPool, rng: np.random.Generator) -> np.ndarray:
    """Simulated scores for every player, correlated within a team."""
    n_sims = 400
    teams = tiny_pool.keys["team"]
    shock = rng.standard_normal((int(teams.max()) + 1, n_sims)) * 0.6
    noise = rng.standard_normal((len(tiny_pool), n_sims))
    return tiny_pool.projections[:, None] + tiny_pool.stddevs[:, None] * (noise + shock[teams])


@pytest.fixture
def candidates(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> np.ndarray:
    return build_lineups(tiny_pool, tiny_spec, num_lineups=400, seed=5)


# --- Scoring -------------------------------------------------------------


def test_scoring_sums_players_across_outcomes(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    universe = np.arange(len(tiny_pool) * 3, dtype=np.float64).reshape(len(tiny_pool), 3)
    lineups = np.array([[0, 8, 9, 16]])
    scored = score_lineups(tiny_pool, tiny_spec, lineups, universe)
    assert scored.shape == (1, 3)
    np.testing.assert_allclose(scored[0], universe[[0, 8, 9, 16]].sum(axis=0))


def test_scoring_applies_slot_score_multipliers(tiny_pool: PlayerPool) -> None:
    # A captain is worth 1.5x when the portfolio is scored, or selection would
    # rank showdown lineups on numbers nobody is actually paid on.
    spec = RosterSpec(
        positions=("P", "C", "OF"),
        slots=(Slot("CPT", ("P", "C", "OF"), score_multiplier=1.5), Slot("FLEX", ("P", "C", "OF"))),
        salary_cap=30_000,
    )
    universe = np.ones((len(tiny_pool), 2), dtype=np.float64)
    scored = score_lineups(tiny_pool, spec, np.array([[0, 1]]), universe)
    np.testing.assert_allclose(scored[0], [2.5, 2.5])


def test_scoring_rejects_a_universe_for_another_pool(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="they must be parallel"):
        score_lineups(tiny_pool, tiny_spec, candidates, np.zeros((3, 10)))


def test_scoring_rejects_a_one_dimensional_universe(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        score_lineups(tiny_pool, tiny_spec, candidates, np.zeros(len(tiny_pool)))


def test_scoring_rejects_lineups_of_the_wrong_width(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, universe: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="to match the specification"):
        score_lineups(tiny_pool, tiny_spec, np.array([[0, 1]]), universe)


def test_scoring_rejects_a_player_outside_the_pool(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, universe: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="but the pool has"):
        score_lineups(tiny_pool, tiny_spec, np.array([[0, 1, 2, 9999]]), universe)


# --- The contest modes ---------------------------------------------------


def test_the_modes_optimize_different_things(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    """Cash wins on cash rate; tournament wins on the chance of winning.

    The load-bearing test for having two modes at all. If either lost on its own
    metric the mode would be mislabelled, and if they tied they would have
    collapsed into one objective.
    """
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    cash_line = tail_line(sim, 0.5)
    win_line = tail_line(sim, 0.99)

    cash_idx = select_portfolio(sim, mode="cash", line=cash_line, n_select=40)
    gpp_idx = select_portfolio(sim, mode="gpp", line=win_line, n_select=40)

    # Truncated to a common length. Tournament mode stops once no candidate
    # covers a new outcome, so it routinely returns fewer entries than asked
    # for — and a mean over 40 entries against a mean over 4 compares portfolio
    # sizes rather than objectives.
    n = min(len(cash_idx), len(gpp_idx))
    assert n > 1, "nothing to compare"
    cash = sim[cash_idx[:n]]
    gpp = sim[gpp_idx[:n]]

    # Cash: what fraction of entry-outcomes clear the line.
    assert float((cash >= cash_line).mean()) > float((gpp >= cash_line).mean())
    # Tournament: how often *some* entry reaches a winning score.
    assert float((gpp.max(axis=0) >= win_line).mean()) > float(
        (cash.max(axis=0) >= win_line).mean()
    )


def test_cash_mode_keeps_near_duplicates(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, universe: np.ndarray
) -> None:
    # Two identical entries both cash, so cash mode must not fade the second.
    # This is the opposite of what a portfolio objective wants, which is exactly
    # why cash is a separate mode rather than a weight.
    sim = np.repeat(np.array([[10.0, 10.0, 0.0, 0.0]], dtype=np.float32), 4, axis=0)
    chosen = select_portfolio(sim, mode="cash", line=5.0, n_select=3)
    assert len(chosen) == 3


def test_tournament_mode_drops_duplicates(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # The same pool under the tournament objective: a second copy covers no new
    # outcome, so one entry is the whole portfolio.
    sim = np.repeat(np.array([[10.0, 10.0, 0.0, 0.0]], dtype=np.float32), 4, axis=0)
    chosen = select_portfolio(sim, mode="gpp", line=5.0, n_select=3)
    assert len(chosen) == 1


def test_tournament_mode_prefers_covering_a_new_outcome(tiny_pool: PlayerPool) -> None:
    # Candidate 2 reaches the line in an outcome no one else does, so it is taken
    # second even though it is the lowest-scoring lineup in the pool.
    sim = np.array(
        [
            [20.0, 20.0, 0.0, 0.0],
            [30.0, 30.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 15.0],
        ],
        dtype=np.float32,
    )
    chosen = select_portfolio(sim, mode="gpp", line=10.0, n_select=2)
    assert chosen.tolist()[1] == 2


def test_tournament_mode_is_indifferent_to_margin(tiny_pool: PlayerPool) -> None:
    """Clearing the line by 20 counts the same as clearing it by 1.

    A real consequence of the coverage objective, and worth pinning so it is a
    decision rather than a surprise: candidates 0 and 1 cover exactly the same
    outcomes, so despite candidate 1 scoring half again as much they tie, and the
    tie breaks on index. Use `"excess"` when the payout keeps climbing with rank
    rather than paying a lump for reaching the line.
    """
    sim = np.array(
        [[20.0, 20.0, 0.0, 0.0], [30.0, 30.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    assert select_portfolio(sim, mode="gpp", line=10.0, n_select=1).tolist() == [0]
    # The excess objective does distinguish them, and takes the bigger score.
    assert select_portfolio(sim, mode="excess", line=10.0, n_select=1).tolist() == [1]


def test_selection_is_ordered_by_contribution(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    # A shorter portfolio must be a prefix of a longer one, so a caller can
    # truncate instead of re-running.
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    long = select_portfolio(sim, mode="excess", line=0.0, n_select=20)
    short = select_portfolio(sim, mode="excess", line=0.0, n_select=5)
    np.testing.assert_array_equal(long[:5], short)


def test_selection_is_deterministic(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    a = select_portfolio(sim, mode="gpp", line=tail_line(sim, 0.9), n_select=25)
    b = select_portfolio(sim, mode="gpp", line=tail_line(sim, 0.9), n_select=25)
    np.testing.assert_array_equal(a, b)


def test_selection_beats_taking_the_top_by_mean(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    """The whole justification for a portfolio objective.

    Ranking candidates by their own mean is the obvious thing to do and is worse
    at the thing that pays, because it buys the same outcomes repeatedly.
    """
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    # Well under the candidate count, or both approaches take everything and the
    # comparison is between two identical sets.
    take = max(2, len(sim) // 4)
    chosen = select_portfolio(sim, mode="excess", line=0.0, n_select=take)
    naive = np.argsort(-sim.mean(axis=1))[:take]
    assert portfolio_value(sim, chosen) > portfolio_value(sim, naive)


# --- Exposure caps -------------------------------------------------------


def test_exposure_caps_bound_appearances(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    n_select = 40
    chosen = select_portfolio(
        sim,
        mode="excess",
        line=0.0,
        n_select=n_select,
        lineups=candidates,
        max_exposure=0.25,
        n_players=len(tiny_pool),
    )
    entered = candidates[chosen]
    for player in range(len(tiny_pool)):
        assert int((entered == player).sum()) <= int(0.25 * n_select)


def test_exposure_caps_need_the_lineups(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    with pytest.raises(ValueError, match="needs `lineups`"):
        select_portfolio(sim, mode="excess", line=0.0, max_exposure=0.5)


def test_exposure_caps_reject_lineups_of_the_wrong_length(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    with pytest.raises(ValueError, match="but sim_scores describes"):
        select_portfolio(sim, mode="excess", line=0.0, lineups=candidates[:3], max_exposure=0.5)


def test_an_out_of_range_exposure_fraction_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    with pytest.raises(ValueError, match="expected a fraction"):
        select_portfolio(sim, mode="excess", line=0.0, lineups=candidates, max_exposure={0: 1.5})


def test_exposure_naming_a_missing_player_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    with pytest.raises(ValueError, match="but the pool has"):
        select_portfolio(
            sim,
            mode="excess",
            line=0.0,
            lineups=candidates,
            max_exposure={9999: 0.5},
            n_players=len(tiny_pool),
        )


# --- Arguments -----------------------------------------------------------


def test_an_unknown_mode_is_rejected() -> None:
    # No default mode: the two are different objectives, and silently picking one
    # would optimize confidently for the contest the caller is not playing.
    with pytest.raises(ValueError, match="unknown mode"):
        select_portfolio(np.ones((2, 2), dtype=np.float32), mode="tournament", line=0.0)


def test_a_negative_selection_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        select_portfolio(np.ones((2, 2), dtype=np.float32), mode="gpp", line=0.0, n_select=-1)


def test_a_one_dimensional_score_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        select_portfolio(np.ones(4, dtype=np.float32), mode="gpp", line=0.0)


def test_selecting_zero_returns_nothing() -> None:
    chosen = select_portfolio(np.ones((4, 2), dtype=np.float32), mode="gpp", line=0.0, n_select=0)
    assert len(chosen) == 0


def test_tail_line_reads_a_quantile() -> None:
    scores = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert tail_line(scores, 0.0) == 1.0
    assert tail_line(scores, 1.0) == 4.0
    assert tail_line(np.empty((0, 0))) == 0.0


def test_tail_line_rejects_a_quantile_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        tail_line(np.ones((2, 2)), 1.5)


def test_portfolio_value_is_the_mean_of_the_best_entry() -> None:
    sim = np.array([[10.0, 0.0], [0.0, 40.0]], dtype=np.float32)
    # Per-outcome max is (10, 40), mean 25.
    assert portfolio_value(sim, np.array([0, 1])) == 25.0
    assert portfolio_value(sim, np.array([0])) == 5.0


def test_min_gain_stops_early() -> None:
    # Three candidates, the third adding almost nothing over the first two.
    sim = np.array(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.01]],
        dtype=np.float32,
    )
    assert len(select_portfolio(sim, mode="excess", line=0.0, n_select=3, min_gain=0.1)) == 2


# --- The line ------------------------------------------------------------


def test_field_line_is_per_outcome(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    sim = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=np.float32)
    np.testing.assert_allclose(field_line(sim, 0.5), [3.0, 30.0])
    assert field_line(sim, 0.5).shape == (2,)


def test_field_line_tracks_the_slate_not_the_lineup(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    """The bar has to move with the outcome, or it measures the wrong thing.

    On a realistic slate the field's score varies far more between outcomes than
    lineups vary within one, so a fixed bar mostly asks "was this a high-scoring
    world?" — which is no edge, since every rival lineup scored more there too.
    """
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    per_outcome = field_line(sim, 0.5)
    spread = float(per_outcome.max() - per_outcome.min())
    within = float(np.median(sim.std(axis=0)))
    assert spread > within, (
        f"outcome-to-outcome spread {spread:.1f} should exceed the within-outcome "
        f"lineup spread {within:.1f}, or a constant line would be harmless"
    )


def test_a_per_outcome_line_is_accepted(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    chosen = select_portfolio(sim, mode="gpp", line=field_line(sim, 0.9), n_select=20)
    assert len(chosen) > 0


def test_a_scalar_line_still_broadcasts(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    # A constant is right when the bar genuinely does not move, so it stays legal.
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    flat = select_portfolio(sim, mode="gpp", line=50.0, n_select=10)
    vector = select_portfolio(sim, mode="gpp", line=np.full(sim.shape[1], 50.0), n_select=10)
    np.testing.assert_array_equal(flat, vector)


def test_a_line_of_the_wrong_length_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, candidates: np.ndarray, universe: np.ndarray
) -> None:
    sim = score_lineups(tiny_pool, tiny_spec, candidates, universe)
    with pytest.raises(ValueError, match="expected one per outcome"):
        select_portfolio(sim, mode="gpp", line=np.zeros(3), n_select=5)


def test_field_line_rejects_a_bad_quantile() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        field_line(np.ones((2, 2)), 1.5)


def test_field_line_rejects_a_one_dimensional_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        field_line(np.ones(4))


def test_field_line_of_an_empty_matrix_is_empty() -> None:
    assert field_line(np.empty((0, 0))).size == 0
