"""Ranking into a realized field, and the tie rule.

The tie rule is the thing under test. Every case here is checked against the
arithmetic done by hand in the comment beside it, because "the code agrees with
itself" is not evidence that it agrees with how an operator settles a tie.
"""

from __future__ import annotations

import numpy as np
import pytest
from dfs_solver.backtest.contest import Contest, PayoutTier
from dfs_solver.backtest.ranking import (
    FieldScores,
    is_void,
    realized_ranks_and_payouts,
    tie_block_payouts,
    tie_blocks,
)
from hypothesis import given, settings
from hypothesis import strategies as st

# Five-entry contest paying 10 / 4 / 4 / 0 / 0.
CONTEST = Contest(
    contest_id=1,
    entry_fee=1.0,
    field_size=5,
    max_entries_per_user=5,
    tiers=(PayoutTier(1, 1, 10.0), PayoutTier(2, 3, 4.0)),
)
FIELD = FieldScores.of(np.array([100.0, 90.0, 90.0, 80.0, 70.0]))


# --- is_void ---------------------------------------------------------------


def test_void_is_every_score_at_or_below_zero() -> None:
    assert is_void(np.zeros(5))
    assert is_void(np.array([0.0, -1.0]))
    assert is_void(np.array([]))
    assert not is_void(np.array([0.0, 0.0, 0.5]))
    assert FieldScores.of(np.zeros(3)).void


# --- FieldScores -----------------------------------------------------------


def test_field_scores_of_sorts() -> None:
    field = FieldScores.of(np.array([3.0, 1.0, 2.0]))
    assert field.scores.tolist() == [1.0, 2.0, 3.0]
    assert len(field) == 3
    assert not field.scores.flags.writeable


def test_field_scores_rejects_unsorted_two_dimensional_and_nan() -> None:
    with pytest.raises(ValueError, match="sorted ascending"):
        FieldScores(np.array([2.0, 1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        FieldScores(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="NaN"):
        FieldScores(np.array([1.0, np.nan]))


def test_field_scores_counts() -> None:
    assert FIELD.count_above(np.array([90.0, 100.0, 50.0])).tolist() == [1, 0, 5]
    assert FIELD.count_tied(np.array([90.0, 100.0, 50.0])).tolist() == [2, 1, 0]
    np.testing.assert_allclose(FIELD.cdf(np.array([90.0, 69.0, 100.0])), [0.8, 0.0, 1.0])


def test_field_scores_cdf_of_an_empty_field_is_zero() -> None:
    empty = FieldScores(np.array([]))
    assert empty.cdf(np.array([1.0, 2.0])).tolist() == [0.0, 0.0]


# --- tie_blocks -------------------------------------------------------------


def test_untied_entry_ranks_by_field_entries_above_it() -> None:
    start, size = tie_blocks(np.array([95.0]), FIELD)
    assert start.tolist() == [2]
    assert size.tolist() == [1]


def test_our_entry_joins_the_field_entries_it_ties_with() -> None:
    # Two field entries at 90 plus ours: three entries share ranks 2..4.
    start, size = tie_blocks(np.array([90.0]), FIELD)
    assert start.tolist() == [2]
    assert size.tolist() == [3]


def test_our_own_entries_tie_with_each_other() -> None:
    # Two of ours at 100 join the one field entry at 100: ranks 1..3 shared.
    start, size = tie_blocks(np.array([100.0, 100.0]), FIELD)
    assert start.tolist() == [1, 1]
    assert size.tolist() == [3, 3]


def test_a_higher_own_entry_pushes_the_next_one_down() -> None:
    # 101 is alone at rank 1; 95 is below it and the field's 100, so rank 3.
    start, size = tie_blocks(np.array([95.0, 101.0]), FIELD)
    assert start.tolist() == [3, 1]
    assert size.tolist() == [1, 1]


def test_tie_blocks_accepts_a_raw_array_and_scalar() -> None:
    start, size = tie_blocks(95.0, np.array([90.0, 100.0]))  # type: ignore[arg-type]
    assert start.tolist() == [2]
    assert size.tolist() == [1]


def test_tie_blocks_rejects_bad_scores() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        tie_blocks(np.zeros((2, 2)), FIELD)
    with pytest.raises(ValueError, match="NaN"):
        tie_blocks(np.array([np.nan]), FIELD)


# --- tie_block_payouts -------------------------------------------------------


def test_block_pays_the_mean_over_its_ranks() -> None:
    table = CONTEST.payout_table
    # Ranks 1..3 shared: (10 + 4 + 4) / 3.
    assert tie_block_payouts(np.array([1]), np.array([3]), table, 5).tolist() == [6.0]
    # Ranks 2..4: (4 + 4 + 0) / 3.
    np.testing.assert_allclose(tie_block_payouts(np.array([2]), np.array([3]), table, 5), [8 / 3])


def test_block_past_the_field_is_truncated() -> None:
    table = CONTEST.payout_table
    # Starts at 3, would run to 7; only ranks 3..5 exist: (4 + 0 + 0) / 3.
    np.testing.assert_allclose(tie_block_payouts(np.array([3]), np.array([5]), table, 5), [4 / 3])
    # Starts past the field entirely: collapses onto the last rank, which pays 0.
    assert tie_block_payouts(np.array([9]), np.array([1]), table, 5).tolist() == [0.0]


def test_block_on_a_table_that_pays_nothing_pays_nothing() -> None:
    table = np.zeros(7)
    assert tie_block_payouts(np.array([1, 2]), np.array([2, 1]), table, 5).tolist() == [0.0, 0.0]


def test_block_payouts_reject_a_wrongly_sized_table_and_bad_blocks() -> None:
    with pytest.raises(ValueError, match="expected \\(7,\\)"):
        tie_block_payouts(np.array([1]), np.array([1]), np.zeros(6), 5)
    with pytest.raises(ValueError, match="rank 1 or later"):
        tie_block_payouts(np.array([0]), np.array([1]), np.zeros(7), 5)
    with pytest.raises(ValueError, match="at least one entry"):
        tie_block_payouts(np.array([1]), np.array([0]), np.zeros(7), 5)


@settings(max_examples=100, deadline=None)
@given(
    scores=st.lists(st.integers(min_value=0, max_value=6), min_size=1, max_size=12),
    payouts=st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=12, max_size=12),
)
def test_a_full_field_of_our_own_entries_is_paid_exactly_the_table(
    scores: list[int], payouts: list[float]
) -> None:
    # Money is conserved: when our entries *are* the whole field, the payouts
    # handed out sum to what the table pays over that many ranks — however the
    # ties fall. Integer scores make ties common.
    n = len(scores)
    table = np.zeros(n + 2)
    table[1 : n + 1] = payouts[:n]
    start, size = tie_blocks(np.asarray(scores, dtype=float), np.array([]))
    paid = tie_block_payouts(start, size, table, n)
    assert paid.sum() == pytest.approx(table[1 : n + 1].sum())


# --- realized_ranks_and_payouts ---------------------------------------------


def test_realized_ranks_and_payouts_end_to_end() -> None:
    ranks, payouts = realized_ranks_and_payouts(
        np.array([90.0, 90.0, 100.0, 50.0, 60.0]), FIELD, CONTEST
    )
    # 100 ties the field's 100: ranks 1..2, (10 + 4) / 2 = 7.
    # The two 90s join the field's two 90s below both 100s: ranks 3..6, capped
    # at 5, so ranks 3..5 pay (4 + 0 + 0) / 3.
    # 50 and 60 are below everyone and below each other; rank capped at 5.
    assert ranks.tolist() == [3, 3, 1, 5, 5]
    np.testing.assert_allclose(payouts, [4 / 3, 4 / 3, 7.0, 0.0, 0.0])


def test_rank_is_capped_at_the_field_size() -> None:
    ranks, payouts = realized_ranks_and_payouts(np.array([1.0, 2.0, 3.0]), FIELD, CONTEST)
    assert ranks.tolist() == [5, 5, 5]
    assert payouts.tolist() == [0.0, 0.0, 0.0]


def test_a_raw_array_field_is_accepted() -> None:
    ranks, _ = realized_ranks_and_payouts(np.array([95.0]), np.array([90.0, 100.0, 80.0]), CONTEST)
    assert ranks.tolist() == [2]


def test_void_field_is_refused() -> None:
    with pytest.raises(ValueError, match="void"):
        realized_ranks_and_payouts(np.array([1.0]), np.zeros(5), CONTEST)


def test_field_larger_than_the_contest_is_refused() -> None:
    with pytest.raises(ValueError, match="holds 6 entries but field_size is 5"):
        realized_ranks_and_payouts(np.array([1.0]), np.arange(1.0, 7.0), CONTEST)
