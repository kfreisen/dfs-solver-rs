"""The empirical field model: pooling, walk-forward windows, expected payout."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from dfs_solver.backtest.contest import synthetic_gpp
from dfs_solver.backtest.field import (
    FieldCdf,
    FieldModel,
    ReferenceContest,
    band_of,
    expected_payouts,
    pooled_cdf,
)


def reference(as_of: int, *scores: float, band: str = "b") -> ReferenceContest:
    return ReferenceContest(band, as_of, np.asarray(scores))


# --- band_of ---------------------------------------------------------------


def test_band_of_brackets_by_edges() -> None:
    edges = (3.0, 10.0, 25.0)
    assert [band_of(x, edges) for x in (1.0, 3.0, 9.99, 10.0, 25.0, 1000.0)] == [0, 1, 1, 2, 3, 3]


# --- FieldCdf ---------------------------------------------------------------


def test_field_cdf_is_the_empirical_distribution() -> None:
    cdf = FieldCdf(np.array([1.0, 2.0, 2.0, 4.0]), n_contests=1)
    np.testing.assert_allclose(cdf.cdf(np.array([0.0, 2.0, 3.0, 5.0])), [0.0, 0.75, 0.75, 1.0])
    assert not cdf.scores.flags.writeable


def test_field_cdf_rejects_empty_and_unsorted() -> None:
    with pytest.raises(ValueError, match="at least one score"):
        FieldCdf(np.array([]), n_contests=0)
    with pytest.raises(ValueError, match="sorted"):
        FieldCdf(np.array([2.0, 1.0]), n_contests=1)


# --- ReferenceContest -------------------------------------------------------


def test_reference_sorts_its_scores_and_refuses_void() -> None:
    assert reference(1, 3.0, 1.0, 2.0).scores.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="void"):
        reference(1, 0.0, 0.0)


# --- pooled_cdf ---------------------------------------------------------------


@pytest.fixture
def references() -> list[ReferenceContest]:
    # One contest per integer "day" 1..10, each with scores identifying its day.
    return [reference(day, float(day), float(day) + 0.5) for day in range(1, 11)]


def test_pooled_cdf_widens_until_enough_contests(references: list[ReferenceContest]) -> None:
    # Window 2 before day 8 admits days 6 and 7; window 5 admits days 3..7.
    cdf = pooled_cdf(references, 8, windows=(2, 5, 100), min_contests=3)
    assert cdf is not None
    assert cdf.window == 5
    assert cdf.n_contests == 5
    assert cdf.scores.min() == 3.0
    assert cdf.scores.max() == 7.5


def test_pooled_cdf_takes_the_narrowest_window_that_suffices(
    references: list[ReferenceContest],
) -> None:
    cdf = pooled_cdf(references, 8, windows=(2, 5), min_contests=2)
    assert cdf is not None
    assert cdf.window == 2
    assert cdf.n_contests == 2


def test_pooled_cdf_is_walk_forward(references: list[ReferenceContest]) -> None:
    # The contest on day 8 itself, and every later one, is never in the pool.
    cdf = pooled_cdf(references, 8, windows=None, min_contests=1)
    assert cdf is not None
    assert cdf.window is None
    assert cdf.n_contests == 7
    assert cdf.scores.max() == 7.5


def test_pooled_cdf_is_none_when_no_window_suffices(references: list[ReferenceContest]) -> None:
    assert pooled_cdf(references, 8, windows=(1,), min_contests=3) is None
    assert pooled_cdf([], 8, windows=None, min_contests=1) is None
    assert pooled_cdf(references, 1, windows=None, min_contests=1) is None


def test_pooled_cdf_subsamples_reproducibly(references: list[ReferenceContest]) -> None:
    a = pooled_cdf(references, 11, windows=None, min_contests=1, max_scores=6, seed=1)
    b = pooled_cdf(references, 11, windows=None, min_contests=1, max_scores=6, seed=1)
    assert a is not None
    assert b is not None
    assert len(a.scores) == 6
    assert a.scores.tolist() == b.scores.tolist()


def test_pooled_cdf_rejects_a_zero_minimum(references: list[ReferenceContest]) -> None:
    with pytest.raises(ValueError, match="min_contests"):
        pooled_cdf(references, 8, windows=None, min_contests=0)


def test_pooled_cdf_works_with_dates() -> None:
    day = dt.date(2025, 6, 1)
    refs = [
        ReferenceContest("b", day - dt.timedelta(days=d), np.array([10.0 * d])) for d in (1, 5, 40)
    ]
    cdf = pooled_cdf(
        refs, day, windows=(dt.timedelta(days=7), dt.timedelta(days=60)), min_contests=2
    )
    assert cdf is not None
    assert cdf.window == dt.timedelta(days=7)
    assert cdf.scores.tolist() == [10.0, 50.0]


# --- FieldModel ---------------------------------------------------------------


def test_field_model_pools_within_a_band_and_caches(references: list[ReferenceContest]) -> None:
    other = [reference(day, 100.0 + day, band="c") for day in range(1, 11)]
    model = FieldModel([*references, *other], windows=(3,), min_contests=2)
    assert model.bands == ("b", "c")
    b = model.for_contest("b", 8)
    c = model.for_contest("c", 8)
    assert b is not None
    assert c is not None
    assert b.scores.max() == 7.5
    assert c.scores.max() == 107.0
    assert model.for_contest("b", 8) is b
    assert model.for_contest("unseen", 8) is None


# --- expected_payouts -----------------------------------------------------------


def test_expected_payouts_reads_rank_from_the_pooled_field() -> None:
    contest = synthetic_gpp(1.0, 100, max_entries_per_user=1)
    cdf = FieldCdf(np.linspace(0.0, 100.0, 101), n_contests=1)
    # A lineup at the top of the field every time is paid first place every
    # time; one at the bottom is paid nothing; one in between is in between.
    sims = np.array([[200.0, 200.0], [-1.0, -1.0], [200.0, -1.0]])
    ev = expected_payouts(sims, cdf, contest.payout_table, contest.field_size)
    first = contest.payout_for_rank(1)
    np.testing.assert_allclose(ev, [first, 0.0, first / 2])


def test_expected_payouts_is_monotone_in_score() -> None:
    contest = synthetic_gpp(1.0, 1_000, max_entries_per_user=1)
    cdf = FieldCdf(np.sort(np.random.default_rng(0).normal(100.0, 20.0, 5_000)), n_contests=10)
    sims = np.linspace(40.0, 180.0, 50)[:, None] + np.zeros((1, 8))
    ev = expected_payouts(sims, cdf, contest.payout_table, contest.field_size)
    assert np.all(np.diff(ev) >= 0)


def test_expected_payouts_rejects_bad_shapes() -> None:
    contest = synthetic_gpp(1.0, 10, max_entries_per_user=1)
    cdf = FieldCdf(np.arange(5.0), n_contests=1)
    with pytest.raises(ValueError, match="two-dimensional"):
        expected_payouts(np.zeros(3), cdf, contest.payout_table, 10)
    with pytest.raises(ValueError, match="expected \\(12,\\)"):
        expected_payouts(np.zeros((3, 2)), cdf, np.zeros(11), 10)
    with pytest.raises(ValueError, match="no outcomes"):
        expected_payouts(np.zeros((3, 0)), cdf, contest.payout_table, 10)
