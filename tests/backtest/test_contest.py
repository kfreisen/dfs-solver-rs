"""Payout tiers, contests, and the synthetic curve.

Most of these are rejections. A payout table that overlaps or skips rank 1 is a
transcription error, and the place to catch one is before any number is
computed from it.
"""

from __future__ import annotations

import numpy as np
import pytest
from dfs_solver.backtest.contest import (
    Contest,
    PayoutTier,
    payout_table_from_tiers,
    synthetic_gpp,
    synthetic_gpp_tiers,
    validate_tiers,
)
from hypothesis import given, settings
from hypothesis import strategies as st

TIERS = (PayoutTier(1, 1, 100.0), PayoutTier(2, 3, 20.0), PayoutTier(4, 10, 5.0))


def contest(**overrides: object) -> Contest:
    kwargs: dict[str, object] = {
        "contest_id": 1,
        "entry_fee": 5.0,
        "field_size": 20,
        "max_entries_per_user": 3,
        "tiers": TIERS,
    }
    kwargs.update(overrides)
    return Contest(**kwargs)  # type: ignore[arg-type]


# --- PayoutTier ----------------------------------------------------------


def test_tier_spots_and_total() -> None:
    tier = PayoutTier(4, 10, 5.0)
    assert tier.spots == 7
    assert tier.total == 35.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_rank": 0, "max_rank": 1, "payout": 1.0}, "1-based"),
        ({"min_rank": 3, "max_rank": 2, "payout": 1.0}, "which is empty"),
        ({"min_rank": 1, "max_rank": 1, "payout": -1.0}, "non-negative"),
        ({"min_rank": 1, "max_rank": 1, "payout": float("nan")}, "finite"),
    ],
)
def test_tier_rejects_junk(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PayoutTier(**kwargs)  # type: ignore[arg-type]


# --- validate_tiers -------------------------------------------------------


def test_validate_sorts_by_rank() -> None:
    assert validate_tiers(reversed(TIERS)) == TIERS


def test_validate_rejects_no_tiers() -> None:
    with pytest.raises(ValueError, match="at least one payout tier"):
        validate_tiers(())


def test_validate_rejects_a_table_that_skips_first_place() -> None:
    with pytest.raises(ValueError, match="rank 1 must be paid"):
        validate_tiers((PayoutTier(2, 3, 1.0),))


def test_validate_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_tiers((PayoutTier(1, 3, 1.0), PayoutTier(3, 5, 1.0)))


def test_validate_allows_a_gap() -> None:
    # Operators do publish tables with unpaid ranks between paid ones.
    tiers = (PayoutTier(1, 1, 5.0), PayoutTier(5, 6, 1.0))
    assert validate_tiers(tiers) == tiers


# --- payout_table_from_tiers ---------------------------------------------


def test_payout_table_is_indexed_by_rank() -> None:
    table = payout_table_from_tiers(TIERS, 20)
    assert table.shape == (22,)
    assert table[0] == 0.0
    assert table[1] == 100.0
    assert table[2] == table[3] == 20.0
    assert table[4] == table[10] == 5.0
    assert table[11] == 0.0
    assert table[21] == 0.0
    assert not table.flags.writeable


def test_payout_table_truncates_a_tier_past_the_field() -> None:
    table = payout_table_from_tiers(TIERS, 5)
    assert table.shape == (7,)
    assert table[5] == 5.0


def test_payout_table_rejects_an_empty_field() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        payout_table_from_tiers(TIERS, 0)


# --- Contest -------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"entry_fee": -1.0}, "entry fee"),
        ({"entry_fee": float("inf")}, "entry fee"),
        ({"field_size": 0}, "field_size"),
        ({"max_entries_per_user": 0}, "max_entries_per_user"),
        ({"total_prizes": -5.0}, "total_prizes"),
        ({"tiers": ()}, "at least one payout tier"),
    ],
)
def test_contest_rejects_junk(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        contest(**overrides)


def test_contest_normalizes_tier_order_and_is_hashable() -> None:
    c = contest(tiers=tuple(reversed(TIERS)))
    assert c.tiers == TIERS
    assert hash(c) == hash(contest())


def test_contest_payout_table_is_cached_and_read_only() -> None:
    c = contest()
    assert c.payout_table is c.payout_table
    with pytest.raises(ValueError, match="read-only"):
        c.payout_table[1] = 0.0


def test_contest_totals() -> None:
    c = contest()
    assert c.implied_total == 100.0 + 2 * 20.0 + 7 * 5.0
    assert c.rake == pytest.approx(1.0 - 175.0 / 100.0)
    assert c.last_paid_rank == 10


def test_contest_implied_total_drops_ranks_past_the_field() -> None:
    assert contest(field_size=5).implied_total == 100.0 + 2 * 20.0 + 2 * 5.0


def test_contest_free_has_no_rake() -> None:
    assert contest(entry_fee=0.0).rake == 0.0


def test_payout_for_rank() -> None:
    c = contest()
    assert c.payout_for_rank(1) == 100.0
    assert c.payout_for_rank(10) == 5.0
    assert c.payout_for_rank(11) == 0.0
    assert c.payout_for_rank(999) == 0.0
    with pytest.raises(ValueError, match="1-based"):
        c.payout_for_rank(0)


# --- synthetic_gpp -------------------------------------------------------


def test_synthetic_tiers_sum_to_the_prize_pool() -> None:
    tiers = synthetic_gpp_tiers(20.0, 10_000)
    pool = 20.0 * 10_000 * (1 - 0.159)
    # Per-rank rounding to the cent accumulates over thousands of spots.
    assert sum(t.total for t in tiers) == pytest.approx(pool, abs=0.01 * 10_000)
    assert tiers[0].min_rank == 1
    # Top-heavy: first place pays more than second, and the flat band at the
    # bottom pays a bit over the fee.
    assert tiers[0].payout > tiers[1].payout
    assert tiers[-1].payout == pytest.approx(1.6 * 20.0, rel=0.05)
    assert tiers[-1].max_rank == 2_200


@pytest.mark.parametrize("field_size", [1, 2, 3, 5, 10, 26, 27, 50, 100, 1_000])
def test_synthetic_tiers_are_valid_for_small_fields(field_size: int) -> None:
    tiers = synthetic_gpp_tiers(5.0, field_size)
    assert validate_tiers(tiers) == tiers
    assert tiers[-1].max_rank <= field_size
    pool = 5.0 * field_size * (1 - 0.159)
    assert sum(t.total for t in tiers) == pytest.approx(pool, abs=0.01 * field_size + 0.01)


@settings(max_examples=60, deadline=None)
@given(
    fee=st.floats(min_value=0.25, max_value=5_000.0, allow_nan=False),
    field_size=st.integers(min_value=1, max_value=250_000),
)
def test_synthetic_tiers_always_validate(fee: float, field_size: int) -> None:
    tiers = synthetic_gpp_tiers(fee, field_size)
    assert validate_tiers(tiers) == tiers
    assert tiers[-1].max_rank <= field_size
    pool = fee * field_size * (1 - 0.159)
    assert sum(t.total for t in tiers) == pytest.approx(
        pool, rel=1e-3, abs=0.01 * field_size + 0.01
    )


def test_synthetic_alpha_override_changes_the_top() -> None:
    flat = synthetic_gpp_tiers(5.0, 1_000, alpha=0.1)
    steep = synthetic_gpp_tiers(5.0, 1_000, alpha=0.9)
    assert steep[0].payout > flat[0].payout


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"field_size": 0}, "field_size"),
        ({"rake": 1.0}, "rake"),
        ({"cash_fraction": 0.0}, "cash_fraction"),
    ],
)
def test_synthetic_rejects_junk(kwargs: dict[str, object], match: str) -> None:
    full: dict[str, object] = {"entry_fee": 5.0, "field_size": 100}
    full.update(kwargs)
    with pytest.raises(ValueError, match=match):
        synthetic_gpp_tiers(**full)  # type: ignore[arg-type]


def test_synthetic_gpp_builds_a_contest() -> None:
    c = synthetic_gpp(3.0, 5_000, max_entries_per_user=20, contest_id="x", slate_key="s", rake=0.1)
    assert c.contest_id == "x"
    assert c.slate_key == "s"
    assert c.max_entries_per_user == 20
    assert c.name == "synthetic $3 GPP (5,000 entries)"
    assert c.rake == pytest.approx(0.1, abs=1e-3)
    assert np.all(c.payout_table >= 0)


def test_payout_table_drops_a_tier_entirely_past_the_field() -> None:
    # A three-entry contest whose table lists ranks 4..10: nobody can finish
    # there, so those ranks contribute nothing and the table is silently shorter.
    table = payout_table_from_tiers(TIERS, 3)
    assert table.tolist() == [0.0, 100.0, 20.0, 20.0, 0.0]
