"""Reconciling real entries against the backtest, per contest."""

from __future__ import annotations

import math

import pytest
from dfs_solver.backtest.entries import Entries
from dfs_solver.backtest.ledger import LedgerEntry, compare, totals


def backtest_rows() -> Entries:
    def row(
        strategy: str, contest_id: int, fee: float, payout: float, rank: int
    ) -> dict[str, object]:
        return {
            "strategy": strategy,
            "slate_key": "k",
            "contest_id": contest_id,
            "lineup_idx": 0,
            "entry_fee": fee,
            "field_size": 100,
            "max_entries_per_user": 20,
            "sim_p50": 0.0,
            "sim_p99": 0.0,
            "ev_pred": float("nan"),
            "lineup_score": 0.0,
            "lineup_rank": rank,
            "lineup_payout": payout,
            "lineup_pit": 0.5,
        }

    return Entries.from_dicts(
        [
            row("main", 1, 5.0, 0.0, 40),
            row("main", 1, 5.0, 20.0, 3),
            row("main", 2, 1.0, 0.0, 90),
            row("other", 1, 5.0, 500.0, 1),
        ]
    )


LEDGER = [
    LedgerEntry(1, 5.0, 8.0, 12, 100),
    LedgerEntry(1, 5.0, 0.0, 30, 100),
    LedgerEntry(3, 2.0, 0.0, 50, 200),
]


def test_ledger_entry_rejects_junk() -> None:
    with pytest.raises(ValueError, match="entry_fee"):
        LedgerEntry(1, -1.0, 0.0, 1, 10)
    with pytest.raises(ValueError, match="winnings"):
        LedgerEntry(1, 1.0, float("nan"), 1, 10)
    with pytest.raises(ValueError, match="place"):
        LedgerEntry(1, 1.0, 0.0, 0, 10)
    with pytest.raises(ValueError, match="contest_entries"):
        LedgerEntry(1, 1.0, 0.0, 1, 0)


def test_compare_joins_per_contest() -> None:
    rows = compare(LEDGER, backtest_rows(), strategy="main")
    assert [r["contest_id"] for r in rows] == [1, 3]
    one, three = rows
    assert one["field"] == 100
    assert one["real_entries"] == 2
    assert one["real_fees"] == 10.0
    assert one["real_won"] == 8.0
    assert one["real_best_place"] == 12
    assert one["real_roi"] == pytest.approx(-0.2)
    assert one["bt_entries"] == 2
    assert one["bt_fees"] == 10.0
    assert one["bt_won"] == 20.0
    assert one["bt_best_rank"] == 3
    assert one["bt_roi"] == pytest.approx(1.0)
    # A contest the backtest never entered keeps its row, with nothing beside it.
    assert three["bt_entries"] is None
    assert three["bt_roi"] is None
    assert three["real_roi"] == pytest.approx(-1.0)


def test_compare_without_a_strategy_filter_uses_every_row() -> None:
    rows = compare(LEDGER, backtest_rows())
    assert rows[0]["bt_entries"] == 3
    assert rows[0]["bt_won"] == 520.0
    assert rows[0]["bt_best_rank"] == 1


def test_totals_sum_the_shared_contests_and_report_the_rest() -> None:
    rows = compare(LEDGER, backtest_rows(), strategy="main")
    summary = totals(iter(rows))
    assert summary["contests_in_both"] == 1
    assert summary["contests_ledger_only"] == [3]
    assert summary["real_entries"] == 2
    assert summary["real_fees"] == 10.0
    assert summary["real_won"] == 8.0
    assert summary["real_roi"] == pytest.approx(-0.2)
    assert summary["bt_entries"] == 2
    assert summary["bt_fees"] == 10.0
    assert summary["bt_won"] == 20.0
    assert summary["bt_roi"] == pytest.approx(1.0)
    # Over the whole ledger: 8 back on 12 staked.
    assert summary["real_all_roi"] == pytest.approx(8.0 / 12.0 - 1.0)


def test_totals_of_nothing_are_nan_not_an_error() -> None:
    summary = totals([])
    assert summary["contests_in_both"] == 0
    assert math.isnan(summary["real_roi"])
    assert math.isnan(summary["real_all_roi"])
