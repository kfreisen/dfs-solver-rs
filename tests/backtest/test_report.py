"""The report reductions, checked against arithmetic done by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest
from dfs_solver.backtest.entries import Entries
from dfs_solver.backtest.report import (
    ev_calibration_table,
    period_table,
    pit_table,
    roi_table,
)


def make(
    rows: list[tuple[str, int, int, float, float, int, float, float]],
) -> Entries:
    """(strategy, year, contest_id, fee, payout, rank, pit, ev_pred) per row."""
    n = len(rows)
    return Entries.from_columns(
        {
            "strategy": [r[0] for r in rows],
            "slate_key": [(r[1], r[2]) for r in rows],
            "contest_id": [r[2] for r in rows],
            "lineup_idx": list(range(n)),
            "entry_fee": [r[3] for r in rows],
            "field_size": [100] * n,
            "max_entries_per_user": [20] * n,
            "sim_p50": [0.0] * n,
            "sim_p99": [0.0] * n,
            "ev_pred": [r[7] for r in rows],
            "lineup_score": [0.0] * n,
            "lineup_rank": [r[5] for r in rows],
            "lineup_payout": [r[4] for r in rows],
            "lineup_pit": [r[6] for r in rows],
        }
    )


NAN = float("nan")


@pytest.fixture
def entries() -> Entries:
    return make(
        [
            # Strategy a, contest 1 (2024): one 150x bomb and two blanks.
            ("a", 2024, 1, 10.0, 1500.0, 1, 0.99, 15.0),
            ("a", 2024, 1, 10.0, 0.0, 50, 0.30, 9.0),
            ("a", 2024, 1, 10.0, 0.0, 60, 0.10, 4.0),
            # Strategy a, contest 2 (2025): two blanks.
            ("a", 2025, 2, 10.0, 0.0, 70, 0.55, NAN),
            ("a", 2025, 2, 10.0, 0.0, 80, 0.65, NAN),
            # Strategy b, contest 1: a min-cash.
            ("b", 2024, 1, 10.0, 15.0, 20, 0.80, 12.0),
            ("b", 2024, 1, 10.0, 0.0, 90, 0.20, 12.0),
        ]
    )


def test_roi_table(entries: Entries) -> None:
    rows = roi_table(entries, period_of=lambda key: key[0])
    assert [r["strategy"] for r in rows] == ["a", "b"]
    a, b = rows
    assert a["contests"] == 2
    assert a["entries"] == 5
    assert a["fees"] == 50.0
    assert a["payout"] == 1500.0
    assert a["profit"] == 1450.0
    assert a["roi"] == pytest.approx(29.0)
    # Without contest 1: two blanks on $20 staked.
    assert a["top_profit"] == 1470.0
    assert a["roi_ex_top"] == pytest.approx(-1.0)
    assert a["n_big"] == 1
    assert a["hit_10x"] == 1
    assert a["wins"] == 1
    assert a["any_win"] == 0.5
    assert a["roi_by_period"] == {2024: pytest.approx(49.0), 2025: pytest.approx(-1.0)}
    assert b["profit"] == -5.0
    assert b["roi_ex_top"] != b["roi_ex_top"]  # only contest removed: nan
    assert b["any_win"] == 1.0
    assert b["n_big"] == 0


def test_roi_table_sorts_by_profit_and_handles_empty(entries: Entries) -> None:
    assert roi_table(Entries.empty()) == []
    swapped = entries.filter(np.asarray([s == "b" for s in entries.strategy.tolist()]))
    assert roi_table(swapped)[0]["strategy"] == "b"
    assert "roi_by_period" not in roi_table(entries)[0]


def test_roi_is_nan_on_free_entries() -> None:
    free = make([("a", 2024, 1, 0.0, 0.0, 1, 0.5, NAN)])
    assert math.isnan(roi_table(free)[0]["roi"])


def test_period_table(entries: Entries) -> None:
    rows = period_table(entries, lambda key: key[0])
    assert [(r["strategy"], r["period"]) for r in rows] == [("a", 2024), ("a", 2025), ("b", 2024)]
    first = rows[0]
    assert first["slates"] == 1
    assert first["contests"] == 1
    assert first["entries"] == 3
    assert first["fees"] == 30.0
    assert first["payout"] == 1500.0
    assert first["roi"] == pytest.approx(49.0)
    assert first["n_big"] == 1
    assert first["cash_rate"] == pytest.approx(1 / 3)
    assert rows[1]["roi"] == pytest.approx(-1.0)
    assert period_table(Entries.empty(), lambda key: key) == []


def test_pit_table(entries: Entries) -> None:
    rows = pit_table(entries, bins=10)
    a = rows[0]
    assert a["strategy"] == "a"
    assert a["entries"] == 5
    # 0.99 -> bin 9, 0.30 -> 3, 0.10 -> 1, 0.55 -> 5, 0.65 -> 6.
    assert a["shares"] == pytest.approx([0, 0.2, 0, 0.2, 0, 0.2, 0.2, 0, 0, 0.2])
    assert a["mean_pit"] == pytest.approx((0.99 + 0.30 + 0.10 + 0.55 + 0.65) / 5)
    assert a["coverage_50"] == pytest.approx(3 / 5)
    assert a["coverage_90"] == pytest.approx(4 / 5)
    # A PIT of exactly 1.0 lands in the last bin, not one past it.
    edge = make([("x", 2024, 1, 1.0, 0.0, 1, 1.0, NAN)])
    assert pit_table(edge, bins=4)[0]["shares"] == [0.0, 0.0, 0.0, 1.0]
    assert pit_table(Entries.empty()) == []
    with pytest.raises(ValueError, match="bins"):
        pit_table(entries, bins=0)


def test_ev_calibration_table(entries: Entries) -> None:
    rows = ev_calibration_table(entries)
    # Predictions in fees: 1.5, 0.9, 0.4, (nan, nan), 1.2, 1.2. An edge belongs
    # to the bin above it, so 1.5 is in "1.5-2".
    assert [r["bin"] for r in rows] == ["<0.5", "0.8-1", "1-1.25", "1.5-2"]
    by_bin = {r["bin"]: r for r in rows}
    assert by_bin["<0.5"]["entries"] == 1
    assert by_bin["<0.5"]["pred_x_fee"] == pytest.approx(0.4)
    assert by_bin["<0.5"]["realized_x_fee"] == 0.0
    assert by_bin["1-1.25"]["entries"] == 2
    assert by_bin["1-1.25"]["realized_x_fee"] == pytest.approx(0.75)
    assert by_bin["1-1.25"]["cash_rate"] == 0.5
    assert by_bin["1.5-2"]["realized_x_fee"] == pytest.approx(150.0)


def test_ev_calibration_top_bin_and_empty() -> None:
    top = make([("a", 2024, 1, 1.0, 0.0, 1, 0.5, 9.0)])
    assert [r["bin"] for r in ev_calibration_table(top)] == [">=3"]
    unmodelled = make([("a", 2024, 1, 1.0, 0.0, 1, 0.5, NAN)])
    assert ev_calibration_table(unmodelled) == []
    assert ev_calibration_table(Entries.empty()) == []
    with pytest.raises(ValueError, match="strictly increasing"):
        ev_calibration_table(top, edges=(1.0, 1.0))
