"""Reconciling a backtest against contests that were actually entered.

A backtest that agrees with itself proves nothing. The check that matters is
against the operator's own record of what you entered and what you were paid:
for every contest in both, fees and winnings side by side, real beside
backtested. Where they diverge is where the backtest is wrong about something
— the field, the payout table, the lineups it thinks you played — and each
divergence has a cause that can be found.

[`LedgerEntry`][dfs_solver.backtest.ledger.LedgerEntry] is one real entry, as
an operator's contest-history export describes it.
[`compare`][dfs_solver.backtest.ledger.compare] joins on `contest_id` and
reports per contest; [`totals`][dfs_solver.backtest.ledger.totals] sums the
rows both sides have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dfs_solver.backtest.entries import Entries

__all__ = ["LedgerEntry", "compare", "totals"]


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One entry that was really placed, as the operator recorded it.

    Attributes:
        contest_id: The contest, matching the `contest_id` of the
            [`Contest`][dfs_solver.backtest.contest.Contest] it was entered in.
        entry_fee: What it cost.
        winnings: What it paid.
        place: Where it finished, 1-based.
        contest_entries: The field size the operator reported.
    """

    contest_id: int | str
    entry_fee: float
    winnings: float
    place: int
    contest_entries: int

    def __post_init__(self) -> None:
        """Reject a row that could not have come from a contest history."""
        for name, value in (("entry_fee", self.entry_fee), ("winnings", self.winnings)):
            if not math.isfinite(value) or value < 0.0:
                msg = f"ledger entry in contest {self.contest_id!r}: {name} {value} must be finite and non-negative"
                raise ValueError(msg)
        if self.place < 1:
            msg = f"ledger entry in contest {self.contest_id!r}: place {self.place} is not a finishing position"
            raise ValueError(msg)
        if self.contest_entries < 1:
            msg = f"ledger entry in contest {self.contest_id!r}: contest_entries {self.contest_entries} must be at least 1"
            raise ValueError(msg)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def compare(
    ledger: Iterable[LedgerEntry], entries: Entries, *, strategy: Any = None
) -> list[dict[str, Any]]:
    """Real fees, winnings and best place beside the backtest's, per contest.

    Every contest in the ledger gets a row. Where the backtest also entered it
    the `bt_*` fields are filled; where it did not — the slate was not
    backtested, or a filter excluded the contest — they are `None`, and the row
    is still there because a contest you played that the backtest skipped is
    itself a finding.

    Args:
        ledger: The real entries.
        entries: The backtest's rows.
        strategy: Restrict the backtest side to one strategy. `None` uses every
            row, which is only sensible when the table holds one.

    Returns:
        One row per ledger contest, sorted by contest id, with `contest_id`,
        `field` (the operator's field size), `real_entries`, `real_fees`,
        `real_won`, `real_best_place`, `real_roi`, and `bt_entries`,
        `bt_fees`, `bt_won`, `bt_best_rank`, `bt_roi`. ROI is `won / fees - 1`.
    """
    real: dict[int | str, dict[str, Any]] = {}
    for entry in ledger:
        row = real.setdefault(
            entry.contest_id,
            {
                "contest_id": entry.contest_id,
                "field": entry.contest_entries,
                "real_entries": 0,
                "real_fees": 0.0,
                "real_won": 0.0,
                "real_best_place": entry.place,
            },
        )
        row["real_entries"] += 1
        row["real_fees"] += entry.entry_fee
        row["real_won"] += entry.winnings
        row["real_best_place"] = min(row["real_best_place"], entry.place)

    backtest = entries if strategy is None else entries.where(strategy=strategy)
    ids = backtest.contest_id.tolist()
    rows: list[dict[str, Any]] = []
    for contest_id, row in real.items():
        mask = np.asarray([cid == contest_id for cid in ids], dtype=bool)
        row["real_roi"] = _ratio(row["real_won"], row["real_fees"]) - 1.0
        if mask.any():
            fees = float(backtest.entry_fee[mask].sum())
            won = float(backtest.lineup_payout[mask].sum())
            row.update(
                bt_entries=int(mask.sum()),
                bt_fees=fees,
                bt_won=won,
                bt_best_rank=int(backtest.lineup_rank[mask].min()),
                bt_roi=_ratio(won, fees) - 1.0,
            )
        else:
            row.update(bt_entries=None, bt_fees=None, bt_won=None, bt_best_rank=None, bt_roi=None)
        rows.append(row)
    rows.sort(key=lambda r: str(r["contest_id"]))
    return rows


def totals(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum a comparison over the contests both sides entered.

    Returns:
        `contests_in_both`, `contests_ledger_only` (a list of ids), and for
        each side its `entries`, `fees`, `won`, and `roi` over the shared
        contests, plus `real_all_roi` over the whole ledger.
    """
    everything = list(rows)
    both = [row for row in everything if row["bt_entries"] is not None]
    only = [row["contest_id"] for row in everything if row["bt_entries"] is None]
    real_fees = sum(row["real_fees"] for row in both)
    real_won = sum(row["real_won"] for row in both)
    bt_fees = sum(row["bt_fees"] for row in both)
    bt_won = sum(row["bt_won"] for row in both)
    all_fees = sum(row["real_fees"] for row in everything)
    all_won = sum(row["real_won"] for row in everything)
    return {
        "contests_in_both": len(both),
        "contests_ledger_only": only,
        "real_entries": sum(row["real_entries"] for row in both),
        "real_fees": real_fees,
        "real_won": real_won,
        "real_roi": _ratio(real_won, real_fees) - 1.0,
        "bt_entries": sum(row["bt_entries"] for row in both),
        "bt_fees": bt_fees,
        "bt_won": bt_won,
        "bt_roi": _ratio(bt_won, bt_fees) - 1.0,
        "real_all_roi": _ratio(all_won, all_fees) - 1.0,
    }
