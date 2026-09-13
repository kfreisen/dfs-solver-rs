"""Reducing entries to the numbers a backtest is run for.

Every function here takes an [`Entries`][dfs_solver.backtest.entries.Entries]
table and returns plain rows — lists of dicts — grouped by strategy. There is no
dataframe library between the numbers and you.

Three of the figures deserve a word, because each guards against a way a
tournament backtest flatters itself:

* **`roi_ex_top`** is ROI with the single most profitable contest removed. A
  tournament's return is dominated by its tail, and one first-place finish in a
  large field can carry a thousand losing contests. The number with that
  contest removed is not the "real" ROI — the win happened — but a strategy
  whose whole edge is one contest is a different thing from one that is
  profitable without it, and both numbers together say which you have.
* **`n_big`** counts entries that paid a hundred times their fee, and
  **`hit_10x`** ten times. Those are the tail: how often the thing a tournament
  is entered for actually happened.
* **`any_win`** is the share of contests in which at least one entry cashed.
  It is a floor, not a ceiling — a strategy cashing one min-cash entry in every
  contest is not thereby good — and it is what falls first when a field model
  is wrong.

[`pit_table`][dfs_solver.backtest.report.pit_table] and
[`ev_calibration_table`][dfs_solver.backtest.report.ev_calibration_table] are
not about money. They check the two models the loop relied on — the simulation
and the field — against what happened, so that a good ROI on a bad model reads
as luck rather than as evidence.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    from dfs_solver.backtest.entries import Entries

__all__ = [
    "BIG_MULTIPLE",
    "EV_BIN_EDGES",
    "ev_calibration_table",
    "period_table",
    "pit_table",
    "roi_table",
]

BIG_MULTIPLE = 100.0
"""A payout at or above this many times the fee counts as `n_big`."""

EV_BIN_EDGES: tuple[float, ...] = (0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0)
"""Default bin edges for `ev_calibration_table`, in multiples of the fee."""


def _group_ids(*columns: np.ndarray) -> tuple[np.ndarray, list[tuple[Any, ...]]]:
    """Dense group id per row over the tuple of the given columns.

    Object columns cannot go through `np.unique` when their values are not
    mutually comparable, so this is a dictionary walk — one pass, and the group
    keys come back in first-seen order.
    """
    keys: dict[tuple[Any, ...], int] = {}
    ids = np.empty(len(columns[0]), dtype=np.int64)
    lists = [column.tolist() for column in columns]
    for i, key in enumerate(zip(*lists, strict=True)):
        ids[i] = keys.setdefault(key, len(keys))
    return ids, list(keys)


def _ratio(numerator: float, denominator: float) -> float:
    """A ratio that is `nan` rather than an exception when nothing was staked."""
    return float(numerator / denominator) if denominator else float("nan")


def roi_table(
    entries: Entries, *, period_of: Callable[[Any], Hashable] | None = None
) -> list[dict[str, Any]]:
    """Per strategy: what was staked, what came back, and the shape of the return.

    Args:
        entries: The rows.
        period_of: Optional map from a `slate_key` to a period — `lambda d:
            d.year` for dates. When given, each row carries `roi_by_period`,
            a mapping from period to that period's ROI.

    Returns:
        One row per strategy, sorted by `profit` descending, with keys
        `strategy`, `contests`, `entries`, `fees`, `payout`, `profit`, `roi`,
        `roi_ex_top`, `top_profit`, `n_big`, `hit_10x`, `wins` (entries that
        finished first), and `any_win` (share of contests with a cashing
        entry). ROI is `profit / fees`, and `nan` when nothing was staked.
    """
    if len(entries) == 0:
        return []
    fee = entries.entry_fee
    pay = entries.lineup_payout
    pnl = pay - fee
    contest_ids, contest_keys = _group_ids(entries.strategy, entries.contest_id)
    n_contests = len(contest_keys)
    contest_fee = np.bincount(contest_ids, weights=fee, minlength=n_contests)
    contest_pnl = np.bincount(contest_ids, weights=pnl, minlength=n_contests)
    contest_cashed = np.bincount(contest_ids, weights=(pay > 0), minlength=n_contests) > 0
    contest_strategy = [key[0] for key in contest_keys]

    rows: list[dict[str, Any]] = []
    for strategy in dict.fromkeys(entries.strategy.tolist()):
        in_rows = np.asarray([s == strategy for s in entries.strategy.tolist()], dtype=bool)
        in_contests = np.asarray([s == strategy for s in contest_strategy], dtype=bool)
        fees = float(fee[in_rows].sum())
        profit = float(pnl[in_rows].sum())
        top = int(np.argmax(np.where(in_contests, contest_pnl, -np.inf)))
        rows.append(
            {
                "strategy": strategy,
                "contests": int(in_contests.sum()),
                "entries": int(in_rows.sum()),
                "fees": fees,
                "payout": float(pay[in_rows].sum()),
                "profit": profit,
                "roi": _ratio(profit, fees),
                "roi_ex_top": _ratio(profit - contest_pnl[top], fees - contest_fee[top]),
                "top_profit": float(contest_pnl[top]),
                "n_big": int((pay[in_rows] >= BIG_MULTIPLE * fee[in_rows]).sum()),
                "hit_10x": int((pay[in_rows] >= 10.0 * fee[in_rows]).sum()),
                "wins": int((entries.lineup_rank[in_rows] == 1).sum()),
                "any_win": float(contest_cashed[in_contests].mean()),
            }
        )
        if period_of is not None:
            by_period = period_table(entries.filter(in_rows), period_of)
            rows[-1]["roi_by_period"] = {row["period"]: row["roi"] for row in by_period}
    rows.sort(key=lambda row: -row["profit"])
    return rows


def period_table(entries: Entries, period_of: Callable[[Any], Hashable]) -> list[dict[str, Any]]:
    """Per strategy and period: the same return figures, one period at a time.

    The usual period is a year, and a year is the usual reason to look: a
    tournament strategy that was profitable in aggregate and lost in each of
    the last two seasons is one whose edge has gone. The period is whatever
    `period_of` makes of a `slate_key` — a year, a month, a season half, a
    regime label.

    Returns:
        One row per `(strategy, period)`, sorted by strategy then period, with
        keys `strategy`, `period`, `slates`, `contests`, `entries`, `fees`,
        `payout`, `roi`, `n_big`, and `cash_rate` (share of entries that paid
        anything).
    """
    if len(entries) == 0:
        return []
    periods = np.empty(len(entries), dtype=object)
    periods[:] = [period_of(key) for key in entries.slate_key.tolist()]
    ids, keys = _group_ids(entries.strategy, periods)
    fee = entries.entry_fee
    pay = entries.lineup_payout
    rows: list[dict[str, Any]] = []
    for group, (strategy, period) in enumerate(keys):
        mask = ids == group
        fees = float(fee[mask].sum())
        payout = float(pay[mask].sum())
        rows.append(
            {
                "strategy": strategy,
                "period": period,
                "slates": len(set(entries.slate_key[mask].tolist())),
                "contests": len(set(entries.contest_id[mask].tolist())),
                "entries": int(mask.sum()),
                "fees": fees,
                "payout": payout,
                "roi": _ratio(payout - fees, fees),
                "n_big": int((pay[mask] >= BIG_MULTIPLE * fee[mask]).sum()),
                "cash_rate": float((pay[mask] > 0).mean()),
            }
        )
    rows.sort(key=lambda row: (str(row["strategy"]), str(row["period"])))
    return rows


def pit_table(entries: Entries, *, bins: int = 10) -> list[dict[str, Any]]:
    """How each lineup's realized score sat in its own simulated distribution.

    `lineup_pit` is a probability integral transform: the fraction of simulated
    outcomes at or below what actually happened. If the simulation is
    calibrated those values are uniform, and every bin holds `1 / bins` of
    them. A hump in the low bins means the simulation runs hot — lineups scored
    less than it thought — and a hump in the high bins means it runs cold. A
    U-shape means the tails are too thin.

    Returns:
        One row per strategy with `strategy`, `entries`, `shares` (a list of
        `bins` fractions summing to one), `mean_pit` (`0.5` when calibrated),
        `coverage_50` and `coverage_90` — the share of realized scores inside
        the simulation's central 50% and 90% intervals, which are `0.5` and
        `0.9` when it is right.
    """
    if bins < 1:
        msg = f"bins must be at least 1, got {bins}"
        raise ValueError(msg)
    if len(entries) == 0:
        return []
    rows: list[dict[str, Any]] = []
    for strategy in dict.fromkeys(entries.strategy.tolist()):
        mask = np.asarray([s == strategy for s in entries.strategy.tolist()], dtype=bool)
        pit = entries.lineup_pit[mask]
        which = np.clip(np.floor(pit * bins).astype(np.int64), 0, bins - 1)
        counts = np.bincount(which, minlength=bins)
        rows.append(
            {
                "strategy": strategy,
                "entries": int(mask.sum()),
                "shares": (counts / counts.sum()).tolist(),
                "mean_pit": float(pit.mean()),
                "coverage_50": float(((pit >= 0.25) & (pit <= 0.75)).mean()),
                "coverage_90": float(((pit >= 0.05) & (pit <= 0.95)).mean()),
            }
        )
    return rows


def ev_calibration_table(
    entries: Entries, *, edges: Sequence[float] = EV_BIN_EDGES
) -> list[dict[str, Any]]:
    """Was the field model's expected payout worth believing?

    Rows are binned by `ev_pred / entry_fee` — what the model said an entry was
    worth, in fees — and each bin reports what those entries actually returned,
    in the same units. A calibrated model has `realized_x_fee` tracking
    `pred_x_fee` bin by bin; a model that flatters its top bin is the one to
    stop sizing entries by.

    Rows without a prediction (`ev_pred` is `nan`) are left out, and the result
    is empty when no row has one.

    Args:
        entries: The rows.
        edges: Bin edges in multiples of the fee. `n` edges make `n + 1` bins;
            the first is everything below the first edge, the last everything
            at or above the last.

    Returns:
        One row per non-empty bin, in bin order, with `bin` (a label such as
        `"1-1.25"`), `entries`, `pred_x_fee`, `realized_x_fee`, and `cash_rate`.
    """
    cuts = np.asarray(edges, dtype=np.float64)
    if cuts.ndim != 1 or cuts.size == 0 or np.any(np.diff(cuts) <= 0):
        msg = "edges must be a non-empty, strictly increasing sequence"
        raise ValueError(msg)
    has_ev = ~np.isnan(entries.ev_pred) & (entries.entry_fee > 0)
    if not has_ev.any():
        return []
    fee = entries.entry_fee[has_ev]
    ev_x = entries.ev_pred[has_ev] / fee
    pay_x = entries.lineup_payout[has_ev] / fee
    which = np.searchsorted(cuts, ev_x, side="right")
    labels = [f"<{cuts[0]:g}"]
    labels += [f"{lo:g}-{hi:g}" for lo, hi in pairwise(cuts.tolist())]
    labels.append(f">={cuts[-1]:g}")
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        mask = which == index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": label,
                "entries": int(mask.sum()),
                "pred_x_fee": float(ev_x[mask].mean()),
                "realized_x_fee": float(pay_x[mask].mean()),
                "cash_rate": float((pay_x[mask] > 0).mean()),
            }
        )
    return rows
