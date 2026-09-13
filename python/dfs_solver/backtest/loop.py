"""The backtest loop for one slate, as a pure function.

[`run_slate`][dfs_solver.backtest.loop.run_slate] takes what the rest of the
package has already produced — candidate lineups scored over simulated outcomes,
an order in which to enter them, what each actually scored — and the contests
that were on the slate with their realized fields, and writes one
[`Entries`][dfs_solver.backtest.entries.Entries] row per lineup per contest.

It does not choose lineups. The `order` comes from the caller — typically
[`select_portfolio`][dfs_solver.select.select_portfolio], whose output is
prefix-consistent: a shorter portfolio is a prefix of a longer one. That
property is what lets one selection serve every contest on the slate. A
20-entry contest takes the first twenty, a 150-entry contest the first hundred
and fifty, and the twenty are the same lineups in both.

Every contest is entered to its cap or to the order's length, whichever is
smaller. There is no sizing logic here, and none should be added: how many
entries to place is a strategy, and a strategy belongs to the caller. Pass a
shorter `order` to enter fewer.

A contest whose field is void — every entry at or below zero, a cancelled or
refunded slate — is refused rather than skipped. Skipping would make a run that
silently entered fewer contests than it was given, and refusing makes the caller
filter deliberately, with [`is_void`][dfs_solver.backtest.ranking.is_void].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from dfs_solver.backtest.contest import Contest
from dfs_solver.backtest.entries import COLUMNS, Entries, RunManifest
from dfs_solver.backtest.field import FieldCdf, expected_payouts
from dfs_solver.backtest.ranking import FieldScores, realized_ranks_and_payouts

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ["SlateContest", "rescore", "run_slate"]


@dataclass(frozen=True)
class SlateContest:
    """A contest on the slate being backtested, with what actually happened in it.

    Attributes:
        contest: The contest — fee, size, cap, payout curve.
        field: Every other entry's realized score.
        cdf: The field model for this contest, if one exists. Supplies
            `ev_pred`; `None` records `nan` there.
    """

    contest: Contest
    field: FieldScores | np.ndarray
    cdf: FieldCdf | None = None

    def __post_init__(self) -> None:
        """Refuse a void field or one larger than the contest, at construction."""
        field = self.field if isinstance(self.field, FieldScores) else FieldScores.of(self.field)
        object.__setattr__(self, "field", field)
        if field.void:
            msg = (
                f"contest {self.contest.contest_id!r}: the field never scored (void — a "
                f"cancelled or refunded contest). Filter it out with is_void() rather than "
                f"entering it; nothing can be ranked in it."
            )
            raise ValueError(msg)
        if len(field) > self.contest.field_size:
            msg = (
                f"contest {self.contest.contest_id!r}: the field holds {len(field)} "
                f"entries but field_size is {self.contest.field_size}"
            )
            raise ValueError(msg)


def run_slate(
    *,
    strategy: Any,
    slate_key: Any,
    sim_scores: np.ndarray,
    order: np.ndarray | Sequence[int],
    realized_scores: np.ndarray,
    contests: Sequence[SlateContest],
    manifest: RunManifest | None = None,
) -> Entries:
    """Enter one slate's lineups into its contests and record what happened.

    Args:
        strategy: Label for whatever chose the lineups. Carried into every row.
        slate_key: Opaque key for this slate. Carried into every row, and what
            the report functions group by when asked for per-period figures.
        sim_scores: An `(n_candidates, n_outcomes)` array from
            [`score_lineups`][dfs_solver.select.score_lineups] — every
            candidate, not just the chosen ones, so `order` can index it.
        order: Indices into the candidates, best first, prefix-consistent. Each
            contest takes `order[:min(cap, len(order))]`. Empty enters nothing
            and returns no rows; that is an answer, not an error, because a
            selection that found nothing worth entering has said so.
        realized_scores: What each candidate actually scored, parallel to
            `sim_scores`.
        contests: The slate's contests with their realized fields.
        manifest: Recorded on the result.

    Returns:
        One row per entered lineup per contest. `lineup_idx` is the position in
        `order`, so the same lineup has the same index across contests.

    Raises:
        ValueError: If the arrays disagree about how many candidates there are,
            or `order` repeats or exceeds them.
    """
    scores = np.asarray(sim_scores, dtype=np.float64)
    if scores.ndim != 2:
        msg = f"sim_scores must be two-dimensional, got shape {scores.shape}"
        raise ValueError(msg)
    realized = np.asarray(realized_scores, dtype=np.float64)
    if realized.shape != (scores.shape[0],):
        msg = (
            f"realized_scores has shape {realized.shape} but sim_scores describes "
            f"{scores.shape[0]} candidates"
        )
        raise ValueError(msg)
    chosen = np.asarray(order, dtype=np.int64).ravel()
    if chosen.size and (chosen.min() < 0 or chosen.max() >= scores.shape[0]):
        msg = f"order names a candidate outside the {scores.shape[0]} scored"
        raise ValueError(msg)
    if len(np.unique(chosen)) != chosen.size:
        msg = "order lists the same candidate more than once"
        raise ValueError(msg)
    if chosen.size == 0 or not contests:
        return Entries.empty(manifest)

    # Per-lineup facts that do not depend on the contest, computed once for the
    # longest prefix any contest can take.
    longest = min(chosen.size, max(c.contest.max_entries_per_user for c in contests))
    picked = chosen[:longest]
    sim = scores[picked]
    actual = realized[picked]
    pit = (sim <= actual[:, None]).mean(axis=1)
    p50 = np.quantile(sim, 0.5, axis=1)
    p99 = np.quantile(sim, 0.99, axis=1)

    parts: list[Entries] = []
    for slate_contest in contests:
        contest = slate_contest.contest
        k = min(contest.max_entries_per_user, chosen.size)
        ranks, payouts = realized_ranks_and_payouts(actual[:k], slate_contest.field, contest)
        ev = (
            expected_payouts(sim[:k], slate_contest.cdf, contest.payout_table, contest.field_size)
            if slate_contest.cdf is not None
            else np.full(k, np.nan)
        )
        parts.append(
            Entries.from_columns(
                {
                    "strategy": [strategy] * k,
                    "slate_key": [slate_key] * k,
                    "contest_id": [contest.contest_id] * k,
                    "lineup_idx": np.arange(k),
                    "entry_fee": np.full(k, contest.entry_fee),
                    "field_size": np.full(k, contest.field_size),
                    "max_entries_per_user": np.full(k, contest.max_entries_per_user),
                    "sim_p50": p50[:k],
                    "sim_p99": p99[:k],
                    "ev_pred": ev,
                    "lineup_score": actual[:k],
                    "lineup_rank": ranks,
                    "lineup_payout": payouts,
                    "lineup_pit": pit[:k],
                },
                manifest,
            )
        )
    return Entries.concat(parts)


def rescore(
    entries: Entries, contests: Iterable[Contest], *, drop_missing: bool = False
) -> Entries:
    """Re-read every stored rank through a new set of payout tables.

    Ranks do not depend on payouts, so a corrected table needs no re-run. Each
    row's `lineup_payout` becomes what its `lineup_rank` pays under the contest
    of the same id in `contests`.

    One approximation, stated: a row that was part of a tie block was paid the
    block's mean, and the block's size is not stored. Rescoring pays the block's
    *top* rank instead. For a tie on the paid ranks that differs from the exact
    split by a few cents; rows that tied for the win differ by more. Re-run the
    slate when that matters.

    Args:
        entries: Rows to rescore.
        contests: The contests, carrying the tables to use now.
        drop_missing: What to do with a row whose contest is not in `contests`.
            `False` raises, since silently losing rows changes every total;
            `True` drops them, for a contest that has since been found void.

    Returns:
        A new table with `lineup_payout` recomputed; everything else, the
        manifest included, is carried over.
    """
    by_id = {contest.contest_id: contest for contest in contests}
    ids = entries.contest_id.tolist()
    present = np.asarray([cid in by_id for cid in ids], dtype=bool)
    if not present.all():
        missing = sorted({str(cid) for cid, ok in zip(ids, present, strict=True) if not ok})
        if not drop_missing:
            msg = (
                f"{int((~present).sum())} row(s) belong to {len(missing)} contest(s) not "
                f"given: {missing[:24]}. Pass drop_missing=True to discard them."
            )
            raise KeyError(msg)
        entries = entries.filter(present)
        ids = entries.contest_id.tolist()
    ranks = entries.lineup_rank
    payouts = np.asarray(
        [by_id[cid].payout_for_rank(int(rank)) for cid, rank in zip(ids, ranks, strict=True)],
        dtype=np.float64,
    )
    columns = {name: getattr(entries, name) for name in COLUMNS}
    columns["lineup_payout"] = payouts
    return Entries(**columns, manifest=entries.manifest)
