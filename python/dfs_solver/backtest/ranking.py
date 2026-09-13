"""Where a lineup finished in a realized field, and what that paid.

A contest's realized field is the score of every entry that was in it. Ranking
our lineups into it is a `searchsorted`, with one rule that has to be exactly
right because tournaments pay by rank and ties at the paid ranks are common:

**Every entry with the same score shares a block of ranks and receives the
average payout over that block.** Two entries tied for first do not each take
first place — they take ranks 1 and 2 between them and each receive the mean of
what those ranks pay. That is how operators settle ties, and any other rule
either invents money (both get first) or destroys it (both get second).

Our own entries follow the same rule. A lineup that ties with a field entry joins
that entry's block; two of our own lineups with the same score share a block with
each other; a higher-scoring lineup of ours pushes the next one down a rank
exactly as a stranger's would. The reported rank is the *top* of the block.

Ranks are capped at the field size. Entering ten lineups into a field whose
scores are already the complete field means the total exceeds `field_size`, and a
lineup below every field entry cannot finish lower than last.

The field is a [`FieldScores`][dfs_solver.backtest.ranking.FieldScores] — a
sorted array, checked once. [`is_void`][dfs_solver.backtest.ranking.is_void]
is the rule for a contest that never happened: a rained-out or cancelled slate
that the operator refunded leaves a field of zeros, and nothing can be ranked in
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dfs_solver.backtest.contest import Contest

__all__ = [
    "FieldScores",
    "is_void",
    "realized_ranks_and_payouts",
    "tie_block_payouts",
    "tie_blocks",
]


def is_void(field_scores: np.ndarray) -> bool:
    """Whether a field records a contest that never took place.

    A cancelled or rained-out slate the operator refunded leaves every entry at
    zero. No entry can be ranked or paid in such a contest, and counting it as a
    loss — or a win — is a bookkeeping error rather than a result. The rule is
    *every* score at or below zero, so a field with a single scoring entry is not
    void, and an empty field is.
    """
    scores = np.asarray(field_scores, dtype=np.float64)
    return bool(scores.size == 0 or np.all(scores <= 0.0))


@dataclass(frozen=True)
class FieldScores:
    """The realized scores of every entry in a contest, sorted ascending.

    Construct with [`FieldScores.of`][dfs_solver.backtest.ranking.FieldScores.of]
    from scores in any order. Holding the sorted array means the ranking
    functions can `searchsorted` straight into it without each caller sorting
    again, and the constructor is where "these are numbers, and there are some"
    is checked once.

    Attributes:
        scores: A float64 array in ascending order.
    """

    scores: np.ndarray

    def __post_init__(self) -> None:
        """Check the array is one-dimensional, finite, and sorted."""
        scores = np.asarray(self.scores, dtype=np.float64)
        if scores.ndim != 1:
            msg = f"field scores must be one-dimensional, got shape {scores.shape}"
            raise ValueError(msg)
        if scores.size and not np.all(np.isfinite(scores)):
            msg = "field scores contain NaN or infinity; a score that is not a number cannot be ranked"
            raise ValueError(msg)
        if scores.size > 1 and np.any(np.diff(scores) < 0):
            msg = "field scores must be sorted ascending; use FieldScores.of() to sort"
            raise ValueError(msg)
        scores = np.ascontiguousarray(scores)
        scores.flags.writeable = False
        object.__setattr__(self, "scores", scores)

    @classmethod
    def of(cls, scores: np.ndarray) -> FieldScores:
        """Sort scores given in any order."""
        return cls(np.sort(np.asarray(scores, dtype=np.float64)))

    def __len__(self) -> int:
        """Number of entries in the field."""
        return int(self.scores.shape[0])

    @property
    def void(self) -> bool:
        """See [`is_void`][dfs_solver.backtest.ranking.is_void]."""
        return is_void(self.scores)

    def count_above(self, x: np.ndarray) -> np.ndarray:
        """How many field entries scored strictly more than each `x`."""
        return len(self) - np.searchsorted(
            self.scores, np.asarray(x, dtype=np.float64), side="right"
        )

    def count_tied(self, x: np.ndarray) -> np.ndarray:
        """How many field entries scored exactly each `x`."""
        values = np.asarray(x, dtype=np.float64)
        return np.searchsorted(self.scores, values, side="right") - np.searchsorted(
            self.scores, values, side="left"
        )

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Fraction of the field scoring at or below each `x`.

        Zero everywhere for an empty field, since there is nothing to be below.
        """
        if len(self) == 0:
            return np.zeros(np.shape(x), dtype=np.float64)
        return np.searchsorted(self.scores, np.asarray(x, dtype=np.float64), side="right") / len(
            self
        )


def tie_blocks(
    lineup_scores: np.ndarray, field: FieldScores | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The block of ranks each of our lineups shares, before any cap.

    Args:
        lineup_scores: One score per lineup of ours. These are ranked together:
            each pushes the others down, and equal scores share a block.
        field: Every other entry's score.

    Returns:
        `(start, size)`: the first rank of each lineup's block, 1-based and
        uncapped, and how many entries — field and our own, self included —
        share it. A lineup that ties with nobody has `size == 1`.
    """
    scores = np.atleast_1d(np.asarray(lineup_scores, dtype=np.float64))
    if scores.ndim != 1:
        msg = f"lineup_scores must be one-dimensional, got shape {scores.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(scores)):
        msg = "lineup_scores contain NaN or infinity; a score that is not a number cannot be ranked"
        raise ValueError(msg)
    if not isinstance(field, FieldScores):
        field = FieldScores.of(field)

    field_above = field.count_above(scores)
    field_tied = field.count_tied(scores)

    # The same two counts against our own entries. Sorting once and searching is
    # the same answer as comparing every pair, at n log n instead of n squared.
    own = np.sort(scores)
    own_above = len(own) - np.searchsorted(own, scores, side="right")
    own_tied = np.searchsorted(own, scores, side="right") - np.searchsorted(
        own, scores, side="left"
    )

    start = field_above + own_above + 1
    size = field_tied + own_tied
    return start.astype(np.int64), size.astype(np.int64)


def tie_block_payouts(
    start: np.ndarray, size: np.ndarray, payout_table: np.ndarray, field_size: int
) -> np.ndarray:
    """What each tie block pays per entry: the mean payout over its ranks.

    Args:
        start: First rank of each block, 1-based, as from
            [`tie_blocks`][dfs_solver.backtest.ranking.tie_blocks].
        size: Entries sharing each block.
        payout_table: What each rank pays, indexed by rank, as from
            [`Contest.payout_table`][dfs_solver.backtest.contest.Contest.payout_table].
            Its length is `field_size + 2`.
        field_size: The contest's size. A block is truncated here: ranks past
            the field do not exist and pay nothing, and a block that starts past
            it collapses onto the last rank.

    Returns:
        One payout per block, in the same order.
    """
    table = np.asarray(payout_table, dtype=np.float64)
    if table.shape != (field_size + 2,):
        msg = (
            f"payout_table has shape {table.shape}; expected ({field_size + 2},) — "
            f"one entry per rank plus the two guards"
        )
        raise ValueError(msg)
    start = np.asarray(start, dtype=np.int64)
    size = np.asarray(size, dtype=np.int64)
    if np.any(start < 1) or np.any(size < 1):
        msg = "every tie block starts at rank 1 or later and holds at least one entry"
        raise ValueError(msg)
    # cum[k] is what ranks 1..k pay in all, so a block's total is one subtraction.
    cum = np.concatenate([[0.0], np.cumsum(table[1:])])
    first = np.minimum(start, field_size)
    last = np.minimum(start + size - 1, field_size)
    return np.asarray((cum[last] - cum[first - 1]) / np.maximum(last - first + 1, 1))


def realized_ranks_and_payouts(
    lineup_scores: np.ndarray,
    field: FieldScores | np.ndarray,
    contest: Contest,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank our lineups into a realized field and read off what each earned.

    The two are returned separately and the rank is the one to keep. A rank is a
    fact about what happened; a payout is that fact read through one table, and
    [`rescore`][dfs_solver.backtest.loop.rescore] exists because tables get
    corrected after the fact.

    Args:
        lineup_scores: One realized score per lineup of ours.
        field: The other entries' scores. Ours are added on top, so pass the
            field *without* them — if the field already contains our entries,
            each is counted twice.
        contest: Supplies the payout table and the field size.

    Returns:
        `(ranks, payouts)`. `ranks` is the top of each lineup's tie block, capped
        at `contest.field_size`; `payouts` is the block's mean payout.

    Raises:
        ValueError: If the field is void, or holds more entries than the contest.
    """
    if not isinstance(field, FieldScores):
        field = FieldScores.of(field)
    if field.void:
        msg = (
            f"contest {contest.contest_id!r}: the field never scored (void — a cancelled "
            f"or refunded contest); nothing can be ranked in it"
        )
        raise ValueError(msg)
    if len(field) > contest.field_size:
        msg = (
            f"contest {contest.contest_id!r}: the field holds {len(field)} entries but "
            f"field_size is {contest.field_size}"
        )
        raise ValueError(msg)
    start, size = tie_blocks(lineup_scores, field)
    payouts = tie_block_payouts(start, size, contest.payout_table, contest.field_size)
    return np.minimum(start, contest.field_size), payouts
