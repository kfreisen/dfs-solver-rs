"""What a contest pays, and to whom.

A contest is an entry fee, a field size, a cap on entries per user, and a payout
curve: which finishing ranks earn what. Everything downstream — ranking, expected
payout, ROI — reads the curve through
[`Contest.payout_table`][dfs_solver.backtest.contest.Contest.payout_table], an
array indexed by 1-based rank, so no consumer ever walks the tiers itself.

Tiers are the operator's own published table, and the validation here is strict
because a table that overlaps or skips ranks is not a contest but a transcription
error: two tiers claiming rank 3 would silently pay whichever was listed last.

[`synthetic_gpp`][dfs_solver.backtest.contest.synthetic_gpp] is the one thing
here that is not data. It is a stand-in curve for when the real one is not to
hand — examples, tests, a rough sizing — shaped like a large-field tournament
without being any particular one. Nothing that reports a result should be run on
it if the official table exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "Contest",
    "PayoutTier",
    "payout_table_from_tiers",
    "synthetic_gpp",
    "synthetic_gpp_tiers",
    "validate_tiers",
]


@dataclass(frozen=True, slots=True)
class PayoutTier:
    """A run of finishing ranks that each pay the same amount.

    Attributes:
        min_rank: First rank in the run, 1-based and inclusive.
        max_rank: Last rank in the run, inclusive. Equal to `min_rank` for a
            single place.
        payout: What *each* rank in the run pays — not the run's total.
    """

    min_rank: int
    max_rank: int
    payout: float

    def __post_init__(self) -> None:
        """Reject a tier that names no rank or pays something that is not money."""
        if self.min_rank < 1:
            msg = f"tier starts at rank {self.min_rank}; ranks are 1-based"
            raise ValueError(msg)
        if self.max_rank < self.min_rank:
            msg = f"tier spans ranks {self.min_rank}..{self.max_rank}, which is empty"
            raise ValueError(msg)
        if not math.isfinite(self.payout) or self.payout < 0.0:
            msg = f"tier {self.min_rank}..{self.max_rank} pays {self.payout}; expected a finite, non-negative amount"
            raise ValueError(msg)

    @property
    def spots(self) -> int:
        """How many ranks the tier covers."""
        return self.max_rank - self.min_rank + 1

    @property
    def total(self) -> float:
        """What the tier pays out in all, across every rank it covers."""
        return self.payout * self.spots


def validate_tiers(tiers: Iterable[PayoutTier]) -> tuple[PayoutTier, ...]:
    """Check a payout table and return it sorted by rank.

    Args:
        tiers: The published tiers, in any order.

    Returns:
        The same tiers, sorted by `min_rank`.

    Raises:
        ValueError: If there are no tiers, if the first tier does not start at
            rank 1, or if two tiers overlap. A gap between tiers is allowed and
            means those ranks pay nothing — operators do publish tables like that.
    """
    ordered = tuple(sorted(tiers, key=lambda t: t.min_rank))
    if not ordered:
        msg = "a contest needs at least one payout tier; a table that pays nobody is missing data"
        raise ValueError(msg)
    if ordered[0].min_rank != 1:
        msg = f"the first tier starts at rank {ordered[0].min_rank}; rank 1 must be paid"
        raise ValueError(msg)
    for earlier, later in pairwise(ordered):
        if later.min_rank <= earlier.max_rank:
            msg = (
                f"tiers {earlier.min_rank}..{earlier.max_rank} and "
                f"{later.min_rank}..{later.max_rank} overlap; each rank pays exactly once"
            )
            raise ValueError(msg)
    return ordered


def payout_table_from_tiers(tiers: Iterable[PayoutTier], field_size: int) -> np.ndarray:
    """Expand tiers into an array indexed by rank.

    Args:
        tiers: The published tiers. Validated on the way through.
        field_size: How many entries the contest holds. Ranks past it are
            dropped, because nobody can finish there.

    Returns:
        A float64 array of length `field_size + 2` where index `r` is what rank
        `r` pays. Index 0 is never a rank and stays zero; the extra trailing
        element lets a cumulative sum be taken without an off-by-one guard at the
        bottom of the field.
    """
    if field_size < 1:
        msg = f"field_size must be at least 1, got {field_size}"
        raise ValueError(msg)
    table = np.zeros(field_size + 2, dtype=np.float64)
    for tier in validate_tiers(tiers):
        lo = tier.min_rank
        hi = min(tier.max_rank, field_size)
        if lo <= hi:
            table[lo : hi + 1] = tier.payout
    table.flags.writeable = False
    return table


@dataclass(frozen=True)
class Contest:
    """One contest: its fee, its size, and its payout curve.

    Frozen and hashable so it can key a cache. The realized field — what every
    entry actually scored — is deliberately not here; it is a separate
    [`FieldScores`][dfs_solver.backtest.ranking.FieldScores], because the same
    contest is ranked against a real field in a backtest and against a modelled
    one when predicting.

    Attributes:
        contest_id: The operator's identifier. Any hashable value; it is only
            ever compared for equality and carried through to entry rows.
        entry_fee: What one entry costs.
        field_size: How many entries the contest holds. Ranks are capped here.
        max_entries_per_user: How many entries one account may submit. The loop
            enters exactly `min(this, entries available)`.
        tiers: The published payout tiers. Validated: sorted, non-overlapping,
            and starting at rank 1.
        name: Optional label for reports.
        slate_key: Optional opaque key for the slate this contest was on. Used
            to group entries; never interpreted.
        total_prizes: The operator's advertised prize pool, if known. Reported
            beside [`implied_total`][dfs_solver.backtest.contest.Contest.implied_total]
            so a mistranscribed table shows up as a disagreement.
    """

    contest_id: int | str
    entry_fee: float
    field_size: int
    max_entries_per_user: int
    tiers: tuple[PayoutTier, ...]
    name: str | None = None
    slate_key: object = None
    total_prizes: float | None = None

    def __post_init__(self) -> None:
        """Validate the scalars and normalize the tiers."""
        if not math.isfinite(self.entry_fee) or self.entry_fee < 0.0:
            msg = f"contest {self.contest_id!r}: entry fee {self.entry_fee} must be finite and non-negative"
            raise ValueError(msg)
        if self.field_size < 1:
            msg = f"contest {self.contest_id!r}: field_size {self.field_size} must be at least 1"
            raise ValueError(msg)
        if self.max_entries_per_user < 1:
            msg = (
                f"contest {self.contest_id!r}: max_entries_per_user "
                f"{self.max_entries_per_user} must be at least 1"
            )
            raise ValueError(msg)
        if self.total_prizes is not None and (
            not math.isfinite(self.total_prizes) or self.total_prizes < 0.0
        ):
            msg = f"contest {self.contest_id!r}: total_prizes {self.total_prizes} must be finite and non-negative"
            raise ValueError(msg)
        object.__setattr__(self, "tiers", validate_tiers(self.tiers))

    @cached_property
    def payout_table(self) -> np.ndarray:
        """What each rank pays, indexed by 1-based rank.

        Length `field_size + 2`; see
        [`payout_table_from_tiers`][dfs_solver.backtest.contest.payout_table_from_tiers].
        Read-only, and computed once.
        """
        return payout_table_from_tiers(self.tiers, self.field_size)

    @property
    def implied_total(self) -> float:
        """What the table pays in all, summed over every rank in the field.

        Differs from the tiers' own total when a tier runs past `field_size`.
        """
        return float(self.payout_table.sum())

    @property
    def rake(self) -> float:
        """The operator's share of the fees, as a fraction.

        `1 - implied_total / (entry_fee * field_size)`. Zero for a free contest,
        since there is nothing to take a share of.
        """
        fees = self.entry_fee * self.field_size
        if fees == 0.0:
            return 0.0
        return 1.0 - self.implied_total / fees

    @property
    def last_paid_rank(self) -> int:
        """The lowest finishing rank that earns anything."""
        paid = np.flatnonzero(self.payout_table)
        return int(paid.max())

    def payout_for_rank(self, rank: int) -> float:
        """What a single, untied entry finishing at `rank` earns.

        Ranks past the field pay nothing. For tied entries — which share a block
        of ranks and split its payouts — use
        [`realized_ranks_and_payouts`][dfs_solver.backtest.ranking.realized_ranks_and_payouts].
        """
        if rank < 1:
            msg = f"rank {rank} is not a finishing position; ranks are 1-based"
            raise ValueError(msg)
        if rank > self.field_size:
            return 0.0
        return float(self.payout_table[rank])


# Tier boundaries of the synthetic curve, as fractions of the field. The first
# six are fixed places; the rest scale with the field so a 100-entry and a
# 100,000-entry contest have the same shape at different resolutions.
_FIXED_BOUNDS: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 25))
_SCALED_BOUNDS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)


def synthetic_gpp_tiers(
    entry_fee: float,
    field_size: int,
    *,
    rake: float = 0.159,
    cash_fraction: float = 0.22,
    min_cash_fraction: float = 0.02,
    min_cash_multiple: float = 1.6,
    alpha: float | None = None,
) -> tuple[PayoutTier, ...]:
    """A stand-in payout curve shaped like a large-field tournament.

    A power law: rank `r` in the graduated tiers is weighted `r ** -alpha`, with
    a flat minimum-cash band at the bottom of the paid ranks. The defaults
    describe the shape of a typical operator's guaranteed-prize-pool tournament
    — roughly a sixth in rake, a little over a fifth of the field paid, and a
    min-cash that returns the fee and a bit — without being any particular
    contest.

    **Use this for examples and tests.** Anything that reports a result should
    run on the operator's published table, because the exact split at the top
    is what a tournament's economics turn on, and no formula reproduces it.

    Args:
        entry_fee: What one entry costs.
        field_size: How many entries the contest holds.
        rake: The operator's share of fees. The prize pool is the rest.
        cash_fraction: Share of the field that earns anything.
        min_cash_fraction: Share of the field in the flat minimum-cash band.
        min_cash_multiple: What the minimum-cash band pays, as a multiple of the
            fee. Its share of the pool is capped at 30% so a tiny field does not
            put everything in the flat band.
        alpha: The power-law exponent. `None` derives it from the field size,
            steeper for larger fields — the same relationship real tables show.

    Returns:
        Validated tiers, starting at rank 1, summing to the prize pool up to
        each rank's payout being rounded to the cent.
    """
    if field_size < 1:
        msg = f"field_size must be at least 1, got {field_size}"
        raise ValueError(msg)
    if not 0.0 <= rake < 1.0:
        msg = f"rake must be in [0, 1), got {rake}"
        raise ValueError(msg)
    if not 0.0 < cash_fraction <= 1.0:
        msg = f"cash_fraction must be in (0, 1], got {cash_fraction}"
        raise ValueError(msg)
    prize_pool = entry_fee * field_size * (1.0 - rake)
    if alpha is None:
        alpha = float(np.clip(0.0175 * math.log(max(field_size, 100)) + 0.5458, 0.55, 0.80))

    last_paid = min(max(2, int(field_size * cash_fraction)), field_size)
    min_cash_spots = max(1, int(field_size * min_cash_fraction))
    graduated_end = last_paid - min_cash_spots
    min_cash_share = min(min_cash_multiple * min_cash_spots / (field_size * (1.0 - rake)), 0.30)

    # Graduated tiers, clipped so each starts after the previous one ends and
    # none runs into the flat band. The raw boundaries overlap for small fields,
    # and a payout table that pays a rank twice is exactly what validation exists
    # to refuse.
    raw = list(_FIXED_BOUNDS)
    previous = 25
    for fraction in _SCALED_BOUNDS:
        edge = int(field_size * fraction)
        raw.append((previous + 1, edge))
        previous = max(previous, edge)
    raw.append((previous + 1, graduated_end))

    bounds: list[tuple[int, int]] = []
    end = 0
    for lo, hi in raw:
        lo = max(lo, end + 1)
        hi = min(hi, graduated_end)
        if lo > hi:
            continue
        bounds.append((lo, hi))
        end = hi

    if not bounds:
        # The field is too small for a graduated top: everything paid is one
        # flat band, which is what a two-entry head-to-head actually looks like.
        return (PayoutTier(1, last_paid, round(prize_pool / last_paid, 2)),)

    weights = [sum(r ** (-alpha) for r in range(lo, hi + 1)) for lo, hi in bounds]
    graduated_pool = prize_pool * (1.0 - min_cash_share)
    total_weight = sum(weights)
    tiers = [
        PayoutTier(lo, hi, round(graduated_pool * w / total_weight / (hi - lo + 1), 2))
        for (lo, hi), w in zip(bounds, weights, strict=True)
    ]
    tiers.append(
        PayoutTier(
            graduated_end + 1, last_paid, round(prize_pool * min_cash_share / min_cash_spots, 2)
        )
    )
    return validate_tiers(tiers)


def synthetic_gpp(
    entry_fee: float,
    field_size: int,
    *,
    max_entries_per_user: int = 150,
    contest_id: int | str = "synthetic",
    slate_key: object = None,
    rake: float = 0.159,
    cash_fraction: float = 0.22,
    min_cash_fraction: float = 0.02,
    min_cash_multiple: float = 1.6,
    alpha: float | None = None,
) -> Contest:
    """A [`Contest`][dfs_solver.backtest.contest.Contest] on a synthetic curve.

    Convenience over [`synthetic_gpp_tiers`][dfs_solver.backtest.contest.synthetic_gpp_tiers],
    which documents the curve parameters. The same caution applies: this is for
    examples and tests, not for anything that reports a result.
    """
    tiers = synthetic_gpp_tiers(
        entry_fee,
        field_size,
        rake=rake,
        cash_fraction=cash_fraction,
        min_cash_fraction=min_cash_fraction,
        min_cash_multiple=min_cash_multiple,
        alpha=alpha,
    )
    return Contest(
        contest_id=contest_id,
        entry_fee=entry_fee,
        field_size=field_size,
        max_entries_per_user=max_entries_per_user,
        tiers=tiers,
        name=f"synthetic ${entry_fee:g} GPP ({field_size:,} entries)",
        slate_key=slate_key,
    )
