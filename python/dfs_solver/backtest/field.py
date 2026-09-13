"""An empirical model of the field: what score does a rank take?

This package ships no model of opponents — how a field is built is sport-specific
and the honest thing is to say so. What it can do is *measure* one. For a contest
of a given kind, pool the realized scores of every entry in every comparable
contest that has already happened, sort them, and ask: for a score `x`, what
fraction of that pool finished at or below it? That fraction is a rank, and the
payout table turns a rank into money.

It is measured, not simulated: no opponent generator, no ownership, no knobs to
tune. And it is **walk-forward** — a contest's reference pool contains only
contests that finished before it. A model that could see the contest it is
predicting would be exactly the leak a backtest exists to avoid.

What "comparable" means is the caller's decision, expressed as an opaque band
key: fee bracket and field-size bracket is the usual choice, and
[`band_of`][dfs_solver.backtest.field.band_of] turns a value and a set of edges
into one. What "before" means is likewise the caller's — an `as_of` key that
sorts, with windows in the same units. Dates and `timedelta`s work; so do
integers.

The model's calibration is checked on every entered lineup:
[`run_slate`][dfs_solver.backtest.loop.run_slate] records the model's expected
payout beside the realized one, and
[`ev_calibration_table`][dfs_solver.backtest.report.ev_calibration_table] bins
them. A bad model is visible, not silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np

from dfs_solver.backtest.ranking import FieldScores, is_void

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

__all__ = [
    "FieldCdf",
    "FieldModel",
    "ReferenceContest",
    "band_of",
    "expected_payouts",
    "pooled_cdf",
]


class SupportsWindow(Protocol):
    """An `as_of` key: ordered, and shiftable back by a window.

    `datetime.date` with `timedelta` windows satisfies it; so does `int` with
    `int` windows.
    """

    def __lt__(self, other: Any, /) -> bool: ...

    def __le__(self, other: Any, /) -> bool: ...

    def __sub__(self, other: Any, /) -> Any: ...


K = TypeVar("K", bound=SupportsWindow)


def band_of(value: float, edges: Sequence[float]) -> int:
    """Which bracket a value falls in, given the bracket edges.

    `band_of(x, (3, 10, 25))` is `0` below 3, `1` in `[3, 10)`, `2` in
    `[10, 25)`, `3` at or above 25. A convenience for building band keys; the
    model itself never reads the number, only compares keys for equality.
    """
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), value, side="right"))


@dataclass(frozen=True)
class FieldCdf:
    """The pooled, sorted scores of a set of reference contests.

    Attributes:
        scores: Every entry's score from every reference contest, ascending.
        n_contests: How many contests were pooled.
        window: The window that was wide enough, in the caller's units. Carried
            so a report can say how far back the model had to reach.
    """

    scores: np.ndarray
    n_contests: int
    window: Any = None

    def __post_init__(self) -> None:
        """Check there is something to pool and it is sorted."""
        scores = np.asarray(self.scores, dtype=np.float64)
        if scores.ndim != 1 or scores.size == 0:
            msg = "a field CDF needs at least one score"
            raise ValueError(msg)
        if np.any(np.diff(scores) < 0):
            msg = "field CDF scores must be sorted ascending"
            raise ValueError(msg)
        scores = np.ascontiguousarray(scores)
        scores.flags.writeable = False
        object.__setattr__(self, "scores", scores)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Fraction of the pooled field scoring at or below each `x`."""
        return np.searchsorted(self.scores, np.asarray(x, dtype=np.float64), side="right") / len(
            self.scores
        )


@dataclass(frozen=True)
class ReferenceContest:
    """One finished contest the model may learn from.

    Attributes:
        band: Opaque key for the kind of contest. Contests are pooled only with
            others in the same band.
        as_of: When it finished, in whatever units the caller windows in.
        scores: Every entry's realized score, in any order.
    """

    band: Hashable
    as_of: Any
    scores: np.ndarray

    def __post_init__(self) -> None:
        """Refuse a void contest; a field of zeros teaches nothing true."""
        scores = np.asarray(self.scores, dtype=np.float64)
        if is_void(scores):
            msg = (
                f"reference contest (band {self.band!r}, as_of {self.as_of!r}) is void — "
                f"no entry scored — and cannot describe a field"
            )
            raise ValueError(msg)
        object.__setattr__(self, "scores", FieldScores.of(scores).scores)


def pooled_cdf(
    references: Iterable[ReferenceContest],
    as_of: K,
    *,
    windows: Sequence[Any] | None,
    min_contests: int = 5,
    max_scores: int | None = None,
    seed: int = 0,
) -> FieldCdf | None:
    """Pool the references that finished before `as_of`, widening until enough do.

    Args:
        references: Candidate contests, already restricted to one band.
        as_of: The contest being modelled. Only references strictly before it
            are used; a reference at the same key is excluded, since two
            contests on the same slate have not finished when either starts.
        windows: How far back to look, narrowest first. Each is tried in turn
            and the first that admits `min_contests` references wins; that is
            what keeps a model from reaching years back when last month has
            plenty, while still finding *something* in a thin stretch of the
            calendar. A reference is in a window when `as_of - window <= ref`.
            `None` uses everything before `as_of`, with no widening.
        min_contests: How many references a window needs before it counts.
        max_scores: Cap on the pooled array; above it a uniform subsample is
            taken, seeded so the model is reproducible.
        seed: For that subsample.

    Returns:
        The pooled CDF, or `None` when no window admits enough references. A
        contest with no model gets no expected payout, and the loop records
        `nan` rather than a guess.
    """
    if min_contests < 1:
        msg = f"min_contests must be at least 1, got {min_contests}"
        raise ValueError(msg)
    earlier = [r for r in references if r.as_of < as_of]
    tried: list[tuple[Any, list[ReferenceContest]]] = (
        [(None, earlier)]
        if windows is None
        else [(w, [r for r in earlier if (as_of - w) <= r.as_of]) for w in windows]
    )
    for window, chosen in tried:
        if len(chosen) < min_contests:
            continue
        pooled = np.concatenate([r.scores for r in chosen])
        if max_scores is not None and len(pooled) > max_scores:
            pooled = np.random.default_rng(seed).choice(pooled, max_scores, replace=False)
        return FieldCdf(np.sort(pooled), len(chosen), window)
    return None


class FieldModel:
    """Walk-forward pooled CDFs, one per band and `as_of`, cached.

    Holds every reference contest grouped by band and answers
    [`for_contest`][dfs_solver.backtest.field.FieldModel.for_contest] with
    [`pooled_cdf`][dfs_solver.backtest.field.pooled_cdf] under the settings
    given here. Nothing is computed until asked, and each answer is kept.

    Args:
        references: Every finished contest the model may learn from.
        windows: Passed to `pooled_cdf`; see there.
        min_contests: Likewise.
        max_scores: Likewise.
        seed: Likewise.
    """

    def __init__(
        self,
        references: Iterable[ReferenceContest],
        *,
        windows: Sequence[Any] | None,
        min_contests: int = 5,
        max_scores: int | None = None,
        seed: int = 0,
    ) -> None:
        self.windows = None if windows is None else tuple(windows)
        self.min_contests = min_contests
        self.max_scores = max_scores
        self.seed = seed
        self._by_band: dict[Hashable, list[ReferenceContest]] = {}
        for reference in references:
            self._by_band.setdefault(reference.band, []).append(reference)
        self._cache: dict[tuple[Hashable, Any], FieldCdf | None] = {}

    @property
    def bands(self) -> tuple[Hashable, ...]:
        """Every band with at least one reference, in first-seen order."""
        return tuple(self._by_band)

    def for_contest(self, band: Hashable, as_of: K) -> FieldCdf | None:
        """The model for a contest in `band` finishing at `as_of`, or `None`.

        `None` means no window admitted enough references — including a band
        the model has never seen. The caller decides what that means; the loop
        records no expected payout for such a contest.
        """
        key = (band, as_of)
        if key not in self._cache:
            self._cache[key] = pooled_cdf(
                self._by_band.get(band, ()),
                as_of,
                windows=self.windows,
                min_contests=self.min_contests,
                max_scores=self.max_scores,
                seed=self.seed,
            )
        return self._cache[key]


def expected_payouts(
    sim_scores: np.ndarray, cdf: FieldCdf, payout_table: np.ndarray, field_size: int
) -> np.ndarray:
    """Each lineup's expected payout under the field model.

    For every simulated outcome, the lineup's score is placed in the pooled
    field, the rank that implies is read from the payout table, and the mean
    over outcomes is taken. Own-entry cannibalization is not modelled: each
    lineup is ranked against the field alone, as if it were the only entry.

    Args:
        sim_scores: An `(n_lineups, n_outcomes)` array.
        cdf: The model for this contest's band.
        payout_table: What each rank pays, length `field_size + 2`.
        field_size: The contest's size.

    Returns:
        One expected payout per lineup.
    """
    scores = np.asarray(sim_scores, dtype=np.float64)
    if scores.ndim != 2:
        msg = f"sim_scores must be two-dimensional, got shape {scores.shape}"
        raise ValueError(msg)
    table = np.asarray(payout_table, dtype=np.float64)
    if table.shape != (field_size + 2,):
        msg = f"payout_table has shape {table.shape}; expected ({field_size + 2},)"
        raise ValueError(msg)
    if scores.shape[1] == 0:
        msg = "sim_scores has no outcomes to average over"
        raise ValueError(msg)
    below = cdf.cdf(scores.ravel()).reshape(scores.shape)
    rank = np.clip(np.floor((1.0 - below) * field_size).astype(np.int64) + 1, 1, field_size)
    return np.asarray(table[rank].mean(axis=1))
