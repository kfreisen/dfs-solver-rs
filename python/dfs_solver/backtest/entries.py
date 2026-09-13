"""The rows a backtest produces: one per lineup per contest.

[`Entries`][dfs_solver.backtest.entries.Entries] is a column store — every
field is a NumPy array and index `i` is the same entry in each — because that
is what the report functions reduce over and what a backtest of a few thousand
contests produces by the million. It is not a dataframe and does not want to
be one; `to_dicts` hands the rows to whatever you prefer.

Two facts about every row are kept apart on purpose: `lineup_rank` is what
happened, and `lineup_payout` is that rank read through one payout table.
Tables get corrected — an operator restates a prize pool, a transcription is
fixed — and [`rescore`][dfs_solver.backtest.loop.rescore] re-reads the stored
ranks through the new table without re-running anything.

A [`RunManifest`][dfs_solver.backtest.entries.RunManifest] says what produced
the rows. Two runs with different seeds or outcome counts are different
experiments, and [`merge`][dfs_solver.backtest.entries.merge] refuses to pool
them into one table unless told the caller knows that.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = ["COLUMNS", "Entries", "RunManifest", "merge"]


@dataclass(frozen=True)
class RunManifest:
    """What produced a set of entries.

    Attributes:
        seed: The seed the simulation and selection ran under.
        n_sims: How many simulated outcomes each lineup was scored over.
        label: A name for the run.
        extra: Anything else worth pinning — a model version, a git hash. Read
            only; compared by value.
    """

    seed: int
    n_sims: int
    label: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze `extra` so equality is stable after construction."""
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunManifest):
            return NotImplemented
        return (
            self.seed == other.seed
            and self.n_sims == other.n_sims
            and self.label == other.label
            and dict(self.extra) == dict(other.extra)
        )

    def __hash__(self) -> int:
        return hash((self.seed, self.n_sims, self.label, tuple(sorted(self.extra.items()))))


# Column name -> dtype. `object` columns hold whatever the caller keys by —
# strategy names, slate keys, contest ids — compared only for equality.
COLUMNS: Mapping[str, type] = MappingProxyType(
    {
        "strategy": object,
        "slate_key": object,
        "contest_id": object,
        "lineup_idx": np.int64,
        "entry_fee": np.float64,
        "field_size": np.int64,
        "max_entries_per_user": np.int64,
        "sim_p50": np.float64,
        "sim_p99": np.float64,
        "ev_pred": np.float64,
        "lineup_score": np.float64,
        "lineup_rank": np.int64,
        "lineup_payout": np.float64,
        "lineup_pit": np.float64,
    }
)
"""Every column of [`Entries`][dfs_solver.backtest.entries.Entries] and its dtype."""


def _column(name: str, values: Any) -> np.ndarray:
    dtype = COLUMNS[name]
    if dtype is object:
        array = np.empty(len(values), dtype=object)
        array[:] = list(values)
        return array
    return np.asarray(values, dtype=dtype)


@dataclass(frozen=True)
class Entries:
    """One row per lineup per contest, as parallel columns.

    Attributes:
        strategy: Label of whatever chose the lineups. Opaque.
        slate_key: Which slate the contest was on. Opaque.
        contest_id: Which contest. Opaque, compared for equality.
        lineup_idx: The lineup's position in the order it was entered, so the
            same lineup carries the same index in every contest on its slate.
        entry_fee: What the entry cost.
        field_size: The contest's size.
        max_entries_per_user: The contest's per-user cap.
        sim_p50: Median of the lineup's simulated scores.
        sim_p99: 99th percentile of the same.
        ev_pred: The field model's expected payout, or `nan` when the contest
            had no model.
        lineup_score: What the lineup actually scored.
        lineup_rank: Where it finished — top of its tie block, capped at the
            field size.
        lineup_payout: What that rank paid under the table in force when the
            row was written. See [`rescore`][dfs_solver.backtest.loop.rescore].
        lineup_pit: The realized score's percentile in the lineup's own
            simulated distribution. Uniform on `[0, 1]` when the simulation is
            calibrated; anything else is the simulation being wrong in a
            direction the histogram shows.
        manifest: What produced these rows, if recorded.
    """

    strategy: np.ndarray
    slate_key: np.ndarray
    contest_id: np.ndarray
    lineup_idx: np.ndarray
    entry_fee: np.ndarray
    field_size: np.ndarray
    max_entries_per_user: np.ndarray
    sim_p50: np.ndarray
    sim_p99: np.ndarray
    ev_pred: np.ndarray
    lineup_score: np.ndarray
    lineup_rank: np.ndarray
    lineup_payout: np.ndarray
    lineup_pit: np.ndarray
    manifest: RunManifest | None = None

    def __post_init__(self) -> None:
        """Coerce every column to its dtype and check they are parallel."""
        n = len(self.strategy)
        for name in COLUMNS:
            values = _column(name, getattr(self, name))
            if values.ndim != 1:
                msg = f"{name} must be one-dimensional, got shape {values.shape}"
                raise ValueError(msg)
            if values.shape[0] != n:
                msg = (
                    f"{name} has {values.shape[0]} entries but strategy has {n}; "
                    f"every column must be parallel"
                )
                raise ValueError(msg)
            object.__setattr__(self, name, values)

    def __len__(self) -> int:
        """Number of rows."""
        return int(self.strategy.shape[0])

    @classmethod
    def from_columns(
        cls, columns: Mapping[str, Any], manifest: RunManifest | None = None
    ) -> Entries:
        """Build from one sequence per column, coercing each to its dtype.

        The constructor itself coerces too, but is typed for arrays; this is
        the entry point for lists and scalars.

        Raises:
            KeyError: If a column is missing or unknown. A row store with a
                column silently dropped is a report with a number missing.
        """
        missing = [name for name in COLUMNS if name not in columns]
        unknown = [name for name in columns if name not in COLUMNS]
        if missing or unknown:
            msg = (
                f"columns missing {missing} and unknown {unknown}; expected exactly {list(COLUMNS)}"
            )
            raise KeyError(msg)
        return cls(**{name: _column(name, columns[name]) for name in COLUMNS}, manifest=manifest)

    @classmethod
    def empty(cls, manifest: RunManifest | None = None) -> Entries:
        """No rows."""
        return cls.from_columns({name: [] for name in COLUMNS}, manifest)

    @classmethod
    def from_dicts(
        cls, rows: Iterable[Mapping[str, Any]], manifest: RunManifest | None = None
    ) -> Entries:
        """Build from one mapping per row, each carrying every column."""
        materialized = list(rows)
        for i, row in enumerate(materialized):
            missing = [name for name in COLUMNS if name not in row]
            if missing:
                msg = f"row {i} is missing column(s): {missing}"
                raise KeyError(msg)
        return cls.from_columns(
            {name: [row[name] for row in materialized] for name in COLUMNS}, manifest
        )

    def to_dicts(self) -> list[dict[str, Any]]:
        """One mapping per row, with Python scalars."""
        columns = {name: getattr(self, name).tolist() for name in COLUMNS}
        return [{name: columns[name][i] for name in COLUMNS} for i in range(len(self))]

    def filter(self, mask: np.ndarray) -> Entries:
        """The rows where `mask` is true. Keeps the manifest."""
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != (len(self),):
            msg = f"mask has shape {keep.shape} but there are {len(self)} rows"
            raise ValueError(msg)
        return Entries(
            **{name: getattr(self, name)[keep] for name in COLUMNS}, manifest=self.manifest
        )

    def where(self, **equal: Any) -> Entries:
        """The rows where every named column equals the value given.

        `entries.where(strategy="a", contest_id=7)` is the common filter, and
        a mask over object columns is fiddly enough to be worth naming.
        """
        keep = np.ones(len(self), dtype=bool)
        for name, value in equal.items():
            if name not in COLUMNS:
                msg = f"no column {name!r}; columns are {list(COLUMNS)}"
                raise KeyError(msg)
            keep &= np.asarray([v == value for v in getattr(self, name).tolist()], dtype=bool)
        return self.filter(keep)

    @staticmethod
    def concat(parts: Sequence[Entries]) -> Entries:
        """Stack several tables into one.

        The manifest is kept only when every part agrees on it; otherwise the
        result has none. To *refuse* a mismatch instead, use
        [`merge`][dfs_solver.backtest.entries.merge].
        """
        if not parts:
            return Entries.empty()
        manifests = {part.manifest for part in parts}
        manifest = parts[0].manifest if len(manifests) == 1 else None
        return Entries(
            **{name: np.concatenate([getattr(part, name) for part in parts]) for name in COLUMNS},
            manifest=manifest,
        )


def merge(parts: Sequence[Entries], *, allow_mismatched_manifests: bool = False) -> Entries:
    """Pool several runs' entries, refusing to mix experiments by accident.

    Args:
        parts: The tables to pool.
        allow_mismatched_manifests: Pool anyway when the manifests differ. The
            result then carries no manifest, since none describes it.

    Raises:
        ValueError: If the manifests differ and mixing was not allowed. Two
            runs at different seeds or outcome counts are two experiments;
            one table of both reports a number that belongs to neither.
    """
    manifests = {part.manifest for part in parts}
    if len(manifests) > 1 and not allow_mismatched_manifests:
        described = sorted(str(m) for m in manifests)
        msg = (
            f"refusing to merge entries from {len(manifests)} different runs: {described}. "
            f"Pass allow_mismatched_manifests=True if pooling them is what you mean."
        )
        raise ValueError(msg)
    return Entries.concat(parts)


# `fields` is imported so the column list can be checked against the dataclass
# once at import time: a column added to one and not the other is a bug that
# would otherwise surface as a KeyError deep in a report.
_declared = {f.name for f in fields(Entries)} - {"manifest"}
if _declared != set(COLUMNS):  # pragma: no cover - import-time consistency check
    msg = f"Entries fields {sorted(_declared)} and COLUMNS {sorted(COLUMNS)} disagree"
    raise RuntimeError(msg)
