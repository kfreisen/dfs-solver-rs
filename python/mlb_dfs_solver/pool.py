"""The player pool: who is available, and what is known about them.

Stored column-wise as NumPy arrays because that is the form the kernel borrows
directly. Building a pool from a list of records is a convenience
([`PlayerPool.from_records`][mlb_dfs_solver.pool.PlayerPool.from_records]); the arrays
are the real interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mlb_dfs_solver.spec import RosterSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["PlayerPool"]


@dataclass(frozen=True, slots=True)
class PlayerPool:
    """Players available to build lineups from.

    Every array is parallel: index `i` is the same player throughout.

    Attributes:
        projections: Expected score.
        stddevs: Standard deviation of score. Drives the upside bias — a player
            with no variance is never preferred for their ceiling.
        salaries: Salary cost.
        ownership: Projected ownership as a fraction in `[0, 1]`. Used to fade
            popular players; pass zeros to disable that entirely.
        positions: Position eligibility bitmasks, encoded against a
            [`RosterSpec`][mlb_dfs_solver.spec.RosterSpec].
        keys: Per-player group keys, e.g. `{"team": array_of_team_ids}`. A negative
            value means the player belongs to no group and is never capped.
        names: Optional labels, carried through so results are readable.
    """

    projections: np.ndarray
    stddevs: np.ndarray
    salaries: np.ndarray
    ownership: np.ndarray
    positions: np.ndarray
    keys: Mapping[str, np.ndarray]
    names: tuple[str, ...] | None = None

    def __len__(self) -> int:
        """Number of players."""
        return int(self.projections.shape[0])

    def __post_init__(self) -> None:
        """Check every column is the same length and one-dimensional."""
        n = len(self)
        columns: dict[str, np.ndarray] = {
            "projections": self.projections,
            "stddevs": self.stddevs,
            "salaries": self.salaries,
            "ownership": self.ownership,
            "positions": self.positions,
            **{f"keys[{k!r}]": v for k, v in self.keys.items()},
        }
        for name, array in columns.items():
            if array.ndim != 1:
                msg = f"{name} must be one-dimensional, got shape {array.shape}"
                raise ValueError(msg)
            if array.shape[0] != n:
                msg = (
                    f"{name} has {array.shape[0]} entries but projections has {n}; "
                    f"every player column must be parallel"
                )
                raise ValueError(msg)
        if self.names is not None and len(self.names) != n:
            msg = f"names has {len(self.names)} entries but the pool has {n} players"
            raise ValueError(msg)

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        spec: RosterSpec,
        *,
        key_fields: Sequence[str] | None = None,
    ) -> PlayerPool:
        """Build a pool from a sequence of per-player mappings.

        Each record needs `positions` (an iterable of position names), `salary`,
        and `projection`. `stddev`, `ownership`, and `name` are optional.

        Group keys are read from the record field named by the constraint's `key`
        and are mapped to dense integers, since string keys cannot cross into the
        kernel. A missing or `None` key becomes `-1`, meaning uncapped.

        Args:
            records: One mapping per player.
            spec: Specification whose positions encode the masks and whose group
                constraints determine which keys are needed.
            key_fields: Key names to extract. Defaults to those the spec's groups
                reference.

        Returns:
            A pool with columns in the order the records were given.

        Raises:
            KeyError: If a record is missing a required field, or names a position
                the specification does not define.
        """
        required = ("salary", "projection", "positions")
        for i, record in enumerate(records):
            missing = [f for f in required if f not in record]
            if missing:
                msg = f"record {i} is missing required field(s): {missing}"
                raise KeyError(msg)

        wanted = tuple(key_fields) if key_fields is not None else spec.group_keys
        keys: dict[str, np.ndarray] = {}
        for key in wanted:
            # Dense integer ids, assigned in first-seen order so the encoding is
            # deterministic for a given record order.
            seen: dict[Any, int] = {}
            encoded = np.empty(len(records), dtype=np.int32)
            for i, record in enumerate(records):
                raw = record.get(key)
                if raw is None:
                    encoded[i] = -1
                    continue
                if raw not in seen:
                    seen[raw] = len(seen)
                encoded[i] = seen[raw]
            keys[key] = encoded

        return cls(
            projections=np.asarray([r["projection"] for r in records], dtype=np.float64),
            stddevs=np.asarray([r.get("stddev", 0.0) for r in records], dtype=np.float64),
            salaries=np.asarray([r["salary"] for r in records], dtype=np.int64),
            ownership=np.asarray([r.get("ownership", 0.0) for r in records], dtype=np.float64),
            positions=np.asarray([spec.mask_for(r["positions"]) for r in records], dtype=np.uint32),
            keys=keys,
            names=tuple(str(r.get("name", f"player-{i}")) for i, r in enumerate(records)),
        )

    def salary_of(self, lineups: np.ndarray) -> np.ndarray:
        """Total salary of each lineup in an `(n, roster_size)` index array."""
        return np.asarray(self.salaries[lineups].sum(axis=1))

    def projection_of(self, lineups: np.ndarray) -> np.ndarray:
        """Total projection of each lineup in an `(n, roster_size)` index array."""
        return np.asarray(self.projections[lineups].sum(axis=1))

    def names_of(self, lineup: np.ndarray) -> list[str]:
        """Player names for one lineup, in slot order."""
        if self.names is None:
            return [str(i) for i in lineup]
        return [self.names[int(i)] for i in lineup]
