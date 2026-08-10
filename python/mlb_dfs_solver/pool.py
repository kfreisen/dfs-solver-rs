"""The player pool: who is available, and what is known about them.

Stored column-wise as NumPy arrays because that is the form the kernel borrows
directly. Building a pool from a list of records is a convenience
([`PlayerPool.from_records`][mlb_dfs_solver.pool.PlayerPool.from_records]); the arrays
are the real interface.

The pool holds no reference to a specification. That is why the methods reporting
what a lineup is worth take one: a player's salary and score depend on the slot
they were rostered in once showdown multipliers exist, and the pool alone cannot
know that.
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

        **All key columns share one id space.** Encoding each column separately
        would be tidier and is wrong: a conflict rule joins one column to another,
        and `"BOS"` has to mean the same integer in `opponent` as it does in
        `team` or the join matches unrelated players. Group caps are unaffected
        either way, since they only ever compare within a single column.

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

        wanted = (
            tuple(key_fields)
            if key_fields is not None
            else tuple(dict.fromkeys(spec.group_keys + spec.conflict_keys))
        )
        keys: dict[str, np.ndarray] = {}
        # One dictionary across every column, so the same value encodes to the
        # same id wherever it appears. Ids are assigned in first-seen order, which
        # keeps the encoding deterministic for a given record order.
        seen: dict[Any, int] = {}
        for key in wanted:
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

    def conflict_pairs(self, spec: RosterSpec) -> np.ndarray:
        """Resolve the specification's conflict rules against this pool.

        Each rule joins one player's `left_key` to another's `right_key`, so this
        buckets the pool by the right-hand key and then walks the left-hand side
        looking each value up — `O(n + pairs)` rather than the `O(n^2)` the rule
        reads like.

        Negative key values match nothing, consistent with group constraints,
        where a negative key means "belongs to no group".

        Args:
            spec: Specification whose `conflicts` are resolved. Its `positions`
                encode the eligibility filters.

        Returns:
            A `(2, n_pairs)` `uint32` array of player indices. Empty when the
            specification declares no conflicts, which is the default.

        Raises:
            KeyError: If a rule reads a key the pool does not carry.
        """
        if not spec.conflicts:
            return np.empty((2, 0), dtype=np.uint32)

        for key in spec.conflict_keys:
            if key not in self.keys:
                available = ", ".join(sorted(self.keys)) or "none"
                msg = (
                    f"a conflict rule reads {key!r} but the pool has no such key "
                    f"column (available: {available}). Pass it in the records, or "
                    f"name it in from_records(key_fields=...)."
                )
                raise KeyError(msg)

        left: list[int] = []
        right: list[int] = []
        for rule in spec.conflicts:
            left_mask = spec.mask_for(rule.left_positions) if rule.left_positions else None
            right_mask = spec.mask_for(rule.right_positions) if rule.right_positions else None
            left_values = self.keys[rule.left_key]
            right_values = self.keys[rule.right_key]

            by_value: dict[int, list[int]] = {}
            for j in range(len(self)):
                value = int(right_values[j])
                if value < 0:
                    continue
                if right_mask is not None and not int(self.positions[j]) & right_mask:
                    continue
                by_value.setdefault(value, []).append(j)

            for i in range(len(self)):
                value = int(left_values[i])
                if value < 0:
                    continue
                if left_mask is not None and not int(self.positions[i]) & left_mask:
                    continue
                for j in by_value.get(value, ()):
                    if i == j:
                        continue
                    left.append(i)
                    right.append(j)

        return np.asarray([left, right], dtype=np.uint32).reshape(2, -1)

    def salary_of(self, lineups: np.ndarray, spec: RosterSpec) -> np.ndarray:
        """Total salary of each lineup in an `(n, roster_size)` index array.

        `spec` is required rather than optional because a showdown captain costs
        1.5x and a total computed without it is not merely approximate, it is a
        different number from the one the kernel checked against the cap.
        """
        base = self.salaries[lineups]
        multipliers = np.asarray(spec.salary_multipliers(), dtype=np.float64)
        if not (multipliers != 1.0).any():
            return np.asarray(base.sum(axis=1))
        # Half away from zero, matching `scaled_salary` and the kernel. Columns
        # whose multiplier is exactly 1.0 keep their integer value rather than
        # making a float round trip, so a large salary cannot drift by an ulp.
        product = base * multipliers
        rounded = np.sign(product) * np.floor(np.abs(product) + 0.5)
        scaled = np.where(multipliers == 1.0, base, rounded).astype(np.int64)
        return np.asarray(scaled.sum(axis=1))

    def projection_of(self, lineups: np.ndarray, spec: RosterSpec) -> np.ndarray:
        """Total projection of each lineup in an `(n, roster_size)` index array.

        This is where a score multiplier finally shows up. It cannot change which
        player construction puts in a captain slot — every candidate for that slot
        is scaled alike — but it very much changes what the lineup is worth.
        """
        weighted = self.projections[lineups] * np.asarray(spec.score_multipliers())
        return np.asarray(weighted.sum(axis=1))

    def names_of(self, lineup: np.ndarray) -> list[str]:
        """Player names for one lineup, in slot order."""
        if self.names is None:
            return [str(i) for i in lineup]
        return [self.names[int(i)] for i in lineup]
