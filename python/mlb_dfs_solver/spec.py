"""Describing what makes a lineup legal.

A [`RosterSpec`][mlb_dfs_solver.spec.RosterSpec] is the sport-independent description of
a contest: which slots a roster has, which positions may fill each, what the salary
budget is, and how many players may share a key such as a team.

The two rules a daily-fantasy contest usually states as separate features — "at most
6 players from one team" and "at most 5 *hitters* from one team" — are the same
constraint here, differing only in which slots they count. That collapse is the
reason this generalizes past the sport it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "GroupConstraint",
    "RosterSpec",
    "Slot",
    "positions_to_mask",
]

# Slot membership is addressed with a 64-bit mask on the Rust side.
MAX_SLOTS = 64
# Positions are addressed with a 32-bit mask.
MAX_POSITIONS = 32


def positions_to_mask(positions: Iterable[str], index: Mapping[str, int]) -> int:
    """Turn position names into a bitmask.

    Args:
        positions: Position names, e.g. `("1B", "OF")` for a player eligible at both.
        index: Maps each position name to its bit number.

    Returns:
        A bitmask with one bit set per named position.

    Raises:
        KeyError: If a position is not in `index`. Silently dropping an unknown
            position would make a player quietly ineligible, which surfaces much
            later as an inexplicably thin pool.
    """
    mask = 0
    for name in positions:
        try:
            bit = index[name]
        except KeyError:
            known = ", ".join(sorted(index))
            msg = f"unknown position {name!r}; this specification knows: {known}"
            raise KeyError(msg) from None
        mask |= 1 << bit
    return mask


@dataclass(frozen=True, slots=True)
class Slot:
    """A run of interchangeable roster positions.

    Attributes:
        name: Label used in error messages and lineup output.
        eligible: Position names a player must have at least one of.
        count: How many slots of this kind the roster has.
    """

    name: str
    eligible: tuple[str, ...]
    count: int = 1

    def __post_init__(self) -> None:
        """Reject a slot that can never be filled."""
        if self.count < 1:
            msg = f"slot {self.name!r} has count {self.count}; remove it instead"
            raise ValueError(msg)
        if not self.eligible:
            msg = f"slot {self.name!r} lists no eligible positions, so nothing can fill it"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GroupConstraint:
    """A cap on how many selected players may share a key.

    Attributes:
        key: Name of the per-player key this counts, e.g. `"team"`.
        max_count: Maximum players sharing one value of that key.
        slots: Slot names that count toward the cap. `None` means every slot.
            Restricting this is how "at most 5 hitters from one team" is expressed
            without a special case.
    """

    key: str
    max_count: int
    slots: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Reject a cap that forbids every lineup."""
        if self.max_count < 1:
            msg = (
                f"group constraint on {self.key!r} has max_count {self.max_count}, "
                f"which forbids every lineup; exclude those players from the pool instead"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RosterSpec:
    """Everything that makes a lineup legal.

    Attributes:
        slots: Slot groups **in fill order**. Order matters: construction fills
            these left to right, so scarce positions belong first. A builder that
            spends its budget on outfielders before looking for a catcher fails to
            find one far more often.
        salary_cap: Total salary a lineup may not exceed.
        salary_floor: Minimum total salary. Zero disables the floor.
        groups: Group caps.
        positions: Position names, in bit order. Determines the mask encoding.
    """

    slots: tuple[Slot, ...]
    salary_cap: int
    positions: tuple[str, ...]
    salary_floor: int = 0
    groups: tuple[GroupConstraint, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the specification as a whole.

        Everything checked here would otherwise surface as "no valid lineups
        found", which is the least actionable failure a solver can produce.
        """
        if not self.slots:
            msg = "a roster specification needs at least one slot"
            raise ValueError(msg)
        if len(self.slots) > MAX_SLOTS:
            msg = (
                f"{len(self.slots)} slot groups exceeds the limit of {MAX_SLOTS}; "
                f"group constraints address slots with a 64-bit mask"
            )
            raise ValueError(msg)
        if len(self.positions) > MAX_POSITIONS:
            msg = f"{len(self.positions)} positions exceeds the limit of {MAX_POSITIONS}"
            raise ValueError(msg)
        if len(set(self.positions)) != len(self.positions):
            msg = f"duplicate position names: {self.positions}"
            raise ValueError(msg)

        slot_names = [s.name for s in self.slots]
        if len(set(slot_names)) != len(slot_names):
            msg = f"duplicate slot names: {slot_names}"
            raise ValueError(msg)

        if self.salary_floor > self.salary_cap:
            msg = (
                f"salary floor {self.salary_floor} exceeds cap {self.salary_cap}, "
                f"so no lineup can be valid"
            )
            raise ValueError(msg)

        known_positions = set(self.positions)
        for slot in self.slots:
            unknown = set(slot.eligible) - known_positions
            if unknown:
                msg = (
                    f"slot {slot.name!r} lists position(s) {sorted(unknown)} that are "
                    f"not in this specification's positions {list(self.positions)}"
                )
                raise ValueError(msg)

        known_slots = set(slot_names)
        for group in self.groups:
            if group.slots is None:
                continue
            unknown_slots = set(group.slots) - known_slots
            if unknown_slots:
                msg = (
                    f"group constraint on {group.key!r} names slot(s) "
                    f"{sorted(unknown_slots)} that do not exist"
                )
                raise ValueError(msg)

    @property
    def roster_size(self) -> int:
        """Total number of players a complete lineup holds."""
        return sum(slot.count for slot in self.slots)

    @property
    def position_index(self) -> dict[str, int]:
        """Map each position name to its bit number."""
        return {name: i for i, name in enumerate(self.positions)}

    @property
    def group_keys(self) -> tuple[str, ...]:
        """Distinct key names the group constraints read, in a stable order."""
        seen: dict[str, None] = {}
        for group in self.groups:
            seen.setdefault(group.key, None)
        return tuple(seen)

    def slot_names(self) -> list[str]:
        """One entry per roster position, expanding multi-count slots."""
        names: list[str] = []
        for slot in self.slots:
            names.extend([slot.name] * slot.count)
        return names

    def mask_for(self, positions: Iterable[str]) -> int:
        """Encode position names as a bitmask under this specification."""
        return positions_to_mask(positions, self.position_index)

    def slot_mask_for(self, slots: Sequence[str] | None) -> int:
        """Encode slot names as a bitmask over slot-group index.

        `None` means every slot group, which is the common case.
        """
        if slots is None:
            return (1 << len(self.slots)) - 1
        index = {slot.name: i for i, slot in enumerate(self.slots)}
        mask = 0
        for name in slots:
            mask |= 1 << index[name]
        return mask
