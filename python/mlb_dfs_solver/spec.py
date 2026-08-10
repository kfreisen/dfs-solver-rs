"""Describing what makes a lineup legal.

A [`RosterSpec`][mlb_dfs_solver.spec.RosterSpec] is the sport-independent description of
a contest: which slots a roster has, which positions may fill each, what the salary
budget is, and how many players may share a key such as a team.

The two rules a daily-fantasy contest usually states as separate features — "at most
6 players from one team" and "at most 5 *hitters* from one team" — are the same
constraint here, differing only in which slots they count. That collapse is the
reason this generalizes past the sport it came from.

Two shapes do not collapse into a group cap, and each gets its own mechanism:

* A **showdown captain** is worth more and costs more than the same player in a
  flex slot, so [`Slot`][mlb_dfs_solver.spec.Slot] carries score and salary
  multipliers.
* **"No hitters against my starting pitcher"** depends on a *pair* of players — the
  pitcher's opponent matching the hitter's team — which is a join rather than a
  grouping. [`ConflictRule`][mlb_dfs_solver.spec.ConflictRule] expresses it, and
  nothing turns it on unless you ask: it is a preference, not a contest rule, and
  a contrarian who wants that correlation is entitled to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "ConflictRule",
    "GroupConstraint",
    "RosterSpec",
    "Slot",
    "positions_to_mask",
    "scaled_salary",
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


def scaled_salary(base: int, multiplier: float) -> int:
    """Apply a slot's salary multiplier to a base salary.

    Rounded half away from zero, and short-circuited at exactly `1.0` so an
    ordinary slot cannot drift by a floating-point ulp.

    The rounding rule is part of the contract, not an implementation detail: the
    Rust kernel does the same thing when it checks the cap, and if the two
    disagreed a lineup the kernel believes is legal would read as over the cap
    here. `round()` would not do — it rounds half to even.
    """
    if multiplier == 1.0:
        return base
    scaled = base * multiplier
    return math.floor(abs(scaled) + 0.5) * (1 if scaled >= 0 else -1)


@dataclass(frozen=True, slots=True)
class Slot:
    """A run of interchangeable roster positions.

    Attributes:
        name: Label used in error messages and lineup output.
        eligible: Position names a player must have at least one of.
        count: How many slots of this kind the roster has.
        score_multiplier: What this slot multiplies its occupant's score by. `1.5`
            is a DraftKings showdown captain.

            This deliberately does **not** affect which player construction puts
            here. Every candidate for a slot is scaled by the same factor, so a
            non-negative multiplier cannot reorder them. It changes what the
            finished lineup is worth, which is what
            [`PlayerPool.projection_of`][mlb_dfs_solver.pool.PlayerPool.projection_of]
            reports and what anything ranking lineups against each other needs.
        salary_multiplier: What this slot multiplies its occupant's salary by.
            `1.5` is a DraftKings showdown captain; FanDuel's MVP leaves salary
            alone, so that is `1.0`.

            Unlike the score multiplier this changes construction throughout: the
            cap check, the cheapest-way-to-finish reservation, and salary repair
            all price a player by the slot they are being considered for.
    """

    name: str
    eligible: tuple[str, ...]
    count: int = 1
    score_multiplier: float = 1.0
    salary_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Reject a slot that can never be filled, or whose multipliers are junk."""
        if self.count < 1:
            msg = f"slot {self.name!r} has count {self.count}; remove it instead"
            raise ValueError(msg)
        if not self.eligible:
            msg = f"slot {self.name!r} lists no eligible positions, so nothing can fill it"
            raise ValueError(msg)
        for which, value in (
            ("score_multiplier", self.score_multiplier),
            ("salary_multiplier", self.salary_multiplier),
        ):
            # A NaN silently poisons every comparison it reaches, and a negative
            # salary multiplier would let a slot pay the caller — which breaks the
            # monotonicity the kernel's reservation bound assumes.
            if not math.isfinite(value) or value < 0.0:
                msg = f"slot {self.name!r} has {which} {value}; it must be finite and non-negative"
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
class ConflictRule:
    """Two players may not be rostered together when their keys match.

    This is the one rule shape a [`GroupConstraint`][mlb_dfs_solver.spec.GroupConstraint]
    cannot express. A group cap counts players who share a value of *one* key; a
    conflict joins *two different* keys across a pair of players. "No hitters
    against my starting pitcher" is the pitcher's `opponent` matching the hitter's
    `team`, which is not a grouping at all.

    Nothing enables this for you. Presets ship without conflicts, because avoiding
    a pitcher's opposing hitters is a preference and not a contest rule — a
    contrarian deliberately wants that correlation, and an optimizer that silently
    forbade it would be wrong for them. Opt in with `dataclasses.replace`:

    ```python
    from dataclasses import replace
    from mlb_dfs_solver.presets import DK_MLB_CLASSIC
    from mlb_dfs_solver.spec import ConflictRule

    spec = replace(
        DK_MLB_CLASSIC,
        conflicts=(ConflictRule(left_key="opponent", right_key="team", left_positions=("P",)),),
    )
    ```

    Attributes:
        left_key: Per-player key read from the first player of the pair.
        right_key: Per-player key read from the second. Matching `left_key`'s
            value against this one is what makes the pair a conflict. Negative key
            values match nothing, the same as everywhere else.
        left_positions: Restricts the rule's first side to players eligible at one
            of these positions. `None` means every player. For the MLB rule this
            is `("P",)`, so it is a *pitcher's* opponent that matters.
        right_positions: The same restriction for the second side. Leaving it
            `None` also forbids two opposing pitchers, which some people want and
            some do not — name the hitter positions if you do not.
    """

    left_key: str
    right_key: str
    left_positions: tuple[str, ...] | None = None
    right_positions: tuple[str, ...] | None = None


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
        conflicts: Pairs of players forbidden from sharing a lineup, expressed as
            key joins. Empty by default and never populated by a preset — see
            [`ConflictRule`][mlb_dfs_solver.spec.ConflictRule] for why the opt-in
            is explicit.
    """

    slots: tuple[Slot, ...]
    salary_cap: int
    positions: tuple[str, ...]
    salary_floor: int = 0
    groups: tuple[GroupConstraint, ...] = field(default_factory=tuple)
    conflicts: tuple[ConflictRule, ...] = field(default_factory=tuple)

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

        for rule in self.conflicts:
            unknown = (set(rule.left_positions or ()) | set(rule.right_positions or ())) - (
                known_positions
            )
            if unknown:
                msg = (
                    f"conflict rule on {rule.left_key!r}/{rule.right_key!r} names "
                    f"position(s) {sorted(unknown)} that are not in this "
                    f"specification's positions {list(self.positions)}"
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

    @property
    def conflict_keys(self) -> tuple[str, ...]:
        """Distinct key names the conflict rules read, in a stable order.

        Kept separate from `group_keys` because the two are consumed differently:
        group keys become a matrix the kernel indexes by column, while conflict
        keys are resolved to explicit player pairs before the boundary.
        """
        seen: dict[str, None] = {}
        for rule in self.conflicts:
            seen.setdefault(rule.left_key, None)
            seen.setdefault(rule.right_key, None)
        return tuple(seen)

    @property
    def has_multipliers(self) -> bool:
        """Whether any slot scales score or salary."""
        return any(
            slot.score_multiplier != 1.0 or slot.salary_multiplier != 1.0 for slot in self.slots
        )

    def slot_names(self) -> list[str]:
        """One entry per roster position, expanding multi-count slots."""
        names: list[str] = []
        for slot in self.slots:
            names.extend([slot.name] * slot.count)
        return names

    def score_multipliers(self) -> list[float]:
        """One score multiplier per roster position, in slot order.

        Parallel to `slot_names()`, so it lines up with a lineup's columns.
        """
        return [slot.score_multiplier for slot in self.slots for _ in range(slot.count)]

    def salary_multipliers(self) -> list[float]:
        """One salary multiplier per roster position, in slot order."""
        return [slot.salary_multiplier for slot in self.slots for _ in range(slot.count)]

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
