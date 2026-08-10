"""The roster specification and its validation.

Most of these test rejections. That is deliberate: every one of these mistakes,
left unchecked, surfaces later as "no valid lineups found", which tells the caller
nothing about what they got wrong.
"""

from __future__ import annotations

import pytest
from mlb_dfs_solver.spec import GroupConstraint, RosterSpec, Slot, positions_to_mask


def test_positions_to_mask_sets_one_bit_per_position() -> None:
    index = {"P": 0, "C": 1, "OF": 2}
    assert positions_to_mask(["P"], index) == 0b001
    assert positions_to_mask(["C", "OF"], index) == 0b110
    assert positions_to_mask([], index) == 0


def test_positions_to_mask_rejects_an_unknown_position() -> None:
    # Dropping it silently would make a player quietly ineligible, which shows up
    # much later as an inexplicably thin pool.
    with pytest.raises(KeyError, match="unknown position 'XX'"):
        positions_to_mask(["XX"], {"P": 0})


def test_slot_rejects_a_zero_count() -> None:
    with pytest.raises(ValueError, match="remove it instead"):
        Slot("C", ("C",), count=0)


def test_slot_rejects_empty_eligibility() -> None:
    with pytest.raises(ValueError, match="nothing can fill it"):
        Slot("C", ())


def test_group_constraint_rejects_a_zero_cap() -> None:
    with pytest.raises(ValueError, match="forbids every lineup"):
        GroupConstraint(key="team", max_count=0)


def simple(**overrides: object) -> RosterSpec:
    """A minimal valid spec, with fields replaceable per test."""
    kwargs: dict[str, object] = {
        "positions": ("P", "C"),
        "slots": (Slot("C", ("C",)), Slot("P", ("P",), count=2)),
        "salary_cap": 100,
        "salary_floor": 0,
        "groups": (),
    }
    kwargs.update(overrides)
    return RosterSpec(**kwargs)  # type: ignore[arg-type]


def test_roster_size_counts_every_slot() -> None:
    assert simple().roster_size == 3


def test_slot_names_expands_multi_count_slots() -> None:
    assert simple().slot_names() == ["C", "P", "P"]


def test_position_index_follows_declaration_order() -> None:
    assert simple().position_index == {"P": 0, "C": 1}


def test_group_keys_are_distinct_and_ordered() -> None:
    spec = simple(
        groups=(
            GroupConstraint(key="team", max_count=2),
            GroupConstraint(key="game", max_count=2),
            GroupConstraint(key="team", max_count=1, slots=("C",)),
        )
    )
    assert spec.group_keys == ("team", "game")


def test_slot_mask_for_none_covers_every_slot() -> None:
    assert simple().slot_mask_for(None) == 0b11


def test_slot_mask_for_named_slots() -> None:
    assert simple().slot_mask_for(("P",)) == 0b10


def test_mask_for_encodes_against_this_spec() -> None:
    assert simple().mask_for(("C",)) == 0b10


def test_a_spec_needs_at_least_one_slot() -> None:
    with pytest.raises(ValueError, match="at least one slot"):
        simple(slots=())


def test_duplicate_slot_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate slot names"):
        simple(slots=(Slot("C", ("C",)), Slot("C", ("P",))))


def test_duplicate_position_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate position names"):
        simple(positions=("P", "P"))


def test_a_floor_above_the_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="no lineup can be valid"):
        simple(salary_cap=10, salary_floor=20)


def test_a_slot_naming_an_unknown_position_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in this specification's positions"):
        simple(slots=(Slot("X", ("QB",)),))


def test_a_group_naming_an_unknown_slot_is_rejected() -> None:
    with pytest.raises(ValueError, match="do not exist"):
        simple(groups=(GroupConstraint(key="team", max_count=1, slots=("DST",)),))


def test_too_many_slot_groups_are_rejected() -> None:
    # Group constraints address slots with a 64-bit mask, so 65 groups cannot be
    # represented. Better to say so than to silently ignore the overflow.
    slots = tuple(Slot(f"S{i}", ("P",)) for i in range(65))
    with pytest.raises(ValueError, match="64-bit mask"):
        simple(slots=slots)


def test_too_many_positions_are_rejected() -> None:
    positions = tuple(f"P{i}" for i in range(33))
    with pytest.raises(ValueError, match="exceeds the limit"):
        simple(positions=positions, slots=(Slot("S", ("P0",)),))


def test_a_spec_is_immutable() -> None:
    spec = simple()
    with pytest.raises(AttributeError):
        spec.salary_cap = 1  # type: ignore[misc]
