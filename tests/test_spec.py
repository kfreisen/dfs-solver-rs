"""The roster specification and its validation.

Most of these test rejections. That is deliberate: every one of these mistakes,
left unchecked, surfaces later as "no valid lineups found", which tells the caller
nothing about what they got wrong.
"""

from __future__ import annotations

import pytest
from dfs_solver.spec import (
    UNCAPPED,
    ConflictRule,
    GroupConstraint,
    RosterSpec,
    Slot,
    positions_to_mask,
    scaled_salary,
)


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


def test_a_slot_defaults_to_neutral_multipliers() -> None:
    slot = Slot("C", ("C",))
    assert slot.score_multiplier == 1.0
    assert slot.salary_multiplier == 1.0


def test_a_nan_multiplier_is_rejected() -> None:
    # NaN poisons every comparison it reaches, so it must not survive into the
    # kernel's cap arithmetic.
    with pytest.raises(ValueError, match="finite and non-negative"):
        Slot("CPT", ("C",), score_multiplier=float("nan"))


def test_a_negative_multiplier_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        Slot("CPT", ("C",), salary_multiplier=-1.0)


def test_an_infinite_multiplier_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        Slot("CPT", ("C",), salary_multiplier=float("inf"))


def test_scaled_salary_returns_the_base_exactly_at_one() -> None:
    # Not merely close: the cap check is an integer compare, and a float round
    # trip could shift a large salary by an ulp.
    for base in (0, 1, 3_700, 50_000, 2**53 + 1):
        assert scaled_salary(base, 1.0) == base


def test_scaled_salary_rounds_half_away_from_zero() -> None:
    # 3_701 * 1.5 is 5_551.5 exactly. Python's round() is half-to-even and would
    # give 5_552 here but 5_550 next door, disagreeing with the kernel.
    assert scaled_salary(3_701, 1.5) == 5_552
    assert scaled_salary(3_703, 1.5) == 5_555
    assert scaled_salary(3_700, 1.5) == 5_550
    assert scaled_salary(1_000, 0.0) == 0


def test_multipliers_are_reported_per_roster_slot() -> None:
    spec = simple(
        slots=(
            Slot("CPT", ("P",), score_multiplier=1.5, salary_multiplier=1.5),
            Slot("FLEX", ("P",), count=2),
        )
    )
    assert spec.score_multipliers() == [1.5, 1.0, 1.0]
    assert spec.salary_multipliers() == [1.5, 1.0, 1.0]
    assert len(spec.salary_multipliers()) == spec.roster_size


def test_has_multipliers_distinguishes_a_showdown() -> None:
    assert not simple().has_multipliers
    assert simple(slots=(Slot("CPT", ("P",), score_multiplier=1.5),)).has_multipliers


def test_a_spec_declares_no_conflicts_by_default() -> None:
    # The opt-in is the point: forbidding a pitcher's opposing hitters is a
    # strategy, and imposing it silently would be wrong for a contrarian.
    assert simple().conflicts == ()
    assert simple().conflict_keys == ()


def test_conflict_keys_are_deduplicated_in_a_stable_order() -> None:
    spec = simple(
        conflicts=(
            ConflictRule(left_key="opponent", right_key="team"),
            ConflictRule(left_key="team", right_key="game"),
        )
    )
    assert spec.conflict_keys == ("opponent", "team", "game")


def test_a_conflict_rule_naming_an_unknown_position_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in this specification's positions"):
        simple(
            conflicts=(ConflictRule(left_key="opponent", right_key="team", left_positions=("QB",)),)
        )


def test_a_constraint_defaults_to_uncapped_with_no_minimums() -> None:
    with pytest.raises(ValueError, match="constrains nothing"):
        GroupConstraint(key="team")


def test_a_negative_minimum_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative min_distinct"):
        GroupConstraint(key="team", min_distinct=-1)


def test_a_stack_above_its_own_cap_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot both hold"):
        GroupConstraint(key="team", max_count=3, min_stack=4)


def test_an_uncapped_constraint_reports_the_sentinel() -> None:
    # The kernel takes a u32, so "no ceiling" has to be a number rather than None.
    assert GroupConstraint(key="team", min_distinct=2).cap == UNCAPPED
    assert GroupConstraint(key="team", max_count=6).cap == 6


def test_a_minimum_larger_than_the_slots_it_counts_is_rejected() -> None:
    # Three slots cannot show four distinct teams. Caught here because the
    # alternative is an empty result the caller cannot distinguish from a thin
    # slate.
    with pytest.raises(ValueError, match="counts only 3 roster slot"):
        simple(groups=(GroupConstraint(key="team", min_distinct=4),))


def test_a_minimum_is_measured_against_only_the_slots_it_names() -> None:
    with pytest.raises(ValueError, match="counts only 1 roster slot"):
        simple(groups=(GroupConstraint(key="team", min_stack=2, slots=("C",)),))
    # The same minimum against both slot groups is fine.
    simple(groups=(GroupConstraint(key="team", min_stack=2, slots=("C", "P")),))


def test_group_keys_include_minimum_only_constraints() -> None:
    spec = simple(groups=(GroupConstraint(key="game", min_distinct=2),))
    assert spec.group_keys == ("game",)
