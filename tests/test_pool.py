"""The player pool and its construction from records."""

from __future__ import annotations

import numpy as np
import pytest
from dfs_solver.pool import PlayerPool
from dfs_solver.spec import ConflictRule, GroupConstraint, RosterSpec, Slot

SPEC = RosterSpec(
    positions=("P", "C"),
    slots=(Slot("C", ("C",)), Slot("P", ("P",))),
    salary_cap=10_000,
    groups=(GroupConstraint(key="team", max_count=1),),
)

RECORDS: list[dict[str, object]] = [
    {"name": "a", "positions": ("C",), "salary": 3000, "projection": 8.0, "team": "X"},
    {"name": "b", "positions": ("P",), "salary": 4000, "projection": 9.0, "team": "Y"},
]


def test_from_records_builds_parallel_columns() -> None:
    pool = PlayerPool.from_records(RECORDS, SPEC)
    assert len(pool) == 2
    assert pool.salaries.tolist() == [3000, 4000]
    assert pool.projections.tolist() == [8.0, 9.0]
    assert pool.names == ("a", "b")


def test_from_records_encodes_positions_against_the_spec() -> None:
    pool = PlayerPool.from_records(RECORDS, SPEC)
    # P is bit 0, C is bit 1, per SPEC.positions.
    assert pool.positions.tolist() == [0b10, 0b01]


def test_from_records_densifies_string_keys() -> None:
    pool = PlayerPool.from_records(RECORDS, SPEC)
    # Ids are assigned in first-seen order, so the encoding is deterministic for a
    # given record order rather than depending on hash iteration.
    assert pool.keys["team"].tolist() == [0, 1]


def test_repeated_key_values_share_an_id() -> None:
    records = [{**r, "team": "SAME"} for r in RECORDS]
    pool = PlayerPool.from_records(records, SPEC)
    assert pool.keys["team"].tolist() == [0, 0]


def test_a_missing_key_becomes_uncapped() -> None:
    records = [dict(RECORDS[0]), {k: v for k, v in RECORDS[1].items() if k != "team"}]
    pool = PlayerPool.from_records(records, SPEC)
    # -1 means "belongs to no group", which group caps skip entirely.
    assert pool.keys["team"].tolist() == [0, -1]


def test_optional_fields_default() -> None:
    pool = PlayerPool.from_records(RECORDS, SPEC)
    assert pool.stddevs.tolist() == [0.0, 0.0]
    assert pool.ownership.tolist() == [0.0, 0.0]


def test_missing_required_field_names_the_record_and_the_field() -> None:
    bad = [{"positions": ("C",), "salary": 1}]
    with pytest.raises(KeyError, match="record 0 is missing"):
        PlayerPool.from_records(bad, SPEC)


def test_an_unknown_position_is_rejected() -> None:
    bad = [{"positions": ("QB",), "salary": 1, "projection": 1.0}]
    with pytest.raises(KeyError, match="unknown position"):
        PlayerPool.from_records(bad, SPEC)


def test_explicit_key_fields_override_the_spec() -> None:
    records = [{**r, "game": "G1"} for r in RECORDS]
    pool = PlayerPool.from_records(records, SPEC, key_fields=["game"])
    assert set(pool.keys) == {"game"}


def test_unnamed_players_get_positional_names() -> None:
    records = [{k: v for k, v in r.items() if k != "name"} for r in RECORDS]
    pool = PlayerPool.from_records(records, SPEC)
    assert pool.names == ("player-0", "player-1")


def make_pool(**overrides: object) -> PlayerPool:
    kwargs: dict[str, object] = {
        "projections": np.array([1.0, 2.0]),
        "stddevs": np.array([1.0, 1.0]),
        "salaries": np.array([10, 20], dtype=np.int64),
        "ownership": np.array([0.0, 0.0]),
        "positions": np.array([1, 2], dtype=np.uint32),
        "keys": {"team": np.array([0, 1], dtype=np.int32)},
    }
    kwargs.update(overrides)
    return PlayerPool(**kwargs)  # type: ignore[arg-type]


def test_a_short_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="every player column must be parallel"):
        make_pool(stddevs=np.array([1.0]))


def test_a_short_key_column_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"keys\['team'\]"):
        make_pool(keys={"team": np.array([0], dtype=np.int32)})


def test_a_two_dimensional_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be one-dimensional"):
        make_pool(salaries=np.zeros((2, 2), dtype=np.int64))


def test_names_of_the_wrong_length_are_rejected() -> None:
    with pytest.raises(ValueError, match="names has 1 entries"):
        make_pool(names=("only-one",))


def test_salary_and_projection_of_sum_across_slots() -> None:
    pool = make_pool()
    lineups = np.array([[0, 1], [1, 1]])
    assert pool.salary_of(lineups, SPEC).tolist() == [30, 40]
    assert pool.projection_of(lineups, SPEC).tolist() == [3.0, 4.0]


def test_salary_and_projection_of_apply_slot_multipliers() -> None:
    # A captain in slot 0 costs and scores 1.5x; the other slot is untouched.
    spec = RosterSpec(
        positions=("P", "C"),
        slots=(
            Slot("CPT", ("C",), score_multiplier=1.5, salary_multiplier=1.5),
            Slot("P", ("P",)),
        ),
        salary_cap=10_000,
    )
    pool = make_pool()
    lineups = np.array([[0, 1]])
    assert pool.salary_of(lineups, spec).tolist() == [15 + 20]
    assert pool.projection_of(lineups, spec).tolist() == [1.0 * 1.5 + 2.0]


def test_salary_of_rounds_half_away_from_zero_like_the_kernel() -> None:
    # 3 * 1.5 is 4.5 exactly. numpy's default rint would give 4 here, and the
    # kernel would have charged 5 — a disagreement about whether a lineup fits.
    spec = RosterSpec(
        positions=("P", "C"),
        slots=(
            Slot("CPT", ("C",), salary_multiplier=1.5),
            Slot("P", ("P",)),
        ),
        salary_cap=10_000,
    )
    pool = make_pool(salaries=np.array([3, 20], dtype=np.int64))
    assert pool.salary_of(np.array([[0, 1]]), spec).tolist() == [5 + 20]


def test_names_of_returns_slot_order() -> None:
    pool = make_pool(names=("first", "second"))
    assert pool.names_of(np.array([1, 0])) == ["second", "first"]


def test_names_of_falls_back_to_indices() -> None:
    assert make_pool().names_of(np.array([1, 0])) == ["1", "0"]


def test_key_columns_share_one_id_space() -> None:
    # A conflict rule joins one column to another, so "X" must encode to the same
    # integer in `opponent` as it does in `team`. Encoding each column separately
    # would make the join match unrelated players.
    records = [
        {**RECORDS[0], "opponent": "Y"},
        {**RECORDS[1], "opponent": "X"},
    ]
    pool = PlayerPool.from_records(records, SPEC, key_fields=["team", "opponent"])
    assert pool.keys["team"].tolist() == [0, 1]
    assert pool.keys["opponent"].tolist() == [1, 0]


def test_conflict_keys_are_extracted_without_being_named() -> None:
    spec = RosterSpec(
        positions=("P", "C"),
        slots=(Slot("C", ("C",)), Slot("P", ("P",))),
        salary_cap=10_000,
        conflicts=(ConflictRule(left_key="opponent", right_key="team"),),
    )
    records = [{**r, "opponent": "Z"} for r in RECORDS]
    pool = PlayerPool.from_records(records, spec)
    assert set(pool.keys) == {"team", "opponent"}


def test_conflict_pairs_are_empty_without_a_rule() -> None:
    pool = PlayerPool.from_records(RECORDS, SPEC)
    assert pool.conflict_pairs(SPEC).shape == (2, 0)


def test_conflict_pairs_resolve_a_key_join() -> None:
    spec = RosterSpec(
        positions=("P", "C"),
        slots=(Slot("C", ("C",)), Slot("P", ("P",))),
        salary_cap=10_000,
        conflicts=(ConflictRule(left_key="opponent", right_key="team", left_positions=("P",)),),
    )
    # Player a is a catcher on X; player b is a pitcher facing X.
    records = [
        {**RECORDS[0], "opponent": "Y"},
        {**RECORDS[1], "team": "Y", "opponent": "X"},
    ]
    pool = PlayerPool.from_records(records, spec)
    pairs = pool.conflict_pairs(spec)
    # Only the pitcher side generates the pair: left_positions restricts it.
    assert pairs.shape == (2, 1)
    assert pairs[:, 0].tolist() == [1, 0]


def test_conflict_pairs_skip_negative_keys() -> None:
    spec = RosterSpec(
        positions=("P", "C"),
        slots=(Slot("C", ("C",)), Slot("P", ("P",))),
        salary_cap=10_000,
        conflicts=(ConflictRule(left_key="opponent", right_key="team"),),
    )
    # Neither player declares an opponent, so the left side joins nothing.
    pool = PlayerPool.from_records(RECORDS, spec, key_fields=["team", "opponent"])
    assert pool.conflict_pairs(spec).shape == (2, 0)

    # And a player with no team is absent from the right side, so a rule that
    # would otherwise have matched them does not.
    records = [
        {**RECORDS[0], "opponent": "Y"},
        {k: v for k, v in RECORDS[1].items() if k != "team"},
    ]
    unteamed = PlayerPool.from_records(records, spec)
    assert unteamed.keys["team"].tolist()[1] == -1
    assert unteamed.conflict_pairs(spec).shape == (2, 0)


def test_conflict_pairs_report_a_missing_key_column() -> None:
    spec = RosterSpec(
        positions=("P", "C"),
        slots=(Slot("C", ("C",)), Slot("P", ("P",))),
        salary_cap=10_000,
        conflicts=(ConflictRule(left_key="opponent", right_key="team"),),
    )
    pool = PlayerPool.from_records(RECORDS, spec, key_fields=["team"])
    with pytest.raises(KeyError, match="conflict rule reads 'opponent'"):
        pool.conflict_pairs(spec)


def test_conflict_pairs_restrict_the_right_side_by_position() -> None:
    # Leaving right_positions open also forbids two opposing pitchers, which some
    # people want and some do not. Naming the hitter positions is how you say so.
    spec = RosterSpec(
        positions=("P", "C"),
        slots=(Slot("C", ("C",)), Slot("P", ("P",))),
        salary_cap=10_000,
        conflicts=(
            ConflictRule(
                left_key="opponent",
                right_key="team",
                left_positions=("P",),
                right_positions=("P",),
            ),
        ),
    )
    records = [
        {**RECORDS[0], "opponent": "Y"},
        {**RECORDS[1], "team": "Y", "opponent": "X"},
    ]
    # Player 0 is a catcher, so the pitcher-versus-pitcher rule matches nothing.
    pool = PlayerPool.from_records(records, spec)
    assert pool.conflict_pairs(spec).shape == (2, 0)


def test_conflict_pairs_drop_a_player_matching_themselves() -> None:
    # A rule joining a key to itself matches every player against themselves.
    # That is not an error to report — the no-duplicate-players rule already
    # covers it — so the pair is simply not generated.
    spec = RosterSpec(
        positions=("P", "C"),
        slots=(Slot("C", ("C",)), Slot("P", ("P",))),
        salary_cap=10_000,
        conflicts=(ConflictRule(left_key="team", right_key="team"),),
    )
    pool = PlayerPool.from_records(RECORDS, spec)
    # Each player is alone on their team, so only the self-pairs could match.
    assert pool.conflict_pairs(spec).shape == (2, 0)
