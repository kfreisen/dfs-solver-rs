"""The player pool and its construction from records."""

from __future__ import annotations

import numpy as np
import pytest
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.spec import GroupConstraint, RosterSpec, Slot

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
    assert pool.salary_of(lineups).tolist() == [30, 40]
    assert pool.projection_of(lineups).tolist() == [3.0, 4.0]


def test_names_of_returns_slot_order() -> None:
    pool = make_pool(names=("first", "second"))
    assert pool.names_of(np.array([1, 0])) == ["second", "first"]


def test_names_of_falls_back_to_indices() -> None:
    assert make_pool().names_of(np.array([1, 0])) == ["1", "0"]
