"""The Python surface of lineup construction.

Algorithm behavior is covered by the Rust unit tests and by `test_parity.py`.
What is tested here is the wrapper: argument validation, encoding, and the
determinism guarantee as seen from Python.
"""

from __future__ import annotations

from dataclasses import replace

import mlb_dfs_solver
import numpy as np
import pytest
from mlb_dfs_solver import CONTRARIAN, STANDARD, JitterProfile, build_lineups
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.presets import DK_NFL_SHOWDOWN
from mlb_dfs_solver.spec import ConflictRule, GroupConstraint, RosterSpec, Slot


def test_returns_indices_shaped_by_the_roster(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=10, seed=1)
    assert lineups.ndim == 2
    assert lineups.shape[1] == tiny_spec.roster_size
    assert lineups.dtype == np.int64
    assert lineups.min() >= 0
    assert lineups.max() < len(tiny_pool)


def test_same_seed_same_output(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    a = build_lineups(tiny_pool, tiny_spec, num_lineups=25, seed=4)
    b = build_lineups(tiny_pool, tiny_spec, num_lineups=25, seed=4)
    np.testing.assert_array_equal(a, b)


def test_different_seed_different_output(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    a = build_lineups(tiny_pool, tiny_spec, num_lineups=25, seed=4)
    b = build_lineups(tiny_pool, tiny_spec, num_lineups=25, seed=5)
    assert not np.array_equal(a, b)


def test_chunks_change_the_result(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    """Chunk count is documented as part of the reproducibility contract.

    If it did not change the output, the contract would be quietly stronger than
    documented — and someone would come to rely on the stronger version.
    """
    a = build_lineups(tiny_pool, tiny_spec, num_lineups=25, seed=4, chunks=4)
    b = build_lineups(tiny_pool, tiny_spec, num_lineups=25, seed=4, chunks=32)
    assert not np.array_equal(a, b)


def test_zero_lineups_returns_an_empty_array(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=0, seed=1)
    assert lineups.shape == (0, tiny_spec.roster_size)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_lineups": -1}, "num_lineups must be non-negative"),
        ({"attempts_per_lineup": 0}, "attempts_per_lineup must be at least 1"),
        ({"chunks": 0}, "chunks must be at least 1"),
        ({"noise": -0.1}, "noise must be non-negative"),
        ({"profiles": []}, "profiles is empty"),
    ],
)
def test_invalid_arguments_are_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_lineups(tiny_pool, tiny_spec, **kwargs)  # type: ignore[arg-type]


def test_a_constrained_key_missing_from_the_pool_is_reported(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    spec = replace(tiny_spec, groups=(GroupConstraint(key="stadium", max_count=2),))
    with pytest.raises(KeyError, match="constrains 'stadium'"):
        build_lineups(tiny_pool, spec, num_lineups=5)


def test_a_spec_with_no_groups_needs_no_keys(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    spec = replace(tiny_spec, groups=())
    assert len(build_lineups(tiny_pool, spec, num_lineups=5, seed=1)) > 0


def test_custom_profiles_are_accepted(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    flat = JitterProfile(ceiling=(0.0, 0.0), leverage=(0.0, 0.0))
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=5, seed=1, profiles=[flat])
    assert len(lineups) > 0


def test_an_inverted_profile_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="inverted"):
        JitterProfile(ceiling=(1.0, 0.0), leverage=(0.0, 1.0))


def test_the_shipped_profiles_differ() -> None:
    # If these were equal, alternating them would buy nothing and the diversity
    # argument in the docs would be false.
    assert CONTRARIAN != STANDARD
    assert CONTRARIAN.leverage[0] > STANDARD.leverage[0]


def test_non_contiguous_input_is_accepted(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    """A strided view must work, because NumPy hands them out constantly.

    The wrapper copies to contiguous rather than letting the kernel refuse — a
    user slicing their own arrays should not have to know what C-contiguous means.
    """
    strided = replace(
        tiny_pool,
        projections=np.repeat(tiny_pool.projections, 2)[::2],
    )
    assert len(build_lineups(strided, tiny_spec, num_lineups=5, seed=1)) > 0


def test_active_isa_reports_a_known_path() -> None:
    assert mlb_dfs_solver.active_isa() in {"avx2", "scalar"}


def test_native_version_is_reported() -> None:
    assert mlb_dfs_solver.native_version()


def test_public_names_are_all_importable() -> None:
    # __all__ drifting from reality breaks `from mlb_dfs_solver import *` and, more
    # importantly, the documented surface.
    for name in mlb_dfs_solver.__all__:
        assert hasattr(mlb_dfs_solver, name), name


# --- Slot multipliers ----------------------------------------------------


def showdown_spec(cap: int = 30_000, *, score: float = 1.5, salary: float = 1.5) -> RosterSpec:
    """A single-game shape over the tiny fixture's positions."""
    return RosterSpec(
        positions=("P", "C", "OF"),
        slots=(
            Slot("CPT", ("P", "C", "OF"), score_multiplier=score, salary_multiplier=salary),
            Slot("FLEX", ("P", "C", "OF"), count=3),
        ),
        salary_cap=cap,
        salary_floor=0,
    )


def test_a_captain_slot_charges_its_multiplied_salary(tiny_pool: PlayerPool) -> None:
    spec = showdown_spec()
    lineups = build_lineups(tiny_pool, spec, num_lineups=40, seed=2)
    assert len(lineups) > 0
    salaries = tiny_pool.salary_of(lineups, spec)
    assert int(salaries.max()) <= spec.salary_cap
    # Independent of salary_of: the captain column really is dearer.
    for lineup in lineups.tolist():
        expected = round(int(tiny_pool.salaries[lineup[0]]) * 1.5) + sum(
            int(tiny_pool.salaries[i]) for i in lineup[1:]
        )
        assert expected <= spec.salary_cap


def test_a_multiplied_salary_binds_the_cap(tiny_pool: PlayerPool) -> None:
    # A cap the same roster clears at 1.0x and cannot at 1.5x. If the multiplier
    # were dropped anywhere in the fill, both would produce the same count.
    plain = build_lineups(tiny_pool, showdown_spec(14_000, salary=1.0), num_lineups=60, seed=2)
    multiplied = build_lineups(tiny_pool, showdown_spec(14_000), num_lineups=60, seed=2)
    assert len(plain) > len(multiplied)


def test_a_score_multiplier_does_not_change_the_selection(tiny_pool: PlayerPool) -> None:
    # Documented behaviour. Every candidate for a slot is scaled by the same
    # factor, so ordering cannot move — and a seed's output must not silently
    # change when someone edits a multiplier.
    plain = build_lineups(tiny_pool, showdown_spec(score=1.0, salary=1.0), num_lineups=30, seed=4)
    scaled = build_lineups(tiny_pool, showdown_spec(score=1.5, salary=1.0), num_lineups=30, seed=4)
    assert np.array_equal(plain, scaled)


def test_a_score_multiplier_changes_the_lineup_value(tiny_pool: PlayerPool) -> None:
    spec = showdown_spec(score=1.5, salary=1.0)
    plain = showdown_spec(score=1.0, salary=1.0)
    lineups = build_lineups(tiny_pool, spec, num_lineups=10, seed=4)
    assert len(lineups) > 0
    assert (tiny_pool.projection_of(lineups, spec) > tiny_pool.projection_of(lineups, plain)).all()


def test_the_showdown_preset_builds(mlb_pool: PlayerPool) -> None:
    # Not about MLB: the preset is the check that a multiplier-carrying spec
    # survives the whole path from data to kernel and back.
    records = [
        {
            "name": f"p{k}",
            "positions": ("QB" if k % 6 == 0 else "WR",),
            "salary": 4_000 + (k % 9) * 900,
            "projection": 6.0 + (k % 9),
            "stddev": 4.0,
        }
        for k in range(40)
    ]
    pool = PlayerPool.from_records(records, DK_NFL_SHOWDOWN)
    lineups = build_lineups(pool, DK_NFL_SHOWDOWN, num_lineups=50, seed=5)
    assert len(lineups) > 0
    assert lineups.shape[1] == 6
    assert int(pool.salary_of(lineups, DK_NFL_SHOWDOWN).max()) <= 50_000


# --- Conflicts -----------------------------------------------------------


def conflicted_spec(tiny_spec: RosterSpec) -> RosterSpec:
    """`tiny_spec` plus the "no hitters against my pitcher" rule."""
    return replace(
        tiny_spec,
        conflicts=(ConflictRule(left_key="opponent", right_key="team", left_positions=("P",)),),
    )


def opposed_pool(tiny_spec: RosterSpec) -> PlayerPool:
    """A pool carrying both a team and an opponent, four teams paired off."""
    records: list[dict[str, object]] = []
    for position in ("P", "C", "OF"):
        for k in range(8):
            team = k % 4
            records.append(
                {
                    "name": f"{position}{k}",
                    "positions": (position,),
                    "salary": 3000 + k * 600,
                    "projection": 5.0 + k * 1.5,
                    "stddev": 2.0 + (k % 3),
                    "team": f"T{team}",
                    # T0 plays T1, T2 plays T3.
                    "opponent": f"T{team ^ 1}",
                }
            )
    return PlayerPool.from_records(records, tiny_spec, key_fields=["team", "opponent"])


def test_a_conflict_rule_keeps_a_pitcher_away_from_opposing_hitters(
    tiny_spec: RosterSpec,
) -> None:
    pool = opposed_pool(tiny_spec)
    spec = conflicted_spec(tiny_spec)
    teams = pool.keys["team"]
    opponents = pool.keys["opponent"]

    # The rule must be doing work, or this passes for the wrong reason.
    unconstrained = build_lineups(pool, tiny_spec, num_lineups=80, seed=6)
    assert any(
        any(opponents[a] == teams[b] for a in lineup for b in lineup)
        for lineup in unconstrained.tolist()
    )

    lineups = build_lineups(pool, spec, num_lineups=80, seed=6)
    assert len(lineups) > 0
    for lineup in lineups.tolist():
        for a in lineup:
            for b in lineup:
                assert not (opponents[a] == teams[b] and int(pool.positions[a]) & 1)


def test_conflicts_are_off_unless_asked_for(tiny_spec: RosterSpec) -> None:
    # The whole reason this is opt-in: a contrarian wanting a pitcher stacked
    # against their own hitters must still be able to build it.
    pool = opposed_pool(tiny_spec)
    assert np.array_equal(
        build_lineups(pool, tiny_spec, num_lineups=40, seed=6),
        build_lineups(pool, replace(tiny_spec, conflicts=()), num_lineups=40, seed=6),
    )


def test_explicit_conflict_pairs_are_honoured(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # Index pairs are a fact about this pool, so they live at the call site
    # rather than on the reusable specification.
    baseline = build_lineups(tiny_pool, tiny_spec, num_lineups=60, seed=7)
    # Take a pair the builder actually produces, rather than guessing one: a pair
    # it never picks anyway would make the assertion below vacuous.
    first = baseline.tolist()[0]
    pair = (first[0], first[-1])
    assert any(set(pair) <= set(lineup) for lineup in baseline.tolist())

    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=60, seed=7, conflict_pairs=[pair])
    assert len(lineups) > 0
    for lineup in lineups.tolist():
        assert not set(pair) <= set(lineup)


def test_conflict_pairs_are_symmetric(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    forward = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=7, conflict_pairs=[(8, 20)])
    reverse = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=7, conflict_pairs=[(20, 8)])
    assert np.array_equal(forward, reverse)


def test_an_out_of_range_conflict_pair_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    # Silently dropping it would forbid nothing, and the caller would never learn
    # that the constraint they asked for was not applied.
    with pytest.raises(ValueError, match="but the pool has"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, conflict_pairs=[(0, 9999)])


def test_a_negative_conflict_pair_is_rejected(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    with pytest.raises(ValueError, match="non-negative pool indices"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, conflict_pairs=[(0, -1)])


def test_a_conflict_rule_needs_its_key_in_the_pool(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    with pytest.raises(KeyError, match="conflict rule reads 'opponent'"):
        build_lineups(tiny_pool, conflicted_spec(tiny_spec), num_lineups=5)
