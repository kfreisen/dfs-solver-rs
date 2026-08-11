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
from conftest import make_records
from mlb_dfs_solver import CONTRARIAN, STANDARD, JitterProfile, build_lineups
from mlb_dfs_solver.greedy import assign_locks
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


# --- Locks ---------------------------------------------------------------


def test_a_locked_player_appears_in_every_lineup(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=8, locks=[9])
    assert len(lineups) > 0
    assert all(9 in lineup for lineup in lineups.tolist())


def test_a_lock_overrides_what_the_greedy_would_have_picked(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    # Player 8 is the cheapest, worst-projected catcher, so an unlocked build
    # almost never takes them. A lock that merely nudged the objective would
    # produce a mixture.
    unlocked = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=8)
    assert sum(8 in lineup for lineup in unlocked.tolist()) < len(unlocked)

    locked = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=8, locks=[8])
    assert len(locked) > 0
    assert all(8 in lineup for lineup in locked.tolist())


def test_locks_land_in_their_slot_column(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # tiny_spec is C, OF x2, P. A catcher lock belongs in column 0.
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=20, seed=8, locks=[9])
    assert all(lineup[0] == 9 for lineup in lineups.tolist())


def test_several_locks_are_honoured_together(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=30, seed=8, locks=[9, 17, 18, 1])
    assert len(lineups) > 0
    for lineup in lineups.tolist():
        assert {9, 17, 18, 1} <= set(lineup)


def test_locking_a_full_roster_yields_exactly_one_lineup(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=30, seed=8, locks=[9, 17, 18, 1])
    assert len(lineups) == 1
    assert lineups[0].tolist() == [9, 17, 18, 1]


def test_locks_can_be_pinned_to_a_named_slot(tiny_pool: PlayerPool) -> None:
    # The reason pinning exists: in a showdown every slot takes every position,
    # so "lock this player" is ambiguous until you say which slot.
    spec = showdown_spec(salary=1.0)
    lineups = build_lineups(tiny_pool, spec, num_lineups=20, seed=9, locks={2: "CPT"})
    assert len(lineups) > 0
    assert all(lineup[0] == 2 for lineup in lineups.tolist())

    flexed = build_lineups(tiny_pool, spec, num_lineups=20, seed=9, locks={2: "FLEX"})
    assert len(flexed) > 0
    assert all(lineup[0] != 2 and 2 in lineup for lineup in flexed.tolist())


def test_locking_nobody_changes_nothing(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    assert np.array_equal(
        build_lineups(tiny_pool, tiny_spec, num_lineups=30, seed=8),
        build_lineups(tiny_pool, tiny_spec, num_lineups=30, seed=8, locks=[]),
    )


def test_locks_survive_a_salary_floor(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # Repair reaches the floor by swapping the cheapest player out, and a lock is
    # often exactly that player.
    spec = replace(tiny_spec, salary_cap=22_000, salary_floor=20_000)
    lineups = build_lineups(tiny_pool, spec, num_lineups=40, seed=10, locks=[8])
    assert len(lineups) > 0, "repair recovered nothing to check"
    assert all(8 in lineup for lineup in lineups.tolist())


def test_a_lock_that_cannot_coexist_yields_nothing_rather_than_dropping_it(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    # Two locked outfielders on the same team, with a team cap that a four-player
    # roster cannot then satisfy.
    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", max_count=1),))
    lineups = build_lineups(tiny_pool, spec, num_lineups=20, seed=8, locks=[16, 20])
    assert len(lineups) == 0


def test_a_lock_outside_the_pool_is_rejected(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    with pytest.raises(ValueError, match="outside a pool of"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, locks=[9999])


def test_a_lock_eligible_for_no_slot_is_rejected(tiny_spec: RosterSpec) -> None:
    # A designated hitter on a slate whose spec has no DH slot: the honest answer
    # names the player, not "no valid lineups found".
    spec = replace(tiny_spec, positions=("P", "C", "OF", "DH"))
    records = [
        *make_records(),
        {"name": "dh", "positions": ("DH",), "salary": 3000, "projection": 5.0, "team": "T0"},
    ]
    pool = PlayerPool.from_records(records, spec)
    with pytest.raises(ValueError, match="not eligible for any slot"):
        build_lineups(pool, spec, num_lineups=5, locks=[len(records) - 1])


def test_a_lock_pinned_to_an_unknown_slot_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    with pytest.raises(ValueError, match="this specification has"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, locks={9: "QB"})


def test_a_lock_pinned_to_an_ineligible_slot_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    with pytest.raises(ValueError, match="not eligible for slot 'C'"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, locks={0: "C"})


def test_the_same_player_locked_twice_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    with pytest.raises(ValueError, match="locked more than once"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, locks=[9, 9])


def test_more_locks_than_a_slot_holds_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    # Two catchers, one catcher slot. Reported rather than returned empty.
    with pytest.raises(ValueError, match="cannot be placed"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, locks=[8, 9])


def test_lock_assignment_backtracks_rather_than_taking_the_first_fit(
    tiny_pool: PlayerPool,
) -> None:
    """A greedy assignment would reject a set of locks that is perfectly legal.

    Two locks, both eligible for the flex; only one of them can also fill the
    dedicated slot. Assigning the first lock to the flex strands the second, even
    though swapping them works. Matching finds the arrangement; greedy does not.
    """
    spec = RosterSpec(
        positions=("P", "C", "OF"),
        slots=(Slot("FLEX", ("C", "OF")), Slot("C", ("C",)), Slot("P", ("P",))),
        salary_cap=30_000,
        salary_floor=0,
    )
    # Player 8 is a catcher (fits both FLEX and C); player 16 is an outfielder
    # (fits only FLEX). Listed so that a first-fit walk puts 8 in FLEX first.
    # 8 ends up in the dedicated catcher slot (group 1) so that 16, which fits
    # nowhere else, can have the flex (group 0).
    assert assign_locks(tiny_pool, spec, [8, 16]) == [(8, 1), (16, 0)]

    lineups = build_lineups(tiny_pool, spec, num_lineups=20, seed=11, locks=[8, 16])
    assert len(lineups) > 0
    for lineup in lineups.tolist():
        assert lineup[0] == 16
        assert lineup[1] == 8


# --- Exposure caps -------------------------------------------------------


def test_an_exposure_cap_bounds_appearances(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    uncapped = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=12)
    counts = [sum(p in lineup for lineup in uncapped.tolist()) for p in range(len(tiny_pool))]
    busiest = max(range(len(tiny_pool)), key=lambda p: counts[p])
    assert counts[busiest] > 10, "nothing appeared often enough to cap"

    capped = build_lineups(
        tiny_pool, tiny_spec, num_lineups=40, seed=12, max_exposure={busiest: 0.25}
    )
    assert len(capped) > 0
    assert sum(busiest in lineup for lineup in capped.tolist()) <= 10


def test_a_global_exposure_cap_applies_to_everyone(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=12, max_exposure=0.2)
    assert len(lineups) > 0
    for player in range(len(tiny_pool)):
        assert sum(player in lineup for lineup in lineups.tolist()) <= 8


def test_a_zero_exposure_cap_excludes_a_player(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    # The natural way to say "fade this player" without rebuilding the pool.
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=12, max_exposure={15: 0.0})
    assert len(lineups) > 0
    assert all(15 not in lineup for lineup in lineups.tolist())


def test_exposure_caps_are_deterministic(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # The cap is applied at the merge precisely so this holds; enforcing it
    # inside the parallel region would make the answer depend on scheduling.
    a = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=12, max_exposure=0.3)
    b = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=12, max_exposure=0.3)
    assert np.array_equal(a, b)


def test_a_full_exposure_cap_changes_nothing(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    assert np.array_equal(
        build_lineups(tiny_pool, tiny_spec, num_lineups=30, seed=12),
        build_lineups(tiny_pool, tiny_spec, num_lineups=30, seed=12, max_exposure=1.0),
    )


def test_a_tight_cap_returns_fewer_lineups_rather_than_breaking_it(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=12, max_exposure=0.05)
    assert 0 < len(lineups) < 40
    for player in range(len(tiny_pool)):
        assert sum(player in lineup for lineup in lineups.tolist()) <= 2


def test_an_out_of_range_exposure_fraction_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    with pytest.raises(ValueError, match="fraction in"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, max_exposure=1.5)


def test_exposure_naming_a_missing_player_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    with pytest.raises(ValueError, match="but the pool has"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, max_exposure={999: 0.5})


def test_locking_and_capping_the_same_player_is_rejected(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    # Both were asked for and they cannot both hold. Honouring one silently would
    # look exactly like the other being ignored.
    with pytest.raises(ValueError, match="cannot both be true"):
        build_lineups(tiny_pool, tiny_spec, num_lineups=5, locks=[9], max_exposure={9: 0.5})


def test_locks_and_a_global_cap_coexist_when_consistent(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    lineups = build_lineups(
        tiny_pool, tiny_spec, num_lineups=30, seed=12, locks=[9], max_exposure={16: 0.2}
    )
    assert len(lineups) > 0
    assert all(9 in lineup for lineup in lineups.tolist())
    assert sum(16 in lineup for lineup in lineups.tolist()) <= 6


# --- Minimums ------------------------------------------------------------


def distinct_teams(pool: PlayerPool, lineup: list[int]) -> int:
    return len({int(pool.keys["team"][p]) for p in lineup})


def biggest_stack(pool: PlayerPool, lineup: list[int]) -> int:
    counts: dict[int, int] = {}
    for p in lineup:
        key = int(pool.keys["team"][p])
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def test_a_distinct_minimum_is_met_by_every_lineup(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", min_distinct=3),))
    lineups = build_lineups(tiny_pool, spec, num_lineups=50, seed=40)
    assert len(lineups) > 0
    assert all(distinct_teams(tiny_pool, lineup) >= 3 for lineup in lineups.tolist())


def test_a_distinct_minimum_actually_binds(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # Without it the builder returns lineups from fewer teams, so the assertion
    # above is not passing for free.
    loose = build_lineups(tiny_pool, tiny_spec, num_lineups=50, seed=40)
    assert any(distinct_teams(tiny_pool, lineup) < 4 for lineup in loose.tolist())

    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", min_distinct=4),))
    lineups = build_lineups(tiny_pool, spec, num_lineups=50, seed=40)
    assert len(lineups) > 0
    assert all(distinct_teams(tiny_pool, lineup) == 4 for lineup in lineups.tolist())


def test_a_distinct_minimum_counts_only_its_named_slots(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    # Three distinct teams among C and the two OF slots; the pitcher does not
    # help satisfy it.
    spec = replace(
        tiny_spec, groups=(GroupConstraint(key="team", min_distinct=3, slots=("C", "OF")),)
    )
    lineups = build_lineups(tiny_pool, spec, num_lineups=40, seed=41)
    assert len(lineups) > 0
    for lineup in lineups.tolist():
        assert distinct_teams(tiny_pool, lineup[:3]) >= 3


def test_a_distinct_minimum_the_pool_cannot_meet_returns_nothing(
    tiny_spec: RosterSpec,
) -> None:
    records = [{**r, "team": "ONE"} for r in make_records()]
    pool = PlayerPool.from_records(records, tiny_spec)
    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", min_distinct=2),))
    assert len(build_lineups(pool, spec, num_lineups=20, seed=41)) == 0


def test_a_stack_minimum_is_met_by_every_lineup(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", min_stack=3),))
    lineups = build_lineups(tiny_pool, spec, num_lineups=50, seed=42)
    assert len(lineups) > 0
    assert all(biggest_stack(tiny_pool, lineup) >= 3 for lineup in lineups.tolist())


def test_a_stack_minimum_actually_binds(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    loose = build_lineups(tiny_pool, tiny_spec, num_lineups=50, seed=42)
    assert any(biggest_stack(tiny_pool, lineup) < 3 for lineup in loose.tolist())


def test_stacks_spread_across_teams(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # The reason the target is redrawn per attempt. A builder that answered
    # "which team?" once would return a portfolio stacked entirely on one team.
    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", min_stack=3),))
    lineups = build_lineups(tiny_pool, spec, num_lineups=60, seed=43)
    assert len(lineups) > 5
    stacked = set()
    for lineup in lineups.tolist():
        counts: dict[int, int] = {}
        for p in lineup:
            key = int(tiny_pool.keys["team"][p])
            counts[key] = counts.get(key, 0) + 1
        stacked.add(max(counts, key=lambda k: counts[k]))
    assert len(stacked) > 1, f"every lineup stacked the same team: {stacked}"


def test_a_stack_the_pool_cannot_supply_returns_nothing(tiny_spec: RosterSpec) -> None:
    # Every player on their own team, so no team has a second player. The roster
    # has room for the stack; the pool simply cannot supply it.
    records = [{**r, "team": f"T{i}"} for i, r in enumerate(make_records())]
    pool = PlayerPool.from_records(records, tiny_spec)
    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", min_stack=2),))
    assert len(build_lineups(pool, spec, num_lineups=20, seed=43)) == 0


def test_a_stack_and_a_cap_coexist(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # The shape a real MLB spec has: stack four hitters from a team, and never
    # more than five from any one.
    spec = replace(
        tiny_spec,
        groups=(GroupConstraint(key="team", max_count=3, min_stack=3),),
    )
    lineups = build_lineups(tiny_pool, spec, num_lineups=40, seed=44)
    assert len(lineups) > 0
    for lineup in lineups.tolist():
        assert biggest_stack(tiny_pool, lineup) == 3


def test_a_stack_and_a_distinct_minimum_hold_together(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    # They pull in opposite directions: one concentrates, the other spreads.
    spec = replace(
        tiny_spec,
        groups=(
            GroupConstraint(key="team", min_stack=2),
            GroupConstraint(key="team", min_distinct=3),
        ),
    )
    lineups = build_lineups(tiny_pool, spec, num_lineups=40, seed=45)
    assert len(lineups) > 0
    for lineup in lineups.tolist():
        assert biggest_stack(tiny_pool, lineup) >= 2
        assert distinct_teams(tiny_pool, lineup) >= 3


def test_minimums_survive_a_salary_floor(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    spec = replace(
        tiny_spec,
        salary_cap=22_000,
        salary_floor=20_000,
        groups=(
            GroupConstraint(key="team", min_stack=2),
            GroupConstraint(key="team", min_distinct=2),
        ),
    )
    lineups = build_lineups(tiny_pool, spec, num_lineups=40, seed=46)
    assert len(lineups) > 0, "repair recovered nothing to check"
    for lineup in lineups.tolist():
        assert biggest_stack(tiny_pool, lineup) >= 2
        assert distinct_teams(tiny_pool, lineup) >= 2


def test_minimums_are_deterministic(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # The stack target is drawn from the same generator as the jitter, so the
    # output must stay a function of the seed alone.
    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", min_stack=3),))
    assert np.array_equal(
        build_lineups(tiny_pool, spec, num_lineups=40, seed=47),
        build_lineups(tiny_pool, spec, num_lineups=40, seed=47),
    )


def test_minimums_compose_with_locks_and_exposure(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", min_distinct=3),))
    lineups = build_lineups(
        tiny_pool, spec, num_lineups=30, seed=48, locks=[9], max_exposure={16: 0.3}
    )
    assert len(lineups) > 0
    for lineup in lineups.tolist():
        assert 9 in lineup
        assert distinct_teams(tiny_pool, lineup) >= 3
    assert sum(16 in lineup for lineup in lineups.tolist()) <= 9
