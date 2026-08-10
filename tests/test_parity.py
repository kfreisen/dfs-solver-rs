"""The kernel and the reference implementation must agree.

This is the most important test file in the package. Every performance claim
mlb_dfs_solver makes is a comparison against `benchmarks/baselines/reference.py`, and a
comparison only means something if both implementations do the same job. A fast
path that has quietly changed its behavior produces an excellent benchmark number
and a wrong answer.

**What parity means here, precisely.** Not byte-identical output: matching the Rust
lineup-for-lineup would require reimplementing xoshiro256++ and its float mapping
in Python, which is a lot of fragile code in service of a weaker guarantee than the
one below. What is asserted instead:

* every lineup either implementation produces satisfies every rule, checked by a
  validator written independently of both builders;
* both produce distinct lineups, and about as many of them;
* both explore comparably — neither collapses onto a narrow set of players;
* both respond the same way to the constraints being tightened.

Those are the properties a caller relies on. Bit-identity is not one of them.
"""

from __future__ import annotations

import numpy as np
import pytest
from baselines.reference import build_lineups_reference, is_valid, validity_report
from mlb_dfs_solver import build_lineups
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.spec import GroupConstraint, RosterSpec


def test_kernel_lineups_are_all_valid(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    lineups = build_lineups(tiny_pool, tiny_spec, num_lineups=200, seed=1)
    assert len(lineups) > 0
    assert validity_report(lineups, tiny_pool, tiny_spec) == ""


def test_reference_lineups_are_all_valid(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # The oracle has to be right, or it certifies nothing.
    lineups = build_lineups_reference(tiny_pool, tiny_spec, num_lineups=200, seed=1)
    assert len(lineups) > 0
    for lineup in lineups:
        assert is_valid(lineup, tiny_pool, tiny_spec), lineup


def test_kernel_lineups_are_valid_for_a_real_slate(
    mlb_pool: PlayerPool, mlb_spec: RosterSpec
) -> None:
    # DK_MLB_CLASSIC has a salary floor, two team caps that interact, and a
    # three-deep outfield — far more chances to produce something illegal.
    lineups = build_lineups(mlb_pool, mlb_spec, num_lineups=300, seed=3)
    assert len(lineups) > 0
    assert validity_report(lineups, mlb_pool, mlb_spec) == ""


def test_reference_lineups_are_valid_for_a_real_slate(
    mlb_pool: PlayerPool, mlb_spec: RosterSpec
) -> None:
    lineups = build_lineups_reference(mlb_pool, mlb_spec, num_lineups=60, seed=3)
    assert len(lineups) > 0
    for lineup in lineups:
        assert is_valid(lineup, mlb_pool, mlb_spec), lineup


def test_both_produce_distinct_lineups(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    kernel = build_lineups(tiny_pool, tiny_spec, num_lineups=150, seed=5)
    reference = build_lineups_reference(tiny_pool, tiny_spec, num_lineups=150, seed=5)

    kernel_keys = {tuple(sorted(row)) for row in kernel.tolist()}
    reference_keys = {tuple(sorted(row)) for row in reference}
    assert len(kernel_keys) == len(kernel)
    assert len(reference_keys) == len(reference)


def test_both_fill_the_request_on_a_pool_that_can_support_it(
    mlb_pool: PlayerPool, mlb_spec: RosterSpec
) -> None:
    requested = 50
    kernel = build_lineups(mlb_pool, mlb_spec, num_lineups=requested, seed=11)
    reference = build_lineups_reference(mlb_pool, mlb_spec, num_lineups=requested, seed=11)
    assert len(kernel) == requested
    assert len(reference) == requested


def test_both_explore_a_comparable_share_of_the_pool(
    mlb_pool: PlayerPool, mlb_spec: RosterSpec
) -> None:
    """Neither implementation may collapse onto a narrow core of players.

    This is the property that catches a real class of porting bug — a broken
    objective or an incorrectly seeded generator still produces valid, distinct lineups
    while drawing from a fraction of the pool, and a validity check alone would
    pass it.
    """
    kernel = build_lineups(mlb_pool, mlb_spec, num_lineups=200, seed=7)
    reference = build_lineups_reference(mlb_pool, mlb_spec, num_lineups=200, seed=7)

    kernel_players = len(set(kernel.ravel().tolist()))
    reference_players = len({p for lineup in reference for p in lineup})

    assert kernel_players > 0.25 * len(mlb_pool)
    assert reference_players > 0.25 * len(mlb_pool)
    # Within a factor of two of each other. A loose bound on purpose: these are
    # different generators, so the claim is "similar breadth", not "same breadth".
    ratio = kernel_players / reference_players
    assert 0.5 < ratio < 2.0, f"{kernel_players} vs {reference_players} distinct players"


def test_both_let_the_objective_drive_selection(mlb_pool: PlayerPool, mlb_spec: RosterSpec) -> None:
    """The objective must actually decide who gets picked.

    Comparing against the pool's mean projection would be the obvious test and it
    is wrong: a salary cap forces cheap players, so a *valid* lineup legitimately
    scores below the unconstrained pool average. Failing that comparison says
    nothing about whether the objective works.

    So vary only the noise. At `noise=0` ordering is the objective alone; at a
    noise level that dwarfs every projection, ordering is effectively random. If
    the objective is wired up, the first must score materially higher — and a
    builder that ignored its objective would score the same either way.
    """
    signal = build_lineups(mlb_pool, mlb_spec, num_lineups=100, seed=13, noise=0.0)
    scrambled = build_lineups(mlb_pool, mlb_spec, num_lineups=100, seed=13, noise=50.0)
    assert len(signal) > 0
    assert len(scrambled) > 0
    assert float(mlb_pool.projection_of(signal, mlb_spec).mean()) > float(
        mlb_pool.projection_of(scrambled, mlb_spec).mean()
    )

    ref_signal = build_lineups_reference(mlb_pool, mlb_spec, num_lineups=30, seed=13, noise=0.0)
    ref_scrambled = build_lineups_reference(mlb_pool, mlb_spec, num_lineups=30, seed=13, noise=50.0)
    assert ref_signal
    assert ref_scrambled

    def mean_projection(lineups: list[list[int]]) -> float:
        return float(np.mean([sum(mlb_pool.projections[i] for i in lineup) for lineup in lineups]))

    assert mean_projection(ref_signal) > mean_projection(ref_scrambled)


@pytest.mark.parametrize("cap", [30_000, 40_000, 50_000])
def test_both_respond_to_a_tightening_cap(
    mlb_pool: PlayerPool, mlb_spec: RosterSpec, cap: int
) -> None:
    """Tightening the budget must bind on both implementations identically."""
    from dataclasses import replace

    spec = replace(mlb_spec, salary_cap=cap, salary_floor=0)
    kernel = build_lineups(mlb_pool, spec, num_lineups=40, seed=17)
    reference = build_lineups_reference(mlb_pool, spec, num_lineups=40, seed=17)

    if len(kernel):
        assert int(mlb_pool.salary_of(kernel, spec).max()) <= cap
    for lineup in reference:
        assert sum(int(mlb_pool.salaries[i]) for i in lineup) <= cap


def test_both_return_nothing_when_the_cap_is_impossible(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    from dataclasses import replace

    spec = replace(tiny_spec, salary_cap=1, salary_floor=0)
    assert len(build_lineups(tiny_pool, spec, num_lineups=10, seed=1)) == 0
    assert build_lineups_reference(tiny_pool, spec, num_lineups=10, seed=1) == []


def test_both_honour_a_salary_floor_that_forces_repair(
    mlb_pool: PlayerPool, mlb_spec: RosterSpec
) -> None:
    """The floor is the path that exercises salary repair in both.

    Repair is the fiddliest part of the algorithm — it removes a player from the
    tally, searches, and has to put them back correctly on failure. A leak there
    shows up as a group-cap violation, which the validator catches.
    """
    kernel = build_lineups(mlb_pool, mlb_spec, num_lineups=100, seed=23)
    reference = build_lineups_reference(mlb_pool, mlb_spec, num_lineups=40, seed=23)

    assert len(kernel) > 0
    assert validity_report(kernel, mlb_pool, mlb_spec) == ""
    assert all(
        sum(int(mlb_pool.salaries[i]) for i in lineup) >= mlb_spec.salary_floor
        for lineup in reference
    )


# --- Slot multipliers and conflicts --------------------------------------


def showdown_spec() -> RosterSpec:
    """A single-game shape over the tiny fixture's positions."""
    from mlb_dfs_solver.spec import Slot

    return RosterSpec(
        positions=("P", "C", "OF"),
        slots=(
            Slot("CPT", ("P", "C", "OF"), score_multiplier=1.5, salary_multiplier=1.5),
            Slot("FLEX", ("P", "C", "OF"), count=3),
        ),
        salary_cap=30_000,
        salary_floor=0,
    )


def test_both_price_a_captain_slot_the_same_way(tiny_pool: PlayerPool) -> None:
    """A multiplier the two implementations disagreed about would be invisible.

    The kernel would call a lineup legal and the oracle would call it over the
    cap, or worse, both would agree on a number neither the operator nor the
    reader recognises. So the check is against the independent validator.
    """
    spec = showdown_spec()
    kernel = build_lineups(tiny_pool, spec, num_lineups=60, seed=21)
    reference = build_lineups_reference(tiny_pool, spec, num_lineups=60, seed=21)
    assert len(kernel) > 0
    assert reference
    assert validity_report(kernel, tiny_pool, spec) == ""
    for lineup in reference:
        assert is_valid(lineup, tiny_pool, spec), lineup


def test_both_build_comparably_many_showdown_lineups(tiny_pool: PlayerPool) -> None:
    # Different RNGs mean different lineups, but a multiplier applied in one
    # implementation and not the other would show up as one of them finding far
    # fewer lineups under the same cap.
    spec = showdown_spec()
    kernel = build_lineups(tiny_pool, spec, num_lineups=80, seed=22)
    reference = build_lineups_reference(tiny_pool, spec, num_lineups=80, seed=22)
    assert 0.5 <= len(kernel) / max(len(reference), 1) <= 2.0


def opposed_pool(spec: RosterSpec) -> PlayerPool:
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
                    "opponent": f"T{team ^ 1}",
                }
            )
    return PlayerPool.from_records(records, spec, key_fields=["team", "opponent"])


def test_both_enforce_a_conflict_rule(tiny_spec: RosterSpec) -> None:
    from dataclasses import replace

    from mlb_dfs_solver.spec import ConflictRule

    spec = replace(
        tiny_spec,
        conflicts=(ConflictRule(left_key="opponent", right_key="team", left_positions=("P",)),),
    )
    pool = opposed_pool(spec)
    kernel = build_lineups(pool, spec, num_lineups=60, seed=23)
    reference = build_lineups_reference(pool, spec, num_lineups=60, seed=23)
    assert len(kernel) > 0
    assert reference
    assert validity_report(kernel, pool, spec) == ""
    for lineup in reference:
        assert is_valid(lineup, pool, spec), lineup


def test_both_enforce_explicit_conflict_pairs(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    pairs = [(0, 8), (1, 9)]
    kernel = build_lineups(tiny_pool, tiny_spec, num_lineups=60, seed=24, conflict_pairs=pairs)
    reference = build_lineups_reference(
        tiny_pool, tiny_spec, num_lineups=60, seed=24, extra_conflict_pairs=pairs
    )
    assert len(kernel) > 0
    assert reference
    assert validity_report(kernel, tiny_pool, tiny_spec, extra_conflict_pairs=pairs) == ""
    for lineup in reference:
        assert is_valid(lineup, tiny_pool, tiny_spec, extra_conflict_pairs=pairs), lineup


def test_both_enforce_conflicts_through_salary_repair(tiny_spec: RosterSpec) -> None:
    """Repair is where conflict bookkeeping is easiest to get wrong.

    It removes a player, searches for a replacement, and must restore the marks
    exactly on failure. A leak produces lineups the fill loop would never build,
    and only a floor tight enough to force repair reaches that code.
    """
    from dataclasses import replace

    from mlb_dfs_solver.spec import ConflictRule

    spec = replace(
        tiny_spec,
        salary_cap=22_000,
        salary_floor=20_000,
        conflicts=(ConflictRule(left_key="opponent", right_key="team", left_positions=("P",)),),
    )
    pool = opposed_pool(spec)
    kernel = build_lineups(pool, spec, num_lineups=60, seed=25)
    reference = build_lineups_reference(pool, spec, num_lineups=60, seed=25)
    assert len(kernel) > 0, "repair recovered nothing to check"
    assert reference
    assert validity_report(kernel, pool, spec) == ""
    for lineup in reference:
        assert is_valid(lineup, pool, spec), lineup


# --- Locks and exposure caps ---------------------------------------------


def test_both_honour_the_same_locks(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    locks = [9, 17]
    kernel = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=31, locks=locks)
    reference = build_lineups_reference(tiny_pool, tiny_spec, num_lineups=40, seed=31, locks=locks)
    assert len(kernel) > 0
    assert reference
    assert validity_report(kernel, tiny_pool, tiny_spec) == ""
    for lineup in kernel.tolist():
        assert set(locks) <= set(lineup)
    for lineup in reference:
        assert is_valid(lineup, tiny_pool, tiny_spec), lineup
        assert set(locks) <= set(lineup)


def test_both_place_locks_in_the_same_slots(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    # The slot a lock lands in is decided by assign_locks, which both call, so
    # the two must agree on the column even though they disagree on the rest.
    locks = [9, 17]
    kernel = build_lineups(tiny_pool, tiny_spec, num_lineups=20, seed=31, locks=locks)
    reference = build_lineups_reference(tiny_pool, tiny_spec, num_lineups=20, seed=31, locks=locks)
    assert all(lineup[0] == 9 and lineup[1] == 17 for lineup in kernel.tolist())
    assert all(lineup[0] == 9 and lineup[1] == 17 for lineup in reference)


def test_both_honour_locks_through_salary_repair(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    from dataclasses import replace

    spec = replace(tiny_spec, salary_cap=22_000, salary_floor=20_000)
    kernel = build_lineups(tiny_pool, spec, num_lineups=40, seed=32, locks=[8])
    reference = build_lineups_reference(tiny_pool, spec, num_lineups=40, seed=32, locks=[8])
    assert len(kernel) > 0, "repair recovered nothing to check"
    assert reference
    assert validity_report(kernel, tiny_pool, spec) == ""
    assert all(8 in lineup for lineup in kernel.tolist())
    for lineup in reference:
        assert is_valid(lineup, tiny_pool, spec), lineup
        assert 8 in lineup


def test_both_return_nothing_for_a_lock_that_cannot_fit(
    tiny_pool: PlayerPool, tiny_spec: RosterSpec
) -> None:
    from dataclasses import replace

    spec = replace(tiny_spec, groups=(GroupConstraint(key="team", max_count=1),))
    assert len(build_lineups(tiny_pool, spec, num_lineups=20, seed=33, locks=[16, 20])) == 0
    assert build_lineups_reference(tiny_pool, spec, num_lineups=20, seed=33, locks=[16, 20]) == []


def test_both_respect_the_same_exposure_cap(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    kernel = build_lineups(tiny_pool, tiny_spec, num_lineups=40, seed=34, max_exposure=0.25)
    reference = build_lineups_reference(
        tiny_pool, tiny_spec, num_lineups=40, seed=34, max_exposure=0.25
    )
    assert len(kernel) > 0
    assert reference
    assert validity_report(kernel, tiny_pool, tiny_spec) == ""
    # 25% of the 40 requested is 10, for both, whatever each actually returned.
    for player in range(len(tiny_pool)):
        assert sum(player in lineup for lineup in kernel.tolist()) <= 10
        assert sum(player in lineup for lineup in reference) <= 10


def test_both_exclude_a_player_capped_at_zero(tiny_pool: PlayerPool, tiny_spec: RosterSpec) -> None:
    kernel = build_lineups(tiny_pool, tiny_spec, num_lineups=30, seed=35, max_exposure={15: 0.0})
    reference = build_lineups_reference(
        tiny_pool, tiny_spec, num_lineups=30, seed=35, max_exposure={15: 0.0}
    )
    assert len(kernel) > 0
    assert reference
    assert all(15 not in lineup for lineup in kernel.tolist())
    assert all(15 not in lineup for lineup in reference)
