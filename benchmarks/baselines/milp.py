"""MILP formulations of the same problem, for comparison.

This is what mlb_dfs_solver is measured against, and it is important to be precise about
what the comparison shows — because the naive reading of it is wrong.

**A solver wins on one lineup.** Asked for the single best lineup, CP-SAT returns
the optimum and the greedy builder returns something slightly worse. That is not
in dispute and this file exists partly to demonstrate it.

**Asked for many, it returns the top N by projection.** Each is the best remaining
roster after forbidding the last, so they differ by a player or two and are built
from a narrow slice of the slate. `benchmarks/run.py` describes that difference
without judging it: whether a tight, high-projection set beats a wide one depends
on the contest and on the projections, and nothing in this repository can settle
that.

**A caveat on this baseline.** `_portfolio` below gets diversity from no-good cuts
alone. The other standard technique is an overlap constraint — bounding how many
players a new lineup may share with each one already found — which produces a
visibly wider set at 150 entries and costs substantially more per solve. It is not
implemented here, so read the 150-entry comparison as against the cheapest way to
get N lineups from a solver, not the best one. The contest-scale table does not
depend on the distinction: at 10,000 lineups both are out of reach.

Speed is the smaller part of the story and was long overstated here. On a
realistic slate a solver produces a 150-entry portfolio in seconds — not the
"15-20 seconds per lineup" this file used to claim, which was measured on a
degenerate slate where eleven clones of every player sent branch-and-bound
hunting through interchangeable optima. What a solver genuinely cannot do is
produce a *candidate pool*: twenty thousand lineups by no-good cut is the better
part of an hour, and they would be the twenty thousand most similar lineups
available.

OR-Tools' CP-SAT is the baseline: open source, installs everywhere, and markedly
faster on this problem shape than the bundled-CBC route PuLP offers — this file
carried a CBC formulation until it was benchmarked nowhere, and beating the
slower of two free solvers was the weaker claim anyway. Gurobi would be faster
still and is deliberately absent: it needs a commercial license, so a benchmark
nobody can reproduce.

**Every constraint the specification can express is modelled here.** That is what
makes the scenario ladder in `benchmarks/scenarios.py` a comparison rather than a
handicap: if the solver only enforced the salary cap while the kernel enforced
stacks and conflicts too, the kernel would look slow for doing more work. The one
thing that is genuinely awkward to state as a linear program is an exposure cap,
because it spans lineups rather than living inside one; it is applied between
solves, which is described where it happens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlb_dfs_solver.spec import UNCAPPED, scaled_salary

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mlb_dfs_solver.pool import PlayerPool
    from mlb_dfs_solver.spec import RosterSpec

__all__ = [
    "solve_milp_ortools",
    "solve_portfolio_ortools",
]


def _slot_assignments(spec: RosterSpec) -> list[tuple[int, int]]:
    """Expand slot groups into `(group_index, occurrence)` pairs."""
    return [(gi, k) for gi, slot in enumerate(spec.slots) for k in range(slot.count)]


def _eligible(pool: PlayerPool, spec: RosterSpec) -> dict[int, list[int]]:
    """Players who may fill each slot group."""
    return {
        gi: [i for i in range(len(pool)) if int(pool.positions[i]) & spec.mask_for(slot.eligible)]
        for gi, slot in enumerate(spec.slots)
    }


def _key_values(pool: PlayerPool, key: str) -> list[int]:
    """Distinct non-negative values of a key, in a stable order."""
    return sorted({int(k) for k in pool.keys[key] if int(k) >= 0})


def _extract(
    spec: RosterSpec,
    eligible: Mapping[int, Sequence[int]],
    is_set: object,
) -> list[int]:
    """Read a lineup out of a solved assignment, in slot order."""
    chosen_by_group: dict[int, list[int]] = {}
    for gi in range(len(spec.slots)):
        chosen_by_group[gi] = [i for i in eligible[gi] if is_set(i, gi)]  # type: ignore[operator]
    lineup: list[int] = []
    taken: set[int] = set()
    for gi, _ in _slot_assignments(spec):
        for i in chosen_by_group[gi]:
            if i not in taken:
                lineup.append(i)
                taken.add(i)
                break
    return lineup


def solve_milp_ortools(
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    excluded: Sequence[frozenset[int]] | None = None,
    banned: Sequence[int] | None = None,
    locks: Sequence[int] | None = None,
    conflict_pairs: Sequence[tuple[int, int]] | None = None,
    time_limit_s: float = 30.0,
) -> list[int] | None:
    """Return the highest-projection valid lineup, or None if infeasible.

    Formulated as an assignment problem — a binary per `(player, slot group)` —
    rather than one binary per player, because a player eligible at several
    positions has to be *placed* somewhere for group constraints restricted to
    particular slots to be expressible at all. Slot multipliers need the same
    thing for a different reason: a showdown captain is worth 1.5x and costs 1.5x,
    so both coefficients depend on where the player went, not just on who they are.

    Args:
        pool: Available players.
        spec: What makes a lineup legal.
        excluded: Player sets that may not all appear together. This is how a
            solver is made to produce a *different* lineup on the next call — one
            no-good cut per lineup already found.
        banned: Players that may not be used at all. The portfolio solver uses
            this to enforce exposure caps between solves.
        locks: Players that must appear.
        conflict_pairs: Pool-index pairs that may not appear together, on top of
            whatever `spec.conflicts` resolves to.
        time_limit_s: Wall-clock limit handed to CP-SAT.

    Returns:
        Pool indices in slot order, or None if no valid lineup exists.
    """
    from ortools.sat.python import cp_model

    n = len(pool)
    groups = list(range(len(spec.slots)))
    eligible = _eligible(pool, spec)

    model = cp_model.CpModel()
    assign = {(i, gi): model.new_bool_var(f"x_{i}_{gi}") for gi in groups for i in eligible[gi]}

    for gi in groups:
        model.add(sum(assign[i, gi] for i in eligible[gi]) == spec.slots[gi].count)
    for i in range(n):
        placements = [assign[i, gi] for gi in groups if (i, gi) in assign]
        if placements:
            model.add(sum(placements) <= 1)

    salary = sum(
        scaled_salary(int(pool.salaries[i]), spec.slots[gi].salary_multiplier) * var
        for (i, gi), var in assign.items()
    )
    model.add(salary <= spec.salary_cap)
    if spec.salary_floor > 0:
        model.add(salary >= spec.salary_floor)

    for group in spec.groups:
        counted = [gi for gi in groups if spec.slot_mask_for(group.slots) & (1 << gi)]
        keys = pool.keys[group.key]
        members_by_key = {
            key: [assign[i, gi] for gi in counted for i in eligible[gi] if int(keys[i]) == key]
            for key in _key_values(pool, group.key)
        }
        for members in members_by_key.values():
            if members and group.cap != UNCAPPED:
                model.add(sum(members) <= group.cap)

        if group.min_distinct:
            used = []
            for key, members in members_by_key.items():
                if not members:
                    continue
                indicator = model.new_bool_var(f"used_{group.key}_{key}")
                model.add(sum(members) >= 1).only_enforce_if(indicator)
                used.append(indicator)
            model.add(sum(used) >= group.min_distinct)

        if group.min_stack:
            reaches = []
            for key, members in members_by_key.items():
                if len(members) < group.min_stack:
                    continue
                indicator = model.new_bool_var(f"stack_{group.key}_{key}")
                model.add(sum(members) >= group.min_stack).only_enforce_if(indicator)
                reaches.append(indicator)
            if not reaches:
                return None
            model.add(sum(reaches) >= 1)

    pairs = list(zip(*pool.conflict_pairs(spec).tolist(), strict=True))
    pairs += list(conflict_pairs or [])
    for a, b in pairs:
        left = [assign[a, gi] for gi in groups if (a, gi) in assign]
        right = [assign[b, gi] for gi in groups if (b, gi) in assign]
        if left and right:
            model.add(sum(left) + sum(right) <= 1)

    for player in locks or []:
        placements = [assign[player, gi] for gi in groups if (player, gi) in assign]
        if not placements:
            return None
        model.add(sum(placements) == 1)

    for player in banned or []:
        for gi in groups:
            if (player, gi) in assign:
                model.add(assign[player, gi] == 0)

    for forbidden in excluded or []:
        members = [assign[i, gi] for gi in groups for i in eligible[gi] if i in forbidden]
        if members:
            model.add(sum(members) <= len(forbidden) - 1)

    # CP-SAT is integral, so projections are scaled rather than rounded away.
    model.maximize(
        sum(
            int(round(float(pool.projections[i]) * spec.slots[gi].score_multiplier * 1000)) * var
            for (i, gi), var in assign.items()
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    # Fixed, so the benchmark measures the solver rather than the machine.
    solver.parameters.num_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    return _extract(spec, eligible, lambda i, gi: bool(solver.value(assign[i, gi])))


def _portfolio(
    solve: object,
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    num_lineups: int,
    locks: Sequence[int] | None,
    conflict_pairs: Sequence[tuple[int, int]] | None,
    max_exposure: float | Mapping[int, float] | None,
    time_limit_s: float,
    budget_s: float | None,
) -> list[list[int]]:
    """Re-solve with no-good cuts until `num_lineups` distinct lineups exist.

    Exposure caps are the one constraint not put into the model. They span
    lineups, so expressing them inside a single solve is impossible and expressing
    them across all of them at once means one giant model with `num_lineups` copies
    of every variable — a different and much harder problem than the one the
    kernel solves. Applied between solves instead: a player who has reached their
    limit is banned from the next model. That is the same greedy rule the kernel
    uses at its merge, so the comparison stays like for like.

    `budget_s` bounds the whole loop by wall clock, checked before each solve.
    Ten thousand sequential solves have no natural upper bound, and a benchmark
    that either finishes in the tens of hours or gets killed mid-run produces no
    number at all. Stopping at the budget and returning what exists turns the
    claim into a measurement: this many lineups is what the budget bought.
    """
    from time import perf_counter

    started = perf_counter()
    limits: dict[int, int] = {}
    if max_exposure is not None:
        fractions = (
            {i: float(max_exposure) for i in range(len(pool))}
            if isinstance(max_exposure, (int, float))
            else {int(k): float(v) for k, v in max_exposure.items()}
        )
        limits = {p: int(f * num_lineups) for p, f in fractions.items()}

    found: list[list[int]] = []
    cuts: list[frozenset[int]] = []
    appearances: dict[int, int] = {}
    for _ in range(num_lineups):
        if budget_s is not None and perf_counter() - started >= budget_s:
            break
        banned = [p for p, limit in limits.items() if appearances.get(p, 0) >= limit]
        lineup = solve(  # type: ignore[operator]
            pool,
            spec,
            excluded=cuts,
            banned=banned,
            locks=locks,
            conflict_pairs=conflict_pairs,
            time_limit_s=time_limit_s,
        )
        if lineup is None:
            break
        found.append(lineup)
        cuts.append(frozenset(lineup))
        for player in lineup:
            appearances[player] = appearances.get(player, 0) + 1
    return found


def solve_portfolio_ortools(
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    num_lineups: int,
    locks: Sequence[int] | None = None,
    conflict_pairs: Sequence[tuple[int, int]] | None = None,
    max_exposure: float | Mapping[int, float] | None = None,
    time_limit_s: float = 30.0,
    budget_s: float | None = None,
) -> list[list[int]]:
    """Produce `num_lineups` distinct lineups with CP-SAT and no-good cuts.

    This is the apples-to-apples comparison against `mlb_dfs_solver.build_lineups`.

    Per-lineup cost is extremely sensitive to the slate rather than to its size.
    On a slate with many tied `(salary, projection)` pairs a solver can run
    hundreds of times slower, because proving optimality means ruling out every
    interchangeable alternative. A benchmark that reports solver time without
    saying whether its inputs are degenerate is reporting the degeneracy.

    `budget_s`, when set, caps the whole loop by wall clock and returns whatever
    was found by then — see `_portfolio` for why that is the honest form of a
    contest-scale claim.
    """
    return _portfolio(
        solve_milp_ortools,
        pool,
        spec,
        num_lineups=num_lineups,
        locks=locks,
        conflict_pairs=conflict_pairs,
        max_exposure=max_exposure,
        time_limit_s=time_limit_s,
        budget_s=budget_s,
    )
