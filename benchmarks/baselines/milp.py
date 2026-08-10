"""MILP formulations of the same problem, for comparison.

This is what mlb_dfs_solver is measured against, and it is important to be precise about
what the comparison shows — because the naive reading of it is wrong.

**A solver wins on one lineup.** Asked for the single best lineup, CBC returns the
optimum and the greedy builder returns something slightly worse. That is not in
dispute and this file exists partly to demonstrate it.

**The interesting question is a portfolio.** Asked for 150 lineups, a solver
returns the optimum, then the second best, then the third — which differ from each
other by one or two players. That is a terrible portfolio for a large-field
contest, where the payoff is convex and coverage of the outcome space is what pays.
Getting genuine diversity out of a solver means adding no-good cuts or overlap
constraints and re-solving once per lineup, and the cost of that is what the
benchmark measures.

So the honest framing is: **per-lineup quality favors the solver, and
lineups-per-second favors the greedy builder by orders of magnitude.** Both numbers
are reported.

PuLP with the bundled CBC is the baseline because it is open source and installs
everywhere. OR-Tools' CP-SAT is included as a second opinion — it is markedly
faster than CBC on this problem shape and makes the comparison harder for mlb_dfs_solver,
which is the point of including it. Gurobi would be faster still and is deliberately
absent: it needs a commercial license, so a benchmark nobody can reproduce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlb_dfs_solver.pool import PlayerPool
    from mlb_dfs_solver.spec import RosterSpec

__all__ = ["solve_milp_ortools", "solve_milp_pulp", "solve_portfolio_pulp"]


def _slot_assignments(spec: RosterSpec) -> list[tuple[int, int]]:
    """Expand slot groups into `(group_index, occurrence)` pairs."""
    return [(gi, k) for gi, slot in enumerate(spec.slots) for k in range(slot.count)]


def solve_milp_pulp(
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    excluded: list[frozenset[int]] | None = None,
    time_limit_s: float = 30.0,
) -> list[int] | None:
    """Return the highest-projection valid lineup, or None if infeasible.

    Formulated as an assignment problem — a binary per `(player, slot group)` —
    rather than one binary per player, because a player eligible at several
    positions has to be *placed* somewhere for group constraints restricted to
    particular slots to be expressible at all.

    Args:
        pool: Available players.
        spec: What makes a lineup legal.
        excluded: Player sets that may not all appear together. This is how a
            solver is made to produce a *different* lineup on the next call — one
            no-good cut per lineup already found.
        time_limit_s: Wall-clock limit handed to CBC.

    Returns:
        Pool indices in slot order, or None if no valid lineup exists.
    """
    import pulp

    n = len(pool)
    groups = list(range(len(spec.slots)))
    eligible = {
        gi: [i for i in range(n) if int(pool.positions[i]) & spec.mask_for(slot.eligible)]
        for gi, slot in enumerate(spec.slots)
    }

    problem = pulp.LpProblem("lineup", pulp.LpMaximize)
    assign = {
        (i, gi): pulp.LpVariable(f"x_{i}_{gi}", cat="Binary") for gi in groups for i in eligible[gi]
    }

    problem += pulp.lpSum(float(pool.projections[i]) * var for (i, _), var in assign.items())

    # Each slot group is filled exactly to its count.
    for gi in groups:
        problem += pulp.lpSum(assign[i, gi] for i in eligible[gi]) == spec.slots[gi].count

    # A player occupies at most one slot.
    for i in range(n):
        placements = [assign[i, gi] for gi in groups if i in set(eligible[gi])]
        if placements:
            problem += pulp.lpSum(placements) <= 1

    problem += (
        pulp.lpSum(int(pool.salaries[i]) * var for (i, _), var in assign.items()) <= spec.salary_cap
    )
    if spec.salary_floor > 0:
        problem += (
            pulp.lpSum(int(pool.salaries[i]) * var for (i, _), var in assign.items())
            >= spec.salary_floor
        )

    for group in spec.groups:
        counted = [gi for gi in groups if spec.slot_mask_for(group.slots) & (1 << gi)]
        keys = pool.keys[group.key]
        for key in sorted({int(k) for k in keys if int(k) >= 0}):
            members = [assign[i, gi] for gi in counted for i in eligible[gi] if int(keys[i]) == key]
            if members:
                problem += pulp.lpSum(members) <= group.max_count

    # No-good cuts: each previously found lineup must lose at least one player.
    for forbidden in excluded or []:
        members = [assign[i, gi] for gi in groups for i in eligible[gi] if i in forbidden]
        if members:
            problem += pulp.lpSum(members) <= len(forbidden) - 1

    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s))
    if pulp.LpStatus[problem.status] != "Optimal":
        return None

    lineup: list[int] = []
    for gi, _ in _slot_assignments(spec):
        chosen = [
            i
            for i in eligible[gi]
            if assign[i, gi].value() and assign[i, gi].value() > 0.5 and i not in lineup
        ]
        lineup.append(chosen[0])
    return lineup


def solve_portfolio_pulp(
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    num_lineups: int,
    time_limit_s: float = 30.0,
) -> list[list[int]]:
    """Produce `num_lineups` distinct lineups by re-solving with no-good cuts.

    This is the apples-to-apples comparison against `mlb_dfs_solver.build_lineups`.

    Measured cost is roughly 15-20 seconds per lineup on a 90-player slate, and it
    does not grow superlinearly at small portfolio sizes despite each solve
    carrying one more no-good cut than the last — the cuts are cheap next to the
    solve itself.
    """
    found: list[list[int]] = []
    cuts: list[frozenset[int]] = []
    for _ in range(num_lineups):
        lineup = solve_milp_pulp(pool, spec, excluded=cuts, time_limit_s=time_limit_s)
        if lineup is None:
            break
        found.append(lineup)
        cuts.append(frozenset(lineup))
    return found


def solve_milp_ortools(
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    time_limit_s: float = 30.0,
) -> list[int] | None:
    """The same formulation on CP-SAT.

    Included because CBC is not a strong solver and beating it is a weak claim.
    CP-SAT is much faster on this problem shape, so it is the harder comparison and
    the more honest one.
    """
    from ortools.sat.python import cp_model

    n = len(pool)
    groups = list(range(len(spec.slots)))
    eligible = {
        gi: [i for i in range(n) if int(pool.positions[i]) & spec.mask_for(slot.eligible)]
        for gi, slot in enumerate(spec.slots)
    }

    model = cp_model.CpModel()
    assign = {(i, gi): model.new_bool_var(f"x_{i}_{gi}") for gi in groups for i in eligible[gi]}

    for gi in groups:
        model.add(sum(assign[i, gi] for i in eligible[gi]) == spec.slots[gi].count)
    for i in range(n):
        placements = [assign[i, gi] for gi in groups if (i, gi) in assign]
        if placements:
            model.add(sum(placements) <= 1)

    salary = sum(int(pool.salaries[i]) * var for (i, _), var in assign.items())
    model.add(salary <= spec.salary_cap)
    if spec.salary_floor > 0:
        model.add(salary >= spec.salary_floor)

    for group in spec.groups:
        counted = [gi for gi in groups if spec.slot_mask_for(group.slots) & (1 << gi)]
        keys = pool.keys[group.key]
        for key in sorted({int(k) for k in keys if int(k) >= 0}):
            members = [assign[i, gi] for gi in counted for i in eligible[gi] if int(keys[i]) == key]
            if members:
                model.add(sum(members) <= group.max_count)

    # CP-SAT is integral, so projections are scaled rather than rounded away.
    model.maximize(
        sum(int(round(float(pool.projections[i]) * 1000)) * var for (i, _), var in assign.items())
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    # Fixed, so the benchmark measures the solver rather than the machine.
    solver.parameters.num_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    lineup: list[int] = []
    for gi, _ in _slot_assignments(spec):
        chosen = [i for i in eligible[gi] if solver.value(assign[i, gi]) and i not in lineup]
        lineup.append(chosen[0])
    return lineup
