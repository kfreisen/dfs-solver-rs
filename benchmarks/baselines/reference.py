"""Pure-Python transcription of the greedy construction algorithm.

This is the most valuable file in the package and it is not shipped in the wheel.
It serves three jobs at once:

1. **The specification.** The Rust is fast and the Rust is not readable. This is
   the same algorithm at a speed nobody cares about, written so the algorithm can
   be checked by reading it.
2. **The parity oracle.** `tests/test_parity.py` asserts this and the kernel
   produce *identical* lineups for the same seed. That is what makes a speedup
   number mean something: the two implementations are doing the same job.
3. **The benchmark baseline.** The headline comparison is against MILP, but the
   Python-versus-Rust number is the honest measure of what the port bought.

Because it is an oracle, it must be a *transcription*, not a reimplementation.
Where a more Pythonic expression would change the order of operations, the clumsy
version wins — the lane-by-lane structure here mirrors the Rust deliberately, down
to the tie-breaks and the RNG call sequence.

Determinism note: exact lineup-for-lineup parity with the Rust would require
reimplementing xoshiro256++ and its `random_range` mapping bit for bit. That is
not a useful thing to maintain, so the parity test asserts the invariants that
actually matter — validity, distinctness, and the *distribution* of the objective
— rather than byte-identical output. See `tests/test_parity.py` for exactly what
is claimed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from mlb_dfs_solver.pool import PlayerPool
    from mlb_dfs_solver.spec import RosterSpec


@dataclass
class _Tally:
    """Running group counts, mirroring `GroupTally` in the Rust core."""

    spec: RosterSpec
    keys: dict[str, Sequence[int]]
    slot_masks: list[int]
    counts: list[dict[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.counts = [{} for _ in self.spec.groups]

    def reset(self) -> None:
        for bucket in self.counts:
            bucket.clear()

    def would_exceed(self, player: int, slot_group: int) -> bool:
        for gi, group in enumerate(self.spec.groups):
            if not self.slot_masks[gi] & (1 << slot_group):
                continue
            key = self.keys[group.key][player]
            if key < 0:
                continue
            if self.counts[gi].get(key, 0) >= group.max_count:
                return True
        return False

    def add(self, player: int, slot_group: int, delta: int = 1) -> None:
        for gi, group in enumerate(self.spec.groups):
            if not self.slot_masks[gi] & (1 << slot_group):
                continue
            key = self.keys[group.key][player]
            if key < 0:
                continue
            self.counts[gi][key] = self.counts[gi].get(key, 0) + delta


def build_lineups_reference(
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    num_lineups: int = 200,
    seed: int = 0,
    noise: float = 0.35,
    attempts_per_lineup: int = 3,
    profiles: Sequence[tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> list[list[int]]:
    """Build distinct valid lineups, in readable Python.

    Mirrors `mlb_dfs_solver.build_lineups`, minus the chunking: this runs serially, so
    there is no `chunks` argument and no cross-chunk merge. Everything else — the
    objective, the fill order, the tie-breaks, the repair — is the same.

    Args:
        pool: Available players.
        spec: What makes a lineup legal.
        num_lineups: How many distinct lineups to return.
        seed: Seed for Python's `random`.
        noise: Uniform noise as a fraction of each player's projection.
        attempts_per_lineup: Attempts per requested lineup before giving up.
        profiles: `((ceiling_low, ceiling_high), (leverage_low, leverage_high))`
            pairs, cycled across attempts. Defaults to the contrarian/standard pair.

    Returns:
        Lineups as lists of pool indices, in slot order.
    """
    if profiles is None:
        profiles = (((0.1, 0.7), (0.6, 1.6)), ((0.3, 1.5), (0.2, 1.2)))

    n = len(pool)
    if num_lineups <= 0 or n < spec.roster_size:
        return []

    projections = pool.projections.tolist()
    stddevs = pool.stddevs.tolist()
    salaries = pool.salaries.tolist()
    ownership = pool.ownership.tolist()
    positions = pool.positions.tolist()

    slot_eligible = [spec.mask_for(slot.eligible) for slot in spec.slots]
    slot_masks = [spec.slot_mask_for(group.slots) for group in spec.groups]
    keys = {k: pool.keys[k].tolist() for k in spec.group_keys}

    # Eligible players per slot group, computed once.
    eligible: list[list[int]] = [
        [i for i in range(n) if positions[i] & mask] for mask in slot_eligible
    ]
    if any(len(e) < slot.count for e, slot in zip(eligible, spec.slots, strict=True)):
        return []

    # Cheapest way to finish, per slot group. Mirrors the kernel exactly.
    #
    # The bound must account for distinctness within a group: a group needing three
    # players cannot fill all three with the single cheapest one. `cheapest[j][m]`
    # is the least group `j` can spend on `m` players. Using `m * min` instead
    # under-reserves, and the symptom is the last slot of a multi-slot group being
    # unfillable — the fill takes the one affordable player and has nothing left
    # for its twin.
    #
    # Double counting across groups is not corrected, which keeps this a lower
    # bound — the safe direction, since it can only admit a pick that later proves
    # infeasible, never reject a feasible one.
    cheapest: list[list[int]] = []
    for group in eligible:
        prefix = [0]
        for salary in sorted(salaries[i] for i in group):
            prefix.append(prefix[-1] + salary)
        cheapest.append(prefix)
    suffix_cost = [0] * (len(spec.slots) + 1)
    for j in range(len(spec.slots) - 1, -1, -1):
        take = min(spec.slots[j].count, len(cheapest[j]) - 1)
        suffix_cost[j] = suffix_cost[j + 1] + cheapest[j][take]

    rng = random.Random(seed)
    tally = _Tally(spec, keys, slot_masks)
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []

    for attempt in range(num_lineups * attempts_per_lineup):
        if len(out) >= num_lineups:
            break

        (ceiling_low, ceiling_high), (leverage_low, leverage_high) = profiles[
            attempt % len(profiles)
        ]
        ceiling = rng.uniform(ceiling_low, ceiling_high)
        leverage = rng.uniform(leverage_low, leverage_high)

        objective = [0.0] * n
        for i in range(n):
            value = projections[i] + ceiling * stddevs[i]
            value *= (1.0 - min(max(ownership[i], 0.0), 0.99)) ** leverage
            if noise > 0.0:
                amplitude = max(abs(projections[i] * noise), 0.1)
                value += rng.uniform(-amplitude, amplitude)
            objective[i] = max(value, 0.01)

        lineup = _fill(spec, eligible, objective, salaries, tally, n, cheapest, suffix_cost)
        if lineup is None:
            continue

        salary = sum(salaries[i] for i in lineup)
        if salary < spec.salary_floor:
            repaired = _repair_up(spec, eligible, salaries, tally, lineup, salary)
            if repaired is None:
                continue
            lineup, salary = repaired

        if not (spec.salary_floor <= salary <= spec.salary_cap):
            continue

        key = tuple(sorted(lineup))
        if key not in seen:
            seen.add(key)
            out.append(lineup)

    return out


def _fill(
    spec: RosterSpec,
    eligible: list[list[int]],
    objective: list[float],
    salaries: list[int],
    tally: _Tally,
    n: int,
    cheapest: list[list[int]],
    suffix_cost: list[int],
) -> list[int] | None:
    """Greedily fill every slot; return None if a slot cannot be filled."""
    used = [False] * n
    tally.reset()
    lineup: list[int] = []
    salary = 0

    for group_idx, slot in enumerate(spec.slots):
        # Descending objective, ties broken by index, matching the Rust sort.
        candidates = sorted(
            (i for i in eligible[group_idx] if not used[i]),
            key=lambda i: (-objective[i], i),
        )
        picked = 0
        for player in candidates:
            if picked == slot.count:
                break
            # Reserve enough for the slots still to be filled, including the
            # rest of this group.
            still_needed = min(slot.count - picked - 1, len(cheapest[group_idx]) - 1)
            remaining = cheapest[group_idx][still_needed] + suffix_cost[group_idx + 1]
            if salary + salaries[player] + remaining > spec.salary_cap:
                continue
            if tally.would_exceed(player, group_idx):
                continue
            used[player] = True
            tally.add(player, group_idx)
            salary += salaries[player]
            lineup.append(player)
            picked += 1
        if picked < slot.count:
            return None
    return lineup


def _repair_up(
    spec: RosterSpec,
    eligible: list[list[int]],
    salaries: list[int],
    tally: _Tally,
    lineup: list[int],
    salary: int,
) -> tuple[list[int], int] | None:
    """Swap one cheap player for a dearer one to clear the salary floor.

    Cheapest slot first, because replacing the cheapest player leaves the most
    headroom under the cap. A single swap must close the whole gap: searching
    combinations would be exponential, and with thousands of attempts available it
    is cheaper to discard an unrepairable lineup than to work harder on it.
    """
    needed = spec.salary_floor - salary
    if needed <= 0:
        return lineup, salary

    slot_of: list[int] = []
    for group_idx, slot in enumerate(spec.slots):
        slot_of.extend([group_idx] * slot.count)

    in_lineup = set(lineup)
    order = sorted(range(len(lineup)), key=lambda s: (salaries[lineup[s]], s))

    for slot_index in order:
        outgoing = lineup[slot_index]
        group_idx = slot_of[slot_index]
        tally.add(outgoing, group_idx, delta=-1)
        in_lineup.discard(outgoing)

        for candidate in eligible[group_idx]:
            if candidate in in_lineup:
                continue
            gain = salaries[candidate] - salaries[outgoing]
            if gain < needed:
                continue
            new_total = salary + gain
            if new_total > spec.salary_cap:
                continue
            if tally.would_exceed(candidate, group_idx):
                continue
            tally.add(candidate, group_idx)
            replaced = list(lineup)
            replaced[slot_index] = candidate
            return replaced, new_total

        tally.add(outgoing, group_idx)
        in_lineup.add(outgoing)

    return None


def is_valid(lineup: Sequence[int], pool: PlayerPool, spec: RosterSpec) -> bool:
    """Whether a lineup satisfies every rule in the specification.

    Written independently of the construction code above so that it is a real
    check rather than a restatement — a bug shared between builder and validator
    would otherwise be invisible.
    """
    if len(lineup) != spec.roster_size:
        return False
    if len(set(lineup)) != len(lineup):
        return False

    salary = sum(int(pool.salaries[i]) for i in lineup)
    if salary > spec.salary_cap or salary < spec.salary_floor:
        return False

    cursor = 0
    slot_of: list[int] = []
    for group_idx, slot in enumerate(spec.slots):
        mask = spec.mask_for(slot.eligible)
        for _ in range(slot.count):
            if not int(pool.positions[lineup[cursor]]) & mask:
                return False
            slot_of.append(group_idx)
            cursor += 1

    for group in spec.groups:
        mask = spec.slot_mask_for(group.slots)
        counts: dict[int, int] = {}
        for position, player in enumerate(lineup):
            if not mask & (1 << slot_of[position]):
                continue
            key = int(pool.keys[group.key][player])
            if key < 0:
                continue
            counts[key] = counts.get(key, 0) + 1
        if any(count > group.max_count for count in counts.values()):
            return False

    return True


def validity_report(lineups: np.ndarray, pool: PlayerPool, spec: RosterSpec) -> str:
    """Describe the first invalid lineup, for a readable assertion failure."""
    for row, lineup in enumerate(lineups.tolist()):
        if not is_valid(lineup, pool, spec):
            salary = sum(int(pool.salaries[i]) for i in lineup)
            return (
                f"lineup {row} is invalid: players={lineup} "
                f"salary={salary} (cap={spec.salary_cap}, floor={spec.salary_floor})"
            )
    return ""
