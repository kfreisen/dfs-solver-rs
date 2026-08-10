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
from math import floor
from typing import TYPE_CHECKING

from mlb_dfs_solver.greedy import assign_locks
from mlb_dfs_solver.spec import scaled_salary

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    extra_conflict_pairs: Sequence[tuple[int, int]] | None = None,
    locks: Sequence[int] | Mapping[int, str | None] | None = None,
    max_exposure: float | Mapping[int, float] | None = None,
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
        extra_conflict_pairs: Pool-index pairs forbidden from sharing a lineup, on
            top of whatever `spec.conflicts` resolves to. Mirrors the kernel's
            `conflict_pairs` argument.
        locks: Players forced into every lineup. Slot assignment goes through the
            same `assign_locks` the wrapper uses — the matching is not part of the
            algorithm being transcribed, and two implementations of it would be
            two chances to disagree about which lineups are even reachable.
        max_exposure: Ceiling on the fraction of returned lineups containing a
            player.

    Returns:
        Lineups as lists of pool indices, in slot order.
    """
    if profiles is None:
        profiles = (((0.1, 0.7), (0.6, 1.6)), ((0.3, 1.5), (0.2, 1.2)))

    n = len(pool)
    if num_lineups <= 0 or n < spec.roster_size:
        return []

    # Locked players, bucketed by the slot group that will hold them.
    locked_by_slot: list[list[int]] = [[] for _ in spec.slots]
    for player, group in assign_locks(pool, spec, locks) if locks else []:
        locked_by_slot[group].append(player)
    is_locked = {player for group in locked_by_slot for player in group}

    # Per-player lineup-count ceilings, against the *requested* portfolio size.
    limits: dict[int, int] = {}
    if max_exposure is not None:
        fractions = (
            {i: float(max_exposure) for i in range(n)}
            if isinstance(max_exposure, (int, float))
            else {int(k): float(v) for k, v in max_exposure.items()}
        )
        limits = {p: floor(f * num_lineups) for p, f in fractions.items()}

    projections = pool.projections.tolist()
    stddevs = pool.stddevs.tolist()
    salaries = pool.salaries.tolist()
    ownership = pool.ownership.tolist()
    positions = pool.positions.tolist()

    slot_eligible = [spec.mask_for(slot.eligible) for slot in spec.slots]
    slot_masks = [spec.slot_mask_for(group.slots) for group in spec.groups]
    keys = {k: pool.keys[k].tolist() for k in spec.group_keys}

    # What each player costs in each slot group. The kernel builds the same table
    # once per solve and takes a row per slot group; here it is a list of lists
    # for the same reason — a captain is priced by the slot, not by the player.
    #
    # Skipped when nothing scales salary, mirroring the kernel's `Uniform` case:
    # every row would be a copy of `salaries`. This is a transcription of a fast
    # path the kernel really has, not an optimization invented here.
    if any(slot.salary_multiplier != 1.0 for slot in spec.slots):
        slot_salaries = [
            [scaled_salary(s, slot.salary_multiplier) for s in salaries] for slot in spec.slots
        ]
    else:
        slot_salaries = [salaries] * len(spec.slots)

    # Conflicts as symmetric adjacency, mirroring the kernel's ConflictGraph. A
    # dict rather than a list indexed by player, so a spec declaring no conflicts
    # allocates nothing — again matching what the kernel does.
    conflicts: dict[int, set[int]] = {}
    pairs = pool.conflict_pairs(spec)
    declared = list(zip(pairs[0].tolist(), pairs[1].tolist(), strict=True))
    declared += list(extra_conflict_pairs or ())
    for a, b in declared:
        if a != b:
            conflicts.setdefault(a, set()).add(b)
            conflicts.setdefault(b, set()).add(a)

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
    #
    # Built from slot-scaled salaries, not raw ones: a captain slot reserving the
    # unmultiplied cheapest player under-reserves by half its cost.
    cheapest: list[list[int]] = []
    for group_idx, group in enumerate(eligible):
        prefix = [0]
        for salary in sorted(slot_salaries[group_idx][i] for i in group):
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
    # How many accepted lineups each capped player has appeared in so far. The
    # kernel keeps the same tally at its merge; here there is only one stream, so
    # the accept step is the merge.
    exposure: dict[int, int] = {}

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

        filled = _fill(
            spec,
            eligible,
            objective,
            slot_salaries,
            tally,
            n,
            cheapest,
            suffix_cost,
            conflicts,
            locked_by_slot,
        )
        if filled is None:
            continue
        lineup, salary = filled

        if salary < spec.salary_floor:
            repaired = _repair_up(
                spec, eligible, slot_salaries, tally, lineup, salary, conflicts, n, is_locked
            )
            if repaired is None:
                continue
            lineup, salary = repaired

        if not (spec.salary_floor <= salary <= spec.salary_cap):
            continue

        if any(exposure.get(p, 0) >= limits[p] for p in lineup if p in limits):
            continue

        key = tuple(sorted(lineup))
        if key not in seen:
            seen.add(key)
            for player in lineup:
                if player in limits:
                    exposure[player] = exposure.get(player, 0) + 1
            out.append(lineup)

    return out


def _fill(
    spec: RosterSpec,
    eligible: list[list[int]],
    objective: list[float],
    slot_salaries: list[list[int]],
    tally: _Tally,
    n: int,
    cheapest: list[list[int]],
    suffix_cost: list[int],
    conflicts: dict[int, set[int]],
    locked_by_slot: list[list[int]],
) -> tuple[list[int], int] | None:
    """Greedily fill every slot; return None if a slot cannot be filled.

    Returns the lineup and its total salary together, because with per-slot
    multipliers the caller can no longer recover the salary by summing raw
    columns — it depends on where each player was rostered.
    """
    used = [False] * n
    # How many rostered players each player conflicts with, mirroring the
    # kernel's counter. A count rather than a flag, because repair removes
    # players and a flag could not tell "unblocked" from "blocked by someone
    # else". Not allocated at all when nothing conflicts, matching the kernel.
    blocked = [0] * n if conflicts else []
    tally.reset()
    lineup: list[int] = []
    salary = 0

    for group_idx, slot in enumerate(spec.slots):
        salaries = slot_salaries[group_idx]
        picked = 0

        # Locked players go in before anything is considered, so the greedy pass
        # sees the budget they have already spent. A lock that will not fit fails
        # the whole attempt: unlike a greedy pick there is no fallback, and
        # dropping it would silently ignore what the caller insisted on.
        for player in locked_by_slot[group_idx]:
            still_needed = min(slot.count - picked - 1, len(cheapest[group_idx]) - 1)
            remaining = cheapest[group_idx][still_needed] + suffix_cost[group_idx + 1]
            if (
                salary + salaries[player] + remaining > spec.salary_cap
                or tally.would_exceed(player, group_idx)
                or (conflicts and blocked[player])
            ):
                return None
            used[player] = True
            tally.add(player, group_idx)
            for other in conflicts.get(player, ()):
                blocked[other] += 1
            salary += salaries[player]
            lineup.append(player)
            picked += 1

        # Descending objective, ties broken by index, matching the Rust sort.
        # The slot's score multiplier is deliberately absent: it scales every
        # candidate alike and so cannot reorder them.
        candidates = sorted(
            (i for i in eligible[group_idx] if not used[i]),
            key=lambda i: (-objective[i], i),
        )
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
            if conflicts and blocked[player]:
                continue
            used[player] = True
            tally.add(player, group_idx)
            for other in conflicts.get(player, ()):
                blocked[other] += 1
            salary += salaries[player]
            lineup.append(player)
            picked += 1
        if picked < slot.count:
            return None
    return lineup, salary


def _repair_up(
    spec: RosterSpec,
    eligible: list[list[int]],
    slot_salaries: list[list[int]],
    tally: _Tally,
    lineup: list[int],
    salary: int,
    conflicts: dict[int, set[int]],
    n: int,
    is_locked: set[int],
) -> tuple[list[int], int] | None:
    """Swap one cheap player for a dearer one to clear the salary floor.

    Cheapest slot first, because replacing the cheapest player leaves the most
    headroom under the cap. A single swap must close the whole gap: searching
    combinations would be exponential, and with thousands of attempts available it
    is cheaper to discard an unrepairable lineup than to work harder on it.

    "Cheapest" means cheapest *as rostered*: a captain at 1.5x can cost more than
    a flex player on a larger base salary.
    """
    needed = spec.salary_floor - salary
    if needed <= 0:
        return lineup, salary

    slot_of: list[int] = []
    for group_idx, slot in enumerate(spec.slots):
        slot_of.extend([group_idx] * slot.count)

    in_lineup = set(lineup)
    # Recomputed from the finished lineup rather than threaded out of the fill:
    # the two are the same set of marks, and deriving it here keeps the fill's
    # bookkeeping local.
    blocked = [0] * n if conflicts else []
    for player in lineup:
        for other in conflicts.get(player, ()):
            blocked[other] += 1

    def cost(slot_index: int) -> int:
        return slot_salaries[slot_of[slot_index]][lineup[slot_index]]

    order = sorted(range(len(lineup)), key=lambda s: (cost(s), s))

    for slot_index in order:
        outgoing = lineup[slot_index]
        if outgoing in is_locked:
            continue
        group_idx = slot_of[slot_index]
        salaries = slot_salaries[group_idx]
        tally.add(outgoing, group_idx, delta=-1)
        in_lineup.discard(outgoing)
        for other in conflicts.get(outgoing, ()):
            blocked[other] -= 1

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
            if conflicts and blocked[candidate]:
                continue
            tally.add(candidate, group_idx)
            replaced = list(lineup)
            replaced[slot_index] = candidate
            return replaced, new_total

        tally.add(outgoing, group_idx)
        in_lineup.add(outgoing)
        for other in conflicts.get(outgoing, ()):
            blocked[other] += 1

    return None


def is_valid(
    lineup: Sequence[int],
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    extra_conflict_pairs: Sequence[tuple[int, int]] | None = None,
) -> bool:
    """Whether a lineup satisfies every rule in the specification.

    Written independently of the construction code above so that it is a real
    check rather than a restatement — a bug shared between builder and validator
    would otherwise be invisible.
    """
    if len(lineup) != spec.roster_size:
        return False
    if len(set(lineup)) != len(lineup):
        return False

    salary = sum(
        scaled_salary(int(pool.salaries[i]), m)
        for i, m in zip(lineup, spec.salary_multipliers(), strict=True)
    )
    if salary > spec.salary_cap or salary < spec.salary_floor:
        return False

    forbidden = {
        (int(a), int(b))
        for a, b in zip(*pool.conflict_pairs(spec).tolist(), strict=True)  # type: ignore[call-overload]
    }
    forbidden |= {(int(a), int(b)) for a, b in extra_conflict_pairs or ()}
    for a in lineup:
        for b in lineup:
            if (int(a), int(b)) in forbidden or (int(b), int(a)) in forbidden:
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


def validity_report(
    lineups: np.ndarray,
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    extra_conflict_pairs: Sequence[tuple[int, int]] | None = None,
) -> str:
    """Describe the first invalid lineup, for a readable assertion failure."""
    for row, lineup in enumerate(lineups.tolist()):
        if not is_valid(lineup, pool, spec, extra_conflict_pairs=extra_conflict_pairs):
            salary = sum(
                scaled_salary(int(pool.salaries[i]), m)
                for i, m in zip(lineup, spec.salary_multipliers(), strict=True)
            )
            return (
                f"lineup {row} is invalid: players={lineup} "
                f"salary={salary} (cap={spec.salary_cap}, floor={spec.salary_floor})"
            )
    return ""
