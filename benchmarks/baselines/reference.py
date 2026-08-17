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
    from mlb_dfs_solver.spec import GroupConstraint, RosterSpec


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
            if self.counts[gi].get(key, 0) >= group.cap:
                return True
        return False

    def distinct_count(self, group_index: int) -> int:
        """How many distinct key values this constraint currently sees."""
        return sum(1 for count in self.counts[group_index].values() if count > 0)

    def max_stack(self, group_index: int) -> int:
        """The largest number of players any one key value has contributed."""
        return max(self.counts[group_index].values(), default=0)

    def top_two(self, group_index: int) -> tuple[int, int]:
        """The two largest per-key counts, descending — what a 4-2 is judged on."""
        return self.top_two_with(group_index, -1)

    def top_two_with(self, group_index: int, extra_key: int) -> tuple[int, int]:
        """`top_two` as it would read with one extra player of `extra_key`.

        `-1` adds nobody. Mirrors `GroupTally::top_two_with` in the Rust core:
        how salary repair asks "would this swap keep the stacks?" without
        mutating the tally.
        """
        bucket = dict(self.counts[group_index])
        if extra_key >= 0:
            bucket[extra_key] = bucket.get(extra_key, 0) + 1
        sizes = sorted(bucket.values(), reverse=True)
        return (sizes[0] if sizes else 0, sizes[1] if len(sizes) > 1 else 0)

    def count_of(self, group_index: int, key: int) -> int:
        """How many players this constraint has counted for `key`."""
        return self.counts[group_index].get(key, 0) if key >= 0 else 0

    def key_of(self, group_index: int, player: int) -> int:
        """The key value `player` carries for this constraint."""
        return self.keys[self.spec.groups[group_index].key][player]

    def counts_slot(self, group_index: int, slot_group: int) -> bool:
        """Whether this constraint counts `slot_group`."""
        return bool(self.slot_masks[group_index] & (1 << slot_group))

    def add(self, player: int, slot_group: int, delta: int = 1) -> None:
        for gi, group in enumerate(self.spec.groups):
            if not self.slot_masks[gi] & (1 << slot_group):
                continue
            key = self.keys[group.key][player]
            if key < 0:
                continue
            self.counts[gi][key] = self.counts[gi].get(key, 0) + delta


def _counted_suffix(spec: RosterSpec, slot_masks: list[int]) -> list[list[int]]:
    """`[g][j]`: roster slots constraint `g` still counts from slot group `j` on.

    Knowing how many counted slots are left is what lets a forward fill tell "this
    pick is fine" from "this pick strands the requirement". Without it a distinct
    minimum could only be checked at the end, by which point every attempt that
    was going to fail has already spent a full roster finding out.
    """
    table: list[list[int]] = []
    for gi in range(len(spec.groups)):
        suffix = [0] * (len(spec.slots) + 1)
        for j in range(len(spec.slots) - 1, -1, -1):
            here = spec.slots[j].count if slot_masks[gi] & (1 << j) else 0
            suffix[j] = suffix[j + 1] + here
        table.append(suffix)
    return table


def _capable_keys(
    spec: RosterSpec,
    slot_masks: list[int],
    eligible: list[list[int]],
    keys: dict[str, Sequence[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Key values each stack constraint could plausibly be built around.

    A team with two eligible outfielders cannot supply a five-stack, and drawing
    it as a target would waste the attempt. Bounded per slot group by both the
    slot count and how many of the key's players are eligible there.

    Returns the primary list and the secondary list per group. The secondary bar
    is never higher than the primary, so its list is a superset — which is what
    the pair feasibility test relies on.
    """
    capable: list[list[int]] = []
    secondary_capable: list[list[int]] = []
    for gi, group in enumerate(spec.groups):
        if not group.min_stack:
            capable.append([])
            secondary_capable.append([])
            continue
        reach: dict[int, int] = {}
        for j, slot in enumerate(spec.slots):
            if not slot_masks[gi] & (1 << j):
                continue
            per_key: dict[int, int] = {}
            for player in eligible[j]:
                key = keys[group.key][player]
                if key >= 0:
                    per_key[key] = per_key.get(key, 0) + 1
            for key, available in per_key.items():
                reach[key] = reach.get(key, 0) + min(available, slot.count)
        capable.append(sorted(k for k, r in reach.items() if r >= group.min_stack))
        secondary_capable.append(
            sorted(k for k, r in reach.items() if r >= group.secondary_min_stack)
            if group.secondary_min_stack
            else []
        )
    return capable, secondary_capable


def build_lineups_reference(
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    num_lineups: int = 200,
    seed: int = 0,
    noise: float = 0.35,
    attempts_per_lineup: int = 3,
    profiles: Sequence[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    value_weight: float = 0.75,
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
        value_weight: How strongly to price salary into a player's value, as a
            multiple of the pool's points-per-dollar rate. Mirrors the kernel.
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

    # Salary priced into every projection, once. The pool's own points-per-dollar
    # rate makes the weight unitless. Only the *ordering* reads this; a finished
    # lineup is still worth the sum of the real projections.
    total_salary = float(pool.salaries.sum())
    rate = float(pool.projections.sum()) / total_salary if total_salary > 0 else 0.0
    projections = (pool.projections - value_weight * rate * pool.salaries).tolist()
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
    #
    # Locked players are excluded, and their own cost added back. A lock cannot
    # fill any other slot, and a group holding an expensive lock has a minimum
    # cost far above its cheapest eligible player — reserve the cheap figure and
    # the earlier groups spend the budget the lock needed, making it unaffordable
    # on every attempt.
    cheapest: list[list[int]] = []
    for group_idx, group in enumerate(eligible):
        prefix = [0]
        for salary in sorted(slot_salaries[group_idx][i] for i in group if i not in is_locked):
            prefix.append(prefix[-1] + salary)
        cheapest.append(prefix)
    locked_cost = [
        sum(slot_salaries[j][p] for p in locked_by_slot[j]) for j in range(len(spec.slots))
    ]
    suffix_cost = [0] * (len(spec.slots) + 1)
    for j in range(len(spec.slots) - 1, -1, -1):
        free = spec.slots[j].count - len(locked_by_slot[j])
        take = min(free, len(cheapest[j]) - 1)
        suffix_cost[j] = suffix_cost[j + 1] + locked_cost[j] + cheapest[j][take]

    counted_suffix = _counted_suffix(spec, slot_masks)
    capable, secondary_capable = _capable_keys(spec, slot_masks, eligible, keys)
    # A stack nobody can supply makes every lineup impossible; say so once
    # rather than discovering it attempt by attempt. A 4-2 needs a *pair* of
    # distinct keys, and the secondary list being a superset makes that exactly
    # "at least two keys reach the secondary bar".
    if any(
        g.min_stack
        and (not capable[gi] or (g.secondary_min_stack and len(secondary_capable[gi]) < 2))
        for gi, g in enumerate(spec.groups)
    ):
        return []

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

        # The key value each stack chases this attempt, drawn before the jitter
        # from the same generator. Redrawing per attempt is the mechanism: the
        # constraint asks for "some" key value, and answering that question the
        # same way every time would stack the whole portfolio on one team.
        stack_targets = [
            rng.choice(capable[gi]) if g.min_stack else -1 for gi, g in enumerate(spec.groups)
        ]
        # The secondary target is a *different* key, drawn right after its
        # primary, mirroring the kernel's draw order.
        secondary_targets = [
            rng.choice([k for k in secondary_capable[gi] if k != stack_targets[gi]])
            if g.secondary_min_stack
            else -1
            for gi, g in enumerate(spec.groups)
        ]

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
            counted_suffix,
            stack_targets,
            secondary_targets,
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
    counted_suffix: list[list[int]],
    stack_targets: list[int],
    secondary_targets: list[int],
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
    has_minimums = any(g.min_distinct or g.min_stack for g in spec.groups)
    has_stacks = any(g.min_stack for g in spec.groups)
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
        free_slots = min(slot.count - len(locked_by_slot[group_idx]), len(cheapest[group_idx]) - 1)
        free_floor = cheapest[group_idx][free_slots]
        locks_to_come = sum(salaries[p] for p in locked_by_slot[group_idx])
        for player in locked_by_slot[group_idx]:
            locks_to_come -= salaries[player]
            remaining = locks_to_come + free_floor + suffix_cost[group_idx + 1]
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
        # Two passes when a stack still wants players this group could supply:
        # the first takes only players that feed it, the second everything else.
        # Filling a stack from the best available stack players rather than
        # whatever is left at the end is what makes it a stack worth having.
        # Both passes walk the same sorted list, so ordering is untouched.
        for stack_pass in (True, False) if has_stacks else (False,):
            for player in candidates:
                if picked == slot.count:
                    break
                if used[player]:
                    continue
                if stack_pass and not _feeds_a_stack(
                    spec, tally, stack_targets, secondary_targets, player, group_idx
                ):
                    continue
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
                if has_minimums and _strands_a_minimum(
                    spec, tally, counted_suffix, player, group_idx, picked
                ):
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

    # Distinct minimums are guaranteed by construction — no pick that would make
    # one unreachable is ever taken. A stack is not: the drawn key may simply not
    # have had enough affordable players in the slots that were left. A pair is
    # judged on the two largest stacks actually present, not the drawn targets.
    if has_stacks and not all(
        _stack_holds(tally, gi, g) for gi, g in enumerate(spec.groups) if g.min_stack
    ):
        return None
    return lineup, salary


def _stack_holds(tally: _Tally, group_index: int, group: GroupConstraint) -> bool:
    """Whether one group's stack requirement — single or 4-2 pair — is met."""
    if not group.secondary_min_stack:
        return tally.max_stack(group_index) >= group.min_stack
    best, second = tally.top_two(group_index)
    return best >= group.min_stack and second >= group.secondary_min_stack


def _feeds_a_stack(
    spec: RosterSpec,
    tally: _Tally,
    stack_targets: list[int],
    secondary_targets: list[int],
    player: int,
    slot_group: int,
) -> bool:
    """Whether a stack still owed players wants this one — either target."""
    for gi, group in enumerate(spec.groups):
        if not group.min_stack or not tally.counts_slot(gi, slot_group):
            continue
        key = tally.key_of(gi, player)
        target = stack_targets[gi]
        if target >= 0 and key == target and tally.count_of(gi, target) < group.min_stack:
            return True
        secondary = secondary_targets[gi]
        if (
            secondary >= 0
            and key == secondary
            and tally.count_of(gi, secondary) < group.secondary_min_stack
        ):
            return True
    return False


def _strands_a_minimum(
    spec: RosterSpec,
    tally: _Tally,
    counted_suffix: list[list[int]],
    player: int,
    slot_group: int,
    picked: int,
) -> bool:
    """Whether taking `player` leaves a distinct minimum out of reach.

    Exact, not a heuristic: after this pick a known number of counted slots
    remain and the requirement needs a known number of further distinct values.
    If the second exceeds the first the attempt is already lost, and continuing
    would only discover that after spending the rest of the roster on it.
    """
    for gi, group in enumerate(spec.groups):
        if not group.min_distinct:
            continue
        counts_here = tally.counts_slot(gi, slot_group)
        left_in_group = slot_group_remaining(spec, slot_group, picked) if counts_here else 0
        remaining = left_in_group + counted_suffix[gi][slot_group + 1]
        gain = 1 if counts_here and tally.count_of(gi, tally.key_of(gi, player)) == 0 else 0
        still_needed = max(0, group.min_distinct - (tally.distinct_count(gi) + gain))
        if still_needed > remaining:
            return True
    return False


def slot_group_remaining(spec: RosterSpec, slot_group: int, picked: int) -> int:
    """Slots left in `slot_group` after the pick being considered."""
    return spec.slots[slot_group].count - picked - 1


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
    has_minimums = any(g.min_distinct or g.min_stack for g in spec.groups)

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
            # The outgoing player is already out of the tally, so this sees the
            # lineup the swap would produce. The lineup is otherwise complete —
            # there are no further picks to fix a requirement this breaks — so
            # the test is for "met", not "still reachable".
            if has_minimums and not _swap_keeps_minimums(spec, tally, candidate, group_idx):
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


def _swap_keeps_minimums(spec: RosterSpec, tally: _Tally, candidate: int, slot_group: int) -> bool:
    """Whether bringing `candidate` in still satisfies every minimum."""
    for gi, group in enumerate(spec.groups):
        if not group.min_distinct and not group.min_stack:
            continue
        counts_here = tally.counts_slot(gi, slot_group)
        key = tally.key_of(gi, candidate) if counts_here else -1
        gain = 1 if counts_here and tally.count_of(gi, key) == 0 and key >= 0 else 0
        if tally.distinct_count(gi) + gain < group.min_distinct:
            return False
        if group.min_stack:
            # The tally already has the outgoing player removed; ask what the
            # top counts would be with the candidate's key bumped by one.
            best, second = tally.top_two_with(gi, key)
            if best < group.min_stack:
                return False
            if group.secondary_min_stack and second < group.secondary_min_stack:
                return False
    return True


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
        if any(count > group.cap for count in counts.values()):
            return False
        if len(counts) < group.min_distinct:
            return False
        if group.min_stack and max(counts.values(), default=0) < group.min_stack:
            return False
        if group.secondary_min_stack:
            sizes = sorted(counts.values(), reverse=True)
            if len(sizes) < 2 or sizes[1] < group.secondary_min_stack:
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
