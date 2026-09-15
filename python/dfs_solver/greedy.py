"""Randomized greedy lineup construction.

Builds a large, diverse pool of *valid* lineups quickly. It is not trying to find
the best lineup — an ILP solver will beat it on any single one — because a
portfolio of 150 optimal lineups is 150 nearly identical lineups. Coverage of the
outcome space is what matters downstream, and randomized construction buys it for
a fraction of the cost.

Determinism is part of the contract: output depends on `seed` and `chunks` and
nothing else. Two runs on machines with different core counts agree exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from dfs_solver import _native
from dfs_solver.pool import PlayerPool
from dfs_solver.spec import RosterSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["CONTRARIAN", "STANDARD", "JitterProfile", "assign_locks", "build_lineups"]


@dataclass(frozen=True, slots=True)
class JitterProfile:
    """How much to perturb the objective on one attempt.

    Drawing these per lineup rather than fixing them is what produces diversity: a
    high `ceiling` draw chases upside, a high `leverage` draw fades popular
    players, and the two together reach different corners of the pool.

    Attributes:
        ceiling: Range for the multiplier on each player's standard deviation.
        leverage: Range for the exponent on `(1 - ownership)`.
    """

    ceiling: tuple[float, float]
    leverage: tuple[float, float]

    def __post_init__(self) -> None:
        """Reject inverted ranges, which silently collapse to a constant."""
        for name, (low, high) in (("ceiling", self.ceiling), ("leverage", self.leverage)):
            if high < low:
                msg = f"{name} range {(low, high)} is inverted"
                raise ValueError(msg)


STANDARD = JitterProfile(ceiling=(0.3, 1.5), leverage=(0.2, 1.2))
"""Balanced: mild upside chasing, mild ownership fade."""

CONTRARIAN = JitterProfile(ceiling=(0.1, 0.7), leverage=(0.6, 1.6))
"""Less upside chasing, stronger ownership fade.

Alternating this with [`STANDARD`][dfs_solver.greedy.STANDARD] reaches lineups
neither profile finds alone — without it the pool collapses onto the same
high-ceiling core.
"""

_DEFAULT_PROFILES: tuple[JitterProfile, ...] = (CONTRARIAN, STANDARD)

# Shared empty pair list. Allocating a fresh one per build is not free at the
# scale a three-lineup build runs at, and there is nothing to distinguish two
# empty arrays. Read-only so a caller cannot make it non-empty for everyone.
_NO_PAIRS = np.empty((2, 0), dtype=np.uint32)
_NO_PAIRS.flags.writeable = False


def index_pairs(
    conflict_pairs: Sequence[tuple[int, int]] | np.ndarray, n_players: int | None
) -> np.ndarray:
    """Validate caller-supplied `(i, j)` pairs and return them as an `(n_pairs, 2)` `int64` array.

    The shape is checked, never reshaped. `PlayerPool.conflict_pairs` returns the
    kernel's `(2, n_pairs)` layout, and passing that here used to be flattened and
    re-paired silently: the left row and the right row were zipped into
    neighbouring indices, which forbids unrelated players (linemates, same-team
    hitters) and can make a stack unsatisfiable. A `(2, n)` array with `n != 2`
    is now an error that names the fix. A `(2, 2)` array is read as two pairs,
    one per row, like every other `(n, 2)` input.

    `n_players=None` skips the range check: a legality check only asks whether
    a lineup holds both players of a pair, and a pair naming an index outside the
    pool can never be held.

    Raises:
        ValueError: If the input is not `(n, 2)`, or (with `n_players`) names an
            index outside the pool.
    """
    extra = np.asarray(conflict_pairs, dtype=np.int64)
    if extra.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if extra.ndim != 2 or extra.shape[1] != 2:
        hint = (
            " That is the (2, n_pairs) layout PlayerPool.conflict_pairs returns: pass `.T`,"
            " or leave conflict_pairs out, since spec.conflicts is already resolved against"
            " the pool."
            if extra.ndim == 2 and extra.shape[0] == 2
            else ""
        )
        msg = (
            f"conflict_pairs must be (n_pairs, 2) pool-index pairs, got shape {extra.shape}.{hint}"
        )
        raise ValueError(msg)
    if (extra < 0).any():
        msg = "conflict_pairs must be non-negative pool indices"
        raise ValueError(msg)
    if n_players is not None and int(extra.max()) >= n_players:
        msg = f"conflict_pairs names player index {int(extra.max())} but the pool has {n_players} players"
        raise ValueError(msg)
    return extra


# Shared "no exposure caps" marker. An empty column is how the kernel is told to
# skip the merge-time bookkeeping entirely.
_NO_LIMITS = np.empty(0, dtype=np.uint32)
_NO_LIMITS.flags.writeable = False

# Likewise for locks. Building two empty arrays per call is measurable against a
# build of a handful of lineups, and there is nothing to distinguish them.
_NO_LOCK_PLAYERS = np.empty(0, dtype=np.uint32)
_NO_LOCK_PLAYERS.flags.writeable = False
_NO_LOCK_SLOTS = np.empty(0, dtype=np.uint64)
_NO_LOCK_SLOTS.flags.writeable = False


@dataclass(frozen=True, slots=True)
class _Encoded:
    """Spec and config flattened into the arrays the kernel expects."""

    slot_eligible: np.ndarray
    slot_counts: np.ndarray
    slot_score_multipliers: np.ndarray
    slot_salary_multipliers: np.ndarray
    group_key_columns: np.ndarray
    group_max_counts: np.ndarray
    group_min_distincts: np.ndarray
    group_min_stacks: np.ndarray
    group_secondary_min_stacks: np.ndarray
    group_slot_masks: np.ndarray
    key_columns: np.ndarray
    conflict_left: np.ndarray
    conflict_right: np.ndarray
    profiles: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


def _encode(
    spec: RosterSpec,
    pool: PlayerPool,
    profiles: Sequence[JitterProfile],
    conflict_pairs: np.ndarray,
) -> _Encoded:
    """Flatten the specification and profiles for the kernel.

    Done once per call. The key matrix is `(n_columns, n_players)` row-major, and
    group constraints index into it by column, so a single `team` column is shared
    by both the total cap and the hitter cap rather than being sent twice.

    Conflicts cross as an explicit pair list rather than as rules. Resolving a key
    join needs the pool, which the kernel does not have in a form it could join
    on, and the resolution is `O(n + pairs)` done once against thousands of
    lineups. The kernel symmetrizes what it receives, so pair order is irrelevant.
    """
    key_order: list[str] = list(spec.group_keys)
    for key in key_order:
        if key not in pool.keys:
            available = ", ".join(sorted(pool.keys)) or "none"
            msg = (
                f"the specification constrains {key!r} but the pool has no such key "
                f"column (available: {available})"
            )
            raise KeyError(msg)

    key_columns = (
        np.concatenate([np.asarray(pool.keys[k], dtype=np.int32) for k in key_order])
        if key_order
        else np.empty(0, dtype=np.int32)
    )
    slots = _encode_slots(spec)

    return _Encoded(
        slot_eligible=slots.slot_eligible,
        slot_counts=slots.slot_counts,
        slot_score_multipliers=slots.slot_score_multipliers,
        slot_salary_multipliers=slots.slot_salary_multipliers,
        group_key_columns=slots.group_key_columns,
        group_max_counts=slots.group_max_counts,
        group_min_distincts=slots.group_min_distincts,
        group_min_stacks=slots.group_min_stacks,
        group_secondary_min_stacks=slots.group_secondary_min_stacks,
        group_slot_masks=slots.group_slot_masks,
        key_columns=key_columns,
        conflict_left=np.ascontiguousarray(conflict_pairs[0], dtype=np.uint32),
        conflict_right=np.ascontiguousarray(conflict_pairs[1], dtype=np.uint32),
        profiles=np.asarray(
            [[*p.ceiling, *p.leverage] for p in profiles], dtype=np.float64
        ).ravel(),
    )


@dataclass(frozen=True, slots=True)
class _SlotArrays:
    """The part of the encoding that depends on the specification alone."""

    slot_eligible: np.ndarray
    slot_counts: np.ndarray
    slot_score_multipliers: np.ndarray
    slot_salary_multipliers: np.ndarray
    group_key_columns: np.ndarray
    group_max_counts: np.ndarray
    group_min_distincts: np.ndarray
    group_min_stacks: np.ndarray
    group_secondary_min_stacks: np.ndarray
    group_slot_masks: np.ndarray


@lru_cache(maxsize=32)
def _encode_slots(spec: RosterSpec) -> _SlotArrays:
    """Encode the slot and group arrays, memoized on the specification.

    Seven small arrays built from Python lists costs several microseconds, which
    is invisible against a thousand-lineup build and a double-digit percentage of
    a three-lineup one. A `RosterSpec` is frozen and hashable, and callers reuse
    one across every build, so caching is both safe and effective.

    The arrays are marked read-only before being handed out. They are shared
    between callers, and a mutation would silently change someone else's
    constraints.
    """
    column_of = {key: i for i, key in enumerate(spec.group_keys)}
    arrays = _SlotArrays(
        slot_eligible=np.asarray(
            [spec.mask_for(slot.eligible) for slot in spec.slots], dtype=np.uint32
        ),
        slot_counts=np.asarray([slot.count for slot in spec.slots], dtype=np.uint64),
        slot_score_multipliers=np.asarray(
            [slot.score_multiplier for slot in spec.slots], dtype=np.float64
        ),
        slot_salary_multipliers=np.asarray(
            [slot.salary_multiplier for slot in spec.slots], dtype=np.float64
        ),
        group_key_columns=np.asarray([column_of[g.key] for g in spec.groups], dtype=np.uint64),
        group_max_counts=np.asarray([g.cap for g in spec.groups], dtype=np.uint32),
        group_min_distincts=np.asarray([g.min_distinct for g in spec.groups], dtype=np.uint32),
        group_min_stacks=np.asarray([g.min_stack for g in spec.groups], dtype=np.uint32),
        group_secondary_min_stacks=np.asarray(
            [g.secondary_min_stack for g in spec.groups], dtype=np.uint32
        ),
        group_slot_masks=np.asarray(
            [spec.slot_mask_for(g.slots) for g in spec.groups], dtype=np.uint64
        ),
    )
    for field_name in _SlotArrays.__slots__:
        getattr(arrays, field_name).flags.writeable = False
    return arrays


def assign_locks(
    pool: PlayerPool,
    spec: RosterSpec,
    locks: Sequence[int] | Mapping[int, str | None],
) -> list[tuple[int, int]]:
    """Work out which slot group will hold each locked player.

    Returns `(player, slot_group)` pairs. Exposed because the answer is worth
    inspecting: if a set of locks is rejected, seeing the partial assignment is
    usually enough to know which one is the problem.

    This is a bipartite matching, not a greedy walk, and the difference matters.
    Two locks both eligible at flex, only one of whom can also play tight end:
    take flex for the wrong one and the other has nowhere to go, even though a
    legal arrangement exists. Assigning greedily would reject rosters that are
    perfectly fine. Kuhn's algorithm finds a complete assignment whenever one
    exists, and at roster-sized inputs its cost is not worth measuring.

    The assignment depends only on eligibility and slot counts, never on salary
    or projection, so it is settled once here rather than being re-derived on
    every one of thousands of attempts.

    Args:
        pool: Players, used for their position eligibility.
        spec: Specification whose slots the locks are matched against.
        locks: Pool indices, or a mapping from pool index to a slot name that
            pins that lock to one slot group. A `None` value means "any slot the
            player is eligible for", the same as passing a bare sequence.

    Returns:
        One `(player, slot_group)` pair per lock, ordered by player so the result
        is stable however the locks were given.

    Raises:
        ValueError: If a lock is out of range, names an unknown slot, is not
            eligible for the slot it was pinned to, is listed twice, or if no
            complete assignment exists.
    """
    pinned: dict[int, str | None] = (
        dict(locks) if isinstance(locks, dict) else {int(p): None for p in locks}
    )
    if isinstance(locks, dict) or not locks:
        pass
    elif len(pinned) != len(locks):
        msg = "the same player is locked more than once"
        raise ValueError(msg)

    slot_index = {slot.name: i for i, slot in enumerate(spec.slots)}
    # Which slot groups each lock could go in, in slot order.
    options: list[list[int]] = []
    players = list(pinned)
    for player in players:
        if not 0 <= player < len(pool):
            msg = f"locked player index {player} is outside a pool of {len(pool)} players"
            raise ValueError(msg)
        mask = int(pool.positions[player])
        name = pinned[player]
        if name is None:
            groups = [i for i, slot in enumerate(spec.slots) if mask & spec.mask_for(slot.eligible)]
            if not groups:
                msg = f"locked player {player} is not eligible for any slot in this specification"
                raise ValueError(msg)
        else:
            if name not in slot_index:
                known = ", ".join(slot_index)
                msg = f"locked player {player} names slot {name!r}; this specification has: {known}"
                raise ValueError(msg)
            group = slot_index[name]
            if not mask & spec.mask_for(spec.slots[group].eligible):
                msg = f"locked player {player} is not eligible for slot {name!r}"
                raise ValueError(msg)
            groups = [group]
        options.append(groups)

    # Kuhn's algorithm, with each slot group holding up to `count` locks. Trying
    # to displace an existing occupant is what separates this from greedy.
    seats: list[list[int]] = [[] for _ in spec.slots]

    def place(lock: int, visited: set[int]) -> bool:
        for group in options[lock]:
            if group in visited:
                continue
            visited.add(group)
            if len(seats[group]) < spec.slots[group].count:
                seats[group].append(lock)
                return True
            for seat, occupant in enumerate(seats[group]):
                seats[group][seat] = lock
                if place(occupant, visited):
                    return True
                seats[group][seat] = occupant
        return False

    for lock in range(len(players)):
        if not place(lock, set()):
            player = players[lock]
            msg = (
                f"locked player {player} cannot be placed: every slot they are "
                f"eligible for is already taken by another lock. Locks so far: "
                f"{sorted((players[o], g) for g, occ in enumerate(seats) for o in occ)}"
            )
            raise ValueError(msg)

    return sorted(
        (players[lock], group) for group, occupants in enumerate(seats) for lock in occupants
    )


def _exposure_limits(
    pool: PlayerPool,
    max_exposure: float | Mapping[int, float],
    num_lineups: int,
    locked: set[int],
) -> np.ndarray:
    """Turn exposure fractions into per-player lineup counts.

    The count is taken against the *requested* portfolio size, not the running
    total. Deriving it from what has been accepted so far would put the very
    first lineup over any cap below 100%.
    """
    uncapped = np.iinfo(np.uint32).max
    limits = np.full(len(pool), uncapped, dtype=np.uint32)
    items = (
        [(i, float(max_exposure)) for i in range(len(pool))]
        if isinstance(max_exposure, (int, float))
        else [(int(k), float(v)) for k, v in max_exposure.items()]
    )
    for player, fraction in items:
        if not 0.0 <= fraction <= 1.0:
            msg = f"max_exposure for player {player} is {fraction}; it must be a fraction in [0, 1]"
            raise ValueError(msg)
        if not 0 <= player < len(pool):
            msg = f"max_exposure names player index {player} but the pool has {len(pool)} players"
            raise ValueError(msg)
        if player in locked and fraction < 1.0:
            # Both were asked for and they cannot both hold. Saying so beats
            # silently honouring one, which would look like the other was ignored.
            msg = (
                f"player {player} is locked into every lineup but capped at "
                f"{fraction:.0%} exposure; those cannot both be true"
            )
            raise ValueError(msg)
        limits[player] = math.floor(fraction * num_lineups)
    return limits


def build_lineups(
    pool: PlayerPool,
    spec: RosterSpec,
    *,
    num_lineups: int = 200,
    seed: int = 0,
    noise: float = 0.35,
    attempts_per_lineup: int = 3,
    chunks: int = 64,
    profiles: Sequence[JitterProfile] | None = None,
    value_weight: float = 0.75,
    diversity_weight: float = 0.0,
    conflict_pairs: Sequence[tuple[int, int]] | np.ndarray | None = None,
    locks: Sequence[int] | Mapping[int, str | None] | None = None,
    max_exposure: float | Mapping[int, float] | None = None,
) -> np.ndarray:
    """Build a pool of distinct, valid lineups.

    Most parameters interact; [`recipes`][dfs_solver.recipes] bundles them
    into named per-contest-type starting points, and the recipes page in the
    docs maps each interaction.

    Args:
        pool: Available players.
        spec: What makes a lineup legal.
        num_lineups: How many distinct lineups to return.
        seed: Base seed. With `chunks`, fixes the output exactly.
        noise: Uniform noise added to each objective, as a fraction of the
            player's projection. Zero removes the per-player randomness and
            collapses diversity to the profile draw alone.
        attempts_per_lineup: Attempts made per requested lineup before giving up.
            Attempts fail when the greedy fill paints itself into a corner, so a
            tightly constrained pool needs more.
        chunks: Upper bound on independent work units. Fixed rather than derived
            from the CPU count, so results do not depend on the machine.

            An upper bound rather than a literal count. A chunk needs a few
            attempts before the profile cycle advances and `diversity_weight`
            has a history to fade against, so the kernel clamps this to
            `total_attempts // 4`. Taken literally, the default would give a
            20-lineup build one attempt per chunk and silently disable both.

            It therefore changes the output only while it is the binding
            constraint: above the clamp, two different values give the same
            answer.
        profiles: Jitter profiles cycled across attempts. Defaults to
            `(CONTRARIAN, STANDARD)`.
        value_weight: How strongly to price salary into a player's value, as a
            multiple of the pool's own points-per-dollar rate.

            Zero ranks by projection alone, which systematically overspends: the
            fill takes the best remaining player at each slot and reaches the
            last ones with no money left. Measured, that capped the best
            candidate at 0.93 of the proven optimum however large the pool grew.
            At `1.0` a player is ranked by their surplus over what a point costs
            on average, which found the exact optimum on the slate it was tested
            against.

            The default stops short of 1.0 because just past it there is a cliff
            — at 1.25 the candidate pool collapsed from fifteen thousand distinct
            lineups to five hundred, every attempt converging on the same cheap
            players. Between 0.75 and 1.0 quality is flat.

            Pricing narrows the pool even at the default, because a sharper
            objective makes attempts agree more often. That is a real cost and it
            was worth paying: on the benchmark slate the pool fell from twenty
            thousand distinct lineups to fifteen while every quality measure
            improved, portfolio diversity included.

            This changes only the *ordering*. A lineup is still worth the sum of
            its players' real projections.
        diversity_weight: How strongly to fade a player already used by the
            lineups accepted so far. A player's value drops by
            `diversity_weight * share * mean_projection`, where `share` is the
            fraction of accepted lineups containing them. Zero disables it and
            reproduces the original output exactly.

            This is a preference, not a constraint, which is why it sits next to
            `max_exposure` rather than replacing it. A cap rejects a finished
            lineup at the merge and costs yield; this steers construction before
            the lineup exists, so the portfolio spreads without anything being
            discarded. Use the cap to enforce a hard ceiling on specific players
            and this to spread everything else.

            Measured on a 288-player slate at 10,000 lineups, against the same
            build with the weight off:

            | | distinct players | mean overlap |
            | ---: | ---: | ---: |
            | off | 92 | 33.8% |
            | `0.6` | 119 | 18.8% |
            | `1.0` | 133 | 15.0% |

            The cost is projection: `1.0` gave up 3% of the median entry's
            points for those. Each chunk fades against its own accepted lineups,
            so a chunk needs a few attempts before this does anything — the
            kernel clamps `chunks` to guarantee that, and no arithmetic is
            required of the caller.
        conflict_pairs: Extra `(i, j)` pool-index pairs forbidden from sharing a
            lineup, on top of anything `spec.conflicts` resolves to. Index pairs
            live here rather than on the specification because they are a fact
            about *this* pool — the same specification against tomorrow's slate
            would silently forbid two unrelated players. Order within a pair does
            not matter; the relation is symmetrized.
        locks: Players forced into every lineup, as pool indices — or as a
            mapping from pool index to a slot name, to pin a lock to one slot
            (`{7: "CPT"}`). Which slot holds each lock is otherwise worked out by
            [`assign_locks`][dfs_solver.greedy.assign_locks].

            A lock is honoured or the lineup is not produced; it is never quietly
            dropped. If a lock cannot coexist with the cap, a group cap, or
            another lock, the result is empty rather than a lineup without it.
        max_exposure: Ceiling on how many lineups may contain a player,
            expressed as a fraction of `num_lineups` **as requested**. A number
            caps every player; a mapping caps only the players it names.

            Applied when the parallel chunks are merged, which is the only place
            that sees the whole portfolio and still reproduces exactly. Over-cap
            lineups are discarded rather than rebuilt.

            **Read the realized exposure, not the number you passed.** The limit
            is `floor(fraction * num_lineups)` appearances, so whenever discards
            push the yield below the request, the share of what comes back runs
            above the fraction asked for — and tightening the cap can raise it.
            On a slate deep enough to absorb the discards, the cap simply holds.
            Measured on a 432-player slate asking for 10,000 stacked lineups:

            | Requested | Returned | Realized top exposure |
            | ---: | ---: | ---: |
            | none | 10,000 | 50.7% |
            | 60% | 10,000 | 50.7% |
            | 40% | 10,000 | 40.0% |
            | 25% | 10,000 | 25.0% |

            If you are running selection, cap there instead:
            [`select_portfolio`][dfs_solver.select.select_portfolio] skips
            over-cap candidates from a pool you have already built rather than
            discarding from a portfolio being assembled, so it holds much closer
            to the count you asked for. Use this parameter when you are entering
            the constructed lineups directly.

    Returns:
        An `(n, roster_size)` array of indices into `pool`, where columns follow
        `spec.slot_names()`. **`n` may be smaller than `num_lineups`** — that means
        the pool could not support more, which is a real answer rather than a
        failure. A 12-player slate with a salary floor may have only a handful of
        valid lineups.

    Raises:
        ValueError: If the specification, pool, or arguments are inconsistent.
        KeyError: If the specification constrains a key the pool does not carry.

    Example:
        ```python
        from dfs_solver import build_lineups
        from dfs_solver.presets import DK_MLB_CLASSIC
        from dfs_solver.pool import PlayerPool

        pool = PlayerPool.from_records(records, DK_MLB_CLASSIC)
        lineups = build_lineups(pool, DK_MLB_CLASSIC, num_lineups=500, seed=1)
        print(lineups.shape, pool.salary_of(lineups, DK_MLB_CLASSIC).max())
        ```
    """
    if num_lineups < 0:
        msg = f"num_lineups must be non-negative, got {num_lineups}"
        raise ValueError(msg)
    if attempts_per_lineup < 1:
        msg = f"attempts_per_lineup must be at least 1, got {attempts_per_lineup}"
        raise ValueError(msg)
    if chunks < 1:
        msg = f"chunks must be at least 1, got {chunks}"
        raise ValueError(msg)
    if noise < 0:
        msg = f"noise must be non-negative, got {noise}"
        raise ValueError(msg)
    if value_weight < 0:
        msg = f"value_weight must be non-negative, got {value_weight}"
        raise ValueError(msg)
    if diversity_weight < 0:
        msg = f"diversity_weight must be non-negative, got {diversity_weight}"
        raise ValueError(msg)

    selected = tuple(profiles) if profiles is not None else _DEFAULT_PROFILES
    if not selected:
        msg = "profiles is empty; supply at least one, or pass None for the default"
        raise ValueError(msg)

    pairs = _NO_PAIRS if not spec.conflicts else pool.conflict_pairs(spec)
    if conflict_pairs is not None:
        extra = index_pairs(conflict_pairs, len(pool))
        pairs = np.concatenate([pairs, extra.T.astype(np.uint32)], axis=1)

    assigned = assign_locks(pool, spec, locks) if locks else []
    limits = (
        _exposure_limits(pool, max_exposure, num_lineups, {p for p, _ in assigned})
        if max_exposure is not None
        else _NO_LIMITS
    )

    encoded = _encode(spec, pool, selected, pairs)

    # Keyword-only at the boundary: the kernel signature has runs of same-dtype
    # arrays, and a positional call that transposed two of them would build
    # silently wrong lineups.
    return _native.build_lineups(
        projections=np.ascontiguousarray(pool.projections, dtype=np.float64),
        stddevs=np.ascontiguousarray(pool.stddevs, dtype=np.float64),
        salaries=np.ascontiguousarray(pool.salaries, dtype=np.int64),
        ownership=np.ascontiguousarray(pool.ownership, dtype=np.float64),
        positions=np.ascontiguousarray(pool.positions, dtype=np.uint32),
        slot_eligible=encoded.slot_eligible,
        slot_counts=encoded.slot_counts,
        slot_score_multipliers=encoded.slot_score_multipliers,
        slot_salary_multipliers=encoded.slot_salary_multipliers,
        salary_cap=int(spec.salary_cap),
        salary_floor=int(spec.salary_floor),
        group_key_columns=encoded.group_key_columns,
        group_max_counts=encoded.group_max_counts,
        group_min_distincts=encoded.group_min_distincts,
        group_min_stacks=encoded.group_min_stacks,
        group_secondary_min_stacks=encoded.group_secondary_min_stacks,
        group_slot_masks=encoded.group_slot_masks,
        key_columns=encoded.key_columns,
        conflict_left=encoded.conflict_left,
        conflict_right=encoded.conflict_right,
        num_lineups=int(num_lineups),
        seed=int(seed),
        noise=float(noise),
        attempts_per_lineup=int(attempts_per_lineup),
        chunks=int(chunks),
        profiles=encoded.profiles,
        value_weight=float(value_weight),
        diversity_weight=float(diversity_weight),
        lock_players=np.asarray([p for p, _ in assigned], dtype=np.uint32)
        if assigned
        else _NO_LOCK_PLAYERS,
        lock_slot_groups=np.asarray([g for _, g in assigned], dtype=np.uint64)
        if assigned
        else _NO_LOCK_SLOTS,
        exposure_limits=limits,
    )
