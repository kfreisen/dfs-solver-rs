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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from mlb_dfs_solver import _native
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.spec import RosterSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["CONTRARIAN", "STANDARD", "JitterProfile", "build_lineups"]


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

Alternating this with [`STANDARD`][mlb_dfs_solver.greedy.STANDARD] reaches lineups
neither profile finds alone — without it the pool collapses onto the same
high-ceiling core.
"""

_DEFAULT_PROFILES: tuple[JitterProfile, ...] = (CONTRARIAN, STANDARD)


@dataclass(frozen=True, slots=True)
class _Encoded:
    """Spec and config flattened into the arrays the kernel expects."""

    slot_eligible: np.ndarray
    slot_counts: np.ndarray
    group_key_columns: np.ndarray
    group_max_counts: np.ndarray
    group_slot_masks: np.ndarray
    key_columns: np.ndarray
    profiles: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


def _encode(spec: RosterSpec, pool: PlayerPool, profiles: Sequence[JitterProfile]) -> _Encoded:
    """Flatten the specification and profiles for the kernel.

    Done once per call. The key matrix is `(n_columns, n_players)` row-major, and
    group constraints index into it by column, so a single `team` column is shared
    by both the total cap and the hitter cap rather than being sent twice.
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
    column_of = {key: i for i, key in enumerate(key_order)}

    return _Encoded(
        slot_eligible=np.asarray(
            [spec.mask_for(slot.eligible) for slot in spec.slots], dtype=np.uint32
        ),
        slot_counts=np.asarray([slot.count for slot in spec.slots], dtype=np.uint64),
        group_key_columns=np.asarray([column_of[g.key] for g in spec.groups], dtype=np.uint64),
        group_max_counts=np.asarray([g.max_count for g in spec.groups], dtype=np.uint32),
        group_slot_masks=np.asarray(
            [spec.slot_mask_for(g.slots) for g in spec.groups], dtype=np.uint64
        ),
        key_columns=key_columns,
        profiles=np.asarray(
            [[*p.ceiling, *p.leverage] for p in profiles], dtype=np.float64
        ).ravel(),
    )


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
) -> np.ndarray:
    """Build a pool of distinct, valid lineups.

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
        chunks: Independent work units. Fixed rather than derived from the CPU
            count, so results do not depend on the machine. Changing it changes
            the output.
        profiles: Jitter profiles cycled across attempts. Defaults to
            `(CONTRARIAN, STANDARD)`.

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
        from mlb_dfs_solver import build_lineups
        from mlb_dfs_solver.presets import DK_MLB_CLASSIC
        from mlb_dfs_solver.pool import PlayerPool

        pool = PlayerPool.from_records(records, DK_MLB_CLASSIC)
        lineups = build_lineups(pool, DK_MLB_CLASSIC, num_lineups=500, seed=1)
        print(lineups.shape, pool.salary_of(lineups).max())
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

    selected = tuple(profiles) if profiles is not None else _DEFAULT_PROFILES
    if not selected:
        msg = "profiles is empty; supply at least one, or pass None for the default"
        raise ValueError(msg)

    encoded = _encode(spec, pool, selected)

    return _native.build_lineups(
        np.ascontiguousarray(pool.projections, dtype=np.float64),
        np.ascontiguousarray(pool.stddevs, dtype=np.float64),
        np.ascontiguousarray(pool.salaries, dtype=np.int64),
        np.ascontiguousarray(pool.ownership, dtype=np.float64),
        np.ascontiguousarray(pool.positions, dtype=np.uint32),
        encoded.slot_eligible,
        encoded.slot_counts,
        int(spec.salary_cap),
        int(spec.salary_floor),
        encoded.group_key_columns,
        encoded.group_max_counts,
        encoded.group_slot_masks,
        encoded.key_columns,
        int(num_lineups),
        int(seed),
        float(noise),
        int(attempts_per_lineup),
        int(chunks),
        encoded.profiles,
    )
