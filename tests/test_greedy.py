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
from mlb_dfs_solver.spec import GroupConstraint, RosterSpec


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
