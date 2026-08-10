"""Shared fixtures.

`benchmarks/baselines/` is not part of the installed package — it is the "before"
that the "after" is measured against, and shipping it would put a deliberately slow
implementation in users' site-packages. Tests still need it, so it goes on the path
here rather than becoming a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.presets import DK_MLB_CLASSIC
from mlb_dfs_solver.spec import GroupConstraint, RosterSpec, Slot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))


@pytest.fixture
def mlb_spec() -> RosterSpec:
    """DraftKings MLB classic."""
    return DK_MLB_CLASSIC


@pytest.fixture
def tiny_spec() -> RosterSpec:
    """A four-player sport, small enough to reason about by hand."""
    return RosterSpec(
        positions=("P", "C", "OF"),
        slots=(Slot("C", ("C",)), Slot("OF", ("OF",), count=2), Slot("P", ("P",))),
        salary_cap=30_000,
        salary_floor=0,
        groups=(GroupConstraint(key="team", max_count=3),),
    )


def make_records(
    n_per_position: int = 8, *, positions: tuple[str, ...] = ("P", "C", "OF")
) -> list[dict[str, object]]:
    """Synthetic players with a spread of salary, projection, and ownership.

    Deterministic and hand-written rather than random: a fixture that changes
    between runs turns a real failure into a flake.
    """
    records: list[dict[str, object]] = []
    for position in positions:
        for k in range(n_per_position):
            records.append(
                {
                    "name": f"{position}{k}",
                    "positions": (position,),
                    "salary": 3000 + k * 600,
                    "projection": 5.0 + k * 1.5,
                    "stddev": 2.0 + (k % 3),
                    "ownership": (k % 5) / 10.0,
                    "team": f"T{k % 4}",
                }
            )
    return records


@pytest.fixture
def tiny_pool(tiny_spec: RosterSpec) -> PlayerPool:
    """A pool matching `tiny_spec`, comfortably large enough to build from."""
    return PlayerPool.from_records(make_records(), tiny_spec)


def make_mlb_records(per_position: int = 10) -> list[dict[str, object]]:
    """A realistic-shaped DraftKings MLB slate."""
    records: list[dict[str, object]] = []
    for position in ("P", "C", "1B", "2B", "3B", "SS", "OF"):
        # Outfield needs three starters, so slates carry far more of them.
        count = per_position * 3 if position == "OF" else per_position
        for k in range(count):
            records.append(
                {
                    "name": f"{position}-{k}",
                    "positions": (position,),
                    "salary": 2500 + (k % 12) * 750,
                    "projection": 4.0 + (k % 12) * 1.2,
                    "stddev": 3.0 + (k % 4),
                    "ownership": ((k * 7) % 30) / 100.0,
                    "team": f"TM{k % 10}",
                }
            )
    return records


@pytest.fixture
def mlb_pool(mlb_spec: RosterSpec) -> PlayerPool:
    """A pool matching `DK_MLB_CLASSIC`."""
    return PlayerPool.from_records(make_mlb_records(), mlb_spec)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, for tests that need arbitrary but fixed numbers."""
    return np.random.default_rng(20260810)
