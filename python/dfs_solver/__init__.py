"""Fast roster optimization.

`dfs_solver` builds valid lineups under salary, position, and group constraints. The
hot path is Rust; this package is the typed, documented surface over it.

```python
from dfs_solver import PlayerPool, build_lineups
from dfs_solver.presets import DK_MLB_CLASSIC

pool = PlayerPool.from_records(records, DK_MLB_CLASSIC)
lineups = build_lineups(pool, DK_MLB_CLASSIC, num_lineups=500, seed=1)
```

Construction answers which rosters are *legal*, not which are good. To pick a
portfolio, generate many more candidates than you need and select from them
against simulated outcomes — see `dfs_solver.select`.

The compiled kernel lives in `dfs_solver._native`, which is private: it takes flat
arrays chosen for cheap marshalling and gives no diagnostics. Everything supported
is re-exported here, so pinning to these names insulates you from changes to the
boundary.
"""

from __future__ import annotations

from importlib.metadata import version as _distribution_version

from dfs_solver._native import __version__ as _native_version
from dfs_solver._native import active_isa
from dfs_solver.greedy import (
    CONTRARIAN,
    STANDARD,
    JitterProfile,
    assign_locks,
    build_lineups,
)
from dfs_solver.pool import PlayerPool
from dfs_solver.select import (
    field_line,
    portfolio_value,
    score_lineups,
    select_portfolio,
    tail_line,
)
from dfs_solver.spec import ConflictRule, GroupConstraint, RosterSpec, Slot

__all__ = [
    "CONTRARIAN",
    "STANDARD",
    "ConflictRule",
    "GroupConstraint",
    "JitterProfile",
    "PlayerPool",
    "RosterSpec",
    "Slot",
    "__version__",
    "active_isa",
    "assign_locks",
    "build_lineups",
    "field_line",
    "native_version",
    "portfolio_value",
    "score_lineups",
    "select_portfolio",
    "tail_line",
]

# Read from the installed distribution rather than restated here. The compiled
# kernel makes this package impossible to import without installing it, so the
# metadata is always present, and one fewer copy is one fewer thing to forget on
# a release. The argument is the distribution name, which differs from the import
# name: you `pip install dfs-solver-rs` and `import dfs_solver`.
__version__ = _distribution_version("dfs-solver-rs")


def native_version() -> str:
    """Return the version of the compiled kernel.

    Tracks the Python package version but is reported separately: a mismatch means
    a stale build artifact is on the path, which otherwise presents as baffling
    behavior rather than an import error.
    """
    return str(_native_version)
