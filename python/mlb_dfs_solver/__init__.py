"""Fast roster optimization.

`mlb_dfs_solver` builds valid lineups under salary, position, and group constraints. The
hot path is Rust; this package is the typed, documented surface over it.

```python
from mlb_dfs_solver import PlayerPool, build_lineups
from mlb_dfs_solver.presets import DK_MLB_CLASSIC

pool = PlayerPool.from_records(records, DK_MLB_CLASSIC)
lineups = build_lineups(pool, DK_MLB_CLASSIC, num_lineups=500, seed=1)
```

Construction answers which rosters are *legal*, not which are good. To pick a
portfolio, generate many more candidates than you need and select from them
against simulated outcomes — see `mlb_dfs_solver.select`.

The compiled kernel lives in `mlb_dfs_solver._native`, which is private: it takes flat
arrays chosen for cheap marshalling and gives no diagnostics. Everything supported
is re-exported here, so pinning to these names insulates you from changes to the
boundary.
"""

from __future__ import annotations

from mlb_dfs_solver._native import __version__ as _native_version
from mlb_dfs_solver._native import active_isa
from mlb_dfs_solver.greedy import (
    CONTRARIAN,
    STANDARD,
    JitterProfile,
    assign_locks,
    build_lineups,
)
from mlb_dfs_solver.pool import PlayerPool
from mlb_dfs_solver.select import (
    field_line,
    portfolio_value,
    score_lineups,
    select_portfolio,
    tail_line,
)
from mlb_dfs_solver.spec import ConflictRule, GroupConstraint, RosterSpec, Slot

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

__version__ = "0.0.1.dev0"


def native_version() -> str:
    """Return the version of the compiled kernel.

    Tracks the Python package version but is reported separately: a mismatch means
    a stale build artifact is on the path, which otherwise presents as baffling
    behavior rather than an import error.
    """
    return str(_native_version)
