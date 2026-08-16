"""A crude Gaussian simulator, for the benchmark workflow only.

The library ships no simulator on purpose: simulating a sport well means
modelling that sport, and a package that works for hockey and golf has no
business modelling how baseball scores. The workflow benchmark still needs a
`(players x outcomes)` matrix to run the scoring and selection stages against,
so this one lives here, beside the other fixtures, where its numbers can only
ever describe the harness.

It is the same stand-in `examples/02_pipeline.py` uses: a per-team shock so
that lineups stacking a team move together — the correlation the portfolio
objective needs to have anything to work with — plus independent noise, stored
as `float16` because real simulators store these narrow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mlb_dfs_solver.pool import PlayerPool

__all__ = ["simulate"]


def simulate(pool: PlayerPool, n_outcomes: int = 1_000, seed: int = 7) -> np.ndarray:
    """What every player scored in every simulated outcome."""
    rng = np.random.default_rng(seed)
    teams = pool.keys["team"]
    shock = rng.standard_normal((int(teams.max()) + 1, n_outcomes)) * 0.55
    noise = rng.standard_normal((len(pool), n_outcomes))
    universe = pool.projections[:, None] + pool.stddevs[:, None] * (0.8 * noise + shock[teams])
    return universe.astype(np.float16)
