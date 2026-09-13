"""Verifying lineups you already have against contests that already happened.

The rest of this package answers "which rosters are legal?" and "which should I
enter?". This subpackage answers the question that comes after: *given a set of
lineups and the contests they would have gone into, what would have happened?*

It knows nothing about how the lineups were chosen. There are no projections
here, no ownership, no stacking rules, no strategy of any kind — only what a
contest pays, where a score finishes in a realized field, whether a lineup
satisfies its specification, and the arithmetic that turns entries into a
return. That is deliberate: the same machinery has to be trusted by someone
whose approach is the opposite of yours.

```
run_slate()                      roi_table()  period_table()
  lineups + contests + fields  ->  Entries  ->  pit_table()   ev_calibration_table()
                                               ledger.compare()
```

What goes in:

* [`Contest`][dfs_solver.backtest.contest.Contest] — a fee, a field size, a
  per-user cap, and the operator's published payout tiers.
* [`FieldScores`][dfs_solver.backtest.ranking.FieldScores] — what every other
  entry in that contest scored.
* An order of candidate lineups, their simulated scores, and their realized
  ones.

What comes out is an [`Entries`][dfs_solver.backtest.entries.Entries] table,
one row per lineup per contest, and the report functions reduce it. Nothing is
hidden in a dataframe: every column is a NumPy array.

Two design choices carry the honesty of the numbers:

* **Ties are settled as operators settle them.** Every entry with the same
  score shares a block of ranks and takes the block's mean payout; our own
  entries join blocks with field entries and with each other. See
  [`ranking`][dfs_solver.backtest.ranking].
* **Rank is stored separately from payout.** A rank is what happened; a payout
  is that rank read through a table that may later be corrected.
  [`rescore`][dfs_solver.backtest.loop.rescore] re-reads without re-running.

What this subpackage still does not ship: a simulator, or a model of who else
enters. [`FieldModel`][dfs_solver.backtest.field.FieldModel] is an empirical
pooled distribution of realized scores from earlier contests, walk-forward and
measured rather than generated, and it is used for one thing — an expected
payout whose calibration is reported beside the realized one, so a bad model
is seen and not believed.
"""

from __future__ import annotations

from dfs_solver.backtest.contest import (
    Contest,
    PayoutTier,
    payout_table_from_tiers,
    synthetic_gpp,
    synthetic_gpp_tiers,
    validate_tiers,
)
from dfs_solver.backtest.entries import COLUMNS, Entries, RunManifest, merge
from dfs_solver.backtest.field import (
    FieldCdf,
    FieldModel,
    ReferenceContest,
    band_of,
    expected_payouts,
    pooled_cdf,
)
from dfs_solver.backtest.ledger import LedgerEntry, compare, totals
from dfs_solver.backtest.legality import IllegalLineupError, assert_legal, check_lineup
from dfs_solver.backtest.loop import SlateContest, rescore, run_slate
from dfs_solver.backtest.ranking import (
    FieldScores,
    is_void,
    realized_ranks_and_payouts,
    tie_block_payouts,
    tie_blocks,
)
from dfs_solver.backtest.report import (
    BIG_MULTIPLE,
    EV_BIN_EDGES,
    ev_calibration_table,
    period_table,
    pit_table,
    roi_table,
)

__all__ = [
    "BIG_MULTIPLE",
    "COLUMNS",
    "EV_BIN_EDGES",
    "Contest",
    "Entries",
    "FieldCdf",
    "FieldModel",
    "FieldScores",
    "IllegalLineupError",
    "LedgerEntry",
    "PayoutTier",
    "ReferenceContest",
    "RunManifest",
    "SlateContest",
    "assert_legal",
    "band_of",
    "check_lineup",
    "compare",
    "ev_calibration_table",
    "expected_payouts",
    "is_void",
    "merge",
    "payout_table_from_tiers",
    "period_table",
    "pit_table",
    "pooled_cdf",
    "realized_ranks_and_payouts",
    "rescore",
    "roi_table",
    "run_slate",
    "synthetic_gpp",
    "synthetic_gpp_tiers",
    "tie_block_payouts",
    "tie_blocks",
    "totals",
    "validate_tiers",
]
