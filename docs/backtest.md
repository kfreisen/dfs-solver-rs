# Backtesting

`dfs_solver.backtest` answers the question that comes after construction and
selection: *given lineups I already have and contests that already happened,
what would have happened?* It ranks your lineups into each contest's realized
field, settles ties the way the operator does, reads the payout table, and
reduces the rows to ROI and its calibration checks.

It knows nothing about how the lineups were chosen. There are no projections
here, no ownership, no stacking rules, no strategy of any kind. That is what
makes it usable by someone whose approach is the opposite of yours, and it is
what stops the arithmetic from quietly favouring one.

```
run_slate()                      roi_table()  period_table()
  lineups + contests + fields  ->  Entries  ->  pit_table()   ev_calibration_table()
                                               ledger.compare()
```

## What goes in

A [`Contest`][dfs_solver.backtest.contest.Contest] is a fee, a field size, a
per-user entry cap, and the operator's published payout tiers. The tiers are
validated strictly — sorted, non-overlapping, starting at rank 1 — because a
table that pays rank 3 twice is a transcription error, and the place to catch
one is before a number is computed from it. Gaps are allowed; operators do
publish tables with unpaid ranks between paid ones.

A [`FieldScores`][dfs_solver.backtest.ranking.FieldScores] is what every
*other* entry in that contest scored. It is kept separate from the contest on
purpose: the same contest is ranked against a realized field in a backtest and
against a modelled one when predicting.

The lineups arrive as three things you already have from the rest of the
package: their simulated scores from
[`score_lineups`][dfs_solver.select.score_lineups], an order from
[`select_portfolio`][dfs_solver.select.select_portfolio], and what each
actually scored.

## Ties

Tournaments pay by rank, and ties at the paid ranks are common. The rule has to
be exactly right:

**Every entry with the same score shares a block of ranks and receives the
average payout over that block.** Two entries tied for first do not each take
first; they take ranks 1 and 2 between them and each receive the mean of what
those ranks pay. Any other rule either invents money or destroys it.

Your own entries follow the same rule, with three consequences worth
stating:

- A lineup of yours that ties with a field entry joins that entry's block.
- Two lineups of yours with the same score share a block with each other.
- A higher-scoring lineup of yours pushes the next one down a rank, exactly
  as a stranger's would. Entering ten lineups is not ten independent draws.

The reported rank is the *top* of the block, and it is capped at the field
size — a lineup below every field entry cannot finish lower than last.

```python
import numpy as np
from dfs_solver.backtest import Contest, FieldScores, PayoutTier, realized_ranks_and_payouts

contest = Contest(
    contest_id=1, entry_fee=1.0, field_size=5, max_entries_per_user=5,
    tiers=(PayoutTier(1, 1, 10.0), PayoutTier(2, 3, 4.0)),  # 10 / 4 / 4 / 0 / 0
)
field = FieldScores.of(np.array([100.0, 90.0, 90.0, 80.0, 70.0]))

ranks, payouts = realized_ranks_and_payouts([90.0, 90.0, 100.0, 50.0], field, contest)
# ranks   -> [3, 3, 1, 5]
# payouts -> [4/3, 4/3, 7.0, 0.0]
```

The 100 ties the field's 100: ranks 1–2 shared, `(10 + 4) / 2 = 7`. The two
90s join the field's two 90s below both 100s: ranks 3–6, capped at 5, so
`(4 + 0 + 0) / 3`. The 50 is below everyone and capped at rank 5.

## Rank is not payout

Every [`Entries`][dfs_solver.backtest.entries.Entries] row records
`lineup_rank` and `lineup_payout` as separate columns. A rank is what happened.
A payout is that rank read through one payout table — and tables get corrected.
An operator restates a prize pool; a transcription is fixed; you discover the
table you scraped was the sample of single-entry cashers and not the official
one. [`rescore`][dfs_solver.backtest.loop.rescore] re-reads the stored ranks
through a new set of contests without re-running anything:

```python
from dfs_solver.backtest import rescore

corrected = rescore(entries, corrected_contests)
```

One approximation, stated in the docstring: a row that was in a tie block was
paid the block's mean, and the block's size is not stored, so rescoring pays
the block's top rank. Re-run the slate when a tie for the win matters.

## Void contests

A rained-out or cancelled slate that the operator refunded leaves a field in
which every entry scored zero. No contest took place; no entry can be ranked
or paid in it; and counting it as a loss — or a win — is a bookkeeping error.
[`is_void`][dfs_solver.backtest.ranking.is_void] is the rule: every score at or
below zero.

The loop *refuses* a void contest rather than skipping it. A run that silently
entered fewer contests than it was given would report totals over a set nobody
chose. Filter deliberately:

```python
from dfs_solver.backtest import FieldScores, SlateContest, is_void

live = [
    SlateContest(contest, FieldScores.of(scores))
    for contest, scores in slate_contests
    if not is_void(scores)
]
```

## The field model

Nothing here simulates opponents; how a field is built is sport-specific. What
the package can do is *measure* one.
[`FieldModel`][dfs_solver.backtest.field.FieldModel] pools the realized scores
of every entry in every comparable earlier contest and asks, for a score `x`,
what fraction of that pool finished at or below it. That fraction is a rank,
and the payout table turns a rank into money —
[`expected_payouts`][dfs_solver.backtest.field.expected_payouts].

It is walk-forward: a contest's reference pool contains only contests that
finished before it. What "comparable" means is yours to say, as an opaque band
key (fee bracket and field-size bracket is the usual choice —
[`band_of`][dfs_solver.backtest.field.band_of] helps). What "before" means is
also yours: an `as_of` key that sorts, with windows in the same units. Dates
with `timedelta`s work; so do integers.

The model is used for one thing, an expected payout recorded as `ev_pred`
beside the realized one, and
[`ev_calibration_table`][dfs_solver.backtest.report.ev_calibration_table]
bins the two against each other. A bad model is visible, not believed.

## Legality

[`check_lineup`][dfs_solver.backtest.legality.check_lineup] re-derives
legality from the [`RosterSpec`][dfs_solver.spec.RosterSpec] alone, in plain
Python, so a lineup the kernel built and one someone typed are held to the
same rules: roster size and distinct players, a complete slot assignment,
salary within the floor and cap priced by slot, every
[`GroupConstraint`][dfs_solver.spec.GroupConstraint] (cap, distinct minimum,
both stacks), and every [`ConflictRule`][dfs_solver.spec.ConflictRule]. It
returns every violation, not the first;
[`assert_legal`][dfs_solver.backtest.legality.assert_legal] raises an
[`IllegalLineupError`][dfs_solver.backtest.legality.IllegalLineupError]
carrying the list.

## A full example

Build candidates, score them, select a portfolio, and enter it into two
contests on the same slate. The simulation and the fields below are synthetic
so the example runs on its own; in use they are your simulator and the
operator's results.

```python
import numpy as np
from dfs_solver import build_lineups, field_line, score_lineups, select_portfolio
from dfs_solver.backtest import (
    FieldCdf, FieldScores, RunManifest, SlateContest,
    assert_legal, pit_table, roi_table, run_slate, synthetic_gpp,
)
from dfs_solver.pool import PlayerPool
from dfs_solver.presets import DK_MLB_CLASSIC

spec = DK_MLB_CLASSIC
pool = PlayerPool.from_records(records, spec)  # your slate

# 1. Candidates, checked against the specification they were built under.
candidates = build_lineups(pool, spec, num_lineups=2_000, seed=1)
for lineup in candidates:
    assert_legal(pool, spec, lineup)

# 2. Simulated outcomes: (players x outcomes), from your simulator.
rng = np.random.default_rng(1)
universe = pool.projections[:, None] + pool.stddevs[:, None] * rng.standard_normal((len(pool), 1_000))
sim_scores = score_lineups(pool, spec, candidates, universe)

# 3. What actually happened: one realized score per player, summed per lineup.
realized_players = pool.projections + pool.stddevs * rng.standard_normal(len(pool))
realized = score_lineups(pool, spec, candidates, realized_players[:, None])[:, 0]

# 4. A prefix-consistent order: the 20 entered into a 20-max are the first 20
#    of the 150 entered into a 150-max.
order = select_portfolio(sim_scores, mode="gpp", line=field_line(sim_scores, 0.95), n_select=150)

# 5. The contests, with every other entry's realized score. Synthetic here.
centre, spread = float(sim_scores.mean()), float(sim_scores.std())
small = synthetic_gpp(20.0, 500, max_entries_per_user=20, contest_id="small")
large = synthetic_gpp(3.0, 20_000, max_entries_per_user=150, contest_id="large")
reference = FieldCdf(np.sort(rng.normal(centre, spread, 50_000)), n_contests=12, window=60)
contests = [
    SlateContest(small, FieldScores.of(rng.normal(centre, spread, 480))),
    SlateContest(large, FieldScores.of(rng.normal(centre, spread, 19_850)), cdf=reference),
]

# 6. Enter, and reduce.
entries = run_slate(
    strategy="gpp-95",
    slate_key="2025-06-01",
    sim_scores=sim_scores,
    order=order,
    realized_scores=realized,
    contests=contests,
    manifest=RunManifest(seed=1, n_sims=1_000, label="example"),
)
for row in roi_table(entries):
    print(row["strategy"], f"{row['roi']:+.1%}", "ex-top", f"{row['roi_ex_top']:+.1%}", "big", row["n_big"])
for row in pit_table(entries):
    print(row["strategy"], "coverage_90", f"{row['coverage_90']:.2f}")
```

One slate proves nothing, and every report function is written for a table
that spans many. Concatenate slates with
[`Entries.concat`][dfs_solver.backtest.entries.Entries.concat], or with
[`merge`][dfs_solver.backtest.entries.merge] when the runs carry a
[`RunManifest`][dfs_solver.backtest.entries.RunManifest] and mixing seeds or
outcome counts by accident would be a mistake worth refusing.

## Reading the reports

[`roi_table`][dfs_solver.backtest.report.roi_table] reports, per strategy,
what was staked and what came back — and three figures that guard against the
ways a tournament backtest flatters itself:

| Column | Reads as |
| --- | --- |
| `roi` | profit over fees |
| `roi_ex_top` | the same with the single most profitable contest removed. Not the "real" ROI — the win happened — but a strategy whose whole edge is one contest is a different thing from one profitable without it |
| `n_big`, `hit_10x` | entries paying 100x and 10x their fee: how often the thing a tournament is entered for actually happened |
| `any_win` | share of contests with at least one cashing entry — a floor, and the first thing to fall when the field model is wrong |

[`period_table`][dfs_solver.backtest.report.period_table] gives the same by
period — a year, usually, because a strategy profitable in aggregate and losing
in each of the last two seasons is one whose edge has gone.

Two tables are not about money.
[`pit_table`][dfs_solver.backtest.report.pit_table] histograms each lineup's
realized score as a percentile of its own simulated distribution: uniform when
the simulation is calibrated, humped low when it runs hot, U-shaped when its
tails are too thin.
[`ev_calibration_table`][dfs_solver.backtest.report.ev_calibration_table]
bins the field model's expected payout against what was realized. A good ROI
on a bad model reads as luck rather than evidence, and these two say which you
have.

## Against your real ledger

A backtest that agrees with itself proves nothing.
[`compare`][dfs_solver.backtest.ledger.compare] joins the operator's export of
what you actually entered — one
[`LedgerEntry`][dfs_solver.backtest.ledger.LedgerEntry] per real entry — to
the backtest's rows per contest, real fees and winnings beside backtested, and
[`totals`][dfs_solver.backtest.ledger.totals] sums the contests both sides
have. Where they diverge is where the backtest is wrong about something — the
field, the table, the lineups it thinks you played — and each divergence has a
cause that can be found.

## What is still not here

- **A simulator.** The `(players x outcomes)` matrix is yours, as everywhere in
  this package.
- **A model of who else enters.** The field model is an empirical pooled
  distribution of realized scores from earlier contests — measured, not
  generated, and without ownership or any notion of how the public builds.
- **Entry sizing.** The loop enters every contest to its cap or to the order's
  length, whichever is smaller. How many to enter is a strategy; pass a shorter
  order.
- **Any strategy at all.** No ownership, no stacks, no projections.
