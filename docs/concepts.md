# How it works

This page is the background. It explains what problem daily fantasy optimization
actually is, why the obvious approach is the wrong one, and what each piece of
this package does about it. The [API reference](api.md) says what the functions
take; this says why they exist.

## Two problems

**One roster.** A few hundred players with a salary, a projection and a position;
fill the slots, stay under the cap, maximize projected points. This is an integer
program. `benchmarks/baselines/milp.py` solves it exactly in under a second and
beats everything else here. If you enter one lineup, use it.

**Many rosters.** Contests take up to 150 entries, and a field simulation takes a
hundred thousand. Repeatedly solving and forbidding the last answer gives the top
N by projection, which is a different object from N independent good rosters.

This package addresses the second. It is a weak optimizer per lineup and a fast
one per thousand, and everything below follows from that trade.

## What a solver gives you for 150 lineups

The obvious extension is to ask the solver for the best lineup, forbid it, ask
again, and repeat. This works, and produces the top 150 rosters by projection.

What that set looks like is the thing to understand, and it is directly
observable — no simulation required:

<!-- headline -->

The solver's entries are individually better: it returns the top rosters by
projection, so its median entry sits at or near the optimum and ours does not.
The same run builds those 150 rosters out of a fraction of the slate, and its
best and worst entries differ by a fraction of a point. Ours spread wider and
draw on roughly twice as many players.

That is the trade, stated as two properties of the output rather than as a
verdict. Which one you want depends on the contest and on how much you trust
your projections — if they are exactly right, the solver's set is correct and
diversity is a cost. Neither of those is something this package can tell you.

**What is not on the table**: any claim about what these lineups would have
scored. Scoring needs a simulator and a field model, both of which this package
deliberately does not provide, and a benchmark that supplies its own is grading
its own fixture. See [what is not here](#what-is-not-here).

Where the two stop being alternatives is scale. The next stage needs a candidate
pool far larger than the portfolio, and at ten thousand lineups the solver's
per-lineup cost puts it in a different category — hours against a fraction of a
second. That, rather than any quality argument, is why construction is
randomized.

## Three stages

The package splits the work into three pieces that know nothing about each other.

```
build_lineups()          score_lineups()            select_portfolio()
  what is legal      ->    what might happen    ->    what to enter
```

### 1. Construction — what is legal

[`build_lineups`][mlb_dfs_solver.greedy.build_lineups] produces a large pool of
valid rosters. It perturbs every player's value, sorts, and fills slots greedily,
repairing the salary total when it lands under the floor. Then it does that
thousands of times with different random draws.

It is a weak optimizer by design. A solver beats it on any single lineup; its job
is throughput, so that something good is in the pool for the next stage to find.
What it does enforce strictly is legality — see
[what a contest can require](#what-a-contest-can-require).

### 2. Simulation — what might happen

Selection needs a matrix: what every player scored in every simulated outcome.

**This package does not produce that matrix**, and that is the one omission that
constrains everything else. Simulating baseball means modelling batting order,
park factors, handedness and the correlation between a team's hitters, none of
which generalizes to the other sports the rest of this library handles.
[`score_lineups`][mlb_dfs_solver.select.score_lineups] takes the matrix and sums
each roster's players out of it, applying slot multipliers on the way.

The quality of what comes out of stage 3 is bounded by this matrix, not by
anything in this package. Budget accordingly.

Correlation comes along for free. Two lineups stacking the same team index the
same rows, so they rise and fall together with no extra machinery — which is
exactly the property that makes them poor portfolio companions, and the next
stage will notice.

### 3. Selection — what to enter

[`select_portfolio`][mlb_dfs_solver.select.select_portfolio] chooses which
candidates to actually enter. A candidate is scored not on its own merit but on
what it *adds* to the entries already chosen:

```
gain(c) = mean over outcomes of  max(0, score[c] - what the portfolio already gets)
```

A candidate that duplicates one you hold contributes zero at every outcome. A
candidate that wins in outcomes you currently lose contributes a great deal. So
the portfolio spreads out on its own, and there is no diversity penalty anywhere
in the code — adding one would double-count something the objective already does,
and would break the mathematical property the method depends on.

That property is **submodularity**: the value of adding a lineup shrinks as the
portfolio grows. It licenses two things. Greedy selection is within `1 - 1/e` of
the best possible portfolio, and no polynomial-time algorithm does better unless
P = NP — so greedy is not a shortcut here, it is the answer. And Minoux's *lazy
evaluation* can skip recomputing candidates that cannot possibly win this round,
producing an identical portfolio for much less work.

Both are properties of the algorithm, provable from the objective. Note what they
are guarantees *about*: the portfolio is within `1 - 1/e` of the best portfolio
**under the outcome matrix you supplied**. If that matrix is a poor model of the
sport, selection will optimize against it faithfully and the guarantee will hold
exactly while the result is worthless. The bound is on the search, not on the
simulation — and the simulation is yours.

## Cash and tournaments are different problems

The most important knob is `mode`, and it has no default, because the two
contest types want genuinely different objectives rather than different weights
on one.

| | Cash game (50/50, double-up) | Tournament (GPP) |
| --- | --- | --- |
| Payout | flat, for beating a line | almost everything in the extreme tail |
| Entry judged | alone | by what it adds |
| Duplicate entries | fine — both cash | wasted — cover the same outcomes |
| Objective | `P(score ≥ line)` per entry | `P(any entry ≥ line)` |
| Structure | modular; greedy is *exactly* optimal | submodular; greedy is within `1 - 1/e` |

In a cash game diversity is harmful, not merely unnecessary: if you have the
roster most likely to beat the line, the next thing to enter is the second most
likely. Cash mode therefore ignores the portfolio, which makes the objective
modular and ranking exactly optimal rather than an approximation.

## The line, and why it is a vector

Every mode needs a score to beat, and it should be one value **per outcome**, not
a constant.

A slate's total scoring varies far more between outcomes than lineups vary within
any one outcome. Against a fixed bar, "did this lineup clear it?" mostly asks
"was it a high-scoring slate?" — which is no edge, because every rival entry also
scored more in those worlds. What pays is beating the field *in the same world*.

Take the bar from a model of the field: the entries other people submit. That is
what you are ranked against, and it is the input that makes the whole selection
stage mean anything.

[`field_line`][mlb_dfs_solver.select.field_line] will read a per-outcome quantile
off a score matrix, but note what it computes if you hand it your own candidates
— a bar your own pool exceeds by construction. Use it on a field matrix, or
compute the quantile yourself. **This package does not model a field**, so if you
have no field model, the line is the weakest part of your pipeline and no amount
of selection machinery repairs it.

## What a contest can require

The specification is sport-independent. Every rule below is data, not a code
path, and nothing in the package branches on which sport it is looking at.

| Rule | Expressed as |
| --- | --- |
| Positional slots, flex slots | [`Slot`][mlb_dfs_solver.spec.Slot] with an eligibility mask |
| Salary cap and floor | `RosterSpec.salary_cap`, `.salary_floor` |
| "At most 6 from one team" | `GroupConstraint(max_count=...)` |
| "At most 5 *hitters* from one team" | the same, with `slots=` naming the hitter slots |
| "Players from 2 different games" | `GroupConstraint(min_distinct=2)` |
| Showdown captain worth and costing 1.5× | `Slot(score_multiplier=1.5, salary_multiplier=1.5)` |
| "At least 4 hitters from one team" | `GroupConstraint(min_stack=4)` |
| "No hitters against my pitcher" | [`ConflictRule`][mlb_dfs_solver.spec.ConflictRule] |
| Locked players | `build_lineups(locks=...)` |
| Exposure caps | `select_portfolio(max_exposure=...)`, or `build_lineups(max_exposure=...)` |

The first four collapse into one another, which is what makes this general: "at
most 6 from a team" and "at most 5 hitters from a team" are the same constraint
differing only in which slots they count.

Three do not collapse, and each needed its own mechanism. A **showdown captain**
is worth and costs more than the same player elsewhere, so value became a
property of the slot rather than the player. A **conflict** is a property of a
*pair* — the pitcher's opponent matching the hitter's team is a join, not a
grouping. And a **stack** is existential: it asks that *some* team be well
represented without saying which, so the builder draws a team per attempt, which
is also what spreads a portfolio's stacks across the slate instead of piling them
onto one.

Two of those are strategies rather than rules, and no preset turns them on.
Avoiding a pitcher's opposing hitters is a choice; a contrarian may want exactly
that correlation, and an optimizer that quietly forbade it would be wrong for
them.

## Putting it in a pipeline

Three things to hand over and two to hand back.

**The pool.** [`PlayerPool.from_records`][mlb_dfs_solver.pool.PlayerPool.from_records]
takes a sequence of mappings, so an adapter from whatever objects your projection
layer produces is a comprehension. Multi-position eligibility is a list of names;
team, opponent and game are ordinary keys.

**The universe.** Any float dtype and any layout. Simulators keep outcome
matrices narrow — a hundred thousand candidates by ten thousand outcomes is
gigabytes — so `float16` is accepted and converted exactly, and an
iteration-major matrix can be handed over as a transposed view without being made
contiguous first.

**The lineups back.** An `(n, roster_size)` array of pool indices, columns in
`spec.slot_names()` order. Mapping those back to your own player objects, or to a
contest upload file, is a lookup.

### Expected payout, not just probability

Selection reads a matrix of *values* per outcome. Nothing requires those to be
points. Hand it a `(lineups x outcomes)` matrix of **payouts in dollars** with
`mode="excess"` and `line=0.0`, and the objective becomes

```text
E[max payout across the portfolio]
```

which is the standard objective for a top-heavy contest. It stays submodular
because payouts are non-negative, so lazy evaluation and the `1 - 1/e` guarantee
carry over unchanged. This is not a special mode; it is what the general one
already computes.

### A ceiling you already estimated

If you have a real upside estimate — a conformal P95, say — it does not need a
column of its own. The objective values a player at `projection + t * stddev`
with `t` drawn per attempt, so

```text
stddev = (ceiling - projection) / 1.645
```

makes `t = 1.645` land exactly on your ceiling. Nothing is lost: only the upside
term is ever used, so an asymmetric right-tail estimate does not drag an implied
downside along with it.

### Memory

Selection holds the whole score matrix: `candidates x outcomes x 4 bytes`. Twenty
thousand candidates against ten thousand outcomes is 800 MB, which is fine;
a hundred thousand is 4 GB, which may not be. If you need that many, score and
select in two passes — a coarse outcome sample to shortlist, then the full
resolution on the survivors — rather than reaching for a narrower dtype. Storing
scores as `float16` was measured and costs real quality, because coverage counts
outcomes above a line and a 0.06-point error flips the ones sitting on it.

## What belongs in here, and what does not

The rule for new per-player inputs: **it belongs here only if the objective
already models that quantity and is currently guessing at it.** Projection,
standard deviation and ownership pass — the objective uses all three. A ceiling
fails, because `stddev` already carries it.

The rule exists because the predecessor to this library accumulated fifteen
calibration weights in one scoring function, each defensible alone and jointly
impossible to reason about.

Preferences about *roster composition* are not inputs at all. Most are already
reachable:

| Want | Reach for |
| --- | --- |
| Prefer popular players | a negative `leverage` exponent — it inverts the fade |
| Concentrate on one team | `GroupConstraint(min_stack=...)` |
| Avoid a pitcher's opposing hitters | [`ConflictRule`][mlb_dfs_solver.spec.ConflictRule] |
| Spread the portfolio off its favourites | `build_lineups(diversity_weight=...)` — see [spreading a portfolio](#spreading-a-portfolio) |
| Cap how often a player is used | `select_portfolio(max_exposure=...)` — see [exposure caps](#exposure-caps) |
| Always use a player | `locks` |
| Weight a player up or down | adjust their projection before building |

The division the rest of this follows: **player facts are columns, roster rules
are the specification, portfolio rules are selection, and calibration constants
live in the caller.**

### Pricing

`build_lineups(value_weight=...)` sets how strongly salary is priced into a
player's rank, as a multiple of the pool's own points-per-dollar rate. It changes
only the ordering; a lineup is still worth the sum of its players' projections.

| `value_weight` | Effect |
| --- | --- |
| `0.0` | rank by projection alone. Overspends early and reaches the last slots with no budget; measured, the best candidate capped at 0.93 of the optimum however large the pool grew |
| `0.75` (default) | flat quality through `1.0`, with more of the pool retained |
| `1.0` | rank by surplus over what a point costs on average. Found the exact optimum on the slate it was tested against |
| `1.25` | cliff. The candidate pool collapsed from 15,000 distinct lineups to 500, every attempt converging on the same cheap players |

Pricing narrows the pool even at the default — a sharper objective makes attempts
agree more often. On the benchmark slate the pool fell from 20,000 distinct
lineups to 15,000 while every quality measure improved.

### Spreading a portfolio

`build_lineups(diversity_weight=...)` fades a player already used by the lineups
built so far. Their value drops by `weight × share × mean_projection`, where
`share` is the fraction of accepted lineups containing them.

Measured on a 288-player slate, 10,000 lineups:

| `diversity_weight` | distinct players | mean overlap | median entry |
| ---: | ---: | ---: | ---: |
| off | 92 | 35.0% | 132.6 |
| `0.6` | 119 | 17.6% | 129.9 |
| `1.0` | 133 | 13.8% | 128.6 |

It is a **preference, not a constraint**, and that is the whole reason it sits
beside `max_exposure` instead of replacing it. A cap rejects a finished lineup at
the merge and costs yield; this steers construction before the lineup exists, so
the portfolio spreads without anything being discarded. Use the cap for a hard
ceiling on named players, and this to spread everything else — which is the
answer to not wanting to write down a ceiling for all 288.

Each chunk fades against its own accepted lineups, so a chunk needs a few
attempts before the mechanism does anything. The kernel guarantees that by
treating `chunks` as an upper bound and clamping it — no arithmetic is required
of the caller. That clamp also fixes two neighbouring defects: below it the
jitter profiles stopped alternating, and the attempt budget ran over.

Two alternatives were built and measured before settling here. A **global
snapshot**, refreshed in sequential waves, spread exposure just as well but left
mean overlap unchanged at 35% — every chunk reads the same snapshot, so they all
avoid the same players and all converge on the same replacements. A **maximum
overlap constraint**, the Hamming rule a solver would use, reduced overlap only
once the threshold fell near the mean, and needed 2.5× the candidate pool to fill
the same portfolio.

### Exposure caps

Caps are available at both stages, and **the stage matters more than the number.**

`build_lineups(max_exposure=...)` applies the cap when parallel chunks are
merged: over-cap lineups are discarded rather than rebuilt. That lowers yield,
and because the limit is `floor(cap × lineups *requested*)`, a lower yield raises
the realized share. Tightening the cap can therefore *increase* the exposure you
actually get:

| Requested cap | Lineups returned (of 10,000) | Realized top exposure |
| ---: | ---: | ---: |
| none | 10,000 | 82.9% |
| 60% | 8,650 | 69.4% |
| 40% | 5,930 | 67.5% |
| 25% | 3,963 | 63.1% |

`select_portfolio(max_exposure=...)` applies the cap while choosing from a pool
you have already built, so it skips over-cap candidates instead of discarding
them. Given a large enough pool it fills more of the portfolio and holds closer
to the number you asked for.

**Build uncapped, cap at selection.** Use the construction-stage cap only when
you are not running selection at all, and read the realized exposure rather than
trusting the requested one.

## What is not here

Stated plainly, because a library's gaps matter as much as its features.

- **Simulation.** By design, as above.
- **A field model.** The score that wins is a property of who else entered.
  Modelling that means ownership projections and a model of how the public
  builds — real work, and sport-specific.
- **Minimums per key value.** "Every team used must contribute at least two" is
  not expressible.
- **An exposure floor.** "Roster the ace in about 37% of entries." Caps bound
  from above and locks pin at 100%, with nothing in between.
- **An accurate exposure cap.** The limit is `floor(cap × lineups requested)`,
  and a cap lowers yield, so the realized share exceeds the cap whenever fewer
  lineups come back than were asked for. See [exposure caps](#exposure-caps).

## Reading the numbers

Every figure on this page comes from [the benchmarks](benchmarks.md), which are
run on a named machine and committed rather than measured in CI.

They measure two things and only two: **how long generation takes**, and **what
it generates** — player coverage, exposure, the spread of projections, the
rosters themselves. Both are properties of the output and of inputs you supply.

They deliberately do not measure what a portfolio would have won. That needs a
simulator and a field model, this package provides neither, and any benchmark
that supplied its own would be reporting on that fixture. The
[methodology](methodology.md) page covers the rest.
