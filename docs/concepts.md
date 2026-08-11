# How it works

This page is the background. It explains what problem daily fantasy optimization
actually is, why the obvious approach is the wrong one, and what each piece of
this package does about it. The [API reference](api.md) says what the functions
take; this says why they exist.

## The problem is not the one it looks like

You are given a few hundred players, each with a salary, a projected score, and a
position. You must pick a roster that fills the positional slots and fits under a
salary cap. Maximize projected points.

Stated that way it is an integer program, a solver answers it exactly in under a
second, and there is nothing more to discuss. `benchmarks/baselines/milp.py`
contains that solver, it works, and on this measure it beats everything else
here. If your contest is one entry against a simple payout, use it.

The real question is different in a way that changes the answer completely.
Contests let you enter **many** lineups — a hundred and fifty is a common
maximum — and the payout is wildly top-heavy. A large tournament might return
nothing at all outside the top fifth of the field, and most of the prize pool to
the top fraction of a percent. What you are choosing is not a lineup. It is a
*portfolio*, and it is judged on whether **any** of your entries lands in the
money.

That distinction is the whole package.

## Why asking a solver for 150 lineups does not work

The obvious extension is to ask the solver for the best lineup, forbid it, ask
again, and repeat. This works and produces the top 150 rosters by projection. It
is also close to the worst thing you could enter.

Those 150 lineups differ from one another by a player or two, because that is
what "second best" means. They therefore win together and lose together. If the
expensive shortstop they all share has a quiet night, all 150 entries fail at
once. You have bought one lineup a hundred and fifty times and paid a hundred and
fifty entry fees for it.

Here is that measured, against an independent field, both approaches asked for
the same 150 entries:

<!-- headline -->

Each column is defined precisely in
[what the columns mean](methodology.md#what-the-columns-mean) — in particular
"in the money", which has a field model and a per-outcome payout line behind it.

Every individual lineup the solver produced is *better* than ours — its median
entry is the optimum, because it returned the top 150 by projection. The
portfolio is worse. Per-lineup quality and portfolio quality are close to
opposites here, and only one of them is what a contest pays for.

Speed is the smaller half of the argument and worth stating precisely: a solver
makes 150 lineups in seconds. What it cannot make is the *candidate pool* the
next section needs — twenty thousand lineups by no-good cut is roughly forty
minutes of solving, and they would be the twenty thousand most similar lineups
available.

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

It is deliberately a weak optimizer. A solver beats it on any single lineup and
that is fine, because its job is coverage: produce many *different* legal rosters
cheaply, so that something good is in the pool. Optimality per candidate would be
wasted work — the next stage decides what is good.

What it does take seriously is legality, which is more varied than it sounds. See
[what a contest can require](#what-a-contest-can-require) below.

### 2. Simulation — what might happen

Selection needs to know how lineups perform across the ways a slate can break,
which means a matrix: what every player scored in every simulated outcome.

**This package does not produce that matrix.** It is the one piece deliberately
left out. Simulating baseball well means modelling batting order, park factors,
pitcher handedness, and the correlation between a team's hitters — and none of
that generalizes to hockey or golf, which the rest of this library does.
[`score_lineups`][mlb_dfs_solver.select.score_lineups] takes the matrix and sums
each roster's players out of it, applying slot multipliers on the way.

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

Diversity is not merely unnecessary in a cash game — it is harmful. If you have
found the roster most likely to beat the line, the second-best thing you can
enter is the *next* most likely, not something different for its own sake. Cash
mode therefore ignores the portfolio entirely, which makes it modular, which
makes ranking exactly optimal rather than an approximation.

## The line, and why it is a vector

Every mode needs a score to beat, and it is one value **per outcome**, not a
constant. This is the single easiest thing to get wrong, and getting it wrong
silently produces numbers that look fine.

On a realistic slate the field's median score swings between simulated outcomes
several times more than lineups differ from each other within any one outcome.
The world matters more than the roster. Judged against a fixed bar, "did this
lineup cash?" turns out to be almost entirely a question of whether it was a
high-scoring slate — and that is no edge at all, because every rival entry also
scored more in those worlds. What pays is beating the field *in the same world*.

The benchmark records the size of that swing, under `win_line_spread`; it is
about half the score of an entire lineup.

[`field_line`][mlb_dfs_solver.select.field_line] reads a per-outcome bar off a
score matrix. Better still is a bar computed from a model of the actual field —
the entries other people submit — because that is what you are being ranked
against. A benchmark that draws the line from its own candidates is measuring a
circle, and will report success no matter what it does.

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
| Exposure caps | `build_lineups(max_exposure=...)` |

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

## What is not here

Stated plainly, because a library's gaps matter as much as its features.

- **Simulation.** By design, as above.
- **A field model.** The score that wins is a property of who else entered.
  Modelling that means ownership projections and a model of how the public
  builds — real work, and sport-specific.
- **Payout curves.** Selection optimizes the probability of clearing a line, not
  expected dollars across a payout structure. For a top-heavy tournament these
  are close; for a flat one they are not.
- **Minimums per key value.** "Every team used must contribute at least two" is
  not expressible. Nobody has needed it.
- **Value-aware construction.** The builder ranks by projection and never by
  points per dollar, so on a slate with mispriced players its best candidate
  plateaus short of the true optimum however many you generate — see
  `best_ratio` on the benchmarks page. Selection cannot fix that; better
  construction would.

## Reading the numbers

Every figure on this page comes from [the benchmarks](benchmarks.md), which are
run on a named machine and committed rather than measured in CI. The
[methodology](methodology.md) page explains why, and what a benchmark here does
and does not tell you.
