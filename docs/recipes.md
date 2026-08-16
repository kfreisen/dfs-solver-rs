# Recipes

[`build_lineups`][mlb_dfs_solver.greedy.build_lineups] takes twelve parameters,
[`select_portfolio`][mlb_dfs_solver.select.select_portfolio] eight, and most of
them interact. This page is the short path through them:
[`mlb_dfs_solver.recipes`][mlb_dfs_solver.recipes] bundles the parameters into
named intents — one per way of playing — and this page shows one recipe per
contest type, what each chose, and which knob to reach for when the output is
not what you want.

A [`Recipe`][mlb_dfs_solver.recipes.Recipe] is a frozen dataclass of the same
explicit parameters, nothing more. Print it to see every choice. Override any
field with `dataclasses.replace`. Its `.build()` and `.select()` are plain
calls to the two functions above — the parity tests assert byte-identical
output — so graduating from a recipe to the full API is deleting the recipe,
not translating it.

Every factory requires `seed`. Output is exactly reproducible from `seed` (and
`chunks`, which recipes never touch), and a hidden seed would be a hidden claim
that two runs agree by accident.

## Cash

A cash game pays a flat amount for beating a line. Each entry is judged alone,
and diversity is actively wrong — the right play is the most probable roster,
then the next.

```python
from mlb_dfs_solver import recipes, score_lineups
from mlb_dfs_solver.presets import DK_MLB_CLASSIC

r = recipes.cash(DK_MLB_CLASSIC, seed=1)
lineups = r.build(pool)
scores = score_lineups(pool, r.spec, lineups, universe)   # universe is yours
entries = lineups[r.select(scores, line=cash_line)]
```

What it chose: a salary floor of `salary_cap - 1_000` (unspent salary is
usually projection given away), `value_weight=1.0` (measured to find the exact
optimum on the slate it was tested against), no diversity of any kind, and
`mode="cash"` with 20 entries.

| Steer | Reach for |
| --- | --- |
| A different floor, or none | `recipes.cash(spec, seed=1, salary_floor=...)`, `0` removes it |
| More or fewer entries | `entries=` |
| The line | Yours to supply — a per-outcome median of a field model beats any constant. See [the line, and why it is a vector](concepts.md#the-line-and-why-it-is-a-vector) |

## Single entry

Be clear-eyed about this one: **on a single roster, a MILP solver wins.** It
proves the optimum; nothing here does. The `single-entry` row in
[the benchmarks](benchmarks.md) exists to show that case rather than hide it.

```python
r = recipes.single_entry(DK_MLB_CLASSIC, seed=1)   # 300 candidates, keep 1
lineups = r.build(pool)
scores = score_lineups(pool, r.spec, lineups, universe)
entry = lineups[r.select(scores, line=cash_line)]
```

Use this when you are already in this pipeline — same pool, same simulator,
same objects — and the gap between the provable best lineup and the best of
three hundred is smaller than your projection error. When you are not, use a
solver: `benchmarks/baselines/milp.py` contains a real one.

## Tournament (GPP / MME)

A tournament pays almost nothing outside the extreme tail, so what matters is
the chance that *some* entry reaches a winning score. Two entries that win in
the same outcomes are largely wasted.

The recipe bundles the build and selection side. Stacks and conflicts are
**your strategy**, stated on the spec, because they need your slot and key
names — a factory that quietly imposed either would be wrong for anyone playing
differently:

```python
from dataclasses import replace
from mlb_dfs_solver import recipes
from mlb_dfs_solver.presets import DK_MLB_CLASSIC
from mlb_dfs_solver.spec import ConflictRule, GroupConstraint

HITTERS = ("C", "SS", "2B", "3B", "1B", "OF")
spec = replace(
    DK_MLB_CLASSIC,
    # A four-hitter team stack ...
    groups=(*DK_MLB_CLASSIC.groups,
            GroupConstraint(key="team", min_stack=4, slots=HITTERS)),
    # ... and no hitters against the rostered pitcher.
    conflicts=(ConflictRule(left_key="opponent", right_key="team",
                            left_positions=("P",), right_positions=HITTERS),),
)

r = recipes.gpp(spec, seed=1)                      # 150 entries, diversity on
lineups = r.build(pool)
scores = score_lineups(pool, spec, lineups, universe)
entries = lineups[r.select(scores, line=win_line, lineups=lineups)]
```

What it chose: `attempts_per_lineup=10` (stacks and conflicts make attempts
fail more often), `diversity_weight=0.6` (fades players the portfolio already
used — measured to take mean overlap from 35% to 18% for a fraction of a point
of median projection), and `mode="gpp"`.

| Steer | Reach for |
| --- | --- |
| Spread wider / tighter | `dataclasses.replace(r, diversity_weight=...)` — `1.0` spreads further at ~3% median projection cost, `0.0` turns it off |
| Hard ceiling on named players | `replace(r, max_exposure=...)` — applied at selection, where a cap skips candidates instead of discarding lineups. See [exposure caps](concepts.md#exposure-caps) |
| Always roster someone | `build_lineups(locks=...)` directly — see [locks and exposure](#locks-and-exposure) |
| A 4-2 secondary stack | Not expressible — `min_stack` is existential over one team. Recorded in `benchmarks/scenarios.py` rather than worked around |
| Chalkier / more contrarian | Your own `profiles=` on `build_lineups` — the `leverage` range fades ownership, and a negative exponent inverts the fade |

## Contest scale — the candidate pool

Selection wants far more candidates than you will enter, and a field model
wants tens of thousands of entries. This is the build where the solver stops
being an alternative at all — see the `contest-scale` benchmark row.

```python
r = recipes.candidate_pool(spec, seed=1)           # 20,000, spread wide
candidates = r.build(pool)
```

There is no `.select()` half — what to keep depends on the contest, which is
one of the recipes above. Build uncapped and cap at selection: the
construction-stage cap discards finished lineups and costs yield, the
selection-stage cap skips candidates from a pool already built.

## Showdown

A single-game contest. The captain slot scores 1.5× and costs 1.5×, which
lives on the *spec* — every stage prices per placement, so nothing else
changes.

```python
from mlb_dfs_solver.presets import DK_MLB_SHOWDOWN
from mlb_dfs_solver.spec import GroupConstraint

spec = replace(
    DK_MLB_SHOWDOWN,
    # DraftKings requires players from both teams; the preset cannot know your
    # records carry a `team` key, so you add it when they do.
    groups=(GroupConstraint(key="team", min_distinct=2),),
)
r = recipes.showdown(spec, seed=1)
```

To pin a specific player into the captain slot, pass a mapping lock to
`build_lineups` directly: `locks={index: "CPT"}`.

## Locks and exposure

Locks and caps interact, and one combination is rejected loudly: a locked
player is in 100% of lineups by construction, so capping them below 100%
raises `ValueError` rather than silently honouring one of the two. The pattern
that works — and the one the `mme` benchmark scenario uses — is **lock your
core, cap everyone else**:

```python
locks = [ace, bargain_bat]
caps = {i: 0.6 for i in range(len(pool)) if i not in locks}
lineups = build_lineups(pool, spec, num_lineups=150, seed=1,
                        locks=locks, max_exposure=caps)
```

Read the realized exposure, not the number you passed: the limit is
`floor(cap × lineups requested)`, a cap lowers yield, and a lower yield raises
the realized share. If you run selection, cap there instead — it holds much
closer to what you asked for. The arithmetic is laid out in
[exposure caps](concepts.md#exposure-caps).

## How the knobs interact

The parameters are not independent, and the interactions are the part nobody
guesses. The load-bearing ones:

| Interaction | What holds |
| --- | --- |
| `seed` + `chunks` → output | These two fix the output exactly, on any machine, at any core count. Change either and the lineups change; change anything else about the machine and they do not |
| `chunks` × everything | An **upper bound**, not a count. The kernel clamps it to `total_attempts // 4` so every chunk gets enough attempts for profiles to cycle and diversity to act. Above the clamp, two values give identical output |
| `diversity_weight` vs `max_exposure` | A preference vs a constraint. The weight steers construction before a lineup exists (no yield loss); the cap discards finished lineups at the merge (yield loss, and realized exposure above the requested cap). Spread with the weight, ceiling with the cap — at selection |
| `max_exposure` denominator | The limit is `floor(cap × requested)`, at build and at selection. Fewer lineups back ⇒ realized share exceeds the cap; tightening a build-stage cap can *raise* realized exposure |
| `noise = 0` | Collapses per-player randomness; diversity comes from the profile draw alone. Not a way to make output "cleaner" |
| `value_weight` | Changes ordering only — lineup value stays raw projection. Flat quality from 0.75 to 1.0; a cliff at 1.25 where the pool collapses |
| `attempts_per_lineup` | Buys yield under tight constraints (floors, stacks, conflicts make attempts fail). The recipes raise it to 10 exactly where those bind |
| `salary_floor` | Lives on the `RosterSpec`, not the call — it is a strategy stated as a rule. Presets ship without one; the cash recipe adds one |
| `locks` × `max_exposure` | A lock capped below 100% raises `ValueError`. Cap everyone *else* |

Everything on this page is a starting point. The measured tables behind each
default are in the
[`build_lineups`][mlb_dfs_solver.greedy.build_lineups] docstring and
[how it works](concepts.md); the benchmarks that check the claims are in
[benchmarks](benchmarks.md).
