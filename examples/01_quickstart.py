import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import mlb_dfs_solver
    import numpy as np
    from mlb_dfs_solver import PlayerPool, build_lineups
    from mlb_dfs_solver.presets import DK_MLB_CLASSIC

    return DK_MLB_CLASSIC, PlayerPool, build_lineups, mo, np, mlb_dfs_solver


@app.cell
def _(mo, mlb_dfs_solver):
    mo.md(
        f"""
        # mlb_dfs_solver — building a lineup pool

        Version `{mlb_dfs_solver.__version__}`, kernel `{mlb_dfs_solver.native_version()}`,
        SIMD path `{mlb_dfs_solver.active_isa()}`.

        That last one is worth noticing: wheels are built portably, without
        `target-cpu=native`, so the vector path is chosen when the process starts
        rather than when the wheel was compiled. A wheel built on a machine with
        AVX-512 that assumed AVX-512 would crash on a machine without it.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## A slate

        Real slates come from a contest export. This one is generated so the
        notebook runs anywhere, offline, in well under a second — which is also
        what lets CI execute it on every change and catch it rotting.
        """
    )
    return


@app.cell
def _(DK_MLB_CLASSIC, PlayerPool):
    # Two things here are load bearing, and getting either wrong makes the whole
    # notebook measure an artefact rather than the package.
    #
    # Projections are distinct per player. The obvious `4.0 + (k % 12) * 1.2`
    # gives twelve distinct (salary, projection) pairs and clones of everybody,
    # which makes "distinct lineups" mean swapping interchangeable players.
    #
    # Projection is not a monotone function of salary either. If it were, every
    # lineup spending the cap would score about the same and there would be no
    # optimization to do. Mispriced players are the reason this problem exists.
    records = []
    for position_index, position in enumerate(("P", "C", "1B", "2B", "3B", "SS", "OF")):
        count = 30 if position == "OF" else 10
        for k in range(count):
            records.append(
                {
                    "name": f"{position}-{k}",
                    "positions": (position,),
                    "salary": 2500 + (k % 12) * 750,
                    "projection": (
                        4.0
                        + (k % 12) * 1.2
                        + (k // 12) * 0.29
                        + position_index * 0.037
                        + (((k * 13 + position_index * 5) % 7) - 3) * 0.9
                    ),
                    "stddev": 3.0 + (k % 4),
                    "ownership": ((k * 7) % 30) / 100.0,
                    # Not `k % 10`: salary is driven by k, so that mapping puts
                    # every expensive player on one team and makes every
                    # team-shaped constraint measure the price bracket instead.
                    "team": f"TM{(7 * k + 3 * position_index) % 10}",
                }
            )

    pool = PlayerPool.from_records(records, DK_MLB_CLASSIC)
    len(pool)
    return pool, records


@app.cell
def _(mo):
    mo.md(
        """
        ## The specification

        `DK_MLB_CLASSIC` is data, not a code path — nothing in mlb_dfs_solver branches on
        which sport it is looking at. Note the two team constraints: "at most 6 from
        a team" and "at most 5 *hitters* from a team" are the same kind of rule,
        differing only in which slots they count. Five hitters plus that team's
        starting pitcher is a legal and popular shape, which is why the second
        constraint cannot simply be derived from the first.

        Presets carry only what the operator enforces. DraftKings sets no salary
        floor, so neither does the preset — a floor is a strategy, and one is
        added below the way you would add any other.
        """
    )
    return


@app.cell
def _(DK_MLB_CLASSIC):
    from dataclasses import replace

    # Leaving salary unspent is usually a mistake, so require nearly all of it.
    spec = replace(DK_MLB_CLASSIC, salary_floor=49_000)
    return replace, spec


@app.cell
def _(mo, spec):
    mo.md(
        "```\n"
        + "\n".join(
            [
                f"roster size : {spec.roster_size}",
                f"slots       : {' '.join(spec.slot_names())}",
                f"salary      : {spec.salary_floor:,} – {spec.salary_cap:,}",
                *[
                    f"group       : max {g.max_count} per {g.key}"
                    + (f", counting {', '.join(g.slots)}" if g.slots else ", counting every slot")
                    for g in spec.groups
                ],
            ]
        )
        + "\n```"
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Build

        `seed` and `chunks` together fix the output completely. Work is split across
        a fixed number of chunks rather than across however many cores happen to be
        available, so this returns the same lineups on a laptop and on a 64-core
        server.
        """
    )
    return


@app.cell
def _(build_lineups, pool, spec):
    lineups = build_lineups(pool, spec, num_lineups=500, seed=1)
    lineups.shape
    return (lineups,)


@app.cell
def _(lineups, mo, pool, spec):
    salaries = pool.salary_of(lineups, spec)
    projections = pool.projection_of(lineups, spec)

    mo.md(
        f"""
        **{len(lineups)} distinct lineups.**

        | | min | median | max |
        | --- | ---: | ---: | ---: |
        | salary | {salaries.min():,} | {int(sorted(salaries)[len(salaries) // 2]):,} | {salaries.max():,} |
        | projection | {projections.min():.1f} | {sorted(projections)[len(projections) // 2]:.1f} | {projections.max():.1f} |

        Every one is inside the salary band `{spec.salary_floor:,}`–`{spec.salary_cap:,}`,
        fills every slot with an eligible player, and respects both team caps.
        """
    )
    return projections, salaries


@app.cell
def _(mo):
    mo.md(
        """
        ## Why not just ask a solver?

        Because the pool is the product, not any single lineup.

        An ILP solver returns the *optimal* lineup and beats this on that one
        lineup. Ask it for 500 and it returns the optimum, then the second best,
        then the third — each the previous one with a player swapped, and all of
        them drawn from a narrow slice of the slate.

        Whether that set or this one suits your contest is not something this
        notebook can answer; it depends on the payout structure and on how far
        you trust the projections. What it can show is how the two differ, so
        the numbers below describe this pool rather than grading it.
        """
    )
    return


@app.cell
def _(lineups, mo, pool):
    distinct_players = len(set(lineups.ravel().tolist()))
    overlaps = [len(set(lineups[0].tolist()) & set(row)) for row in lineups[1:].tolist()]
    mean_overlap = sum(overlaps) / len(overlaps)

    mo.md(
        f"""
        - **{distinct_players} of {len(pool)} players** appear somewhere in the pool.
        - A given lineup shares **{mean_overlap:.1f} of 10** players with the first one
          on average.

        Both numbers are properties of what came back — count them off the array
        yourself. The equivalent figures for a solver enumerating with no-good
        cuts are on the
        [benchmarks page](https://kfreisen.github.io/mlb-dfs-solver/benchmarks/),
        measured rather than asserted here.
        """
    )
    return distinct_players, mean_overlap, overlaps


@app.cell
def _(lineups, mo, pool):
    mo.md("**A sample lineup**\n\n" + "\n".join(f"- {name}" for name in pool.names_of(lineups[0])))
    return


if __name__ == "__main__":
    app.run()
