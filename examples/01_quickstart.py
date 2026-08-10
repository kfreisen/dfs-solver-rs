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
    records = []
    for position in ("P", "C", "1B", "2B", "3B", "SS", "OF"):
        count = 30 if position == "OF" else 10
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
        """
    )
    return


@app.cell
def _(DK_MLB_CLASSIC, mo):
    mo.md(
        "```\n"
        + "\n".join(
            [
                f"roster size : {DK_MLB_CLASSIC.roster_size}",
                f"slots       : {' '.join(DK_MLB_CLASSIC.slot_names())}",
                f"salary      : {DK_MLB_CLASSIC.salary_floor} – {DK_MLB_CLASSIC.salary_cap}",
                *[
                    f"group       : max {g.max_count} per {g.key}"
                    + (f", counting {', '.join(g.slots)}" if g.slots else ", counting every slot")
                    for g in DK_MLB_CLASSIC.groups
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
def _(DK_MLB_CLASSIC, build_lineups, pool):
    lineups = build_lineups(pool, DK_MLB_CLASSIC, num_lineups=500, seed=1)
    lineups.shape
    return (lineups,)


@app.cell
def _(DK_MLB_CLASSIC, lineups, mo, pool):
    salaries = pool.salary_of(lineups)
    projections = pool.projection_of(lineups)

    mo.md(
        f"""
        **{len(lineups)} distinct lineups.**

        | | min | median | max |
        | --- | ---: | ---: | ---: |
        | salary | {salaries.min():,} | {int(sorted(salaries)[len(salaries) // 2]):,} | {salaries.max():,} |
        | projection | {projections.min():.1f} | {sorted(projections)[len(projections) // 2]:.1f} | {projections.max():.1f} |

        Every one is inside the salary band `{DK_MLB_CLASSIC.salary_floor:,}`–`{DK_MLB_CLASSIC.salary_cap:,}`,
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

        An ILP solver returns the *optimal* lineup, and it beats this on that one
        lineup. Ask it for 500 and it returns the optimum, then the second best,
        then the third — differing by a player or two each time. In a large-field
        contest, where the payoff is convex, 500 near-identical lineups is close to
        the worst thing you can enter.

        Diversity is the thing being bought here, so it is worth measuring rather
        than asserting.
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

        A solver-with-cuts pool would sit far nearer 9 of 10.
        """
    )
    return distinct_players, mean_overlap, overlaps


@app.cell
def _(lineups, mo, pool):
    mo.md("**A sample lineup**\n\n" + "\n".join(f"- {name}" for name in pool.names_of(lineups[0])))
    return


if __name__ == "__main__":
    app.run()
