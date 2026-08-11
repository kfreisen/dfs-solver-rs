import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from dataclasses import dataclass

    import marimo as mo
    import numpy as np
    from mlb_dfs_solver import (
        ConflictRule,
        GroupConstraint,
        PlayerPool,
        build_lineups,
        portfolio_value,
        score_lineups,
        select_portfolio,
    )
    from mlb_dfs_solver.presets import DK_MLB_CLASSIC

    return (
        ConflictRule,
        DK_MLB_CLASSIC,
        GroupConstraint,
        PlayerPool,
        build_lineups,
        dataclass,
        mo,
        np,
        portfolio_value,
        score_lineups,
        select_portfolio,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Wiring this into an existing pipeline

        The quickstart builds a pool of legal lineups. This one is the whole job:
        take the objects a projection system already produces, get a portfolio of
        contest entries back, and hand them to whatever uploads them.

        Nothing here is MLB-specific except the preset. The adapter at the top is
        the only code you would actually write.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 1. Your objects

        A projection pipeline has its own player type, with more on it than a
        solver needs. This one stands in for it — including a `ceiling` from a
        conformal-quantile model, which is the interesting field.
        """
    )
    return


@app.cell
def _(dataclass):
    @dataclass
    class Player:
        """Whatever your projection layer produces."""

        index: int
        name: str
        team: str
        opponent: str
        game: str
        positions: list[str]
        salary: int
        projection: float
        ceiling: float  # e.g. a conformal P95
        ownership: float

    return (Player,)


@app.cell
def _(Player):
    # A slate, generated so this notebook runs offline in under a second.
    def make_players() -> list[Player]:
        players: list[Player] = []
        index = 0
        for position_index, position in enumerate(("P", "C", "1B", "2B", "3B", "SS", "OF")):
            for k in range(96 if position == "OF" else 32):
                team = (7 * k + 3 * position_index) % 10
                projection = 4.0 + (k % 12) * 1.2 + (((k * 13) % 7) - 3) * 0.9
                players.append(
                    Player(
                        index=index,
                        name=f"{position}-{k}",
                        team=f"TM{team}",
                        opponent=f"TM{team ^ 1}",
                        game=f"G{min(team, team ^ 1)}",
                        positions=[position],
                        salary=2500 + (k % 12) * 750,
                        projection=projection,
                        # A right-skewed ceiling, not a fixed multiple of anything.
                        ceiling=projection + 4.0 + (k % 5) * 1.6,
                        ownership=((k * 7) % 30) / 100.0,
                    )
                )
                index += 1
        return players

    players = make_players()
    len(players)
    return make_players, players


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. The adapter

        One comprehension. The only line worth explaining is the standard
        deviation.

        The objective values a player at $projection + t \cdot stddev$, with $t$
        drawn afresh on every attempt — that draw is what makes the pool diverse.
        A ceiling is not a separate input to that; it is a better estimate of the
        same spread. Converting it means $t = 1.645$ lands exactly on your P95:

        $$stddev = \frac{ceiling - projection}{1.645}$$

        Nothing is lost. Only the upside term is ever used, so an asymmetric
        right-tail estimate does not drag an implied downside along with it.
        """
    )
    return


@app.cell
def _(PlayerPool, players):
    z_p95 = 1.645

    def to_pool(players, spec):
        """Your objects in, a `PlayerPool` out."""
        return PlayerPool.from_records(
            [
                {
                    "name": p.name,
                    "positions": tuple(p.positions),
                    "salary": p.salary,
                    "projection": p.projection,
                    "stddev": (p.ceiling - p.projection) / z_p95,
                    "ownership": p.ownership,
                    "team": p.team,
                    "opponent": p.opponent,
                    "game": p.game,
                }
                for p in players
            ],
            spec,
            key_fields=["team", "opponent", "game"],
        )

    return z_p95, to_pool


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. The contest rules

        The preset carries what DraftKings enforces. Everything below it is a
        choice — the distinct-games rule is theirs, the rest is strategy, and
        none of it is turned on for you.
        """
    )
    return


@app.cell
def _(ConflictRule, DK_MLB_CLASSIC, GroupConstraint, players, to_pool):
    from dataclasses import replace

    hitters = ("C", "SS", "2B", "3B", "1B", "OF")

    spec = replace(
        DK_MLB_CLASSIC,
        groups=(
            *DK_MLB_CLASSIC.groups,
            # DraftKings requires this; a cap cannot say it.
            GroupConstraint(key="game", min_distinct=2),
            # This is strategy, not a rule.
            GroupConstraint(key="team", min_stack=4, slots=hitters),
        ),
        conflicts=(
            ConflictRule(
                left_key="opponent",
                right_key="team",
                left_positions=("P",),
                right_positions=hitters,
            ),
        ),
    )
    pool = to_pool(players, spec)
    (len(pool), spec.roster_size)
    return hitters, pool, replace, spec


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Candidates

        Far more than you will enter. Selection needs something to choose from,
        and the cost of the ten-thousandth lineup is what makes that affordable.

        Ask for twenty thousand and you will get fewer: a stack, a conflict rule
        and a salary floor between them rule out most of the space, and returning
        what exists is more honest than padding. Watch that number if you tighten
        the strategy — a pool of a few hundred is selection with nothing to
        select from.
        """
    )
    return


@app.cell
def _(build_lineups, pool, spec):
    candidates = build_lineups(
        pool,
        spec,
        num_lineups=20_000,
        seed=1,
        attempts_per_lineup=5,
        max_exposure=0.6,
    )
    candidates.shape
    return (candidates,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Your simulator

        This is the one stage the library does not do. Simulating a sport well
        means modelling that sport, and this works for any of them, so it takes
        the matrix and does not produce it.

        What it needs is `(players x outcomes)`: what everyone scored in each way
        the slate could break. Any float dtype, any layout — simulators store
        these narrow because they are large, and `float16` converts exactly.

        The crude stand-in below has a per-team shock so that lineups stacking a
        team move together. That correlation is the whole reason the portfolio
        objective has anything to work with.
        """
    )
    return


@app.cell
def _(np, pool):
    def simulate(pool, n_outcomes=1_000, seed=7):
        rng = np.random.default_rng(seed)
        teams = pool.keys["team"]
        shock = rng.standard_normal((int(teams.max()) + 1, n_outcomes)) * 0.55
        noise = rng.standard_normal((len(pool), n_outcomes))
        universe = pool.projections[:, None] + pool.stddevs[:, None] * (0.8 * noise + shock[teams])
        # Stored narrow, as a real simulator would.
        return universe.astype(np.float16)

    universe = simulate(pool)
    (universe.shape, universe.dtype)
    return simulate, universe


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. The field

        The score that wins is a property of *who else entered*, not of your own
        candidates. Scoring against a quantile of your own pool is circular and
        will report success no matter what you do.

        Note which rules the crowd is built under. They obey the *contest* — the
        cap, the team limits, the two-games requirement — and not our strategy.
        Nobody else is bound by our four-hitter stack or our rule about opposing
        hitters. Forcing them to obey it shrinks the simulated field sevenfold on
        this slate and draws the payout line from far too few entries.

        A real pipeline puts an ownership-driven field model here. This one
        rebuilds the crowd with the same constructor and no ownership fade, which
        is at least independent of the portfolio being judged.
        """
    )
    return


@app.cell
def _(build_lineups, np, pool, replace, score_lineups, spec, universe):
    from mlb_dfs_solver import JitterProfile

    crowd = [
        JitterProfile(ceiling=(0.1, 0.6), leverage=(0.0, 0.1)),
        JitterProfile(ceiling=(0.3, 1.2), leverage=(0.1, 0.5)),
    ]
    # Contest rules only. The crowd is not bound by our stack or our conflict
    # rule, and forcing them to be shrinks the simulated field sevenfold.
    contest = replace(spec, conflicts=(), groups=tuple(g for g in spec.groups if not g.min_stack))
    field = build_lineups(pool, contest, num_lineups=100_000, seed=99, noise=0.45, profiles=crowd)
    field_scores = score_lineups(pool, spec, field, universe)

    # The bar is per outcome. A high-scoring slate lifts everyone, so clearing a
    # fixed number in one is no edge at all.
    win_line = np.quantile(field_scores, 0.999, axis=0).astype(np.float32)
    cash_line = np.quantile(field_scores, 0.50, axis=0).astype(np.float32)
    (len(field), float(np.median(win_line)), float(win_line.max() - win_line.min()))
    return JitterProfile, cash_line, contest, crowd, field, field_scores, win_line


@app.cell
def _(mo):
    mo.md(
        """
        ## 7. Selection

        Cash and tournaments are different objectives, not different weights, so
        `mode` has no default.
        """
    )
    return


@app.cell
def _(candidates, cash_line, pool, score_lineups, select_portfolio, spec, universe, win_line):
    scores = score_lineups(pool, spec, candidates, universe)

    gpp = candidates[select_portfolio(scores, mode="gpp", line=win_line, n_select=150)]
    cash = candidates[select_portfolio(scores, mode="cash", line=cash_line, n_select=20)]
    (scores.shape, len(gpp), len(cash))
    return cash, gpp, scores


@app.cell
def _(cash, cash_line, gpp, mo, np, pool, score_lineups, spec, universe, win_line):
    gpp_scores = score_lineups(pool, spec, gpp, universe)
    cash_scores = score_lineups(pool, spec, cash, universe)

    mo.md(
        f"""
        | Portfolio | Entries | Mean salary | Result |
        | --- | ---: | ---: | ---: |
        | Tournament | {len(gpp)} | ${float(np.mean(pool.salary_of(gpp, spec))):,.0f} | some entry in the top 0.1% in **{float((gpp_scores.max(axis=0) >= win_line).mean()):.1%}** of outcomes |
        | Cash | {len(cash)} | ${float(np.mean(pool.salary_of(cash, spec))):,.0f} | the average entry beats the field in **{float((cash_scores >= cash_line).mean()):.1%}** of outcomes |

        Two different jobs, from one candidate pool.
        """
    )
    return cash_scores, gpp_scores


@app.cell
def _(mo):
    mo.md(
        """
        ## 8. Expected payout, if you have a payout curve

        Selection reads values per outcome. Nothing requires them to be points.
        Hand it dollars — a `(lineups x outcomes)` payout matrix — with
        `mode="excess"` and `line=0.0`, and the objective is

        ```text
        E[max payout across the portfolio]
        ```

        which is the usual objective for a top-heavy contest. It stays submodular
        because payouts are non-negative, so lazy evaluation and its guarantee
        carry over. This is not a special mode; it is what the general one
        already computes.
        """
    )
    return


@app.cell
def _(field_scores, np, portfolio_value, scores, select_portfolio):
    # A payout curve is a set of per-outcome thresholds: what it took to reach
    # each paying tier in that outcome. Comparing every candidate against every
    # field entry would be a (candidates x field x outcomes) tensor and is never
    # necessary — the quantiles are the curve.
    tiers = [(0.9999, 10_000.0), (0.999, 250.0), (0.99, 25.0)]
    payouts = np.zeros_like(scores)
    for quantile, prize in tiers:
        line = np.quantile(field_scores, quantile, axis=0).astype(np.float32)
        payouts = np.where((payouts == 0) & (scores >= line), prize, payouts)

    by_payout = select_portfolio(payouts, mode="excess", line=0.0, n_select=150)
    f"E[max payout] = ${portfolio_value(payouts, by_payout):,.2f} over {len(by_payout)} entries"
    return by_payout, payouts, tiers


@app.cell
def _(mo):
    mo.md(
        """
        ## 9. Back to your objects

        A lineup is an array of pool indices, in `spec.slot_names()` order. The
        pool was built from your players in order, so the index is your index.
        """
    )
    return


@app.cell
def _(gpp, players, spec):
    def to_entries(lineups, players, spec):
        """Pool indices back to your own objects, labelled by slot."""
        slots = spec.slot_names()
        return [
            [(slot, players[i]) for slot, i in zip(slots, lineup, strict=True)]
            for lineup in lineups.tolist()
        ]

    entries = to_entries(gpp, players, spec)
    [f"{slot}: {player.name}" for slot, player in entries[0]]
    return entries, to_entries


@app.cell
def _(mo):
    mo.md(
        """
        ## What you would replace

        Two things, and neither is in this library:

        - **The simulator.** Section 5 is a per-team shock and Gaussian noise.
          A real one models batting order, park, handedness and the correlation
          structure between teammates.
        - **The field.** Section 6 rebuilds the crowd with the same constructor.
          A real one is driven by ownership projections and how the public
          actually builds.

        Everything else — the adapter, the rules, the candidate pool, selection,
        the mapping back — is what you see.
        """
    )
    return


if __name__ == "__main__":
    app.run()
