"""Turning a pool of candidate lineups into a portfolio to enter.

[`build_lineups`][mlb_dfs_solver.greedy.build_lineups] answers "which rosters are
legal?" and deliberately does not answer "which are good?" — it returns the first
distinct lineups it finds, in no useful order. On a slate where price and
projection disagree, the median of those sits well below the best of them.

Selection is the other half. Generate far more candidates than you need, score
them against simulated outcomes, and keep the set that works best *together*:

```python
from mlb_dfs_solver import build_lineups, score_lineups, select_portfolio

pool_of = build_lineups(pool, spec, num_lineups=20_000, seed=1)
scores = score_lineups(pool, spec, pool_of, universe)  # (n_lineups, n_sims)
entries = pool_of[select_portfolio(scores, mode="gpp", line=..., n_select=150)]
```

Nothing here simulates. `universe` — what every player scored in every simulated
outcome — comes from the caller, because a library that works for hockey and golf
has no business modelling how baseball scores. Correlation lives in that matrix:
two lineups sharing a stack move together automatically, because they read the
same rows.

## Cash and tournaments want different things

Not different weights on one objective — different objectives. A cash game pays a
flat amount for beating a line and nothing for beating it well, so each entry is
judged alone and *diversity is actively wrong*: the right play is the most
probable roster, then the next. A tournament pays almost nothing outside the
extreme tail, so what matters is the chance that some entry reaches a winning
score, and two entries that win in the same outcomes are largely wasted.

Pick with `mode`. There is no default, because guessing wrong produces a portfolio
that is confidently optimized for the contest you are not playing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mlb_dfs_solver import _native

if TYPE_CHECKING:
    from mlb_dfs_solver.pool import PlayerPool
    from mlb_dfs_solver.spec import RosterSpec

__all__ = [
    "field_line",
    "portfolio_value",
    "score_lineups",
    "select_portfolio",
    "tail_line",
]

_MODES = {
    # A cash game: each entry judged alone on its chance of clearing the line.
    "cash": "cash",
    # A tournament: the chance that *some* entry clears the line.
    "gpp": "cover",
    # Neither, and the general case: how far past the line the best entry got,
    # which suits a payout that keeps climbing with rank.
    "excess": "excess",
}


def score_lineups(
    pool: PlayerPool,
    spec: RosterSpec,
    lineups: np.ndarray,
    universe: np.ndarray,
) -> np.ndarray:
    """Score each lineup in every simulated outcome.

    Args:
        pool: The players the lineups index into. Used only for its size.
        spec: Supplies the per-slot score multipliers, so a showdown captain is
            worth 1.5x here exactly as they are everywhere else.
        lineups: An `(n_lineups, roster_size)` index array, as returned by
            `build_lineups`.
        universe: An `(n_players, n_outcomes)` array of simulated player scores.
            Rows must line up with `pool`.

    Returns:
        An `(n_lineups, n_outcomes)` float32 array.

    Raises:
        ValueError: If `universe` does not describe this pool, or the lineups
            index a player it lacks.
    """
    universe = np.ascontiguousarray(universe, dtype=np.float32)
    if universe.ndim != 2:
        msg = f"universe must be two-dimensional, got shape {universe.shape}"
        raise ValueError(msg)
    if universe.shape[0] != len(pool):
        msg = (
            f"universe has {universe.shape[0]} player rows but the pool has "
            f"{len(pool)}; they must be parallel"
        )
        raise ValueError(msg)

    lineups = np.asarray(lineups)
    if lineups.size and int(lineups.max()) >= len(pool):
        msg = (
            f"a lineup names player index {int(lineups.max())} but the pool has {len(pool)} players"
        )
        raise ValueError(msg)
    roster_size = spec.roster_size
    if lineups.ndim != 2 or lineups.shape[1] != roster_size:
        msg = (
            f"lineups must be (n, {roster_size}) to match the specification, "
            f"got shape {lineups.shape}"
        )
        raise ValueError(msg)

    return _native.score_lineups(
        universe.ravel(),
        int(universe.shape[1]),
        np.ascontiguousarray(lineups.ravel(), dtype=np.uint32),
        roster_size,
        np.asarray(spec.score_multipliers(), dtype=np.float32),
    )


def field_line(sim_scores: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """The score to beat *in each outcome*, taken across candidates.

    This is almost always the line you want, and a single number almost always
    is not. The field's score swings enormously between simulated outcomes: on a
    realistic slate the median lineup varies about six times as much across
    outcomes as lineups vary within one. Judged against a fixed bar, "did this
    lineup cash?" then correlates 0.98 with "was the slate high-scoring" — which
    is no edge at all, because everyone else scored more in those worlds too.
    Beating the field *in the same world* is what pays.

    Args:
        sim_scores: An `(n_candidates, n_outcomes)` array from `score_lineups`.
        quantile: Where in the field to draw the line. `0.5` is a cash game's
            roughly-half-the-field; `0.99` is a tournament-winning score.

    Returns:
        One score per outcome.

    A candidate pool is still a stand-in for the field — it is your lineups, not
    the ones other people entered. If you have a field model, take its quantile
    per outcome instead. But it is a far better stand-in than a constant.
    """
    if not 0.0 <= quantile <= 1.0:
        msg = f"quantile must be in [0, 1], got {quantile}"
        raise ValueError(msg)
    scores = np.asarray(sim_scores)
    if scores.ndim != 2:
        msg = f"sim_scores must be two-dimensional, got shape {scores.shape}"
        raise ValueError(msg)
    if scores.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(np.quantile(scores, quantile, axis=0), dtype=np.float32)


def tail_line(sim_scores: np.ndarray, quantile: float = 0.99) -> float:
    """A single score at the given quantile of a whole simulated distribution.

    Kept for the case where the line genuinely does not move between outcomes.
    For judging lineups against a field, reach for
    [`field_line`][mlb_dfs_solver.select.field_line] instead — pooling every
    outcome into one number mostly measures how high-scoring the slate was rather
    than how good the lineup is.
    """
    if not 0.0 <= quantile <= 1.0:
        msg = f"quantile must be in [0, 1], got {quantile}"
        raise ValueError(msg)
    scores = np.asarray(sim_scores)
    if scores.size == 0:
        return 0.0
    return float(np.quantile(scores, quantile))


def select_portfolio(
    sim_scores: np.ndarray,
    *,
    mode: str,
    line: float | np.ndarray,
    n_select: int = 150,
    lineups: np.ndarray | None = None,
    max_exposure: float | dict[int, float] | None = None,
    n_players: int | None = None,
    min_gain: float = 0.0,
) -> np.ndarray:
    """Choose which candidates to enter.

    Args:
        sim_scores: An `(n_candidates, n_outcomes)` array from `score_lineups`.
        mode: `"cash"`, `"gpp"`, or `"excess"`. See the module docstring — these
            are different objectives, not settings on one, and there is no
            default because the wrong one is confidently wrong.
        line: The score that has to be beaten. Either one value per outcome —
            which is what you almost always want, see
            [`field_line`][mlb_dfs_solver.select.field_line] — or a single number
            broadcast across all of them, which is only right when the bar
            genuinely does not move.
        n_select: How many entries to choose. Fewer come back when no remaining
            candidate improves the portfolio — in `"gpp"` mode especially, a pool
            can be exhausted well before this.
        lineups: The `(n_candidates, roster_size)` rosters, required only when
            `max_exposure` is set.
        max_exposure: Ceiling on the fraction of chosen entries containing a
            player. A number caps everyone; a mapping caps only those named.
        n_players: Size of the player pool, needed with `max_exposure` when it is
            a mapping that does not name the highest-indexed player.
        min_gain: Stop once the best remaining candidate adds less than this.

    Returns:
        Indices into `sim_scores`, in the order chosen — which is by decreasing
        contribution, so a shorter portfolio is a prefix of a longer one.

    Raises:
        ValueError: If the mode is unknown, or the arrays are inconsistent.
    """
    if mode not in _MODES:
        known = ", ".join(sorted(_MODES))
        msg = f"unknown mode {mode!r}; expected one of: {known}"
        raise ValueError(msg)
    if n_select < 0:
        msg = f"n_select must be non-negative, got {n_select}"
        raise ValueError(msg)

    scores = np.ascontiguousarray(sim_scores, dtype=np.float32)
    if scores.ndim != 2:
        msg = f"sim_scores must be two-dimensional, got shape {scores.shape}"
        raise ValueError(msg)

    rosters = np.empty(0, dtype=np.uint32)
    roster_size = 0
    limits = np.empty(0, dtype=np.uint32)

    if max_exposure is not None:
        if lineups is None:
            msg = "max_exposure needs `lineups`, since a cap is a fact about players"
            raise ValueError(msg)
        rosters_2d = np.asarray(lineups)
        if rosters_2d.ndim != 2 or rosters_2d.shape[0] != scores.shape[0]:
            msg = (
                f"lineups has shape {rosters_2d.shape} but sim_scores describes "
                f"{scores.shape[0]} candidates"
            )
            raise ValueError(msg)
        roster_size = int(rosters_2d.shape[1])
        rosters = np.ascontiguousarray(rosters_2d.ravel(), dtype=np.uint32)

        size = n_players if n_players is not None else int(rosters.max()) + 1
        uncapped = np.iinfo(np.uint32).max
        limits = np.full(size, uncapped, dtype=np.uint32)
        items = (
            [(i, float(max_exposure)) for i in range(size)]
            if isinstance(max_exposure, (int, float))
            else [(int(k), float(v)) for k, v in max_exposure.items()]
        )
        for player, fraction in items:
            if not 0.0 <= fraction <= 1.0:
                msg = f"max_exposure for player {player} is {fraction}; expected a fraction"
                raise ValueError(msg)
            if not 0 <= player < size:
                msg = f"max_exposure names player {player} but the pool has {size}"
                raise ValueError(msg)
            # Against the requested portfolio size, for the same reason the
            # builder does it: a running total would put the first entry over any
            # cap below 100%.
            limits[player] = int(fraction * n_select)

    line_values = np.ascontiguousarray(np.atleast_1d(np.asarray(line)), dtype=np.float32)
    if line_values.ndim != 1 or line_values.size not in (1, scores.shape[1]):
        msg = (
            f"line has {line_values.size} entries; expected one per outcome "
            f"({scores.shape[1]}) or a single value to broadcast"
        )
        raise ValueError(msg)

    chosen = _native.select_portfolio(
        scores.ravel(),
        int(scores.shape[1]),
        rosters,
        roster_size,
        limits,
        int(n_select),
        _MODES[mode],
        line_values,
        float(min_gain),
    )
    return np.asarray(chosen, dtype=np.int64)


def portfolio_value(sim_scores: np.ndarray, chosen: np.ndarray, line: float = 0.0) -> float:
    """What a portfolio is worth under the excess objective.

    The mean over outcomes of how far the portfolio's best entry cleared `line`.
    Exposed so two portfolios can be compared without anyone re-deriving a subtly
    different definition.
    """
    scores = np.ascontiguousarray(sim_scores, dtype=np.float32)
    return float(
        _native.portfolio_value(
            scores.ravel(),
            int(scores.shape[1]),
            np.ascontiguousarray(np.asarray(chosen).ravel(), dtype=np.uint32),
            float(line),
        )
    )
