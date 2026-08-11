"""Pure-NumPy transcription of portfolio selection.

The counterpart to `reference.py`, serving the same three jobs for the selection
half: it is the readable statement of the algorithm, the parity oracle, and the
baseline the Rust kernel is measured against.

Unlike the construction oracle, this one can be held to *exact* equality. There
is no random number generator involved — selection is deterministic given a score
matrix — so `tests/test_parity.py` asserts the two produce identical portfolios
rather than merely comparable ones. That is a much stronger check, and it is
available here only because nothing is sampled.

Written as the naive greedy: recompute every candidate's marginal gain every
round, take the best. The kernel's lazy evaluation is an optimization over
exactly this, and the claim it makes — same answer, less work — is only testable
against a version that does the work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["portfolio_value_reference", "select_portfolio_reference"]


def _gain(mode: str, scores: np.ndarray, bar: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Marginal value of each candidate against the portfolio's current state.

    One row per candidate; the mean is taken over outcomes. Modes are named as
    the public API names them, not as the kernel spells them internally — an
    oracle that agreed with the wrapper only after a translation step would not
    be checking the translation.
    """
    if mode == "excess":
        return np.maximum(scores - bar, 0.0).mean(axis=1)
    if mode == "gpp":
        # An outcome counts only where this candidate clears the line and the
        # portfolio does not already have an entry that does.
        return ((scores >= line) & (bar < line)).mean(axis=1)
    if mode == "cash":
        # No reference to `bar`: entries are judged alone, which is what makes
        # cash modular rather than submodular.
        return (scores >= line).mean(axis=1)
    msg = f"unknown mode {mode!r}"
    raise ValueError(msg)


def _floor(mode: str, line: np.ndarray) -> np.ndarray:
    """The per-outcome state before anything is chosen."""
    if mode == "excess":
        return line.astype(np.float32).copy()
    if mode == "gpp":
        # Anything under the line reads as "not covered"; the line itself would
        # wrongly read as covered.
        return (line - 1.0).astype(np.float32)
    return np.full(line.shape, -np.inf, dtype=np.float32)


def select_portfolio_reference(
    sim_scores: np.ndarray,
    *,
    mode: str,
    line: float | np.ndarray,
    n_select: int = 150,
    lineups: np.ndarray | None = None,
    max_exposure: float | Mapping[int, float] | None = None,
    n_players: int | None = None,
    min_gain: float = 0.0,
) -> list[int]:
    """Choose a portfolio, in readable NumPy.

    Mirrors `mlb_dfs_solver.select_portfolio`, including its tie-break: on equal
    gains the lower candidate index wins. That matters more than it sounds —
    tournament gains are counts of outcomes, so ties are the norm, and a
    different tie-break produces a different portfolio.
    """
    scores = np.asarray(sim_scores, dtype=np.float32)
    n_cand, n_outcomes = scores.shape
    if n_select <= 0 or n_cand == 0:
        return []

    # One value per outcome. A scalar broadcasts, which is only right when the
    # bar genuinely does not move between outcomes.
    line = np.broadcast_to(np.atleast_1d(np.asarray(line, dtype=np.float32)), (n_outcomes,)).astype(
        np.float32
    )

    limits: np.ndarray | None = None
    rosters: np.ndarray | None = None
    if max_exposure is not None:
        if lineups is None:
            msg = "max_exposure needs `lineups`"
            raise ValueError(msg)
        rosters = np.asarray(lineups)
        size = n_players if n_players is not None else int(rosters.max()) + 1
        limits = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
        items = (
            [(i, float(max_exposure)) for i in range(size)]
            if isinstance(max_exposure, (int, float))
            else [(int(k), float(v)) for k, v in max_exposure.items()]
        )
        for player, fraction in items:
            limits[player] = int(fraction * n_select)

    appearances = np.zeros(0 if limits is None else len(limits), dtype=np.int64)
    bar = _floor(mode, line)
    available = np.ones(n_cand, dtype=bool)
    chosen: list[int] = []

    while len(chosen) < n_select:
        eligible = available.copy()
        if limits is not None and rosters is not None:
            # A candidate is out once any of its players is at their ceiling.
            over = appearances >= limits
            if over.any():
                eligible &= ~over[rosters].any(axis=1)
        if not eligible.any():
            break

        gains = _gain(mode, scores, bar, line)
        gains[~eligible] = -np.inf
        # argmax returns the first maximum, which is the lowest index — the same
        # tie-break the kernel uses.
        best = int(np.argmax(gains))
        if gains[best] <= min_gain and chosen:
            break

        available[best] = False
        if mode != "cash":
            bar = np.maximum(bar, scores[best])
        if limits is not None and rosters is not None:
            np.add.at(appearances, rosters[best], 1)
        chosen.append(best)

    return chosen


def portfolio_value_reference(
    sim_scores: np.ndarray, chosen: list[int] | np.ndarray, line: float = 0.0
) -> float:
    """Mean excess of the portfolio's best entry over `line`."""
    scores = np.asarray(sim_scores, dtype=np.float32)
    chosen = np.asarray(chosen, dtype=np.int64)
    if len(chosen) == 0:
        return 0.0
    return float(np.maximum(scores[chosen].max(axis=0), line).mean() - line)
