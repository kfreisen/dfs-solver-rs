"""Type stubs for the compiled kernel.

Hand-maintained, because the extension is a shared object: mypy cannot infer these
and griffe (which builds the docs) cannot import them. This file is the single
declaration of the FFI surface — keep it in step with `rust/native/src/lib.rs`.

Nothing here is public API. Use the wrappers in :mod:`mlb_dfs_solver`.
"""

import numpy as np

__version__: str

def active_isa() -> str:
    """Return the SIMD path selected at runtime: ``"avx2"`` or ``"scalar"``."""

def build_lineups(
    projections: np.ndarray,
    stddevs: np.ndarray,
    salaries: np.ndarray,
    ownership: np.ndarray,
    positions: np.ndarray,
    slot_eligible: np.ndarray,
    slot_counts: np.ndarray,
    slot_score_multipliers: np.ndarray,
    slot_salary_multipliers: np.ndarray,
    salary_cap: int,
    salary_floor: int,
    group_key_columns: np.ndarray,
    group_max_counts: np.ndarray,
    group_min_distincts: np.ndarray,
    group_min_stacks: np.ndarray,
    group_slot_masks: np.ndarray,
    key_columns: np.ndarray,
    conflict_left: np.ndarray,
    conflict_right: np.ndarray,
    num_lineups: int,
    seed: int,
    noise: float,
    attempts_per_lineup: int,
    chunks: int,
    profiles: np.ndarray,
    value_weight: float,
    diversity_weight: float,
    lock_players: np.ndarray,
    lock_slot_groups: np.ndarray,
    exposure_limits: np.ndarray,
) -> np.ndarray:
    """Build lineups. Returns an ``(n, roster_size)`` int64 index array."""

def score_lineups(
    universe: np.ndarray,
    n_outcomes: int,
    lineups: np.ndarray,
    roster_size: int,
    slot_multipliers: np.ndarray,
) -> np.ndarray:
    """Score lineups against a universe. Returns ``(n_lineups, n_outcomes)`` float32."""

def select_portfolio(
    scores: np.ndarray,
    n_outcomes: int,
    rosters: np.ndarray,
    roster_size: int,
    exposure_limits: np.ndarray,
    n_select: int,
    mode: str,
    line: np.ndarray,
    min_gain: float,
) -> np.ndarray:
    """Select a portfolio. Returns candidate indices in the order chosen."""

def portfolio_value(
    scores: np.ndarray,
    n_outcomes: int,
    chosen: np.ndarray,
    threshold: float,
) -> float:
    """Mean excess of the portfolio's best entry over ``threshold``."""
