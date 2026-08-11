//! PyO3 bindings for `mlb_dfs_solver-core`.
//!
//! Marshalling only. Every algorithm lives in the core crate, which has no Python
//! dependency and carries its own tests — so there is nothing here that needs
//! testing from Rust, and `cargo llvm-cov` over the core is not diluted by glue.
//!
//! Nothing in this module is public API. Each function is wrapped by a typed,
//! documented function in the `mlb_dfs_solver` Python package; the flat argument shapes
//! below are chosen for cheap marshalling, not for anyone to call by hand.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use mlb_dfs_solver_core::convert::{
    config_from_arrays, flatten_lineups, spec_from_arrays, ConfigArrays, SpecArrays,
};
use mlb_dfs_solver_core::greedy::{self, PlayerPool};
use mlb_dfs_solver_core::select;
use mlb_dfs_solver_core::simd;

/// Borrow a NumPy array as a contiguous slice.
///
/// NumPy hands out non-contiguous views freely — a stride, a transpose, a column
/// of a 2-D array — and `as_slice` fails on those. Saying so plainly beats the
/// default `unwrap`, which reaches the user as a `PanicException` with no hint
/// that `np.ascontiguousarray` is the fix.
fn contiguous<'py, T: numpy::Element>(
    array: &'py PyReadonlyArray1<'py, T>,
    name: &str,
) -> PyResult<&'py [T]> {
    array.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "'{name}' is not C-contiguous; pass np.ascontiguousarray({name})"
        ))
    })
}

/// Returns the SIMD instruction set the kernel selected at runtime.
///
/// Exposed because wheels are built without `target-cpu=native` and dispatch
/// happens per-process: anyone comparing a performance report against ours needs
/// to know which path their machine took.
#[pyfunction]
fn active_isa() -> &'static str {
    simd::active_isa()
}

/// Build a pool of distinct, valid lineups.
///
/// Returns an `(n_lineups, roster_size)` array of indices into the player pool.
/// Fewer rows than requested means the pool could not support more.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn build_lineups<'py>(
    py: Python<'py>,
    projections: PyReadonlyArray1<'py, f64>,
    stddevs: PyReadonlyArray1<'py, f64>,
    salaries: PyReadonlyArray1<'py, i64>,
    ownership: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, u32>,
    slot_eligible: PyReadonlyArray1<'py, u32>,
    slot_counts: PyReadonlyArray1<'py, u64>,
    slot_score_multipliers: PyReadonlyArray1<'py, f64>,
    slot_salary_multipliers: PyReadonlyArray1<'py, f64>,
    salary_cap: i64,
    salary_floor: i64,
    group_key_columns: PyReadonlyArray1<'py, u64>,
    group_max_counts: PyReadonlyArray1<'py, u32>,
    group_min_distincts: PyReadonlyArray1<'py, u32>,
    group_min_stacks: PyReadonlyArray1<'py, u32>,
    group_slot_masks: PyReadonlyArray1<'py, u64>,
    key_columns: PyReadonlyArray1<'py, i32>,
    conflict_left: PyReadonlyArray1<'py, u32>,
    conflict_right: PyReadonlyArray1<'py, u32>,
    num_lineups: usize,
    seed: u64,
    noise: f64,
    attempts_per_lineup: usize,
    chunks: usize,
    profiles: PyReadonlyArray1<'py, f64>,
    lock_players: PyReadonlyArray1<'py, u32>,
    lock_slot_groups: PyReadonlyArray1<'py, u64>,
    exposure_limits: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let pool = PlayerPool {
        projections: contiguous(&projections, "projections")?,
        stddevs: contiguous(&stddevs, "stddevs")?,
        salaries: contiguous(&salaries, "salaries")?,
        ownership: contiguous(&ownership, "ownership")?,
        positions: contiguous(&positions, "positions")?,
    };

    let spec = spec_from_arrays(
        SpecArrays {
            slot_eligible: contiguous(&slot_eligible, "slot_eligible")?,
            slot_counts: contiguous(&slot_counts, "slot_counts")?,
            slot_score_multipliers: contiguous(&slot_score_multipliers, "slot_score_multipliers")?,
            slot_salary_multipliers: contiguous(
                &slot_salary_multipliers,
                "slot_salary_multipliers",
            )?,
            salary_cap,
            salary_floor,
            group_key_columns: contiguous(&group_key_columns, "group_key_columns")?,
            group_max_counts: contiguous(&group_max_counts, "group_max_counts")?,
            group_min_distincts: contiguous(&group_min_distincts, "group_min_distincts")?,
            group_min_stacks: contiguous(&group_min_stacks, "group_min_stacks")?,
            group_slot_masks: contiguous(&group_slot_masks, "group_slot_masks")?,
            key_columns: contiguous(&key_columns, "key_columns")?,
            conflict_left: contiguous(&conflict_left, "conflict_left")?,
            conflict_right: contiguous(&conflict_right, "conflict_right")?,
        },
        pool.len(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let config = config_from_arrays(ConfigArrays {
        num_lineups,
        seed,
        noise,
        attempts_per_lineup,
        chunks,
        profiles: contiguous(&profiles, "profiles")?,
        lock_players: contiguous(&lock_players, "lock_players")?,
        lock_slot_groups: contiguous(&lock_slot_groups, "lock_slot_groups")?,
        exposure_limits: contiguous(&exposure_limits, "exposure_limits")?,
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let roster_size = spec.roster_size();

    // Detach from the interpreter for the work itself: construction is pure
    // computation over borrowed buffers and touches no Python object, so staying
    // attached would serialize every caller in a threaded process for nothing.
    // (`detach` is pyo3 0.27's name for what was `allow_threads` — the rename is
    // deliberate, since on a free-threaded build there is no GIL to release.)
    let lineups = py
        .detach(|| greedy::build_lineups(&pool, &spec, &config))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = lineups.len();
    let flat = flatten_lineups(&lineups, roster_size)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let array = numpy::ndarray::Array2::from_shape_vec((rows, roster_size), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

/// Score lineups against a simulated player universe.
///
/// Returns an `(n_lineups, n_outcomes)` float32 matrix.
#[pyfunction]
fn score_lineups<'py>(
    py: Python<'py>,
    universe: PyReadonlyArray1<'py, f32>,
    n_outcomes: usize,
    lineups: PyReadonlyArray1<'py, u32>,
    roster_size: usize,
    slot_multipliers: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let universe = contiguous(&universe, "universe")?;
    let lineups = contiguous(&lineups, "lineups")?;
    let multipliers = contiguous(&slot_multipliers, "slot_multipliers")?;

    let scored = py
        .detach(|| select::score_lineups(universe, n_outcomes, lineups, roster_size, multipliers))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rows = if n_outcomes == 0 {
        0
    } else {
        scored.len() / n_outcomes
    };
    let array = numpy::ndarray::Array2::from_shape_vec((rows, n_outcomes), scored)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

/// Select a portfolio from a scored candidate pool.
///
/// `mode` is `"excess"`, `"cover"` or `"cash"`; `line` is the score each reads.
/// Returns candidate indices in the order chosen.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn select_portfolio<'py>(
    py: Python<'py>,
    scores: PyReadonlyArray1<'py, f32>,
    n_outcomes: usize,
    rosters: PyReadonlyArray1<'py, u32>,
    roster_size: usize,
    exposure_limits: PyReadonlyArray1<'py, u32>,
    n_select: usize,
    mode: &str,
    line: PyReadonlyArray1<'py, f32>,
    min_gain: f32,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let objective = match mode {
        "excess" => select::Objective::Excess,
        "cover" => select::Objective::Cover,
        "cash" => select::Objective::Cash,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown selection mode {other:?}; expected 'excess', 'cover' or 'cash'"
            )))
        }
    };

    let candidates = select::Candidates {
        scores: contiguous(&scores, "scores")?,
        n_outcomes,
        rosters: contiguous(&rosters, "rosters")?,
        roster_size,
    };
    let config = select::SelectConfig {
        n_select,
        objective,
        line: contiguous(&line, "line")?.to_vec(),
        exposure_limits: contiguous(&exposure_limits, "exposure_limits")?.to_vec(),
        min_gain,
    };

    let chosen = py
        .detach(|| select::select_portfolio(&candidates, &config))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, chosen))
}

/// The value of a portfolio under a selection objective.
#[pyfunction]
fn portfolio_value<'py>(
    scores: PyReadonlyArray1<'py, f32>,
    n_outcomes: usize,
    chosen: PyReadonlyArray1<'py, u32>,
    threshold: f32,
) -> PyResult<f32> {
    let candidates = select::Candidates {
        scores: contiguous(&scores, "scores")?,
        n_outcomes,
        rosters: &[],
        roster_size: 0,
    };
    Ok(select::portfolio_value(
        &candidates,
        contiguous(&chosen, "chosen")?,
        threshold,
    ))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(active_isa, m)?)?;
    m.add_function(wrap_pyfunction!(build_lineups, m)?)?;
    m.add_function(wrap_pyfunction!(score_lineups, m)?)?;
    m.add_function(wrap_pyfunction!(select_portfolio, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_value, m)?)?;
    Ok(())
}
