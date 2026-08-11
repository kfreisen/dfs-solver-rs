//! Roster optimization algorithms.
//!
//! This crate is deliberately free of any Python dependency. Everything here is
//! ordinary Rust operating on slices, which means it unit-tests and
//! coverage-measures without an interpreter, and the binding layer in
//! `mlb_dfs_solver-native` has nothing in it worth testing separately.
//!
//! * [`roster`] — what makes a lineup legal: slots, eligibility, group caps.
//! * [`greedy`] — randomized greedy construction of a diverse pool of lineups.
//! * [`select`] — lazy-greedy submodular selection of a portfolio from that pool.
//! * [`convert`] — reassembling a specification from the flat arrays that cross
//!   the language boundary.
//! * [`simd`] — runtime-dispatched vector kernels, each paired with a scalar
//!   reference that produces bit-identical results.
//!
//! Construction and selection are separate on purpose. Construction knows what
//! makes a lineup *legal* and nothing about what makes one good; selection knows
//! what makes a portfolio good and nothing about the rules. Neither simulates —
//! the outcome matrix selection reads comes from the caller, because a
//! sport-agnostic library has no business modelling how baseball scores.

pub mod convert;
pub mod greedy;
pub mod roster;
pub mod select;
pub mod simd;
