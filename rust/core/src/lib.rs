//! Roster optimization algorithms.
//!
//! This crate is deliberately free of any Python dependency. Everything here is
//! ordinary Rust operating on slices, which means it unit-tests and
//! coverage-measures without an interpreter, and the binding layer in
//! `mlb_dfs_solver-native` has nothing in it worth testing separately.
//!
//! * [`roster`] — what makes a lineup legal: slots, eligibility, group caps.
//! * [`greedy`] — randomized greedy construction of a diverse pool of lineups.
//! * [`convert`] — reassembling a specification from the flat arrays that cross
//!   the language boundary.
//! * [`simd`] — runtime-dispatched vector kernels, each paired with a scalar
//!   reference that produces bit-identical results.

pub mod convert;
pub mod greedy;
pub mod roster;
pub mod simd;
