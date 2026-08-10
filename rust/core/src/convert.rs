//! Reassembling a specification from flat arrays.
//!
//! The roster specification crosses the language boundary once per solve, as a
//! handful of parallel arrays rather than as an object graph. Rebuilding a
//! `RosterSpec` per lineup would cost more than building the lineup does.
//!
//! This module contains no Python types — it takes slices and returns a
//! [`RosterSpec`] or a [`ConvertError`]. The binding crate turns that error into
//! an exception. Keeping the split here is what lets these cases be tested
//! without an interpreter.

use crate::greedy::{GreedyConfig, JitterProfile};
use crate::roster::{GroupConstraint, RosterSpec, SlotGroup};

/// A malformed set of flat arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvertError {
    /// The per-slot arrays disagree on length.
    SlotArrayMismatch { eligible: usize, counts: usize },
    /// The per-group arrays disagree on length.
    GroupArrayMismatch {
        key_columns: usize,
        max_counts: usize,
        slot_masks: usize,
    },
    /// No players were supplied.
    EmptyPool,
    /// The key matrix is not a whole number of columns.
    RaggedKeyMatrix { len: usize, n_players: usize },
    /// The profile matrix is not a whole number of 4-tuples.
    RaggedProfileMatrix { len: usize },
    /// A lineup came back the wrong length, which would be a bug in this crate.
    WrongLineupLength { got: usize, expected: usize },
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SlotArrayMismatch { eligible, counts } => write!(
                f,
                "slot_eligible has {eligible} entries but slot_counts has {counts}"
            ),
            Self::GroupArrayMismatch {
                key_columns,
                max_counts,
                slot_masks,
            } => write!(
                f,
                "group arrays must be the same length, got key_columns={key_columns}, \
                 max_counts={max_counts}, slot_masks={slot_masks}"
            ),
            Self::EmptyPool => write!(f, "the player pool is empty"),
            Self::RaggedKeyMatrix { len, n_players } => write!(
                f,
                "key_columns has {len} entries, which is not a multiple of the \
                 {n_players} players in the pool"
            ),
            Self::RaggedProfileMatrix { len } => write!(
                f,
                "profiles has {len} values, which is not a multiple of 4 \
                 (ceiling low, ceiling high, leverage low, leverage high)"
            ),
            Self::WrongLineupLength { got, expected } => write!(
                f,
                "internal error: built a lineup of {got} players for a roster of {expected}"
            ),
        }
    }
}

impl std::error::Error for ConvertError {}

/// Arrays describing a roster specification, as they arrive from the caller.
///
/// Grouped into a struct rather than passed as nine positional arguments, because
/// nine same-typed slices in a row is a bug waiting to happen.
#[derive(Debug, Clone, Copy)]
pub struct SpecArrays<'a> {
    /// Eligibility mask per slot group.
    pub slot_eligible: &'a [u32],
    /// Number of slots per slot group.
    pub slot_counts: &'a [u64],
    /// Maximum total salary.
    pub salary_cap: i64,
    /// Minimum total salary; 0 disables.
    pub salary_floor: i64,
    /// Which key column each group constraint reads.
    pub group_key_columns: &'a [u64],
    /// The cap for each group constraint.
    pub group_max_counts: &'a [u32],
    /// Which slot groups each constraint counts.
    pub group_slot_masks: &'a [u64],
    /// Flattened `(n_columns, n_players)` key matrix.
    pub key_columns: &'a [i32],
}

/// Rebuild a [`RosterSpec`] from flat arrays.
pub fn spec_from_arrays(
    arrays: SpecArrays<'_>,
    n_players: usize,
) -> Result<RosterSpec, ConvertError> {
    if arrays.slot_eligible.len() != arrays.slot_counts.len() {
        return Err(ConvertError::SlotArrayMismatch {
            eligible: arrays.slot_eligible.len(),
            counts: arrays.slot_counts.len(),
        });
    }
    let n_groups = arrays.group_max_counts.len();
    if arrays.group_key_columns.len() != n_groups || arrays.group_slot_masks.len() != n_groups {
        return Err(ConvertError::GroupArrayMismatch {
            key_columns: arrays.group_key_columns.len(),
            max_counts: n_groups,
            slot_masks: arrays.group_slot_masks.len(),
        });
    }
    if n_players == 0 {
        return Err(ConvertError::EmptyPool);
    }
    // Not `is_multiple_of`: stable only from 1.87, and this crate supports older
    // toolchains.
    if arrays.key_columns.len() % n_players != 0 {
        return Err(ConvertError::RaggedKeyMatrix {
            len: arrays.key_columns.len(),
            n_players,
        });
    }

    Ok(RosterSpec {
        slots: arrays
            .slot_eligible
            .iter()
            .zip(arrays.slot_counts)
            .map(|(&eligible, &count)| SlotGroup {
                eligible,
                count: count as usize,
            })
            .collect(),
        salary_cap: arrays.salary_cap,
        salary_floor: arrays.salary_floor,
        groups: (0..n_groups)
            .map(|i| GroupConstraint {
                key_column: arrays.group_key_columns[i] as usize,
                max_count: arrays.group_max_counts[i],
                slots: arrays.group_slot_masks[i],
            })
            .collect(),
        key_columns: arrays
            .key_columns
            .chunks(n_players)
            .map(<[i32]>::to_vec)
            .collect(),
    })
}

/// Rebuild a [`GreedyConfig`]. `profiles` is a flattened `(n, 4)` matrix of
/// ceiling low/high then leverage low/high.
pub fn config_from_arrays(
    num_lineups: usize,
    seed: u64,
    noise: f64,
    attempts_per_lineup: usize,
    chunks: usize,
    profiles: &[f64],
) -> Result<GreedyConfig, ConvertError> {
    if profiles.len() % 4 != 0 {
        return Err(ConvertError::RaggedProfileMatrix {
            len: profiles.len(),
        });
    }
    Ok(GreedyConfig {
        num_lineups,
        seed,
        noise,
        attempts_per_lineup,
        chunks,
        profiles: profiles
            .chunks(4)
            .map(|p| JitterProfile {
                ceiling: (p[0], p[1]),
                leverage: (p[2], p[3]),
            })
            .collect(),
    })
}

/// Flatten lineups into a row-major `(n_lineups, roster_size)` buffer.
///
/// An empty result is not an error: the caller asked whether any valid lineup
/// exists and the answer was no.
pub fn flatten_lineups(lineups: &[Vec<u32>], roster_size: usize) -> Result<Vec<i64>, ConvertError> {
    let mut flat = Vec::with_capacity(lineups.len() * roster_size);
    for lineup in lineups {
        if lineup.len() != roster_size {
            return Err(ConvertError::WrongLineupLength {
                got: lineup.len(),
                expected: roster_size,
            });
        }
        flat.extend(lineup.iter().map(|&i| i64::from(i)));
    }
    Ok(flat)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrays<'a>(key_columns: &'a [i32]) -> SpecArrays<'a> {
        SpecArrays {
            slot_eligible: &[0b010, 0b100, 0b001],
            slot_counts: &[1, 2, 1],
            salary_cap: 50_000,
            salary_floor: 49_000,
            group_key_columns: &[0, 0],
            group_max_counts: &[6, 5],
            group_slot_masks: &[0b111, 0b011],
            key_columns,
        }
    }

    #[test]
    fn spec_round_trips_through_flat_arrays() {
        let keys = [0, 1, 0, 1, 2, 2];
        let spec = spec_from_arrays(arrays(&keys), 6).unwrap();
        assert_eq!(spec.roster_size(), 4);
        assert_eq!(spec.slots.len(), 3);
        assert_eq!(spec.slots[1].count, 2);
        assert_eq!(spec.groups.len(), 2);
        assert_eq!(spec.groups[1].max_count, 5);
        assert_eq!(spec.groups[1].slots, 0b011);
        assert_eq!(spec.key_columns, vec![vec![0, 1, 0, 1, 2, 2]]);
    }

    #[test]
    fn two_key_columns_split_at_the_pool_boundary() {
        let keys = [0, 0, 1, 5, 6, 7];
        let mut a = arrays(&keys);
        a.slot_eligible = &[1];
        a.slot_counts = &[1];
        a.group_key_columns = &[0, 1];
        let spec = spec_from_arrays(a, 3).unwrap();
        assert_eq!(spec.key_columns, vec![vec![0, 0, 1], vec![5, 6, 7]]);
    }

    #[test]
    fn mismatched_slot_arrays_are_rejected() {
        let keys = [0];
        let mut a = arrays(&keys);
        a.slot_eligible = &[1, 2];
        a.slot_counts = &[1];
        assert_eq!(
            spec_from_arrays(a, 1),
            Err(ConvertError::SlotArrayMismatch {
                eligible: 2,
                counts: 1
            })
        );
    }

    #[test]
    fn mismatched_group_arrays_are_rejected() {
        let keys = [0];
        let mut a = arrays(&keys);
        a.group_key_columns = &[0];
        a.group_max_counts = &[1, 2];
        a.group_slot_masks = &[1];
        assert!(matches!(
            spec_from_arrays(a, 1),
            Err(ConvertError::GroupArrayMismatch { .. })
        ));
    }

    #[test]
    fn a_ragged_key_matrix_is_rejected() {
        let keys = [0, 1, 2, 3, 4];
        assert_eq!(
            spec_from_arrays(arrays(&keys), 2),
            Err(ConvertError::RaggedKeyMatrix {
                len: 5,
                n_players: 2
            })
        );
    }

    #[test]
    fn an_empty_pool_is_rejected() {
        let keys: [i32; 0] = [];
        assert_eq!(
            spec_from_arrays(arrays(&keys), 0),
            Err(ConvertError::EmptyPool)
        );
    }

    #[test]
    fn config_round_trips_through_a_flat_profile_matrix() {
        let config = config_from_arrays(
            10,
            42,
            0.25,
            3,
            8,
            &[0.1, 0.7, 0.6, 1.6, 0.3, 1.5, 0.2, 1.2],
        )
        .unwrap();
        assert_eq!(config.num_lineups, 10);
        assert_eq!(config.seed, 42);
        assert_eq!(
            config.profiles,
            vec![JitterProfile::CONTRARIAN, JitterProfile::STANDARD]
        );
    }

    #[test]
    fn a_ragged_profile_matrix_is_rejected() {
        assert_eq!(
            config_from_arrays(1, 0, 0.0, 1, 1, &[0.1, 0.7, 0.6]),
            Err(ConvertError::RaggedProfileMatrix { len: 3 })
        );
    }

    #[test]
    fn flattening_is_row_major() {
        assert_eq!(
            flatten_lineups(&[vec![1, 2, 3], vec![4, 5, 6]], 3).unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn flattening_no_lineups_is_not_an_error() {
        assert!(flatten_lineups(&[], 10).unwrap().is_empty());
    }

    #[test]
    fn a_wrong_length_lineup_is_an_internal_error() {
        assert_eq!(
            flatten_lineups(&[vec![1, 2]], 3),
            Err(ConvertError::WrongLineupLength {
                got: 2,
                expected: 3
            })
        );
    }

    #[test]
    fn every_error_renders_a_message() {
        // Display is what the user actually sees, so it must not be empty for any
        // variant — including ones only reachable from a bug in this crate.
        let errors = [
            ConvertError::SlotArrayMismatch {
                eligible: 1,
                counts: 2,
            },
            ConvertError::GroupArrayMismatch {
                key_columns: 1,
                max_counts: 2,
                slot_masks: 3,
            },
            ConvertError::EmptyPool,
            ConvertError::RaggedKeyMatrix {
                len: 5,
                n_players: 2,
            },
            ConvertError::RaggedProfileMatrix { len: 3 },
            ConvertError::WrongLineupLength {
                got: 1,
                expected: 2,
            },
        ];
        for error in errors {
            assert!(!error.to_string().is_empty(), "{error:?} renders empty");
        }
    }
}
