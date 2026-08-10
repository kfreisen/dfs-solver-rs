//! The roster specification: what makes a lineup legal.
//!
//! The original of this code had DraftKings MLB baked in — roster size 10, seven
//! named positions in a `u8` bitmask, `max_per_team`, and a second special-cased
//! `max_hitters_per_team`. Every one of those is a instance of something more
//! general, and this module is that generalization:
//!
//! * **Slots** are groups of interchangeable roster positions, each with a
//!   bitmask of the player positions eligible to fill them and a count.
//! * **Group constraints** cap how many chosen players may share a key. "At most
//!   six players from one team" and "at most five *hitters* from one team" are the
//!   same constraint differing only in which slots they count, so there is no
//!   special case for the second.
//!
//! One deliberate behavior change from the original is recorded here. The original
//! decided "is this a hitter?" from the slot being filled during construction but
//! from the player's own position mask during salary repair. Those agree for
//! DraftKings MLB, where pitchers are eligible for nothing else, and disagree for
//! any sport with overlapping eligibility. This implementation counts by **slot**
//! everywhere: the slot is what the player was actually rostered as, and it is
//! unambiguous for multi-position players.

/// Bitmask over player positions. 32 positions is comfortably more than any real
/// sport uses, and keeping it a single word keeps eligibility a branchless `&`.
pub type PositionMask = u32;

/// Bitmask over slot groups, used to say which slots a group constraint counts.
pub type SlotMask = u64;

/// The maximum number of distinct slot groups a specification may declare.
pub const MAX_SLOT_GROUPS: usize = 64;

/// A run of interchangeable roster slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotGroup {
    /// Positions a player must have at least one of to fill this slot.
    pub eligible: PositionMask,
    /// How many slots of this kind the roster has.
    pub count: usize,
}

/// A cap on how many selected players may share a key.
///
/// `max_per_team = 6` is `{ key_column: 0, max_count: 6, slots: all }`.
/// `max_hitters_per_team = 5` is the same with `slots` restricted to hitter slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupConstraint {
    /// Index into [`RosterSpec::key_columns`] naming the key this counts.
    pub key_column: usize,
    /// Maximum number of players sharing one key value.
    pub max_count: u32,
    /// Which slot groups count toward this cap, as a bitmask over slot-group index.
    pub slots: SlotMask,
}

/// Everything that makes a lineup legal, independent of any sport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RosterSpec {
    /// Slot groups in fill order. Order matters: the greedy construction fills
    /// these left to right, so scarce positions belong first — a lineup that
    /// spends its budget on outfielders before looking for a catcher will fail to
    /// find one far more often.
    pub slots: Vec<SlotGroup>,
    /// Total salary a lineup may not exceed.
    pub salary_cap: i64,
    /// Minimum total salary. Zero disables the floor.
    pub salary_floor: i64,
    /// Group caps, evaluated against `key_columns`.
    pub groups: Vec<GroupConstraint>,
    /// Per-player key values, one column per distinct key. A negative value means
    /// the player belongs to no group for that column and is never capped by it.
    pub key_columns: Vec<Vec<i32>>,
}

/// Why a [`RosterSpec`] could not be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecError {
    NoSlots,
    TooManySlotGroups(usize),
    EmptySlotGroup(usize),
    SlotWithNoEligiblePositions(usize),
    FloorAboveCap {
        floor: i64,
        cap: i64,
    },
    UnknownKeyColumn {
        group: usize,
        column: usize,
    },
    KeyColumnLengthMismatch {
        column: usize,
        len: usize,
        expected: usize,
    },
    ZeroMaxCount(usize),
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSlots => write!(f, "roster specification declares no slots"),
            Self::TooManySlotGroups(n) => write!(
                f,
                "roster specification declares {n} slot groups; the limit is {MAX_SLOT_GROUPS} \
                 because group constraints address slots with a 64-bit mask"
            ),
            Self::EmptySlotGroup(i) => {
                write!(f, "slot group {i} has count 0; remove it instead")
            }
            Self::SlotWithNoEligiblePositions(i) => write!(
                f,
                "slot group {i} has an empty eligibility mask, so no player can ever fill it"
            ),
            Self::FloorAboveCap { floor, cap } => write!(
                f,
                "salary floor {floor} exceeds cap {cap}, so no lineup can be valid"
            ),
            Self::UnknownKeyColumn { group, column } => write!(
                f,
                "group constraint {group} refers to key column {column}, which does not exist"
            ),
            Self::KeyColumnLengthMismatch {
                column,
                len,
                expected,
            } => write!(
                f,
                "key column {column} has {len} entries but the player pool has {expected}"
            ),
            Self::ZeroMaxCount(i) => write!(
                f,
                "group constraint {i} has max_count 0, which forbids every lineup; \
                 exclude those players from the pool instead"
            ),
        }
    }
}

impl std::error::Error for SpecError {}

impl RosterSpec {
    /// Total number of players a complete lineup holds.
    pub fn roster_size(&self) -> usize {
        self.slots.iter().map(|s| s.count).sum()
    }

    /// Number of distinct slot groups.
    pub fn n_slot_groups(&self) -> usize {
        self.slots.len()
    }

    /// Check the specification is self-consistent and matches a pool of `n_players`.
    ///
    /// Called once per solve rather than per lineup. Every error here would
    /// otherwise surface as "no valid lineups found", which is the least
    /// actionable failure a solver can produce.
    pub fn validate(&self, n_players: usize) -> Result<(), SpecError> {
        if self.slots.is_empty() {
            return Err(SpecError::NoSlots);
        }
        if self.slots.len() > MAX_SLOT_GROUPS {
            return Err(SpecError::TooManySlotGroups(self.slots.len()));
        }
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.count == 0 {
                return Err(SpecError::EmptySlotGroup(i));
            }
            if slot.eligible == 0 {
                return Err(SpecError::SlotWithNoEligiblePositions(i));
            }
        }
        if self.salary_floor > self.salary_cap {
            return Err(SpecError::FloorAboveCap {
                floor: self.salary_floor,
                cap: self.salary_cap,
            });
        }
        for (i, column) in self.key_columns.iter().enumerate() {
            if column.len() != n_players {
                return Err(SpecError::KeyColumnLengthMismatch {
                    column: i,
                    len: column.len(),
                    expected: n_players,
                });
            }
        }
        for (i, group) in self.groups.iter().enumerate() {
            if group.key_column >= self.key_columns.len() {
                return Err(SpecError::UnknownKeyColumn {
                    group: i,
                    column: group.key_column,
                });
            }
            if group.max_count == 0 {
                return Err(SpecError::ZeroMaxCount(i));
            }
        }
        Ok(())
    }

    /// Highest key value across all columns, used to size counting arrays.
    pub fn max_key(&self) -> i32 {
        self.key_columns
            .iter()
            .flat_map(|c| c.iter().copied())
            .max()
            .unwrap_or(-1)
    }
}

/// Running tally of the group constraints while a lineup is being built.
///
/// Counts live in one flat array indexed by `(group, key)` so that incrementing on
/// each pick is a single add rather than a walk over the partial lineup — which is
/// what the original did during repair, at O(roster_size) per candidate examined.
pub struct GroupTally<'a> {
    spec: &'a RosterSpec,
    n_keys: usize,
    counts: Vec<u32>,
}

impl<'a> GroupTally<'a> {
    /// Allocate a tally for `spec`. Reused across lineups via [`Self::reset`].
    pub fn new(spec: &'a RosterSpec) -> Self {
        let n_keys = (spec.max_key() + 1).max(0) as usize;
        Self {
            spec,
            n_keys,
            counts: vec![0; spec.groups.len() * n_keys],
        }
    }

    /// Zero every count, keeping the allocation.
    pub fn reset(&mut self) {
        self.counts.fill(0);
    }

    /// Whether adding `player` into slot group `slot_group` would breach a cap.
    pub fn would_exceed(&self, player: usize, slot_group: usize) -> bool {
        for (gi, group) in self.spec.groups.iter().enumerate() {
            if group.slots & (1u64 << slot_group) == 0 {
                continue;
            }
            let key = self.spec.key_columns[group.key_column][player];
            if key < 0 {
                continue;
            }
            if self.counts[gi * self.n_keys + key as usize] >= group.max_count {
                return true;
            }
        }
        false
    }

    /// Record that `player` was placed into slot group `slot_group`.
    pub fn add(&mut self, player: usize, slot_group: usize) {
        self.apply(player, slot_group, 1);
    }

    /// Undo a previous [`Self::add`]. Used by salary repair, which tries a swap
    /// out before it knows whether a replacement exists.
    pub fn remove(&mut self, player: usize, slot_group: usize) {
        self.apply(player, slot_group, -1);
    }

    fn apply(&mut self, player: usize, slot_group: usize, delta: i32) {
        for (gi, group) in self.spec.groups.iter().enumerate() {
            if group.slots & (1u64 << slot_group) == 0 {
                continue;
            }
            let key = self.spec.key_columns[group.key_column][player];
            if key < 0 {
                continue;
            }
            let cell = &mut self.counts[gi * self.n_keys + key as usize];
            *cell = cell.wrapping_add_signed(delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DraftKings MLB classic, expressed in the general form. Used throughout the
    /// tests and mirrored by `mlb_dfs_solver.presets.DK_MLB_CLASSIC` on the Python side —
    /// the parity test against the reference implementation is only meaningful if
    /// these two agree.
    fn dk_mlb(team_ids: Vec<i32>) -> RosterSpec {
        const P: PositionMask = 1 << 0;
        const C: PositionMask = 1 << 1;
        const B1: PositionMask = 1 << 2;
        const B2: PositionMask = 1 << 3;
        const B3: PositionMask = 1 << 4;
        const SS: PositionMask = 1 << 5;
        const OF: PositionMask = 1 << 6;

        // Fill order is scarce-first, which is why C leads and P trails.
        let slots = vec![
            SlotGroup {
                eligible: C,
                count: 1,
            },
            SlotGroup {
                eligible: SS,
                count: 1,
            },
            SlotGroup {
                eligible: B2,
                count: 1,
            },
            SlotGroup {
                eligible: B3,
                count: 1,
            },
            SlotGroup {
                eligible: B1,
                count: 1,
            },
            SlotGroup {
                eligible: OF,
                count: 3,
            },
            SlotGroup {
                eligible: P,
                count: 2,
            },
        ];
        // Slot group 6 is the pitchers; every other group is a hitter slot.
        let hitter_slots: SlotMask = 0b0111111;
        let all_slots: SlotMask = 0b1111111;

        RosterSpec {
            slots,
            salary_cap: 50_000,
            salary_floor: 49_000,
            groups: vec![
                GroupConstraint {
                    key_column: 0,
                    max_count: 6,
                    slots: all_slots,
                },
                GroupConstraint {
                    key_column: 0,
                    max_count: 5,
                    slots: hitter_slots,
                },
            ],
            key_columns: vec![team_ids],
        }
    }

    #[test]
    fn dk_mlb_roster_size_is_ten() {
        assert_eq!(dk_mlb(vec![0; 4]).roster_size(), 10);
    }

    #[test]
    fn valid_spec_passes_validation() {
        assert_eq!(dk_mlb(vec![0, 1, 2, 3]).validate(4), Ok(()));
    }

    #[test]
    fn key_column_must_match_pool_size() {
        let spec = dk_mlb(vec![0, 1]);
        assert_eq!(
            spec.validate(5),
            Err(SpecError::KeyColumnLengthMismatch {
                column: 0,
                len: 2,
                expected: 5
            })
        );
    }

    #[test]
    fn floor_above_cap_is_rejected() {
        let mut spec = dk_mlb(vec![0]);
        spec.salary_floor = 60_000;
        assert_eq!(
            spec.validate(1),
            Err(SpecError::FloorAboveCap {
                floor: 60_000,
                cap: 50_000
            })
        );
    }

    #[test]
    fn a_group_naming_a_missing_key_column_is_rejected() {
        let mut spec = dk_mlb(vec![0]);
        spec.groups[0].key_column = 7;
        assert_eq!(
            spec.validate(1),
            Err(SpecError::UnknownKeyColumn {
                group: 0,
                column: 7
            })
        );
    }

    #[test]
    fn a_zero_cap_is_rejected_rather_than_silently_forbidding_every_lineup() {
        let mut spec = dk_mlb(vec![0]);
        spec.groups[0].max_count = 0;
        assert_eq!(spec.validate(1), Err(SpecError::ZeroMaxCount(0)));
    }

    #[test]
    fn empty_eligibility_is_rejected() {
        let mut spec = dk_mlb(vec![0]);
        spec.slots[0].eligible = 0;
        assert_eq!(
            spec.validate(1),
            Err(SpecError::SlotWithNoEligiblePositions(0))
        );
    }

    #[test]
    fn tally_caps_total_players_per_team() {
        // Six players, all on team 0.
        let spec = dk_mlb(vec![0; 6]);
        let mut tally = GroupTally::new(&spec);
        // Fill hitter slots; the hitter cap of 5 bites before the total cap of 6.
        for i in 0..5 {
            assert!(!tally.would_exceed(i, 0), "hitter {i} should fit");
            tally.add(i, 0);
        }
        assert!(
            tally.would_exceed(5, 0),
            "a sixth hitter breaches the 5-hitter cap"
        );
        // The same player in a pitcher slot is fine: the hitter cap does not count
        // pitcher slots, and the total cap still has room for a sixth.
        assert!(!tally.would_exceed(5, 6), "a pitcher is not a hitter");
        tally.add(5, 6);
        // Now the total cap of 6 is reached, so nothing else from team 0 fits.
        assert!(tally.would_exceed(0, 6));
    }

    #[test]
    fn negative_keys_are_never_capped() {
        // A player with no team — a spec might use -1 for free agents.
        let spec = dk_mlb(vec![-1; 8]);
        let mut tally = GroupTally::new(&spec);
        for i in 0..8 {
            assert!(!tally.would_exceed(i, 0));
            tally.add(i, 0);
        }
    }

    #[test]
    fn remove_undoes_add() {
        let spec = dk_mlb(vec![0; 6]);
        let mut tally = GroupTally::new(&spec);
        for i in 0..5 {
            tally.add(i, 0);
        }
        assert!(tally.would_exceed(5, 0));
        tally.remove(4, 0);
        assert!(!tally.would_exceed(5, 0), "removing a hitter frees the cap");
    }

    #[test]
    fn reset_clears_every_count() {
        let spec = dk_mlb(vec![0; 6]);
        let mut tally = GroupTally::new(&spec);
        for i in 0..5 {
            tally.add(i, 0);
        }
        tally.reset();
        assert!(!tally.would_exceed(5, 0));
    }

    #[test]
    fn a_spec_with_no_groups_caps_nothing() {
        let mut spec = dk_mlb(vec![0; 12]);
        spec.groups.clear();
        let mut tally = GroupTally::new(&spec);
        for i in 0..12 {
            assert!(!tally.would_exceed(i, 0));
            tally.add(i, 0);
        }
    }
}
