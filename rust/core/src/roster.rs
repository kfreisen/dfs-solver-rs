//! The roster specification: what makes a lineup legal.
//!
//! The original of this code had DraftKings MLB baked in — roster size 10, seven
//! named positions in a `u8` bitmask, `max_per_team`, and a second special-cased
//! `max_hitters_per_team`. Every one of those is a instance of something more
//! general, and this module is that generalization:
//!
//! * **Slots** are groups of interchangeable roster positions, each with a
//!   bitmask of the player positions eligible to fill them, a count, and the
//!   multipliers a showdown-style slot applies to score and salary.
//! * **Group constraints** cap how many chosen players may share a key. "At most
//!   six players from one team" and "at most five *hitters* from one team" are the
//!   same constraint differing only in which slots they count, so there is no
//!   special case for the second.
//! * **Conflicts** forbid two specific players appearing together. This is the one
//!   rule shape a group cap cannot express, because it is a property of a *pair*
//!   rather than of a shared key: "no hitters against my starting pitcher" depends
//!   on the pitcher's opponent matching the hitter's team, which is a join, not a
//!   grouping.
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
#[derive(Debug, Clone, PartialEq)]
pub struct SlotGroup {
    /// Positions a player must have at least one of to fill this slot.
    pub eligible: PositionMask,
    /// How many slots of this kind the roster has.
    pub count: usize,
    /// What this slot multiplies the occupant's score by. `1.5` is a DraftKings
    /// showdown captain.
    ///
    /// This deliberately does **not** enter the candidate ordering. Every
    /// candidate for a given slot is scaled by the same factor, so a non-negative
    /// multiplier cannot reorder them; applying it would cost a multiply per
    /// candidate and change nothing. It matters for scoring a finished lineup and
    /// for anything downstream that ranks lineups against each other.
    pub score_multiplier: f64,
    /// What this slot multiplies the occupant's salary by. `1.5` is a DraftKings
    /// showdown captain; FanDuel's MVP leaves salary alone, so that is `1.0`.
    ///
    /// Unlike the score multiplier this changes construction throughout: it feeds
    /// the cap check, the cheapest-way-to-finish reservation, and salary repair.
    /// The scaled salary is rounded half away from zero — see [`scaled_salary`].
    pub salary_multiplier: f64,
}

impl SlotGroup {
    /// A plain slot: no multipliers, which is every slot outside a showdown.
    pub fn new(eligible: PositionMask, count: usize) -> Self {
        Self {
            eligible,
            count,
            score_multiplier: 1.0,
            salary_multiplier: 1.0,
        }
    }

    /// A slot that scales what its occupant is worth and what they cost.
    pub fn multiplied(
        eligible: PositionMask,
        count: usize,
        score_multiplier: f64,
        salary_multiplier: f64,
    ) -> Self {
        Self {
            eligible,
            count,
            score_multiplier,
            salary_multiplier,
        }
    }
}

/// Apply a slot's salary multiplier to a base salary.
///
/// Rounded half away from zero, and short-circuited at exactly `1.0` so an
/// ordinary slot cannot drift by a floating-point ulp. The rounding rule is part
/// of the contract rather than an implementation detail: the Python wrapper
/// reproduces it when it reports a lineup's salary, and the two must agree or a
/// lineup the kernel believes is legal reads as over the cap.
pub fn scaled_salary(base: i64, multiplier: f64) -> i64 {
    if multiplier == 1.0 {
        base
    } else {
        (base as f64 * multiplier).round() as i64
    }
}

/// Pairs of players that may not appear in the same lineup.
///
/// Stored as a symmetric adjacency list rather than an `n x n` bitset. Real
/// conflict graphs are sparse — a hitter opposes one starting pitcher, not five
/// hundred — so adjacency makes maintaining the "is this candidate blocked?"
/// state cost `O(degree)` per pick instead of `O(n/64)` per *candidate examined*.
///
/// Symmetry is an invariant, established by [`Self::from_pairs`]. The builder
/// marks a candidate blocked when a player it conflicts with is rostered; if the
/// relation were directed, whichever of the pair was picked first would decide
/// whether the rule applied at all.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConflictGraph {
    /// Neighbours of each player. Empty when no conflicts were declared, which is
    /// the case this type is careful to keep free.
    adjacency: Vec<Vec<u32>>,
}

impl ConflictGraph {
    /// An empty graph: nothing conflicts with anything.
    pub fn none() -> Self {
        Self::default()
    }

    /// Build from unordered pairs over `n_players`.
    ///
    /// Symmetrized and deduplicated. Self-pairs are dropped rather than rejected:
    /// a rule derived from key columns can legitimately match a player against
    /// themselves — a two-way pitcher whose team is also their opponent's — and
    /// the "no duplicate players" rule already covers it.
    ///
    /// Out-of-range indices are dropped for the same reason a caller cannot
    /// usefully act on them: the pairs are usually generated, not typed.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, u32)>, n_players: usize) -> Self {
        // Filtered to the surviving pairs before anything is allocated. The
        // overwhelmingly common call declares no conflicts at all, and eagerly
        // building `vec![Vec::new(); n_players]` only to throw it away would put
        // an allocation proportional to the pool on every single solve.
        let mut kept: Vec<(usize, usize)> = pairs
            .into_iter()
            .map(|(a, b)| (a as usize, b as usize))
            .filter(|&(a, b)| a != b && a < n_players && b < n_players)
            .collect();
        if kept.is_empty() {
            return Self::none();
        }

        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n_players];
        for &(a, b) in &kept {
            adjacency[a].push(b as u32);
            adjacency[b].push(a as u32);
        }
        kept.clear();
        for row in &mut adjacency {
            row.sort_unstable();
            row.dedup();
        }
        Self { adjacency }
    }

    /// Whether any conflict was declared. The hot loop tests this first so that a
    /// spec without conflicts pays nothing.
    pub fn is_empty(&self) -> bool {
        self.adjacency.is_empty()
    }

    /// Players that may not be rostered alongside `player`.
    pub fn neighbors(&self, player: usize) -> &[u32] {
        match self.adjacency.get(player) {
            Some(row) => row,
            None => &[],
        }
    }

    /// Number of players the graph is defined over; 0 when empty.
    fn len(&self) -> usize {
        self.adjacency.len()
    }
}

/// A cap on how many selected players may share a key.
///
/// `max_per_team = 6` is `{ key_column: 0, max_count: 6, slots: all }`.
/// `max_hitters_per_team = 5` is the same with `slots` restricted to hitter slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupConstraint {
    /// Index into [`RosterSpec::key_columns`] naming the key this counts.
    pub key_column: usize,
    /// Maximum number of players sharing one key value. [`UNCAPPED`] for a
    /// constraint that only imposes a minimum.
    pub max_count: u32,
    /// How many *distinct* values of the key must appear. Zero imposes nothing.
    ///
    /// This is the shape behind "players from at least two different games",
    /// which most classic contests require and which a cap cannot express: a cap
    /// bounds one key value, while this counts how many are used at all.
    ///
    /// See [`Self::min_stack`] for the other, quite different, minimum.
    pub min_distinct: u32,
    /// How many players at least one key value must contribute: a stack.
    ///
    /// "At least four hitters from one team" — existential over teams, which is
    /// what separates it from everything else here. A cap and a distinct-minimum
    /// are both properties a finished lineup either has or does not, testable one
    /// player at a time. This one is a property of a *choice*: the builder has to
    /// decide which team it is stacking before it can act on it, because the
    /// constraint says nothing about which team should be the one.
    ///
    /// That choice is drawn per attempt from the same generator as the jitter, so
    /// a portfolio ends up stacked across many teams rather than piling onto one.
    /// See `greedy::Builder::choose_stack_targets`.
    ///
    /// This is a strategy rather than a rule an operator enforces. It lives here
    /// anyway because it is the single most common thing a daily-fantasy builder
    /// is asked to do, and expressing it any other way means post-filtering a
    /// pool that was never built to contain it.
    pub min_stack: u32,
    /// How many players a *second, distinct* key value must contribute.
    ///
    /// `min_stack: 4, secondary_min_stack: 2` is the classic "4-2": one team
    /// stacks four hitters and a different team stacks two. Zero imposes
    /// nothing. Never larger than [`Self::min_stack`] — the primary is the
    /// larger stack by definition, which is what keeps the top-two test in
    /// `GroupTally::top_two` unambiguous.
    pub secondary_min_stack: u32,
    /// Which slot groups count toward this constraint, as a bitmask over
    /// slot-group index.
    pub slots: SlotMask,
}

/// A `max_count` that imposes no ceiling, for a minimum-only constraint.
pub const UNCAPPED: u32 = u32::MAX;

impl GroupConstraint {
    /// A plain cap: at most `max_count` players sharing a key value.
    pub fn at_most(key_column: usize, max_count: u32, slots: SlotMask) -> Self {
        Self {
            key_column,
            max_count,
            min_distinct: 0,
            min_stack: 0,
            secondary_min_stack: 0,
            slots,
        }
    }

    /// A requirement that at least `min_distinct` values of the key be used.
    pub fn distinct(key_column: usize, min_distinct: u32, slots: SlotMask) -> Self {
        Self {
            key_column,
            max_count: UNCAPPED,
            min_distinct,
            min_stack: 0,
            secondary_min_stack: 0,
            slots,
        }
    }

    /// A requirement that some one key value contribute `min_stack` players.
    pub fn stack(key_column: usize, min_stack: u32, slots: SlotMask) -> Self {
        Self {
            key_column,
            max_count: UNCAPPED,
            min_distinct: 0,
            min_stack,
            secondary_min_stack: 0,
            slots,
        }
    }

    /// A "4-2": one key value contributes `min_stack` players and a *different*
    /// one contributes `secondary_min_stack`.
    pub fn stack_pair(
        key_column: usize,
        min_stack: u32,
        secondary_min_stack: u32,
        slots: SlotMask,
    ) -> Self {
        Self {
            key_column,
            max_count: UNCAPPED,
            min_distinct: 0,
            min_stack,
            secondary_min_stack,
            slots,
        }
    }

    /// Whether this constraint imposes anything at all.
    fn is_inert(&self) -> bool {
        self.max_count == UNCAPPED && self.min_distinct == 0 && self.min_stack == 0
    }
}

/// Everything that makes a lineup legal, independent of any sport.
#[derive(Debug, Clone, PartialEq)]
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
    /// Players that may not be rostered together. Empty unless the caller opted
    /// in: "no hitters against my pitcher" is a preference, not a contest rule,
    /// and a contrarian who wants that correlation must be able to have it.
    pub conflicts: ConflictGraph,
}

/// Why a [`RosterSpec`] could not be used.
///
/// Not `Eq`: `BadMultiplier` carries the offending `f64` so the message can name
/// it, and reporting `NaN` is exactly the case that matters most.
#[derive(Debug, Clone, PartialEq)]
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
    InertGroup(usize),
    /// A minimum asks for more players than the slots it counts can hold.
    MinimumExceedsSlots {
        group: usize,
        which: &'static str,
        wanted: u32,
        slots: usize,
    },
    /// A stack minimum is larger than the cap on the same key.
    StackAboveCap {
        group: usize,
        min_stack: u32,
        max_count: u32,
    },
    /// A secondary stack without a primary, or larger than the primary.
    SecondaryStackShape {
        group: usize,
        min_stack: u32,
        secondary: u32,
    },
    /// The two stacks together need more players than the counted slots hold.
    StackPairExceedsSlots {
        group: usize,
        min_stack: u32,
        secondary: u32,
        slots: usize,
    },
    BadMultiplier {
        slot: usize,
        which: &'static str,
        value: f64,
    },
    ConflictGraphLengthMismatch {
        len: usize,
        expected: usize,
    },
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
            Self::InertGroup(i) => write!(
                f,
                "group constraint {i} sets no cap, no distinct minimum and no stack, \
                 so it constrains nothing; remove it"
            ),
            Self::MinimumExceedsSlots {
                group,
                which,
                wanted,
                slots,
            } => write!(
                f,
                "group constraint {group} requires {which} {wanted} but counts only \
                 {slots} roster slot(s), so no lineup can satisfy it"
            ),
            Self::StackAboveCap {
                group,
                min_stack,
                max_count,
            } => write!(
                f,
                "group constraint {group} requires a stack of {min_stack} but caps the \
                 same key at {max_count}; those cannot both hold"
            ),
            Self::SecondaryStackShape {
                group,
                min_stack,
                secondary,
            } => write!(
                f,
                "group constraint {group} has secondary_min_stack {secondary} against \
                 min_stack {min_stack}; the secondary needs a primary and may not \
                 exceed it"
            ),
            Self::StackPairExceedsSlots {
                group,
                min_stack,
                secondary,
                slots,
            } => write!(
                f,
                "group constraint {group} requires stacks of {min_stack} and {secondary} \
                 from different key values but counts only {slots} roster slot(s), so no \
                 lineup can satisfy both"
            ),
            Self::BadMultiplier { slot, which, value } => write!(
                f,
                "slot group {slot} has {which} multiplier {value}; it must be finite \
                 and non-negative"
            ),
            Self::ConflictGraphLengthMismatch { len, expected } => write!(
                f,
                "the conflict graph is defined over {len} players but the pool has {expected}"
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
            for (which, value) in [
                ("score", slot.score_multiplier),
                ("salary", slot.salary_multiplier),
            ] {
                // A negative salary multiplier would let a slot pay the caller,
                // which breaks the monotonicity the reservation bound assumes; a
                // NaN silently poisons every comparison it reaches.
                if !value.is_finite() || value < 0.0 {
                    return Err(SpecError::BadMultiplier {
                        slot: i,
                        which,
                        value,
                    });
                }
            }
        }
        if !self.conflicts.is_empty() && self.conflicts.len() != n_players {
            return Err(SpecError::ConflictGraphLengthMismatch {
                len: self.conflicts.len(),
                expected: n_players,
            });
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
            if group.is_inert() {
                return Err(SpecError::InertGroup(i));
            }
            // How many roster slots this constraint actually counts. A minimum
            // asking for more than that can never be met, and would otherwise
            // burn the entire attempt budget proving it.
            let counted = self.counted_slots(group.slots);
            for (which, wanted) in [
                ("min_distinct", group.min_distinct),
                ("min_stack", group.min_stack),
            ] {
                if wanted as usize > counted {
                    return Err(SpecError::MinimumExceedsSlots {
                        group: i,
                        which,
                        wanted,
                        slots: counted,
                    });
                }
            }
            if group.min_stack > group.max_count {
                return Err(SpecError::StackAboveCap {
                    group: i,
                    min_stack: group.min_stack,
                    max_count: group.max_count,
                });
            }
            if group.secondary_min_stack > 0
                && (group.min_stack == 0 || group.secondary_min_stack > group.min_stack)
            {
                return Err(SpecError::SecondaryStackShape {
                    group: i,
                    min_stack: group.min_stack,
                    secondary: group.secondary_min_stack,
                });
            }
            // The two stacks occupy disjoint key values, so they need room for
            // their sum, not for each alone.
            if (group.min_stack + group.secondary_min_stack) as usize > counted {
                return Err(SpecError::StackPairExceedsSlots {
                    group: i,
                    min_stack: group.min_stack,
                    secondary: group.secondary_min_stack,
                    slots: counted,
                });
            }
        }
        Ok(())
    }

    /// What every player costs in every slot group, as a flat
    /// `(n_slot_groups, n_players)` row-major table.
    ///
    /// Materialized once per solve rather than multiplied at the point of use.
    /// The fill loop holds one slot group at a time, so it takes a row up front
    /// and indexes it exactly as it used to index `pool.salaries` — the multiplier
    /// costs nothing in the inner loop, which is the whole reason for the table.
    /// A pool of 1000 players across 9 slot groups is 72 KB, built once against
    /// thousands of lineups.
    ///
    /// Callers should skip this entirely when [`Self::has_salary_multipliers`] is
    /// false, in which case every row would be a copy of `salaries`.
    pub fn slot_salary_table(&self, salaries: &[i64]) -> Vec<i64> {
        let mut table = Vec::with_capacity(self.slots.len() * salaries.len());
        for slot in &self.slots {
            table.extend(
                salaries
                    .iter()
                    .map(|&base| scaled_salary(base, slot.salary_multiplier)),
            );
        }
        table
    }

    /// How many roster slots a slot mask covers, expanding multi-count groups.
    fn counted_slots(&self, mask: SlotMask) -> usize {
        self.slots
            .iter()
            .enumerate()
            .filter(|(j, _)| mask & (1u64 << j) != 0)
            .map(|(_, slot)| slot.count)
            .sum()
    }

    /// `[g][j]` is how many roster slots constraint `g` still counts from slot
    /// group `j` onward, as a flat `(n_groups, n_slot_groups + 1)` table.
    ///
    /// This is what makes a distinct-minimum enforceable during a forward fill
    /// rather than only checkable at the end. Knowing how many counted slots are
    /// left is what lets the builder tell "this pick is fine" from "this pick
    /// strands the requirement", and reject the second before it has wasted the
    /// rest of the attempt.
    pub fn counted_suffix(&self) -> Vec<usize> {
        let width = self.slots.len() + 1;
        let mut table = vec![0usize; self.groups.len() * width];
        for (gi, group) in self.groups.iter().enumerate() {
            for j in (0..self.slots.len()).rev() {
                let here = if group.slots & (1u64 << j) != 0 {
                    self.slots[j].count
                } else {
                    0
                };
                table[gi * width + j] = table[gi * width + j + 1] + here;
            }
        }
        table
    }

    /// Whether any constraint imposes a minimum, of either kind.
    pub fn has_minimums(&self) -> bool {
        self.groups
            .iter()
            .any(|g| g.min_distinct > 0 || g.min_stack > 0)
    }

    /// Whether any constraint asks for a stack.
    pub fn has_stacks(&self) -> bool {
        self.groups.iter().any(|g| g.min_stack > 0)
    }

    /// Whether any slot scales salary. Lets a caller skip work that only matters
    /// for showdown-style rosters.
    pub fn has_salary_multipliers(&self) -> bool {
        self.slots.iter().any(|s| s.salary_multiplier != 1.0)
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
///
/// Crate-private on purpose: it is the mutable scratch state of the builder's
/// fill loop, and its `add`/`remove`/`reset` only make sense interleaved with
/// that loop — exposing them would invite a caller to desynchronize the counts
/// from the lineup.
pub(crate) struct GroupTally<'a> {
    spec: &'a RosterSpec,
    n_keys: usize,
    counts: Vec<u32>,
    /// How many distinct key values each constraint currently sees. Maintained
    /// incrementally on the 0↔1 transitions rather than recounted, because the
    /// feasibility test below runs per candidate examined.
    distinct: Vec<u32>,
}

impl<'a> GroupTally<'a> {
    /// Allocate a tally for `spec`. Reused across lineups via [`Self::reset`].
    pub fn new(spec: &'a RosterSpec) -> Self {
        let n_keys = (spec.max_key() + 1).max(0) as usize;
        Self {
            spec,
            n_keys,
            counts: vec![0; spec.groups.len() * n_keys],
            distinct: vec![0; spec.groups.len()],
        }
    }

    /// Zero every count, keeping the allocation.
    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.distinct.fill(0);
    }

    /// How many distinct key values constraint `group_index` currently sees.
    pub fn distinct_count(&self, group_index: usize) -> u32 {
        self.distinct[group_index]
    }

    /// How many players constraint `group_index` has counted for `key`.
    pub fn count_of(&self, group_index: usize, key: i32) -> u32 {
        if key < 0 {
            return 0;
        }
        self.counts[group_index * self.n_keys + key as usize]
    }

    /// The largest number of players any one key value has contributed to
    /// constraint `group_index` — the size of its biggest stack.
    ///
    /// Scanned rather than maintained incrementally. A running maximum is wrong
    /// the moment salary repair removes a player, and this is checked once per
    /// finished lineup rather than once per candidate.
    pub fn max_stack(&self, group_index: usize) -> u32 {
        let start = group_index * self.n_keys;
        self.counts[start..start + self.n_keys]
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// The two largest per-key counts for constraint `group_index`, descending.
    ///
    /// What a secondary stack is judged against: a 4-2 holds when the top count
    /// reaches the primary minimum and the *second* count — necessarily a
    /// different key — reaches the secondary. Same single scan as
    /// [`Self::max_stack`], with two accumulators.
    pub fn top_two(&self, group_index: usize) -> (u32, u32) {
        self.top_two_with(group_index, -1)
    }

    /// [`Self::top_two`] as it would read with one extra player of `extra_key`
    /// counted. `-1` adds nobody. This is how salary repair asks "would this
    /// swap keep the stacks?" without mutating the tally.
    pub fn top_two_with(&self, group_index: usize, extra_key: i32) -> (u32, u32) {
        let start = group_index * self.n_keys;
        let (mut best, mut second) = (0u32, 0u32);
        for (key, &count) in self.counts[start..start + self.n_keys].iter().enumerate() {
            let count = count + u32::from(extra_key == key as i32);
            if count > best {
                second = best;
                best = count;
            } else if count > second {
                second = count;
            }
        }
        (best, second)
    }

    /// The key value `player` carries for constraint `group_index`, or `-1` if
    /// they belong to no group under it.
    pub fn key_of(&self, group_index: usize, player: usize) -> i32 {
        self.spec.key_columns[self.spec.groups[group_index].key_column][player]
    }

    /// Whether placing `player` into `slot_group` would introduce a key value
    /// that constraint `group_index` has not seen yet.
    pub fn is_new_key(&self, group_index: usize, player: usize, slot_group: usize) -> bool {
        let group = &self.spec.groups[group_index];
        if group.slots & (1u64 << slot_group) == 0 {
            return false;
        }
        let key = self.spec.key_columns[group.key_column][player];
        if key < 0 {
            return false;
        }
        self.counts[group_index * self.n_keys + key as usize] == 0
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
            let before = *cell;
            *cell = cell.wrapping_add_signed(delta);
            // Only the transitions in and out of zero move the distinct count.
            if before == 0 && *cell > 0 {
                self.distinct[gi] += 1;
            } else if before > 0 && *cell == 0 {
                self.distinct[gi] -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DraftKings MLB classic, expressed in the general form. Used throughout the
    /// tests and mirrored by `dfs_solver.presets.DK_MLB_CLASSIC` on the Python side —
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
            SlotGroup::new(C, 1),
            SlotGroup::new(SS, 1),
            SlotGroup::new(B2, 1),
            SlotGroup::new(B3, 1),
            SlotGroup::new(B1, 1),
            SlotGroup::new(OF, 3),
            SlotGroup::new(P, 2),
        ];
        // Slot group 6 is the pitchers; every other group is a hitter slot.
        let hitter_slots: SlotMask = 0b0111111;
        let all_slots: SlotMask = 0b1111111;

        RosterSpec {
            slots,
            salary_cap: 50_000,
            salary_floor: 49_000,
            groups: vec![
                GroupConstraint::at_most(0, 6, all_slots),
                GroupConstraint::at_most(0, 5, hitter_slots),
            ],
            key_columns: vec![team_ids],
            conflicts: ConflictGraph::none(),
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
    fn top_two_reports_the_two_biggest_stacks() {
        // Teams 0, 0, 0, 1, 1, 2 across the first six players.
        let spec = dk_mlb(vec![0, 0, 0, 1, 1, 2]);
        let mut tally = GroupTally::new(&spec);
        for i in 0..6 {
            tally.add(i, 0);
        }
        assert_eq!(tally.top_two(0), (3, 2));
        // A virtual extra player of team 2 lifts it to 2, tying second place.
        assert_eq!(tally.top_two_with(0, 2), (3, 2));
        // Of team 1, it takes the lead outright.
        assert_eq!(tally.top_two_with(0, 1), (3, 3));
        // Of nobody, nothing changes.
        assert_eq!(tally.top_two_with(0, -1), (3, 2));
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
    fn an_ordinary_slot_carries_neutral_multipliers() {
        let slot = SlotGroup::new(1, 1);
        assert_eq!(slot.score_multiplier, 1.0);
        assert_eq!(slot.salary_multiplier, 1.0);
    }

    #[test]
    fn scaling_by_one_returns_the_base_exactly() {
        // Not merely "close": routing an ordinary slot through the f64 round-trip
        // would let a large salary drift, and the cap check is an integer compare.
        for base in [0, 1, 3_700, 50_000, i64::MAX / 4] {
            assert_eq!(scaled_salary(base, 1.0), base);
        }
    }

    #[test]
    fn scaling_rounds_half_away_from_zero() {
        // 3_701 * 1.5 is 5_551.5 exactly. Half-to-even would give 5_552 here and
        // 5_550 for the neighbouring case, which is not what the Python side does.
        assert_eq!(scaled_salary(3_701, 1.5), 5_552);
        assert_eq!(scaled_salary(3_703, 1.5), 5_555);
        assert_eq!(scaled_salary(3_700, 1.5), 5_550);
        assert_eq!(scaled_salary(1_000, 0.0), 0);
    }

    #[test]
    fn the_slot_salary_table_is_row_major_by_slot_group() {
        let mut spec = dk_mlb(vec![0; 3]);
        spec.slots = vec![SlotGroup::multiplied(1, 1, 1.5, 1.5), SlotGroup::new(1, 2)];
        let table = spec.slot_salary_table(&[1_000, 2_000, 3_000]);
        assert_eq!(table, vec![1_500, 3_000, 4_500, 1_000, 2_000, 3_000]);
    }

    #[test]
    fn a_spec_without_multipliers_says_so() {
        assert!(!dk_mlb(vec![0]).has_salary_multipliers());
        let mut spec = dk_mlb(vec![0]);
        spec.slots[0].salary_multiplier = 1.5;
        assert!(spec.has_salary_multipliers());
    }

    #[test]
    fn a_nan_multiplier_is_rejected_rather_than_poisoning_every_comparison() {
        let mut spec = dk_mlb(vec![0]);
        spec.slots[2].score_multiplier = f64::NAN;
        assert!(matches!(
            spec.validate(1),
            Err(SpecError::BadMultiplier {
                slot: 2,
                which: "score",
                ..
            })
        ));
    }

    #[test]
    fn a_negative_salary_multiplier_is_rejected() {
        let mut spec = dk_mlb(vec![0]);
        spec.slots[0].salary_multiplier = -1.0;
        assert_eq!(
            spec.validate(1),
            Err(SpecError::BadMultiplier {
                slot: 0,
                which: "salary",
                value: -1.0
            })
        );
    }

    #[test]
    fn conflicts_are_symmetric_however_the_pair_was_given() {
        let graph = ConflictGraph::from_pairs([(0, 3)], 4);
        assert_eq!(graph.neighbors(0), &[3]);
        assert_eq!(graph.neighbors(3), &[0]);
        assert!(graph.neighbors(1).is_empty());
    }

    #[test]
    fn duplicate_pairs_collapse() {
        let graph = ConflictGraph::from_pairs([(0, 1), (1, 0), (0, 1)], 2);
        assert_eq!(graph.neighbors(0), &[1]);
        assert_eq!(graph.neighbors(1), &[0]);
    }

    #[test]
    fn self_pairs_and_out_of_range_pairs_are_dropped_not_rejected() {
        // Both arise from generated rules rather than typed input: a two-way
        // player can match themselves, and the caller cannot act on the report.
        let graph = ConflictGraph::from_pairs([(1, 1), (0, 9)], 3);
        assert!(graph.is_empty());
    }

    #[test]
    fn a_graph_with_no_pairs_is_empty_so_the_hot_loop_can_skip_it() {
        let graph = ConflictGraph::from_pairs([], 100);
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert!(graph.neighbors(0).is_empty());
    }

    #[test]
    fn a_conflict_graph_sized_for_another_pool_is_rejected() {
        let mut spec = dk_mlb(vec![0; 4]);
        spec.conflicts = ConflictGraph::from_pairs([(0, 1)], 4);
        assert_eq!(
            spec.validate(6),
            Err(SpecError::ConflictGraphLengthMismatch {
                len: 4,
                expected: 6
            })
        );
        assert_eq!(spec.validate(4), Ok(()));
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
