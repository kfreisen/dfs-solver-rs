//! Randomized greedy lineup construction.
//!
//! One lineup is built by perturbing every player's objective, sorting, and
//! filling slots scarce-first with the best eligible player who does not break a
//! constraint. That is a weak optimizer on its own — it is not trying to find the
//! best lineup, and an ILP solver will beat it on any single one.
//!
//! It is the right tool for the job it actually has, which is producing a large,
//! *diverse* pool of valid lineups to select a portfolio from. Asking a solver for
//! 150 optimal lineups gets 150 nearly identical lineups; the interesting variance
//! is in the tail, and randomized construction covers it for a fraction of the
//! cost. See `submodular.rs` for the selection step that consumes this pool.
//!
//! **Determinism.** Work is split across a fixed number of chunks — not
//! `rayon::current_num_threads()` — and each chunk seeds its own generator from
//! `(seed, chunk_index)`. The output therefore depends on the seed and the chunk
//! count and nothing else, so a result reproduces across machines with different
//! core counts, and serial and parallel runs agree. Deriving chunk count from the
//! thread pool would make every benchmark irreproducible on another laptop.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::roster::{GroupTally, PositionMask, RosterSpec, SlotGroup, SpecError};

/// What every player costs in every slot group.
///
/// The fill loop takes one row per slot group and then indexes it by player, so
/// a showdown captain costs its multiplied salary at exactly the cost of the
/// plain lookup it replaced.
///
/// Two shapes, because a roster without multipliers is the common one and would
/// otherwise pay for a table whose rows are all identical: `Uniform` hands back
/// the pool's own salaries. `row` is called once per slot group per lineup —
/// roughly ten times against thousands of player comparisons — so the branch is
/// free at the scale that matters.
#[derive(Debug, Clone, Copy)]
enum SlotSalaries<'a> {
    /// No slot scales salary; every row is the base column.
    Uniform(&'a [i64]),
    /// Flat `(n_slot_groups, n_players)` row-major table.
    PerSlot { table: &'a [i64], n_players: usize },
}

impl<'a> SlotSalaries<'a> {
    fn row(&self, slot_group: usize) -> &'a [i64] {
        match *self {
            Self::Uniform(salaries) => salaries,
            Self::PerSlot { table, n_players } => {
                let start = slot_group * n_players;
                &table[start..start + n_players]
            }
        }
    }
}

/// The players available to build from. Columns rather than a struct-of-arrays
/// because this arrives from NumPy and is read in tight loops.
#[derive(Debug, Clone, Copy)]
pub struct PlayerPool<'a> {
    /// Expected score.
    pub projections: &'a [f64],
    /// Standard deviation of score, used to bias toward upside.
    pub stddevs: &'a [f64],
    /// Salary cost.
    pub salaries: &'a [i64],
    /// Projected ownership in [0, 1]. Used to fade popular players.
    pub ownership: &'a [f64],
    /// Eligible positions, as a bitmask matching the spec's slot masks.
    pub positions: &'a [PositionMask],
}

impl PlayerPool<'_> {
    /// Number of players.
    pub fn len(&self) -> usize {
        self.projections.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.projections.is_empty()
    }

    /// Check every column has the same length.
    fn validate(&self) -> Result<(), BuildError> {
        let n = self.projections.len();
        for (name, len) in [
            ("stddevs", self.stddevs.len()),
            ("salaries", self.salaries.len()),
            ("ownership", self.ownership.len()),
            ("positions", self.positions.len()),
        ] {
            if len != n {
                return Err(BuildError::ColumnLengthMismatch {
                    column: name,
                    len,
                    expected: n,
                });
            }
        }
        Ok(())
    }
}

/// How much to perturb the objective on one attempt.
///
/// Sampling these per lineup rather than fixing them is what produces diversity:
/// a high `ceiling` draw chases upside, a high `leverage` draw fades chalk, and
/// the two together explore different corners of the pool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JitterProfile {
    /// Range for the multiplier on each player's standard deviation. Higher means
    /// more willing to take a volatile player over a steady one.
    pub ceiling: (f64, f64),
    /// Range for the exponent on `(1 - ownership)`. Higher means a stronger
    /// discount on popular players.
    pub leverage: (f64, f64),
}

impl JitterProfile {
    /// Balanced: mild upside chasing, mild ownership fade.
    pub const STANDARD: Self = Self {
        ceiling: (0.3, 1.5),
        leverage: (0.2, 1.2),
    };

    /// Contrarian: less upside chasing, stronger ownership fade. Alternating this
    /// with [`Self::STANDARD`] reaches lineups neither profile finds alone — the
    /// contrarian draw stops the pool collapsing onto the same high-ceiling core.
    pub const CONTRARIAN: Self = Self {
        ceiling: (0.1, 0.7),
        leverage: (0.6, 1.6),
    };
}

/// Knobs for a construction run.
#[derive(Debug, Clone, PartialEq)]
pub struct GreedyConfig {
    /// How many distinct lineups to return. Fewer may come back if the pool
    /// cannot support that many.
    pub num_lineups: usize,
    /// Base seed. Together with the chunk count this fixes the output exactly.
    pub seed: u64,
    /// Scale of the uniform noise added to each objective, as a fraction of the
    /// player's projection. Zero makes construction deterministic given a profile
    /// draw, which collapses diversity — it is allowed, for testing.
    pub noise: f64,
    /// Attempts to make per requested lineup before giving up. Attempts fail when
    /// the greedy fill paints itself into a corner, so a tight pool needs more.
    pub attempts_per_lineup: usize,
    /// Upper bound on independent work units. Fixed rather than thread-derived,
    /// so results do not depend on the machine.
    ///
    /// An upper bound rather than a literal count: splitting fewer than
    /// [`MIN_ATTEMPTS_PER_CHUNK`] attempts into a chunk breaks the profile cycle
    /// and `diversity_weight`, so [`effective_chunks`] clamps it. Consequently
    /// this changes the output only while it is the binding constraint — above
    /// the clamp, two different values give the same answer.
    pub chunks: usize,
    /// Profiles cycled across attempts.
    pub profiles: Vec<JitterProfile>,
    /// How strongly to price salary into a player's value, as a multiple of the
    /// pool's own points-per-dollar rate.
    ///
    /// Zero ranks players by projection alone, which is what this did
    /// originally and which systematically overspends: the greedy takes the
    /// best remaining player at every slot and arrives at the cheap ones with
    /// no money left. Measured, its best candidate reached 0.93 of the proven
    /// optimum however large the pool grew, and the rosters it liked used
    /// players at points-per-dollar rank 62 of 288 where the optimal roster
    /// used rank 20.
    ///
    /// At 1.0 a player is ranked by their surplus over what the pool charges
    /// for a point on average — the Lagrangian view of the salary constraint,
    /// with the shadow price approximated by that rate. That found the exact
    /// optimum on the slate it was measured against.
    ///
    /// The default is deliberately short of it. Quality is flat between 0.75
    /// and 1.0 across every constraint rung tested (0.981-0.982 of optimum),
    /// and just above there is a cliff: at 1.25 the pool collapsed from fifteen
    /// thousand distinct lineups to five hundred, because once cheap players
    /// dominate outright every attempt converges on the same ones.
    pub value_weight: f64,
    /// Players forced into every lineup, each already assigned to the slot group
    /// that will hold it: `(player, slot_group)`.
    ///
    /// The assignment arrives decided rather than being worked out here. Choosing
    /// which slot holds which lock is a bipartite matching — two locks both
    /// eligible for the flex, only one of which can also play tight end — and
    /// resolving it greedily mid-fill would reject rosters that are perfectly
    /// legal. It depends only on eligibility and slot counts, so it is settled
    /// once, before construction, where it can be done properly and explained.
    pub locks: Vec<(u32, usize)>,
    /// How strongly to fade a player already used by the lineups accepted so far.
    ///
    /// A player's value drops by `diversity_weight * share * mean_projection`,
    /// where `share` is the fraction of accepted lineups already containing them.
    /// Zero disables it and reproduces the original output exactly.
    ///
    /// This is a *preference*, not a constraint, which is the whole reason it
    /// exists next to `exposure_limits`. A cap rejects a finished lineup at the
    /// merge and costs yield; this steers construction before the lineup exists,
    /// so the portfolio spreads out without any lineup being thrown away.
    ///
    /// Subtractive rather than multiplicative because `value` is a *priced*
    /// projection and can be negative — scaling a negative number down would
    /// make an over-used, over-priced player look better.
    ///
    /// Each chunk fades against its own accepted lineups. That needs enough
    /// attempts per chunk to have a history worth reading:
    /// `num_lineups * attempts_per_lineup / chunks`, which wants to be at least
    /// about three. Below that the mechanism is a no-op and the fix is fewer
    /// `chunks`, not a larger weight.
    pub diversity_weight: f64,
    /// Per-player ceiling on how many returned lineups may contain that player.
    /// Empty means uncapped; `u32::MAX` means uncapped for one player.
    ///
    /// A count rather than a fraction, because the fraction is of the *requested*
    /// portfolio size. Deriving it from the running total instead would put the
    /// very first lineup over any cap below 100%.
    pub exposure_limits: Vec<u32>,
}

impl Default for GreedyConfig {
    fn default() -> Self {
        Self {
            num_lineups: 200,
            seed: 0,
            noise: 0.35,
            attempts_per_lineup: 3,
            chunks: 64,
            profiles: vec![JitterProfile::CONTRARIAN, JitterProfile::STANDARD],
            value_weight: 0.75,
            diversity_weight: 0.0,
            locks: Vec::new(),
            exposure_limits: Vec::new(),
        }
    }
}

/// Why construction could not run.
///
/// Not `Eq`, because [`SpecError`] is not: it carries the offending multiplier so
/// that a `NaN` can be named in the message.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    /// The roster specification is itself invalid.
    Spec(SpecError),
    /// A pool column disagrees with the others on length.
    ColumnLengthMismatch {
        column: &'static str,
        len: usize,
        expected: usize,
    },
    /// No profiles supplied, so there is nothing to sample.
    NoProfiles,
    /// Zero chunks, which would do no work.
    NoChunks,
    /// A lock names a player or a slot group that does not exist.
    LockOutOfRange {
        player: u32,
        slot_group: usize,
        n_players: usize,
        n_slot_groups: usize,
    },
    /// A lock puts a player in a slot they cannot fill.
    LockNotEligible { player: u32, slot_group: usize },
    /// More locks were assigned to a slot group than it has slots.
    OverfilledSlotGroup {
        slot_group: usize,
        locks: usize,
        count: usize,
    },
    /// The same player is locked twice.
    DuplicateLock(u32),
    /// The exposure-limit column disagrees with the pool.
    ExposureLengthMismatch { len: usize, expected: usize },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spec(e) => write!(f, "{e}"),
            Self::ColumnLengthMismatch {
                column,
                len,
                expected,
            } => write!(
                f,
                "player pool column '{column}' has {len} entries but 'projections' has {expected}"
            ),
            Self::NoProfiles => write!(f, "config.profiles is empty; supply at least one"),
            Self::NoChunks => write!(f, "config.chunks is 0; supply at least one"),
            Self::LockOutOfRange {
                player,
                slot_group,
                n_players,
                n_slot_groups,
            } => write!(
                f,
                "lock ({player}, slot group {slot_group}) is out of range for a pool of \
                 {n_players} players and {n_slot_groups} slot groups"
            ),
            Self::LockNotEligible { player, slot_group } => write!(
                f,
                "player {player} is locked into slot group {slot_group}, which they are \
                 not eligible to fill"
            ),
            Self::OverfilledSlotGroup {
                slot_group,
                locks,
                count,
            } => write!(
                f,
                "slot group {slot_group} has {count} slot(s) but {locks} locked player(s)"
            ),
            Self::DuplicateLock(player) => write!(
                f,
                "player {player} is locked more than once, but a lineup cannot hold \
                 the same player twice"
            ),
            Self::ExposureLengthMismatch { len, expected } => write!(
                f,
                "exposure_limits has {len} entries but the pool has {expected} players"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<SpecError> for BuildError {
    fn from(e: SpecError) -> Self {
        Self::Spec(e)
    }
}

/// A lineup, as indices into the player pool, in slot order.
pub type Lineup = Vec<u32>;

/// Build a pool of distinct, valid lineups.
///
/// Returns at most `config.num_lineups`. Fewer means the pool could not support
/// more within the attempt budget, which is a real answer rather than a failure:
/// a 12-player slate with a salary floor may genuinely have only a handful of
/// valid lineups.
pub fn build_lineups(
    pool: &PlayerPool<'_>,
    spec: &RosterSpec,
    config: &GreedyConfig,
) -> Result<Vec<Lineup>, BuildError> {
    pool.validate()?;
    spec.validate(pool.len())?;
    if config.profiles.is_empty() {
        return Err(BuildError::NoProfiles);
    }
    if config.chunks == 0 {
        return Err(BuildError::NoChunks);
    }
    if !config.exposure_limits.is_empty() && config.exposure_limits.len() != pool.len() {
        return Err(BuildError::ExposureLengthMismatch {
            len: config.exposure_limits.len(),
            expected: pool.len(),
        });
    }
    // Locks are grouped by the slot that will hold them, so the fill can place
    // them without searching. Validated first: every one of these mistakes would
    // otherwise present as an empty result, and "no valid lineups" says nothing
    // about a player having been locked into a slot they cannot fill.
    //
    // Left empty when nothing is locked, so the fill can test one length and skip
    // the whole mechanism — including the per-chunk `is_locked` column, which
    // would otherwise be an allocation proportional to the pool on every chunk of
    // every build that does not use locks.
    let mut locked_by_slot: Vec<Vec<u32>> = if config.locks.is_empty() {
        Vec::new()
    } else {
        vec![Vec::new(); spec.slots.len()]
    };
    let mut seen_locks: Vec<u32> = Vec::with_capacity(config.locks.len());
    for &(player, slot_group) in &config.locks {
        if player as usize >= pool.len() || slot_group >= spec.slots.len() {
            return Err(BuildError::LockOutOfRange {
                player,
                slot_group,
                n_players: pool.len(),
                n_slot_groups: spec.slots.len(),
            });
        }
        if pool.positions[player as usize] & spec.slots[slot_group].eligible == 0 {
            return Err(BuildError::LockNotEligible { player, slot_group });
        }
        if seen_locks.contains(&player) {
            return Err(BuildError::DuplicateLock(player));
        }
        seen_locks.push(player);
        locked_by_slot[slot_group].push(player);
    }
    for (slot_group, locks) in locked_by_slot.iter().enumerate() {
        if locks.len() > spec.slots[slot_group].count {
            return Err(BuildError::OverfilledSlotGroup {
                slot_group,
                locks: locks.len(),
                count: spec.slots[slot_group].count,
            });
        }
    }
    // Membership by player, for the salary reservation below.
    let mut is_locked_anywhere = vec![false; pool.len()];
    for &player in &seen_locks {
        is_locked_anywhere[player as usize] = true;
    }
    if config.num_lineups == 0 || pool.len() < spec.roster_size() {
        return Ok(Vec::new());
    }

    // Eligible players per slot group, computed once rather than per attempt.
    let eligible: Vec<Vec<u32>> = spec
        .slots
        .iter()
        .map(|slot| {
            (0..pool.len() as u32)
                .filter(|&i| pool.positions[i as usize] & slot.eligible != 0)
                .collect()
        })
        .collect();
    // A slot nobody can fill makes every lineup impossible; say so cheaply rather
    // than burning the whole attempt budget discovering it.
    if eligible
        .iter()
        .zip(&spec.slots)
        .any(|(e, s)| e.len() < s.count)
    {
        return Ok(Vec::new());
    }

    // What each player costs in each slot group. Built once, outside the parallel
    // region: it depends only on the pool and the spec, so building it per chunk
    // would multiply the work by the chunk count for an identical answer. Not
    // built at all when no slot scales salary, which is every roster outside a
    // showdown.
    let salary_table = if spec.has_salary_multipliers() {
        spec.slot_salary_table(pool.salaries)
    } else {
        Vec::new()
    };
    let slot_salaries = if salary_table.is_empty() {
        SlotSalaries::Uniform(pool.salaries)
    } else {
        SlotSalaries::PerSlot {
            table: &salary_table,
            n_players: pool.len(),
        }
    };

    // Cheapest way to finish, per slot group.
    //
    // Without a reservation the fill spends its whole budget on whichever slots it
    // visits first and then cannot afford the last ones — on a tight cap that
    // fails every attempt rather than merely producing a worse lineup.
    //
    // The bound has to account for distinctness *within* a group: a group needing
    // three players cannot fill all three with the single cheapest one. So this
    // takes the prefix sums of each group's sorted salaries, and `cheapest[j][m]`
    // is the least a group can spend on `m` players. Using `m * min` instead
    // under-reserves, which shows up as the last slot of a multi-slot group being
    // unfillable — the fill happily takes the one affordable player and then has
    // nothing left for its twin.
    //
    // Double counting *across* groups is not corrected: a player eligible for two
    // groups is counted in both. That keeps this a lower bound, which is the safe
    // direction — it can only admit a pick that later proves infeasible, never
    // reject a feasible one.
    //
    // Built from slot-scaled salaries, not raw ones: a captain slot reserving the
    // unmultiplied cheapest player under-reserves by half its cost, and the
    // symptom is the last slot being unaffordable on every attempt.
    // Locked players are excluded from every group's prefix, and their own cost
    // is added back below. A lock is not available to fill some other slot, so
    // counting it among the cheap options overstates what is left; and, far worse,
    // a group holding an expensive lock has a minimum cost far above its cheapest
    // eligible player. Reserving the cheap figure lets the earlier slot groups
    // spend the budget the lock needed, and the lock is then unaffordable on every
    // single attempt — which presents as locking a star returning nothing at all
    // while locking a cheap player works fine.
    let cheapest: Vec<Vec<i64>> = eligible
        .iter()
        .enumerate()
        .map(|(group_idx, group)| {
            let row = slot_salaries.row(group_idx);
            let mut salaries: Vec<i64> = group
                .iter()
                .filter(|&&i| !is_locked_anywhere[i as usize])
                .map(|&i| row[i as usize])
                .collect();
            salaries.sort_unstable();
            let mut prefix = Vec::with_capacity(salaries.len() + 1);
            prefix.push(0);
            let mut running = 0;
            for salary in salaries {
                running += salary;
                prefix.push(running);
            }
            prefix
        })
        .collect();
    // What the locks in each group cost, as rostered.
    let locked_cost: Vec<i64> = locked_by_slot
        .iter()
        .enumerate()
        .map(|(group_idx, locks)| {
            let row = slot_salaries.row(group_idx);
            locks.iter().map(|&p| row[p as usize]).sum()
        })
        .collect();
    // Least a group can spend filling all of its own slots: its locks, plus the
    // cheapest unlocked players for whatever slots they leave.
    let group_floor = |j: usize| -> i64 {
        let locks = locked_by_slot.get(j).map_or(0, Vec::len);
        let free = spec.slots[j].count - locks;
        locked_cost.get(j).copied().unwrap_or(0) + cheapest[j][free.min(cheapest[j].len() - 1)]
    };
    let mut suffix_cost = vec![0i64; spec.slots.len() + 1];
    for j in (0..spec.slots.len()).rev() {
        suffix_cost[j] = suffix_cost[j + 1] + group_floor(j);
    }

    // Key values each stack constraint could actually be built around.
    //
    // A team with two eligible outfielders cannot supply a five-stack, and
    // drawing it as a target would waste the attempt. The bound is per slot
    // group — a team's contribution to a group is capped by both that group's
    // slot count and how many of its players are eligible for it — which makes
    // this exact enough to eliminate the hopeless targets without pretending to
    // know about salary.
    let counted_suffix = spec.counted_suffix();
    let n_keys = (spec.max_key() + 1).max(0) as usize;
    let mut capable_keys: Vec<Vec<i32>> = Vec::with_capacity(spec.groups.len());
    let mut secondary_capable_keys: Vec<Vec<i32>> = Vec::with_capacity(spec.groups.len());
    for group in &spec.groups {
        if group.min_stack == 0 {
            capable_keys.push(Vec::new());
            secondary_capable_keys.push(Vec::new());
            continue;
        }
        let mut reach = vec![0usize; n_keys];
        for (j, slot) in spec.slots.iter().enumerate() {
            if group.slots & (1u64 << j) == 0 {
                continue;
            }
            let mut per_key = vec![0usize; n_keys];
            for &player in &eligible[j] {
                let key = spec.key_columns[group.key_column][player as usize];
                if key >= 0 {
                    per_key[key as usize] += 1;
                }
            }
            for (key, available) in per_key.into_iter().enumerate() {
                reach[key] += available.min(slot.count);
            }
        }
        capable_keys.push(
            (0..n_keys)
                .filter(|&key| reach[key] >= group.min_stack as usize)
                .map(|key| key as i32)
                .collect(),
        );
        // The secondary bar is lower (never above the primary), so this list is
        // a superset of the one above — which is what the pair test relies on.
        secondary_capable_keys.push(if group.secondary_min_stack == 0 {
            Vec::new()
        } else {
            (0..n_keys)
                .filter(|&key| reach[key] >= group.secondary_min_stack as usize)
                .map(|key| key as i32)
                .collect()
        });
    }
    // A stack nobody can supply makes every lineup impossible. Saying so here
    // costs one pass over the pool; discovering it by attempt costs all of them.
    // For a 4-2 the requirement is a *pair* of distinct keys: one on the primary
    // list and a different one on the secondary list. The secondary list being a
    // superset makes that exactly "at least two keys reach the secondary bar".
    if spec.groups.iter().enumerate().any(|(gi, g)| {
        g.min_stack > 0
            && (capable_keys[gi].is_empty()
                || (g.secondary_min_stack > 0 && secondary_capable_keys[gi].len() < 2))
    }) {
        return Ok(Vec::new());
    }

    // Salary priced into every projection, once. The rate is the pool's own
    // points per dollar, so `value_weight` is unitless and means the same thing
    // on a $50,000 cap as on a $60,000 one.
    let total_salary: i64 = pool.salaries.iter().sum();
    let rate = if total_salary > 0 {
        pool.projections.iter().sum::<f64>() / total_salary as f64
    } else {
        0.0
    };
    let value: Vec<f64> = pool
        .projections
        .iter()
        .zip(pool.salaries)
        .map(|(&projection, &salary)| projection - config.value_weight * rate * salary as f64)
        .collect();

    let tables = Tables {
        eligible: &eligible,
        cheapest: &cheapest,
        suffix_cost: &suffix_cost,
        slot_salaries,
        counted_suffix: &counted_suffix,
        capable_keys: &capable_keys,
        secondary_capable_keys: &secondary_capable_keys,
        value: &value,
        locked_by_slot: &locked_by_slot,
    };

    let total_attempts = config
        .num_lineups
        .saturating_mul(config.attempts_per_lineup);
    let effective_chunks = effective_chunks(total_attempts, config.chunks);
    let per_chunk = (total_attempts / effective_chunks).max(1);
    // Scaling by the pool's mean projection is what makes the weight unitless:
    // 1.0 docks a player used in every lineup by one average player's worth of
    // value, on a slate scoring eight points a man or eighty.
    let fading = config.diversity_weight > 0.0;
    let fade_scale = if fading && !pool.is_empty() {
        config.diversity_weight * pool.projections.iter().sum::<f64>() / pool.len() as f64
    } else {
        0.0
    };

    let chunk_results: Vec<Vec<Lineup>> = (0..config.chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut builder = Builder::new(pool, spec, tables);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                config.seed ^ (chunk as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut seen: HashSet<Lineup> = HashSet::new();
            let mut out: Vec<Lineup> = Vec::new();
            // Usage over this chunk's own lineups. Deliberately not shared: a
            // counter visible to every chunk would have to be mutated as they
            // ran, and the answer would then depend on how rayon interleaved
            // them, which is the one thing this file guarantees it does not do.
            //
            // Keeping it local costs nothing in effect. Chunks start from
            // different seeds and so explore different regions anyway; fading
            // each against its own history is what makes consecutive lineups
            // within a chunk differ, and pairwise overlap is what that moves. A
            // shared snapshot, tried and measured, steers every chunk the same
            // way — it spreads exposure but leaves lineups as alike as before.
            let mut counts = vec![0u32; if fading { pool.len() } else { 0 }];

            for attempt in 0..per_chunk {
                let profile = config.profiles[attempt % config.profiles.len()];
                let fade = fading.then(|| Fade {
                    counts: &counts,
                    accepted: out.len(),
                    scale: fade_scale,
                });
                builder.choose_stack_targets(&mut rng);
                builder.randomize_objective(&mut rng, profile, config.noise, fade.as_ref());
                if let Some(lineup) = builder.fill() {
                    let mut key = lineup.clone();
                    key.sort_unstable();
                    if seen.insert(key) {
                        if fading {
                            for &player in &lineup {
                                counts[player as usize] += 1;
                            }
                        }
                        out.push(lineup);
                    }
                }
            }
            out
        })
        .collect();

    // Merge in chunk order, deduplicating again: two chunks can independently
    // find the same lineup, and chunk order is fixed so the merge is stable.
    //
    // Exposure caps are applied *here* rather than during construction, and that
    // placement is the whole design. A cap is a property of the portfolio, not of
    // a lineup; enforcing it inside the chunks would mean either sharing mutable
    // counts between them — which destroys the determinism guarantee, since the
    // answer would depend on how rayon interleaved them — or capping each chunk
    // separately, which is a different and weaker constraint. The merge is
    // already sequential and already in a fixed order, so it is the one place
    // that can see the whole portfolio and still reproduce exactly.
    //
    // `diversity_weight` is the other half of that trade and lives in the chunks
    // precisely because it is a preference rather than a ceiling: an approximate
    // answer computed from partial information is fine, and it steers a lineup
    // before it exists instead of discarding one that already does.
    let capping = !config.exposure_limits.is_empty();
    let mut exposure = vec![0u32; if capping { pool.len() } else { 0 }];
    let mut seen: HashSet<Lineup> = HashSet::new();
    let mut all: Vec<Lineup> = Vec::with_capacity(config.num_lineups);
    for chunk in chunk_results {
        for lineup in chunk {
            if all.len() >= config.num_lineups {
                return Ok(all);
            }
            if capping
                && lineup
                    .iter()
                    .any(|&p| exposure[p as usize] >= config.exposure_limits[p as usize])
            {
                continue;
            }
            let mut key = lineup.clone();
            key.sort_unstable();
            if seen.insert(key) {
                if capping {
                    for &player in &lineup {
                        exposure[player as usize] += 1;
                    }
                }
                all.push(lineup);
            }
        }
    }
    Ok(all)
}

/// Attempts a chunk needs before the per-chunk mechanisms mean anything.
///
/// Two things read the per-chunk attempt index, and both stop working when it is
/// too small:
///
/// * the profile cycle is `profiles[attempt % profiles.len()]`, so at one attempt
///   per chunk every chunk draws `profiles[0]` and no other. With the default
///   pair that means `CONTRARIAN` runs and `STANDARD` never does — silently
///   deleting half the mechanism whose whole job is to stop the pool collapsing
///   onto one high-ceiling core.
/// * `diversity_weight` fades against the chunk's own accepted lineups, and a
///   chunk with no history has nothing to fade against.
///
/// Four is enough for the default two-profile cycle to come round twice and for a
/// fade to have something to read.
const MIN_ATTEMPTS_PER_CHUNK: usize = 4;

/// How many chunks to actually split the work across.
///
/// `config.chunks` is an *upper bound* on parallel width, not a literal count.
/// Taken literally it silently breaks small builds: at the shipped defaults a
/// 20-lineup portfolio is 60 attempts over 64 chunks, so every chunk gets one
/// attempt, the profile cycle never advances, `diversity_weight` does nothing,
/// and 64 attempts run where 60 were asked for.
///
/// Clamping here rather than asking the caller to compute it is deliberate. The
/// arithmetic couples four parameters — `num_lineups`, `attempts_per_lineup`,
/// `chunks`, `diversity_weight` — and nothing in the signature or the types hints
/// that the first three decide whether the fourth functions.
///
/// Determinism is unaffected: this is a pure function of values the caller
/// supplied, so the output remains fixed by the inputs. What it does change is
/// that `chunks` only moves the result while it is the binding constraint —
/// above the clamp two different chunk counts give the same answer. That is a
/// weaker contract than "chunks always changes the output" and is stated as such
/// on `GreedyConfig::chunks`.
fn effective_chunks(total_attempts: usize, chunks: usize) -> usize {
    let supportable = (total_attempts / MIN_ATTEMPTS_PER_CHUNK).max(1);
    chunks.min(supportable)
}

/// How much to dock a player for already being used.
///
/// Holds a borrowed count column and the divisor, rather than a precomputed
/// per-player vector, because the counts change as lineups are accepted and a
/// vector would have to be rebuilt each time. `penalty` is one multiply.
struct Fade<'a> {
    counts: &'a [u32],
    accepted: usize,
    /// `diversity_weight * mean_projection`, folded together once. Scaling by
    /// the pool's mean projection is what makes the weight unitless: 1.0 docks
    /// a player used in every lineup by one average player's worth of value.
    scale: f64,
}

impl Fade<'_> {
    fn penalty(&self, player: usize) -> f64 {
        if self.accepted == 0 {
            return 0.0;
        }
        self.scale * (self.counts[player] as f64 / self.accepted as f64)
    }
}

/// Everything derived from the pool and the spec once, before the parallel
/// region, and then shared by every chunk.
///
/// These all depend only on inputs that do not change between attempts, so
/// building them per chunk would multiply the work by the chunk count for an
/// identical answer. Grouped into one struct because passing seven borrowed
/// tables positionally is a transposition waiting to happen.
#[derive(Clone, Copy)]
struct Tables<'a> {
    /// Players eligible for each slot group.
    eligible: &'a [Vec<u32>],
    /// `cheapest[j][m]` is the least slot group `j` can spend on `m` players.
    cheapest: &'a [Vec<i64>],
    /// `suffix_cost[j]` is the cheapest possible total for slot groups after `j`.
    suffix_cost: &'a [i64],
    /// What every player costs in every slot group.
    slot_salaries: SlotSalaries<'a>,
    /// How many counted slots each constraint has left from each slot group on.
    counted_suffix: &'a [usize],
    /// Key values each stack constraint could be built around.
    capable_keys: &'a [Vec<i32>],
    /// Key values that could supply the *secondary* stack — a superset of the
    /// primary list, since the secondary bar is never higher.
    secondary_capable_keys: &'a [Vec<i32>],
    /// Projections with salary priced in.
    value: &'a [f64],
    /// Players forced into each slot group.
    locked_by_slot: &'a [Vec<u32>],
}

/// Per-chunk scratch space. Allocated once and reused across attempts, because
/// the allocation dominates otherwise — a lineup is built in microseconds.
struct Builder<'a> {
    pool: &'a PlayerPool<'a>,
    spec: &'a RosterSpec,
    eligible: &'a [Vec<u32>],
    /// `cheapest[j][m]` is the least slot group `j` can spend on `m` players.
    cheapest: &'a [Vec<i64>],
    /// `suffix_cost[j]` is the cheapest possible total for slot groups after `j`.
    suffix_cost: &'a [i64],
    /// What every player costs in every slot group.
    slot_salaries: SlotSalaries<'a>,
    /// `[g * (n_slot_groups + 1) + j]` is how many roster slots constraint `g`
    /// still counts from slot group `j` onward.
    counted_suffix: &'a [usize],
    /// Key values each stack constraint could plausibly be built around, so a
    /// target is never drawn from a team that cannot supply the players.
    capable_keys: &'a [Vec<i32>],
    /// Key values that could supply the secondary stack, when one is asked for.
    secondary_capable_keys: &'a [Vec<i32>],
    /// The key value each stack constraint is chasing this attempt, or `-1`.
    /// Redrawn per attempt, which is what spreads a portfolio's stacks across
    /// teams instead of piling every lineup onto the same one.
    stack_targets: Vec<i32>,
    /// The *second* key value each constraint is chasing, or `-1`. Always
    /// distinct from the primary target when set.
    secondary_targets: Vec<i32>,
    /// Projections with salary priced in, which is what the candidate ordering
    /// reads. The pool's own projections stay untouched, because those are what
    /// a finished lineup is reported as being worth.
    value: &'a [f64],
    /// Players forced into each slot group, bucketed by that group.
    locked_by_slot: &'a [Vec<u32>],
    /// Whether each player is locked, by pool index. Repair consults this to
    /// leave locks alone; a lock swapped out to reach the salary floor is not a
    /// lock.
    is_locked: Vec<bool>,
    objective: Vec<f64>,
    used: Vec<bool>,
    /// How many rostered players each player conflicts with. A candidate is
    /// blocked while this is non-zero.
    ///
    /// Counted rather than flagged because repair swaps a player back out, and a
    /// flag could not tell "unblocked" from "blocked by someone else". Maintained
    /// per *pick* over the conflict adjacency, so the check a candidate pays is a
    /// single load.
    blocked: Vec<u16>,
    candidates: Vec<u32>,
    lineup: Vec<u32>,
    slot_of: Vec<usize>,
    tally: GroupTally<'a>,
}

impl<'a> Builder<'a> {
    fn new(pool: &'a PlayerPool<'a>, spec: &'a RosterSpec, tables: Tables<'a>) -> Self {
        let Tables {
            eligible,
            cheapest,
            suffix_cost,
            slot_salaries,
            counted_suffix,
            capable_keys,
            secondary_capable_keys,
            value,
            locked_by_slot,
        } = tables;
        let n = pool.len();
        let roster_size = spec.roster_size();
        // Not allocated at all without locks, which is nearly every build; this
        // runs once per chunk, so an unconditional column would be 64 pool-sized
        // allocations for nothing.
        let mut is_locked = vec![false; if locked_by_slot.is_empty() { 0 } else { n }];
        for group in locked_by_slot {
            for &player in group {
                is_locked[player as usize] = true;
            }
        }
        Self {
            pool,
            spec,
            eligible,
            cheapest,
            suffix_cost,
            slot_salaries,
            counted_suffix,
            capable_keys,
            secondary_capable_keys,
            value,
            stack_targets: vec![-1; spec.groups.len()],
            secondary_targets: vec![-1; spec.groups.len()],
            locked_by_slot,
            is_locked,
            objective: vec![0.0; n],
            used: vec![false; n],
            // Not allocated at all without conflicts, so the spec that does not
            // use them does not pay for the memory either.
            blocked: vec![0; if spec.conflicts.is_empty() { 0 } else { n }],
            candidates: Vec::with_capacity(n),
            lineup: Vec::with_capacity(roster_size),
            slot_of: Vec::with_capacity(roster_size),
            tally: GroupTally::new(spec),
        }
    }

    /// Draw a fresh perturbed objective for every player.
    ///
    /// `projection + ceiling * stddev`, discounted by `(1 - ownership)^leverage`,
    /// plus uniform noise proportional to the projection. Clamped to a small
    /// positive value so that ordering stays total and a heavily faded player is
    /// still reachable rather than being excluded outright.
    fn randomize_objective(
        &mut self,
        rng: &mut Xoshiro256PlusPlus,
        profile: JitterProfile,
        noise: f64,
        fade: Option<&Fade<'_>>,
    ) {
        let ceiling = sample_range(rng, profile.ceiling);
        let leverage = sample_range(rng, profile.leverage);

        for i in 0..self.pool.len() {
            // The salary-priced projection, not the raw one — including for the
            // noise amplitude, so a player the pricing has written off does not
            // also get the largest random kick.
            let projection = self.value[i] - fade.map_or(0.0, |f| f.penalty(i));
            let mut value = projection + ceiling * self.pool.stddevs[i];
            // Clamped below 1 so a 100%-owned player is faded, not zeroed.
            let owned = self.pool.ownership[i].clamp(0.0, 0.99);
            value *= (1.0 - owned).powf(leverage);
            if noise > 0.0 {
                let amplitude = (projection * noise).abs().max(0.1);
                value += rng.random_range(-amplitude..amplitude);
            }
            self.objective[i] = value.max(0.01);
        }
    }

    /// Draw the key value each stack constraint will chase this attempt.
    ///
    /// Uniform over the values that could actually supply the stack. Drawing per
    /// attempt is the whole mechanism: the constraint says "some team", not
    /// "which team", and a builder that always answered that question the same
    /// way would return a portfolio stacked entirely on one team — the opposite
    /// of the diversity the rest of this file exists to produce.
    ///
    /// Drawn from the same generator as the jitter, before it, so the sequence
    /// stays a function of the seed alone.
    fn choose_stack_targets(&mut self, rng: &mut Xoshiro256PlusPlus) {
        for (gi, group) in self.spec.groups.iter().enumerate() {
            if group.min_stack == 0 {
                continue;
            }
            let capable = &self.capable_keys[gi];
            self.stack_targets[gi] = if capable.is_empty() {
                -1
            } else {
                capable[rng.random_range(0..capable.len())]
            };
            if group.secondary_min_stack == 0 {
                continue;
            }
            // The secondary target is a *different* key, drawn uniformly from
            // the secondary-capable list with the primary skipped. Rejection-
            // free — one draw over `len - 1` and an index shift — so a config
            // without a secondary stack consumes exactly the RNG calls it
            // always did, and one with it consumes exactly one more per group.
            let capable = &self.secondary_capable_keys[gi];
            let primary = self.stack_targets[gi];
            self.secondary_targets[gi] = match capable.iter().position(|&k| k == primary) {
                Some(skip) if capable.len() >= 2 => {
                    let mut idx = rng.random_range(0..capable.len() - 1);
                    if idx >= skip {
                        idx += 1;
                    }
                    capable[idx]
                }
                None if !capable.is_empty() => capable[rng.random_range(0..capable.len())],
                _ => -1,
            };
        }
    }

    /// Whether taking `player` into `slot_group` leaves every distinct-minimum
    /// still reachable.
    ///
    /// The test is exact rather than a heuristic: after this pick there are a
    /// known number of counted slots left, and the requirement needs a known
    /// number of further distinct values. If the second exceeds the first, the
    /// attempt is already lost and continuing would only discover that later,
    /// having spent the rest of the roster on it.
    ///
    /// `picked` is how many of `slot_group`'s own slots are already filled.
    fn strands_a_minimum(&self, player: usize, slot_group: usize, picked: usize) -> bool {
        let width = self.spec.slots.len() + 1;
        for (gi, group) in self.spec.groups.iter().enumerate() {
            if group.min_distinct == 0 {
                continue;
            }
            let counts_here = group.slots & (1u64 << slot_group) != 0;
            let left_in_group = if counts_here {
                self.spec.slots[slot_group].count - picked - 1
            } else {
                0
            };
            let remaining = left_in_group + self.counted_suffix[gi * width + slot_group + 1];
            let gain = u32::from(self.tally.is_new_key(gi, player, slot_group));
            let still_needed = group
                .min_distinct
                .saturating_sub(self.tally.distinct_count(gi) + gain);
            if still_needed as usize > remaining {
                return true;
            }
        }
        false
    }

    /// Whether `player` is one this slot group should take first because a stack
    /// still wants them — the primary target short of its minimum, or the
    /// secondary target short of its smaller one.
    fn feeds_a_stack(&self, player: usize, slot_group: usize) -> bool {
        for (gi, group) in self.spec.groups.iter().enumerate() {
            if group.min_stack == 0 || group.slots & (1u64 << slot_group) == 0 {
                continue;
            }
            let key = self.tally.key_of(gi, player);
            let target = self.stack_targets[gi];
            if target >= 0 && key == target && self.tally.count_of(gi, target) < group.min_stack {
                return true;
            }
            let secondary = self.secondary_targets[gi];
            if secondary >= 0
                && key == secondary
                && self.tally.count_of(gi, secondary) < group.secondary_min_stack
            {
                return true;
            }
        }
        false
    }

    /// Whether every stack requirement is met by the finished lineup.
    ///
    /// Checked against the largest stacks actually present, not against the
    /// targets: the constraint asks for *some* key value (or some pair), and a
    /// lineup that happened to stack different teams than the ones drawn
    /// satisfies it just as well.
    fn stacks_satisfied(&self) -> bool {
        self.spec.groups.iter().enumerate().all(|(gi, group)| {
            if group.min_stack == 0 {
                return true;
            }
            if group.secondary_min_stack == 0 {
                return self.tally.max_stack(gi) >= group.min_stack;
            }
            let (best, second) = self.tally.top_two(gi);
            best >= group.min_stack && second >= group.secondary_min_stack
        })
    }

    /// Whether bringing `candidate` into `slot_group` still satisfies every
    /// minimum, assuming the outgoing player has already been taken out of the
    /// tally.
    ///
    /// Used by salary repair, where the lineup is otherwise complete: there are
    /// no further picks to fix a requirement this swap would break, so
    /// "reachable" is not good enough and the test is for "met".
    fn swap_keeps_minimums(&self, candidate: usize, slot_group: usize) -> bool {
        for (gi, group) in self.spec.groups.iter().enumerate() {
            if group.min_distinct == 0 && group.min_stack == 0 {
                continue;
            }
            let counts_here = group.slots & (1u64 << slot_group) != 0;
            let key = if counts_here {
                self.tally.key_of(gi, candidate)
            } else {
                -1
            };
            let gain = u32::from(counts_here && self.tally.is_new_key(gi, candidate, slot_group));
            if self.tally.distinct_count(gi) + gain < group.min_distinct {
                return false;
            }
            if group.min_stack > 0 {
                // The tally already has the outgoing player removed; ask what
                // the top counts would be with the candidate's key bumped by
                // one, without mutating anything.
                let (best, second) = self.tally.top_two_with(gi, key);
                if best < group.min_stack
                    || (group.secondary_min_stack > 0 && second < group.secondary_min_stack)
                {
                    return false;
                }
            }
        }
        true
    }

    /// Whether every distinct-minimum is met by the finished lineup.
    fn distincts_satisfied(&self) -> bool {
        self.spec.groups.iter().enumerate().all(|(gi, group)| {
            group.min_distinct == 0 || self.tally.distinct_count(gi) >= group.min_distinct
        })
    }

    /// Fill every slot greedily, then repair salary if needed.
    fn fill(&mut self) -> Option<Lineup> {
        self.used.fill(false);
        self.blocked.fill(0);
        self.lineup.clear();
        self.slot_of.clear();
        self.tally.reset();
        let mut salary: i64 = 0;
        // Hoisted so the inner loop's conflict test short-circuits on a value the
        // branch predictor sees as constant, and never touches `blocked` at all
        // for the overwhelmingly common spec that declares no conflicts.
        let has_conflicts = !self.spec.conflicts.is_empty();
        let has_locks = !self.locked_by_slot.is_empty();
        let has_minimums = self.spec.has_minimums();
        let has_stacks = self.spec.has_stacks();

        for (group_idx, slot) in self.spec.slots.iter().enumerate() {
            // One row for the whole slot group: what each player costs *here*.
            let salaries = self.slot_salaries.row(group_idx);
            let mut picked = 0;

            // Locked players go in before anything is considered, so the greedy
            // pass sees the budget and the group counts they have already spent
            // rather than discovering them afterwards. Placing them last would
            // routinely leave nothing affordable for a lock the caller insisted
            // on, which is the opposite of what locking is for.
            //
            // A lock that will not fit fails the whole attempt. Unlike a greedy
            // pick there is no alternative to fall back to, and the failure is
            // deterministic — the same lock fails on every attempt — so this
            // surfaces as an empty result rather than a slow one.
            let group_locks: &[u32] = if has_locks {
                &self.locked_by_slot[group_idx]
            } else {
                &[]
            };
            // What this group's remaining locks and free slots will cost. The
            // free-slot figure is the cheapest *unlocked* players, which is what
            // `cheapest` holds.
            let free_slots = slot.count - group_locks.len();
            let prefix = &self.cheapest[group_idx];
            let free_floor = prefix[free_slots.min(prefix.len() - 1)];
            let mut locks_to_come: i64 = group_locks.iter().map(|&p| salaries[p as usize]).sum();

            for &player in group_locks {
                let p = player as usize;
                locks_to_come -= salaries[p];
                let remaining = locks_to_come + free_floor + self.suffix_cost[group_idx + 1];
                if salary + salaries[p] + remaining > self.spec.salary_cap
                    || self.tally.would_exceed(p, group_idx)
                    || (has_conflicts && self.blocked[p] != 0)
                {
                    return None;
                }
                self.used[p] = true;
                self.tally.add(p, group_idx);
                if has_conflicts {
                    for &other in self.spec.conflicts.neighbors(p) {
                        self.blocked[other as usize] += 1;
                    }
                }
                salary += salaries[p];
                self.lineup.push(player);
                self.slot_of.push(group_idx);
                picked += 1;
            }

            self.candidates.clear();
            self.candidates.extend(
                self.eligible[group_idx]
                    .iter()
                    .copied()
                    .filter(|&i| !self.used[i as usize]),
            );
            // Descending objective, ties broken by index so the sort is total and
            // the result does not depend on sort stability.
            let objective = &self.objective;
            self.candidates.sort_unstable_by(|&a, &b| {
                objective[b as usize]
                    .partial_cmp(&objective[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });

            // Two passes when a stack is still owed players this group could
            // supply: the first takes only players that feed it, the second
            // everything else. Filling the stack from the *best* available
            // stack players rather than whatever is left at the end is the
            // difference between a stack worth having and a legal accident.
            //
            // Both passes walk the same objective-sorted list, so within each
            // the ordering is untouched — this reorders which players are
            // considered, never how they are ranked.
            let passes = if has_stacks { 2 } else { 1 };
            for pass in 0..passes {
                // Indexed rather than iterated: the loop body mutates `used`,
                // which the priority pass then has to observe.
                for ci in 0..self.candidates.len() {
                    if picked == slot.count {
                        break;
                    }
                    let player = self.candidates[ci];
                    let p = player as usize;
                    if has_stacks {
                        if self.used[p] {
                            continue;
                        }
                        if pass == 0 && !self.feeds_a_stack(p, group_idx) {
                            continue;
                        }
                    }
                    // Reserve enough for the slots still to be filled, including
                    // the rest of this group.
                    let still_needed = slot.count - picked - 1;
                    let prefix = &self.cheapest[group_idx];
                    let remaining = prefix[still_needed.min(prefix.len() - 1)]
                        + self.suffix_cost[group_idx + 1];
                    if salary + salaries[p] + remaining > self.spec.salary_cap {
                        continue;
                    }
                    if self.tally.would_exceed(p, group_idx) {
                        continue;
                    }
                    if has_conflicts && self.blocked[p] != 0 {
                        continue;
                    }
                    if has_minimums && self.strands_a_minimum(p, group_idx, picked) {
                        continue;
                    }
                    self.used[p] = true;
                    self.tally.add(p, group_idx);
                    if has_conflicts {
                        for &other in self.spec.conflicts.neighbors(p) {
                            self.blocked[other as usize] += 1;
                        }
                    }
                    salary += salaries[p];
                    self.lineup.push(player);
                    self.slot_of.push(group_idx);
                    picked += 1;
                }
            }
            if picked < slot.count {
                return None;
            }
        }

        // Distinct minimums are guaranteed by construction — `strands_a_minimum`
        // refuses any pick that would make one unreachable, and at the last
        // counted slot "reachable" and "met" are the same thing. A stack is not:
        // the drawn team may simply not have had enough affordable players in
        // the slots that were left, and that only becomes known here.
        if has_stacks && !self.stacks_satisfied() {
            return None;
        }

        if salary < self.spec.salary_floor && !self.repair_up(&mut salary) {
            return None;
        }
        // The fill loop never exceeds the cap, so only the floor can be violated
        // here; repair_up respects the cap, which this re-check asserts.
        debug_assert!(salary <= self.spec.salary_cap);
        if salary < self.spec.salary_floor {
            return None;
        }
        // Repair swaps a player, which can drop the last of a key value or
        // shrink a stack. It refuses swaps that do, and this re-check is what
        // makes that a guarantee rather than an intention.
        debug_assert!(!has_minimums || (self.distincts_satisfied() && self.stacks_satisfied()));
        Some(self.lineup.clone())
    }

    /// Swap a cheap player for a more expensive eligible one to clear the floor.
    ///
    /// Slots are tried cheapest-first, because replacing the cheapest player has
    /// the most headroom under the cap. One successful swap ends the repair: the
    /// swap must close the whole gap by itself, which is a weaker repair than
    /// searching combinations but keeps this O(slots x pool) rather than
    /// exponential. An attempt that cannot be repaired is discarded, and with
    /// thousands of attempts that is cheaper than repairing hard cases.
    fn repair_up(&mut self, salary: &mut i64) -> bool {
        let needed = self.spec.salary_floor - *salary;
        if needed <= 0 {
            return true;
        }

        let has_conflicts = !self.spec.conflicts.is_empty();
        let has_minimums = self.spec.has_minimums();

        // Cost is per-slot now, so the cheapest player is the cheapest *as
        // rostered*: a captain at 1.5x may be dearer than a flex player on a
        // larger base salary, and swapping the wrong one out has less headroom.
        let mut order: Vec<usize> = (0..self.lineup.len()).collect();
        let slot_salaries = self.slot_salaries;
        let lineup = &self.lineup;
        let slot_of = &self.slot_of;
        let cost = |i: usize| slot_salaries.row(slot_of[i])[lineup[i] as usize];
        order.sort_unstable_by(|&a, &b| cost(a).cmp(&cost(b)).then(a.cmp(&b)));

        for slot_index in order {
            let outgoing = self.lineup[slot_index] as usize;
            if !self.is_locked.is_empty() && self.is_locked[outgoing] {
                continue;
            }
            let group_idx = self.slot_of[slot_index];
            let salaries = self.slot_salaries.row(group_idx);
            let outgoing_salary = salaries[outgoing];

            // Take the outgoing player out of the tally first, so a replacement
            // from the same team is not rejected by the slot it is about to free.
            // The conflict marks it laid down come off for the same reason: a
            // candidate the outgoing player was blocking is a legitimate
            // replacement for them.
            self.tally.remove(outgoing, group_idx);
            self.used[outgoing] = false;
            if has_conflicts {
                for &other in self.spec.conflicts.neighbors(outgoing) {
                    self.blocked[other as usize] -= 1;
                }
            }

            let mut swapped = None;
            for &candidate in &self.eligible[group_idx] {
                let c = candidate as usize;
                if self.used[c] {
                    continue;
                }
                let gain = salaries[c] - outgoing_salary;
                if gain < needed {
                    continue;
                }
                let new_total = *salary + gain;
                if new_total > self.spec.salary_cap {
                    continue;
                }
                if self.tally.would_exceed(c, group_idx) {
                    continue;
                }
                if has_conflicts && self.blocked[c] != 0 {
                    continue;
                }
                // The outgoing player is already out of the tally here, so this
                // sees exactly the lineup the swap would produce. A swap that
                // drops the last player of a key value, or shrinks the only
                // stack, is refused rather than discovered by the re-check.
                if has_minimums && !self.swap_keeps_minimums(c, group_idx) {
                    continue;
                }
                swapped = Some((candidate, new_total));
                break;
            }

            match swapped {
                Some((candidate, new_total)) => {
                    let c = candidate as usize;
                    self.used[c] = true;
                    self.tally.add(c, group_idx);
                    if has_conflicts {
                        for &other in self.spec.conflicts.neighbors(c) {
                            self.blocked[other as usize] += 1;
                        }
                    }
                    self.lineup[slot_index] = candidate;
                    *salary = new_total;
                    return true;
                }
                None => {
                    // Restore and try the next slot.
                    self.used[outgoing] = true;
                    self.tally.add(outgoing, group_idx);
                    if has_conflicts {
                        for &other in self.spec.conflicts.neighbors(outgoing) {
                            self.blocked[other as usize] += 1;
                        }
                    }
                }
            }
        }
        false
    }
}

/// Uniform draw from an inclusive-ish range, tolerating a degenerate range.
fn sample_range(rng: &mut Xoshiro256PlusPlus, (low, high): (f64, f64)) -> f64 {
    if high <= low {
        low
    } else {
        rng.random_range(low..high)
    }
}

/// Walk a lineup alongside the slot group each of its entries occupies.
///
/// A lineup is stored in slot order with the group boundaries implicit, so
/// anything that needs per-slot multipliers has to rebuild that pairing. Doing it
/// in one place keeps the two scoring functions below from disagreeing about it.
fn with_slot_groups<'a>(
    spec: &'a RosterSpec,
    lineup: &'a [u32],
) -> impl Iterator<Item = (usize, &'a SlotGroup)> + 'a {
    spec.slots
        .iter()
        .flat_map(|slot| std::iter::repeat_n(slot, slot.count))
        .zip(lineup)
        .map(|(slot, &player)| (player as usize, slot))
}

/// Total salary of a lineup, with each slot's multiplier applied.
///
/// Exposed because callers routinely want it and recomputing it in Python defeats
/// the point of building here.
pub fn lineup_salary(pool: &PlayerPool<'_>, spec: &RosterSpec, lineup: &[u32]) -> i64 {
    with_slot_groups(spec, lineup)
        .map(|(player, slot)| {
            crate::roster::scaled_salary(pool.salaries[player], slot.salary_multiplier)
        })
        .sum()
}

/// Total projection of a lineup, with each slot's multiplier applied.
///
/// This is where a score multiplier finally shows up. It cannot change which
/// player construction puts in a captain slot — every candidate for that slot is
/// scaled alike — but it very much changes what the finished lineup is worth, and
/// a showdown lineup scored without it is wrong by half a player.
pub fn lineup_projection(pool: &PlayerPool<'_>, spec: &RosterSpec, lineup: &[u32]) -> f64 {
    with_slot_groups(spec, lineup)
        .map(|(player, slot)| pool.projections[player] * slot.score_multiplier)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roster::{ConflictGraph, GroupConstraint, SlotGroup};

    const P: PositionMask = 1 << 0;
    const C: PositionMask = 1 << 1;
    const OF: PositionMask = 1 << 2;

    /// A deliberately tiny sport: 1 catcher, 2 outfielders, 1 pitcher.
    fn tiny_spec(team_ids: Vec<i32>, cap: i64, floor: i64) -> RosterSpec {
        RosterSpec {
            slots: vec![
                SlotGroup::new(C, 1),
                SlotGroup::new(OF, 2),
                SlotGroup::new(P, 1),
            ],
            salary_cap: cap,
            salary_floor: floor,
            groups: vec![GroupConstraint::at_most(0, 3, 0b111)],
            key_columns: vec![team_ids],
            conflicts: ConflictGraph::none(),
        }
    }

    struct Pool {
        projections: Vec<f64>,
        stddevs: Vec<f64>,
        salaries: Vec<i64>,
        ownership: Vec<f64>,
        positions: Vec<PositionMask>,
    }

    impl Pool {
        fn view(&self) -> PlayerPool<'_> {
            PlayerPool {
                projections: &self.projections,
                stddevs: &self.stddevs,
                salaries: &self.salaries,
                ownership: &self.ownership,
                positions: &self.positions,
            }
        }
    }

    /// Four catchers, eight outfielders, four pitchers, alternating teams.
    fn make_pool() -> Pool {
        let mut positions = Vec::new();
        let mut salaries = Vec::new();
        let mut projections = Vec::new();
        for (position, count) in [(C, 4), (OF, 8), (P, 4)] {
            for k in 0..count {
                positions.push(position);
                salaries.push(3000 + (k as i64) * 700);
                projections.push(5.0 + k as f64);
            }
        }
        let n = positions.len();
        Pool {
            stddevs: vec![3.0; n],
            ownership: (0..n).map(|i| (i % 10) as f64 / 20.0).collect(),
            projections,
            salaries,
            positions,
        }
    }

    fn team_ids(n: usize) -> Vec<i32> {
        (0..n as i32).map(|i| i % 4).collect()
    }

    fn config(num_lineups: usize) -> GreedyConfig {
        GreedyConfig {
            num_lineups,
            seed: 7,
            chunks: 8,
            ..GreedyConfig::default()
        }
    }

    /// Every returned lineup must actually be legal. This is the invariant the
    /// whole package rests on — a fast generator of invalid lineups is worthless.
    fn assert_all_valid(lineups: &[Lineup], pool: &PlayerPool<'_>, spec: &RosterSpec) {
        for lineup in lineups {
            assert_eq!(lineup.len(), spec.roster_size(), "wrong roster size");

            let mut sorted = lineup.clone();
            sorted.sort_unstable();
            let before = sorted.len();
            sorted.dedup();
            assert_eq!(before, sorted.len(), "a player appears twice: {lineup:?}");

            let salary = lineup_salary(pool, spec, lineup);
            assert!(salary <= spec.salary_cap, "over cap: {salary}");
            assert!(salary >= spec.salary_floor, "under floor: {salary}");

            // No two rostered players may conflict. Checked from the graph rather
            // than from the builder's `blocked` counters, so a bug in the counter
            // maintenance cannot certify itself.
            for &a in lineup.iter() {
                for &b in lineup.iter() {
                    assert!(
                        !spec.conflicts.neighbors(a as usize).contains(&b),
                        "conflicting players {a} and {b} share a lineup"
                    );
                }
            }

            // Positional eligibility, slot by slot.
            let mut cursor = 0;
            for slot in &spec.slots {
                for _ in 0..slot.count {
                    let player = lineup[cursor] as usize;
                    assert!(
                        pool.positions[player] & slot.eligible != 0,
                        "player {player} is not eligible for the slot it fills"
                    );
                    cursor += 1;
                }
            }

            // Group caps and minimums.
            for group in &spec.groups {
                let mut counts = std::collections::HashMap::new();
                let mut cursor = 0;
                for (group_idx, slot) in spec.slots.iter().enumerate() {
                    for _ in 0..slot.count {
                        if group.slots & (1u64 << group_idx) != 0 {
                            let key = spec.key_columns[group.key_column][lineup[cursor] as usize];
                            if key >= 0 {
                                *counts.entry(key).or_insert(0u32) += 1;
                            }
                        }
                        cursor += 1;
                    }
                }
                for (key, count) in &counts {
                    assert!(
                        *count <= group.max_count,
                        "group cap breached for key {key}"
                    );
                }
                assert!(
                    counts.len() as u32 >= group.min_distinct,
                    "only {} distinct key(s), needed {}: {lineup:?}",
                    counts.len(),
                    group.min_distinct
                );
                assert!(
                    group.min_stack == 0
                        || counts.values().copied().max().unwrap_or(0) >= group.min_stack,
                    "biggest stack is {:?}, needed {}: {lineup:?}",
                    counts.values().copied().max(),
                    group.min_stack
                );
                if group.secondary_min_stack > 0 {
                    let mut sizes: Vec<u32> = counts.values().copied().collect();
                    sizes.sort_unstable_by(|a, b| b.cmp(a));
                    assert!(
                        sizes.get(1).copied().unwrap_or(0) >= group.secondary_min_stack,
                        "second-biggest stack is {:?}, needed {}: {lineup:?}",
                        sizes.get(1),
                        group.secondary_min_stack
                    );
                }
            }
        }
    }

    #[test]
    fn produces_valid_lineups() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn lineups_are_distinct() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        let mut keys: Vec<Lineup> = lineups
            .iter()
            .map(|l| {
                let mut k = l.clone();
                k.sort_unstable();
                k
            })
            .collect();
        let before = keys.len();
        keys.sort();
        keys.dedup();
        assert_eq!(before, keys.len(), "duplicate lineups returned");
    }

    /// The determinism guarantee: same seed, same output, regardless of how rayon
    /// happens to schedule the chunks.
    #[test]
    fn identical_seeds_give_identical_output() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let a = build_lineups(&pool.view(), &spec, &config(40)).unwrap();
        let b = build_lineups(&pool.view(), &spec, &config(40)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_give_different_output() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let a = build_lineups(&pool.view(), &spec, &config(40)).unwrap();
        let mut other = config(40);
        other.seed = 99;
        let b = build_lineups(&pool.view(), &spec, &other).unwrap();
        assert_ne!(a, b, "the seed is not reaching the generator");
    }

    #[test]
    fn a_chunk_always_gets_enough_attempts_to_cycle_and_fade() {
        // The arithmetic behind `effective_chunks`. Taken literally, the shipped
        // default of 64 chunks gives a 20-lineup build one attempt each, which
        // silently deletes the second jitter profile and `diversity_weight`.
        for (total, chunks, expected) in [
            (3, 64, 1),   // one lineup: a single chunk, three attempts
            (60, 64, 15), // twenty lineups at the default attempt budget
            (192, 64, 48),
            (450, 64, 64), // large builds keep the full width
            (10_000, 64, 64),
        ] {
            let effective = effective_chunks(total, chunks);
            assert_eq!(effective, expected, "total={total} chunks={chunks}");
            assert!(
                total / effective.max(1) >= MIN_ATTEMPTS_PER_CHUNK
                    || total < MIN_ATTEMPTS_PER_CHUNK,
                "total={total} left {} attempts per chunk",
                total / effective.max(1)
            );
        }
    }

    #[test]
    fn the_attempt_budget_is_never_exceeded() {
        // `.max(1)` on a literal chunk count used to run more attempts than were
        // asked for: one lineup at three attempts ran sixty-four.
        for num_lineups in [1usize, 5, 10, 20, 43, 64, 150, 1000] {
            let attempts = 3;
            let total = num_lineups * attempts;
            let effective = effective_chunks(total, 64);
            let per_chunk = (total / effective).max(1);
            assert!(
                effective * per_chunk <= total,
                "num_lineups={num_lineups}: ran {} of {total} requested",
                effective * per_chunk
            );
        }
    }

    #[test]
    fn both_profiles_are_drawn_on_a_small_build() {
        // The regression that motivated the clamp. With one attempt per chunk the
        // cycle never advances, so the whole build ran on `profiles[0]` and the
        // second profile — the one that stops the pool collapsing onto a single
        // high-ceiling core — never executed.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(20);
        cfg.chunks = 64;

        let mixed = build_lineups(&pool.view(), &spec, &cfg).unwrap();

        let mut first_only = cfg.clone();
        first_only.profiles = vec![cfg.profiles[0]];
        let collapsed = build_lineups(&pool.view(), &spec, &first_only).unwrap();

        assert_ne!(
            mixed, collapsed,
            "a 20-lineup build drew only the first profile, so the pair is decorative"
        );
    }

    #[test]
    fn diversity_weight_works_on_a_small_build() {
        // Same root cause: a chunk with no accepted lineups has nothing to fade
        // against, so the weight was silently inert below ~64 lineups.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(20);
        cfg.chunks = 64;

        let plain = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        let mut faded = cfg.clone();
        faded.diversity_weight = 1.0;
        let spread = build_lineups(&pool.view(), &spec, &faded).unwrap();

        assert_ne!(plain, spread, "diversity_weight was inert at this size");
    }

    /// Chunk count is part of the reproducibility contract, so it must change the
    /// Above the clamp, `chunks` stops mattering — and that is worth a test
    /// rather than a footnote, because it is a real weakening of the contract
    /// below. Both values here ask for more chunks than the attempt budget can
    /// support, so both resolve to the same width and the same lineups.
    #[test]
    fn chunk_counts_above_the_clamp_agree() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        // 20 lineups x 3 attempts = 60, so the clamp bites at 15 chunks.
        let mut a_cfg = config(20);
        a_cfg.chunks = 64;
        let mut b_cfg = config(20);
        b_cfg.chunks = 256;
        assert_eq!(effective_chunks(60, 64), effective_chunks(60, 256));
        assert_eq!(
            build_lineups(&pool.view(), &spec, &a_cfg).unwrap(),
            build_lineups(&pool.view(), &spec, &b_cfg).unwrap()
        );
    }

    /// Chunk count is part of the reproducibility contract *while it binds*, so
    /// it must change the result — otherwise the contract would be silently
    /// weaker than documented. Both values here are below the clamp.
    #[test]
    fn chunk_count_is_part_of_the_contract() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut a_cfg = config(40);
        a_cfg.chunks = 4;
        let mut b_cfg = config(40);
        b_cfg.chunks = 16;
        assert_ne!(
            build_lineups(&pool.view(), &spec, &a_cfg).unwrap(),
            build_lineups(&pool.view(), &spec, &b_cfg).unwrap()
        );
    }

    #[test]
    fn salary_floor_is_respected_and_requires_repair() {
        let pool = make_pool();
        // A floor high enough that the greedy fill will frequently land under it.
        let spec = tiny_spec(team_ids(16), 22_000, 20_000);
        let lineups = build_lineups(&pool.view(), &spec, &config(30)).unwrap();
        assert!(!lineups.is_empty(), "repair should recover some lineups");
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn an_impossible_cap_returns_nothing_rather_than_an_error() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 1, 0);
        assert!(build_lineups(&pool.view(), &spec, &config(10))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_pool_smaller_than_the_roster_returns_nothing() {
        let pool = Pool {
            projections: vec![1.0],
            stddevs: vec![1.0],
            salaries: vec![1],
            ownership: vec![0.0],
            positions: vec![C],
        };
        let spec = tiny_spec(vec![0], 10_000, 0);
        assert!(build_lineups(&pool.view(), &spec, &config(10))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_slot_no_player_can_fill_returns_nothing() {
        let mut pool = make_pool();
        // Remove every pitcher by making them catchers.
        for position in pool.positions.iter_mut() {
            if *position == P {
                *position = C;
            }
        }
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        assert!(build_lineups(&pool.view(), &spec, &config(10))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn group_caps_bind() {
        let pool = make_pool();
        // Everyone on one team, cap of 3 — impossible for a 4-player roster.
        let spec = tiny_spec(vec![0; 16], 30_000, 0);
        assert!(build_lineups(&pool.view(), &spec, &config(10))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn requesting_zero_lineups_returns_nothing() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        assert!(build_lineups(&pool.view(), &spec, &config(0))
            .unwrap()
            .is_empty());
    }

    // --- Slot multipliers -------------------------------------------------

    /// A showdown-shaped roster: one multiplied slot plus a run of plain ones.
    fn showdown_spec(cap: i64, score: f64, salary: f64) -> RosterSpec {
        RosterSpec {
            slots: vec![
                SlotGroup::multiplied(C | OF | P, 1, score, salary),
                SlotGroup::new(C | OF | P, 3),
            ],
            salary_cap: cap,
            salary_floor: 0,
            groups: Vec::new(),
            key_columns: vec![vec![0; 16]],
            conflicts: ConflictGraph::none(),
        }
    }

    #[test]
    fn a_captain_slot_charges_its_multiplied_salary() {
        let pool = make_pool();
        let spec = showdown_spec(30_000, 1.5, 1.5);
        let lineups = build_lineups(&pool.view(), &spec, &config(30)).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);

        // The validator above uses lineup_salary, which is the same code the cap
        // check uses. Recompute it independently so this is a real check.
        for lineup in &lineups {
            let captain = lineup[0] as usize;
            let expected: i64 = (pool.salaries[captain] as f64 * 1.5).round() as i64
                + lineup[1..]
                    .iter()
                    .map(|&i| pool.salaries[i as usize])
                    .sum::<i64>();
            assert_eq!(lineup_salary(&pool.view(), &spec, lineup), expected);
            assert!(expected <= 30_000);
        }
    }

    #[test]
    fn a_multiplied_salary_actually_binds_the_cap() {
        let pool = make_pool();
        // A cap that the same roster clears comfortably at 1.0x and cannot at
        // 1.5x, so this fails if the multiplier is dropped anywhere in the fill.
        let plain =
            build_lineups(&pool.view(), &showdown_spec(13_000, 1.0, 1.0), &config(30)).unwrap();
        assert!(!plain.is_empty(), "the plain cap should be satisfiable");

        let spec = showdown_spec(13_000, 1.5, 1.5);
        let multiplied = build_lineups(&pool.view(), &spec, &config(30)).unwrap();
        assert_all_valid(&multiplied, &pool.view(), &spec);
        let cheapest_plain = plain
            .iter()
            .map(|l| lineup_salary(&pool.view(), &showdown_spec(13_000, 1.0, 1.0), l))
            .min()
            .unwrap();
        assert!(
            multiplied.len() < plain.len() || cheapest_plain > 13_000,
            "a 1.5x captain must make the cap harder, not easier"
        );
    }

    #[test]
    fn a_score_multiplier_does_not_change_which_players_are_picked() {
        // Documented behaviour, and worth pinning: every candidate for a slot is
        // scaled by the same factor, so the ordering cannot move. If this ever
        // fails, the multiplier has leaked into the sort and determinism claims
        // about seeds no longer hold across spec edits.
        let pool = make_pool();
        let plain = build_lineups(&pool.view(), &showdown_spec(30_000, 1.0, 1.0), &config(30));
        let scaled = build_lineups(&pool.view(), &showdown_spec(30_000, 1.5, 1.0), &config(30));
        assert_eq!(plain.unwrap(), scaled.unwrap());
    }

    #[test]
    fn a_score_multiplier_changes_what_the_lineup_is_worth() {
        let pool = make_pool();
        let spec = showdown_spec(30_000, 1.5, 1.0);
        let lineups = build_lineups(&pool.view(), &spec, &config(5)).unwrap();
        let lineup = &lineups[0];
        let captain = lineup[0] as usize;
        let expected = pool.projections[captain] * 1.5
            + lineup[1..]
                .iter()
                .map(|&i| pool.projections[i as usize])
                .sum::<f64>();
        assert!((lineup_projection(&pool.view(), &spec, lineup) - expected).abs() < 1e-9);
    }

    #[test]
    fn a_multiplied_slot_reserves_its_multiplied_cost() {
        // The reservation is what stops the fill spending everything before it
        // reaches the last slot. Put the captain *last* so the earlier slots have
        // to leave room for 1.5x, which raw-salary prefix sums would not do.
        let pool = make_pool();
        let spec = RosterSpec {
            slots: vec![
                SlotGroup::new(C | OF | P, 3),
                SlotGroup::multiplied(C | OF | P, 1, 1.5, 1.5),
            ],
            salary_cap: 16_000,
            salary_floor: 0,
            groups: Vec::new(),
            key_columns: vec![vec![0; 16]],
            conflicts: ConflictGraph::none(),
        };
        let lineups = build_lineups(&pool.view(), &spec, &config(30)).unwrap();
        assert!(!lineups.is_empty(), "the reservation is over-tight");
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn repair_prices_the_swap_in_the_slot_it_happens_in() {
        // A floor forces repair, and the captain slot makes the cheapest player
        // as-rostered differ from the cheapest by base salary.
        let pool = make_pool();
        let spec = RosterSpec {
            slots: vec![
                SlotGroup::multiplied(C | OF | P, 1, 1.5, 1.5),
                SlotGroup::new(C | OF | P, 3),
            ],
            salary_cap: 25_000,
            salary_floor: 23_000,
            groups: Vec::new(),
            key_columns: vec![vec![0; 16]],
            conflicts: ConflictGraph::none(),
        };
        let lineups = build_lineups(&pool.view(), &spec, &config(30)).unwrap();
        assert!(!lineups.is_empty(), "repair found nothing to fix");
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    // --- Conflicts --------------------------------------------------------

    /// Each catcher conflicts with exactly one pitcher, the way a hitter opposes
    /// exactly one starter. A complete bipartite graph would be unsatisfiable
    /// rather than constraining, which tests nothing about enforcement.
    fn opposing_pairs() -> Vec<(u32, u32)> {
        (0..4u32).map(|c| (c, 12 + c)).collect()
    }

    #[test]
    fn conflicting_players_never_share_a_lineup() {
        let pool = make_pool();
        let unconstrained = tiny_spec(team_ids(16), 30_000, 0);
        let mut spec = tiny_spec(team_ids(16), 30_000, 0);
        spec.conflicts = ConflictGraph::from_pairs(opposing_pairs(), 16);

        // The rule has to be doing work, or this passes for the wrong reason.
        let before = build_lineups(&pool.view(), &unconstrained, &config(50)).unwrap();
        assert!(
            before.iter().any(|l| opposing_pairs()
                .iter()
                .any(|&(a, b)| l.contains(&a) && l.contains(&b))),
            "the unconstrained build never produced a conflicting pair, so \
             forbidding them proves nothing"
        );

        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn conflicts_are_enforced_whichever_side_is_picked_first() {
        // The pair is given one way round only. The catcher slot fills before the
        // pitcher slot, so a directed relation would let the pitcher through.
        let pool = make_pool();
        let mut spec = tiny_spec(team_ids(16), 30_000, 0);
        spec.conflicts = ConflictGraph::from_pairs([(12u32, 0u32)], 16);
        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        assert!(!lineups.is_empty());
        for lineup in &lineups {
            assert!(
                !(lineup.contains(&0) && lineup.contains(&12)),
                "conflict ignored in one direction: {lineup:?}"
            );
        }
    }

    #[test]
    fn an_unsatisfiable_conflict_graph_returns_nothing() {
        // Every pitcher conflicts with every catcher and the roster needs one of
        // each, so no lineup exists. This must come back empty rather than loop.
        let pool = make_pool();
        let mut spec = tiny_spec(team_ids(16), 30_000, 0);
        let pairs: Vec<(u32, u32)> = (0..4u32)
            .flat_map(|c| (12..16u32).map(move |p| (c, p)))
            .chain((0..4u32).flat_map(|c| (4..12u32).map(move |o| (c, o))))
            .collect();
        spec.conflicts = ConflictGraph::from_pairs(pairs, 16);
        assert!(build_lineups(&pool.view(), &spec, &config(10))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn conflicts_survive_salary_repair() {
        // Repair swaps players in and out after the fill, with its own blocked
        // bookkeeping. A floor tight enough to force repair is the only way to
        // reach that code, and a missed decrement there produces lineups the fill
        // loop would never have built.
        let pool = make_pool();
        let mut spec = tiny_spec(team_ids(16), 22_000, 20_000);
        spec.conflicts = ConflictGraph::from_pairs(opposing_pairs(), 16);
        let lineups = build_lineups(&pool.view(), &spec, &config(40)).unwrap();
        assert!(!lineups.is_empty(), "repair recovered nothing to check");
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn declaring_no_conflicts_changes_nothing() {
        // The opt-in has to be genuinely inert: a spec that names no conflicts
        // must produce byte-identical output to one built before conflicts
        // existed, or every committed benchmark and seed became a lie.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut empty = tiny_spec(team_ids(16), 30_000, 0);
        empty.conflicts = ConflictGraph::from_pairs([], 16);
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(40)).unwrap(),
            build_lineups(&pool.view(), &empty, &config(40)).unwrap()
        );
    }

    #[test]
    fn a_conflict_graph_for_the_wrong_pool_is_rejected() {
        let pool = make_pool();
        let mut spec = tiny_spec(team_ids(16), 30_000, 0);
        spec.conflicts = ConflictGraph::from_pairs([(0, 1)], 8);
        assert!(matches!(
            build_lineups(&pool.view(), &spec, &config(10)),
            Err(BuildError::Spec(
                SpecError::ConflictGraphLengthMismatch { .. }
            ))
        ));
    }

    // --- Pricing salary into value ----------------------------------------

    #[test]
    fn pricing_salary_prefers_the_cheaper_of_two_equal_players() {
        // Two catchers projected the same, one costing far more. Ranked by
        // projection alone they tie and index decides; priced against the pool's
        // rate the cheaper one wins outright, which is the entire correction.
        let mut pool = make_pool();
        pool.projections[0] = 8.0;
        pool.projections[1] = 8.0;
        pool.salaries[0] = 3_000;
        pool.salaries[1] = 6_000;
        let spec = tiny_spec(team_ids(16), 30_000, 0);

        let mut priced = config(1);
        priced.noise = 0.0;
        priced.value_weight = 1.0;
        let lineups = build_lineups(&pool.view(), &spec, &priced).unwrap();
        assert!(
            lineups.iter().all(|l| l[0] == 0),
            "the dearer of two equal catchers was preferred"
        );
    }

    #[test]
    fn a_zero_weight_ranks_by_projection_alone() {
        // The old behaviour, still reachable. A caller who wants raw projection
        // ordering — or who is reproducing a result from before this existed —
        // gets exactly it.
        let mut pool = make_pool();
        pool.projections[0] = 8.0;
        pool.projections[1] = 8.5;
        pool.salaries[0] = 3_000;
        pool.salaries[1] = 9_000;
        let spec = tiny_spec(team_ids(16), 30_000, 0);

        let mut plain = config(1);
        plain.noise = 0.0;
        plain.value_weight = 0.0;
        let lineups = build_lineups(&pool.view(), &spec, &plain).unwrap();
        assert!(
            lineups.iter().all(|l| l[0] == 1),
            "with salary unpriced the higher projection should win regardless of cost"
        );
    }

    #[test]
    fn pricing_does_not_change_what_a_lineup_is_reported_as_worth() {
        // The adjustment is a ranking device. A lineup's value is still the sum
        // of its players' real projections, or every number downstream — the
        // benchmark ratios, the simulated scores — would be quietly deflated.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(5);
        cfg.value_weight = 1.0;
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        let lineup = &lineups[0];
        let expected: f64 = lineup.iter().map(|&i| pool.projections[i as usize]).sum();
        assert!((lineup_projection(&pool.view(), &spec, lineup) - expected).abs() < 1e-9);
    }

    #[test]
    fn pricing_salary_is_scale_free() {
        // The weight multiplies the pool's own points-per-dollar rate, so the
        // same number means the same thing whatever the currency. Doubling every
        // salary must not change the ordering.
        let pool = make_pool();
        let mut doubled = make_pool();
        for salary in doubled.salaries.iter_mut() {
            *salary *= 2;
        }
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let doubled_spec = tiny_spec(team_ids(16), 60_000, 0);
        let mut cfg = config(20);
        cfg.value_weight = 1.0;

        assert_eq!(
            build_lineups(&pool.view(), &spec, &cfg).unwrap(),
            build_lineups(&doubled.view(), &doubled_spec, &cfg).unwrap()
        );
    }

    // --- Minimums ---------------------------------------------------------

    /// `tiny_spec` with its cap replaced by the given minimum constraint.
    fn min_spec(team_ids: Vec<i32>, group: GroupConstraint) -> RosterSpec {
        let mut spec = tiny_spec(team_ids, 30_000, 0);
        spec.groups = vec![group];
        spec
    }

    #[test]
    fn a_distinct_minimum_is_met_by_every_lineup() {
        let pool = make_pool();
        // Four teams cycling through the pool; require three of them.
        let spec = min_spec(team_ids(16), GroupConstraint::distinct(0, 3, 0b111));
        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn a_distinct_minimum_actually_binds() {
        // Without the requirement the builder happily returns lineups drawn from
        // fewer teams; with it, none survive. If this passed either way the test
        // above would be proving nothing.
        let pool = make_pool();
        let loose = min_spec(team_ids(16), GroupConstraint::at_most(0, 4, 0b111));
        let unconstrained = build_lineups(&pool.view(), &loose, &config(50)).unwrap();
        let distinct_of = |lineup: &Lineup| {
            lineup
                .iter()
                .map(|&p| team_ids(16)[p as usize])
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        assert!(
            unconstrained.iter().any(|l| distinct_of(l) < 4),
            "every unconstrained lineup already used four teams"
        );

        let spec = min_spec(team_ids(16), GroupConstraint::distinct(0, 4, 0b111));
        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        assert!(!lineups.is_empty());
        assert!(lineups.iter().all(|l| distinct_of(l) == 4));
    }

    #[test]
    fn a_distinct_minimum_counts_only_the_slots_it_names() {
        // Slot group 2 is the pitcher. Requiring three distinct teams among the
        // hitters only must not be satisfiable by the pitcher's team.
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::distinct(0, 3, 0b011));
        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        assert!(!lineups.is_empty());
        for lineup in &lineups {
            let hitters: std::collections::HashSet<i32> = lineup[..3]
                .iter()
                .map(|&p| team_ids(16)[p as usize])
                .collect();
            assert!(hitters.len() >= 3, "{lineup:?}");
        }
    }

    #[test]
    fn an_unreachable_distinct_minimum_is_rejected_at_validation() {
        // Three counted slots cannot show four distinct values, and saying so
        // beats spending the whole attempt budget failing.
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::distinct(0, 4, 0b011));
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(10)),
            Err(BuildError::Spec(SpecError::MinimumExceedsSlots {
                group: 0,
                which: "min_distinct",
                wanted: 4,
                slots: 3
            }))
        );
    }

    #[test]
    fn a_distinct_minimum_the_pool_cannot_meet_returns_nothing() {
        // Every player on one team: reachable in principle, impossible here.
        let pool = make_pool();
        let spec = min_spec(vec![0; 16], GroupConstraint::distinct(0, 2, 0b111));
        assert!(build_lineups(&pool.view(), &spec, &config(10))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_distinct_minimum_survives_salary_repair() {
        let pool = make_pool();
        let mut spec = min_spec(team_ids(16), GroupConstraint::distinct(0, 3, 0b111));
        spec.salary_cap = 22_000;
        spec.salary_floor = 20_000;
        let lineups = build_lineups(&pool.view(), &spec, &config(40)).unwrap();
        assert!(!lineups.is_empty(), "repair recovered nothing to check");
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn a_stack_minimum_is_met_by_every_lineup() {
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::stack(0, 3, 0b111));
        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn stacks_are_drawn_across_teams_rather_than_always_the_same_one() {
        // The point of drawing a target per attempt. A builder that answered
        // "which team?" the same way every time would return a portfolio stacked
        // entirely on one team, which is the opposite of what a portfolio is for.
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::stack(0, 3, 0b111));
        let lineups = build_lineups(&pool.view(), &spec, &config(60)).unwrap();
        assert!(lineups.len() > 5);
        let stacked: std::collections::HashSet<i32> = lineups
            .iter()
            .map(|lineup| {
                let mut counts = std::collections::HashMap::new();
                for &p in lineup {
                    *counts.entry(team_ids(16)[p as usize]).or_insert(0u32) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(k, _)| k)
                    .unwrap()
            })
            .collect();
        assert!(
            stacked.len() > 1,
            "every lineup stacked the same team: {stacked:?}"
        );
    }

    #[test]
    fn a_stack_minimum_actually_binds() {
        let pool = make_pool();
        let loose = min_spec(team_ids(16), GroupConstraint::at_most(0, 4, 0b111));
        let biggest = |lineup: &Lineup| {
            let mut counts = std::collections::HashMap::new();
            for &p in lineup {
                *counts.entry(team_ids(16)[p as usize]).or_insert(0u32) += 1;
            }
            counts.into_values().max().unwrap_or(0)
        };
        let unconstrained = build_lineups(&pool.view(), &loose, &config(50)).unwrap();
        assert!(
            unconstrained.iter().any(|l| biggest(l) < 3),
            "every unconstrained lineup already had a three-stack"
        );

        let spec = min_spec(team_ids(16), GroupConstraint::stack(0, 3, 0b111));
        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        assert!(!lineups.is_empty());
        assert!(lineups.iter().all(|l| biggest(l) >= 3));
    }

    #[test]
    fn a_stack_no_key_can_supply_returns_nothing() {
        // Every player on their own team, so no key has a second player to
        // contribute. The roster has room for the stack — this is not the
        // validation case — the pool simply cannot supply it, and the capability
        // scan catches that once rather than once per attempt.
        let pool = make_pool();
        let spec = min_spec((0..16).collect(), GroupConstraint::stack(0, 2, 0b111));
        assert!(build_lineups(&pool.view(), &spec, &config(10))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_stack_minimum_survives_salary_repair() {
        let pool = make_pool();
        let mut spec = min_spec(team_ids(16), GroupConstraint::stack(0, 3, 0b111));
        spec.salary_cap = 22_000;
        spec.salary_floor = 20_000;
        let lineups = build_lineups(&pool.view(), &spec, &config(40)).unwrap();
        assert!(!lineups.is_empty(), "repair recovered nothing to check");
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn a_stack_and_a_distinct_minimum_hold_together() {
        // They pull in opposite directions — one concentrates, the other spreads
        // — so a roster satisfying both is the real test of the two mechanisms
        // not fighting each other.
        let pool = make_pool();
        let mut spec = tiny_spec(team_ids(16), 30_000, 0);
        spec.groups = vec![
            GroupConstraint::stack(0, 2, 0b111),
            GroupConstraint::distinct(0, 3, 0b111),
        ];
        let lineups = build_lineups(&pool.view(), &spec, &config(50)).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn a_stack_above_its_own_cap_is_rejected() {
        let pool = make_pool();
        let mut group = GroupConstraint::stack(0, 4, 0b111);
        group.max_count = 3;
        let spec = min_spec(team_ids(16), group);
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(10)),
            Err(BuildError::Spec(SpecError::StackAboveCap {
                group: 0,
                min_stack: 4,
                max_count: 3
            }))
        );
    }

    #[test]
    fn a_stack_pair_is_met_by_every_lineup() {
        // 2-2 on a four-slot roster: some team supplies two players and a
        // *different* team supplies two more. assert_all_valid checks the
        // top-two counts of every lineup.
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::stack_pair(0, 2, 2, 0b111));
        let lineups = build_lineups(&pool.view(), &spec, &config(30)).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn a_stack_pair_actually_binds() {
        // The same build without the secondary produces at least one lineup the
        // pair constraint would reject — otherwise the test above proves nothing.
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::stack(0, 2, 0b111));
        let lineups = build_lineups(&pool.view(), &spec, &config(30)).unwrap();
        let violates = lineups.iter().any(|lineup| {
            let mut counts = std::collections::HashMap::new();
            for &p in lineup {
                *counts.entry(team_ids(16)[p as usize]).or_insert(0u32) += 1;
            }
            let mut sizes: Vec<u32> = counts.values().copied().collect();
            sizes.sort_unstable_by(|a, b| b.cmp(a));
            sizes.get(1).copied().unwrap_or(0) < 2
        });
        assert!(
            violates,
            "every lineup was already a 2-2; the pair never binds"
        );
    }

    #[test]
    fn a_secondary_above_its_primary_is_rejected() {
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::stack_pair(0, 2, 3, 0b111));
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(10)),
            Err(BuildError::Spec(SpecError::SecondaryStackShape {
                group: 0,
                min_stack: 2,
                secondary: 3
            }))
        );
    }

    #[test]
    fn a_stack_pair_beyond_the_counted_slots_is_rejected() {
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::stack_pair(0, 3, 2, 0b111));
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(10)),
            Err(BuildError::Spec(SpecError::StackPairExceedsSlots {
                group: 0,
                min_stack: 3,
                secondary: 2,
                slots: 4
            }))
        );
    }

    #[test]
    fn a_stack_pair_one_team_cannot_supply_returns_nothing() {
        // Every player on one team: the primary is trivially satisfiable and
        // the secondary can never be, which the capability scan must catch
        // before the attempt budget is burned proving it.
        let pool = make_pool();
        let spec = min_spec(vec![0; 16], GroupConstraint::stack_pair(0, 2, 2, 0b111));
        let lineups = build_lineups(&pool.view(), &spec, &config(10)).unwrap();
        assert!(lineups.is_empty());
    }

    #[test]
    fn a_stack_pair_is_deterministic() {
        // The secondary target is drawn from the same generator as everything
        // else, so the output must still be a function of the seed alone.
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::stack_pair(0, 2, 2, 0b111));
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(30)).unwrap(),
            build_lineups(&pool.view(), &spec, &config(30)).unwrap()
        );
    }

    #[test]
    fn a_constraint_that_constrains_nothing_is_rejected() {
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::distinct(0, 0, 0b111));
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(10)),
            Err(BuildError::Spec(SpecError::InertGroup(0)))
        );
    }

    #[test]
    fn minimums_are_deterministic() {
        // The stack target is drawn from the same generator as the jitter, so the
        // output must still be a function of the seed alone.
        let pool = make_pool();
        let spec = min_spec(team_ids(16), GroupConstraint::stack(0, 3, 0b111));
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(40)).unwrap(),
            build_lineups(&pool.view(), &spec, &config(40)).unwrap()
        );
    }

    // --- Locks ------------------------------------------------------------

    #[test]
    fn a_locked_player_appears_in_every_lineup() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        // Player 1 is a catcher; slot group 0 is the catcher slot.
        let mut cfg = config(40);
        cfg.locks = vec![(1, 0)];
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);
        for lineup in &lineups {
            assert_eq!(lineup[0], 1, "the lock is not in its slot: {lineup:?}");
        }
    }

    #[test]
    fn locking_a_player_the_greedy_would_not_pick_still_works() {
        // The cheapest catcher has the worst projection, so an unlocked build
        // rarely takes them. If locking only nudged the objective rather than
        // forcing the pick, this would come back mixed.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(40);
        cfg.locks = vec![(0, 0)];
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert!(!lineups.is_empty());
        assert!(lineups.iter().all(|l| l[0] == 0));
    }

    #[test]
    fn locks_fill_several_slots_of_one_group() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        // Both outfield slots, which is slot group 1.
        let mut cfg = config(40);
        cfg.locks = vec![(4, 1), (5, 1)];
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);
        for lineup in &lineups {
            assert_eq!(&lineup[1..3], &[4, 5]);
        }
    }

    #[test]
    fn locks_survive_salary_repair() {
        // Repair reaches the floor by swapping the cheapest player out. A lock is
        // often exactly that player, and swapping it would silently undo the one
        // thing the caller insisted on.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 22_000, 20_000);
        let mut cfg = config(40);
        cfg.locks = vec![(0, 0)];
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert!(!lineups.is_empty(), "repair recovered nothing to check");
        assert_all_valid(&lineups, &pool.view(), &spec);
        for lineup in &lineups {
            assert!(
                lineup.contains(&0),
                "repair swapped the lock out: {lineup:?}"
            );
        }
    }

    #[test]
    fn a_lock_that_cannot_fit_returns_nothing_rather_than_looping() {
        let pool = make_pool();
        // The cheapest legal roster costs 12_700; the dearest catcher pushes the
        // same roster to 14_800. A cap between the two is satisfiable in general
        // and unsatisfiable with that catcher locked in.
        let spec = tiny_spec(team_ids(16), 14_000, 0);
        assert!(
            !build_lineups(&pool.view(), &spec, &config(20))
                .unwrap()
                .is_empty(),
            "the cap alone should be satisfiable"
        );

        let mut cfg = config(20);
        cfg.locks = vec![(3, 0)];
        // Empty, not a lineup without the lock: a lock the caller cannot have is
        // not quietly dropped.
        assert!(build_lineups(&pool.view(), &spec, &cfg).unwrap().is_empty());
    }

    #[test]
    fn an_expensive_lock_in_a_late_slot_group_is_affordable() {
        // The regression that made locks useless for exactly the players anyone
        // wants to lock. The salary reservation used to hold back the *cheapest*
        // eligible player for each upcoming group, so the earlier groups happily
        // spent the budget an expensive lock in a later group needed, and the
        // lock was then unaffordable on every attempt. Locking a star returned
        // nothing while locking a scrub worked, which reads as the feature being
        // broken rather than the pool being tight.
        //
        // Pitchers are the last slot group here and player 15 is the dearest at
        // 5_100. The cap has to be tight enough for the reservation to bind: the
        // catcher and outfielders can spend up to 20_200 between them, so at
        // 23_000 a run that reserved only the cheapest pitcher (3_000) would
        // routinely leave under 5_100 for the lock. With the lock's own cost
        // reserved there is always room.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 23_000, 0);
        let mut cfg = config(30);
        cfg.locks = vec![(15, 2)];
        // A yield, not merely a non-empty result: the old behaviour still found
        // the occasional lineup on an attempt that happened to pick cheaply, so
        // `is_empty` would not have caught this. Measured at 6 of 30 before the
        // fix and 27 after.
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert!(
            lineups.len() >= 20,
            "only {} of 30 lineups: the reservation is holding back the cheapest \
             pitcher rather than the locked one",
            lineups.len()
        );
        assert_all_valid(&lineups, &pool.view(), &spec);
        assert!(lineups.iter().all(|l| l.contains(&15)));
    }

    #[test]
    fn a_locked_player_is_not_counted_as_a_cheap_option_elsewhere() {
        // A lock is unavailable to fill any other slot, so leaving it in the
        // "cheapest way to finish" prefix understates what the rest of the roster
        // must cost. Locking the cheapest outfielder must not make the reservation
        // for the *other* outfield slots any cheaper.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(30);
        cfg.locks = vec![(4, 1)];
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert!(!lineups.is_empty());
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn locks_respect_group_caps() {
        let pool = make_pool();
        // Everyone on one team with a cap of 3, and a 4-player roster: locking
        // does not get to override a constraint.
        let spec = tiny_spec(vec![0; 16], 30_000, 0);
        let mut cfg = config(10);
        cfg.locks = vec![(0, 0)];
        assert!(build_lineups(&pool.view(), &spec, &cfg).unwrap().is_empty());
    }

    #[test]
    fn locking_two_conflicting_players_yields_nothing() {
        let pool = make_pool();
        let mut spec = tiny_spec(team_ids(16), 30_000, 0);
        spec.conflicts = ConflictGraph::from_pairs([(0u32, 12u32)], 16);
        let mut cfg = config(20);
        cfg.locks = vec![(0, 0), (12, 2)];
        assert!(build_lineups(&pool.view(), &spec, &cfg).unwrap().is_empty());
    }

    #[test]
    fn locking_no_one_changes_nothing() {
        // The opt-in must be inert, or every committed seed and benchmark moved.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut empty = config(40);
        empty.locks = Vec::new();
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(40)).unwrap(),
            build_lineups(&pool.view(), &spec, &empty).unwrap()
        );
    }

    #[test]
    fn a_lock_naming_a_missing_player_is_rejected() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(10);
        cfg.locks = vec![(99, 0)];
        assert!(matches!(
            build_lineups(&pool.view(), &spec, &cfg),
            Err(BuildError::LockOutOfRange { player: 99, .. })
        ));
    }

    #[test]
    fn a_lock_into_a_slot_the_player_cannot_fill_is_rejected() {
        // Reported rather than returned as an empty result: "no valid lineups"
        // says nothing about a pitcher having been locked into the catcher slot.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(10);
        cfg.locks = vec![(12, 0)];
        assert_eq!(
            build_lineups(&pool.view(), &spec, &cfg),
            Err(BuildError::LockNotEligible {
                player: 12,
                slot_group: 0
            })
        );
    }

    #[test]
    fn more_locks_than_a_slot_group_holds_is_rejected() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(10);
        cfg.locks = vec![(0, 0), (1, 0)];
        assert_eq!(
            build_lineups(&pool.view(), &spec, &cfg),
            Err(BuildError::OverfilledSlotGroup {
                slot_group: 0,
                locks: 2,
                count: 1
            })
        );
    }

    #[test]
    fn locking_the_same_player_twice_is_rejected() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(10);
        cfg.locks = vec![(4, 1), (4, 1)];
        assert_eq!(
            build_lineups(&pool.view(), &spec, &cfg),
            Err(BuildError::DuplicateLock(4))
        );
    }

    // --- Exposure caps ----------------------------------------------------

    #[test]
    fn an_exposure_cap_bounds_how_often_a_player_appears() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let uncapped = build_lineups(&pool.view(), &spec, &config(40)).unwrap();
        let busiest = (0..16u32)
            .max_by_key(|p| uncapped.iter().filter(|l| l.contains(p)).count())
            .unwrap();
        let before = uncapped.iter().filter(|l| l.contains(&busiest)).count();
        assert!(before > 3, "nothing appeared often enough to cap");

        let mut cfg = config(40);
        let mut limits = vec![u32::MAX; 16];
        limits[busiest as usize] = 3;
        cfg.exposure_limits = limits;
        let capped = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert_all_valid(&capped, &pool.view(), &spec);
        assert!(capped.iter().filter(|l| l.contains(&busiest)).count() <= 3);
    }

    #[test]
    fn an_exposure_cap_of_zero_excludes_a_player_entirely() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(40);
        let mut limits = vec![u32::MAX; 16];
        limits[4] = 0;
        cfg.exposure_limits = limits;
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert!(!lineups.is_empty());
        assert!(lineups.iter().all(|l| !l.contains(&4)));
    }

    #[test]
    fn exposure_caps_are_deterministic() {
        // The cap is applied at the merge, which is sequential and in fixed chunk
        // order. If it had been applied inside the parallel region the answer
        // would depend on how rayon interleaved the chunks.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(40);
        cfg.exposure_limits = vec![2; 16];
        let a = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        let b = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_tight_exposure_cap_returns_fewer_lineups_rather_than_illegal_ones() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(40);
        cfg.exposure_limits = vec![1; 16];
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert!(lineups.len() < 40, "the cap did not bind");
        assert_all_valid(&lineups, &pool.view(), &spec);
        // At most one appearance each means no player is repeated across lineups.
        let mut seen = std::collections::HashSet::new();
        for lineup in &lineups {
            for &player in lineup {
                assert!(seen.insert(player), "player {player} exceeded a cap of 1");
            }
        }
    }

    #[test]
    fn uncapped_exposure_changes_nothing() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut all_max = config(40);
        all_max.exposure_limits = vec![u32::MAX; 16];
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(40)).unwrap(),
            build_lineups(&pool.view(), &spec, &all_max).unwrap()
        );
    }

    #[test]
    fn an_exposure_column_for_the_wrong_pool_is_rejected() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(10);
        cfg.exposure_limits = vec![1; 8];
        assert_eq!(
            build_lineups(&pool.view(), &spec, &cfg),
            Err(BuildError::ExposureLengthMismatch {
                len: 8,
                expected: 16
            })
        );
    }

    #[test]
    fn mismatched_pool_columns_are_rejected() {
        let mut pool = make_pool();
        pool.stddevs.pop();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        assert_eq!(
            build_lineups(&pool.view(), &spec, &config(10)),
            Err(BuildError::ColumnLengthMismatch {
                column: "stddevs",
                len: 15,
                expected: 16
            })
        );
    }

    #[test]
    fn an_invalid_spec_surfaces_as_a_spec_error() {
        let pool = make_pool();
        let mut spec = tiny_spec(team_ids(16), 30_000, 0);
        spec.salary_floor = 99_999;
        assert!(matches!(
            build_lineups(&pool.view(), &spec, &config(10)),
            Err(BuildError::Spec(SpecError::FloorAboveCap { .. }))
        ));
    }

    #[test]
    fn empty_profiles_are_rejected() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(10);
        cfg.profiles.clear();
        assert_eq!(
            build_lineups(&pool.view(), &spec, &cfg),
            Err(BuildError::NoProfiles)
        );
    }

    #[test]
    fn diversity_weight_off_by_default_and_reproduces_the_plain_build() {
        // The guarantee that lets this ship without changing anybody's output.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(40);
        cfg.attempts_per_lineup = 8;
        cfg.chunks = 4;
        let plain = build_lineups(&pool.view(), &spec, &cfg).unwrap();

        let mut explicit = cfg.clone();
        explicit.diversity_weight = 0.0;
        assert_eq!(
            build_lineups(&pool.view(), &spec, &explicit).unwrap(),
            plain
        );
    }

    #[test]
    fn diversity_weight_spreads_the_portfolio_across_more_players() {
        // The claim the parameter exists to make. Asserted rather than left to a
        // benchmark, because a knob that changed nothing measurable would be
        // documentation.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(40);
        cfg.attempts_per_lineup = 8;
        cfg.chunks = 4;

        let plain = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        let mut faded = cfg.clone();
        faded.diversity_weight = 1.0;
        let spread = build_lineups(&pool.view(), &spec, &faded).unwrap();

        let distinct = |lineups: &[Lineup]| {
            lineups
                .iter()
                .flat_map(|l| l.iter().copied())
                .collect::<HashSet<u32>>()
                .len()
        };
        assert!(
            distinct(&spread) > distinct(&plain),
            "fading used players should reach more of the pool: {} against {}",
            distinct(&spread),
            distinct(&plain)
        );
    }

    #[test]
    fn diversity_weight_is_deterministic() {
        // It reads a per-chunk count, never a shared one, precisely so this holds.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(40);
        cfg.attempts_per_lineup = 8;
        cfg.diversity_weight = 0.75;
        let a = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        let b = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn diversity_weight_never_costs_lineups() {
        // The reason it is a preference and not a cap: an exposure limit reaches
        // the same spread by discarding finished lineups, and this must not.
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(30);
        cfg.attempts_per_lineup = 12;
        cfg.chunks = 4;
        let plain = build_lineups(&pool.view(), &spec, &cfg).unwrap().len();

        let mut faded = cfg.clone();
        faded.diversity_weight = 2.0;
        assert_eq!(
            build_lineups(&pool.view(), &spec, &faded).unwrap().len(),
            plain
        );
    }

    #[test]
    fn zero_chunks_is_rejected() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(10);
        cfg.chunks = 0;
        assert_eq!(
            build_lineups(&pool.view(), &spec, &cfg),
            Err(BuildError::NoChunks)
        );
    }

    #[test]
    fn zero_noise_still_produces_valid_lineups() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let mut cfg = config(20);
        cfg.noise = 0.0;
        let lineups = build_lineups(&pool.view(), &spec, &cfg).unwrap();
        assert_all_valid(&lineups, &pool.view(), &spec);
    }

    #[test]
    fn never_returns_more_than_requested() {
        let pool = make_pool();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let lineups = build_lineups(&pool.view(), &spec, &config(5)).unwrap();
        assert!(lineups.len() <= 5);
    }

    #[test]
    fn salary_and_projection_helpers_agree_with_the_pool() {
        let pool = make_pool();
        let view = pool.view();
        let spec = tiny_spec(team_ids(16), 30_000, 0);
        let lineup = vec![0u32, 4, 5, 12];
        assert_eq!(
            lineup_salary(&view, &spec, &lineup),
            lineup
                .iter()
                .map(|&i| pool.salaries[i as usize])
                .sum::<i64>()
        );
        let expected: f64 = lineup.iter().map(|&i| pool.projections[i as usize]).sum();
        assert!((lineup_projection(&view, &spec, &lineup) - expected).abs() < 1e-12);
    }
}
