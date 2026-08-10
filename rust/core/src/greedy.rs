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

use crate::roster::{GroupTally, PositionMask, RosterSpec, SpecError};

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
    /// Independent work units. Fixed rather than thread-derived, so results do not
    /// depend on the machine.
    pub chunks: usize,
    /// Profiles cycled across attempts.
    pub profiles: Vec<JitterProfile>,
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
        }
    }
}

/// Why construction could not run.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    let cheapest: Vec<Vec<i64>> = eligible
        .iter()
        .map(|group| {
            let mut salaries: Vec<i64> = group.iter().map(|&i| pool.salaries[i as usize]).collect();
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
    // Least a group can spend filling all of its own slots.
    let group_floor =
        |j: usize| -> i64 { cheapest[j][spec.slots[j].count.min(cheapest[j].len() - 1)] };
    let mut suffix_cost = vec![0i64; spec.slots.len() + 1];
    for j in (0..spec.slots.len()).rev() {
        suffix_cost[j] = suffix_cost[j + 1] + group_floor(j);
    }

    let total_attempts = config
        .num_lineups
        .saturating_mul(config.attempts_per_lineup);
    let per_chunk = (total_attempts / config.chunks).max(1);

    let chunk_results: Vec<Vec<Lineup>> = (0..config.chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut builder = Builder::new(pool, spec, &eligible, &cheapest, &suffix_cost);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                config.seed ^ (chunk as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut seen: HashSet<Lineup> = HashSet::new();
            let mut out: Vec<Lineup> = Vec::new();

            for attempt in 0..per_chunk {
                let profile = config.profiles[attempt % config.profiles.len()];
                builder.randomize_objective(&mut rng, profile, config.noise);
                if let Some(lineup) = builder.fill() {
                    let mut key = lineup.clone();
                    key.sort_unstable();
                    if seen.insert(key) {
                        out.push(lineup);
                    }
                }
            }
            out
        })
        .collect();

    // Merge in chunk order, deduplicating again: two chunks can independently
    // find the same lineup, and chunk order is fixed so the merge is stable.
    let mut seen: HashSet<Lineup> = HashSet::new();
    let mut all: Vec<Lineup> = Vec::with_capacity(config.num_lineups);
    for chunk in chunk_results {
        for lineup in chunk {
            if all.len() >= config.num_lineups {
                return Ok(all);
            }
            let mut key = lineup.clone();
            key.sort_unstable();
            if seen.insert(key) {
                all.push(lineup);
            }
        }
    }
    Ok(all)
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
    objective: Vec<f64>,
    used: Vec<bool>,
    candidates: Vec<u32>,
    lineup: Vec<u32>,
    slot_of: Vec<usize>,
    tally: GroupTally<'a>,
}

impl<'a> Builder<'a> {
    fn new(
        pool: &'a PlayerPool<'a>,
        spec: &'a RosterSpec,
        eligible: &'a [Vec<u32>],
        cheapest: &'a [Vec<i64>],
        suffix_cost: &'a [i64],
    ) -> Self {
        let n = pool.len();
        let roster_size = spec.roster_size();
        Self {
            pool,
            spec,
            eligible,
            cheapest,
            suffix_cost,
            objective: vec![0.0; n],
            used: vec![false; n],
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
    ) {
        let ceiling = sample_range(rng, profile.ceiling);
        let leverage = sample_range(rng, profile.leverage);

        for i in 0..self.pool.len() {
            let projection = self.pool.projections[i];
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

    /// Fill every slot greedily, then repair salary if needed.
    fn fill(&mut self) -> Option<Lineup> {
        self.used.fill(false);
        self.lineup.clear();
        self.slot_of.clear();
        self.tally.reset();
        let mut salary: i64 = 0;

        for (group_idx, slot) in self.spec.slots.iter().enumerate() {
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

            let mut picked = 0;
            for &player in &self.candidates {
                if picked == slot.count {
                    break;
                }
                let p = player as usize;
                // Reserve enough for the slots still to be filled, including the
                // rest of this group.
                let still_needed = slot.count - picked - 1;
                let prefix = &self.cheapest[group_idx];
                let remaining =
                    prefix[still_needed.min(prefix.len() - 1)] + self.suffix_cost[group_idx + 1];
                if salary + self.pool.salaries[p] + remaining > self.spec.salary_cap {
                    continue;
                }
                if self.tally.would_exceed(p, group_idx) {
                    continue;
                }
                self.used[p] = true;
                self.tally.add(p, group_idx);
                salary += self.pool.salaries[p];
                self.lineup.push(player);
                self.slot_of.push(group_idx);
                picked += 1;
            }
            if picked < slot.count {
                return None;
            }
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

        let mut order: Vec<usize> = (0..self.lineup.len()).collect();
        let salaries = self.pool.salaries;
        let lineup = &self.lineup;
        order.sort_unstable_by(|&a, &b| {
            salaries[lineup[a] as usize]
                .cmp(&salaries[lineup[b] as usize])
                .then(a.cmp(&b))
        });

        for slot_index in order {
            let outgoing = self.lineup[slot_index] as usize;
            let group_idx = self.slot_of[slot_index];
            let outgoing_salary = self.pool.salaries[outgoing];

            // Take the outgoing player out of the tally first, so a replacement
            // from the same team is not rejected by the slot it is about to free.
            self.tally.remove(outgoing, group_idx);
            self.used[outgoing] = false;

            let mut swapped = None;
            for &candidate in &self.eligible[group_idx] {
                let c = candidate as usize;
                if self.used[c] {
                    continue;
                }
                let gain = self.pool.salaries[c] - outgoing_salary;
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
                swapped = Some((candidate, new_total));
                break;
            }

            match swapped {
                Some((candidate, new_total)) => {
                    let c = candidate as usize;
                    self.used[c] = true;
                    self.tally.add(c, group_idx);
                    self.lineup[slot_index] = candidate;
                    *salary = new_total;
                    return true;
                }
                None => {
                    // Restore and try the next slot.
                    self.used[outgoing] = true;
                    self.tally.add(outgoing, group_idx);
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

/// Total salary of a lineup. Exposed because callers routinely want it and
/// recomputing it in Python defeats the point of building here.
pub fn lineup_salary(pool: &PlayerPool<'_>, lineup: &[u32]) -> i64 {
    lineup.iter().map(|&i| pool.salaries[i as usize]).sum()
}

/// Total projection of a lineup.
pub fn lineup_projection(pool: &PlayerPool<'_>, lineup: &[u32]) -> f64 {
    lineup.iter().map(|&i| pool.projections[i as usize]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roster::{GroupConstraint, SlotGroup};

    const P: PositionMask = 1 << 0;
    const C: PositionMask = 1 << 1;
    const OF: PositionMask = 1 << 2;

    /// A deliberately tiny sport: 1 catcher, 2 outfielders, 1 pitcher.
    fn tiny_spec(team_ids: Vec<i32>, cap: i64, floor: i64) -> RosterSpec {
        RosterSpec {
            slots: vec![
                SlotGroup {
                    eligible: C,
                    count: 1,
                },
                SlotGroup {
                    eligible: OF,
                    count: 2,
                },
                SlotGroup {
                    eligible: P,
                    count: 1,
                },
            ],
            salary_cap: cap,
            salary_floor: floor,
            groups: vec![GroupConstraint {
                key_column: 0,
                max_count: 3,
                slots: 0b111,
            }],
            key_columns: vec![team_ids],
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

            let salary = lineup_salary(pool, lineup);
            assert!(salary <= spec.salary_cap, "over cap: {salary}");
            assert!(salary >= spec.salary_floor, "under floor: {salary}");

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

            // Group caps.
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
                for (key, count) in counts {
                    assert!(count <= group.max_count, "group cap breached for key {key}");
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

    /// Chunk count is part of the reproducibility contract, so it must change the
    /// result — otherwise the contract would be silently weaker than documented.
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
        let lineup = vec![0u32, 4, 5, 12];
        assert_eq!(
            lineup_salary(&view, &lineup),
            lineup
                .iter()
                .map(|&i| pool.salaries[i as usize])
                .sum::<i64>()
        );
        let expected: f64 = lineup.iter().map(|&i| pool.projections[i as usize]).sum();
        assert!((lineup_projection(&view, &lineup) - expected).abs() < 1e-12);
    }
}
