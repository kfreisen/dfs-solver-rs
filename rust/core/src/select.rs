//! Choosing which lineups to enter, from a pool of candidates.
//!
//! Construction produces a large, diverse pool of *valid* lineups. It does not
//! produce good ones in any particular order — `build_lineups` returns the first
//! N distinct rosters it happened to find, and on a realistically priced slate
//! the median of those sits well below the best of them. Selection is the step
//! that turns a pool into a portfolio.
//!
//! # The objective
//!
//! A lineup is scored not on its own merits but on what it adds to the entries
//! already chosen. Given a matrix of simulated outcomes, the marginal gain of
//! adding candidate `c` has the closed form
//!
//! ```text
//! gain(c) = mean over outcomes s of  g(score[c][s], bar[s])
//! ```
//!
//! where `bar[s]` is what the portfolio already achieves at outcome `s`. That is
//! one subtract, one clamp and one add per outcome — see
//! [`crate::simd::sum_excess_over`].
//!
//! **Which `g` depends on the contest, and this is not a tuning knob.** See
//! [`Objective`]: a cash game pays a flat amount for clearing a line and nothing
//! for clearing it well, so entries are judged independently and diversity is
//! actively wrong. A tournament pays almost nothing outside the extreme tail, so
//! what matters is whether *any* entry reaches a winning score and entries that
//! win together are wasted on each other. Those are different functionals, not
//! different parameters of one.
//!
//! **Diversity is not a term in any of them.** It does not need to be, in the
//! portfolio modes. A candidate that duplicates a lineup already held scores zero
//! at every outcome and so gains nothing; a candidate that wins where the
//! portfolio currently loses gains a great deal. Bolting a separate overlap
//! penalty on top double-counts what the objective already does, and — because
//! penalties subtract — destroys the submodularity that the whole method rests
//! on. The measured effect of leaving it out is *more* diversity, not less.
//!
//! # Why greedy, and why that is not a compromise
//!
//! `f` is monotone and submodular: taking the elementwise maximum over a larger
//! set can only help, and helps less the more you already hold. For maximizing a
//! monotone submodular function under a cardinality limit, the greedy algorithm
//! is within `1 - 1/e` of optimal, and no polynomial-time algorithm does better
//! unless P = NP. Greedy is not a heuristic stand-in for something better here;
//! it is the best available answer.
//!
//! Lazy evaluation (Minoux) exploits the same property for speed. Gains only ever
//! shrink as the portfolio grows, so a candidate whose gain was computed earlier
//! and is already below the best gain found this round cannot win, and need not
//! be recomputed. The output is *identical* to the naive loop — this buys time,
//! not a different answer, and the tests assert exactly that.
//!
//! Cash is the exception: it is modular rather than submodular, so greedy over
//! it is not an approximation at all — ranking by cash probability is exactly
//! optimal, and falls out of the same loop because the state never moves.
//!
//! # Where the line comes from
//!
//! Every objective takes a score, and it is deliberately a *constant* rather
//! than a quantile of the portfolio's own distribution. "Mean of the top q
//! outcomes" sounds more principled and is not submodular — which outcomes count
//! would depend on the set being chosen — so the guarantee above, and lazy
//! evaluation with it, would be lost. Compute the line up front from whatever
//! you know about the field ([`quantile`] does it from the candidate pool) and
//! hand it in.

use std::collections::BinaryHeap;

use crate::simd::{raise_to, sum_excess_over};

/// Score every lineup against a simulated player universe.
///
/// `universe` is a flat `(n_players, n_outcomes)` row-major matrix: what each
/// player scored in each simulated outcome. A lineup's score in an outcome is
/// the sum of its players' scores there, weighted by the slot each occupies —
/// which is how a showdown captain is worth 1.5x here as everywhere else.
///
/// The whole point of taking a *universe* rather than per-lineup distributions
/// is that correlation is already baked into the matrix by whoever produced it.
/// Two lineups sharing a stack move together automatically, because they index
/// the same rows. Adding a lineup costs one gather-and-sum per outcome rather
/// than a fresh correlated draw, which is what makes a hundred thousand
/// candidates tractable.
///
/// Returns a flat `(n_lineups, n_outcomes)` matrix.
pub fn score_lineups(
    universe: &[f32],
    n_outcomes: usize,
    lineups: &[u32],
    roster_size: usize,
    slot_multipliers: &[f32],
) -> Result<Vec<f32>, SelectError> {
    if n_outcomes == 0 {
        return Err(SelectError::NoOutcomes);
    }
    if universe.len() % n_outcomes != 0 {
        return Err(SelectError::RaggedScores {
            len: universe.len(),
            n_outcomes,
        });
    }
    if roster_size == 0 || lineups.len() % roster_size != 0 {
        return Err(SelectError::RaggedRosters {
            len: lineups.len(),
            roster_size,
        });
    }
    if slot_multipliers.len() != roster_size {
        return Err(SelectError::CandidateCountMismatch {
            rosters: roster_size,
            scores: slot_multipliers.len(),
        });
    }

    let n_lineups = lineups.len() / roster_size;
    let mut out = vec![0.0f32; n_lineups * n_outcomes];
    for lineup in 0..n_lineups {
        let roster = &lineups[lineup * roster_size..(lineup + 1) * roster_size];
        let row = &mut out[lineup * n_outcomes..(lineup + 1) * n_outcomes];
        for (slot, &player) in roster.iter().enumerate() {
            let start = player as usize * n_outcomes;
            let scores = &universe[start..start + n_outcomes];
            let weight = slot_multipliers[slot];
            if weight == 1.0 {
                for (o, &s) in row.iter_mut().zip(scores) {
                    *o += s;
                }
            } else {
                for (o, &s) in row.iter_mut().zip(scores) {
                    *o += s * weight;
                }
            }
        }
    }
    Ok(out)
}

/// Why a selection could not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectError {
    /// The score matrix is not a whole number of candidates.
    RaggedScores { len: usize, n_outcomes: usize },
    /// No outcomes to average over.
    NoOutcomes,
    /// The exposure column disagrees with the roster matrix.
    ExposureLengthMismatch { len: usize, expected: usize },
    /// The roster matrix is not a whole number of lineups.
    RaggedRosters { len: usize, roster_size: usize },
    /// The roster matrix and the score matrix describe different pools.
    CandidateCountMismatch { rosters: usize, scores: usize },
}

impl std::fmt::Display for SelectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RaggedScores { len, n_outcomes } => write!(
                f,
                "the score matrix has {len} entries, which is not a multiple of the \
                 {n_outcomes} outcomes"
            ),
            Self::NoOutcomes => write!(f, "the score matrix has no outcomes to average over"),
            Self::ExposureLengthMismatch { len, expected } => write!(
                f,
                "exposure_limits has {len} entries but the pool has {expected} players"
            ),
            Self::RaggedRosters { len, roster_size } => write!(
                f,
                "the roster matrix has {len} entries, which is not a multiple of the \
                 roster size {roster_size}"
            ),
            Self::CandidateCountMismatch { rosters, scores } => write!(
                f,
                "the roster matrix describes {rosters} lineups but the score matrix \
                 describes {scores}"
            ),
        }
    }
}

impl std::error::Error for SelectError {}

/// What a portfolio is being optimized *for*.
///
/// Two contest types want genuinely different things, and the difference is not
/// a parameter of one objective — it is a different functional.
///
/// **Cash** games pay a flat amount for beating a line and nothing for beating
/// it well. Every entry is judged alone; there is no credit for covering
/// outcomes another entry already covers, because a second entry cashing in the
/// same outcome pays a second time. Diversity is not merely unnecessary here, it
/// is *harmful* — the right play is the highest-probability roster, and the
/// runner-up shape after it. So [`Self::Cash`] is modular, not submodular, and
/// greedy over it degenerates to taking the top candidates by cash probability,
/// which is exactly correct.
///
/// **Tournaments** pay almost nothing outside the extreme tail. What matters is
/// the chance that *at least one* entry reaches a winning score, so entries that
/// win in the same outcomes are largely wasted on each other.
/// [`Self::Cover`] is precisely that: the fraction of outcomes in which the
/// portfolio has any entry above the line. It is weighted set cover over
/// simulated outcomes — submodular, and the sharpest statement of "only the tail
/// pays".
///
/// [`Self::Excess`] sits between them: it credits *how far* above the line the
/// best entry got, which suits a payout curve that keeps climbing with rank
/// rather than one that pays a lump for finishing in the money.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Objective {
    /// Mean excess of the portfolio's best entry over `threshold`.
    ///
    /// The general-purpose choice, and the one the prior art used with
    /// `threshold = 0`.
    Excess { threshold: f32 },
    /// Fraction of outcomes in which some entry reaches `line`. Tournaments.
    Cover { line: f32 },
    /// Mean probability that an entry reaches `line`, judged independently.
    /// Cash games.
    Cash { line: f32 },
}

impl Objective {
    /// The marginal value of adding `scores` to a portfolio whose per-outcome
    /// state is `bar`. Not yet divided by the outcome count.
    fn gain(&self, scores: &[f32], bar: &[f32]) -> f32 {
        match *self {
            Self::Excess { .. } => sum_excess_over(scores, bar),
            // An outcome counts only if this entry clears the line and the
            // portfolio does not already have one that does.
            Self::Cover { line } => scores[..bar.len()]
                .iter()
                .zip(bar)
                .filter(|&(&s, &b)| s >= line && b < line)
                .count() as f32,
            // No reference to `bar` at all: that is what makes cash modular.
            Self::Cash { line } => {
                scores[..bar.len()].iter().filter(|&&s| s >= line).count() as f32
            }
        }
    }

    /// Fold a chosen entry into the portfolio's per-outcome state.
    fn absorb(&self, bar: &mut [f32], scores: &[f32]) {
        match self {
            // Cash entries do not interact, so the state never moves and every
            // candidate's gain stays what it was. Greedy then simply ranks.
            Self::Cash { .. } => {}
            _ => raise_to(bar, scores),
        }
    }

    /// The per-outcome state before anything is chosen.
    fn floor(&self) -> f32 {
        match *self {
            Self::Excess { threshold } => threshold,
            // Any value below the line means "not yet covered"; the line itself
            // would wrongly read as covered.
            Self::Cover { line } => line - 1.0,
            Self::Cash { .. } => f32::NEG_INFINITY,
        }
    }
}

/// Knobs for a selection run.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectConfig {
    /// How many lineups to enter. Fewer come back if the pool runs out of
    /// candidates that help.
    pub n_select: usize,
    /// What the portfolio is optimized for.
    pub objective: Objective,
    /// Per-player ceiling on how many selected lineups may contain that player.
    /// Empty means uncapped.
    ///
    /// A constraint rather than a penalty, so it cannot distort the objective —
    /// it only removes candidates from consideration. Note that this does take
    /// the problem outside plain cardinality, so the `1 - 1/e` guarantee no
    /// longer strictly applies; what remains is a greedy under a knapsack-like
    /// constraint, which is the same trade every exposure-capped optimizer makes.
    pub exposure_limits: Vec<u32>,
    /// Stop once the best remaining gain falls below this. Zero runs to
    /// `n_select`, but still stops if nothing can improve the portfolio at all.
    pub min_gain: f32,
}

impl Default for SelectConfig {
    fn default() -> Self {
        Self {
            n_select: 150,
            objective: Objective::Excess { threshold: 0.0 },
            exposure_limits: Vec::new(),
            min_gain: 0.0,
        }
    }
}

/// The candidate pool a selection reads.
#[derive(Debug, Clone, Copy)]
pub struct Candidates<'a> {
    /// Flat `(n_candidates, n_outcomes)` row-major simulated scores.
    pub scores: &'a [f32],
    /// How many simulated outcomes each candidate carries.
    pub n_outcomes: usize,
    /// Flat `(n_candidates, roster_size)` player indices, needed only for
    /// exposure caps. May be empty when no cap is set.
    pub rosters: &'a [u32],
    /// Players per lineup.
    pub roster_size: usize,
}

impl Candidates<'_> {
    /// Number of candidate lineups.
    pub fn len(&self) -> usize {
        if self.n_outcomes == 0 {
            0
        } else {
            self.scores.len() / self.n_outcomes
        }
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn row(&self, candidate: usize) -> &[f32] {
        let start = candidate * self.n_outcomes;
        &self.scores[start..start + self.n_outcomes]
    }

    fn roster(&self, candidate: usize) -> &[u32] {
        let start = candidate * self.roster_size;
        &self.rosters[start..start + self.roster_size]
    }

    fn validate(&self, config: &SelectConfig) -> Result<(), SelectError> {
        if self.n_outcomes == 0 {
            return Err(SelectError::NoOutcomes);
        }
        if self.scores.len() % self.n_outcomes != 0 {
            return Err(SelectError::RaggedScores {
                len: self.scores.len(),
                n_outcomes: self.n_outcomes,
            });
        }
        if !config.exposure_limits.is_empty() {
            if self.roster_size == 0 || self.rosters.len() % self.roster_size != 0 {
                return Err(SelectError::RaggedRosters {
                    len: self.rosters.len(),
                    roster_size: self.roster_size,
                });
            }
            let n_rosters = self.rosters.len() / self.roster_size;
            if n_rosters != self.len() {
                return Err(SelectError::CandidateCountMismatch {
                    rosters: n_rosters,
                    scores: self.len(),
                });
            }
            let highest = self.rosters.iter().copied().max().unwrap_or(0) as usize;
            if highest >= config.exposure_limits.len() {
                return Err(SelectError::ExposureLengthMismatch {
                    len: config.exposure_limits.len(),
                    expected: highest + 1,
                });
            }
        }
        Ok(())
    }
}

/// A candidate's most recent gain, ordered for the lazy-greedy heap.
///
/// `f32` is not `Ord`, and the ordering has to be total for a heap to be
/// correct rather than merely usually correct. Ties break on the *lower* index
/// so that a tie resolves the same way every run — two candidates with identical
/// simulated scores are genuinely interchangeable, and without this the choice
/// between them would be left to heap internals.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    gain: f32,
    candidate: usize,
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gain
            .partial_cmp(&other.gain)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.candidate.cmp(&self.candidate))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Select a portfolio from a scored candidate pool.
///
/// Returns candidate indices in the order they were chosen, which is by
/// decreasing marginal contribution — so a caller who wants a smaller portfolio
/// can truncate rather than re-run.
pub fn select_portfolio(
    candidates: &Candidates<'_>,
    config: &SelectConfig,
) -> Result<Vec<u32>, SelectError> {
    candidates.validate(config)?;
    let n = candidates.len();
    if n == 0 || config.n_select == 0 {
        return Ok(Vec::new());
    }

    let capping = !config.exposure_limits.is_empty();
    let mut appearances = vec![0u32; config.exposure_limits.len()];
    let mut bar = vec![config.objective.floor(); candidates.n_outcomes];
    let inverse = 1.0 / candidates.n_outcomes as f32;

    // Seed the heap with every candidate's standalone gain. These are upper
    // bounds from here on: the bar only rises, so no gain can grow.
    let mut heap: BinaryHeap<Entry> = (0..n)
        .map(|candidate| Entry {
            gain: config.objective.gain(candidates.row(candidate), &bar) * inverse,
            candidate,
        })
        .collect();

    let mut taken = vec![false; n];
    let mut chosen: Vec<u32> = Vec::with_capacity(config.n_select.min(n));

    while chosen.len() < config.n_select {
        // Pop until the top of the heap is a gain recomputed against the current
        // bar. Anything stale is re-costed and pushed back; because gains only
        // shrink, a stale entry that still tops the heap after recomputation is
        // genuinely the best available.
        let best = loop {
            let Some(entry) = heap.pop() else {
                break None;
            };
            if taken[entry.candidate] {
                continue;
            }
            if capping
                && exceeds_exposure(
                    candidates,
                    &config.exposure_limits,
                    &appearances,
                    entry.candidate,
                )
            {
                // Dropped rather than pushed back: caps only tighten, so a
                // candidate blocked now is blocked for the rest of the run.
                continue;
            }
            let fresh = Entry {
                gain: config.objective.gain(candidates.row(entry.candidate), &bar) * inverse,
                candidate: entry.candidate,
            };
            // Still at least as good as the best remaining *bound*, so nothing
            // under it can overtake it and it can be taken.
            //
            // Compared with the full `Entry` order rather than the gain alone,
            // which matters more than it looks: `Cover` gains are counts of
            // outcomes, so exact ties are the norm rather than the exception. On
            // a tie the bare-float test accepted whichever candidate happened to
            // be popped first, while the naive loop kept the lowest index — the
            // two produced different portfolios, and only the parity test caught
            // it.
            if heap.peek().is_none_or(|next| fresh >= *next) {
                break Some(fresh);
            }
            heap.push(fresh);
        };

        let Some(best) = best else { break };
        // A gain of zero means this candidate wins in no outcome the portfolio
        // does not already win. Taking it would pad the entry count without
        // improving anything, so stopping is the honest answer.
        if best.gain <= config.min_gain && !chosen.is_empty() {
            break;
        }

        taken[best.candidate] = true;
        config
            .objective
            .absorb(&mut bar, candidates.row(best.candidate));
        if capping {
            for &player in candidates.roster(best.candidate) {
                appearances[player as usize] += 1;
            }
        }
        chosen.push(best.candidate as u32);
    }

    Ok(chosen)
}

/// Whether taking `candidate` would put any of its players over their cap.
fn exceeds_exposure(
    candidates: &Candidates<'_>,
    limits: &[u32],
    appearances: &[u32],
    candidate: usize,
) -> bool {
    candidates
        .roster(candidate)
        .iter()
        .any(|&player| appearances[player as usize] >= limits[player as usize])
}

/// The score at a given quantile of a flat matrix of simulated outcomes.
///
/// A starting point for the line an [`Objective`] needs, when nothing better is
/// known: the 99th percentile of what the candidate pool itself produces is a
/// reasonable stand-in for "a score that wins a tournament".
///
/// It is only a stand-in. The score that actually wins is a property of the
/// *field* — how many entries, built how — and a pool of your own candidates is
/// not that. If you have a field model, its quantile is the number to use.
pub fn quantile(scores: &[f32], q: f64) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = scores.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let clamped = q.clamp(0.0, 1.0);
    // Nearest-rank, which needs no interpolation and cannot invent a score no
    // simulation produced.
    let rank = (clamped * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// The value of a portfolio under the selection objective.
///
/// Exposed because it is what a caller compares two portfolios with, and
/// recomputing it by hand invites a subtly different definition.
pub fn portfolio_value(candidates: &Candidates<'_>, chosen: &[u32], threshold: f32) -> f32 {
    if candidates.n_outcomes == 0 {
        return 0.0;
    }
    let mut bar = vec![threshold; candidates.n_outcomes];
    for &candidate in chosen {
        raise_to(&mut bar, candidates.row(candidate as usize));
    }
    let total: f32 = bar.iter().map(|&b| b - threshold).sum();
    total / candidates.n_outcomes as f32
}

/// The naive greedy, kept for tests.
///
/// Recomputes every candidate's gain every round. Lazy evaluation must agree
/// with this exactly — that is the whole claim it makes — and the only way to
/// assert it is to have both.
pub fn select_portfolio_naive(
    candidates: &Candidates<'_>,
    config: &SelectConfig,
) -> Result<Vec<u32>, SelectError> {
    candidates.validate(config)?;
    let n = candidates.len();
    if n == 0 || config.n_select == 0 {
        return Ok(Vec::new());
    }

    let capping = !config.exposure_limits.is_empty();
    let mut appearances = vec![0u32; config.exposure_limits.len()];
    let mut bar = vec![config.objective.floor(); candidates.n_outcomes];
    let inverse = 1.0 / candidates.n_outcomes as f32;
    let mut taken = vec![false; n];
    let mut chosen: Vec<u32> = Vec::new();

    while chosen.len() < config.n_select {
        let mut best: Option<Entry> = None;
        for (candidate, &already) in taken.iter().enumerate() {
            if already
                || (capping
                    && exceeds_exposure(
                        candidates,
                        &config.exposure_limits,
                        &appearances,
                        candidate,
                    ))
            {
                continue;
            }
            let gain = config.objective.gain(candidates.row(candidate), &bar) * inverse;
            let entry = Entry { gain, candidate };
            if best.is_none_or(|current| entry > current) {
                best = Some(entry);
            }
        }

        let Some(best) = best else { break };
        if best.gain <= config.min_gain && !chosen.is_empty() {
            break;
        }
        taken[best.candidate] = true;
        config
            .objective
            .absorb(&mut bar, candidates.row(best.candidate));
        if capping {
            for &player in candidates.roster(best.candidate) {
                appearances[player as usize] += 1;
            }
        }
        chosen.push(best.candidate as u32);
    }
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three candidates over four outcomes. The first is best on average; the
    /// third is worse on average but wins the outcomes the first loses.
    fn pool() -> Vec<f32> {
        vec![
            10.0, 10.0, 10.0, 10.0, // candidate 0: steady
            9.0, 9.0, 9.0, 9.0, // candidate 1: strictly worse than 0
            0.0, 0.0, 40.0, 0.0, // candidate 2: spiky
        ]
    }

    fn candidates(scores: &[f32]) -> Candidates<'_> {
        Candidates {
            scores,
            n_outcomes: 4,
            rosters: &[],
            roster_size: 0,
        }
    }

    #[test]
    fn the_best_standalone_lineup_is_taken_first() {
        let scores = pool();
        let config = SelectConfig {
            n_select: 1,
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&candidates(&scores), &config).unwrap(),
            [0]
        );
    }

    #[test]
    fn a_complementary_lineup_beats_a_better_average_one() {
        // The property that makes this a portfolio objective rather than a
        // ranking. Candidate 1 has the higher mean of the two remaining, but
        // adds nothing the portfolio does not already have; candidate 2 is worse
        // on average and wins an outcome outright.
        let scores = pool();
        let config = SelectConfig {
            n_select: 2,
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&candidates(&scores), &config).unwrap(),
            [0, 2]
        );
    }

    #[test]
    fn a_dominated_lineup_is_never_taken() {
        // Candidate 1 is worse than candidate 0 at every outcome, so once 0 is
        // held its gain is exactly zero and selection stops rather than padding.
        let scores = pool();
        let config = SelectConfig {
            n_select: 3,
            ..SelectConfig::default()
        };
        let chosen = select_portfolio(&candidates(&scores), &config).unwrap();
        assert_eq!(chosen, [0, 2], "a zero-gain lineup was entered");
    }

    #[test]
    fn a_duplicate_pool_yields_one_lineup() {
        // Diversity without a diversity term: fifty copies of one lineup are
        // worth exactly one entry.
        let scores: Vec<f32> = std::iter::repeat_n([5.0f32, 6.0, 7.0, 8.0], 50)
            .flatten()
            .collect();
        let config = SelectConfig {
            n_select: 20,
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&candidates(&scores), &config)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn lazy_and_naive_agree_exactly() {
        // The only claim lazy evaluation makes. Randomized over many shapes,
        // because the interesting failures are ordering ones that a single
        // hand-built pool will not produce.
        use rand::{RngExt, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5EED);
        for n_cand in [1usize, 2, 5, 17, 40] {
            for n_outcomes in [1usize, 3, 8, 11, 32] {
                let scores: Vec<f32> = (0..n_cand * n_outcomes)
                    .map(|_| rng.random_range(0.0..100.0f32))
                    .collect();
                let pool = Candidates {
                    scores: &scores,
                    n_outcomes,
                    rosters: &[],
                    roster_size: 0,
                };
                for n_select in [1usize, 3, 10, n_cand] {
                    let config = SelectConfig {
                        n_select,
                        ..SelectConfig::default()
                    };
                    assert_eq!(
                        select_portfolio(&pool, &config).unwrap(),
                        select_portfolio_naive(&pool, &config).unwrap(),
                        "n_cand={n_cand} n_outcomes={n_outcomes} n_select={n_select}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_threshold_ignores_outcomes_below_it() {
        // Candidate 2 spikes to 40 in one outcome and is worthless elsewhere;
        // candidate 0 is a steady 10. Above a line of 20, only the spike counts,
        // so the spiky lineup is taken first.
        let scores = pool();
        let config = SelectConfig {
            n_select: 1,
            objective: Objective::Excess { threshold: 20.0 },
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&candidates(&scores), &config).unwrap(),
            [2]
        );
    }

    #[test]
    fn gains_are_non_increasing_which_is_what_lazy_evaluation_relies_on() {
        use rand::{RngExt, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let (n_cand, n_outcomes) = (30usize, 16usize);
        let scores: Vec<f32> = (0..n_cand * n_outcomes)
            .map(|_| rng.random_range(0.0..100.0f32))
            .collect();
        let pool = Candidates {
            scores: &scores,
            n_outcomes,
            rosters: &[],
            roster_size: 0,
        };
        let chosen = select_portfolio(
            &pool,
            &SelectConfig {
                n_select: n_cand,
                ..SelectConfig::default()
            },
        )
        .unwrap();

        // Submodularity, observed: the value added by each successive pick must
        // not exceed the one before it. If this ever fails the heap's bounds are
        // not bounds and the lazy result is arbitrary.
        let mut previous = f32::INFINITY;
        let mut running = 0.0f32;
        for k in 1..=chosen.len() {
            let value = portfolio_value(&pool, &chosen[..k], 0.0);
            let gain = value - running;
            assert!(
                gain <= previous + 1e-4,
                "gain rose at pick {k}: {gain} after {previous}"
            );
            previous = gain;
            running = value;
        }
    }

    #[test]
    fn exposure_caps_bound_appearances() {
        // Four candidates over two players each; player 0 appears in three of
        // them but may be entered only once.
        let scores: Vec<f32> = vec![
            10.0, 1.0, //
            9.0, 2.0, //
            8.0, 3.0, //
            1.0, 9.0, //
        ];
        let rosters: Vec<u32> = vec![0, 1, 0, 2, 0, 3, 4, 5];
        let pool = Candidates {
            scores: &scores,
            n_outcomes: 2,
            rosters: &rosters,
            roster_size: 2,
        };
        let config = SelectConfig {
            n_select: 4,
            exposure_limits: vec![1, 9, 9, 9, 9, 9],
            ..SelectConfig::default()
        };
        let chosen = select_portfolio(&pool, &config).unwrap();
        let uses = chosen
            .iter()
            .filter(|&&c| pool.roster(c as usize).contains(&0))
            .count();
        assert_eq!(uses, 1, "player 0 exceeded a cap of one: {chosen:?}");
    }

    #[test]
    fn exposure_caps_agree_between_lazy_and_naive() {
        let scores: Vec<f32> = vec![10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 1.0, 9.0];
        let rosters: Vec<u32> = vec![0, 1, 0, 2, 0, 3, 4, 5];
        let pool = Candidates {
            scores: &scores,
            n_outcomes: 2,
            rosters: &rosters,
            roster_size: 2,
        };
        let config = SelectConfig {
            n_select: 4,
            exposure_limits: vec![1, 9, 9, 9, 9, 9],
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&pool, &config).unwrap(),
            select_portfolio_naive(&pool, &config).unwrap()
        );
    }

    #[test]
    fn selection_is_deterministic_under_ties() {
        // Every candidate identical: the tie-break must pick the lowest index
        // rather than whatever the heap happens to surface.
        let scores: Vec<f32> = std::iter::repeat_n([1.0f32, 2.0], 6).flatten().collect();
        let pool = Candidates {
            scores: &scores,
            n_outcomes: 2,
            rosters: &[],
            roster_size: 0,
        };
        let config = SelectConfig {
            n_select: 1,
            ..SelectConfig::default()
        };
        assert_eq!(select_portfolio(&pool, &config).unwrap(), [0]);
    }

    #[test]
    fn portfolio_value_is_the_mean_of_the_running_maximum() {
        let scores = pool();
        let c = candidates(&scores);
        // Candidates 0 and 2 together: max is (10, 10, 40, 10), mean 17.5.
        assert_eq!(portfolio_value(&c, &[0, 2], 0.0), 17.5);
        assert_eq!(portfolio_value(&c, &[], 0.0), 0.0);
    }

    #[test]
    fn an_empty_pool_selects_nothing() {
        let pool = Candidates {
            scores: &[],
            n_outcomes: 4,
            rosters: &[],
            roster_size: 0,
        };
        assert!(select_portfolio(&pool, &SelectConfig::default())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn selecting_zero_returns_nothing() {
        let scores = pool();
        let config = SelectConfig {
            n_select: 0,
            ..SelectConfig::default()
        };
        assert!(select_portfolio(&candidates(&scores), &config)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_pool_with_no_outcomes_is_rejected() {
        let pool = Candidates {
            scores: &[1.0],
            n_outcomes: 0,
            rosters: &[],
            roster_size: 0,
        };
        assert_eq!(
            select_portfolio(&pool, &SelectConfig::default()),
            Err(SelectError::NoOutcomes)
        );
    }

    #[test]
    fn a_ragged_score_matrix_is_rejected() {
        let pool = Candidates {
            scores: &[1.0, 2.0, 3.0],
            n_outcomes: 2,
            rosters: &[],
            roster_size: 0,
        };
        assert_eq!(
            select_portfolio(&pool, &SelectConfig::default()),
            Err(SelectError::RaggedScores {
                len: 3,
                n_outcomes: 2
            })
        );
    }

    #[test]
    fn an_exposure_column_shorter_than_the_pool_is_rejected() {
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let rosters: Vec<u32> = vec![0, 5, 1, 2];
        let pool = Candidates {
            scores: &scores,
            n_outcomes: 2,
            rosters: &rosters,
            roster_size: 2,
        };
        let config = SelectConfig {
            exposure_limits: vec![1, 1],
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&pool, &config),
            Err(SelectError::ExposureLengthMismatch {
                len: 2,
                expected: 6
            })
        );
    }

    #[test]
    fn a_roster_matrix_for_another_pool_is_rejected() {
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let rosters: Vec<u32> = vec![0, 1];
        let pool = Candidates {
            scores: &scores,
            n_outcomes: 2,
            rosters: &rosters,
            roster_size: 2,
        };
        let config = SelectConfig {
            exposure_limits: vec![9, 9],
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&pool, &config),
            Err(SelectError::CandidateCountMismatch {
                rosters: 1,
                scores: 2
            })
        );
    }

    // --- Contest modes ----------------------------------------------------

    /// Four candidates over eight outcomes, built so cash and tournament want
    /// different answers. Candidate 0 clears 100 in six outcomes but never gets
    /// near 200. Candidate 1 is a lottery ticket: it clears 200 twice and is
    /// worthless otherwise. Candidate 2 duplicates 0's coverage. Candidate 3
    /// clears 200 in the same two outcomes as candidate 1.
    fn contest_pool() -> Vec<f32> {
        vec![
            120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 10.0, 10.0, // 0: steady casher
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 250.0, 250.0, // 1: tail spike
            118.0, 118.0, 118.0, 118.0, 118.0, 118.0, 9.0, 9.0, // 2: near-copy of 0
            9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 240.0, 240.0, // 3: same tail as 1
        ]
    }

    fn contest_candidates(scores: &[f32]) -> Candidates<'_> {
        Candidates {
            scores,
            n_outcomes: 8,
            rosters: &[],
            roster_size: 0,
        }
    }

    #[test]
    fn cash_mode_ranks_by_probability_of_clearing_the_line() {
        // Cash pays flat for beating 100, so the steady lineups win and the
        // lottery tickets are worthless — the opposite of the tournament answer
        // below, from the same pool.
        let scores = contest_pool();
        let config = SelectConfig {
            n_select: 2,
            objective: Objective::Cash { line: 100.0 },
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&contest_candidates(&scores), &config).unwrap(),
            [0, 2]
        );
    }

    #[test]
    fn cash_mode_does_not_penalize_overlap() {
        // Candidate 2 covers exactly the outcomes candidate 0 already covers. In
        // a portfolio objective it would be worthless; in cash it cashes a second
        // time and is correctly taken second.
        let scores = contest_pool();
        let config = SelectConfig {
            n_select: 2,
            objective: Objective::Cash { line: 100.0 },
            ..SelectConfig::default()
        };
        let chosen = select_portfolio(&contest_candidates(&scores), &config).unwrap();
        assert!(chosen.contains(&2), "cash mode should not fade a near-copy");
    }

    #[test]
    fn cash_mode_gains_do_not_decay() {
        // Modular, not submodular: an entry is worth the same whenever it is
        // taken. Worth pinning, because it is why greedy is exactly optimal here
        // rather than a (1 - 1/e) approximation.
        let scores = contest_pool();
        let pool = contest_candidates(&scores);
        let objective = Objective::Cash { line: 100.0 };
        let mut bar = vec![objective.floor(); 8];
        let before = objective.gain(pool.row(2), &bar);
        objective.absorb(&mut bar, pool.row(0));
        assert_eq!(objective.gain(pool.row(2), &bar), before);
    }

    #[test]
    fn tournament_mode_takes_the_lottery_ticket_over_the_steady_lineup() {
        // Above a line of 200, the steady casher never scores at all. Only the
        // tail matters, which is the whole point of the mode.
        let scores = contest_pool();
        let config = SelectConfig {
            n_select: 1,
            objective: Objective::Cover { line: 200.0 },
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&contest_candidates(&scores), &config).unwrap(),
            [1]
        );
    }

    #[test]
    fn tournament_mode_will_not_pay_twice_for_the_same_tail() {
        // Candidate 3 clears the line in exactly the outcomes candidate 1 already
        // covers, so it adds nothing and selection stops at one entry.
        let scores = contest_pool();
        let config = SelectConfig {
            n_select: 4,
            objective: Objective::Cover { line: 200.0 },
            ..SelectConfig::default()
        };
        assert_eq!(
            select_portfolio(&contest_candidates(&scores), &config).unwrap(),
            [1]
        );
    }

    #[test]
    fn the_two_modes_disagree_on_the_same_pool() {
        // The reason both exist. If these ever coincide the modes have collapsed
        // into each other and one of them is not doing its job.
        let scores = contest_pool();
        let pool = contest_candidates(&scores);
        let cash = select_portfolio(
            &pool,
            &SelectConfig {
                n_select: 1,
                objective: Objective::Cash { line: 100.0 },
                ..SelectConfig::default()
            },
        )
        .unwrap();
        let gpp = select_portfolio(
            &pool,
            &SelectConfig {
                n_select: 1,
                objective: Objective::Cover { line: 200.0 },
                ..SelectConfig::default()
            },
        )
        .unwrap();
        assert_ne!(cash, gpp);
    }

    #[test]
    fn cover_counts_the_fraction_of_outcomes_reached() {
        // Two of eight outcomes cleared, so a portfolio value of 0.25.
        let scores = contest_pool();
        let pool = contest_candidates(&scores);
        let objective = Objective::Cover { line: 200.0 };
        let mut bar = vec![objective.floor(); 8];
        let gain = objective.gain(pool.row(1), &bar) / 8.0;
        assert_eq!(gain, 0.25);
        objective.absorb(&mut bar, pool.row(1));
        assert_eq!(objective.gain(pool.row(3), &bar), 0.0);
    }

    #[test]
    fn a_score_exactly_on_the_line_counts_as_clearing_it() {
        let scores = [100.0f32, 99.0];
        let pool = Candidates {
            scores: &scores,
            n_outcomes: 2,
            rosters: &[],
            roster_size: 0,
        };
        let objective = Objective::Cover { line: 100.0 };
        let bar = vec![objective.floor(); 2];
        assert_eq!(objective.gain(pool.row(0), &bar), 1.0);
    }

    #[test]
    fn every_mode_agrees_between_lazy_and_naive() {
        use rand::{RngExt, SeedableRng};
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA11CE);
        for objective in [
            Objective::Excess { threshold: 0.0 },
            Objective::Excess { threshold: 50.0 },
            Objective::Cover { line: 60.0 },
            Objective::Cash { line: 40.0 },
        ] {
            for n_cand in [1usize, 4, 19] {
                for n_outcomes in [1usize, 7, 24] {
                    let scores: Vec<f32> = (0..n_cand * n_outcomes)
                        .map(|_| rng.random_range(0.0..100.0f32))
                        .collect();
                    let pool = Candidates {
                        scores: &scores,
                        n_outcomes,
                        rosters: &[],
                        roster_size: 0,
                    };
                    let config = SelectConfig {
                        n_select: n_cand,
                        objective,
                        ..SelectConfig::default()
                    };
                    assert_eq!(
                        select_portfolio(&pool, &config).unwrap(),
                        select_portfolio_naive(&pool, &config).unwrap(),
                        "{objective:?} n_cand={n_cand} n_outcomes={n_outcomes}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_quantile_reads_off_the_sorted_scores() {
        let scores = [5.0f32, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(quantile(&scores, 0.0), 1.0);
        assert_eq!(quantile(&scores, 1.0), 5.0);
        assert_eq!(quantile(&scores, 0.5), 3.0);
        // Out-of-range quantiles clamp rather than panic.
        assert_eq!(quantile(&scores, 2.0), 5.0);
        assert_eq!(quantile(&[], 0.9), 0.0);
    }

    #[test]
    fn scoring_sums_a_lineup_across_the_universe() {
        // Two players, three outcomes.
        let universe = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let scored = score_lineups(&universe, 3, &[0, 1], 2, &[1.0, 1.0]).unwrap();
        assert_eq!(scored, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn scoring_applies_the_slot_multiplier() {
        // A captain slot is worth 1.5x here as it is everywhere else, or a
        // showdown portfolio would be ranked on the wrong numbers.
        let universe = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let scored = score_lineups(&universe, 3, &[0, 1], 2, &[1.5, 1.0]).unwrap();
        assert_eq!(scored, vec![11.5, 23.0, 34.5]);
    }

    #[test]
    fn scoring_handles_several_lineups() {
        let universe = [1.0f32, 2.0, 10.0, 20.0, 100.0, 200.0];
        let scored = score_lineups(&universe, 2, &[0, 1, 1, 2], 2, &[1.0, 1.0]).unwrap();
        assert_eq!(scored, vec![11.0, 22.0, 110.0, 220.0]);
    }

    #[test]
    fn scoring_rejects_a_multiplier_list_of_the_wrong_length() {
        let universe = [1.0f32, 2.0];
        assert!(score_lineups(&universe, 2, &[0], 1, &[1.0, 1.0]).is_err());
    }

    #[test]
    fn scoring_rejects_a_ragged_roster_matrix() {
        let universe = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(
            score_lineups(&universe, 2, &[0, 1, 0], 2, &[1.0, 1.0]),
            Err(SelectError::RaggedRosters {
                len: 3,
                roster_size: 2
            })
        );
    }

    #[test]
    fn every_error_renders_a_message() {
        let errors = [
            SelectError::RaggedScores {
                len: 3,
                n_outcomes: 2,
            },
            SelectError::NoOutcomes,
            SelectError::ExposureLengthMismatch {
                len: 1,
                expected: 2,
            },
            SelectError::RaggedRosters {
                len: 3,
                roster_size: 2,
            },
            SelectError::CandidateCountMismatch {
                rosters: 1,
                scores: 2,
            },
        ];
        for error in errors {
            assert!(!error.to_string().is_empty(), "{error:?} renders empty");
        }
    }
}
