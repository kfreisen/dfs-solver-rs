//! Runtime-dispatched SIMD kernels.
//!
//! Wheels are built portably — no `-C target-cpu=native` — so the ISA is chosen
//! per-process by feature detection rather than at compile time. Every vector
//! kernel here is paired with a scalar reference that produces a *bit-identical*
//! result, which is what makes the parity tests exact rather than approximate.
//!
//! Bit-identity is not free for floating-point reductions: summing eight lanes in
//! parallel and folding at the end rounds differently from a left-to-right scalar
//! sum. The scalar reference therefore reproduces the vector accumulator layout
//! deliberately (see [`sum_positive_excess_scalar`]). Getting a matching answer out
//! of a naive `iter().sum()` would require a tolerance, and a tolerance hides
//! exactly the lane-handling bugs these tests exist to catch.

/// Number of `f32` lanes in one AVX2 register. The scalar reference mirrors this.
const LANES: usize = 8;

/// Returns the instruction set this process will use.
pub fn active_isa() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    "scalar"
}

/// Sums `max(score - threshold, 0)` over `scores`.
///
/// This is the CVaR-upside inner loop: only outcomes above the threshold
/// contribute, and the contribution is the excess. Dispatches to AVX2 where
/// available.
pub fn sum_positive_excess(scores: &[f32], threshold: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime feature check immediately above.
            return unsafe { sum_positive_excess_avx2(scores, threshold) };
        }
    }
    sum_positive_excess_scalar(scores, threshold)
}

/// Scalar reference for [`sum_positive_excess`].
///
/// Accumulates into [`LANES`] partial sums in the same striping order the vector
/// kernel uses, then folds them identically, so both paths produce the same bits.
pub fn sum_positive_excess_scalar(scores: &[f32], threshold: f32) -> f32 {
    let mut acc = [0.0f32; LANES];
    let chunks = scores.chunks_exact(LANES);
    let tail = chunks.remainder();

    for chunk in chunks {
        for lane in 0..LANES {
            let excess = chunk[lane] - threshold;
            acc[lane] += if excess > 0.0 { excess } else { 0.0 };
        }
    }
    // The vector kernel folds its accumulator before handling the tail, so the
    // reference must too.
    let mut total = fold(acc);
    for &score in tail {
        let excess = score - threshold;
        if excess > 0.0 {
            total += excess;
        }
    }
    total
}

/// Sums `max(score - bar, 0)` elementwise against a per-outcome bar.
///
/// The selection inner loop. Where [`sum_positive_excess`] compares every
/// outcome against one constant, this compares each against the portfolio's
/// current best in that outcome — which is what makes the result the *marginal*
/// gain of adding a lineup rather than its standalone value. A candidate that
/// duplicates one already held scores zero here on every outcome, which is how
/// the objective produces diversity without anyone asking it to.
///
/// Reads `bar.len()` outcomes; `scores` must be at least as long.
pub fn sum_excess_over(scores: &[f32], bar: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime feature check immediately above.
            return unsafe { sum_excess_over_avx2(scores, bar) };
        }
    }
    sum_excess_over_scalar(scores, bar)
}

/// Scalar reference for [`sum_excess_over`], striped to match the vector kernel.
pub fn sum_excess_over_scalar(scores: &[f32], bar: &[f32]) -> f32 {
    let n = bar.len();
    let scores = &scores[..n];
    let mut acc = [0.0f32; LANES];
    let full = n - n % LANES;

    for start in (0..full).step_by(LANES) {
        for lane in 0..LANES {
            let excess = scores[start + lane] - bar[start + lane];
            acc[lane] += if excess > 0.0 { excess } else { 0.0 };
        }
    }
    let mut total = fold(acc);
    for i in full..n {
        let excess = scores[i] - bar[i];
        if excess > 0.0 {
            total += excess;
        }
    }
    total
}

/// Raises each entry of `bar` to `scores`, elementwise.
///
/// How the portfolio's per-outcome best is updated once a lineup is taken.
pub fn raise_to(bar: &mut [f32], scores: &[f32]) {
    for (b, &s) in bar.iter_mut().zip(scores) {
        if s > *b {
            *b = s;
        }
    }
}

/// Folds lane accumulators into a scalar, in the same pairwise order as the
/// AVX2 horizontal reduction below.
#[inline]
fn fold(acc: [f32; LANES]) -> f32 {
    let a = [
        acc[0] + acc[4],
        acc[1] + acc[5],
        acc[2] + acc[6],
        acc[3] + acc[7],
    ];
    let b = [a[0] + a[2], a[1] + a[3]];
    b[0] + b[1]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_positive_excess_avx2(scores: &[f32], threshold: f32) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps, _mm256_sub_ps,
    };

    let thresh = _mm256_set1_ps(threshold);
    let zero = _mm256_setzero_ps();
    let mut acc_v = _mm256_setzero_ps();

    let chunks = scores.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        let v = _mm256_loadu_ps(chunk.as_ptr());
        // max(v - threshold, 0) — branchless, and NaN-consistent with the scalar
        // path because `_mm256_max_ps` returns the second operand for NaN inputs
        // and `excess > 0.0` is false for NaN.
        acc_v = _mm256_add_ps(acc_v, _mm256_max_ps(_mm256_sub_ps(v, thresh), zero));
    }

    let mut acc = [0.0f32; LANES];
    _mm256_storeu_ps(acc.as_mut_ptr(), acc_v);
    let mut total = fold(acc);

    for &score in tail {
        let excess = score - threshold;
        if excess > 0.0 {
            total += excess;
        }
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_excess_over_avx2(scores: &[f32], bar: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };

    let n = bar.len();
    let zero = _mm256_setzero_ps();
    let mut acc_v = _mm256_setzero_ps();
    let full = n - n % LANES;

    for start in (0..full).step_by(LANES) {
        let v = _mm256_loadu_ps(scores.as_ptr().add(start));
        let b = _mm256_loadu_ps(bar.as_ptr().add(start));
        acc_v = _mm256_add_ps(acc_v, _mm256_max_ps(_mm256_sub_ps(v, b), zero));
    }

    let mut acc = [0.0f32; LANES];
    _mm256_storeu_ps(acc.as_mut_ptr(), acc_v);
    let mut total = fold(acc);

    for i in full..n {
        let excess = scores[i] - bar[i];
        if excess > 0.0 {
            total += excess;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    // rand 0.10 moved the range/sampling helpers off `Rng` onto `RngExt`.
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(sum_positive_excess(&[], 1.0), 0.0);
    }

    #[test]
    fn only_values_above_threshold_contribute() {
        let scores = [1.0, 2.0, 3.0, 4.0];
        // 3 and 4 exceed 2.5, contributing 0.5 and 1.5.
        assert_eq!(sum_positive_excess_scalar(&scores, 2.5), 2.0);
    }

    #[test]
    fn values_exactly_at_threshold_do_not_contribute() {
        assert_eq!(sum_positive_excess_scalar(&[2.0, 2.0], 2.0), 0.0);
    }

    /// The test that protects the portable-wheel change: every tail length from
    /// zero through several full vectors must agree bit-for-bit.
    #[test]
    fn avx2_matches_scalar_bit_for_bit_at_every_length() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE);
        for len in 0..=64usize {
            let scores: Vec<f32> = (0..len).map(|_| rng.random_range(-50.0..150.0)).collect();
            for &threshold in &[-10.0f32, 0.0, 37.5, 1000.0] {
                let expected = sum_positive_excess_scalar(&scores, threshold);
                let actual = sum_positive_excess(&scores, threshold);
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "len={len} threshold={threshold} dispatched={} scalar={expected} got={actual}",
                    active_isa(),
                );
            }
        }
    }

    #[test]
    fn excess_over_a_bar_is_elementwise() {
        let scores = [1.0, 5.0, 3.0];
        let bar = [0.0, 6.0, 3.0];
        // 1 beats 0 by one; 5 loses to 6; 3 ties 3 and ties do not contribute.
        assert_eq!(sum_excess_over_scalar(&scores, &bar), 1.0);
    }

    #[test]
    fn a_duplicate_lineup_gains_nothing() {
        // The property the whole selection objective rests on: a candidate whose
        // outcomes the portfolio already matches contributes exactly zero, which
        // is why no separate diversity penalty is needed.
        let scores = [10.0, 20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(sum_excess_over_scalar(&scores, &scores), 0.0);
    }

    #[test]
    fn excess_over_a_bar_matches_avx2_bit_for_bit_at_every_length() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF);
        for len in 0..=64usize {
            let scores: Vec<f32> = (0..len).map(|_| rng.random_range(-50.0..150.0)).collect();
            let bar: Vec<f32> = (0..len).map(|_| rng.random_range(-50.0..150.0)).collect();
            let expected = sum_excess_over_scalar(&scores, &bar);
            let actual = sum_excess_over(&scores, &bar);
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "len={len} dispatched={}",
                active_isa(),
            );
        }
    }

    #[test]
    fn raising_the_bar_takes_the_elementwise_maximum() {
        let mut bar = [1.0f32, 5.0, 3.0];
        raise_to(&mut bar, &[2.0, 4.0, 3.0]);
        assert_eq!(bar, [2.0, 5.0, 3.0]);
    }

    #[test]
    fn active_isa_is_one_of_the_known_paths() {
        assert!(matches!(active_isa(), "avx2" | "scalar"));
    }
}
