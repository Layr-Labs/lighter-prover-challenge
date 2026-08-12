//! NEON external linear layer (`M_E`) for the width-12 Goldilocks Poseidon2.
//!
//! The scalar path widens each of the twelve state words to `u128`, runs the
//! whole `M_E` add graph in 128-bit arithmetic and reduces each result with
//! `reduce128_with_96_bits`. That needs twenty-four general-purpose registers
//! for the state alone, so it spills, and every 128-bit add costs an
//! `adds`/`adcs` pair.
//!
//! `M_E` has only tiny coefficients, so the extra width is almost entirely
//! wasted. Writing each state word as `x = lo + 2**32 * hi` with
//! `lo, hi < 2**32`, the matrix product is
//!
//! ```ignore
//! V_i = sum_j c_ij * x_j = (sum_j c_ij * lo_j) + 2**32 * (sum_j c_ij * hi_j)
//!     = L_i + 2**32 * H_i
//! ```
//!
//! and every row of `M_E` has coefficient sum 28 — each `M_4` row sums to 7 and
//! the outer circulant adds one row of the own block twice and one row of each
//! other block once — so `L_i, H_i <= 28 * (2**32 - 1) < 2**36.81`. Both halves fit a
//! 64-bit lane with 27 bits to spare, and no carry ever has to cross from `L`
//! to `H` while the add graph runs.
//!
//! So each state word lives in one `uint64x2_t` as `[lo, hi]`, one `vaddq_u64`
//! does the work of an `adds`/`adcs` pair, and twelve vector registers hold the
//! entire state. The lane split is undone once, right before the reduction,
//! which reconstructs `V_i = L_i + 2**32 * H_i` exactly and then reproduces
//! `reduce128_with_96_bits` operation for operation — the result is the same
//! non-canonical `u64` representative as the scalar path, not merely a
//! congruent one.

use core::arch::aarch64::*;

use static_assertions::const_assert;

use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::poseidon2::config::WIDTH;

const EPSILON: u64 = 0xffffffff;

// The block decomposition below is written out for exactly three `M_4` blocks;
// fail to compile rather than silently mismatch a re-parameterised Poseidon2.
const_assert!(WIDTH == 12);

/// Load `[x_i, x_{i+1}]` and return them split as `[lo_i, hi_i]` and
/// `[lo_j, hi_j]`.
///
/// Reinterpreting the loaded pair as four `u32` lanes already lays the halves
/// out as `[lo_i, hi_i, lo_j, hi_j]`, so the split is one `ushll`/`ushll2` pair
/// rather than a mask and a shift per element.
///
/// # Safety
/// `p` must point to two readable, 8-byte-aligned `u64`s.
#[inline(always)]
unsafe fn split_pair(p: *const u64) -> (uint64x2_t, uint64x2_t) {
    let packed = vreinterpretq_u32_u64(vld1q_u64(p));
    (
        vmovl_u32(vget_low_u32(packed)),
        vmovl_u32(vget_high_u32(packed)),
    )
}

/// Multiply a four-element block by
/// ```ignore
/// [ 2 3 1 1 ]
/// [ 1 2 3 1 ]
/// [ 1 1 2 3 ]
/// [ 3 1 1 2 ]
/// ```
/// using the same add sequence as the scalar `external_linear_layer_u128`.
/// Lanes hold exact integers here, so the sequence only has to agree with the
/// scalar one as integer arithmetic, which it does term for term.
#[inline(always)]
unsafe fn apply_mat4(x: &mut [uint64x2_t; 4]) {
    let t01 = vaddq_u64(x[0], x[1]);
    let t23 = vaddq_u64(x[2], x[3]);
    let t0123 = vaddq_u64(t01, t23);

    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let x3 = x[3];

    x[0] = vaddq_u64(vaddq_u64(t0123, t01), x1); // 2*x0 + 3*x1 +   x2 +   x3
    x[1] = vaddq_u64(vaddq_u64(t0123, x1), vaddq_u64(x2, x2)); //   x0 + 2*x1 + 3*x2 +   x3
    x[2] = vaddq_u64(vaddq_u64(t0123, t23), x3); //   x0 +   x1 + 2*x2 + 3*x3
    x[3] = vaddq_u64(vaddq_u64(t0123, x3), vaddq_u64(x0, x0)); // 3*x0 +   x1 +   x2 + 2*x3
}

/// Reduce two accumulators to two field elements.
///
/// `a` is `[L_i, H_i]` and `b` is `[L_j, H_j]`, each half below `2**37`. The
/// transposes regroup them into `[L_i, L_j]` and `[H_i, H_j]`, after which the
/// two elements reduce together.
///
/// `V = L + 2**32 * H` is at most `28 * (2**64 - 1) < 2**68.81`, so its high
/// word is at most 27 and `t1 = V_hi * EPSILON < 2**36.76`. `V_lo + t1` is
/// therefore below `2**64 + ORDER`, which is exactly the precondition of
/// `add_no_canonicalize_trashing_input`; the single conditional `EPSILON` fold
/// at the end cannot itself overflow because a carry there leaves `s < t1`.
#[inline(always)]
unsafe fn reduce_pair(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let l = vtrn1q_u64(a, b);
    let h = vtrn2q_u64(a, b);

    // Low 64 bits of V, plus the carry out of bit 63 as an all-ones mask.
    let v_lo = vaddq_u64(l, vshlq_n_u64::<32>(h));
    let carry = vcltq_u64(v_lo, l);
    // High word of V. Subtracting the all-ones mask adds the carry.
    let v_hi = vsubq_u64(vshrq_n_u64::<32>(h), carry);

    // t1 = V_hi * EPSILON == (V_hi << 32) - V_hi, exact because V_hi <= 27.
    let t1 = vsubq_u64(vshlq_n_u64::<32>(v_hi), v_hi);
    let s = vaddq_u64(v_lo, t1);
    let wraparound = vcltq_u64(s, v_lo);
    // `wraparound >> 32` is EPSILON on overflow and 0 otherwise.
    vsraq_n_u64::<32>(s, wraparound)
}

/// `M_E` applied to a width-12 Goldilocks state.
///
/// Bit-identical to the portable path: it produces the same non-canonical
/// `u64` representatives, which matters because `GoldilocksField` is compared
/// and hashed raw.
///
/// # Safety
/// Requires NEON, which is baseline on AArch64 and is what gates this module.
#[inline(always)]
pub unsafe fn external_linear_layer(state: &mut [GoldilocksField; WIDTH]) {
    // `GoldilocksField` is `#[repr(transparent)]` over `u64`, and a
    // `[GoldilocksField; 12]` reference is aligned and long enough for six
    // 128-bit accesses.
    let p = state.as_mut_ptr().cast::<u64>();

    let (s0, s1) = split_pair(p);
    let (s2, s3) = split_pair(p.add(2));
    let (s4, s5) = split_pair(p.add(4));
    let (s6, s7) = split_pair(p.add(6));
    let (s8, s9) = split_pair(p.add(8));
    let (s10, s11) = split_pair(p.add(10));

    let mut b0 = [s0, s1, s2, s3];
    let mut b1 = [s4, s5, s6, s7];
    let mut b2 = [s8, s9, s10, s11];

    apply_mat4(&mut b0);
    apply_mat4(&mut b1);
    apply_mat4(&mut b2);

    // Outer circulant matrix: y_i = x_i' + sum of the x_j' with j == i (mod 4).
    let mut out = [vdupq_n_u64(0); WIDTH];
    for k in 0..4 {
        let sum = vaddq_u64(vaddq_u64(b0[k], b1[k]), b2[k]);
        out[k] = vaddq_u64(b0[k], sum);
        out[k + 4] = vaddq_u64(b1[k], sum);
        out[k + 8] = vaddq_u64(b2[k], sum);
    }

    vst1q_u64(p, reduce_pair(out[0], out[1]));
    vst1q_u64(p.add(2), reduce_pair(out[2], out[3]));
    vst1q_u64(p.add(4), reduce_pair(out[4], out[5]));
    vst1q_u64(p.add(6), reduce_pair(out[6], out[7]));
    vst1q_u64(p.add(8), reduce_pair(out[8], out[9]));
    vst1q_u64(p.add(10), reduce_pair(out[10], out[11]));
}

#[cfg(test)]
mod tests {
    use plonky2_field::types::{Field, PrimeField64};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// The two `(L, H)` accumulator halves of one element, reduced the way the
    /// layer's last step reduces them.
    fn reduce_halves(l: u64, h: u64) -> u64 {
        unsafe {
            let a = vld1q_u64([l, h].as_ptr());
            let mut out = [0u64; 2];
            vst1q_u64(out.as_mut_ptr(), reduce_pair(a, a));
            assert_eq!(out[0], out[1]);
            out[0]
        }
    }

    /// What the scalar path computes for the same accumulator.
    fn reduce_halves_reference(l: u64, h: u64) -> u64 {
        let v = (l as u128) + ((h as u128) << 32);
        GoldilocksField::from_noncanonical_u128_with_96_bits(v).to_noncanonical_u64()
    }

    #[test]
    fn reduction_matches_scalar_on_random_accumulators() {
        // Both halves stay below 28 * (2**32 - 1) < 2**36.81 in the layer.
        const BOUND: u64 = 28 * ((1 << 32) - 1);

        let mut rng = ChaCha8Rng::seed_from_u64(0x9e3779b97f4a7c15);
        for _ in 0..200_000 {
            let l = rng.gen_range(0..BOUND);
            let h = rng.gen_range(0..BOUND);
            assert_eq!(
                reduce_halves(l, h),
                reduce_halves_reference(l, h),
                "L = {l:#x}, H = {h:#x}"
            );
        }
    }

    /// The final `EPSILON` fold only fires when `V_lo` is within `2**37` of
    /// `2**64`, which uniform sampling reaches with probability about `2**-27`.
    /// These accumulators put `V_lo` there on purpose, and also cover the carry
    /// out of `L + (H << 32)`.
    #[test]
    fn reduction_matches_scalar_on_wraparound_accumulators() {
        for v_hi in 0..32u64 {
            for h_lo in [0u64, 1, 0x7fff_ffff, 0xffff_ffff] {
                let h = (v_hi << 32) | h_lo;
                for delta in 0..64u64 {
                    for l in [delta, delta.wrapping_neg() & 0xffff_ffff_ffff, 1 << 36] {
                        let l = l.min(28 * ((1 << 32) - 1));
                        assert_eq!(
                            reduce_halves(l, h),
                            reduce_halves_reference(l, h),
                            "L = {l:#x}, H = {h:#x}"
                        );
                    }
                }
            }
        }

        // V_lo lands just under 2**64 so that adding t1 = V_hi * EPSILON wraps.
        for v_hi in 1..28u64 {
            for k in 0..1024u64 {
                let h = (v_hi << 32) | 0xffff_ffff;
                let l = 0xffff_ffffu64 - k;
                assert_eq!(
                    reduce_halves(l, h),
                    reduce_halves_reference(l, h),
                    "L = {l:#x}, H = {h:#x}"
                );
            }
        }
    }
}
