//! Value-exact NEON for Poseidon2's multiply-free external linear layer.
//!
//! The three M4 blocks are independent. The first two run as a 2-wide pair of
//! 128-bit add chains (the same association as [`super::hash`]'s scalar
//! `external_linear_layer_u128`); the third block stays on that scalar form.
//! Results are reduced with the same `from_noncanonical_u128_with_96_bits`
//! the scalar path uses, so raw `u64` limbs match.
//!
//! `LIGHTER_DISABLE_P2_EXT_NEON=1` keeps the scalar u128 loop in the same
//! binary. This is not packed Goldilocks multiply (NEON has no 64×64 widening
//! mul) and not the retired two-wide permutation-factor NEON.

use core::arch::aarch64::{
    uint64x2_t, vaddq_u64, vcombine_u64, vcreate_u64, vgetq_lane_u64, vshrq_n_u64, vcltq_u64,
};

use super::config::WIDTH;

/// `LIGHTER_DISABLE_P2_EXT_NEON=1` restores the scalar u128 external layer.
#[inline]
pub(crate) fn external_neon_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var_os("LIGHTER_DISABLE_P2_EXT_NEON").is_some_and(|v| v == "1")
    })
}

#[inline(always)]
unsafe fn load2(a: u128, b: u128) -> (uint64x2_t, uint64x2_t) {
    let lo = vcombine_u64(vcreate_u64(a as u64), vcreate_u64(b as u64));
    let hi = vcombine_u64(vcreate_u64((a >> 64) as u64), vcreate_u64((b >> 64) as u64));
    (lo, hi)
}

#[inline(always)]
unsafe fn store2(lo: uint64x2_t, hi: uint64x2_t) -> (u128, u128) {
    let a = vgetq_lane_u64(lo, 0) as u128 | ((vgetq_lane_u64(hi, 0) as u128) << 64);
    let b = vgetq_lane_u64(lo, 1) as u128 | ((vgetq_lane_u64(hi, 1) as u128) << 64);
    (a, b)
}

/// Two independent 128-bit adds. Accumulators stay well below 2^96 on this
/// layer (≤ ~28 × (2^64−1)), so the high-word add cannot wrap.
#[inline(always)]
unsafe fn add2(
    a_lo: uint64x2_t,
    a_hi: uint64x2_t,
    b_lo: uint64x2_t,
    b_hi: uint64x2_t,
) -> (uint64x2_t, uint64x2_t) {
    let sum_lo = vaddq_u64(a_lo, b_lo);
    let carry = vshrq_n_u64(vcltq_u64(sum_lo, a_lo), 63);
    let sum_hi = vaddq_u64(vaddq_u64(a_hi, b_hi), carry);
    (sum_lo, sum_hi)
}

/// M4 on two independent 4-vectors. Association matches the scalar loop:
/// `(t0123 + t01) + x1`, `((t0123 + x1) + x2) + x2`, and so on.
#[inline(always)]
unsafe fn mat4_pair(state: &mut [u128; WIDTH]) {
    let (x0_lo, x0_hi) = load2(state[0], state[4]);
    let (x1_lo, x1_hi) = load2(state[1], state[5]);
    let (x2_lo, x2_hi) = load2(state[2], state[6]);
    let (x3_lo, x3_hi) = load2(state[3], state[7]);

    let (t01_lo, t01_hi) = add2(x0_lo, x0_hi, x1_lo, x1_hi);
    let (t23_lo, t23_hi) = add2(x2_lo, x2_hi, x3_lo, x3_hi);
    let (t0123_lo, t0123_hi) = add2(t01_lo, t01_hi, t23_lo, t23_hi);

    let (y0_lo, y0_hi) = {
        let (s, sh) = add2(t0123_lo, t0123_hi, t01_lo, t01_hi);
        add2(s, sh, x1_lo, x1_hi)
    };
    let (y1_lo, y1_hi) = {
        let (s, sh) = add2(t0123_lo, t0123_hi, x1_lo, x1_hi);
        let (s, sh) = add2(s, sh, x2_lo, x2_hi);
        add2(s, sh, x2_lo, x2_hi)
    };
    let (y2_lo, y2_hi) = {
        let (s, sh) = add2(t0123_lo, t0123_hi, t23_lo, t23_hi);
        add2(s, sh, x3_lo, x3_hi)
    };
    let (y3_lo, y3_hi) = {
        let (s, sh) = add2(t0123_lo, t0123_hi, x3_lo, x3_hi);
        let (s, sh) = add2(s, sh, x0_lo, x0_hi);
        add2(s, sh, x0_lo, x0_hi)
    };

    let (a0, b0) = store2(y0_lo, y0_hi);
    let (a1, b1) = store2(y1_lo, y1_hi);
    let (a2, b2) = store2(y2_lo, y2_hi);
    let (a3, b3) = store2(y3_lo, y3_hi);
    state[0] = a0;
    state[1] = a1;
    state[2] = a2;
    state[3] = a3;
    state[4] = b0;
    state[5] = b1;
    state[6] = b2;
    state[7] = b3;
}

#[inline(always)]
fn mat4_scalar(state: &mut [u128; WIDTH], i: usize) {
    let t01 = state[i] + state[i + 1];
    let t23 = state[i + 2] + state[i + 3];
    let t0123 = t01 + t23;
    let x0 = state[i];
    let x2 = state[i + 2];
    state[i] = t0123 + t01 + state[i + 1];
    state[i + 1] = t0123 + state[i + 1] + x2 + x2;
    state[i + 2] = t0123 + t23 + state[i + 3];
    state[i + 3] = t0123 + state[i + 3] + x0 + x0;
}

/// In-place external linear layer on raw u128 accumulators. Bit-identical to
/// [`super::hash::external_linear_layer_u128`].
#[inline]
pub(crate) fn external_linear_layer_u128_neon(state: &mut [u128; WIDTH]) {
    unsafe {
        mat4_pair(state);
    }
    mat4_scalar(state, 8);

    let mut sums = [0u128; 4];
    unsafe {
        let (a_lo, a_hi) = load2(state[0], state[1]);
        let (b_lo, b_hi) = load2(state[4], state[5]);
        let (c_lo, c_hi) = load2(state[8], state[9]);
        let (s, sh) = add2(a_lo, a_hi, b_lo, b_hi);
        let (s, sh) = add2(s, sh, c_lo, c_hi);
        let (s0, s1) = store2(s, sh);
        sums[0] = s0;
        sums[1] = s1;

        let (a_lo, a_hi) = load2(state[2], state[3]);
        let (b_lo, b_hi) = load2(state[6], state[7]);
        let (c_lo, c_hi) = load2(state[10], state[11]);
        let (s, sh) = add2(a_lo, a_hi, b_lo, b_hi);
        let (s, sh) = add2(s, sh, c_lo, c_hi);
        let (s2, s3) = store2(s, sh);
        sums[2] = s2;
        sums[3] = s3;
    }

    unsafe {
        for base in (0..WIDTH).step_by(2) {
            let (x_lo, x_hi) = load2(state[base], state[base + 1]);
            let (s_lo, s_hi) = load2(sums[base % 4], sums[(base + 1) % 4]);
            let (y_lo, y_hi) = add2(x_lo, x_hi, s_lo, s_hi);
            let (a, b) = store2(y_lo, y_hi);
            state[base] = a;
            state[base + 1] = b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::external_linear_layer_u128_neon;
    use crate::hash::poseidon2::hash::external_linear_layer_u128;
    use super::WIDTH;

    #[test]
    fn external_linear_layer_neon_matches_u128_raw() {
        let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
        let next = |rng: &mut u64| {
            *rng = rng.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            *rng
        };
        for case in 0..64 {
            let mut scalar = [0u128; WIDTH];
            for lane in scalar.iter_mut() {
                // Mix canonical, non-canonical, and near-2^64 limbs.
                let limb = next(&mut rng);
                *lane = if case % 3 == 0 {
                    limb as u128
                } else if case % 3 == 1 {
                    (limb as u128) + (1u128 << 64)
                } else {
                    u64::MAX as u128
                };
            }
            let mut neon = scalar;
            external_linear_layer_u128(&mut scalar);
            external_linear_layer_u128_neon(&mut neon);
            assert_eq!(neon, scalar, "case {case}");
        }
    }
}
