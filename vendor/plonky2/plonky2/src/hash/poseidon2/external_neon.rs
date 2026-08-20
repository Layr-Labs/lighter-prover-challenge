//! 2-wide NEON for the multiply-free Poseidon2 external linear layer.
//!
//! The three 4-wide M4 blocks are independent. The first two run as a pair of
//! 128-bit add chains with the same association as `external_linear_layer_u128`;
//! the third stays on that scalar form. Results still go through
//! `from_noncanonical_u128_with_96_bits` in the caller.

use super::config::WIDTH;

pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var_os("LIGHTER_DISABLE_P2_EXT_NEON").is_some_and(|v| v == "1")
    })
}

#[inline]
pub(crate) fn external_linear_layer_u128(state: &mut [u128; WIDTH]) {
    apply_mat4_x2(&mut state[..8]);
    apply_mat4_scalar(&mut state[8..12]);
    let mut sums = [0u128; 4];
    for i in 0..4 {
        sums[i] = state[i] + state[i + 4] + state[i + 8];
    }
    for i in 0..WIDTH {
        state[i] += sums[i % 4];
    }
}

#[inline]
fn apply_mat4_scalar(block: &mut [u128]) {
    debug_assert_eq!(block.len(), 4);
    let t01 = block[0] + block[1];
    let t23 = block[2] + block[3];
    let t0123 = t01 + t23;
    let x0 = block[0];
    let x2 = block[2];
    block[0] = t0123 + t01 + block[1];
    block[1] = t0123 + block[1] + x2 + x2;
    block[2] = t0123 + t23 + block[3];
    block[3] = t0123 + block[3] + x0 + x0;
}

/// Two independent M4 blocks in `block[0..4]` and `block[4..8]`.
#[inline]
fn apply_mat4_x2(block: &mut [u128]) {
    debug_assert_eq!(block.len(), 8);
    use core::arch::aarch64::{
        uint64x2_t, vaddq_u64, vcltq_u64, vcombine_u64, vcreate_u64, vgetq_lane_u64, vshrq_n_u64,
    };

    let pack = |a: u128, b: u128| -> (uint64x2_t, uint64x2_t) {
        let lo = unsafe { vcombine_u64(vcreate_u64(a as u64), vcreate_u64(b as u64)) };
        let hi = unsafe {
            vcombine_u64(
                vcreate_u64((a >> 64) as u64),
                vcreate_u64((b >> 64) as u64),
            )
        };
        (lo, hi)
    };
    let unpack = |lo: uint64x2_t, hi: uint64x2_t| -> (u128, u128) {
        let a = unsafe {
            (vgetq_lane_u64(lo, 0) as u128) | ((vgetq_lane_u64(hi, 0) as u128) << 64)
        };
        let b = unsafe {
            (vgetq_lane_u64(lo, 1) as u128) | ((vgetq_lane_u64(hi, 1) as u128) << 64)
        };
        (a, b)
    };
    let add = |a_lo: uint64x2_t, a_hi: uint64x2_t, b_lo: uint64x2_t, b_hi: uint64x2_t| {
        unsafe {
            let sum_lo = vaddq_u64(a_lo, b_lo);
            let carry = vshrq_n_u64(vcltq_u64(sum_lo, a_lo), 63);
            let sum_hi = vaddq_u64(vaddq_u64(a_hi, b_hi), carry);
            (sum_lo, sum_hi)
        }
    };

    let (s0_lo, s0_hi) = pack(block[0], block[4]);
    let (s1_lo, s1_hi) = pack(block[1], block[5]);
    let (s2_lo, s2_hi) = pack(block[2], block[6]);
    let (s3_lo, s3_hi) = pack(block[3], block[7]);

    let (t01_lo, t01_hi) = add(s0_lo, s0_hi, s1_lo, s1_hi);
    let (t23_lo, t23_hi) = add(s2_lo, s2_hi, s3_lo, s3_hi);
    let (t0123_lo, t0123_hi) = add(t01_lo, t01_hi, t23_lo, t23_hi);

    let (tmp_lo, tmp_hi) = add(t0123_lo, t0123_hi, t01_lo, t01_hi);
    let (n0_lo, n0_hi) = add(tmp_lo, tmp_hi, s1_lo, s1_hi);

    let (tmp_lo, tmp_hi) = add(t0123_lo, t0123_hi, s1_lo, s1_hi);
    let (tmp_lo, tmp_hi) = add(tmp_lo, tmp_hi, s2_lo, s2_hi);
    let (n1_lo, n1_hi) = add(tmp_lo, tmp_hi, s2_lo, s2_hi);

    let (tmp_lo, tmp_hi) = add(t0123_lo, t0123_hi, t23_lo, t23_hi);
    let (n2_lo, n2_hi) = add(tmp_lo, tmp_hi, s3_lo, s3_hi);

    let (tmp_lo, tmp_hi) = add(t0123_lo, t0123_hi, s3_lo, s3_hi);
    let (tmp_lo, tmp_hi) = add(tmp_lo, tmp_hi, s0_lo, s0_hi);
    let (n3_lo, n3_hi) = add(tmp_lo, tmp_hi, s0_lo, s0_hi);

    let (n0a, n0b) = unpack(n0_lo, n0_hi);
    let (n1a, n1b) = unpack(n1_lo, n1_hi);
    let (n2a, n2b) = unpack(n2_lo, n2_hi);
    let (n3a, n3b) = unpack(n3_lo, n3_hi);
    block[0] = n0a;
    block[4] = n0b;
    block[1] = n1a;
    block[5] = n1b;
    block[2] = n2a;
    block[6] = n2b;
    block[3] = n3a;
    block[7] = n3b;
}

#[cfg(test)]
mod tests {
    use super::{apply_mat4_scalar, apply_mat4_x2, external_linear_layer_u128, WIDTH};

    fn scalar_mat4(block: &mut [u128]) {
        apply_mat4_scalar(block);
    }

    #[test]
    fn external_linear_layer_neon_matches_u128_raw() {
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let next = |s: &mut u64| {
            *s = s.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            *s
        };
        for case in 0..64 {
            let mut state = [0u128; WIDTH];
            for slot in state.iter_mut() {
                let lo = match case % 4 {
                    0 => next(&mut seed) as u128,
                    1 => (next(&mut seed) as u128) + (1u128 << 64),
                    2 => u64::MAX as u128,
                    _ => u128::MAX,
                };
                *slot = lo;
            }
            let mut neon = state;
            let mut scalar = state;
            for i in (0..WIDTH).step_by(4) {
                scalar_mat4(&mut scalar[i..i + 4]);
            }
            let mut sums = [0u128; 4];
            for i in 0..4 {
                sums[i] = scalar[i] + scalar[i + 4] + scalar[i + 8];
            }
            for i in 0..WIDTH {
                scalar[i] += sums[i % 4];
            }
            external_linear_layer_u128(&mut neon);
            assert_eq!(neon, scalar, "case {case}");
            let mut x2_in = state;
            apply_mat4_x2(&mut x2_in[..8]);
            apply_mat4_scalar(&mut state[0..4]);
            apply_mat4_scalar(&mut state[4..8]);
            assert_eq!(&x2_in[..8], &state[..8], "mat4 x2 case {case}");
        }
    }
}
