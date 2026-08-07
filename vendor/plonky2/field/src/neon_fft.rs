//! NEON vectorized add/sub reduction for the FFT butterfly.
//!
//! Only activated for `WideGoldilocksField` (WIDTH 4) on aarch64.
//! The scalar multiply stays in `NeonGoldilocksField::mul` (paired asm).
//! The add/sub modular reduction moves to the vector unit via `.2d` intrinsics,
//! bit-identical to the scalar `GoldilocksField::add`/`sub`.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::arch::aarch64::wide_goldilocks_field::WideGoldilocksField;
use crate::fft::FftRootTable;
use crate::packed::PackedField;

#[cfg(target_arch = "aarch64")]
const EPSILON: u64 = (1 << 32) - 1;

/// Vectorized Goldilocks modular addition on two `uint64x2_t` lanes.
///
/// Matches the scalar `GoldilocksField::add` bit-for-bit:
///   sum = x + y
///   if overflow: sum += EPSILON
///   if second overflow: sum += EPSILON again (unconditional masked add, no branch)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn goldilocks_add_v2d(x: uint64x2_t, y: uint64x2_t, eps_vec: uint64x2_t) -> uint64x2_t {
    let sum0 = vaddq_u64(x, y);
    let over1 = vcgtq_u64(x, sum0); // x > sum0 -> overflow occurred
    let eps1 = vandq_u64(over1, eps_vec);
    let sum1 = vaddq_u64(sum0, eps1);
    let over2 = vcgtq_u64(eps1, sum1); // eps1 > sum1 -> second overflow
    let eps2 = vandq_u64(over2, eps_vec);
    vaddq_u64(sum1, eps2) // guaranteed not to overflow further
}

/// Vectorized Goldilocks modular subtraction on two `uint64x2_t` lanes.
///
/// Matches the scalar `GoldilocksField::sub` bit-for-bit:
///   diff = x - y
///   if underflow: diff -= EPSILON
///   if second underflow: diff -= EPSILON again
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn goldilocks_sub_v2d(x: uint64x2_t, y: uint64x2_t, eps_vec: uint64x2_t) -> uint64x2_t {
    let diff0 = vsubq_u64(x, y);
    let under1 = vcgtq_u64(y, x); // y > x -> underflow occurred
    let eps1 = vandq_u64(under1, eps_vec);
    let diff1 = vsubq_u64(diff0, eps1);
    // Second underflow: diff0 < EPSILON when under1 is true.
    // Computed from diff0 (before first correction) to avoid a stale comparison.
    let under2 = vandq_u64(under1, vcgtq_u64(eps_vec, diff0)); // EPSILON > diff0
    let eps2 = vandq_u64(under2, eps_vec);
    vsubq_u64(diff1, eps2) // guaranteed not to underflow further
}

/// Load a `WideGoldilocksField` (4 × u64) into two `uint64x2_t` registers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_wide(ptr: *const u64) -> (uint64x2_t, uint64x2_t) {
    (vld1q_u64(ptr), vld1q_u64(ptr.add(2)))
}

/// Store two `uint64x2_t` registers to a `WideGoldilocksField` (4 × u64).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_wide(ptr: *mut u64, lo: uint64x2_t, hi: uint64x2_t) {
    vst1q_u64(ptr, lo);
    vst1q_u64(ptr.add(2), hi);
}

/// NEON-specialized single FFT layer butterfly.
///
/// Receives `&mut [WideGoldilocksField]` directly (TypeId-guarded dispatch from
/// the generic `fft_classic_simd_single_layer`). The per-lane scalar multiply
/// (`omega * values[k + half_packed_m + j]`) stays on the GPR side
/// (`NeonGoldilocksField::mul`); only the modular add/sub reduction moves to
/// NEON `.2d` instructions.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub unsafe fn fft_classic_simd_single_layer_neon(
    values: &mut [WideGoldilocksField],
    lg_half_m: usize,
    root_table: &FftRootTable<<WideGoldilocksField as PackedField>::Scalar>,
) {
    let eps_vec = vdupq_n_u64(EPSILON);
    let lg_packed_width = 2; // log2(WideGoldilocksField::WIDTH) = log2(4) = 2
    let lg_m = lg_half_m + 1;
    let m = 1usize << lg_m;
    let packed_m = m >> lg_packed_width;
    let half_packed_m = packed_m / 2;
    debug_assert!(half_packed_m != 0);

    let omega_table = WideGoldilocksField::pack_slice(&root_table[lg_half_m]);
    let base_ptr = values.as_mut_ptr() as *mut u64;

    for k in (0..values.len()).step_by(packed_m) {
        for j in 0..half_packed_m {
            // Scalar multiply: t = omega * values[k + half_packed_m + j]
            let omega = omega_table[j];
            let v_val = values[k + half_packed_m + j];
            let t: WideGoldilocksField = omega * v_val;

            // Load u = values[k + j] into NEON
            let u_idx = k + j;
            let (u_lo, u_hi) = load_wide(base_ptr.add(u_idx * 4));

            // Load t into NEON (crosses register file: GPR → NEON via memory)
            let t_ptr = &t as *const WideGoldilocksField as *const u64;
            let (t_lo, t_hi) = load_wide(t_ptr);

            // Vector add: u + t
            let add_lo = goldilocks_add_v2d(u_lo, t_lo, eps_vec);
            let add_hi = goldilocks_add_v2d(u_hi, t_hi, eps_vec);

            // Vector sub: u - t
            let sub_lo = goldilocks_sub_v2d(u_lo, t_lo, eps_vec);
            let sub_hi = goldilocks_sub_v2d(u_hi, t_hi, eps_vec);

            // Store back
            store_wide(base_ptr.add(u_idx * 4), add_lo, add_hi);
            let v_idx = k + half_packed_m + j;
            store_wide(base_ptr.add(v_idx * 4), sub_lo, sub_hi);
        }
    }
}

/// NEON-specialized fused two-layer (radix-4) FFT butterfly.
///
/// Same pattern as the single layer: scalar multiply via `NeonGoldilocksField::mul`,
/// NEON add/sub for the modular reduction on all four butterfly operations
/// (first stage: a±w1*b, c±w1*d; second stage: ab0±w2*cd0, ab1±w2*cd1).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub unsafe fn fft_classic_simd_fused_two_layers_neon(
    values: &mut [WideGoldilocksField],
    lg_half_m: usize,
    root_table: &FftRootTable<<WideGoldilocksField as PackedField>::Scalar>,
) {
    let eps_vec = vdupq_n_u64(EPSILON);
    let lg_packed_width = 2;
    let q = (1usize << lg_half_m) >> lg_packed_width;
    debug_assert!(q != 0);

    let stage1_omegas = WideGoldilocksField::pack_slice(&root_table[lg_half_m]);
    let stage2_omegas = WideGoldilocksField::pack_slice(&root_table[lg_half_m + 1]);
    let base_ptr = values.as_mut_ptr() as *mut u64;

    for k in (0..values.len()).step_by(4 * q) {
        for j in 0..q {
            let w1 = stage1_omegas[j];

            // Load a, b, c, d
            let a_val = values[k + j];
            let b_val = values[k + q + j];
            let c_val = values[k + 2 * q + j];
            let d_val = values[k + 3 * q + j];

            // --- First stage ---
            // Butterfly [a, b]: t = w1 * b, ab0 = a + t, ab1 = a - t
            let t_ab: WideGoldilocksField = w1 * b_val;
            // Butterfly [c, d]: t = w1 * d, cd0 = c + t, cd1 = c - t
            let t_cd: WideGoldilocksField = w1 * d_val;

            // Load a, c, t_ab, t_cd into NEON
            let a_ptr = &a_val as *const WideGoldilocksField as *const u64;
            let (a_lo, a_hi) = load_wide(a_ptr);
            let c_ptr = &c_val as *const WideGoldilocksField as *const u64;
            let (c_lo, c_hi) = load_wide(c_ptr);
            let tab_ptr = &t_ab as *const WideGoldilocksField as *const u64;
            let (tab_lo, tab_hi) = load_wide(tab_ptr);
            let tcd_ptr = &t_cd as *const WideGoldilocksField as *const u64;
            let (tcd_lo, tcd_hi) = load_wide(tcd_ptr);

            // ab0 = a + t_ab, ab1 = a - t_ab
            let ab0_lo = goldilocks_add_v2d(a_lo, tab_lo, eps_vec);
            let ab0_hi = goldilocks_add_v2d(a_hi, tab_hi, eps_vec);
            let ab1_lo = goldilocks_sub_v2d(a_lo, tab_lo, eps_vec);
            let ab1_hi = goldilocks_sub_v2d(a_hi, tab_hi, eps_vec);

            // cd0 = c + t_cd, cd1 = c - t_cd
            let cd0_lo = goldilocks_add_v2d(c_lo, tcd_lo, eps_vec);
            let cd0_hi = goldilocks_add_v2d(c_hi, tcd_hi, eps_vec);
            let cd1_lo = goldilocks_sub_v2d(c_lo, tcd_lo, eps_vec);
            let cd1_hi = goldilocks_sub_v2d(c_hi, tcd_hi, eps_vec);

            // --- Second stage ---
            let w2_j = stage2_omegas[j];
            let w2_qj = stage2_omegas[q + j];

            // Need cd0 and cd1 as WideGoldilocksField for the scalar multiply.
            // Reconstruct them from the NEON registers via stack.
            let mut cd0_buf: [u64; 4] = [0; 4];
            let cd0_ptr = cd0_buf.as_mut_ptr();
            store_wide(cd0_ptr, cd0_lo, cd0_hi);
            let cd0_ref = &*(cd0_ptr as *const WideGoldilocksField);

            let mut cd1_buf: [u64; 4] = [0; 4];
            let cd1_ptr = cd1_buf.as_mut_ptr();
            store_wide(cd1_ptr, cd1_lo, cd1_hi);
            let cd1_ref = &*(cd1_ptr as *const WideGoldilocksField);

            // t2a = w2_j * cd0, t2b = w2_qj * cd1
            let t2a: WideGoldilocksField = w2_j * *cd0_ref;
            let t2b: WideGoldilocksField = w2_qj * *cd1_ref;

            let t2a_ptr = &t2a as *const WideGoldilocksField as *const u64;
            let (t2a_lo, t2a_hi) = load_wide(t2a_ptr);
            let t2b_ptr = &t2b as *const WideGoldilocksField as *const u64;
            let (t2b_lo, t2b_hi) = load_wide(t2b_ptr);

            // values[k + j] = ab0 + t2a
            let r0_lo = goldilocks_add_v2d(ab0_lo, t2a_lo, eps_vec);
            let r0_hi = goldilocks_add_v2d(ab0_hi, t2a_hi, eps_vec);
            store_wide(base_ptr.add((k + j) * 4), r0_lo, r0_hi);

            // values[k + 2*q + j] = ab0 - t2a
            let r1_lo = goldilocks_sub_v2d(ab0_lo, t2a_lo, eps_vec);
            let r1_hi = goldilocks_sub_v2d(ab0_hi, t2a_hi, eps_vec);
            store_wide(base_ptr.add((k + 2 * q + j) * 4), r1_lo, r1_hi);

            // values[k + q + j] = ab1 + t2b
            let r2_lo = goldilocks_add_v2d(ab1_lo, t2b_lo, eps_vec);
            let r2_hi = goldilocks_add_v2d(ab1_hi, t2b_hi, eps_vec);
            store_wide(base_ptr.add((k + q + j) * 4), r2_lo, r2_hi);

            // values[k + 3*q + j] = ab1 - t2b
            let r3_lo = goldilocks_sub_v2d(ab1_lo, t2b_lo, eps_vec);
            let r3_hi = goldilocks_sub_v2d(ab1_hi, t2b_hi, eps_vec);
            store_wide(base_ptr.add((k + 3 * q + j) * 4), r3_lo, r3_hi);
        }
    }
}