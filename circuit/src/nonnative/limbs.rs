// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Fixed-width stack-limb arithmetic for hot nonnative witness generators.
//!
//! The nonnative generators (`NonNative*Generator`) historically did all of
//! their per-run arithmetic with heap-allocated `BigUint`s, generic long
//! division, and Fermat-exponentiation inverses. For the secp256k1 base and
//! scalar fields (the ECDSA hot path, thousands of generator runs per signed
//! transaction) that cost is dominated by allocator churn and `BigUint`
//! division. This module provides drop-in `[u64; 4]` / `[u64; 8]` arithmetic
//! for exactly those two fixed 256-bit moduli.
//!
//! Bit-identity: every function here computes the *same integers* the
//! `BigUint` reference path computes (same canonicalization rule, the same
//! strict/non-strict comparisons, the unique modular inverse), and the
//! witness writer mirrors `GeneratedValuesBigUint::set_biguint_target`
//! digit-for-digit (including its `assert!`). Since the generators only ever
//! write `F::from_canonical_u32` / `F::from_bool` values derived from these
//! integers, equal integers imply bit-identical witness values.
//!
//! Any input shape this module does not handle exactly (extra-wide targets
//! with non-zero high limbs, out-of-range limb values, non-secp fields)
//! returns `None` from the readers / dispatcher so callers fall back to the
//! original `BigUint` path unchanged.

use core::any::TypeId;
use core::cmp::Ordering;

use anyhow::Result;
use plonky2::field::secp256k1_base::Secp256K1Base;
use plonky2::field::secp256k1_scalar::Secp256K1Scalar;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::iop::generator::GeneratedValues;
use plonky2::iop::witness::Witness;

use crate::bigint::biguint::BigUintTarget;
use crate::uint::u32::witness::GeneratedValuesU32;

/// Little-endian 256-bit unsigned integer.
pub type U256 = [u64; 4];
/// Little-endian 512-bit unsigned integer.
pub type U512 = [u64; 8];

/// A fixed 256-bit modulus with its top bit set (i.e. `2^255 <= p < 2^256`).
#[derive(Debug)]
pub struct FixedModulus {
    pub limbs: U256,
}

/// `p = 2^256 - 2^32 - 977`, the secp256k1 base field order.
pub const SECP256K1_BASE_MODULUS: FixedModulus = FixedModulus {
    limbs: [
        0xFFFFFFFEFFFFFC2F,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
    ],
};

/// `n = 2^256 - 432420386565659656852420866394968145599`, the secp256k1
/// scalar field order.
pub const SECP256K1_SCALAR_MODULUS: FixedModulus = FixedModulus {
    limbs: [
        0xBFD25E8CD0364141,
        0xBAAEDCE6AF48A03B,
        0xFFFFFFFFFFFFFFFE,
        0xFFFFFFFFFFFFFFFF,
    ],
};

/// Returns the fixed modulus for `FF` when `FF` is one of the two secp256k1
/// fields, and `None` otherwise (callers fall back to the `BigUint` path).
#[inline]
pub fn fixed_modulus_for_field<FF: Field>() -> Option<&'static FixedModulus> {
    let id = TypeId::of::<FF>();
    if id == TypeId::of::<Secp256K1Base>() {
        Some(&SECP256K1_BASE_MODULUS)
    } else if id == TypeId::of::<Secp256K1Scalar>() {
        Some(&SECP256K1_SCALAR_MODULUS)
    } else {
        None
    }
}

impl FixedModulus {
    /// Reduces a raw `< 2^256` value exactly like
    /// `FF::from_noncanonical_biguint(x).to_canonical_biguint()`: one
    /// conditional subtraction of the modulus. (Since `2p > 2^256`, a single
    /// subtraction is always sufficient.)
    #[inline]
    pub fn canonicalize(&self, x: &U256) -> U256 {
        if cmp_256(x, &self.limbs) != Ordering::Less {
            sub_256(x, &self.limbs).0
        } else {
            *x
        }
    }
}

// ---------------------------------------------------------------------------
// Core limb primitives.
// ---------------------------------------------------------------------------

#[inline]
pub fn cmp_256(a: &U256, b: &U256) -> Ordering {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

#[inline]
pub fn cmp_512(a: &U512, b: &U512) -> Ordering {
    for i in (0..8).rev() {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
    }
    Ordering::Equal
}

#[inline]
pub fn is_zero_256(a: &U256) -> bool {
    a.iter().all(|&l| l == 0)
}

/// `a + b`, returning the wrapped sum and the carry-out bit.
#[inline]
pub fn add_256(a: &U256, b: &U256) -> (U256, u64) {
    let mut out = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        let t = a[i] as u128 + b[i] as u128 + carry as u128;
        out[i] = t as u64;
        carry = (t >> 64) as u64;
    }
    (out, carry)
}

/// `a - b` (wrapping), returning the difference and the borrow-out bit.
#[inline]
pub fn sub_256(a: &U256, b: &U256) -> (U256, u64) {
    let mut out = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        out[i] = d2;
        borrow = (b1 as u64) | (b2 as u64);
    }
    (out, borrow)
}

/// `a - b` for 512-bit values; requires `a >= b`.
#[inline]
pub fn sub_512(a: &U512, b: &U512) -> U512 {
    let mut out = [0u64; 8];
    let mut borrow = 0u64;
    for i in 0..8 {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        out[i] = d2;
        borrow = (b1 as u64) | (b2 as u64);
    }
    debug_assert_eq!(borrow, 0);
    out
}

#[inline]
pub fn u256_to_u512(a: &U256) -> U512 {
    [a[0], a[1], a[2], a[3], 0, 0, 0, 0]
}

/// Schoolbook 256x256 -> 512 multiplication.
#[inline]
pub fn mul_wide_256(a: &U256, b: &U256) -> U512 {
    let mut out = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u64;
        for j in 0..4 {
            let t = out[i + j] as u128 + a[i] as u128 * b[j] as u128 + carry as u128;
            out[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        out[i + 4] = carry;
    }
    out
}

/// Knuth Algorithm D: divides a 512-bit value by a 256-bit divisor whose top
/// bit is set (already normalized, so no shift is needed). Returns the
/// (up to 320-bit) quotient and the remainder. Exact integer division:
/// results are identical to `BigUint::div_rem`.
pub fn div_rem_512_by_256(u: &U512, v: &U256) -> ([u64; 5], U256) {
    debug_assert_eq!(v[3] >> 63, 1, "divisor must be normalized");
    let mut rem = [u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], 0u64];
    let mut q = [0u64; 5];
    let v3 = v[3] as u128;
    for j in (0..5).rev() {
        let num = ((rem[j + 4] as u128) << 64) | rem[j + 3] as u128;
        // Knuth's estimate: q_hat = min(floor(num / v3), B - 1).
        let (mut qhat, mut rhat): (u128, u128) = if rem[j + 4] >= v[3] {
            let qh = (1u128 << 64) - 1;
            (qh, num - qh * v3)
        } else {
            (num / v3, num % v3)
        };
        // Refine with the second divisor limb (at most two decrements).
        while rhat < (1u128 << 64) && qhat * (v[2] as u128) > ((rhat << 64) | rem[j + 2] as u128) {
            qhat -= 1;
            rhat += v3;
        }
        // Multiply-and-subtract: rem[j..=j+4] -= q_hat * v.
        let mut qh = qhat as u64;
        let mut borrow = 0u64;
        let mut carry = 0u64;
        for i in 0..4 {
            let p = qh as u128 * v[i] as u128 + carry as u128;
            carry = (p >> 64) as u64;
            let (d1, b1) = rem[j + i].overflowing_sub(p as u64);
            let (d2, b2) = d1.overflowing_sub(borrow);
            rem[j + i] = d2;
            borrow = (b1 as u64) | (b2 as u64);
        }
        let (d1, b1) = rem[j + 4].overflowing_sub(carry);
        let (d2, b2) = d1.overflowing_sub(borrow);
        rem[j + 4] = d2;
        if b1 || b2 {
            // q_hat was one too large (rare); add the divisor back.
            qh -= 1;
            let mut c = 0u64;
            for i in 0..4 {
                let (s1, c1) = rem[j + i].overflowing_add(v[i]);
                let (s2, c2) = s1.overflowing_add(c);
                rem[j + i] = s2;
                c = (c1 as u64) | (c2 as u64);
            }
            rem[j + 4] = rem[j + 4].wrapping_add(c);
        }
        q[j] = qh;
    }
    (q, [rem[0], rem[1], rem[2], rem[3]])
}

#[inline]
fn shr1_256(x: &mut U256, hi: u64) {
    debug_assert!(hi <= 1);
    x[0] = (x[0] >> 1) | (x[1] << 63);
    x[1] = (x[1] >> 1) | (x[2] << 63);
    x[2] = (x[2] >> 1) | (x[3] << 63);
    x[3] = (x[3] >> 1) | (hi << 63);
}

#[inline]
fn add_assign_256(x: &mut U256, y: &U256) -> u64 {
    let mut carry = 0u64;
    for i in 0..4 {
        let t = x[i] as u128 + y[i] as u128 + carry as u128;
        x[i] = t as u64;
        carry = (t >> 64) as u64;
    }
    carry
}

#[inline]
fn sub_assign_256(x: &mut U256, y: &U256) {
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d1, b1) = x[i].overflowing_sub(y[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        x[i] = d2;
        borrow = (b1 as u64) | (b2 as u64);
    }
    debug_assert_eq!(borrow, 0);
}

/// `x = (x - y) mod p`, with `x, y` in `[0, p)`.
#[inline]
fn sub_mod_assign_256(x: &mut U256, y: &U256, p: &U256) {
    if cmp_256(x, y) != Ordering::Less {
        sub_assign_256(x, y);
    } else {
        let mut t = *y;
        sub_assign_256(&mut t, x);
        *x = *p;
        sub_assign_256(x, &t);
    }
}

/// Modular inverse via the binary extended Euclidean algorithm.
///
/// Requires `p` odd (both secp256k1 moduli are odd primes) and
/// `a` canonical and non-zero. Returns the unique inverse in `[0, p)` —
/// the exact same integer the `BigUint` Fermat-exponentiation path
/// (`FF::try_inverse`) produces, at a small fraction of the cost.
pub fn mod_inverse(a: &U256, p: &U256) -> U256 {
    debug_assert_eq!(p[0] & 1, 1, "modulus must be odd");
    debug_assert!(!is_zero_256(a));
    debug_assert_eq!(cmp_256(a, p), Ordering::Less);
    let one: U256 = [1, 0, 0, 0];
    let mut u = *a;
    let mut v = *p;
    let mut x1: U256 = one;
    let mut x2: U256 = [0; 4];
    while u != one && v != one {
        while u[0] & 1 == 0 {
            shr1_256(&mut u, 0);
            if x1[0] & 1 == 0 {
                shr1_256(&mut x1, 0);
            } else {
                let carry = add_assign_256(&mut x1, p);
                shr1_256(&mut x1, carry);
            }
        }
        while v[0] & 1 == 0 {
            shr1_256(&mut v, 0);
            if x2[0] & 1 == 0 {
                shr1_256(&mut x2, 0);
            } else {
                let carry = add_assign_256(&mut x2, p);
                shr1_256(&mut x2, carry);
            }
        }
        if cmp_256(&u, &v) != Ordering::Less {
            sub_assign_256(&mut u, &v);
            sub_mod_assign_256(&mut x1, &x2, p);
        } else {
            sub_assign_256(&mut v, &u);
            sub_mod_assign_256(&mut x2, &x1, p);
        }
    }
    if u == one { x1 } else { x2 }
}

// ---------------------------------------------------------------------------
// Generator arithmetic (pure integer semantics of each `run_once` body).
// ---------------------------------------------------------------------------

pub struct AddOutcome {
    pub sum: U256,
    pub overflow: bool,
}

/// Mirrors `NonNativeAdditionGenerator::run_once`. Note the original uses a
/// *strict* comparison (`sum > modulus`), so a sum equal to the modulus is
/// written back unreduced; we reproduce that exactly.
pub fn add_generator_math(m: &FixedModulus, a_raw: &U256, b_raw: &U256) -> AddOutcome {
    let a = m.canonicalize(a_raw);
    let b = m.canonicalize(b_raw);
    let (sum, carry) = add_256(&a, &b);
    if carry != 0 || cmp_256(&sum, &m.limbs) == Ordering::Greater {
        AddOutcome {
            sum: sub_256(&sum, &m.limbs).0,
            overflow: true,
        }
    } else {
        AddOutcome {
            sum,
            overflow: false,
        }
    }
}

pub struct SubOutcome {
    pub diff: U256,
    pub overflow: bool,
}

/// Mirrors `NonNativeSubtractionGenerator::run_once`.
pub fn sub_generator_math(m: &FixedModulus, a_raw: &U256, b_raw: &U256) -> SubOutcome {
    let a = m.canonicalize(a_raw);
    let b = m.canonicalize(b_raw);
    if cmp_256(&a, &b) != Ordering::Less {
        SubOutcome {
            diff: sub_256(&a, &b).0,
            overflow: false,
        }
    } else {
        let t = sub_256(&b, &a).0;
        SubOutcome {
            diff: sub_256(&m.limbs, &t).0,
            overflow: true,
        }
    }
}

pub struct MulOutcome {
    pub prod: U256,
    pub overflow: U256,
}

/// Mirrors `NonNativeMultiplicationGenerator::run_once`:
/// `(overflow, prod) = (a * b).div_rem(p)` on canonicalized inputs.
pub fn mul_generator_math(m: &FixedModulus, a_raw: &U256, b_raw: &U256) -> MulOutcome {
    let a = m.canonicalize(a_raw);
    let b = m.canonicalize(b_raw);
    let wide = mul_wide_256(&a, &b);
    let (q, r) = div_rem_512_by_256(&wide, &m.limbs);
    debug_assert_eq!(q[4], 0);
    MulOutcome {
        prod: r,
        overflow: [q[0], q[1], q[2], q[3]],
    }
}

pub struct MulDivOutcome {
    pub result: U256,
    pub overflow: U256,
    pub add_to_lhs: bool,
}

/// Mirrors `NonNativeMulDivGenerator::run_once`. Returns `None` when the
/// (canonical) divisor is zero, matching the original's error case.
/// The original reduces the 768-bit product `a * b * d^-1` in one step; we
/// reduce after each multiplication, which yields the identical remainder.
pub fn mul_div_generator_math(
    m: &FixedModulus,
    a_raw: &U256,
    b_raw: &U256,
    d_raw: &U256,
) -> Option<MulDivOutcome> {
    let a = m.canonicalize(a_raw);
    let b = m.canonicalize(b_raw);
    let d = m.canonicalize(d_raw);
    if is_zero_256(&d) {
        return None;
    }
    let d_inv = mod_inverse(&d, &m.limbs);
    let lhs = mul_wide_256(&a, &b);
    let (_, ab_red) = div_rem_512_by_256(&lhs, &m.limbs);
    let t = mul_wide_256(&ab_red, &d_inv);
    let (_, result) = div_rem_512_by_256(&t, &m.limbs);
    let rhs = mul_wide_256(&d, &result);
    // Original: `if lhs > rhs { .. add_to_lhs = false } else { .. true }`.
    if cmp_512(&lhs, &rhs) == Ordering::Greater {
        let diff = sub_512(&lhs, &rhs);
        let (q, _) = div_rem_512_by_256(&diff, &m.limbs);
        debug_assert_eq!(q[4], 0);
        Some(MulDivOutcome {
            result,
            overflow: [q[0], q[1], q[2], q[3]],
            add_to_lhs: false,
        })
    } else {
        let diff = sub_512(&rhs, &lhs);
        let (q, _) = div_rem_512_by_256(&diff, &m.limbs);
        debug_assert_eq!(q[4], 0);
        Some(MulDivOutcome {
            result,
            overflow: [q[0], q[1], q[2], q[3]],
            add_to_lhs: true,
        })
    }
}

pub struct InverseOutcome {
    pub inv: U256,
    pub div: U256,
}

/// Mirrors `NonNativeInverseGenerator::run_once`. Returns `None` when the
/// (canonical) input is zero, in which case the caller writes zeros.
pub fn inverse_generator_math(m: &FixedModulus, x_raw: &U256) -> Option<InverseOutcome> {
    let x = m.canonicalize(x_raw);
    if is_zero_256(&x) {
        return None;
    }
    let inv = mod_inverse(&x, &m.limbs);
    let wide = mul_wide_256(&x, &inv);
    let (q, _) = div_rem_512_by_256(&wide, &m.limbs);
    debug_assert_eq!(q[4], 0);
    Some(InverseOutcome {
        inv,
        div: [q[0], q[1], q[2], q[3]],
    })
}

pub struct DivisionOutcome {
    pub div: U256,
    pub overflow: U256,
    pub add_to_a: bool,
}

/// Mirrors `NonNativeDivisionGenerator::run_once`. Returns `None` when the
/// (canonical) divisor is zero, in which case the caller writes zeros.
/// NOTE: like the original, this uses the *raw* (possibly unreduced) `a` and
/// `b` witness values everywhere except in the inverse computation.
pub fn division_generator_math(
    m: &FixedModulus,
    a_raw: &U256,
    b_raw: &U256,
) -> Option<DivisionOutcome> {
    let b_c = m.canonicalize(b_raw);
    if is_zero_256(&b_c) {
        return None;
    }
    let b_inv = mod_inverse(&b_c, &m.limbs);
    let wide = mul_wide_256(a_raw, &b_inv);
    let (_, div) = div_rem_512_by_256(&wide, &m.limbs);
    let b_times_div = mul_wide_256(b_raw, &div);
    let a_512 = u256_to_u512(a_raw);
    let (diff, add_to_a) = if cmp_512(&b_times_div, &a_512) != Ordering::Greater {
        (sub_512(&a_512, &b_times_div), false)
    } else {
        (sub_512(&b_times_div, &a_512), true)
    };
    let (q, r) = div_rem_512_by_256(&diff, &m.limbs);
    // Mirrors `assert_eq!(should_be_zero, BigUint::ZERO)` in the original.
    assert!(
        is_zero_256(&r),
        "non-zero remainder in NonNativeDivisionGenerator"
    );
    debug_assert_eq!(q[4], 0);
    Some(DivisionOutcome {
        div,
        overflow: [q[0], q[1], q[2], q[3]],
        add_to_a,
    })
}

// ---------------------------------------------------------------------------
// Witness IO helpers.
// ---------------------------------------------------------------------------

/// Reads a `BigUintTarget` into a `U256` without heap allocation.
///
/// Returns `None` (caller must fall back to the `BigUint` path) if any limb
/// holds a value `>= 2^32`, or if the target has more than 8 limbs with a
/// non-zero high limb — the only cases in which
/// `WitnessBigUint::get_biguint_target` could yield a different integer.
#[inline]
pub fn try_read_u256<F: PrimeField64, W: Witness<F>>(
    witness: &W,
    target: &BigUintTarget,
) -> Option<U256> {
    let limbs = &target.limbs;
    if limbs.len() > 8 {
        for l in &limbs[8..] {
            if witness.get_target(l.0).to_canonical_u64() != 0 {
                return None;
            }
        }
    }
    let mut out = [0u64; 4];
    for (i, l) in limbs.iter().take(8).enumerate() {
        let v = witness.get_target(l.0).to_canonical_u64();
        if v > u32::MAX as u64 {
            return None;
        }
        out[i >> 1] |= v << ((i & 1) * 32);
    }
    Some(out)
}

/// 512-bit variant of [`try_read_u256`] (up to 16 u32 limbs).
#[inline]
pub fn try_read_u512<F: PrimeField64, W: Witness<F>>(
    witness: &W,
    target: &BigUintTarget,
) -> Option<U512> {
    let limbs = &target.limbs;
    if limbs.len() > 16 {
        for l in &limbs[16..] {
            if witness.get_target(l.0).to_canonical_u64() != 0 {
                return None;
            }
        }
    }
    let mut out = [0u64; 8];
    for (i, l) in limbs.iter().take(16).enumerate() {
        let v = witness.get_target(l.0).to_canonical_u64();
        if v > u32::MAX as u64 {
            return None;
        }
        out[i >> 1] |= v << ((i & 1) * 32);
    }
    Some(out)
}

#[inline]
pub fn u256_digits(x: &U256) -> [u32; 8] {
    let mut d = [0u32; 8];
    for i in 0..4 {
        d[2 * i] = x[i] as u32;
        d[2 * i + 1] = (x[i] >> 32) as u32;
    }
    d
}

#[inline]
pub fn u320_digits(x: &[u64; 5]) -> [u32; 10] {
    let mut d = [0u32; 10];
    for i in 0..5 {
        d[2 * i] = x[i] as u32;
        d[2 * i + 1] = (x[i] >> 32) as u32;
    }
    d
}

/// Writes little-endian u32 digits to a `BigUintTarget`, mirroring
/// `GeneratedValuesBigUint::set_biguint_target` exactly: the same
/// `assert!` on the stripped digit count and the same
/// `F::from_canonical_u32` writes (zero-padded to the target width).
pub fn set_limb_digits_target<F: Field>(
    out_buffer: &mut GeneratedValues<F>,
    target: &BigUintTarget,
    digits: &[u32],
) -> Result<()> {
    let mut len = digits.len();
    while len > 0 && digits[len - 1] == 0 {
        len -= 1;
    }
    assert!(target.num_limbs() >= len);
    for i in 0..target.num_limbs() {
        let v = if i < digits.len() { digits[i] } else { 0 };
        out_buffer.set_u32_target(target.get_limb(i), v)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use num::bigint::RandBigInt;
    use num::{BigUint, Integer, One, Zero};
    use plonky2::field::types::PrimeField;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    fn to_biguint_256(x: &U256) -> BigUint {
        let mut bytes = Vec::with_capacity(32);
        for l in x {
            bytes.extend_from_slice(&l.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }

    fn to_biguint_512(x: &U512) -> BigUint {
        let mut bytes = Vec::with_capacity(64);
        for l in x {
            bytes.extend_from_slice(&l.to_le_bytes());
        }
        BigUint::from_bytes_le(&bytes)
    }

    fn from_biguint_256(x: &BigUint) -> U256 {
        let digits = x.to_u64_digits();
        assert!(digits.len() <= 4);
        let mut out = [0u64; 4];
        out[..digits.len()].copy_from_slice(&digits);
        out
    }

    fn rand_u256(rng: &mut StdRng) -> U256 {
        [rng.r#gen(), rng.r#gen(), rng.r#gen(), rng.r#gen()]
    }

    fn rand_u512(rng: &mut StdRng) -> U512 {
        core::array::from_fn(|_| rng.r#gen())
    }

    /// Canonicalization used by the original generators:
    /// `FF::from_noncanonical_biguint(x).to_canonical_biguint()`.
    fn ref_canon(x: &BigUint, m: &BigUint) -> BigUint {
        if x >= m { x - m } else { x.clone() }
    }

    fn moduli() -> [(&'static FixedModulus, BigUint); 2] {
        [
            (&SECP256K1_BASE_MODULUS, Secp256K1Base::order()),
            (&SECP256K1_SCALAR_MODULUS, Secp256K1Scalar::order()),
        ]
    }

    #[test]
    fn test_moduli_match_field_orders() {
        assert_eq!(
            to_biguint_256(&SECP256K1_BASE_MODULUS.limbs),
            Secp256K1Base::order()
        );
        assert_eq!(
            to_biguint_256(&SECP256K1_SCALAR_MODULUS.limbs),
            Secp256K1Scalar::order()
        );
        // Both moduli must be normalized (top bit set) for Knuth division.
        assert_eq!(SECP256K1_BASE_MODULUS.limbs[3] >> 63, 1);
        assert_eq!(SECP256K1_SCALAR_MODULUS.limbs[3] >> 63, 1);
    }

    #[test]
    fn test_dispatch() {
        assert!(core::ptr::eq(
            fixed_modulus_for_field::<Secp256K1Base>().unwrap(),
            &SECP256K1_BASE_MODULUS
        ));
        assert!(core::ptr::eq(
            fixed_modulus_for_field::<Secp256K1Scalar>().unwrap(),
            &SECP256K1_SCALAR_MODULUS
        ));
        assert!(
            fixed_modulus_for_field::<crate::blob::bls12_381_scalar_field::BLS12381Scalar>()
                .is_none()
        );
        assert!(fixed_modulus_for_field::<plonky2::field::goldilocks_field::GoldilocksField>().is_none());
    }

    #[test]
    fn test_mul_wide_and_add_sub_differential() {
        let mut rng = StdRng::seed_from_u64(0x11ab5);
        for _ in 0..20_000 {
            let a = rand_u256(&mut rng);
            let b = rand_u256(&mut rng);
            let a_big = to_biguint_256(&a);
            let b_big = to_biguint_256(&b);

            let wide = mul_wide_256(&a, &b);
            assert_eq!(to_biguint_512(&wide), &a_big * &b_big);

            let (sum, carry) = add_256(&a, &b);
            assert_eq!(
                to_biguint_256(&sum) + (BigUint::from(carry) << 256),
                &a_big + &b_big
            );

            let (diff, borrow) = sub_256(&a, &b);
            if a_big >= b_big {
                assert_eq!(borrow, 0);
                assert_eq!(to_biguint_256(&diff), &a_big - &b_big);
            } else {
                assert_eq!(borrow, 1);
                assert_eq!(
                    to_biguint_256(&diff),
                    (BigUint::one() << 256) + &a_big - &b_big
                );
            }

            assert_eq!(cmp_256(&a, &b), a_big.cmp(&b_big));
        }
    }

    #[test]
    fn test_div_rem_differential_random() {
        let mut rng = StdRng::seed_from_u64(0xd117);
        for iter in 0..40_000 {
            // Random normalized divisor; bias towards the two secp moduli.
            let v: U256 = match iter % 4 {
                0 => SECP256K1_BASE_MODULUS.limbs,
                1 => SECP256K1_SCALAR_MODULUS.limbs,
                _ => {
                    let mut v = rand_u256(&mut rng);
                    v[3] |= 1 << 63;
                    v
                }
            };
            // Random dividend with a variety of shapes.
            let mut u = rand_u512(&mut rng);
            match iter % 8 {
                2 => {
                    // Force the q_hat == B-1 clamp: top limbs mirror the divisor.
                    u[7] = v[3];
                    u[6] = v[2];
                }
                3 => {
                    // Exact multiple of v.
                    let k = rand_u256(&mut rng);
                    u = mul_wide_256(&k, &v);
                }
                4 => {
                    // Multiple of v minus one (max remainder edge nearby).
                    let k = rand_u256(&mut rng);
                    let prod = mul_wide_256(&k, &v);
                    if !prod.iter().all(|&l| l == 0) {
                        u = sub_512(&prod, &[1, 0, 0, 0, 0, 0, 0, 0]);
                    }
                }
                5 => {
                    // Small dividend (quotient zero).
                    u = [u[0], 0, 0, 0, 0, 0, 0, 0];
                }
                6 => {
                    // Dividend just below 2^512.
                    u = [u64::MAX; 8];
                    u[0] = rng.r#gen();
                }
                _ => {}
            }
            let (q, r) = div_rem_512_by_256(&u, &v);
            let u_big = to_biguint_512(&u);
            let v_big = to_biguint_256(&v);
            let (q_big, r_big) = u_big.div_rem(&v_big);
            let mut q_bytes = Vec::with_capacity(40);
            for l in &q {
                q_bytes.extend_from_slice(&l.to_le_bytes());
            }
            assert_eq!(BigUint::from_bytes_le(&q_bytes), q_big);
            assert_eq!(to_biguint_256(&r), r_big);
        }
    }

    #[test]
    fn test_div_rem_edge_cases() {
        for (m, m_big) in moduli() {
            let v = m.limbs;
            let cases: Vec<U512> = vec![
                [0; 8],
                [1, 0, 0, 0, 0, 0, 0, 0],
                u256_to_u512(&v),
                {
                    let mut x = u256_to_u512(&v);
                    x[0] = x[0].wrapping_sub(1);
                    x
                },
                {
                    let mut x = u256_to_u512(&v);
                    let (s, c) = add_256(&[x[0], x[1], x[2], x[3]], &[1, 0, 0, 0]);
                    x[0] = s[0];
                    x[1] = s[1];
                    x[2] = s[2];
                    x[3] = s[3];
                    x[4] = c;
                    x
                },
                [u64::MAX; 8],
                mul_wide_256(&v, &v),
            ];
            for u in cases {
                let (q, r) = div_rem_512_by_256(&u, &v);
                let (q_big, r_big) = to_biguint_512(&u).div_rem(&m_big);
                let mut q_bytes = Vec::with_capacity(40);
                for l in &q {
                    q_bytes.extend_from_slice(&l.to_le_bytes());
                }
                assert_eq!(BigUint::from_bytes_le(&q_bytes), q_big);
                assert_eq!(to_biguint_256(&r), r_big);
            }
        }
    }

    #[test]
    fn test_mod_inverse_self_consistency() {
        let mut rng = StdRng::seed_from_u64(0x1237);
        for (m, m_big) in moduli() {
            for i in 0..2_000u64 {
                let a = if i == 0 {
                    [1, 0, 0, 0]
                } else if i == 1 {
                    sub_256(&m.limbs, &[1, 0, 0, 0]).0 // p - 1
                } else if i == 2 {
                    [2, 0, 0, 0]
                } else {
                    let x = rng.gen_biguint_below(&m_big);
                    if x.is_zero() {
                        continue;
                    }
                    from_biguint_256(&x)
                };
                let inv = mod_inverse(&a, &m.limbs);
                assert_eq!(cmp_256(&inv, &m.limbs), Ordering::Less);
                let prod = mul_wide_256(&a, &inv);
                let (_, r) = div_rem_512_by_256(&prod, &m.limbs);
                assert_eq!(to_biguint_256(&r), BigUint::one());
            }
        }
    }

    #[test]
    fn test_mod_inverse_matches_fermat_reference() {
        // The generators' reference path computes inverses with
        // `FF::try_inverse` (Fermat exponentiation). The modular inverse is
        // unique in [0, p), but check directly against the field
        // implementation anyway.
        let mut rng = StdRng::seed_from_u64(0xfe12);
        for i in 0..64u64 {
            let x_big = rng.gen_biguint_below(&Secp256K1Base::order());
            if x_big.is_zero() {
                continue;
            }
            let expected = Secp256K1Base::from_noncanonical_biguint(x_big.clone())
                .try_inverse()
                .unwrap()
                .to_canonical_biguint();
            let inv = mod_inverse(&from_biguint_256(&x_big), &SECP256K1_BASE_MODULUS.limbs);
            assert_eq!(to_biguint_256(&inv), expected, "base field iter {i}");

            let y_big = rng.gen_biguint_below(&Secp256K1Scalar::order());
            if y_big.is_zero() {
                continue;
            }
            let expected = Secp256K1Scalar::from_noncanonical_biguint(y_big.clone())
                .try_inverse()
                .unwrap()
                .to_canonical_biguint();
            let inv = mod_inverse(&from_biguint_256(&y_big), &SECP256K1_SCALAR_MODULUS.limbs);
            assert_eq!(to_biguint_256(&inv), expected, "scalar field iter {i}");
        }
        // Edges: 1, 2, p-1, p-2.
        for (m, m_big) in moduli() {
            for edge in [
                BigUint::one(),
                BigUint::from(2u32),
                &m_big - BigUint::one(),
                &m_big - BigUint::from(2u32),
            ] {
                let inv = mod_inverse(&from_biguint_256(&edge), &m.limbs);
                assert_eq!(
                    (to_biguint_256(&inv) * &edge) % &m_big,
                    BigUint::one(),
                    "edge {edge}"
                );
            }
        }
    }

    /// Raw witness values a generator might see: canonical, unreduced
    /// (>= p), zero, the modulus itself, and all-ones.
    fn raw_cases(rng: &mut StdRng, m_big: &BigUint) -> Vec<BigUint> {
        let max: BigUint = (BigUint::one() << 256) - BigUint::one();
        vec![
            BigUint::zero(),
            BigUint::one(),
            m_big.clone(),
            m_big + BigUint::one(),
            m_big - BigUint::one(),
            max.clone(),
            &max - BigUint::one(),
            rng.gen_biguint_below(&max),
            rng.gen_biguint_below(m_big),
            rng.gen_biguint_below(&BigUint::from(u64::MAX)),
        ]
    }

    #[test]
    fn test_add_sub_generator_math_differential() {
        let mut rng = StdRng::seed_from_u64(0xadd5);
        for (m, m_big) in moduli() {
            let mut cases = Vec::new();
            for _ in 0..40 {
                cases.extend(raw_cases(&mut rng, &m_big));
            }
            for a_big in &cases {
                for b_big in cases.iter().step_by(7) {
                    let a = from_biguint_256(a_big);
                    let b = from_biguint_256(b_big);

                    // Reference: original addition generator body.
                    let sum_big = ref_canon(a_big, &m_big) + ref_canon(b_big, &m_big);
                    let (exp_overflow, exp_sum) = if sum_big > m_big {
                        (true, &sum_big - &m_big)
                    } else {
                        (false, sum_big.clone())
                    };
                    let got = add_generator_math(m, &a, &b);
                    assert_eq!(to_biguint_256(&got.sum), exp_sum);
                    assert_eq!(got.overflow, exp_overflow);

                    // Reference: original subtraction generator body.
                    let a_c = ref_canon(a_big, &m_big);
                    let b_c = ref_canon(b_big, &m_big);
                    let (exp_diff, exp_overflow) = if a_c >= b_c {
                        (&a_c - &b_c, false)
                    } else {
                        (&m_big + &a_c - &b_c, true)
                    };
                    let got = sub_generator_math(m, &a, &b);
                    assert_eq!(to_biguint_256(&got.diff), exp_diff);
                    assert_eq!(got.overflow, exp_overflow);
                }
            }
        }
    }

    #[test]
    fn test_mul_generator_math_differential() {
        let mut rng = StdRng::seed_from_u64(0x30125);
        for (m, m_big) in moduli() {
            let mut cases = Vec::new();
            for _ in 0..20 {
                cases.extend(raw_cases(&mut rng, &m_big));
            }
            for a_big in &cases {
                for b_big in cases.iter().step_by(5) {
                    let got = mul_generator_math(m, &from_biguint_256(a_big), &from_biguint_256(b_big));
                    let prod = ref_canon(a_big, &m_big) * ref_canon(b_big, &m_big);
                    let (exp_q, exp_r) = prod.div_rem(&m_big);
                    assert_eq!(to_biguint_256(&got.prod), exp_r);
                    assert_eq!(to_biguint_256(&got.overflow), exp_q);
                }
            }
        }
    }

    #[test]
    fn test_mul_div_generator_math_differential() {
        let mut rng = StdRng::seed_from_u64(0x310d1);
        for (m, m_big) in moduli() {
            let mut cases = Vec::new();
            for _ in 0..6 {
                cases.extend(raw_cases(&mut rng, &m_big));
            }
            for a_big in &cases {
                for (bi, b_big) in cases.iter().enumerate().step_by(5) {
                    let d_big = &cases[(bi + 3) % cases.len()];
                    let a_c = ref_canon(a_big, &m_big);
                    let b_c = ref_canon(b_big, &m_big);
                    let d_c = ref_canon(d_big, &m_big);
                    let got = mul_div_generator_math(
                        m,
                        &from_biguint_256(a_big),
                        &from_biguint_256(b_big),
                        &from_biguint_256(d_big),
                    );
                    if d_c.is_zero() {
                        assert!(got.is_none());
                        continue;
                    }
                    let got = got.unwrap();
                    // Reference: original mul-div generator body.
                    let d_inv = Secp256K1BaseAgnosticInverse::inverse(&d_c, &m_big);
                    let prod = &a_c * &b_c * &d_inv;
                    let (_, exp_result) = prod.div_rem(&m_big);
                    assert_eq!(to_biguint_256(&got.result), exp_result);
                    let lhs = &a_c * &b_c;
                    let rhs = &d_c * &exp_result;
                    if lhs > rhs {
                        let (exp_q, _) = (&lhs - &rhs).div_rem(&m_big);
                        assert!(!got.add_to_lhs);
                        assert_eq!(to_biguint_256(&got.overflow), exp_q);
                    } else {
                        let (exp_q, _) = (&rhs - &lhs).div_rem(&m_big);
                        assert!(got.add_to_lhs);
                        assert_eq!(to_biguint_256(&got.overflow), exp_q);
                    }
                }
            }
        }
    }

    /// BigUint modular inverse reference (extended Euclid via BigUint), used
    /// to keep the mul-div reference test independent of `mod_inverse`.
    struct Secp256K1BaseAgnosticInverse;
    impl Secp256K1BaseAgnosticInverse {
        fn inverse(x: &BigUint, m: &BigUint) -> BigUint {
            // x^(m-2) mod m (m prime), matching FF::try_inverse.
            x.modpow(&(m - BigUint::from(2u32)), m)
        }
    }

    #[test]
    fn test_inverse_and_division_generator_math_differential() {
        let mut rng = StdRng::seed_from_u64(0xd1015);
        for (m, m_big) in moduli() {
            let mut cases = Vec::new();
            for _ in 0..12 {
                cases.extend(raw_cases(&mut rng, &m_big));
            }
            for x_big in &cases {
                let x_c = ref_canon(x_big, &m_big);
                // Inverse generator.
                let got = inverse_generator_math(m, &from_biguint_256(x_big));
                if x_c.is_zero() {
                    assert!(got.is_none());
                } else {
                    let got = got.unwrap();
                    let exp_inv = Secp256K1BaseAgnosticInverse::inverse(&x_c, &m_big);
                    let (exp_div, _) = (&x_c * &exp_inv).div_rem(&m_big);
                    assert_eq!(to_biguint_256(&got.inv), exp_inv);
                    assert_eq!(to_biguint_256(&got.div), exp_div);
                }
            }
            for a_big in &cases {
                for b_big in cases.iter().step_by(7) {
                    // Division generator (uses raw a and b).
                    let b_c = ref_canon(b_big, &m_big);
                    let got = division_generator_math(m, &from_biguint_256(a_big), &from_biguint_256(b_big));
                    if b_c.is_zero() {
                        assert!(got.is_none());
                        continue;
                    }
                    let got = got.unwrap();
                    let b_inv = Secp256K1BaseAgnosticInverse::inverse(&b_c, &m_big);
                    let (_, exp_div) = (a_big * &b_inv).div_rem(&m_big);
                    assert_eq!(to_biguint_256(&got.div), exp_div);
                    let b_times_div = b_big * &exp_div;
                    let (exp_overflow, exp_add_to_a) = if b_times_div <= *a_big {
                        ((a_big - &b_times_div).div_rem(&m_big).0, false)
                    } else {
                        ((&b_times_div - a_big).div_rem(&m_big).0, true)
                    };
                    assert_eq!(got.add_to_a, exp_add_to_a);
                    assert_eq!(to_biguint_256(&got.overflow), exp_overflow);
                }
            }
        }
    }

    #[test]
    fn test_digit_conversions() {
        let mut rng = StdRng::seed_from_u64(0xd161);
        for _ in 0..1_000 {
            let x = rand_u256(&mut rng);
            let digits = u256_digits(&x);
            let mut expected = to_biguint_256(&x).to_u32_digits();
            expected.resize(8, 0);
            assert_eq!(digits.to_vec(), expected);

            let q: [u64; 5] = core::array::from_fn(|_| rng.r#gen());
            let digits = u320_digits(&q);
            let mut bytes = Vec::new();
            for l in &q {
                bytes.extend_from_slice(&l.to_le_bytes());
            }
            let mut expected = BigUint::from_bytes_le(&bytes).to_u32_digits();
            expected.resize(10, 0);
            assert_eq!(digits.to_vec(), expected);
        }
    }
}
