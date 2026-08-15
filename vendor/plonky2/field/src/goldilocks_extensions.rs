use alloc::vec::Vec;
#[cfg(target_arch = "aarch64")]
use core::arch::asm;
use core::ops::Mul;

use static_assertions::const_assert;
use unroll::unroll_for_loops;

use crate::extension::quadratic::QuadraticExtension;
use crate::extension::quartic::QuarticExtension;
use crate::extension::quintic::{QuinticExtension, QuinticFirstCoeff};
use crate::extension::{Extendable, Frobenius};
use crate::goldilocks_field::{reduce160, GoldilocksField};
use crate::types::Field;

impl Frobenius<1> for GoldilocksField {}

impl Extendable<2> for GoldilocksField {
    type Extension = QuadraticExtension<Self>;

    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^2 - 7).is_irreducible()`.
    const W: Self = Self(7);

    // DTH_ROOT = W^((ORDER - 1)/2)
    const DTH_ROOT: Self = Self(18446744069414584320);

    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 2] = [Self(0), Self(11713931119993638672)];

    const EXT_POWER_OF_TWO_GENERATOR: [Self; 2] = [Self(0), Self(7226896044987257365)];

    #[inline]
    fn extension_base_dot_product(
        extension_values: &[QuadraticExtension<Self>],
        base_scalars: &[Self],
    ) -> QuadraticExtension<Self> {
        ext2_base_scalar_dot_product(extension_values, base_scalars)
    }

    #[inline(always)]
    fn mul_fft_quadratic_base_twiddle(twiddle: [Self; 2], value: [Self; 2]) -> [Self; 2] {
        // FFT rows below the quadratic extension's extra two-adic level
        // contain [w, 0], so scalar-multiply the two value limbs. Each limb
        // is one widening product instead of general ext2's four total.
        let [w, _] = twiddle;
        let [a0, a1] = value;
        [w * a0, w * a1]
    }

    #[inline(always)]
    fn fri_fold_arity16(
        terms: &[QuadraticExtension<Self>; 16],
        _beta: QuadraticExtension<Self>,
        beta_powers: &[QuadraticExtension<Self>; 16],
    ) -> QuadraticExtension<Self> {
        ext2_dot_product_arity16(terms, beta_powers)
    }
}

impl Mul for QuadraticExtension<GoldilocksField> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1]) = self;
        let Self([b0, b1]) = rhs;
        let c = ext2_mul([a0.0, a1.0], [b0.0, b1.0]);
        Self(c)
    }
}

impl Extendable<4> for GoldilocksField {
    type Extension = QuarticExtension<Self>;

    const W: Self = Self(7);

    // DTH_ROOT = W^((ORDER - 1)/4)
    const DTH_ROOT: Self = Self(281474976710656);

    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 4] =
        [Self(0), Self(8295451483910296135), Self(0), Self(0)];

    const EXT_POWER_OF_TWO_GENERATOR: [Self; 4] =
        [Self(0), Self(0), Self(0), Self(17216955519093520442)];
}

impl Mul for QuarticExtension<GoldilocksField> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3]) = self;
        let Self([b0, b1, b2, b3]) = rhs;
        let c = ext4_mul([a0.0, a1.0, a2.0, a3.0], [b0.0, b1.0, b2.0, b3.0]);
        Self(c)
    }
}

impl Extendable<5> for GoldilocksField {
    type Extension = QuinticExtension<Self>;

    const W: Self = Self(3);

    // DTH_ROOT = W^((ORDER - 1)/5)
    const DTH_ROOT: Self = Self(1041288259238279555);

    const EXT_MULTIPLICATIVE_GROUP_GENERATOR: [Self; 5] = [
        Self(4624713872807171977),
        Self(381988216716071028),
        Self(14499722700050429911),
        Self(4870631734967222356),
        Self(4518902370426242880),
    ];

    const EXT_POWER_OF_TWO_GENERATOR: [Self; 5] = [
        Self::POWER_OF_TWO_GENERATOR,
        Self(0),
        Self(0),
        Self(0),
        Self(0),
    ];
}

impl Mul for QuinticExtension<GoldilocksField> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let Self([a0, a1, a2, a3, a4]) = self;
        let Self([b0, b1, b2, b3, b4]) = rhs;
        let c = ext5_mul(
            [a0.0, a1.0, a2.0, a3.0, a4.0],
            [b0.0, b1.0, b2.0, b3.0, b4.0],
        );
        Self(c)
    }
}

impl Frobenius<5> for QuinticExtension<GoldilocksField> {
    fn repeated_frobenius(&self, count: usize) -> Self {
        // The code below assumes DTH_ROOT = W^((p - 1)/5) = 1041288259238279555,
        // which has multiplicative order 5.
        const_assert!(
            <GoldilocksField as Extendable<5>>::DTH_ROOT.0 == 1041288259238279555u64
        );

        // FROB_COEFFS[c - 1][i - 1] = DTH_ROOT^(c * i mod 5), the coefficient of
        // limb i under the c-fold Frobenius automorphism.
        const FROB_COEFFS: [[GoldilocksField; 4]; 4] = [
            [
                GoldilocksField(1041288259238279555),
                GoldilocksField(15820824984080659046),
                GoldilocksField(211587555138949697),
                GoldilocksField(1373043270956696022),
            ],
            [
                GoldilocksField(15820824984080659046),
                GoldilocksField(1373043270956696022),
                GoldilocksField(1041288259238279555),
                GoldilocksField(211587555138949697),
            ],
            [
                GoldilocksField(211587555138949697),
                GoldilocksField(1041288259238279555),
                GoldilocksField(1373043270956696022),
                GoldilocksField(15820824984080659046),
            ],
            [
                GoldilocksField(1373043270956696022),
                GoldilocksField(211587555138949697),
                GoldilocksField(15820824984080659046),
                GoldilocksField(1041288259238279555),
            ],
        ];

        let count = count % 5;
        if count == 0 {
            return *self;
        }
        let z = &FROB_COEFFS[count - 1];
        let Self([a0, a1, a2, a3, a4]) = *self;
        Self([a0, a1 * z[0], a2 * z[1], a3 * z[2], a4 * z[3]])
    }
}

impl QuinticFirstCoeff<GoldilocksField> for QuinticExtension<GoldilocksField> {
    #[inline]
    fn mul_first_coeff(&self, rhs: &Self) -> GoldilocksField {
        let Self([a0, a1, a2, a3, a4]) = *self;
        let Self([b0, b1, b2, b3, b4]) = *rhs;
        ext5_add_prods0(
            &[a0.0, a1.0, a2.0, a3.0, a4.0],
            &[b0.0, b1.0, b2.0, b3.0, b4.0],
        )
    }
}

/*
 * The functions extD_add_prods[0-4] are helper functions for
 * computing products for extensions of degree D over the Goldilocks
 * field. They are faster than the generic method because all
 * reductions are delayed until the end which means only one per
 * result coefficient is necessary.
 */

/// Return `a`, `b` such that `a + b*2^128 = 3*(x + y*2^128)` with `a < 2^128` and `b < 2^32`.
#[inline(always)]
const fn u160_times_3(x: u128, y: u32) -> (u128, u32) {
    let (s, cy) = x.overflowing_add(x << 1);
    (s, 3 * y + (x >> 127) as u32 + cy as u32)
}

/// Return `a`, `b` such that `a + b*2^128 = 7*(x + y*2^128)` with `a < 2^128` and `b < 2^32`.
#[inline(always)]
const fn u160_times_7(x: u128, y: u32) -> (u128, u32) {
    let (d, br) = (x << 3).overflowing_sub(x);
    // NB: subtracting the borrow can't underflow
    (d, 7 * y + (x >> (128 - 3)) as u32 - br as u32)
}

/// Add one 64-by-64-bit product to a little-endian 160-bit accumulator.
/// Callers bound their term counts so the high limb cannot overflow.
#[inline(always)]
fn u160_add_product(lo: &mut u128, hi: &mut u32, a: u64, b: u64) {
    let (sum, carry) = lo.overflowing_add((a as u128) * (b as u128));
    *lo = sum;
    *hi += carry as u32;
}

/// Split 160-bit accumulator used by the AArch64 pair kernel so the inner
/// loop never re-packs a `u128` between terms.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct U160Acc {
    lo: u64,
    mid: u64,
    hi: u64,
}

#[cfg(target_arch = "aarch64")]
impl U160Acc {
    const ZERO: Self = Self {
        lo: 0,
        mid: 0,
        hi: 0,
    };

    #[inline(always)]
    fn reduce(self) -> GoldilocksField {
        // SAFETY: callers inherit `ext2_base_scalar_dot_slots`'s term-count
        // bound, which is far under `reduce160`'s precondition.
        unsafe { reduce160((self.lo as u128) | ((self.mid as u128) << 64), self.hi as u32) }
    }
}

/// Two independent `u160_add_product` steps, interleaved so the `umulh`/`mul`
/// pair for each lane hides the other's latency. Accumulators stay in the
/// split `(lo, mid, hi)` form; the mathematical 160-bit sum is identical to
/// two scalar adds.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn u160_add_product_pair(
    acc0: &mut U160Acc,
    a0: u64,
    b0: u64,
    acc1: &mut U160Acc,
    a1: u64,
    b1: u64,
) {
    let mut lo0 = acc0.lo;
    let mut mid0 = acc0.mid;
    let mut hi0 = acc0.hi;
    let mut lo1 = acc1.lo;
    let mut mid1 = acc1.mid;
    let mut hi1 = acc1.hi;
    let prod0 = a0;
    let prod1 = a1;

    unsafe {
        asm!(
            "umulh {ph0}, {prod0}, {scratch0}",
            "umulh {ph1}, {prod1}, {scratch1}",
            "mul   {prod0}, {prod0}, {scratch0}",
            "mul   {prod1}, {prod1}, {scratch1}",
            "adds  {lo0}, {lo0}, {prod0}",
            "adcs  {mid0}, {mid0}, {ph0}",
            "adc   {hi0}, {hi0}, xzr",
            "adds  {lo1}, {lo1}, {prod1}",
            "adcs  {mid1}, {mid1}, {ph1}",
            "adc   {hi1}, {hi1}, xzr",
            prod0 = inout(reg) prod0 => _,
            prod1 = inout(reg) prod1 => _,
            scratch0 = in(reg) b0,
            scratch1 = in(reg) b1,
            ph0 = out(reg) _,
            ph1 = out(reg) _,
            lo0 = inout(reg) lo0,
            mid0 = inout(reg) mid0,
            hi0 = inout(reg) hi0,
            lo1 = inout(reg) lo1,
            mid1 = inout(reg) mid1,
            hi1 = inout(reg) hi1,
            options(pure, nomem, nostack),
        );
    }

    acc0.lo = lo0;
    acc0.mid = mid0;
    acc0.hi = hi0;
    acc1.lo = lo1;
    acc1.mid = mid1;
    acc1.hi = hi1;
}

/// Compute `sum_i extension_values[i].scalar_mul(base_scalars[i])` in
/// GF(p^2), delaying reduction across the complete dot product.
///
/// The iterator-compatible result uses the shorter input length. Raw
/// Goldilocks limbs, including non-canonical representatives, are at most
/// `2^64 - 1`; therefore each limb product is at most `(2^64 - 1)^2`.
/// A chunk contains at most `2^32 - 1` terms, whose worst-case sum is
///
/// ```text
/// (2^32 - 1)(2^64 - 1)^2
///   = 2^160 - 2^128 - 2^97 + 2^65 + 2^32 - 1
///   < 2^160 - 2^128 + 2^96.
/// ```
///
/// This is exactly `reduce160`'s precondition. It also leaves the u32 high
/// accumulator below `2^32 - 1`, so `u160_add_product` cannot overflow it.
/// Inputs longer than one safe chunk are reduced chunk-wise; production
/// openings are many orders of magnitude smaller and take the one-reduction
/// path. The returned representative need not match reduce-per-term addition,
/// but it represents the same field element.
#[inline]
fn ext2_base_scalar_dot_product(
    extension_values: &[QuadraticExtension<GoldilocksField>],
    base_scalars: &[GoldilocksField],
) -> QuadraticExtension<GoldilocksField> {
    const MAX_TERMS_PER_REDUCTION: usize = u32::MAX as usize;

    let len = extension_values.len().min(base_scalars.len());
    if len == 0 {
        return QuadraticExtension::ZERO;
    }

    let reduce_chunk = |values: &[QuadraticExtension<GoldilocksField>],
                        scalars: &[GoldilocksField]| {
        debug_assert_eq!(values.len(), scalars.len());
        debug_assert!(values.len() <= MAX_TERMS_PER_REDUCTION);
        let (mut lo0, mut hi0) = (0u128, 0u32);
        let (mut lo1, mut hi1) = (0u128, 0u32);
        for (&QuadraticExtension([a0, a1]), &scalar) in values.iter().zip(scalars) {
            u160_add_product(&mut lo0, &mut hi0, a0.0, scalar.0);
            u160_add_product(&mut lo1, &mut hi1, a1.0, scalar.0);
        }
        // SAFETY: the exact worst-case bound above covers arbitrary u64
        // representatives for every term in this chunk.
        QuadraticExtension([unsafe { reduce160(lo0, hi0) }, unsafe {
            reduce160(lo1, hi1)
        }])
    };

    let first_end = len.min(MAX_TERMS_PER_REDUCTION);
    let mut result = reduce_chunk(&extension_values[..first_end], &base_scalars[..first_end]);
    let mut start = first_end;
    while start < len {
        let end = len.min(start + MAX_TERMS_PER_REDUCTION);
        result += reduce_chunk(&extension_values[start..end], &base_scalars[start..end]);
        start = end;
    }
    result
}

/// Compute `sum_i terms[i] * powers[i]` in GF(p^2), delaying reduction
/// across the complete production FRI arity. For raw limbs below 2^64,
///
/// - c0 < 16 * (1 + 7) * 2^128 = 2^135;
/// - c1 < 16 * 2 * 2^128 = 2^133.
///
/// Both coefficients therefore satisfy `reduce160`'s bound with ample room.
#[inline(always)]
#[unroll_for_loops]
fn ext2_dot_product_arity16(
    terms: &[QuadraticExtension<GoldilocksField>; 16],
    powers: &[QuadraticExtension<GoldilocksField>; 16],
) -> QuadraticExtension<GoldilocksField> {
    const_assert!(<GoldilocksField as Extendable<2>>::W.0 == 7u64);

    let (mut c0_plain_lo, mut c0_plain_hi) = (0u128, 0u32);
    let (mut c0_w_lo, mut c0_w_hi) = (0u128, 0u32);
    let (mut c1_lo, mut c1_hi) = (0u128, 0u32);

    for i in 0..16 {
        let QuadraticExtension([a0, a1]) = terms[i];
        let QuadraticExtension([b0, b1]) = powers[i];
        u160_add_product(&mut c0_plain_lo, &mut c0_plain_hi, a0.0, b0.0);
        u160_add_product(&mut c0_w_lo, &mut c0_w_hi, a1.0, b1.0);
        u160_add_product(&mut c1_lo, &mut c1_hi, a0.0, b1.0);
        u160_add_product(&mut c1_lo, &mut c1_hi, a1.0, b0.0);
    }

    let (c0_w_lo, c0_w_hi) = u160_times_7(c0_w_lo, c0_w_hi);
    let (c0_lo, carry) = c0_plain_lo.overflowing_add(c0_w_lo);
    let c0_hi = c0_plain_hi + c0_w_hi + carry as u32;

    // SAFETY: the bounds documented above are far below reduce160's
    // `2^160 - 2^128 + 2^96` precondition.
    let c0 = unsafe { reduce160(c0_lo, c0_hi) };
    let c1 = unsafe { reduce160(c1_lo, c1_hi) };
    QuadraticExtension([c0, c1])
}

/// For each output slot `i`, compute
/// `out[i] = sum_j powers[j].scalar_mul(polys[j][start + i])`
/// over every polynomial long enough to reach that slot, delaying modular
/// reduction across the whole polynomial batch: one `reduce160` per extension
/// limb per slot instead of one `reduce128` per limb per *term* plus a
/// canonicalizing extension add per term.
///
/// Scalar multiplication by a base-field coefficient never mixes the two
/// extension limbs, so each limb is a plain dot product of 64-bit raw
/// representatives:
///
/// - `limb0 = sum_j powers[j].0[0] * c_j`
/// - `limb1 = sum_j powers[j].0[1] * c_j`
///
/// Each 64x64 product is below `2^128`, so after `n` terms the 160-bit
/// accumulator holds less than `n * 2^128`: its high limb stays below `n`,
/// and the exact worst-case calculation above shows that `reduce160`'s
/// `2^160 - 2^128 + 2^96` precondition holds through `n = 2^32 - 1`.
/// The asserted bound below is far stricter than that limit and covers every
/// production batch (a few hundred polynomials).
///
/// The result is the same field element as the reduce-per-term form; the raw
/// representative may differ (both forms produce sub-2^64 representatives
/// that later consumers treat value-wise, and proof serialization
/// canonicalizes every limb).
pub fn ext2_base_scalar_dot_slots(
    out: &mut [QuadraticExtension<GoldilocksField>],
    start: usize,
    polys: &[&[GoldilocksField]],
    powers: &[QuadraticExtension<GoldilocksField>],
) {
    assert_eq!(polys.len(), powers.len());
    assert!(polys.len() < 1 << 24);
    let end = start + out.len();
    // Split once so the dense inner loop over fully-covering polynomials
    // runs without per-slot bounds checks; only boundary-length polynomials
    // take the checked loop.
    let mut full: Vec<(&[GoldilocksField], QuadraticExtension<GoldilocksField>)> =
        Vec::with_capacity(polys.len());
    let mut partial: Vec<(&[GoldilocksField], QuadraticExtension<GoldilocksField>)> = Vec::new();
    for (&p, &pw) in polys.iter().zip(powers) {
        if p.len() >= end {
            full.push((&p[start..end], pw));
        } else if p.len() > start {
            partial.push((&p[start..], pw));
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        ext2_base_scalar_dot_slots_neon(out, &full, &partial);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        ext2_base_scalar_dot_slots_scalar(out, &full, &partial);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn ext2_base_scalar_dot_slots_scalar(
    out: &mut [QuadraticExtension<GoldilocksField>],
    full: &[(&[GoldilocksField], QuadraticExtension<GoldilocksField>)],
    partial: &[(&[GoldilocksField], QuadraticExtension<GoldilocksField>)],
) {
    for (i, o) in out.iter_mut().enumerate() {
        let (mut lo0, mut hi0) = (0u128, 0u32);
        let (mut lo1, mut hi1) = (0u128, 0u32);
        for &(p, QuadraticExtension([b0, b1])) in full {
            // SAFETY: every slice in `full` has length exactly `out.len()`.
            let c = unsafe { p.get_unchecked(i).0 };
            u160_add_product(&mut lo0, &mut hi0, b0.0, c);
            u160_add_product(&mut lo1, &mut hi1, b1.0, c);
        }
        for &(p, QuadraticExtension([b0, b1])) in partial {
            if i < p.len() {
                let c = p[i].0;
                u160_add_product(&mut lo0, &mut hi0, b0.0, c);
                u160_add_product(&mut lo1, &mut hi1, b1.0, c);
            }
        }
        // SAFETY: the accumulator bound documented above — below
        // `polys.len() * 2^128 < 2^152` — is far under reduce160's
        // precondition.
        *o = QuadraticExtension([
            unsafe { reduce160(lo0, hi0) },
            unsafe { reduce160(lo1, hi1) },
        ]);
    }
}

/// AArch64 path: two output slots (or the two extension limbs of one leftover
/// slot) share one interleaved `umulh`/`mul` u160 pair so the widening
/// multiplies stay in flight. Coefficients are copied into a local `[u64; 2]`
/// rather than reinterpreting a possibly short slice as a packed view.
#[cfg(target_arch = "aarch64")]
fn ext2_base_scalar_dot_slots_neon(
    out: &mut [QuadraticExtension<GoldilocksField>],
    full: &[(&[GoldilocksField], QuadraticExtension<GoldilocksField>)],
    partial: &[(&[GoldilocksField], QuadraticExtension<GoldilocksField>)],
) {
    let n = out.len();
    let mut i = 0;
    while i + 1 < n {
        let mut s0l0 = U160Acc::ZERO;
        let mut s0l1 = U160Acc::ZERO;
        let mut s1l0 = U160Acc::ZERO;
        let mut s1l1 = U160Acc::ZERO;
        for &(p, QuadraticExtension([b0, b1])) in full {
            // SAFETY: every slice in `full` has length exactly `out.len()`.
            let coeffs = [
                unsafe { p.get_unchecked(i).0 },
                unsafe { p.get_unchecked(i + 1).0 },
            ];
            u160_add_product_pair(&mut s0l0, b0.0, coeffs[0], &mut s1l0, b0.0, coeffs[1]);
            u160_add_product_pair(&mut s0l1, b1.0, coeffs[0], &mut s1l1, b1.0, coeffs[1]);
        }
        for &(p, QuadraticExtension([b0, b1])) in partial {
            if i + 1 < p.len() {
                let coeffs = [p[i].0, p[i + 1].0];
                u160_add_product_pair(&mut s0l0, b0.0, coeffs[0], &mut s1l0, b0.0, coeffs[1]);
                u160_add_product_pair(&mut s0l1, b1.0, coeffs[0], &mut s1l1, b1.0, coeffs[1]);
            } else if i < p.len() {
                let c = p[i].0;
                u160_add_product_pair(&mut s0l0, b0.0, c, &mut s0l1, b1.0, c);
            }
        }
        out[i] = QuadraticExtension([s0l0.reduce(), s0l1.reduce()]);
        out[i + 1] = QuadraticExtension([s1l0.reduce(), s1l1.reduce()]);
        i += 2;
    }
    if i < n {
        let mut acc0 = U160Acc::ZERO;
        let mut acc1 = U160Acc::ZERO;
        for &(p, QuadraticExtension([b0, b1])) in full {
            // SAFETY: every slice in `full` has length exactly `out.len()`.
            let c = unsafe { p.get_unchecked(i).0 };
            u160_add_product_pair(&mut acc0, b0.0, c, &mut acc1, b1.0, c);
        }
        for &(p, QuadraticExtension([b0, b1])) in partial {
            if i < p.len() {
                let c = p[i].0;
                u160_add_product_pair(&mut acc0, b0.0, c, &mut acc1, b1.0, c);
            }
        }
        out[i] = QuadraticExtension([acc0.reduce(), acc1.reduce()]);
    }
}

/*
 * Quadratic multiplication and squaring
 */

#[inline(always)]
fn ext2_add_prods0(a: &[u64; 2], b: &[u64; 2]) -> GoldilocksField {
    // Computes a0 * b0 + W * a1 * b1;
    let [a0, a1] = *a;
    let [b0, b1] = *b;

    let cy;

    // W * a1 * b1
    let (mut cumul_lo, mut cumul_hi) = u160_times_7((a1 as u128) * (b1 as u128), 0u32);

    // a0 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext2_add_prods1(a: &[u64; 2], b: &[u64; 2]) -> GoldilocksField {
    // Computes a0 * b1 + a1 * b0;
    let [a0, a1] = *a;
    let [b0, b1] = *b;

    let cy;

    // a0 * b1
    let mut cumul_lo = (a0 as u128) * (b1 as u128);

    // a1 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
    let cumul_hi = cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

/// Multiply a and b considered as elements of GF(p^2).
#[inline(always)]
pub(crate) fn ext2_mul(a: [u64; 2], b: [u64; 2]) -> [GoldilocksField; 2] {
    // The code in ext2_add_prods[01] assumes the quadratic extension
    // generator is 7.
    const_assert!(<GoldilocksField as Extendable<2>>::W.0 == 7u64);

    let c0 = ext2_add_prods0(&a, &b);
    let c1 = ext2_add_prods1(&a, &b);
    [c0, c1]
}

/*
 * Quartic multiplication and squaring
 */

#[inline(always)]
fn ext4_add_prods0(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
    // Computes c0 = a0 * b0 + W * (a1 * b3 + a2 * b2 + a3 * b1)

    let [a0, a1, a2, a3] = *a;
    let [b0, b1, b2, b3] = *b;

    let mut cy;

    // a1 * b3
    let mut cumul_lo = (a1 as u128) * (b3 as u128);

    // a2 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b2 as u128));
    let mut cumul_hi = cy as u32;

    // a3 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_7(cumul_lo, cumul_hi);

    // a0 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext4_add_prods1(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
    // Computes c1 = a0 * b1 + a1 * b0 + W * (a2 * b3 + a3 * b2);

    let [a0, a1, a2, a3] = *a;
    let [b0, b1, b2, b3] = *b;

    let mut cy;

    // a2 * b3
    let mut cumul_lo = (a2 as u128) * (b3 as u128);

    // a3 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b2 as u128));
    let mut cumul_hi = cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_7(cumul_lo, cumul_hi);

    // a0 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a1 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext4_add_prods2(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
    // Computes c2 = a0 * b2 + a1 * b1 + a2 * b0 + W * a3 * b3;

    let [a0, a1, a2, a3] = *a;
    let [b0, b1, b2, b3] = *b;

    let mut cy;

    // W * a3 * b3
    let (mut cumul_lo, mut cumul_hi) = u160_times_7((a3 as u128) * (b3 as u128), 0u32);

    // a0 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a1 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a2 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext4_add_prods3(a: &[u64; 4], b: &[u64; 4]) -> GoldilocksField {
    // Computes c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;

    let [a0, a1, a2, a3] = *a;
    let [b0, b1, b2, b3] = *b;

    let mut cy;

    // a0 * b3
    let mut cumul_lo = (a0 as u128) * (b3 as u128);

    // a1 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b2 as u128));
    let mut cumul_hi = cy as u32;

    // a2 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a3 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

/// Multiply a and b considered as elements of GF(p^4).
#[inline(always)]
pub(crate) fn ext4_mul(a: [u64; 4], b: [u64; 4]) -> [GoldilocksField; 4] {
    // The code in ext4_add_prods[0-3] assumes the quartic extension
    // generator is 7.
    const_assert!(<GoldilocksField as Extendable<4>>::W.0 == 7u64);

    let c0 = ext4_add_prods0(&a, &b);
    let c1 = ext4_add_prods1(&a, &b);
    let c2 = ext4_add_prods2(&a, &b);
    let c3 = ext4_add_prods3(&a, &b);
    [c0, c1, c2, c3]
}

/*
 * Quintic multiplication and squaring
 */

#[inline(always)]
fn ext5_add_prods0(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c0 = a0 * b0 + W * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1)

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // a1 * b4
    let mut cumul_lo = (a1 as u128) * (b4 as u128);

    // a2 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b3 as u128));
    let mut cumul_hi = cy as u32;

    // a3 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a4 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_3(cumul_lo, cumul_hi);

    // a0 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext5_add_prods1(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c1 = a0 * b1 + a1 * b0 + W * (a2 * b4 + a3 * b3 + a4 * b2);

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // a2 * b4
    let mut cumul_lo = (a2 as u128) * (b4 as u128);

    // a3 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b3 as u128));
    let mut cumul_hi = cy as u32;

    // a4 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_3(cumul_lo, cumul_hi);

    // a0 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a1 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext5_add_prods2(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c2 = a0 * b2 + a1 * b1 + a2 * b0 + W * (a3 * b4 + a4 * b3);

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // a3 * b4
    let mut cumul_lo = (a3 as u128) * (b4 as u128);

    // a4 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b3 as u128));
    let mut cumul_hi = cy as u32;

    // * W
    (cumul_lo, cumul_hi) = u160_times_3(cumul_lo, cumul_hi);

    // a0 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a1 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a2 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext5_add_prods3(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + W * a4 * b4;

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // W * a4 * b4
    let (mut cumul_lo, mut cumul_hi) = u160_times_3((a4 as u128) * (b4 as u128), 0u32);

    // a0 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a0 as u128) * (b3 as u128));
    cumul_hi += cy as u32;

    // a1 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a2 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a3 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

#[inline(always)]
fn ext5_add_prods4(a: &[u64; 5], b: &[u64; 5]) -> GoldilocksField {
    // Computes c4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

    let [a0, a1, a2, a3, a4] = *a;
    let [b0, b1, b2, b3, b4] = *b;

    let mut cy;

    // a0 * b4
    let mut cumul_lo = (a0 as u128) * (b4 as u128);

    // a1 * b3
    (cumul_lo, cy) = cumul_lo.overflowing_add((a1 as u128) * (b3 as u128));
    let mut cumul_hi = cy as u32;

    // a2 * b2
    (cumul_lo, cy) = cumul_lo.overflowing_add((a2 as u128) * (b2 as u128));
    cumul_hi += cy as u32;

    // a3 * b1
    (cumul_lo, cy) = cumul_lo.overflowing_add((a3 as u128) * (b1 as u128));
    cumul_hi += cy as u32;

    // a4 * b0
    (cumul_lo, cy) = cumul_lo.overflowing_add((a4 as u128) * (b0 as u128));
    cumul_hi += cy as u32;

    unsafe { reduce160(cumul_lo, cumul_hi) }
}

/// Multiply a and b considered as elements of GF(p^5).
#[inline(always)]
pub(crate) fn ext5_mul(a: [u64; 5], b: [u64; 5]) -> [GoldilocksField; 5] {
    // The code in ext5_add_prods[0-4] assumes the quintic extension
    // generator is 3.
    const_assert!(<GoldilocksField as Extendable<5>>::W.0 == 3u64);

    let c0 = ext5_add_prods0(&a, &b);
    let c1 = ext5_add_prods1(&a, &b);
    let c2 = ext5_add_prods2(&a, &b);
    let c3 = ext5_add_prods3(&a, &b);
    let c4 = ext5_add_prods4(&a, &b);
    [c0, c1, c2, c3, c4]
}

#[cfg(test)]
mod tests {
    use crate::extension::quadratic::QuadraticExtension;
    use crate::extension::quartic::QuarticExtension;
    use crate::extension::quintic::{QuinticExtension, QuinticFirstCoeff};
    use crate::extension::{Extendable, FieldExtension, Frobenius, OEF};
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::{Field, Field64, PrimeField64};

    type GF = GoldilocksField;
    type Q2 = QuadraticExtension<GoldilocksField>;
    type Q4 = QuarticExtension<GoldilocksField>;
    type QE = QuinticExtension<GoldilocksField>;

    fn generic_extension_base_dot_product(values: &[Q2], scalars: &[GF]) -> Q2 {
        values
            .iter()
            .zip(scalars)
            .map(|(&value, &scalar)| <Q2 as FieldExtension<2>>::scalar_mul(&value, scalar))
            .sum()
    }

    #[test]
    fn extension_base_dot_product_default_matches_scalar_mul_sum() {
        let values: Vec<Q4> = (0..17)
            .map(|i| {
                QuarticExtension(core::array::from_fn(|limb| {
                    GoldilocksField(
                        (i as u64 + 1)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15u64.rotate_left(limb as u32)),
                    )
                }))
            })
            .collect();
        let scalars: Vec<GF> = (0..16)
            .map(|i| GoldilocksField((i as u64).wrapping_mul(u64::MAX - 1)))
            .collect();
        let expected: Q4 = values
            .iter()
            .zip(&scalars)
            .map(|(&value, &scalar)| <Q4 as FieldExtension<4>>::scalar_mul(&value, scalar))
            .sum();
        let actual = <GF as Extendable<4>>::extension_base_dot_product(&values, &scalars);
        assert_eq!(actual, expected);
        assert_eq!(
            <GF as Extendable<4>>::extension_base_dot_product(&[], &scalars),
            Q4::ZERO
        );
    }

    #[test]
    fn ext2_extension_base_dot_product_matches_generic_at_boundaries() {
        let p = GF::ORDER;
        let raw_specials = [0, 1, 2, p - 1, p, p + 1, u64::MAX];
        // Zero/one, unequal zip lengths, SIMD/cache-sized powers of two, and
        // the neighboring lengths most likely to expose loop-tail mistakes.
        let lengths = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 1),
            (15, 15),
            (16, 16),
            (17, 17),
            (63, 64),
            (64, 63),
            (65, 65),
            (255, 255),
            (256, 256),
            (257, 257),
            (2047, 2047),
            (2048, 2048),
            (2049, 2049),
            (4095, 4096),
            (4096, 4095),
            (4097, 4097),
        ];

        let mut state = 0xA076_1D64_78BD_642Fu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (values_len, scalars_len) in lengths {
            let values: Vec<Q2> = (0..values_len)
                .map(|i| {
                    let a0 = if i < raw_specials.len() {
                        raw_specials[i]
                    } else {
                        next()
                    };
                    let a1 = if i < raw_specials.len() {
                        raw_specials[raw_specials.len() - 1 - i]
                    } else {
                        next()
                    };
                    QuadraticExtension([GoldilocksField(a0), GoldilocksField(a1)])
                })
                .collect();
            let scalars: Vec<GF> = (0..scalars_len)
                .map(|i| {
                    GoldilocksField(if i < raw_specials.len() {
                        raw_specials[(i * 3) % raw_specials.len()]
                    } else {
                        next()
                    })
                })
                .collect();

            let expected = generic_extension_base_dot_product(&values, &scalars);
            let actual = <GF as Extendable<2>>::extension_base_dot_product(&values, &scalars);
            for limb in 0..2 {
                assert_eq!(
                    actual.0[limb].to_canonical_u64(),
                    expected.0[limb].to_canonical_u64(),
                    "canonical limb {limb} mismatch at ({values_len}, {scalars_len})"
                );
            }
        }
        // Raw representatives are deliberately not part of the assertion:
        // delayed and per-term reduction are required to agree as field
        // values, including when every input can occupy the full u64 range.
    }

    #[test]
    fn ext2_extension_base_dot_product_reduce160_bound() {
        use num::BigUint;

        let one = BigUint::from(1u8);
        let max_product = ((&one << 64usize) - &one) * ((&one << 64usize) - &one);
        let reduce160_limit = (&one << 160usize) - (&one << 128usize) + (&one << 96usize);
        let max_safe_sum = BigUint::from(u32::MAX) * &max_product;
        let first_unsafe_worst_case = BigUint::from(u64::from(u32::MAX) + 1) * max_product;
        assert!(max_safe_sum < reduce160_limit);
        assert!(first_unsafe_worst_case >= reduce160_limit);
    }

    #[test]
    fn fri_fold_arity16_matches_horner_raw() {
        let check = |terms: [Q2; 16], beta: Q2| {
            let mut beta_powers = [Q2::ONE; 16];
            for i in 1..16 {
                beta_powers[i] = beta_powers[i - 1] * beta;
            }
            let expected = terms
                .iter()
                .rev()
                .fold(Q2::ZERO, |acc, &term| acc * beta + term);
            let actual = <GF as Extendable<2>>::fri_fold_arity16(
                &terms,
                beta,
                &beta_powers,
            );
            for limb in 0..2 {
                assert_eq!(
                    actual.0[limb].0, expected.0[limb].0,
                    "raw limb {limb} mismatch for beta={beta:?}"
                );
            }
        };

        check([Q2::ZERO; 16], Q2::ZERO);
        check([Q2::ONE; 16], Q2::ONE);

        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let terms = core::array::from_fn(|_| {
                QuadraticExtension(core::array::from_fn(|_| {
                    GF::from_noncanonical_u64(next())
                }))
            });
            let beta = QuadraticExtension(core::array::from_fn(|_| {
                GF::from_noncanonical_u64(next())
            }));
            check(terms, beta);
        }
    }

    /// The generic `Frobenius::repeated_frobenius` default implementation
    /// (from `extension/mod.rs`), reconstructed as the reference oracle.
    fn generic_repeated_frobenius(x: QE, count: usize) -> QE {
        if count == 0 {
            return x;
        } else if count >= 5 {
            return generic_repeated_frobenius(x, count % 5);
        }
        let arr = x.0;

        let mut z0 = <GF as Extendable<5>>::DTH_ROOT;
        for _ in 1..count {
            z0 *= <GF as Extendable<5>>::DTH_ROOT;
        }

        let mut res = [GF::ZERO; 5];
        for (i, z) in z0.powers().take(5).enumerate() {
            res[i] = arr[i] * z;
        }

        QuinticExtension(res)
    }

    #[test]
    fn quintic_frobenius_specialization_matches_generic() {
        let check = |x: QE| {
            for count in 0..=12 {
                let expected = generic_repeated_frobenius(x, count);
                let actual = x.repeated_frobenius(count);
                for j in 0..5 {
                    assert_eq!(
                        actual.0[j].to_canonical_u64(),
                        expected.0[j].to_canonical_u64(),
                        "limb {j} mismatch for count {count}, x={x:?}"
                    );
                }
                // `frobenius` is defined in terms of `repeated_frobenius`.
                if count == 1 {
                    let frob = x.frobenius();
                    for j in 0..5 {
                        assert_eq!(
                            frob.0[j].to_canonical_u64(),
                            expected.0[j].to_canonical_u64()
                        );
                    }
                }
            }
        };

        // Edge cases: 0, 1, 2, -1, scaled basis vectors, a low-order element
        // and non-canonical representations.
        let p = GF::ORDER;
        check(QE::ZERO);
        check(QE::ONE);
        check(QE::TWO);
        check(QE::NEG_ONE);
        check(QuinticExtension([
            <GF as Extendable<5>>::DTH_ROOT,
            GF::ZERO,
            GF::ZERO,
            GF::ZERO,
            GF::ZERO,
        ]));
        for j in 0..5 {
            for v in [1, p - 1, p, u64::MAX] {
                let mut limbs = [GF::ZERO; 5];
                limbs[j] = GoldilocksField(v);
                check(QuinticExtension(limbs));
            }
        }

        // Randomized differential over the full u64 (non-canonical included) range.
        let mut state = 0xB7E1_5162_8AED_2A6Au64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let limbs = core::array::from_fn(|_| GoldilocksField(next()));
            check(QuinticExtension(limbs));
        }
    }

    /// The generic `QuinticFirstCoeff` default implementation (the `c0` row of the generic
    /// `Mul`), reconstructed as the reference oracle for the widening specialization above.
    fn generic_mul_first_coeff(a: QE, b: QE) -> GF {
        let QuinticExtension([a0, a1, a2, a3, a4]) = a;
        let QuinticExtension([b0, b1, b2, b3, b4]) = b;
        a0 * b0
            + <QE as OEF<5>>::W * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1)
    }

    /// The specialized (delayed-reduction) first-coefficient helper must agree with the
    /// generic expression as a field value on edge cases and random non-canonical inputs,
    /// and `try_inverse` (its only consumer) must still return exact inverses.
    #[test]
    fn quintic_first_coeff_specialization_matches_generic() {
        let canon =
            |x: GF| x.to_canonical_u64();
        let check_pair = |a: QE, b: QE| {
            assert_eq!(
                canon(a.mul_first_coeff(&b)),
                canon(generic_mul_first_coeff(a, b)),
                "first coeff mismatch for a={a:?}, b={b:?}"
            );
            // The first coefficient of a full product must also match Mul's c0.
            assert_eq!(canon(a.mul_first_coeff(&b)), canon((a * b).0[0]));
        };

        let p = GF::ORDER;
        let specials = [0u64, 1, 2, p - 1, p, u64::MAX];
        for &u in &specials {
            for &v in &specials {
                let a = QuinticExtension([GoldilocksField(u); 5]);
                let b = QuinticExtension([GoldilocksField(v); 5]);
                check_pair(a, b);
            }
        }

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let a = QuinticExtension(core::array::from_fn(|_| GoldilocksField(next())));
            let b = QuinticExtension(core::array::from_fn(|_| GoldilocksField(next())));
            check_pair(a, b);

            // try_inverse consumes the helper: x * x^-1 == 1 exactly.
            if !a.is_zero() {
                let inv = a.try_inverse().expect("nonzero element must have an inverse");
                let prod = a * inv;
                let limbs = FieldExtension::<5>::to_basefield_array(&prod);
                assert_eq!(canon(limbs[0]), 1, "a * a^-1 != 1 for a={a:?}");
                for limb in &limbs[1..] {
                    assert_eq!(canon(*limb), 0, "a * a^-1 != 1 for a={a:?}");
                }
            }
        }
        assert!(QE::ZERO.try_inverse().is_none());
    }

    fn reference_dot_slots(
        out_len: usize,
        start: usize,
        polys: &[&[GF]],
        powers: &[Q2],
    ) -> Vec<Q2> {
        let mut out = vec![Q2::ZERO; out_len];
        for (&p, &pw) in polys.iter().zip(powers) {
            for (i, slot) in out.iter_mut().enumerate() {
                let idx = start + i;
                if idx < p.len() {
                    *slot += <Q2 as FieldExtension<2>>::scalar_mul(&pw, p[idx]);
                }
            }
        }
        out
    }

    fn assert_dot_slots_canonical(
        out_len: usize,
        start: usize,
        polys: &[&[GF]],
        powers: &[Q2],
    ) {
        let mut actual = vec![Q2::ZERO; out_len];
        super::ext2_base_scalar_dot_slots(&mut actual, start, polys, powers);
        let expected = reference_dot_slots(out_len, start, polys, powers);
        for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
            for limb in 0..2 {
                assert_eq!(
                    got.0[limb].to_canonical_u64(),
                    want.0[limb].to_canonical_u64(),
                    "canonical limb {limb} mismatch at slot {i} (out_len={out_len}, start={start})"
                );
            }
        }
    }

    #[test]
    fn ext2_base_scalar_dot_slots_matches_scalar_mul_sum() {
        let p = GF::ORDER;
        let raw_specials = [0u64, 1, 2, p - 1, p, p + 1, u64::MAX];

        let mut state = 0xC2B2_AE3D_27D4_EB4Fu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // Empty output, empty batch, and a single slot (the leftover path).
        assert_dot_slots_canonical(0, 0, &[], &[]);
        assert_dot_slots_canonical(3, 0, &[], &[]);
        let one_poly = [GF::ONE, GF::TWO];
        let one_power = [Q2::ONE];
        assert_dot_slots_canonical(1, 0, &[&one_poly], &one_power);
        assert_dot_slots_canonical(2, 0, &[&one_poly], &one_power);

        // Lengths that hit the 2-slot kernel, the odd leftover, and mixed
        // full / partial covering relative to `start`.
        let out_lens = [0, 1, 2, 3, 5, 16, 17, 33];
        let starts = [0, 1, 7];
        let poly_lens = [0, 1, 2, 3, 8, 16, 17, 32, 40];

        for &out_len in &out_lens {
            for &start in &starts {
                for n_polys in [0, 1, 2, 3, 7, 16, 17] {
                    let owned: Vec<Vec<GF>> = (0..n_polys)
                        .map(|j| {
                            let len = poly_lens[(j + start) % poly_lens.len()];
                            (0..len)
                                .map(|k| {
                                    let raw = if k < raw_specials.len() {
                                        raw_specials[(k + j) % raw_specials.len()]
                                    } else {
                                        next()
                                    };
                                    GoldilocksField(raw)
                                })
                                .collect()
                        })
                        .collect();
                    let polys: Vec<&[GF]> = owned.iter().map(|v| v.as_slice()).collect();
                    let powers: Vec<Q2> = (0..n_polys)
                        .map(|j| {
                            QuadraticExtension([
                                GoldilocksField(if j < raw_specials.len() {
                                    raw_specials[j]
                                } else {
                                    next()
                                }),
                                GoldilocksField(if j < raw_specials.len() {
                                    raw_specials[raw_specials.len() - 1 - j]
                                } else {
                                    next()
                                }),
                            ])
                        })
                        .collect();
                    assert_dot_slots_canonical(out_len, start, &polys, &powers);
                }
            }
        }

        // Random batches with every raw limb in the full u64 range, including
        // non-canonical representatives.
        for trial in 0..40 {
            let out_len = 1 + (next() as usize % 48);
            let start = next() as usize % 8;
            let n_polys = 1 + (next() as usize % 24);
            let owned: Vec<Vec<GF>> = (0..n_polys)
                .map(|_| {
                    let len = start + out_len + (next() as usize % 5);
                    let len = if trial % 5 == 0 {
                        start + (next() as usize % (out_len + 1))
                    } else {
                        len
                    };
                    (0..len).map(|_| GoldilocksField(next())).collect()
                })
                .collect();
            let polys: Vec<&[GF]> = owned.iter().map(|v| v.as_slice()).collect();
            let powers: Vec<Q2> = (0..n_polys)
                .map(|_| QuadraticExtension([GoldilocksField(next()), GoldilocksField(next())]))
                .collect();
            assert_dot_slots_canonical(out_len, start, &polys, &powers);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn u160_add_product_pair_matches_scalar_u160() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let specials = [
            0u64,
            1,
            2,
            GF::ORDER - 1,
            GF::ORDER,
            GF::ORDER + 1,
            u64::MAX,
        ];
        for &a0 in &specials {
            for &b0 in &specials {
                for &a1 in &specials {
                    for &b1 in &specials {
                        let mut acc0 = super::U160Acc::ZERO;
                        let mut acc1 = super::U160Acc::ZERO;
                        super::u160_add_product_pair(&mut acc0, a0, b0, &mut acc1, a1, b1);
                        let (mut lo0, mut hi0) = (0u128, 0u32);
                        let (mut lo1, mut hi1) = (0u128, 0u32);
                        super::u160_add_product(&mut lo0, &mut hi0, a0, b0);
                        super::u160_add_product(&mut lo1, &mut hi1, a1, b1);
                        assert_eq!(
                            (acc0.lo as u128) | ((acc0.mid as u128) << 64),
                            lo0
                        );
                        assert_eq!(acc0.hi as u32, hi0);
                        assert_eq!(
                            (acc1.lo as u128) | ((acc1.mid as u128) << 64),
                            lo1
                        );
                        assert_eq!(acc1.hi as u32, hi1);
                    }
                }
            }
        }

        let mut acc0 = super::U160Acc::ZERO;
        let mut acc1 = super::U160Acc::ZERO;
        let (mut lo0, mut hi0) = (0u128, 0u32);
        let (mut lo1, mut hi1) = (0u128, 0u32);
        for _ in 0..64 {
            let a0 = next();
            let b0 = next();
            let a1 = next();
            let b1 = next();
            super::u160_add_product_pair(&mut acc0, a0, b0, &mut acc1, a1, b1);
            super::u160_add_product(&mut lo0, &mut hi0, a0, b0);
            super::u160_add_product(&mut lo1, &mut hi1, a1, b1);
        }
        assert_eq!((acc0.lo as u128) | ((acc0.mid as u128) << 64), lo0);
        assert_eq!(acc0.hi as u32, hi0);
        assert_eq!((acc1.lo as u128) | ((acc1.mid as u128) << 64), lo1);
        assert_eq!(acc1.hi as u32, hi1);
    }
}
