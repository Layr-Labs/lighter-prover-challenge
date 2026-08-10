use alloc::vec::Vec;
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
/// The fixed 16-term quadratic dot product keeps the high limb below 2^7.
#[inline(always)]
fn u160_add_product(lo: &mut u128, hi: &mut u32, a: u64, b: u64) {
    let (sum, carry) = lo.overflowing_add((a as u128) * (b as u128));
    *lo = sum;
    *hi += carry as u32;
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
/// and `reduce160`'s `2^160 - 2^128 + 2^96` precondition holds for any
/// `n <= 2^32 - 2`. The asserted bound below is far stricter than either
/// limit and covers every production batch (a few hundred polynomials).
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
    for (i, o) in out.iter_mut().enumerate() {
        let (mut lo0, mut hi0) = (0u128, 0u32);
        let (mut lo1, mut hi1) = (0u128, 0u32);
        for &(p, QuadraticExtension([b0, b1])) in &full {
            // SAFETY: every slice in `full` has length exactly `out.len()`.
            let c = unsafe { p.get_unchecked(i).0 };
            u160_add_product(&mut lo0, &mut hi0, b0.0, c);
            u160_add_product(&mut lo1, &mut hi1, b1.0, c);
        }
        for &(p, QuadraticExtension([b0, b1])) in &partial {
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

/// `a * b + c` in GF(p^2), folding the addend into the 160-bit accumulators
/// so each output limb still costs exactly one `reduce160` — the plain
/// `ext2_mul` cost, with the two canonicalizing extension adds of a separate
/// `+ c` deleted.
///
/// Bounds (all raw limbs below 2^64): the first-limb accumulator is below
/// `(1 + 7) * 2^128 + 2^64 < 2^131` after the `u160_times_7` fold, the
/// second below `2 * 2^128 + 2^64` — both far under `reduce160`'s
/// `2^160 - 2^128 + 2^96` precondition. Field-equal to `a * b + c`; the raw
/// representative may differ from the mul-then-add spelling.
#[inline(always)]
pub fn ext2_mul_add(
    a: QuadraticExtension<GoldilocksField>,
    b: QuadraticExtension<GoldilocksField>,
    c: QuadraticExtension<GoldilocksField>,
) -> QuadraticExtension<GoldilocksField> {
    const_assert!(<GoldilocksField as Extendable<2>>::W.0 == 7u64);
    let QuadraticExtension([a0, a1]) = a;
    let QuadraticExtension([b0, b1]) = b;
    let QuadraticExtension([c0, c1]) = c;

    let (mut plain_lo, mut plain_hi) = (c0.0 as u128, 0u32);
    let (mut w_lo, mut w_hi) = (0u128, 0u32);
    let (mut out1_lo, mut out1_hi) = (c1.0 as u128, 0u32);
    u160_add_product(&mut plain_lo, &mut plain_hi, a0.0, b0.0);
    u160_add_product(&mut w_lo, &mut w_hi, a1.0, b1.0);
    u160_add_product(&mut out1_lo, &mut out1_hi, a0.0, b1.0);
    u160_add_product(&mut out1_lo, &mut out1_hi, a1.0, b0.0);

    let (w_lo, w_hi) = u160_times_7(w_lo, w_hi);
    let (out0_lo, carry) = plain_lo.overflowing_add(w_lo);
    let out0_hi = plain_hi + w_hi + carry as u32;

    // SAFETY: the accumulator bounds documented above are far below
    // reduce160's precondition.
    QuadraticExtension([
        unsafe { reduce160(out0_lo, out0_hi) },
        unsafe { reduce160(out1_lo, out1_hi) },
    ])
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
    use crate::extension::quintic::{QuinticExtension, QuinticFirstCoeff};
    use crate::extension::{Extendable, FieldExtension, Frobenius, OEF};
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::{Field, Field64, PrimeField64};

    type GF = GoldilocksField;
    type Q2 = QuadraticExtension<GoldilocksField>;
    type QE = QuinticExtension<GoldilocksField>;

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
}
