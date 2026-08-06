use core::ops::Mul;

use static_assertions::const_assert;

use crate::extension::quadratic::QuadraticExtension;
use crate::extension::quartic::QuarticExtension;
use crate::extension::quintic::{QuinticExtension, QuinticInverse};
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

impl QuinticInverse for QuinticExtension<GoldilocksField> {
    #[inline(always)]
    fn try_inverse_quintic(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        let Self([a0, a1, a2, a3, a4]) = *self;
        let a = [a0.0, a1.0, a2.0, a3.0, a4.0];
        let b = ext5_adjugate(a);
        let norm = ext5_add_prods0(&a, &[b[0].0, b[1].0, b[2].0, b[3].0, b[4].0]);
        let inv_norm = norm.try_inverse()?;

        Some(Self([
            b[0] * inv_norm,
            b[1] * inv_norm,
            b[2] * inv_norm,
            b[3] * inv_norm,
            b[4] * inv_norm,
        ]))
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

/// Unsigned 160-bit accumulator seeded with 16*p^2. The seed is zero modulo p
/// and is larger than the negative part of every expression below.
struct AdjugateAccumulator {
    lo: u128,
    hi: u32,
}

impl AdjugateAccumulator {
    #[inline(always)]
    fn new() -> Self {
        const P: u128 = 0xffff_ffff_0000_0001;
        const P2: u128 = P * P;
        Self {
            lo: P2 << 4,
            hi: (P2 >> 124) as u32,
        }
    }

    #[inline(always)]
    fn scaled_value(p: u128, scale: u32) -> (u128, u32) {
        match scale {
            1 => (p, 0),
            3 => u160_times_3(p, 0),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    fn add_product(&mut self, x: u64, y: u64, scale: u32) {
        self.add_value((x as u128) * (y as u128), scale);
    }

    #[inline(always)]
    fn add_value(&mut self, value: u128, scale: u32) {
        let (lo, hi) = Self::scaled_value(value, scale);
        let (sum, carry) = self.lo.overflowing_add(lo);
        self.lo = sum;
        self.hi += hi + carry as u32;
    }

    #[inline(always)]
    fn sub_product(&mut self, x: u64, y: u64, scale: u32) {
        self.sub_value((x as u128) * (y as u128), scale);
    }

    #[inline(always)]
    fn sub_value(&mut self, value: u128, scale: u32) {
        let (lo, hi) = Self::scaled_value(value, scale);
        let (difference, borrow) = self.lo.overflowing_sub(lo);
        self.lo = difference;
        self.hi -= hi + borrow as u32;
    }

    #[inline(always)]
    fn scale3(self) -> Self {
        let (lo, hi) = u160_times_3(self.lo, self.hi);
        Self { lo, hi }
    }

    #[inline(always)]
    fn add_accumulator(&mut self, rhs: Self) {
        let (sum, carry) = self.lo.overflowing_add(rhs.lo);
        self.lo = sum;
        self.hi += rhs.hi + carry as u32;
    }

    #[inline(always)]
    fn reduce(self) -> GoldilocksField {
        // The largest accumulator is below 81*p^2, well within reduce160's bound.
        unsafe { reduce160(self.lo, self.hi) }
    }
}

#[inline(always)]
fn adjugate_minor(
    positive: u128,
    positive_scale: u32,
    negative: u128,
    negative_scale: u32,
) -> GoldilocksField {
    let mut acc = AdjugateAccumulator::new();
    acc.add_value(positive, positive_scale);
    acc.sub_value(negative, negative_scale);
    acc.reduce()
}

/// Compute the adjugate of multiplication by `a` in GF(p)[X]/(X^5 - 3).
///
/// Splitting the four-row cofactor matrix into two pairs of rows yields ten
/// shared 2x2 minors. The other ten minors are signed/scaled copies. Each
/// cofactor is then a factored Plücker expansion. This takes 15 base products
/// and 10 reductions for the minors, then 25 products and 5 reductions for the
/// adjugate.
#[inline(always)]
fn ext5_adjugate(a: [u64; 5]) -> [GoldilocksField; 5] {
    let [a0, a1, a2, a3, a4] = a;
    let product = |x: u64, y: u64| (x as u128) * (y as u128);

    let q02 = product(a0, a2);
    let n01 = adjugate_minor(product(a1, a1), 1, q02, 1);
    let n24 = adjugate_minor(product(a3, a4), 3, q02, 1);

    let q24 = product(a2, a4);
    let n02 = adjugate_minor(product(a0, a1), 1, q24, 3);
    let n34 = adjugate_minor(product(a3, a3), 1, q24, 1);

    let q14 = product(a1, a4);
    let n03 = adjugate_minor(q14, 1, product(a2, a3), 1);
    let n12 = adjugate_minor(product(a0, a0), 1, q14, 3);

    let q13 = product(a1, a3);
    let n04 = adjugate_minor(q13, 1, product(a2, a2), 1);
    let n13 = adjugate_minor(product(a0, a4), 1, q13, 1);

    let q03 = product(a0, a3);
    let n14 = adjugate_minor(q03, 1, product(a1, a2), 1);
    let n23 = adjugate_minor(product(a4, a4), 3, q03, 1);

    let [n01, n02, n03, n04, n12, n13, n14, n23, n24, n34] = [
        n01.0, n02.0, n03.0, n04.0, n12.0, n13.0, n14.0, n23.0, n24.0, n34.0,
    ];
    let n01_plus_n24 = (GoldilocksField(n01) + GoldilocksField(n24)).0;
    let n02_plus_3n34 = (GoldilocksField(n02) + GoldilocksField(n34).triple()).0;
    let n13_minus_n04 = (GoldilocksField(n13) - GoldilocksField(n04)).0;

    let mut b0 = AdjugateAccumulator::new();
    b0.add_product(n12, n12, 1);
    let mut b0_scaled = AdjugateAccumulator::new();
    b0_scaled.add_product(n14, n01_plus_n24, 1);
    b0_scaled.sub_product(n02, n13, 1);
    b0_scaled.sub_product(n04, n34, 3);
    b0_scaled.sub_product(n23, n24, 1);
    b0.add_accumulator(b0_scaled.scale3());

    let mut b1 = AdjugateAccumulator::new();
    b1.sub_product(n02, n12, 1);
    let mut b1_scaled = AdjugateAccumulator::new();
    b1_scaled.sub_product(n01, n04, 1);
    b1_scaled.add_product(n03, n02_plus_3n34, 1);
    b1_scaled.sub_product(n13, n24, 1);
    b1_scaled.add_product(n23, n23, 1);
    b1.add_accumulator(b1_scaled.scale3());

    let mut b2 = AdjugateAccumulator::new();
    b2.add_product(n01, n12, 1);
    let mut b2_scaled = AdjugateAccumulator::new();
    b2_scaled.add_product(n03, n24, 1);
    b2_scaled.add_product(n14, n13_minus_n04, 1);
    b2_scaled.sub_product(n13, n23, 1);
    b2_scaled.add_product(n34, n34, 3);
    b2.add_accumulator(b2_scaled.scale3());

    let mut b3 = AdjugateAccumulator::new();
    b3.sub_product(n02, n01_plus_n24, 1);
    b3.add_product(n12, n23, 1);
    let mut b3_scaled = AdjugateAccumulator::new();
    b3_scaled.sub_product(n03, n14, 1);
    b3_scaled.add_product(n04, n04, 1);
    b3_scaled.sub_product(n24, n34, 1);
    b3.add_accumulator(b3_scaled.scale3());

    let mut b4 = AdjugateAccumulator::new();
    b4.add_product(n01, n01, 1);
    b4.add_product(n02, n14, 1);
    b4.sub_product(n12, n13, 1);
    let mut b4_scaled = AdjugateAccumulator::new();
    b4_scaled.add_product(n03, n13_minus_n04, 1);
    b4_scaled.add_product(n23, n34, 1);
    b4.add_accumulator(b4_scaled.scale3());

    [
        b0.reduce(),
        b1.reduce(),
        b2.reduce(),
        b3.reduce(),
        b4.reduce(),
    ]
}

/// Return `a`, `b` such that `a + b*2^128 = 7*(x + y*2^128)` with `a < 2^128` and `b < 2^32`.
#[inline(always)]
const fn u160_times_7(x: u128, y: u32) -> (u128, u32) {
    let (d, br) = (x << 3).overflowing_sub(x);
    // NB: subtracting the borrow can't underflow
    (d, 7 * y + (x >> (128 - 3)) as u32 - br as u32)
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
    use crate::extension::quintic::QuinticExtension;
    use crate::extension::{Extendable, FieldExtension, Frobenius};
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::{Field, Field64, PrimeField64};

    type GF = GoldilocksField;
    type QE = QuinticExtension<GoldilocksField>;

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

    /// The previous generic `QuinticExtension::try_inverse` implementation.
    fn frobenius_inverse_reference(x: QE) -> Option<QE> {
        if x.is_zero() {
            return None;
        }

        let d = x.frobenius();
        let e = d * d.frobenius();
        let f = e * e.repeated_frobenius(2);
        let QuinticExtension([a0, a1, a2, a3, a4]) = x;
        let QuinticExtension([b0, b1, b2, b3, b4]) = f;
        let norm = a0 * b0 + <GF as Extendable<5>>::W * (a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1);
        Some(FieldExtension::<5>::scalar_mul(&f, norm.inverse()))
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

    #[test]
    fn quintic_inverse_matches_frobenius_reference_million_cases() {
        let check = |x: QE| {
            assert_eq!(x.try_inverse(), frobenius_inverse_reference(x), "x={x:?}");
        };

        let p = GF::ORDER;
        for x in [QE::ZERO, QE::ONE, QE::TWO, QE::NEG_ONE] {
            check(x);
        }
        for limb in 0..5 {
            for value in [0, 1, 2, p - 1, p, p + 1, u64::MAX] {
                let mut x = [GF::ZERO; 5];
                x[limb] = GoldilocksField(value);
                check(QuinticExtension(x));
            }
        }
        for value in [0, 1, p - 1, p, u64::MAX] {
            check(QuinticExtension([GoldilocksField(value); 5]));
        }

        // Exercise canonical and non-canonical limbs across the full u64 range.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for case in 0..1_000_000 {
            let x = QuinticExtension(core::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                GoldilocksField(state)
            }));
            assert_eq!(
                x.try_inverse(),
                frobenius_inverse_reference(x),
                "differential case {case}, x={x:?}"
            );
        }
    }

    #[test]
    fn quintic_inverse_law_and_zero_handling() {
        let p = GF::ORDER;
        assert_eq!(QE::ZERO.try_inverse(), None);
        assert_eq!(
            QuinticExtension([GoldilocksField(p); 5]).try_inverse(),
            None
        );

        let check_nonzero = |x: QE| {
            let inverse = x.try_inverse().expect("nonzero element must invert");
            assert_eq!(x * inverse, QE::ONE);
            assert_eq!(inverse * x, QE::ONE);
        };
        for limb in 0..5 {
            for value in [1, 2, p - 1, p + 1, u64::MAX] {
                let mut x = [GF::ZERO; 5];
                x[limb] = GoldilocksField(value);
                check_nonzero(QuinticExtension(x));
            }
        }

        let mut state = 0x1319_8a2e_0370_7344u64;
        for _ in 0..10_000 {
            let x = QuinticExtension(core::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                GoldilocksField(state)
            }));
            if !x.is_zero() {
                check_nonzero(x);
            }
        }
    }
}
