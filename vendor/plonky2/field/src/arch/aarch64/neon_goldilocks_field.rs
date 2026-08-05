use core::arch::aarch64::*;
use core::fmt;
use core::fmt::{Debug, Formatter};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::goldilocks_field::GoldilocksField;
use crate::ops::Square;
use crate::packed::PackedField;
use crate::types::{Field, Field64};

/// NEON Goldilocks Field
///
/// Two-lane packed Goldilocks arithmetic on AArch64 NEON. The algorithms mirror
/// `Avx2GoldilocksField`, with one simplification: AArch64 has native unsigned
/// 64-bit vector comparisons (`cmhi`), so the sign-shift trick the AVX2 code
/// uses to emulate unsigned compares is unnecessary here.
///
/// Like the AVX2 version, this wraps `[GoldilocksField; 2]` rather than
/// `uint64x2_t` so the alignment matches `GoldilocksField` and slices can be
/// cast losslessly in both directions.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct NeonGoldilocksField(pub [GoldilocksField; 2]);

const EPSILON: u64 = GoldilocksField::ORDER.wrapping_neg();

impl NeonGoldilocksField {
    #[inline]
    fn new(x: uint64x2_t) -> Self {
        unsafe { transmute(x) }
    }
    #[inline]
    fn get(&self) -> uint64x2_t {
        unsafe { transmute(*self) }
    }
}

unsafe impl PackedField for NeonGoldilocksField {
    const WIDTH: usize = 2;

    type Scalar = GoldilocksField;

    const ZEROS: Self = Self([GoldilocksField::ZERO; 2]);
    const ONES: Self = Self([GoldilocksField::ONE; 2]);

    #[inline]
    fn from_slice(slice: &[GoldilocksField]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &*slice.as_ptr().cast() }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [GoldilocksField]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &mut *slice.as_mut_ptr().cast() }
    }
    #[inline]
    fn as_slice(&self) -> &[GoldilocksField] {
        &self.0[..]
    }
    #[inline]
    fn as_slice_mut(&mut self) -> &mut [GoldilocksField] {
        &mut self.0[..]
    }

    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (v0, v1) = (self.get(), other.get());
        let (res0, res1) = match block_len {
            1 => unsafe { interleave1(v0, v1) },
            2 => (v0, v1),
            _ => panic!("unsupported block_len"),
        };
        (Self::new(res0), Self::new(res1))
    }
}

impl Add<Self> for NeonGoldilocksField {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(unsafe { add(self.get(), rhs.get()) })
    }
}
impl Add<GoldilocksField> for NeonGoldilocksField {
    type Output = Self;
    #[inline]
    fn add(self, rhs: GoldilocksField) -> Self {
        self + Self::from(rhs)
    }
}
impl Add<NeonGoldilocksField> for GoldilocksField {
    type Output = NeonGoldilocksField;
    #[inline]
    fn add(self, rhs: Self::Output) -> Self::Output {
        Self::Output::from(self) + rhs
    }
}
impl AddAssign<Self> for NeonGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl AddAssign<GoldilocksField> for NeonGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: GoldilocksField) {
        *self = *self + rhs;
    }
}

impl Debug for NeonGoldilocksField {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.0[0], self.0[1])
    }
}

impl Default for NeonGoldilocksField {
    #[inline]
    fn default() -> Self {
        Self::ZEROS
    }
}

impl Div<GoldilocksField> for NeonGoldilocksField {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: GoldilocksField) -> Self {
        self * rhs.inverse()
    }
}
impl DivAssign<GoldilocksField> for NeonGoldilocksField {
    #[allow(clippy::suspicious_op_assign_impl)]
    #[inline]
    fn div_assign(&mut self, rhs: GoldilocksField) {
        *self *= rhs.inverse();
    }
}

impl From<GoldilocksField> for NeonGoldilocksField {
    fn from(x: GoldilocksField) -> Self {
        Self([x; 2])
    }
}

impl Mul<Self> for NeonGoldilocksField {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(unsafe { mul(self.get(), rhs.get()) })
    }
}
impl Mul<GoldilocksField> for NeonGoldilocksField {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: GoldilocksField) -> Self {
        self * Self::from(rhs)
    }
}
impl Mul<NeonGoldilocksField> for GoldilocksField {
    type Output = NeonGoldilocksField;
    #[inline]
    fn mul(self, rhs: NeonGoldilocksField) -> Self::Output {
        Self::Output::from(self) * rhs
    }
}
impl MulAssign<Self> for NeonGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl MulAssign<GoldilocksField> for NeonGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: GoldilocksField) {
        *self = *self * rhs;
    }
}

impl Neg for NeonGoldilocksField {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(unsafe { neg(self.get()) })
    }
}

impl Product for NeonGoldilocksField {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONES, |acc, x| acc * x)
    }
}

impl Square for NeonGoldilocksField {
    #[inline]
    fn square(&self) -> Self {
        Self::new(unsafe { square(self.get()) })
    }
}

impl Sub<Self> for NeonGoldilocksField {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(unsafe { sub(self.get(), rhs.get()) })
    }
}
impl Sub<GoldilocksField> for NeonGoldilocksField {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: GoldilocksField) -> Self {
        self - Self::from(rhs)
    }
}
impl Sub<NeonGoldilocksField> for GoldilocksField {
    type Output = NeonGoldilocksField;
    #[inline]
    fn sub(self, rhs: NeonGoldilocksField) -> Self::Output {
        Self::Output::from(self) - rhs
    }
}
impl SubAssign<Self> for NeonGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl SubAssign<GoldilocksField> for NeonGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: GoldilocksField) {
        *self = *self - rhs;
    }
}

impl Sum for NeonGoldilocksField {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZEROS, |acc, x| acc + x)
    }
}

#[inline]
unsafe fn epsilon() -> uint64x2_t {
    vdupq_n_u64(EPSILON)
}

/// Convert to canonical representation: subtract the field order once if the
/// input is >= ORDER. Adding EPSILON (= 2^64 - ORDER) with wraparound is the
/// same as subtracting ORDER when the addition overflows.
#[inline]
unsafe fn canonicalize(x: uint64x2_t) -> uint64x2_t {
    let mask = vcgeq_u64(x, vdupq_n_u64(GoldilocksField::ORDER));
    vaddq_u64(x, vandq_u64(mask, epsilon()))
}

/// Addition x + y mod ORDER for arbitrary x < 2^64 and canonical y < ORDER.
/// On wraparound the missing 2^64 is congruent to EPSILON. The correction
/// cannot re-overflow: on overflow the wrapped sum is < ORDER, and
/// ORDER + EPSILON < 2^64.
#[inline]
unsafe fn add_no_double_overflow_64_64c(x: uint64x2_t, y_c: uint64x2_t) -> uint64x2_t {
    let res_wrapped = vaddq_u64(x, y_c);
    let mask = vcgtq_u64(x, res_wrapped); // -1 if overflowed else 0.
    vaddq_u64(res_wrapped, vandq_u64(mask, epsilon()))
}

#[inline]
unsafe fn add(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    add_no_double_overflow_64_64c(x, canonicalize(y))
}

/// Subtraction x - y mod ORDER for arbitrary x < 2^64 and canonical y < ORDER.
/// On borrow the wrapped result is x - y + 2^64; subtracting EPSILON turns it
/// into x - y + ORDER.
#[inline]
unsafe fn sub(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    let y_c = canonicalize(y);
    let mask = vcgtq_u64(y_c, x); // -1 if underflow (y_c > x) else 0.
    let res_wrapped = vsubq_u64(x, y_c);
    vsubq_u64(res_wrapped, vandq_u64(mask, epsilon()))
}

#[inline]
unsafe fn neg(y: uint64x2_t) -> uint64x2_t {
    vsubq_u64(vdupq_n_u64(GoldilocksField::ORDER), canonicalize(y))
}

#[inline]
unsafe fn lo32(x: uint64x2_t) -> uint32x2_t {
    vmovn_u64(x)
}

#[inline]
unsafe fn hi32(x: uint64x2_t) -> uint32x2_t {
    vshrn_n_u64(x, 32)
}

/// Full 64-bit by 64-bit multiplication via four 32x32->64 partial products,
/// mirroring the AVX2 bignum-addition scheme.
#[inline]
unsafe fn mul64_64(x: uint64x2_t, y: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    let x_lo = lo32(x);
    let x_hi = hi32(x);
    let y_lo = lo32(y);
    let y_hi = hi32(y);

    let mul_ll = vmull_u32(x_lo, y_lo);
    let mul_lh = vmull_u32(x_lo, y_hi);
    let mul_hl = vmull_u32(x_hi, y_lo);
    let mul_hh = vmull_u32(x_hi, y_hi);

    // Bignum addition. Extract high 32 bits of mul_ll and add to mul_hl. This
    // cannot overflow; neither can the two additions after it.
    let mul_ll_hi = vshrq_n_u64(mul_ll, 32);
    let t0 = vaddq_u64(mul_hl, mul_ll_hi);
    let t0_lo = vandq_u64(t0, epsilon());
    let t0_hi = vshrq_n_u64(t0, 32);
    let t1 = vaddq_u64(mul_lh, t0_lo);
    let t2 = vaddq_u64(mul_hh, t0_hi);
    let t1_hi = vshrq_n_u64(t1, 32);
    let res_hi = vaddq_u64(t2, t1_hi);

    // res_lo = (t1 << 32) | (mul_ll & 0xFFFFFFFF); vsli keeps the low 32 bits
    // of mul_ll and inserts t1 shifted into the high half.
    let res_lo = vsliq_n_u64(mul_ll, t1, 32);

    (res_hi, res_lo)
}

/// Full 64-bit squaring; saves one 32x32 multiply versus `mul64_64`. The lh
/// cross term appears twice, so it is shifted by 33 (not 32) when combined.
#[inline]
unsafe fn square64(x: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    let x_lo = lo32(x);
    let x_hi = hi32(x);

    let mul_ll = vmull_u32(x_lo, x_lo);
    let mul_lh = vmull_u32(x_lo, x_hi);
    let mul_hh = vmull_u32(x_hi, x_hi);

    let mul_ll_hi = vshrq_n_u64(mul_ll, 33);
    let t0 = vaddq_u64(mul_lh, mul_ll_hi);
    let t0_hi = vshrq_n_u64(t0, 31);
    let res_hi = vaddq_u64(mul_hh, t0_hi);

    // res_lo = mul_ll + (mul_lh << 33); the addition cannot overflow the low
    // word by construction of the shifts above.
    let mul_lh_lo = vshlq_n_u64(mul_lh, 33);
    let res_lo = vaddq_u64(mul_ll, mul_lh_lo);

    (res_hi, res_lo)
}

/// Goldilocks addition of a "small" y <= 0xFFFFFFFF00000000. A single
/// overflow correction suffices because the wrapped sum is then < ORDER.
#[inline]
unsafe fn add_small(x: uint64x2_t, y_small: uint64x2_t) -> uint64x2_t {
    let res_wrapped = vaddq_u64(x, y_small);
    let mask = vcgtq_u64(x, res_wrapped); // -1 if overflowed else 0.
    vaddq_u64(res_wrapped, vandq_u64(mask, epsilon()))
}

/// Goldilocks subtraction of a "small" y <= 0xFFFFFFFF00000000.
#[inline]
unsafe fn sub_small(x: uint64x2_t, y_small: uint64x2_t) -> uint64x2_t {
    let mask = vcgtq_u64(y_small, x); // -1 if underflowed else 0.
    let res_wrapped = vsubq_u64(x, y_small);
    vsubq_u64(res_wrapped, vandq_u64(mask, epsilon()))
}

/// Reduce a 128-bit product (hi, lo) modulo ORDER:
/// hi·2^64 + lo ≡ lo - hi_hi + hi_lo·EPSILON (mod ORDER).
#[inline]
unsafe fn reduce128(x: (uint64x2_t, uint64x2_t)) -> uint64x2_t {
    let (hi0, lo0) = x;
    let hi_hi0 = vshrq_n_u64(hi0, 32);
    let lo1 = sub_small(lo0, hi_hi0);
    let t1 = vmull_u32(lo32(hi0), vdup_n_u32(u32::MAX));
    add_small(lo1, t1)
}

#[inline]
unsafe fn mul(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    reduce128(mul64_64(x, y))
}

#[inline]
unsafe fn square(x: uint64x2_t) -> uint64x2_t {
    reduce128(square64(x))
}

#[inline]
unsafe fn interleave1(x: uint64x2_t, y: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    (vtrn1q_u64(x, y), vtrn2q_u64(x, y))
}

#[cfg(test)]
mod tests {
    use crate::arch::aarch64::neon_goldilocks_field::NeonGoldilocksField;
    use crate::goldilocks_field::GoldilocksField;
    use crate::ops::Square;
    use crate::packed::PackedField;
    use crate::types::{Field, Field64, PrimeField64};

    /// Edge cases around 0, ORDER, EPSILON, and u64::MAX plus arbitrary
    /// values; noncanonical inputs (>= ORDER) are deliberately included since
    /// the scalar field accepts them too.
    fn interesting_vals() -> Vec<u64> {
        let mut vals = vec![
            0,
            1,
            2,
            GoldilocksField::ORDER - 2,
            GoldilocksField::ORDER - 1,
            GoldilocksField::ORDER,
            GoldilocksField::ORDER + 1,
            GoldilocksField::ORDER.wrapping_neg(), // EPSILON
            u32::MAX as u64,
            (u32::MAX as u64) << 32,
            u64::MAX - 1,
            u64::MAX,
            14479013849828404771,
            9087029921428221768,
            2441288194761790662,
            5646033492608483824,
            17891926589593242302,
            11009798273260028228,
        ];
        // Deterministic pseudo-random extension (xorshift), no rand dependency.
        let mut state = 0x9E3779B97F4A7C15u64;
        for _ in 0..64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            vals.push(state);
        }
        vals
    }

    fn packed_from(a: u64, b: u64) -> NeonGoldilocksField {
        *NeonGoldilocksField::from_slice(&[
            GoldilocksField::from_noncanonical_u64(a),
            GoldilocksField::from_noncanonical_u64(b),
        ])
    }

    #[test]
    fn test_add_sub_mul_matches_scalar_exhaustive_pairs() {
        let vals = interesting_vals();
        for (i, &a) in vals.iter().enumerate() {
            for &b in &vals[i..] {
                let sa = GoldilocksField::from_noncanonical_u64(a);
                let sb = GoldilocksField::from_noncanonical_u64(b);
                let p = packed_from(a, b);
                let q = packed_from(b, a);

                let sum = (p + q).as_slice().to_vec();
                assert_eq!(sum[0].to_canonical_u64(), (sa + sb).to_canonical_u64());
                assert_eq!(sum[1].to_canonical_u64(), (sb + sa).to_canonical_u64());

                let diff = (p - q).as_slice().to_vec();
                assert_eq!(diff[0].to_canonical_u64(), (sa - sb).to_canonical_u64());
                assert_eq!(diff[1].to_canonical_u64(), (sb - sa).to_canonical_u64());

                let prod = (p * q).as_slice().to_vec();
                assert_eq!(prod[0].to_canonical_u64(), (sa * sb).to_canonical_u64());
                assert_eq!(prod[1].to_canonical_u64(), (sb * sa).to_canonical_u64());
            }
        }
    }

    #[test]
    fn test_square_neg_matches_scalar() {
        for &a in &interesting_vals() {
            let sa = GoldilocksField::from_noncanonical_u64(a);
            let p = packed_from(a, a);

            let sq = p.square().as_slice().to_vec();
            assert_eq!(sq[0].to_canonical_u64(), sa.square().to_canonical_u64());
            assert_eq!(sq[1].to_canonical_u64(), sa.square().to_canonical_u64());

            let ng = (-p).as_slice().to_vec();
            assert_eq!(ng[0].to_canonical_u64(), (-sa).to_canonical_u64());
            assert_eq!(ng[1].to_canonical_u64(), (-sa).to_canonical_u64());
        }
    }

    #[test]
    fn test_interleave() {
        let a = packed_from(1, 2);
        let b = packed_from(3, 4);

        let (x1, y1) = a.interleave(b, 1);
        assert_eq!(x1.as_slice()[0].to_canonical_u64(), 1);
        assert_eq!(x1.as_slice()[1].to_canonical_u64(), 3);
        assert_eq!(y1.as_slice()[0].to_canonical_u64(), 2);
        assert_eq!(y1.as_slice()[1].to_canonical_u64(), 4);

        let (x2, y2) = a.interleave(b, 2);
        assert_eq!(x2.as_slice()[0].to_canonical_u64(), 1);
        assert_eq!(x2.as_slice()[1].to_canonical_u64(), 2);
        assert_eq!(y2.as_slice()[0].to_canonical_u64(), 3);
        assert_eq!(y2.as_slice()[1].to_canonical_u64(), 4);
    }

    #[test]
    fn test_scalar_mixed_ops_and_pack_slice() {
        let s = GoldilocksField::from_noncanonical_u64(7);
        let p = packed_from(10, 20);

        assert_eq!(
            (p + s).as_slice()[0].to_canonical_u64(),
            17,
            "packed + scalar"
        );
        assert_eq!((s + p).as_slice()[1].to_canonical_u64(), 27);
        assert_eq!((p - s).as_slice()[0].to_canonical_u64(), 3);
        assert_eq!((p * s).as_slice()[1].to_canonical_u64(), 140);

        let buf: Vec<GoldilocksField> = (0..8).map(GoldilocksField::from_canonical_u64).collect();
        let packed = NeonGoldilocksField::pack_slice(&buf);
        assert_eq!(packed.len(), 4);
        assert_eq!(packed[3].as_slice()[1].to_canonical_u64(), 7);
    }
}
