use core::arch::asm;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::goldilocks_field::GoldilocksField;
use crate::ops::Square;
use crate::packed::PackedField;
use crate::types::{Field, Field64};

/// Two independent Goldilocks field elements evaluated in parallel on AArch64.
///
/// The representation deliberately retains the scalar field's eight-byte alignment so that a
/// scalar slice can be reinterpreted as packed values. Multiplication interleaves both lanes in one
/// assembly block to hide the latency of AArch64's scalar `mul` and `umulh` instructions.
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct NeonGoldilocksField(pub [GoldilocksField; 2]);

impl Add<Self> for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
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
    fn add(self, rhs: NeonGoldilocksField) -> NeonGoldilocksField {
        NeonGoldilocksField::from(self) + rhs
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

impl fmt::Debug for NeonGoldilocksField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NeonGoldilocksField").field(&self.0).finish()
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
    #[inline]
    fn from(value: GoldilocksField) -> Self {
        Self([value; 2])
    }
}

impl Mul<Self> for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (lane0, lane1) = mul_reduce_pair(self.0[0].0, rhs.0[0].0, self.0[1].0, rhs.0[1].0);
        Self([GoldilocksField(lane0), GoldilocksField(lane1)])
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
    fn mul(self, rhs: NeonGoldilocksField) -> NeonGoldilocksField {
        NeonGoldilocksField::from(self) * rhs
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
        Self([-self.0[0], -self.0[1]])
    }
}

impl Product for NeonGoldilocksField {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONES)
    }
}

unsafe impl PackedField for NeonGoldilocksField {
    type Scalar = GoldilocksField;

    const WIDTH: usize = 2;
    const ZEROS: Self = Self([GoldilocksField::ZERO; 2]);
    const ONES: Self = Self([GoldilocksField::ONE; 2]);

    #[inline]
    fn from_slice(slice: &[Self::Scalar]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &*slice.as_ptr().cast() }
    }

    #[inline]
    fn from_slice_mut(slice: &mut [Self::Scalar]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &mut *slice.as_mut_ptr().cast() }
    }

    #[inline]
    fn as_slice(&self) -> &[Self::Scalar] {
        &self.0
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Scalar] {
        &mut self.0
    }

    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        match block_len {
            1 => (Self([self.0[0], other.0[0]]), Self([self.0[1], other.0[1]])),
            2 => (*self, other),
            _ => panic!("unsupported block length"),
        }
    }

    #[inline]
    fn multiply_accumulate(&self, x: Self, y: Self) -> Self {
        let (lane0, lane1) = mul_add_reduce_pair(
            self.0[0].0,
            x.0[0].0,
            y.0[0].0,
            self.0[1].0,
            x.0[1].0,
            y.0[1].0,
        );
        Self([GoldilocksField(lane0), GoldilocksField(lane1)])
    }
}

impl Square for NeonGoldilocksField {
    #[inline]
    fn square(&self) -> Self {
        let (lane0, lane1) = mul_reduce_pair(self.0[0].0, self.0[0].0, self.0[1].0, self.0[1].0);
        Self([GoldilocksField(lane0), GoldilocksField(lane1)])
    }
}

impl Sub<Self> for NeonGoldilocksField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
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
    fn sub(self, rhs: NeonGoldilocksField) -> NeonGoldilocksField {
        NeonGoldilocksField::from(self) - rhs
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
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZEROS)
    }
}

/// Reduce two independent 128-bit products modulo
/// `2^64 - 2^32 + 1`, interleaving the instruction streams for ILP.
#[inline(always)]
fn mul_reduce_pair(a0: u64, b0: u64, a1: u64, b1: u64) -> (u64, u64) {
    let result0: u64;
    let result1: u64;

    unsafe {
        asm!(
            "mul   {lo0}, {a0}, {b0}",
            "mul   {lo1}, {a1}, {b1}",
            "umulh {hi0}, {a0}, {b0}",
            "umulh {hi1}, {a1}, {b1}",
            "lsr   {hi_hi0}, {hi0}, #32",
            "lsr   {hi_hi1}, {hi1}, #32",
            "subs  {tmp0}, {lo0}, {hi_hi0}",
            "csetm {adjust0:w}, cc",
            "subs  {tmp1}, {lo1}, {hi_hi1}",
            "csetm {adjust1:w}, cc",
            "sub   {tmp0}, {tmp0}, {adjust0}",
            "sub   {tmp1}, {tmp1}, {adjust1}",
            "and   {hi_lo0}, {hi0}, {epsilon}",
            "and   {hi_lo1}, {hi1}, {epsilon}",
            "lsl   {fold0}, {hi_lo0}, #32",
            "lsl   {fold1}, {hi_lo1}, #32",
            "sub   {fold0}, {fold0}, {hi_lo0}",
            "sub   {fold1}, {fold1}, {hi_lo1}",
            "adds  {result0}, {tmp0}, {fold0}",
            "csetm {adjust0:w}, cs",
            "adds  {result1}, {tmp1}, {fold1}",
            "csetm {adjust1:w}, cs",
            "add   {result0}, {result0}, {adjust0}",
            "add   {result1}, {result1}, {adjust1}",
            a0 = in(reg) a0,
            b0 = in(reg) b0,
            a1 = in(reg) a1,
            b1 = in(reg) b1,
            epsilon = in(reg) GoldilocksField::ORDER.wrapping_neg(),
            lo0 = out(reg) _,
            lo1 = out(reg) _,
            hi0 = out(reg) _,
            hi1 = out(reg) _,
            hi_hi0 = out(reg) _,
            hi_hi1 = out(reg) _,
            tmp0 = out(reg) _,
            tmp1 = out(reg) _,
            hi_lo0 = out(reg) _,
            hi_lo1 = out(reg) _,
            fold0 = out(reg) _,
            fold1 = out(reg) _,
            adjust0 = out(reg) _,
            adjust1 = out(reg) _,
            result0 = out(reg) result0,
            result1 = out(reg) result1,
            options(pure, nomem, nostack),
        );
    }

    (result0, result1)
}

/// Reduce two independent `accumulator + lhs * rhs` values modulo
/// `2^64 - 2^32 + 1`, interleaving both reduction chains for ILP.
///
/// `adds` plus `adc` constructs the exact 128-bit value used by scalar
/// `Field::multiply_accumulate`. The maximum possible sum is `2^128 - 2^64`, so the carry into
/// the product-high word cannot overflow it.
#[inline(always)]
fn mul_add_reduce_pair(
    accumulator0: u64,
    lhs0: u64,
    rhs0: u64,
    accumulator1: u64,
    lhs1: u64,
    rhs1: u64,
) -> (u64, u64) {
    let result0: u64;
    let result1: u64;

    unsafe {
        asm!(
            "mul   {lo0}, {lhs0}, {rhs0}",
            "mul   {lo1}, {lhs1}, {rhs1}",
            "umulh {hi0}, {lhs0}, {rhs0}",
            "umulh {hi1}, {lhs1}, {rhs1}",
            "adds  {lo0}, {lo0}, {accumulator0}",
            "adc   {hi0}, {hi0}, xzr",
            "adds  {lo1}, {lo1}, {accumulator1}",
            "adc   {hi1}, {hi1}, xzr",
            "lsr   {hi_hi0}, {hi0}, #32",
            "lsr   {hi_hi1}, {hi1}, #32",
            "subs  {tmp0}, {lo0}, {hi_hi0}",
            "csetm {adjust0:w}, cc",
            "subs  {tmp1}, {lo1}, {hi_hi1}",
            "csetm {adjust1:w}, cc",
            "sub   {tmp0}, {tmp0}, {adjust0}",
            "sub   {tmp1}, {tmp1}, {adjust1}",
            "and   {hi_lo0}, {hi0}, {epsilon}",
            "and   {hi_lo1}, {hi1}, {epsilon}",
            "lsl   {fold0}, {hi_lo0}, #32",
            "lsl   {fold1}, {hi_lo1}, #32",
            "sub   {fold0}, {fold0}, {hi_lo0}",
            "sub   {fold1}, {fold1}, {hi_lo1}",
            "adds  {result0}, {tmp0}, {fold0}",
            "csetm {adjust0:w}, cs",
            "adds  {result1}, {tmp1}, {fold1}",
            "csetm {adjust1:w}, cs",
            "add   {result0}, {result0}, {adjust0}",
            "add   {result1}, {result1}, {adjust1}",
            accumulator0 = in(reg) accumulator0,
            lhs0 = in(reg) lhs0,
            rhs0 = in(reg) rhs0,
            accumulator1 = in(reg) accumulator1,
            lhs1 = in(reg) lhs1,
            rhs1 = in(reg) rhs1,
            epsilon = in(reg) GoldilocksField::ORDER.wrapping_neg(),
            lo0 = out(reg) _,
            lo1 = out(reg) _,
            hi0 = out(reg) _,
            hi1 = out(reg) _,
            hi_hi0 = out(reg) _,
            hi_hi1 = out(reg) _,
            tmp0 = out(reg) _,
            tmp1 = out(reg) _,
            hi_lo0 = out(reg) _,
            hi_lo1 = out(reg) _,
            fold0 = out(reg) _,
            fold1 = out(reg) _,
            adjust0 = out(reg) _,
            adjust1 = out(reg) _,
            result0 = out(reg) result0,
            result1 = out(reg) result1,
            options(pure, nomem, nostack),
        );
    }

    (result0, result1)
}

#[cfg(test)]
mod tests {
    use super::NeonGoldilocksField;
    use crate::goldilocks_field::GoldilocksField;
    use crate::ops::Square;
    use crate::packed::PackedField;
    use crate::types::{Field, Field64};

    fn values() -> [GoldilocksField; 12] {
        [
            GoldilocksField::ZERO,
            GoldilocksField::ONE,
            GoldilocksField::TWO,
            GoldilocksField::from_noncanonical_u64(GoldilocksField::ORDER - 1),
            GoldilocksField::from_noncanonical_u64(GoldilocksField::ORDER),
            GoldilocksField::from_noncanonical_u64(GoldilocksField::ORDER + 1),
            GoldilocksField::from_noncanonical_u64(u32::MAX as u64),
            GoldilocksField::from_noncanonical_u64(1 << 32),
            GoldilocksField::from_noncanonical_u64(u64::MAX),
            GoldilocksField::from_noncanonical_u64(14_479_013_849_828_404_771),
            GoldilocksField::from_noncanonical_u64(9_087_029_921_428_221_768),
            GoldilocksField::from_noncanonical_u64(2_441_288_194_761_790_662),
        ]
    }

    #[test]
    fn arithmetic_matches_scalar_field() {
        let values = values();
        for i in 0..values.len() {
            for j in 0..values.len() {
                let a = NeonGoldilocksField([values[i], values[(i + 5) % values.len()]]);
                let b = NeonGoldilocksField([values[j], values[(j + 7) % values.len()]]);
                let scalar = values[(i + j) % values.len()];

                assert_eq!((a + b).0, [a.0[0] + b.0[0], a.0[1] + b.0[1]]);
                assert_eq!((a - b).0, [a.0[0] - b.0[0], a.0[1] - b.0[1]]);
                assert_eq!((a * b).0, [a.0[0] * b.0[0], a.0[1] * b.0[1]]);
                assert_eq!((a * scalar).0, [a.0[0] * scalar, a.0[1] * scalar]);
                assert_eq!((-a).0, [-a.0[0], -a.0[1]]);
                assert_eq!(a.square().0, [a.0[0].square(), a.0[1].square()]);

                for k in 0..values.len() {
                    let accumulator =
                        NeonGoldilocksField([values[k], values[(k + 11) % values.len()]]);
                    assert_eq!(
                        accumulator.multiply_accumulate(a, b).0,
                        [
                            Field::multiply_accumulate(&accumulator.0[0], a.0[0], b.0[0]),
                            Field::multiply_accumulate(&accumulator.0[1], a.0[1], b.0[1]),
                        ]
                    );
                }
            }
        }
    }

    #[test]
    fn packed_views_and_interleave_preserve_lane_order() {
        let mut values = [
            GoldilocksField::from_canonical_u64(0),
            GoldilocksField::from_canonical_u64(1),
            GoldilocksField::from_canonical_u64(10),
            GoldilocksField::from_canonical_u64(11),
        ];
        let packed = NeonGoldilocksField::pack_slice(&values);
        let (left, right) = packed[0].interleave(packed[1], 1);
        assert_eq!(left.0, [values[0], values[2]]);
        assert_eq!(right.0, [values[1], values[3]]);
        assert_eq!(left.interleave(right, 1), (packed[0], packed[1]));

        NeonGoldilocksField::pack_slice_mut(&mut values)[1] += GoldilocksField::ONE;
        assert_eq!(values[2], GoldilocksField::from_canonical_u64(11));
        assert_eq!(values[3], GoldilocksField::from_canonical_u64(12));
    }
}
