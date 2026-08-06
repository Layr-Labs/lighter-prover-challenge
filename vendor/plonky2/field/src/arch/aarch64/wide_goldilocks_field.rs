use core::arch::asm;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use core::slice;

use super::neon_goldilocks_field::NeonGoldilocksField;
use crate::goldilocks_field::GoldilocksField;
use crate::ops::Square;
use crate::packed::PackedField;
use crate::types::{Field, Field64};

/// Four packed Goldilocks elements implemented as two independent AArch64 lane pairs.
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct WideGoldilocksField([NeonGoldilocksField; 2]);

impl WideGoldilocksField {
    #[inline]
    fn from_lanes(lanes: [GoldilocksField; 4]) -> Self {
        Self([
            NeonGoldilocksField([lanes[0], lanes[1]]),
            NeonGoldilocksField([lanes[2], lanes[3]]),
        ])
    }

    #[inline]
    fn lanes(self) -> [GoldilocksField; 4] {
        [
            self.0[0].0[0],
            self.0[0].0[1],
            self.0[1].0[0],
            self.0[1].0[1],
        ]
    }
}

impl Add<Self> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl Add<GoldilocksField> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: GoldilocksField) -> Self {
        Self([self.0[0] + rhs, self.0[1] + rhs])
    }
}

impl Add<WideGoldilocksField> for GoldilocksField {
    type Output = WideGoldilocksField;

    #[inline]
    fn add(self, rhs: WideGoldilocksField) -> WideGoldilocksField {
        rhs + self
    }
}

impl AddAssign<Self> for WideGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl AddAssign<GoldilocksField> for WideGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: GoldilocksField) {
        *self = *self + rhs;
    }
}

impl fmt::Debug for WideGoldilocksField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("WideGoldilocksField")
            .field(&self.lanes())
            .finish()
    }
}

impl Default for WideGoldilocksField {
    #[inline]
    fn default() -> Self {
        Self::ZEROS
    }
}

impl Div<GoldilocksField> for WideGoldilocksField {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: GoldilocksField) -> Self {
        self * rhs.inverse()
    }
}

impl From<GoldilocksField> for WideGoldilocksField {
    #[inline]
    fn from(value: GoldilocksField) -> Self {
        Self([NeonGoldilocksField::from(value); 2])
    }
}

impl Mul<Self> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let lhs = self.lanes().map(|value| value.0);
        let rhs = rhs.lanes().map(|value| value.0);
        Self::from_lanes(mul_reduce_quad(lhs, rhs).map(GoldilocksField))
    }
}

impl Mul<GoldilocksField> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: GoldilocksField) -> Self {
        Self([self.0[0] * rhs, self.0[1] * rhs])
    }
}

impl Mul<WideGoldilocksField> for GoldilocksField {
    type Output = WideGoldilocksField;

    #[inline]
    fn mul(self, rhs: WideGoldilocksField) -> WideGoldilocksField {
        rhs * self
    }
}

impl MulAssign<Self> for WideGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<GoldilocksField> for WideGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: GoldilocksField) {
        *self = *self * rhs;
    }
}

impl Neg for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

impl Product for WideGoldilocksField {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONES)
    }
}

unsafe impl PackedField for WideGoldilocksField {
    type Scalar = GoldilocksField;

    const WIDTH: usize = 4;
    const ZEROS: Self = Self([NeonGoldilocksField::ZEROS; 2]);
    const ONES: Self = Self([NeonGoldilocksField::ONES; 2]);

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
        unsafe { slice::from_raw_parts(self.0.as_ptr().cast(), Self::WIDTH) }
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Scalar] {
        unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), Self::WIDTH) }
    }

    #[inline]
    fn multiply_accumulate(&self, x: Self, y: Self) -> Self {
        let low = mul_add_reduce_pair(
            [self.0[0].0[0].0, self.0[0].0[1].0],
            [x.0[0].0[0].0, x.0[0].0[1].0],
            [y.0[0].0[0].0, y.0[0].0[1].0],
        );
        let high = mul_add_reduce_pair(
            [self.0[1].0[0].0, self.0[1].0[1].0],
            [x.0[1].0[0].0, x.0[1].0[1].0],
            [y.0[1].0[0].0, y.0[1].0[1].0],
        );
        Self([
            NeonGoldilocksField(low.map(GoldilocksField)),
            NeonGoldilocksField(high.map(GoldilocksField)),
        ])
    }

    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let a = self.lanes();
        let b = other.lanes();
        match block_len {
            1 => (
                Self::from_lanes([a[0], b[0], a[2], b[2]]),
                Self::from_lanes([a[1], b[1], a[3], b[3]]),
            ),
            2 => (
                Self::from_lanes([a[0], a[1], b[0], b[1]]),
                Self::from_lanes([a[2], a[3], b[2], b[3]]),
            ),
            4 => (*self, other),
            _ => panic!("unsupported block length"),
        }
    }
}

impl Square for WideGoldilocksField {
    #[inline]
    fn square(&self) -> Self {
        let values = self.lanes().map(|value| value.0);
        Self::from_lanes(mul_reduce_quad(values, values).map(GoldilocksField))
    }
}

impl Sub<Self> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl Sub<GoldilocksField> for WideGoldilocksField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: GoldilocksField) -> Self {
        Self([self.0[0] - rhs, self.0[1] - rhs])
    }
}

impl Sub<WideGoldilocksField> for GoldilocksField {
    type Output = WideGoldilocksField;

    #[inline]
    fn sub(self, rhs: WideGoldilocksField) -> WideGoldilocksField {
        WideGoldilocksField::from(self) - rhs
    }
}

impl SubAssign<Self> for WideGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl SubAssign<GoldilocksField> for WideGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: GoldilocksField) {
        *self = *self - rhs;
    }
}

impl Sum for WideGoldilocksField {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZEROS)
    }
}

/// Reduce four independent 128-bit products modulo `2^64 - 2^32 + 1`.
///
/// Each lane reuses its two input registers after the product is available. This keeps four
/// independent reduction chains in one assembly block without exhausting AArch64's register file.
#[inline(always)]
fn mul_reduce_quad(lhs: [u64; 4], rhs: [u64; 4]) -> [u64; 4] {
    let [mut result0, mut result1, mut result2, mut result3] = lhs;
    let [scratch0, scratch1, scratch2, scratch3] = rhs;

    unsafe {
        asm!(
            "umulh {hi0}, {result0}, {scratch0}",
            "umulh {hi1}, {result1}, {scratch1}",
            "umulh {hi2}, {result2}, {scratch2}",
            "umulh {hi3}, {result3}, {scratch3}",
            "mul   {result0}, {result0}, {scratch0}",
            "mul   {result1}, {result1}, {scratch1}",
            "mul   {result2}, {result2}, {scratch2}",
            "mul   {result3}, {result3}, {scratch3}",
            "lsr   {scratch0}, {hi0}, #32",
            "lsr   {scratch1}, {hi1}, #32",
            "lsr   {scratch2}, {hi2}, #32",
            "lsr   {scratch3}, {hi3}, #32",
            "subs  {result0}, {result0}, {scratch0}",
            "csetm {scratch0:w}, cc",
            "subs  {result1}, {result1}, {scratch1}",
            "csetm {scratch1:w}, cc",
            "subs  {result2}, {result2}, {scratch2}",
            "csetm {scratch2:w}, cc",
            "subs  {result3}, {result3}, {scratch3}",
            "csetm {scratch3:w}, cc",
            "sub   {result0}, {result0}, {scratch0}",
            "sub   {result1}, {result1}, {scratch1}",
            "sub   {result2}, {result2}, {scratch2}",
            "sub   {result3}, {result3}, {scratch3}",
            "and   {scratch0}, {hi0}, {epsilon}",
            "and   {scratch1}, {hi1}, {epsilon}",
            "and   {scratch2}, {hi2}, {epsilon}",
            "and   {scratch3}, {hi3}, {epsilon}",
            "lsl   {hi0}, {scratch0}, #32",
            "lsl   {hi1}, {scratch1}, #32",
            "lsl   {hi2}, {scratch2}, #32",
            "lsl   {hi3}, {scratch3}, #32",
            "sub   {hi0}, {hi0}, {scratch0}",
            "sub   {hi1}, {hi1}, {scratch1}",
            "sub   {hi2}, {hi2}, {scratch2}",
            "sub   {hi3}, {hi3}, {scratch3}",
            "adds  {result0}, {result0}, {hi0}",
            "csetm {scratch0:w}, cs",
            "adds  {result1}, {result1}, {hi1}",
            "csetm {scratch1:w}, cs",
            "adds  {result2}, {result2}, {hi2}",
            "csetm {scratch2:w}, cs",
            "adds  {result3}, {result3}, {hi3}",
            "csetm {scratch3:w}, cs",
            "add   {result0}, {result0}, {scratch0}",
            "add   {result1}, {result1}, {scratch1}",
            "add   {result2}, {result2}, {scratch2}",
            "add   {result3}, {result3}, {scratch3}",
            result0 = inout(reg) result0,
            result1 = inout(reg) result1,
            result2 = inout(reg) result2,
            result3 = inout(reg) result3,
            scratch0 = inout(reg) scratch0 => _,
            scratch1 = inout(reg) scratch1 => _,
            scratch2 = inout(reg) scratch2 => _,
            scratch3 = inout(reg) scratch3 => _,
            hi0 = out(reg) _,
            hi1 = out(reg) _,
            hi2 = out(reg) _,
            hi3 = out(reg) _,
            epsilon = in(reg) GoldilocksField::ORDER.wrapping_neg(),
            options(pure, nomem, nostack),
        );
    }

    [result0, result1, result2, result3]
}

/// Reduce two independent `addend + lhs * rhs` expressions modulo
/// `2^64 - 2^32 + 1`.
#[inline(always)]
fn mul_add_reduce_pair(addends: [u64; 2], lhs: [u64; 2], rhs: [u64; 2]) -> [u64; 2] {
    let [mut result0, mut result1] = lhs;
    let [scratch0, scratch1] = rhs;
    let [addend0, addend1] = addends;

    unsafe {
        asm!(
            "umulh {hi0}, {result0}, {scratch0}",
            "umulh {hi1}, {result1}, {scratch1}",
            "mul   {result0}, {result0}, {scratch0}",
            "mul   {result1}, {result1}, {scratch1}",
            "adds  {result0}, {result0}, {addend0}",
            "adc   {hi0}, {hi0}, xzr",
            "adds  {result1}, {result1}, {addend1}",
            "adc   {hi1}, {hi1}, xzr",
            "lsr   {scratch0}, {hi0}, #32",
            "lsr   {scratch1}, {hi1}, #32",
            "subs  {result0}, {result0}, {scratch0}",
            "csetm {scratch0:w}, cc",
            "subs  {result1}, {result1}, {scratch1}",
            "csetm {scratch1:w}, cc",
            "sub   {result0}, {result0}, {scratch0}",
            "sub   {result1}, {result1}, {scratch1}",
            "and   {scratch0}, {hi0}, {epsilon}",
            "and   {scratch1}, {hi1}, {epsilon}",
            "lsl   {hi0}, {scratch0}, #32",
            "lsl   {hi1}, {scratch1}, #32",
            "sub   {hi0}, {hi0}, {scratch0}",
            "sub   {hi1}, {hi1}, {scratch1}",
            "adds  {result0}, {result0}, {hi0}",
            "csetm {scratch0:w}, cs",
            "adds  {result1}, {result1}, {hi1}",
            "csetm {scratch1:w}, cs",
            "add   {result0}, {result0}, {scratch0}",
            "add   {result1}, {result1}, {scratch1}",
            result0 = inout(reg) result0,
            result1 = inout(reg) result1,
            scratch0 = inout(reg) scratch0 => _,
            scratch1 = inout(reg) scratch1 => _,
            addend0 = in(reg) addend0,
            addend1 = in(reg) addend1,
            hi0 = out(reg) _,
            hi1 = out(reg) _,
            epsilon = in(reg) GoldilocksField::ORDER.wrapping_neg(),
            options(pure, nomem, nostack),
        );
    }

    [result0, result1]
}

#[cfg(test)]
mod tests {
    use super::WideGoldilocksField;
    use crate::goldilocks_field::GoldilocksField;
    use crate::ops::Square;
    use crate::packed::PackedField;
    use crate::types::{Field, Field64};

    fn boundary_values() -> [GoldilocksField; 12] {
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
    fn four_lane_operations_match_scalar_field() {
        let a = [
            GoldilocksField::ZERO,
            GoldilocksField::ONE,
            GoldilocksField::from_noncanonical_u64(GoldilocksField::ORDER),
            GoldilocksField::from_noncanonical_u64(u64::MAX),
        ];
        let b = [
            GoldilocksField::from_noncanonical_u64(14_479_013_849_828_404_771),
            GoldilocksField::from_noncanonical_u64(9_087_029_921_428_221_768),
            GoldilocksField::from_noncanonical_u64(2_441_288_194_761_790_662),
            GoldilocksField::TWO,
        ];
        let packed_a = *WideGoldilocksField::from_slice(&a);
        let packed_b = *WideGoldilocksField::from_slice(&b);
        assert_eq!(
            (packed_a + packed_b).as_slice(),
            core::array::from_fn::<_, 4, _>(|i| a[i] + b[i])
        );
        assert_eq!(
            (packed_a - packed_b).as_slice(),
            core::array::from_fn::<_, 4, _>(|i| a[i] - b[i])
        );
        assert_eq!(
            (packed_a * packed_b).as_slice(),
            core::array::from_fn::<_, 4, _>(|i| a[i] * b[i])
        );
        assert_eq!(
            packed_a.square().as_slice(),
            core::array::from_fn::<_, 4, _>(|i| a[i].square())
        );

        for block_len in [1, 2, 4] {
            let (left, right) = packed_a.interleave(packed_b, block_len);
            assert_eq!(left.interleave(right, block_len), (packed_a, packed_b));
        }
    }

    #[test]
    fn fused_reduction_matches_scalar_boundary_products() {
        let values = boundary_values();
        for i in 0..values.len() {
            for j in 0..values.len() {
                let a: [GoldilocksField; 4] =
                    core::array::from_fn(|lane| values[(i + 2 * lane) % values.len()]);
                let b: [GoldilocksField; 4] =
                    core::array::from_fn(|lane| values[(j + 3 * lane) % values.len()]);
                let c: [GoldilocksField; 4] =
                    core::array::from_fn(|lane| values[(i + j + 5 * lane) % values.len()]);
                let packed_a = *WideGoldilocksField::from_slice(&a);
                let packed_b = *WideGoldilocksField::from_slice(&b);
                let packed_c = *WideGoldilocksField::from_slice(&c);

                assert_eq!(
                    (packed_a * packed_b).as_slice(),
                    core::array::from_fn::<_, 4, _>(|lane| a[lane] * b[lane])
                );
                assert_eq!(
                    <WideGoldilocksField as PackedField>::multiply_accumulate(
                        &packed_c, packed_a, packed_b,
                    )
                    .as_slice(),
                    core::array::from_fn::<_, 4, _>(|lane| {
                        <GoldilocksField as Field>::multiply_accumulate(&c[lane], a[lane], b[lane])
                    })
                );
                assert_eq!(
                    packed_a.square().as_slice(),
                    core::array::from_fn::<_, 4, _>(|lane| a[lane].square())
                );
            }
        }
    }
}
