use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use core::slice;

use super::wide_goldilocks_field::WideGoldilocksField;
use crate::goldilocks_field::GoldilocksField;
use crate::ops::Square;
use crate::packed::PackedField;
use crate::types::Field;

/// Eight packed Goldilocks elements: two `WideGoldilocksField` vectors evaluated in lockstep.
///
/// This exists purely to raise instruction-level parallelism in latency-bound constraint
/// folds (the quotient-polynomial accumulate path). A constraint expression evaluated at one
/// packed vector forms a single serial dependency chain through the ~10-cycle
/// multiply-reduce kernels; evaluating two independent vectors in lockstep gives the
/// out-of-order core two chains to overlap, following Plonky3 PR #1977. It is deliberately
/// NOT the `Packable::Packing` type — the FFT and hashing keep WIDTH 4 — and it is
/// deliberately no wider than 2x to avoid register spills.
///
/// Like the narrower packings, it retains the scalar field's eight-byte alignment so that a
/// scalar slice can be reinterpreted as packed values.
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct PairedGoldilocksField(pub [WideGoldilocksField; 2]);

impl Add<Self> for PairedGoldilocksField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl Add<GoldilocksField> for PairedGoldilocksField {
    type Output = Self;

    #[inline]
    fn add(self, rhs: GoldilocksField) -> Self {
        Self([self.0[0] + rhs, self.0[1] + rhs])
    }
}

impl Add<PairedGoldilocksField> for GoldilocksField {
    type Output = PairedGoldilocksField;

    #[inline]
    fn add(self, rhs: PairedGoldilocksField) -> PairedGoldilocksField {
        rhs + self
    }
}

impl AddAssign<Self> for PairedGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl AddAssign<GoldilocksField> for PairedGoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: GoldilocksField) {
        *self = *self + rhs;
    }
}

impl fmt::Debug for PairedGoldilocksField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PairedGoldilocksField")
            .field(&self.as_slice())
            .finish()
    }
}

impl Default for PairedGoldilocksField {
    #[inline]
    fn default() -> Self {
        Self::ZEROS
    }
}

impl Div<GoldilocksField> for PairedGoldilocksField {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: GoldilocksField) -> Self {
        self * rhs.inverse()
    }
}

impl From<GoldilocksField> for PairedGoldilocksField {
    #[inline]
    fn from(value: GoldilocksField) -> Self {
        Self([WideGoldilocksField::from(value); 2])
    }
}

impl Mul<Self> for PairedGoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self([self.0[0] * rhs.0[0], self.0[1] * rhs.0[1]])
    }
}

impl Mul<GoldilocksField> for PairedGoldilocksField {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: GoldilocksField) -> Self {
        Self([self.0[0] * rhs, self.0[1] * rhs])
    }
}

impl Mul<PairedGoldilocksField> for GoldilocksField {
    type Output = PairedGoldilocksField;

    #[inline]
    fn mul(self, rhs: PairedGoldilocksField) -> PairedGoldilocksField {
        rhs * self
    }
}

impl MulAssign<Self> for PairedGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<GoldilocksField> for PairedGoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: GoldilocksField) {
        *self = *self * rhs;
    }
}

impl Neg for PairedGoldilocksField {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

impl Product for PairedGoldilocksField {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONES)
    }
}

unsafe impl PackedField for PairedGoldilocksField {
    type Scalar = GoldilocksField;

    const WIDTH: usize = 8;
    const ZEROS: Self = Self([WideGoldilocksField::ZEROS; 2]);
    const ONES: Self = Self([WideGoldilocksField::ONES; 2]);

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
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        match block_len {
            1 | 2 => {
                // Blockwise 2x2 transpose applied within each corresponding
                // half pair, matching the generic interleave semantics.
                let (a0, b0) = self.0[0].interleave(other.0[0], block_len);
                let (a1, b1) = self.0[1].interleave(other.0[1], block_len);
                (Self([a0, a1]), Self([b0, b1]))
            }
            4 => (
                Self([self.0[0], other.0[0]]),
                Self([self.0[1], other.0[1]]),
            ),
            8 => (*self, other),
            _ => panic!("unsupported block length"),
        }
    }

    /// Delegates to the two `WideGoldilocksField` halves so all four two-lane
    /// multiply-accumulate-reduce assembly blocks are emitted back to back with
    /// no cross-half data dependencies.
    #[inline]
    fn multiply_accumulate(&self, x: Self, y: Self) -> Self {
        Self([
            self.0[0].multiply_accumulate(x.0[0], y.0[0]),
            self.0[1].multiply_accumulate(x.0[1], y.0[1]),
        ])
    }
}

impl Square for PairedGoldilocksField {
    #[inline]
    fn square(&self) -> Self {
        Self([self.0[0].square(), self.0[1].square()])
    }
}

impl Sub<Self> for PairedGoldilocksField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl Sub<GoldilocksField> for PairedGoldilocksField {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: GoldilocksField) -> Self {
        Self([self.0[0] - rhs, self.0[1] - rhs])
    }
}

impl Sub<PairedGoldilocksField> for GoldilocksField {
    type Output = PairedGoldilocksField;

    #[inline]
    fn sub(self, rhs: PairedGoldilocksField) -> PairedGoldilocksField {
        PairedGoldilocksField::from(self) - rhs
    }
}

impl SubAssign<Self> for PairedGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl SubAssign<GoldilocksField> for PairedGoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: GoldilocksField) {
        *self = *self - rhs;
    }
}

impl Sum for PairedGoldilocksField {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZEROS)
    }
}

#[cfg(test)]
mod tests {
    use super::PairedGoldilocksField;
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
    fn eight_lane_operations_match_scalar_field() {
        let values = boundary_values();
        let a: [GoldilocksField; 8] = core::array::from_fn(|lane| values[(2 * lane) % values.len()]);
        let b: [GoldilocksField; 8] =
            core::array::from_fn(|lane| values[(3 * lane + 1) % values.len()]);
        let packed_a = *PairedGoldilocksField::from_slice(&a);
        let packed_b = *PairedGoldilocksField::from_slice(&b);
        assert_eq!(
            (packed_a + packed_b).as_slice(),
            core::array::from_fn::<_, 8, _>(|i| a[i] + b[i])
        );
        assert_eq!(
            (packed_a - packed_b).as_slice(),
            core::array::from_fn::<_, 8, _>(|i| a[i] - b[i])
        );
        assert_eq!(
            (packed_a * packed_b).as_slice(),
            core::array::from_fn::<_, 8, _>(|i| a[i] * b[i])
        );
        assert_eq!(
            packed_a.square().as_slice(),
            core::array::from_fn::<_, 8, _>(|i| a[i].square())
        );
        assert_eq!(
            (-packed_a).as_slice(),
            core::array::from_fn::<_, 8, _>(|i| -a[i])
        );
        assert_eq!(
            packed_a.multiply_accumulate(packed_b, packed_a).as_slice(),
            core::array::from_fn::<_, 8, _>(|i| a[i] + b[i] * a[i])
        );

        for block_len in [1, 2, 4, 8] {
            let (left, right) = packed_a.interleave(packed_b, block_len);
            assert_eq!(left.interleave(right, block_len), (packed_a, packed_b));
        }
    }

    /// Interleave must match the generic blockwise-transpose semantics, not
    /// merely round-trip.
    #[test]
    fn interleave_matches_reference() {
        let a: [GoldilocksField; 8] =
            core::array::from_fn(|i| GoldilocksField::from_canonical_u64(i as u64));
        let b: [GoldilocksField; 8] =
            core::array::from_fn(|i| GoldilocksField::from_canonical_u64(8 + i as u64));
        let packed_a = *PairedGoldilocksField::from_slice(&a);
        let packed_b = *PairedGoldilocksField::from_slice(&b);
        // block_len == WIDTH is specified to be a no-op.
        assert_eq!(packed_a.interleave(packed_b, 8), (packed_a, packed_b));
        for block_len in [1usize, 2, 4] {
            let (got0, got1) = packed_a.interleave(packed_b, block_len);
            let mut want0 = [GoldilocksField::ZERO; 8];
            let mut want1 = [GoldilocksField::ZERO; 8];
            let num_blocks = 8 / block_len;
            for blk in 0..num_blocks {
                for j in 0..block_len {
                    let (src, src_blk) = if blk % 2 == 0 {
                        (&a, blk)
                    } else {
                        (&b, blk - 1)
                    };
                    want0[blk * block_len + j] = src[src_blk * block_len + j];
                    let (src, src_blk) = if blk % 2 == 0 { (&a, blk + 1) } else { (&b, blk) };
                    want1[blk * block_len + j] = src[src_blk * block_len + j];
                }
            }
            assert_eq!(got0.as_slice(), want0, "block_len {block_len} lo");
            assert_eq!(got1.as_slice(), want1, "block_len {block_len} hi");
        }
    }

    /// The packed `multiply_accumulate` must agree with the scalar one on the
    /// RAW `u64` representative, matching the WIDTH-4 test's guarantee.
    #[test]
    fn multiply_accumulate_matches_scalar_raw_representative() {
        fn raw(x: GoldilocksField) -> u64 {
            x.0
        }

        let values = boundary_values();
        let mut s: u64 = 0x1234_5678_9ABC_DEF1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            GoldilocksField(s)
        };

        let mut check = |acc: [GoldilocksField; 8],
                         x: [GoldilocksField; 8],
                         y: [GoldilocksField; 8]| {
            let packed = PairedGoldilocksField::from_slice(&acc).multiply_accumulate(
                *PairedGoldilocksField::from_slice(&x),
                *PairedGoldilocksField::from_slice(&y),
            );
            let got: [u64; 8] = core::array::from_fn(|lane| raw(packed.as_slice()[lane]));
            let want: [u64; 8] = core::array::from_fn(|lane| {
                raw(Field::multiply_accumulate(&acc[lane], x[lane], y[lane]))
            });
            assert_eq!(got, want, "acc={acc:?} x={x:?} y={y:?}");
        };

        for i in 0..values.len() {
            for j in 0..values.len() {
                for k in 0..values.len() {
                    let acc = core::array::from_fn(|lane| values[(i + lane) % values.len()]);
                    let x = core::array::from_fn(|lane| values[(j + 2 * lane) % values.len()]);
                    let y = core::array::from_fn(|lane| values[(k + 3 * lane) % values.len()]);
                    check(acc, x, y);
                }
            }
        }

        for _ in 0..50_000 {
            let acc = core::array::from_fn(|_| next());
            let x = core::array::from_fn(|_| next());
            let y = core::array::from_fn(|_| next());
            check(acc, x, y);
        }
    }
}
