#[cfg(target_arch = "aarch64")]
use core::any::TypeId;

#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::neon_goldilocks_field::NeonGoldilocksField;
#[cfg(target_arch = "aarch64")]
use crate::extension::quadratic::QuadraticExtension;
use crate::extension::FieldExtension;
#[cfg(target_arch = "aarch64")]
use crate::goldilocks_field::GoldilocksField;
use crate::packable::Packable;
use crate::packed::PackedField;
use crate::types::Field;

const fn pack_with_leftovers_split_point<P: PackedField>(slice: &[P::Scalar]) -> usize {
    let n = slice.len();
    let n_leftover = n % P::WIDTH;
    n - n_leftover
}

fn pack_slice_with_leftovers<P: PackedField>(slice: &[P::Scalar]) -> (&[P], &[P::Scalar]) {
    let split_point = pack_with_leftovers_split_point::<P>(slice);
    let (slice_packable, slice_leftovers) = slice.split_at(split_point);
    let slice_packed = P::pack_slice(slice_packable);
    (slice_packed, slice_leftovers)
}

fn pack_slice_with_leftovers_mut<P: PackedField>(
    slice: &mut [P::Scalar],
) -> (&mut [P], &mut [P::Scalar]) {
    let split_point = pack_with_leftovers_split_point::<P>(slice);
    let (slice_packable, slice_leftovers) = slice.split_at_mut(split_point);
    let slice_packed = P::pack_slice_mut(slice_packable);
    (slice_packed, slice_leftovers)
}

/// Elementwise inplace multiplication of two slices of field elements.
/// Implementation be faster than the trivial for loop.
pub fn batch_multiply_inplace<F: Field>(out: &mut [F], a: &[F]) {
    let n = out.len();
    assert_eq!(n, a.len(), "both arrays must have the same length");

    // Split out slice of vectors, leaving leftovers as scalars
    let (out_packed, out_leftovers) =
        pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out);
    let (a_packed, a_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(a);

    // Multiply packed and the leftovers
    for (x_out, x_a) in out_packed.iter_mut().zip(a_packed) {
        *x_out *= *x_a;
    }
    for (x_out, x_a) in out_leftovers.iter_mut().zip(a_leftovers) {
        *x_out *= *x_a;
    }
}
/// Elementwise `out[i] = a[i] * b[i]`, writing a destination that is not also an
/// input.
///
/// The LDE fill previously reached its coset-scaled state in two passes over
/// `degree` words: `copy_from_slice` (read `a`, write `out`) followed by
/// `batch_multiply_inplace` (read `out`, read `b`, write `out`). The
/// intermediate unscaled copy is never observed — the FFT only ever sees the
/// scaled values — so the copy is a materialization that can be deleted by
/// folding the multiply into the same pass.
///
/// The packed/scalar split is identical to `batch_multiply_inplace`'s: the
/// maximal `P::WIDTH` prefix uses packed multiplication and the ragged tail uses
/// the same scalar operation, so every produced word is bit-identical to the
/// two-pass form.
pub fn batch_multiply_into<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = out.len();
    assert_eq!(n, a.len(), "output and first input must have the same length");
    assert_eq!(n, b.len(), "output and second input must have the same length");

    let (out_packed, out_leftovers) =
        pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out);
    let (a_packed, a_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(a);
    let (b_packed, b_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(b);

    for ((x_out, x_a), x_b) in out_packed.iter_mut().zip(a_packed).zip(b_packed) {
        *x_out = *x_a * *x_b;
    }
    for ((x_out, x_a), x_b) in out_leftovers.iter_mut().zip(a_leftovers).zip(b_leftovers) {
        *x_out = *x_a * *x_b;
    }
}

/// Elementwise multiply two slices and add the products to an output slice.
pub fn batch_multiply_add_inplace<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = out.len();
    assert_eq!(
        n,
        a.len(),
        "output and first input must have the same length"
    );
    assert_eq!(
        n,
        b.len(),
        "output and second input must have the same length"
    );

    let (out_packed, out_leftovers) =
        pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out);
    let (a_packed, a_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(a);
    let (b_packed, b_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(b);

    for ((x_out, x_a), x_b) in out_packed.iter_mut().zip(a_packed).zip(b_packed) {
        *x_out = x_out.multiply_accumulate(*x_a, *x_b);
    }
    for ((x_out, x_a), x_b) in out_leftovers.iter_mut().zip(a_leftovers).zip(b_leftovers) {
        *x_out += *x_a * *x_b;
    }
}

/// Accumulates `out[i] += factor * scalars[i]` where `factor` and `out` are
/// extension-field elements and each input scalar is in their base field.
///
/// Performing `FieldExtension::scalar_mul` and then adding would reduce every
/// base-field product before reducing the addition. This spelling delegates
/// each limb to [`Field::multiply_accumulate`], allowing one widened product
/// plus accumulator to share a single reduction. On ranked AArch64 Goldilocks
/// quadratic extensions, both independent limbs are issued through the
/// existing two-lane assembly block so their multiply latency overlaps.
pub fn batch_extension_scalar_multiply_add_inplace<BF, FE, const D: usize>(
    out: &mut [FE],
    factor: FE,
    scalars: &[BF],
) where
    BF: Field,
    FE: FieldExtension<D, BaseField = BF>,
{
    assert_eq!(out.len(), scalars.len(), "output and scalar lengths differ");

    #[cfg(target_arch = "aarch64")]
    if D == 2
        && TypeId::of::<BF>() == TypeId::of::<GoldilocksField>()
        && TypeId::of::<FE>() == TypeId::of::<QuadraticExtension<GoldilocksField>>()
    {
        // SAFETY: both TypeId comparisons establish the exact concrete slice
        // element types. Slice lengths are unchanged, and both concrete types
        // retain their ordinary alignment and layout through the cast.
        let factor =
            unsafe { *(&factor as *const FE).cast::<QuadraticExtension<GoldilocksField>>() };
        let out = unsafe {
            core::slice::from_raw_parts_mut(
                out.as_mut_ptr()
                    .cast::<QuadraticExtension<GoldilocksField>>(),
                out.len(),
            )
        };
        let scalars = unsafe {
            core::slice::from_raw_parts(scalars.as_ptr().cast::<GoldilocksField>(), scalars.len())
        };
        let factor = NeonGoldilocksField(factor.0);
        for (acc, &scalar) in out.iter_mut().zip(scalars) {
            acc.0 = NeonGoldilocksField(acc.0)
                .multiply_accumulate(factor, NeonGoldilocksField::from(scalar))
                .0;
        }
        return;
    }

    let factor_limbs = factor.to_basefield_array();
    for (acc, &scalar) in out.iter_mut().zip(scalars) {
        let mut limbs = acc.to_basefield_array();
        for i in 0..D {
            limbs[i] = limbs[i].multiply_accumulate(factor_limbs[i], scalar);
        }
        *acc = FE::from_basefield_array(limbs);
    }
}

/// Elementwise inplace addition of two slices of field elements.
/// Implementation be faster than the trivial for loop.
pub fn batch_add_inplace<F: Field>(out: &mut [F], a: &[F]) {
    let n = out.len();
    assert_eq!(n, a.len(), "both arrays must have the same length");

    // Split out slice of vectors, leaving leftovers as scalars
    let (out_packed, out_leftovers) =
        pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out);
    let (a_packed, a_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(a);

    // Add packed and the leftovers
    for (x_out, x_a) in out_packed.iter_mut().zip(a_packed) {
        *x_out += *x_a;
    }
    for (x_out, x_a) in out_leftovers.iter_mut().zip(a_leftovers) {
        *x_out += *x_a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::{Extendable, FieldExtension};
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::Field64;

    #[test]
    fn batch_multiply_add_matches_scalar_with_packed_leftovers() {
        let mut out = (0..11)
            .map(|i| GoldilocksField::from_canonical_usize(i + 1))
            .collect::<Vec<_>>();
        let a = (0..11)
            .map(|i| GoldilocksField::from_canonical_usize(2 * i + 3))
            .collect::<Vec<_>>();
        let b = (0..11)
            .map(|i| GoldilocksField::from_canonical_usize(3 * i + 5))
            .collect::<Vec<_>>();
        let expected = out
            .iter()
            .zip(&a)
            .zip(&b)
            .map(|((&x_out, &x_a), &x_b)| x_out + x_a * x_b)
            .collect::<Vec<_>>();

        batch_multiply_add_inplace(&mut out, &a, &b);

        assert_eq!(out, expected);
    }

    #[test]
    fn extension_scalar_multiply_add_matches_limb_fma() {
        const D: usize = 2;
        type F = GoldilocksField;
        type FE = <F as Extendable<D>>::Extension;

        let words = [
            0,
            1,
            F::ORDER - 1,
            F::ORDER,
            F::ORDER + 1,
            u64::MAX - 1,
            u64::MAX,
        ];
        let factor =
            FE::from_basefield_array([GoldilocksField(words[5]), GoldilocksField(words[3])]);
        let scalars = words
            .iter()
            .copied()
            .map(GoldilocksField)
            .collect::<Vec<_>>();
        let mut actual = words
            .iter()
            .enumerate()
            .map(|(i, &word)| {
                FE::from_basefield_array([
                    GoldilocksField(word),
                    GoldilocksField(words[words.len() - 1 - i]),
                ])
            })
            .collect::<Vec<_>>();
        let factor_limbs: [F; D] = factor.to_basefield_array();
        let expected = actual
            .iter()
            .zip(&scalars)
            .map(|(acc, &scalar)| {
                let mut limbs: [F; D] = acc.to_basefield_array();
                for i in 0..D {
                    limbs[i] = Field::multiply_accumulate(&limbs[i], factor_limbs[i], scalar);
                }
                FE::from_basefield_array(limbs)
            })
            .collect::<Vec<_>>();

        batch_extension_scalar_multiply_add_inplace::<F, FE, D>(&mut actual, factor, &scalars);

        assert_eq!(actual, expected);
    }
}
