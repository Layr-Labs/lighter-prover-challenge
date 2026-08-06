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
        *x_out += *x_a * *x_b;
    }
    for ((x_out, x_a), x_b) in out_leftovers.iter_mut().zip(a_leftovers).zip(b_leftovers) {
        *x_out += *x_a * *x_b;
    }
}

/// Multiply every row of `a` by the shared vector `b` and add it to the
/// corresponding row of `out`.
pub fn batch_multiply_add_rows_inplace<F: Field>(out: &mut [F], a: &[F], b: &[F]) {
    let n = b.len();
    assert!(n > 0, "shared input must not be empty");
    assert_eq!(out.len(), a.len(), "output and input sizes must match");
    assert_eq!(
        out.len() % n,
        0,
        "rows must all have the shared input length"
    );

    let paired_len = (out.len() / (2 * n)) * (2 * n);
    let (out_pairs, out_tail) = out.split_at_mut(paired_len);
    let (a_pairs, a_tail) = a.split_at(paired_len);

    for (out_pair, a_pair) in out_pairs
        .chunks_exact_mut(2 * n)
        .zip(a_pairs.chunks_exact(2 * n))
    {
        let (out_0, out_1) = out_pair.split_at_mut(n);
        let (a_0, a_1) = a_pair.split_at(n);
        let (out_0_packed, out_0_leftovers) =
            pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out_0);
        let (out_1_packed, out_1_leftovers) =
            pack_slice_with_leftovers_mut::<<F as Packable>::Packing>(out_1);
        let (a_0_packed, a_0_leftovers) =
            pack_slice_with_leftovers::<<F as Packable>::Packing>(a_0);
        let (a_1_packed, a_1_leftovers) =
            pack_slice_with_leftovers::<<F as Packable>::Packing>(a_1);
        let (b_packed, b_leftovers) = pack_slice_with_leftovers::<<F as Packable>::Packing>(b);

        for ((((out_0, out_1), a_0), a_1), b) in out_0_packed
            .iter_mut()
            .zip(out_1_packed)
            .zip(a_0_packed)
            .zip(a_1_packed)
            .zip(b_packed)
        {
            *out_0 += *a_0 * *b;
            *out_1 += *a_1 * *b;
        }
        for ((((out_0, out_1), a_0), a_1), b) in out_0_leftovers
            .iter_mut()
            .zip(out_1_leftovers)
            .zip(a_0_leftovers)
            .zip(a_1_leftovers)
            .zip(b_leftovers)
        {
            *out_0 += *a_0 * *b;
            *out_1 += *a_1 * *b;
        }
    }

    if !out_tail.is_empty() {
        batch_multiply_add_inplace(out_tail, a_tail, b);
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
    use crate::goldilocks_field::GoldilocksField;

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
    fn batch_multiply_add_rows_matches_scalar_for_odd_rows_and_leftovers() {
        const ROW_LEN: usize = 11;
        const ROWS: usize = 3;
        let mut out = (0..ROW_LEN * ROWS)
            .map(|i| GoldilocksField::from_canonical_usize(i + 1))
            .collect::<Vec<_>>();
        let a = (0..ROW_LEN * ROWS)
            .map(|i| GoldilocksField::from_canonical_usize(2 * i + 3))
            .collect::<Vec<_>>();
        let b = (0..ROW_LEN)
            .map(|i| GoldilocksField::from_canonical_usize(3 * i + 5))
            .collect::<Vec<_>>();
        let expected = out
            .iter()
            .zip(&a)
            .enumerate()
            .map(|(i, (&x_out, &x_a))| x_out + x_a * b[i % ROW_LEN])
            .collect::<Vec<_>>();

        batch_multiply_add_rows_inplace(&mut out, &a, &b);

        assert_eq!(out, expected);
    }
}
