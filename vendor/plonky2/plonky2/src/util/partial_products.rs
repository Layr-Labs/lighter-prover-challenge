#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::iter;

use itertools::Itertools;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::plonk::circuit_builder::CircuitBuilder;

pub(crate) fn quotient_chunk_products<F: Field>(
    quotient_values: &[F],
    max_degree: usize,
) -> Vec<F> {
    debug_assert!(max_degree > 1);
    assert!(!quotient_values.is_empty());
    let chunk_size = max_degree;
    quotient_values
        .chunks(chunk_size)
        .map(|chunk| chunk.iter().copied().product())
        .collect()
}

/// Compute partial products of the original vector `v` such that all products consist of `max_degree`
/// or less elements. This is done until we've computed the product `P` of all elements in the vector.
pub(crate) fn partial_products_and_z_gx<F: Field>(z_x: F, quotient_chunk_products: &[F]) -> Vec<F> {
    assert!(!quotient_chunk_products.is_empty());
    let mut res = Vec::with_capacity(quotient_chunk_products.len());
    let mut acc = z_x;
    for &quotient_chunk_product in quotient_chunk_products {
        acc *= quotient_chunk_product;
        res.push(acc);
    }
    res
}

/// Returns the length of the output of `partial_products()` on a vector of length `n`.
pub(crate) fn num_partial_products(n: usize, max_degree: usize) -> usize {
    debug_assert!(max_degree > 1);
    let chunk_size = max_degree;
    // We'll split the product into `n.div_ceil( chunk_size)` chunks, but the last chunk will
    // be associated with Z(gx) itself. Thus we subtract one to get the chunks associated with
    // partial products.
    n.div_ceil(chunk_size) - 1
}

/// Appends the checks relating each pair of partial product accumulators to `out`. In
/// particular, the sequence of accumulators starts with `Z(x)`, then contains each partial
/// product polynomial `p_i(x)`, and finally `Z(g x)`. See the partial products section of the
/// Plonky2 paper.
pub(crate) fn append_partial_product_checks<F: Field>(
    numerators: &[F],
    denominators: &[F],
    partials: &[F],
    z_x: F,
    z_gx: F,
    max_degree: usize,
    out: &mut Vec<F>,
) {
    assert!(max_degree > 1);
    assert_eq!(numerators.len(), denominators.len());
    let num_chunks = numerators.len().div_ceil(max_degree);
    assert_eq!(partials.len() + 1, num_chunks);
    out.reserve(num_chunks);

    let mut prev_acc = z_x;
    for chunk_index in 0..num_chunks {
        let start = chunk_index * max_degree;
        let end = (start + max_degree).min(numerators.len());
        let mut num_chunk_product = F::ONE;
        let mut den_chunk_product = F::ONE;
        for (&numerator, &denominator) in
            numerators[start..end].iter().zip(&denominators[start..end])
        {
            num_chunk_product *= numerator;
            den_chunk_product *= denominator;
        }
        let next_acc = partials.get(chunk_index).copied().unwrap_or(z_gx);
        // Assert that next_acc * deno_product = prev_acc * nume_product.
        out.push(prev_acc * num_chunk_product - next_acc * den_chunk_product);
        prev_acc = next_acc;
    }
}

/// Checks the relationship between each pair of partial product accumulators. In particular, this
/// sequence of accumulators starts with `Z(x)`, then contains each partial product polynomials
/// `p_i(x)`, and finally `Z(g x)`. See the partial products section of the Plonky2 paper.
pub(crate) fn check_partial_products_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    numerators: &[ExtensionTarget<D>],
    denominators: &[ExtensionTarget<D>],
    partials: &[ExtensionTarget<D>],
    z_x: ExtensionTarget<D>,
    z_gx: ExtensionTarget<D>,
    max_degree: usize,
) -> Vec<ExtensionTarget<D>> {
    debug_assert!(max_degree > 1);
    let product_accs = iter::once(&z_x)
        .chain(partials.iter())
        .chain(iter::once(&z_gx));
    let chunk_size = max_degree;
    numerators
        .chunks(chunk_size)
        .zip_eq(denominators.chunks(chunk_size))
        .zip_eq(product_accs.tuple_windows())
        .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
            let nume_product = builder.mul_many_extension(nume_chunk);
            let deno_product = builder.mul_many_extension(deno_chunk);
            let next_acc_deno = builder.mul_extension(next_acc, deno_product);
            // Assert that next_acc * deno_product = prev_acc * nume_product.
            builder.mul_sub_extension(prev_acc, nume_product, next_acc_deno)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;

    #[test]
    fn test_partial_products() {
        type F = GoldilocksField;
        let denominators = vec![F::ONE; 6];
        let z_x = F::ONE;
        let v = field_vec(&[1, 2, 3, 4, 5, 6]);
        let z_gx = F::from_canonical_u64(720);
        let mut checks = Vec::new();
        let quotient_chunks_prods = quotient_chunk_products(&v, 2);
        assert_eq!(quotient_chunks_prods, field_vec(&[2, 12, 30]));
        let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
        let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
        assert_eq!(pps_and_z_gx, field_vec(&[2, 24, 720]));

        let nums = num_partial_products(v.len(), 2);
        assert_eq!(pps.len(), nums);
        append_partial_product_checks(&v, &denominators, pps, z_x, z_gx, 2, &mut checks);
        assert!(checks.iter().all(|x| x.is_zero()));

        let quotient_chunks_prods = quotient_chunk_products(&v, 3);
        assert_eq!(quotient_chunks_prods, field_vec(&[6, 120]));
        let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
        let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
        assert_eq!(pps_and_z_gx, field_vec(&[6, 720]));
        let nums = num_partial_products(v.len(), 3);
        assert_eq!(pps.len(), nums);
        checks.clear();
        append_partial_product_checks(&v, &denominators, pps, z_x, z_gx, 3, &mut checks);
        assert!(checks.iter().all(|x| x.is_zero()));
    }

    fn partial_product_checks_reference<F: Field>(
        numerators: &[F],
        denominators: &[F],
        partials: &[F],
        z_x: F,
        z_gx: F,
        max_degree: usize,
    ) -> Vec<F> {
        let product_accs = iter::once(&z_x)
            .chain(partials.iter())
            .chain(iter::once(&z_gx));
        numerators
            .chunks(max_degree)
            .zip_eq(denominators.chunks(max_degree))
            .zip_eq(product_accs.tuple_windows())
            .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
                let num_chunk_product = nume_chunk.iter().copied().product();
                let den_chunk_product = deno_chunk.iter().copied().product();
                prev_acc * num_chunk_product - next_acc * den_chunk_product
            })
            .collect()
    }

    #[test]
    fn appended_partial_product_checks_match_iterator_reference() {
        type F = GoldilocksField;

        for len in [1, 2, 3, 7, 8, 9, 17, 79, 80, 81] {
            let numerators: Vec<F> = (0..len)
                .map(|i| F::from_canonical_usize(3 * i + 5))
                .collect();
            let denominators: Vec<F> = (0..len)
                .map(|i| F::from_canonical_usize(7 * i + 11))
                .collect();

            for degree in [2, 3, 7, 8, 11] {
                let num_chunks = len.div_ceil(degree);
                let partials: Vec<F> = (0..num_chunks - 1)
                    .map(|i| F::from_canonical_usize(13 * i + 17))
                    .collect();
                let z_x = F::from_canonical_u64(19);
                let z_gx = F::from_canonical_u64(23);
                let reference = partial_product_checks_reference(
                    &numerators,
                    &denominators,
                    &partials,
                    z_x,
                    z_gx,
                    degree,
                );

                let prefix = F::from_canonical_u64(29);
                let mut actual = vec![prefix];
                append_partial_product_checks(
                    &numerators,
                    &denominators,
                    &partials,
                    z_x,
                    z_gx,
                    degree,
                    &mut actual,
                );
                assert_eq!(actual[0], prefix);
                assert_eq!(&actual[1..], reference, "len={len}, degree={degree}");
            }
        }
    }

    fn field_vec<F: Field>(xs: &[usize]) -> Vec<F> {
        xs.iter().map(|&x| F::from_canonical_usize(x)).collect()
    }
}
