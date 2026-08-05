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

/// Checks the relationship between each pair of partial product accumulators. In particular, this
/// sequence of accumulators starts with `Z(x)`, then contains each partial product polynomials
/// `p_i(x)`, and finally `Z(g x)`. See the partial products section of the Plonky2 paper.
pub(crate) fn check_partial_products<'a, F: Field>(
    numerators: &'a [F],
    denominators: &'a [F],
    partials: &'a [F],
    z_x: F,
    z_gx: F,
    max_degree: usize,
) -> impl Iterator<Item = F> + 'a {
    debug_assert!(max_degree > 1);
    let product_accs = iter::once(z_x)
        .chain(partials.iter().copied())
        .chain(iter::once(z_gx));
    let chunk_size = max_degree;
    numerators
        .chunks(chunk_size)
        .zip_eq(denominators.chunks(chunk_size))
        .zip_eq(product_accs.tuple_windows())
        .map(|((nume_chunk, deno_chunk), (prev_acc, next_acc))| {
            let num_chunk_product = nume_chunk.iter().copied().product();
            let den_chunk_product = deno_chunk.iter().copied().product();
            // Assert that next_acc * deno_product = prev_acc * nume_product.
            prev_acc * num_chunk_product - next_acc * den_chunk_product
        })
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
        let quotient_chunks_prods = quotient_chunk_products(&v, 2);
        assert_eq!(quotient_chunks_prods, field_vec(&[2, 12, 30]));
        let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
        let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
        assert_eq!(pps_and_z_gx, field_vec(&[2, 24, 720]));

        let nums = num_partial_products(v.len(), 2);
        assert_eq!(pps.len(), nums);
        assert!(check_partial_products(&v, &denominators, pps, z_x, z_gx, 2)
            .all(|x| x.is_zero()));

        let quotient_chunks_prods = quotient_chunk_products(&v, 3);
        assert_eq!(quotient_chunks_prods, field_vec(&[6, 120]));
        let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
        let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
        assert_eq!(pps_and_z_gx, field_vec(&[6, 720]));
        let nums = num_partial_products(v.len(), 3);
        assert_eq!(pps.len(), nums);
        assert!(check_partial_products(&v, &denominators, pps, z_x, z_gx, 3)
            .all(|x| x.is_zero()));
    }

    #[test]
    fn check_partial_products_iterator_matches_materialized_reference_raw() {
        type F = GoldilocksField;

        for len in 1_usize..=40 {
            for max_degree in 2_usize..=9 {
                let numerators = (0..len)
                    .map(|i| {
                        GoldilocksField((i as u64)
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            .wrapping_add(u64::MAX - len as u64))
                    })
                    .collect::<Vec<_>>();
                let denominators = (0..len)
                    .map(|i| {
                        GoldilocksField((i as u64)
                            .wrapping_mul(0xd1b5_4a32_d192_ed03)
                            .wrapping_add(0xffff_ffff_0000_0001))
                    })
                    .collect::<Vec<_>>();
                let partials = (0..len.div_ceil(max_degree) - 1)
                    .map(|i| {
                        GoldilocksField((i as u64).wrapping_mul(17).wrapping_add(u64::MAX - 3))
                    })
                    .collect::<Vec<_>>();
                let z_x = GoldilocksField(u64::MAX);
                let z_gx = GoldilocksField(0xffff_ffff_0000_0001);

                let product_accs = iter::once(&z_x)
                    .chain(partials.iter())
                    .chain(iter::once(&z_gx));
                let expected = numerators
                    .chunks(max_degree)
                    .zip_eq(denominators.chunks(max_degree))
                    .zip_eq(product_accs.tuple_windows())
                    .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
                        let num_chunk_product: F = nume_chunk.iter().copied().product();
                        let den_chunk_product: F = deno_chunk.iter().copied().product();
                        prev_acc * num_chunk_product - next_acc * den_chunk_product
                    })
                    .collect::<Vec<F>>();
                let actual = check_partial_products(
                    &numerators,
                    &denominators,
                    &partials,
                    z_x,
                    z_gx,
                    max_degree,
                )
                .collect::<Vec<_>>();

                assert_eq!(
                    actual.iter().map(|value| value.0).collect::<Vec<_>>(),
                    expected.iter().map(|value| value.0).collect::<Vec<_>>(),
                    "len={len}, max_degree={max_degree}",
                );
            }
        }
    }

    fn field_vec<F: Field>(xs: &[usize]) -> Vec<F> {
        xs.iter().map(|&x| F::from_canonical_usize(x)).collect()
    }
}
