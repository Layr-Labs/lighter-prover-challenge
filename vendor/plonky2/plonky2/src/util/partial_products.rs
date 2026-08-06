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
pub(crate) fn check_partial_products<F: Field>(
    numerators: &[F],
    denominators: &[F],
    partials: &[F],
    z_x: F,
    z_gx: F,
    max_degree: usize,
) -> Vec<F> {
    let mut result = Vec::with_capacity(numerators.len().div_ceil(max_degree));
    check_partial_products_into(
        numerators,
        denominators,
        partials,
        z_x,
        z_gx,
        max_degree,
        &mut result,
    );
    result
}

/// Appends the checks for each pair of partial product accumulators to `out`.
pub(crate) fn check_partial_products_into<F: Field>(
    numerators: &[F],
    denominators: &[F],
    partials: &[F],
    z_x: F,
    z_gx: F,
    max_degree: usize,
    out: &mut Vec<F>,
) {
    debug_assert!(max_degree > 1);
    let product_accs = iter::once(&z_x)
        .chain(partials.iter())
        .chain(iter::once(&z_gx));
    let chunk_size = max_degree;
    out.extend(
        numerators
            .chunks(chunk_size)
            .zip_eq(denominators.chunks(chunk_size))
            .zip_eq(product_accs.tuple_windows())
            .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
                let num_chunk_product = nume_chunk.iter().copied().product();
                let den_chunk_product = deno_chunk.iter().copied().product();
                // Assert that next_acc * deno_product = prev_acc * nume_product.
                prev_acc * num_chunk_product - next_acc * den_chunk_product
            }),
    );
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
    use crate::field::types::{Field64, PrimeField64};

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
            .iter()
            .all(|x| x.is_zero()));

        let quotient_chunks_prods = quotient_chunk_products(&v, 3);
        assert_eq!(quotient_chunks_prods, field_vec(&[6, 120]));
        let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
        let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
        assert_eq!(pps_and_z_gx, field_vec(&[6, 720]));
        let nums = num_partial_products(v.len(), 3);
        assert_eq!(pps.len(), nums);
        assert!(check_partial_products(&v, &denominators, pps, z_x, z_gx, 3)
            .iter()
            .all(|x| x.is_zero()));
    }

    #[test]
    fn check_partial_products_into_matches_raw_output_for_full_and_partial_chunks() {
        type F = GoldilocksField;

        for (len, max_degree) in [(8, 4), (10, 4)] {
            let numerators = noncanonical_values(len, 1);
            let denominators = noncanonical_values(len, 101);
            let partials = noncanonical_values(len.div_ceil(max_degree) - 1, 201);
            let z_x = GoldilocksField(F::ORDER + 301);
            let z_gx = GoldilocksField(F::ORDER + 302);
            let expected = legacy_check_partial_products(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
            );
            let legacy_actual = check_partial_products(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
            );
            assert_eq!(legacy_actual.capacity(), len.div_ceil(max_degree));
            assert_eq!(raw_values(&legacy_actual), raw_values(&expected));

            let mut actual = Vec::new();
            check_partial_products_into(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
                &mut actual,
            );

            assert_eq!(raw_values(&actual), raw_values(&expected));
        }
    }

    #[test]
    fn check_partial_products_into_appends_without_reallocating_spare_capacity() {
        type F = GoldilocksField;
        let numerators = noncanonical_values(10, 1);
        let denominators = noncanonical_values(10, 101);
        let partials = noncanonical_values(2, 201);
        let z_x = GoldilocksField(F::ORDER + 301);
        let z_gx = GoldilocksField(F::ORDER + 302);
        let expected =
            legacy_check_partial_products(&numerators, &denominators, &partials, z_x, z_gx, 4);
        let prefix = [
            GoldilocksField(F::ORDER + 401),
            GoldilocksField(F::ORDER + 402),
        ];
        let mut actual = Vec::with_capacity(8);
        actual.extend(prefix);
        let original_capacity = actual.capacity();

        check_partial_products_into(
            &numerators,
            &denominators,
            &partials,
            z_x,
            z_gx,
            4,
            &mut actual,
        );

        assert_eq!(actual.capacity(), original_capacity);
        assert_eq!(raw_values(&actual[..prefix.len()]), raw_values(&prefix));
        assert_eq!(raw_values(&actual[prefix.len()..]), raw_values(&expected));
    }

    #[test]
    fn check_partial_products_into_grows_when_capacity_is_exhausted() {
        type F = GoldilocksField;
        let numerators = noncanonical_values(80, 1);
        let denominators = noncanonical_values(80, 101);
        let partials = noncanonical_values(9, 201);
        let z_x = GoldilocksField(F::ORDER + 301);
        let z_gx = GoldilocksField(F::ORDER + 302);
        let expected =
            legacy_check_partial_products(&numerators, &denominators, &partials, z_x, z_gx, 8);
        let prefix = [GoldilocksField(F::ORDER + 401)];
        let mut actual = Vec::with_capacity(prefix.len());
        actual.extend(prefix);
        let original_capacity = actual.capacity();

        check_partial_products_into(
            &numerators,
            &denominators,
            &partials,
            z_x,
            z_gx,
            8,
            &mut actual,
        );

        assert!(actual.capacity() > original_capacity);
        assert!(actual.capacity() >= actual.len());
        assert_eq!(raw_values(&actual[..prefix.len()]), raw_values(&prefix));
        assert_eq!(raw_values(&actual[prefix.len()..]), raw_values(&expected));
    }

    #[test]
    fn check_partial_products_into_preserves_multi_call_term_sequence() {
        type F = GoldilocksField;
        let numerators = noncanonical_values(10, 1);
        let denominators = noncanonical_values(10, 101);
        let partials = noncanonical_values(2, 201);
        let next_numerators = noncanonical_values(10, 501);
        let next_denominators = noncanonical_values(10, 601);
        let next_partials = noncanonical_values(2, 701);
        let mut expected = legacy_check_partial_products(
            &numerators,
            &denominators,
            &partials,
            GoldilocksField(F::ORDER + 301),
            GoldilocksField(F::ORDER + 302),
            4,
        );
        expected.extend(legacy_check_partial_products(
            &next_numerators,
            &next_denominators,
            &next_partials,
            GoldilocksField(F::ORDER + 801),
            GoldilocksField(F::ORDER + 802),
            4,
        ));

        let mut actual = Vec::new();
        check_partial_products_into(
            &numerators,
            &denominators,
            &partials,
            GoldilocksField(F::ORDER + 301),
            GoldilocksField(F::ORDER + 302),
            4,
            &mut actual,
        );
        check_partial_products_into(
            &next_numerators,
            &next_denominators,
            &next_partials,
            GoldilocksField(F::ORDER + 801),
            GoldilocksField(F::ORDER + 802),
            4,
            &mut actual,
        );

        assert_eq!(raw_values(&actual), raw_values(&expected));
    }

    fn noncanonical_values(len: usize, offset: u64) -> Vec<GoldilocksField> {
        (0..len)
            .map(|i| GoldilocksField(GoldilocksField::ORDER + offset + i as u64))
            .collect()
    }

    fn raw_values(values: &[GoldilocksField]) -> Vec<u64> {
        values
            .iter()
            .map(PrimeField64::to_noncanonical_u64)
            .collect()
    }

    fn legacy_check_partial_products<F: Field>(
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

    fn field_vec<F: Field>(xs: &[usize]) -> Vec<F> {
        xs.iter().map(|&x| F::from_canonical_usize(x)).collect()
    }
}
