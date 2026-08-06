#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::iter;

use itertools::Itertools;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::plonk::circuit_builder::CircuitBuilder;

#[cfg_attr(not(test), allow(dead_code))]
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

pub(crate) fn quotient_chunk_products_from_numerators_and_inverses<F: Field>(
    mut numerators: impl ExactSizeIterator<Item = F>,
    denominator_inverses: &[F],
    max_degree: usize,
) -> Vec<F> {
    debug_assert!(max_degree > 1);
    let num_values = numerators.len();
    debug_assert!(!denominator_inverses.is_empty());
    debug_assert_eq!(num_values, denominator_inverses.len());

    let mut result = Vec::with_capacity(num_values.div_ceil(max_degree));
    let mut denominator_inverses = denominator_inverses.iter().copied();
    let mut remaining = num_values;
    while remaining != 0 {
        let products_in_chunk = remaining.min(max_degree);
        result.push(next_quotient_chunk_product(
            &mut numerators,
            &mut denominator_inverses,
            products_in_chunk,
        ));
        remaining -= products_in_chunk;
    }
    result
}

fn next_quotient_chunk_product<F: Field>(
    numerators: &mut impl Iterator<Item = F>,
    denominator_inverses: &mut impl Iterator<Item = F>,
    products_in_chunk: usize,
) -> F {
    let mut chunk_product = F::ONE;
    for _ in 0..products_in_chunk {
        let numerator = numerators.next().unwrap();
        let denominator_inverse = denominator_inverses.next().unwrap();
        chunk_product *= numerator * denominator_inverse;
    }
    chunk_product
}

pub(crate) fn quotient_chunk_products_80_by_8_from_numerators_and_inverses<F: Field>(
    mut numerators: impl ExactSizeIterator<Item = F>,
    denominator_inverses: &[F],
) -> [F; 10] {
    debug_assert_eq!(numerators.len(), 80);
    debug_assert_eq!(denominator_inverses.len(), 80);
    let mut denominator_inverses = denominator_inverses.iter().copied();
    core::array::from_fn(|_| {
        next_quotient_chunk_product(&mut numerators, &mut denominator_inverses, 8)
    })
}

pub(crate) const fn use_fixed_permutation_chunk_products(
    num_routed_wires: usize,
    max_degree: usize,
    row_width: usize,
) -> bool {
    num_routed_wires == 80 && max_degree == 8 && row_width == 10
}

/// Compute partial products of the original vector `v` such that all products consist of `max_degree`
/// or less elements. This is done until we've computed the product `P` of all elements in the vector.
#[cfg_attr(not(test), allow(dead_code))]
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

pub(crate) fn partial_products_and_zs_from_chunk_rows<F: Field, R: AsRef<[F]>>(
    chunk_rows: &[R],
    num_partial_products: usize,
) -> Vec<Vec<F>> {
    assert!(!chunk_rows.is_empty());
    let row_width = num_partial_products + 1;
    let mut columns = (0..row_width)
        .map(|_| Vec::with_capacity(chunk_rows.len()))
        .collect::<Vec<_>>();
    let (z_column, partial_product_columns) = columns.split_last_mut().unwrap();
    let mut z_x = F::ONE;

    for chunk_row in chunk_rows {
        let chunk_row = chunk_row.as_ref();
        debug_assert_eq!(chunk_row.len(), row_width);
        let (&last_chunk_product, prefix_chunk_products) = chunk_row.split_last().unwrap();
        debug_assert_eq!(prefix_chunk_products.len(), partial_product_columns.len());
        let incoming_z_x = z_x;
        let mut acc = incoming_z_x;
        for (column, &chunk_product) in partial_product_columns
            .iter_mut()
            .zip(prefix_chunk_products)
        {
            acc *= chunk_product;
            column.push(acc);
        }
        acc *= last_chunk_product;
        z_column.push(incoming_z_x);
        z_x = acc;
    }

    columns
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
            let num_chunk_product = nume_chunk.iter().copied().product();
            let den_chunk_product = deno_chunk.iter().copied().product();
            // Assert that next_acc * deno_product = prev_acc * nume_product.
            prev_acc * num_chunk_product - next_acc * den_chunk_product
        })
        .collect()
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
    use core::mem::swap;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field64, PrimeField64};
    use crate::util::transpose;

    #[test]
    fn fused_quotient_chunk_products_match_independent_materialization_raw() {
        type F = GoldilocksField;

        for len in [1usize, 7, 8, 9, 10, 80] {
            let numerators = (0..len)
                .map(|i| GoldilocksField(F::ORDER + 1 + i as u64))
                .collect::<Vec<_>>();
            let denominator_inverses = (0..len)
                .map(|i| GoldilocksField(u64::MAX - i as u64))
                .collect::<Vec<_>>();
            let quotient_values = numerators
                .iter()
                .zip(&denominator_inverses)
                .map(|(&numerator, &inverse)| numerator * inverse)
                .collect::<Vec<_>>();
            let expected = quotient_chunk_products(&quotient_values, 8);

            let actual = quotient_chunk_products_from_numerators_and_inverses(
                numerators.iter().copied(),
                &denominator_inverses,
                8,
            );

            assert_eq!(actual.len(), len.div_ceil(8));
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_noncanonical_u64())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_noncanonical_u64())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn fixed_80_by_8_chunk_products_match_dynamic_path_raw() {
        type F = GoldilocksField;

        let numerators = (0..80)
            .map(|i| GoldilocksField(F::ORDER + 1 + i as u64))
            .collect::<Vec<_>>();
        let denominator_inverses = (0..80)
            .map(|i| GoldilocksField(u64::MAX - i as u64))
            .collect::<Vec<_>>();
        let expected = quotient_chunk_products_from_numerators_and_inverses(
            numerators.iter().copied(),
            &denominator_inverses,
            8,
        );

        let actual = quotient_chunk_products_80_by_8_from_numerators_and_inverses(
            numerators.iter().copied(),
            &denominator_inverses,
        );

        assert_eq!(actual.len(), 10);
        assert_eq!(
            actual.map(|value| value.to_noncanonical_u64()).as_slice(),
            expected
                .iter()
                .map(|value| value.to_noncanonical_u64())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fixed_permutation_shape_is_exact_and_other_shapes_fall_back() {
        assert!(use_fixed_permutation_chunk_products(80, 8, 10));
        assert!(!use_fixed_permutation_chunk_products(79, 8, 10));
        assert!(!use_fixed_permutation_chunk_products(80, 4, 20));
        assert!(!use_fixed_permutation_chunk_products(80, 8, 9));
        assert!(!use_fixed_permutation_chunk_products(88, 8, 11));
    }

    #[test]
    fn direct_partial_product_columns_accept_fixed_rows_and_match_dynamic_path_raw() {
        type F = GoldilocksField;

        let fixed_rows = (0..5)
            .map(|row| {
                core::array::from_fn(|chunk| {
                    GoldilocksField(F::ORDER + 1 + (row * 10 + chunk) as u64)
                })
            })
            .collect::<Vec<[F; 10]>>();
        let dynamic_rows = fixed_rows
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        let expected = partial_products_and_zs_from_chunk_rows(&dynamic_rows, 9);

        let actual = partial_products_and_zs_from_chunk_rows(&fixed_rows, 9);

        assert_eq!(actual.len(), 10);
        assert!(actual.iter().all(|column| column.len() == 5));
        assert_eq!(
            actual
                .iter()
                .flatten()
                .map(|value| value.to_noncanonical_u64())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .flatten()
                .map(|value| value.to_noncanonical_u64())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn direct_partial_product_columns_match_legacy_rows_swap_and_transpose_raw() {
        type F = GoldilocksField;

        for num_rows in [1usize, 5, 32] {
            for num_chunks in [1usize, 2, 10] {
                let chunk_rows = (0..num_rows)
                    .map(|row| {
                        (0..num_chunks)
                            .map(|chunk| {
                                GoldilocksField(F::ORDER + 1 + (row * num_chunks + chunk) as u64)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();

                let mut z_x = F::ONE;
                let mut legacy_rows = Vec::with_capacity(num_rows);
                for chunk_row in &chunk_rows {
                    let mut row = partial_products_and_z_gx(z_x, chunk_row);
                    swap(&mut z_x, &mut row[num_chunks - 1]);
                    legacy_rows.push(row);
                }
                let expected = transpose(&legacy_rows);

                let actual = partial_products_and_zs_from_chunk_rows(&chunk_rows, num_chunks - 1);

                assert_eq!(actual.len(), num_chunks);
                assert!(actual.iter().all(|column| column.len() == num_rows));
                assert_eq!(
                    actual
                        .iter()
                        .flatten()
                        .map(|value| value.to_noncanonical_u64())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .flatten()
                        .map(|value| value.to_noncanonical_u64())
                        .collect::<Vec<_>>()
                );
            }
        }
    }

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

    fn field_vec<F: Field>(xs: &[usize]) -> Vec<F> {
        xs.iter().map(|&x| F::from_canonical_usize(x)).collect()
    }
}
