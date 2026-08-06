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

/// Checks the relationship between each pair of partial product accumulators, appending the check
/// terms to `out`. In particular, this sequence of accumulators starts with `Z(x)`, then contains
/// each partial product polynomials `p_i(x)`, and finally `Z(g x)`. See the partial products
/// section of the Plonky2 paper.
///
/// This is the allocation-free form of [`check_partial_products`]: the batched vanishing-poly
/// evaluator calls it once per point and challenge with its worker-local scratch vector, instead
/// of collecting a fresh ten-element `Vec` and immediately `extend`ing it into that same scratch.
/// `out` is deliberately not cleared: for a given point, challenge `i + 1` must append after
/// challenge `i`.
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

/// Builds permutation numerator and denominator chunk products directly from the
/// wire and sigma evaluations, then appends their partial-product checks to `out`.
/// Both products preserve the original routed-wire multiplication order independently.
/// Interleaving their instructions changes scheduling only; neither fold depends on the other.
/// Partial-product accumulator selection is unchanged at every chunk boundary,
/// starting from `Z(x)` and including the final transition to `Z(gx)`.
/// Intermediate chunks consume the partial-product slice in the same index order.
/// Each wire evaluation is loaded once and shared by its numerator and denominator factors.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn check_permutation_partial_products_into<F: Field>(
    num_wires: usize,
    mut local_wire: impl FnMut(usize) -> F,
    s_sigmas: &[F],
    beta_k_is: &[F],
    beta: F,
    gamma: F,
    x: F,
    partials: &[F],
    z_x: F,
    z_gx: F,
    max_degree: usize,
    out: &mut Vec<F>,
) {
    debug_assert!(max_degree > 1);
    debug_assert!(num_wires > 0);
    debug_assert_eq!(s_sigmas.len(), num_wires);
    debug_assert_eq!(beta_k_is.len(), num_wires);

    let num_chunks = num_wires.div_ceil(max_degree);
    debug_assert_eq!(partials.len(), num_chunks - 1);
    for chunk_index in 0..num_chunks {
        let start = chunk_index * max_degree;
        let end = (start + max_degree).min(num_wires);
        let mut numerator_product = F::ONE;
        let mut denominator_product = F::ONE;
        for j in start..end {
            let wire_value = local_wire(j);
            numerator_product *= wire_value + beta_k_is[j] * x + gamma;
            denominator_product *= wire_value + beta * s_sigmas[j] + gamma;
        }

        let prev_acc = if chunk_index == 0 {
            z_x
        } else {
            partials[chunk_index - 1]
        };
        let next_acc = partials.get(chunk_index).copied().unwrap_or(z_gx);
        out.push(prev_acc * numerator_product - next_acc * denominator_product);
    }
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
    // One term per numerator/denominator chunk: the historical exact capacity of the
    // `collect()` this wrapper replaces.
    let mut res = Vec::with_capacity(numerators.len().div_ceil(max_degree));
    check_partial_products_into(
        numerators,
        denominators,
        partials,
        z_x,
        z_gx,
        max_degree,
        &mut res,
    );
    res
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
    use crate::field::types::PrimeField64;

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

    /// Independent copy of the pre-`_into` implementation, retained only as a
    /// differential-test oracle. Comparing `check_partial_products` (which now
    /// delegates to `_into`) against `_into` would be self-referential.
    fn check_partial_products_legacy<F: Field>(
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
                prev_acc * num_chunk_product - next_acc * den_chunk_product
            })
            .collect()
    }

    /// Deterministic, mostly-noncanonical Goldilocks values so raw-representation
    /// comparisons are meaningful.
    fn noncanonical_vec(len: usize, seed: u64) -> Vec<GoldilocksField> {
        (0..len as u64)
            .map(|i| {
                GoldilocksField::from_noncanonical_u64(
                    u64::MAX - seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(3 * i),
                )
            })
            .collect()
    }

    fn raw(values: &[GoldilocksField]) -> Vec<u64> {
        values.iter().map(|x| x.to_noncanonical_u64()).collect()
    }

    /// `_into` must append exactly the legacy terms (raw representation equality),
    /// for both a full final chunk and a partial final chunk, and the allocating
    /// wrapper must preserve the historical exact capacity.
    #[test]
    fn test_check_partial_products_into_matches_legacy() {
        type F = GoldilocksField;
        // (len, max_degree): 8/2 has all-full chunks, 7/3 has a partial final chunk.
        for &(len, max_degree) in &[(8usize, 2usize), (7, 3)] {
            let numerators = noncanonical_vec(len, 1);
            let denominators = noncanonical_vec(len, 2);
            let num_partials = len.div_ceil(max_degree) - 1;
            let partials = noncanonical_vec(num_partials, 3);
            let z_x = F::from_noncanonical_u64(u64::MAX);
            let z_gx = F::from_noncanonical_u64(u64::MAX - 7);

            let expected = check_partial_products_legacy(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
            );
            assert_eq!(expected.len(), len.div_ceil(max_degree));

            let mut out = Vec::new();
            check_partial_products_into(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
                &mut out,
            );
            assert_eq!(raw(&out), raw(&expected));

            let wrapped = check_partial_products(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
            );
            assert_eq!(raw(&wrapped), raw(&expected));
            assert_eq!(wrapped.capacity(), len.div_ceil(max_degree));
        }
    }

    /// Two sequential `_into` calls (the per-point challenge loop) must append in
    /// order after an existing prefix, without clearing and without reallocating
    /// when spare capacity suffices.
    #[test]
    fn test_check_partial_products_into_appends_after_prefix() {
        type F = GoldilocksField;
        let len = 8usize;
        let max_degree = 2usize;
        let num_partials = len.div_ceil(max_degree) - 1;

        let mut expected_terms = Vec::new();
        let mut challenge_inputs = Vec::new();
        for challenge in 0..2u64 {
            let numerators = noncanonical_vec(len, 10 + challenge);
            let denominators = noncanonical_vec(len, 20 + challenge);
            let partials = noncanonical_vec(num_partials, 30 + challenge);
            let z_x = F::from_noncanonical_u64(u64::MAX - challenge);
            let z_gx = F::from_noncanonical_u64(u64::MAX - 100 - challenge);
            expected_terms.extend(check_partial_products_legacy(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
            ));
            challenge_inputs.push((numerators, denominators, partials, z_x, z_gx));
        }

        let sentinel = F::from_noncanonical_u64(u64::MAX - 42);
        let mut out = Vec::with_capacity(1 + expected_terms.len());
        out.push(sentinel);
        let base_ptr = out.as_ptr();
        for (numerators, denominators, partials, z_x, z_gx) in &challenge_inputs {
            check_partial_products_into(
                numerators,
                denominators,
                partials,
                *z_x,
                *z_gx,
                max_degree,
                &mut out,
            );
        }
        // Sufficient spare capacity: appending must not have reallocated.
        assert_eq!(out.as_ptr(), base_ptr);
        assert_eq!(out[0].to_noncanonical_u64(), sentinel.to_noncanonical_u64());
        assert_eq!(raw(&out[1..]), raw(&expected_terms));

        // Forced growth: capacity exhausted by the sentinel alone. Prefix and
        // terms must survive the reallocation.
        let mut small = Vec::with_capacity(1);
        small.push(sentinel);
        let (numerators, denominators, partials, z_x, z_gx) = &challenge_inputs[0];
        check_partial_products_into(
            numerators,
            denominators,
            partials,
            *z_x,
            *z_gx,
            max_degree,
            &mut small,
        );
        assert_eq!(
            small[0].to_noncanonical_u64(),
            sentinel.to_noncanonical_u64()
        );
        assert_eq!(
            raw(&small[1..]),
            raw(&expected_terms[..len.div_ceil(max_degree)])
        );
    }

    #[test]
    fn test_check_permutation_partial_products_into_matches_legacy() {
        type F = GoldilocksField;

        for &(len, max_degree) in &[(8usize, 2usize), (7, 3)] {
            let local_wires = noncanonical_vec(len, 41);
            let s_sigmas = noncanonical_vec(len, 42);
            let beta_k_is = noncanonical_vec(len, 43);
            let beta = F::from_noncanonical_u64(u64::MAX - 44);
            let gamma = F::from_noncanonical_u64(u64::MAX - 45);
            let x = F::from_noncanonical_u64(u64::MAX - 46);
            let partials = noncanonical_vec(len.div_ceil(max_degree) - 1, 47);
            let z_x = F::from_noncanonical_u64(u64::MAX - 48);
            let z_gx = F::from_noncanonical_u64(u64::MAX - 49);

            let numerators = (0..len)
                .map(|j| local_wires[j] + beta_k_is[j] * x + gamma)
                .collect::<Vec<_>>();
            let denominators = (0..len)
                .map(|j| local_wires[j] + beta * s_sigmas[j] + gamma)
                .collect::<Vec<_>>();
            let expected = check_partial_products_legacy(
                &numerators,
                &denominators,
                &partials,
                z_x,
                z_gx,
                max_degree,
            );

            let sentinel = F::from_noncanonical_u64(u64::MAX - 50);
            let mut actual = vec![sentinel];
            check_permutation_partial_products_into(
                len,
                |j| local_wires[j],
                &s_sigmas,
                &beta_k_is,
                beta,
                gamma,
                x,
                &partials,
                z_x,
                z_gx,
                max_degree,
                &mut actual,
            );

            assert_eq!(
                actual[0].to_noncanonical_u64(),
                sentinel.to_noncanonical_u64()
            );
            assert_eq!(raw(&actual[1..]), raw(&expected));
        }
    }
}
