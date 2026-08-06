//! plonky2 prover implementation.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
use core::cmp::min;

use anyhow::{ensure, Result};
use hashbrown::HashMap;
use plonky2_maybe_rayon::*;

use super::circuit_builder::{LookupChallenges, LookupWire};
use crate::field::extension::Extendable;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::{BatchLayout, PolynomialBatch};
use crate::gates::lookup::LookupGate;
use crate::gates::lookup_table::LookupTableGate;
use crate::gates::selectors::LookupSelectors;
use crate::hash::hash_types::RichField;
use crate::iop::challenger::Challenger;
use crate::iop::generator::generate_partial_witness;
use crate::iop::target::Target;
use crate::iop::witness::{MatrixWitness, PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::NUM_COINS_LOOKUP;
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};
use crate::plonk::vanishing_poly::{
    eval_vanishing_poly_base_batch, get_lut_poly, VanishingScratch,
};
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::timed;
use crate::util::log2_ceil;
use crate::util::timing::TimingTree;

/// Set all the lookup gate wires (including multiplicities) and pad unused LU slots.
/// Warning: rows are in descending order: the first gate to appear is the last LU gate, and
/// the last gate to appear is the first LUT gate.
pub fn set_lookup_wires<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    pw: &mut PartitionWitness<F>,
) -> Result<()> {
    for (
        lut_index,
        &LookupWire {
            last_lu_gate: _,
            last_lut_gate,
            first_lut_gate,
        },
    ) in prover_data.lookup_rows.iter().enumerate()
    {
        let lut_len = common_data.luts[lut_index].len();
        let num_entries = LookupGate::num_slots(&common_data.config);
        let num_lut_entries = LookupTableGate::num_slots(&common_data.config);

        // Compute multiplicities.
        let mut multiplicities = vec![0; lut_len];

        let table_value_to_idx: HashMap<u16, usize> = common_data.luts[lut_index]
            .iter()
            .enumerate()
            .map(|(i, (inp_target, _))| (*inp_target, i))
            .collect();

        for (inp_target, _) in prover_data.lut_to_lookups[lut_index].iter() {
            let inp_value = pw.get_target(*inp_target);
            let idx = table_value_to_idx
                .get(&u16::try_from(inp_value.to_canonical_u64()).unwrap())
                .unwrap();

            multiplicities[*idx] += 1;
        }

        // Pad the last `LookupGate` with the first entry from the LUT.
        let remaining_slots = (num_entries
            - (prover_data.lut_to_lookups[lut_index].len() % num_entries))
            % num_entries;
        let (first_inp_value, first_out_value) = common_data.luts[lut_index][0];
        for slot in (num_entries - remaining_slots)..num_entries {
            let inp_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_inp(slot));
            let out_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_out(slot));
            pw.set_target(inp_target, F::from_canonical_u16(first_inp_value))?;
            pw.set_target(out_target, F::from_canonical_u16(first_out_value))?;

            multiplicities[0] += 1;
        }

        for lut_entry in 0..lut_len {
            let row = first_lut_gate - lut_entry / num_lut_entries;
            let col = lut_entry % num_lut_entries;

            let mul_target = Target::wire(row, LookupTableGate::wire_ith_multiplicity(col));

            pw.set_target(
                mul_target,
                F::from_canonical_usize(multiplicities[lut_entry]),
            )?;
        }
    }

    Ok(())
}

pub fn prove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    inputs: PartialWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let partition_witness = timed!(
        timing,
        &format!("run {} generators", prover_data.generators.len()),
        generate_partial_witness(inputs, prover_data, common_data)?
    );

    prove_with_partition_witness(prover_data, common_data, partition_witness, timing)
}

pub fn prove_with_partition_witness<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    mut partition_witness: PartitionWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let has_lookup = !common_data.luts.is_empty();
    let config = &common_data.config;
    let num_challenges = config.num_challenges;
    let quotient_degree = common_data.quotient_degree();
    let degree = common_data.degree();

    set_lookup_wires(prover_data, common_data, &mut partition_witness)?;

    let public_inputs = partition_witness.get_targets(&prover_data.public_inputs);
    let public_inputs_hash = C::InnerHasher::hash_no_pad(&public_inputs);

    let mut witness = timed!(
        timing,
        "compute full witness",
        partition_witness.full_witness()
    );

    // Only the routed columns are read again after this point (the
    // permutation argument covers wires `j < num_routed_wires`; nothing else
    // consumes the matrix), so move the non-routed columns out instead of
    // cloning them.
    let num_routed_wires = common_data.config.num_routed_wires;
    let wires_values: Vec<PolynomialValues<F>> = timed!(
        timing,
        "compute wire polynomials",
        witness
            .wire_values
            .par_iter_mut()
            .enumerate()
            .map(|(j, column)| {
                if j < num_routed_wires {
                    PolynomialValues::new(column.clone())
                } else {
                    PolynomialValues::new(core::mem::take(column))
                }
            })
            .collect()
    );

    let wires_commitment = timed!(
        timing,
        "compute wires commitment",
        PolynomialBatch::<F, C, D>::from_values(
            wires_values,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::WIRES.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    let mut challenger = Challenger::<F, C::Hasher>::new();

    // Observe the FRI config
    common_data.fri_params.observe(&mut challenger);

    // Observe the instance.
    challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
    challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);

    challenger.observe_cap::<C::Hasher>(&wires_commitment.merkle_tree.cap);

    // We need 4 values per challenge: 2 for the combos, 1 for (X-combo) in the accumulators and 1 to prove that the lookup table was computed correctly.
    // We can reuse betas and gammas for two of them.
    let num_lookup_challenges = NUM_COINS_LOOKUP * num_challenges;

    let betas = challenger.get_n_challenges(num_challenges);
    let gammas = challenger.get_n_challenges(num_challenges);
    // The quotient numerator uses `beta_i * (k_j * x)` for every routed wire
    // and quotient point. Reassociate this finite-field product once per
    // challenge and wire; the resulting coefficient is reused across all
    // quotient batches.
    let beta_k_is: Vec<F> = betas
        .iter()
        .flat_map(|&beta| common_data.k_is.iter().map(move |&k_i| beta * k_i))
        .collect();

    let deltas = if has_lookup {
        let mut delts = Vec::with_capacity(2 * num_challenges);
        let num_additional_challenges = num_lookup_challenges - 2 * num_challenges;
        let additional = challenger.get_n_challenges(num_additional_challenges);
        delts.extend(&betas);
        delts.extend(&gammas);
        delts.extend(additional);
        delts
    } else {
        vec![]
    };

    assert!(
        common_data.quotient_degree_factor < common_data.config.num_routed_wires,
        "When the number of routed wires is smaller that the degree, we should change the logic to avoid computing partial products."
    );
    let mut partial_products_and_zs = timed!(
        timing,
        "compute partial products",
        all_wires_permutation_partial_products(
            &witness,
            &betas,
            &beta_k_is,
            &gammas,
            prover_data,
            common_data,
        )
    );

    // Z is expected at the front of our batch; see `zs_range` and `partial_products_range`.
    let plonk_z_vecs: Vec<_> = partial_products_and_zs
        .iter_mut()
        .map(|partial_products_and_z| partial_products_and_z.pop().unwrap())
        .collect();
    let partial_products_len = partial_products_and_zs.iter().map(Vec::len).sum::<usize>();
    let mut zs_partial_products = Vec::with_capacity(plonk_z_vecs.len() + partial_products_len);
    zs_partial_products.extend(plonk_z_vecs);
    zs_partial_products.extend(partial_products_and_zs.into_iter().flatten());

    // All lookup polys: RE and partial SLDCs.
    let lookup_polys =
        compute_all_lookup_polys(&witness, &deltas, prover_data, common_data, has_lookup);

    // Partial products and optional lookup polynomials are the final consumers of the full witness.
    // Release its routed columns before allocating the Z commitment, quotient values, and FRI
    // buffers instead of retaining up to `num_routed_wires * degree` dead field elements.
    drop(witness);

    if has_lookup {
        zs_partial_products.extend(lookup_polys);
    }

    let partial_products_zs_and_lookup_commitment = timed!(
        timing,
        "commit to partial products, Z's and, if any, lookup polynomials",
        PolynomialBatch::from_values(
            zs_partial_products,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&partial_products_zs_and_lookup_commitment.merkle_tree.cap);

    let alphas = challenger.get_n_challenges(num_challenges);

    let quotient_polys = timed!(
        timing,
        "compute quotient polys",
        compute_quotient_polys::<F, C, D>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &betas,
            &gammas,
            &beta_k_is,
            &deltas,
            &alphas,
        )
    );

    let all_quotient_poly_chunks: Vec<PolynomialCoeffs<F>> = timed!(
        timing,
        "split up quotient polys",
        quotient_polys
            .into_par_iter()
            .flat_map(|mut quotient_poly| {
                quotient_poly.trim_to_len(quotient_degree).expect(
                    "Quotient has failed, the vanishing polynomial is not divisible by Z_H",
                );
                // Split quotient into degree-n chunks.
                quotient_poly.chunks(degree)
            })
            .collect()
    );

    let quotient_polys_commitment = timed!(
        timing,
        "commit to quotient polys",
        PolynomialBatch::<F, C, D>::from_coeffs(
            all_quotient_poly_chunks,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::QUOTIENT.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&quotient_polys_commitment.merkle_tree.cap);

    let zeta = challenger.get_extension_challenge::<D>();
    // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
    // `g * zeta`, are not in our subgroup `H`. It suffices to check `zeta` only, since
    // `(g * zeta)^n = zeta^n`, where `n` is the order of `g`.
    let g = F::Extension::primitive_root_of_unity(common_data.degree_bits());
    ensure!(
        zeta.exp_power_of_2(common_data.degree_bits()) != F::Extension::ONE,
        "Opening point is in the subgroup."
    );

    let openings = timed!(
        timing,
        "construct the opening set, including lookups",
        OpeningSet::new(
            zeta,
            g,
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &quotient_polys_commitment,
            common_data
        )
    );
    challenger.observe_openings(&openings.to_fri_openings());
    let instance = common_data.get_fri_instance(zeta);

    let opening_proof = timed!(
        timing,
        "compute opening proofs",
        PolynomialBatch::<F, C, D>::prove_openings(
            &instance,
            &[
                &prover_data.constants_sigmas_commitment,
                &wires_commitment,
                &partial_products_zs_and_lookup_commitment,
                &quotient_polys_commitment,
            ],
            &mut challenger,
            &common_data.fri_params,
            None,
            None,
            timing,
        )
    );

    let proof = Proof::<F, C, D> {
        wires_cap: wires_commitment.merkle_tree.cap,
        plonk_zs_partial_products_cap: partial_products_zs_and_lookup_commitment.merkle_tree.cap,
        quotient_polys_cap: quotient_polys_commitment.merkle_tree.cap,
        openings,
        opening_proof,
    };
    Ok(ProofWithPublicInputs::<F, C, D> {
        proof,
        public_inputs,
    })
}

/// Compute the partial products used in the `Z` polynomials.
fn all_wires_permutation_partial_products<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    betas: &[F],
    beta_k_is: &[F],
    gammas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<Vec<PolynomialValues<F>>> {
    let num_challenges = common_data.config.num_challenges;
    let num_routed_wires = common_data.config.num_routed_wires;
    debug_assert_eq!(betas.len(), num_challenges);
    debug_assert_eq!(beta_k_is.len(), num_challenges * num_routed_wires);
    (0..common_data.config.num_challenges)
        .map(|i| {
            wires_permutation_partial_products_and_zs(
                witness,
                betas[i],
                &beta_k_is[i * num_routed_wires..(i + 1) * num_routed_wires],
                gammas[i],
                prover_data,
                common_data,
            )
        })
        .collect()
}

/// Accumulate numerator and denominator products for each permutation chunk. Individual wire
/// quotients are never observed by the permutation argument, so form the observable products
/// before the one batched inversion.
fn permutation_chunk_product_pairs<F: Field>(
    terms: impl IntoIterator<Item = (F, F)>,
    chunk_size: usize,
    numerator_products: &mut [F],
    denominator_products: &mut [F],
) {
    debug_assert!(chunk_size > 1);
    assert_eq!(numerator_products.len(), denominator_products.len());

    let mut terms = terms.into_iter();
    let eight_wide_terms = numerator_products.len() * 8;
    if chunk_size == 8 && terms.size_hint() == (eight_wide_terms, Some(eight_wide_terms)) {
        for (numerator_product, denominator_product) in numerator_products
            .iter_mut()
            .zip(denominator_products.iter_mut())
        {
            let [
                (n0, d0),
                (n1, d1),
                (n2, d2),
                (n3, d3),
                (n4, d4),
                (n5, d5),
                (n6, d6),
                (n7, d7),
            ] = core::array::from_fn(|_| {
                terms
                    .next()
                    .expect("exact-size permutation iterator ended early")
            });
            *numerator_product = ((n0 * n1) * (n2 * n3)) * ((n4 * n5) * (n6 * n7));
            *denominator_product = ((d0 * d1) * (d2 * d3)) * ((d4 * d5) * (d6 * d7));
        }
        debug_assert!(terms.next().is_none());
        return;
    }

    let mut num_chunks = 0;
    for (j, (numerator, denominator)) in terms.enumerate() {
        if j % chunk_size == 0 {
            assert!(
                num_chunks < numerator_products.len(),
                "chunk-product scratch space is too small"
            );
            numerator_products[num_chunks] = numerator;
            denominator_products[num_chunks] = denominator;
            num_chunks += 1;
        } else {
            numerator_products[num_chunks - 1] *= numerator;
            denominator_products[num_chunks - 1] *= denominator;
        }
    }
    assert!(num_chunks != 0, "permutation needs at least one routed wire");
    assert_eq!(num_chunks, numerator_products.len());
}

/// Compute the partial products used in the `Z` polynomial.
/// Returns the polynomials interpolating `partial_products(f / g)`
/// where `f, g` are the products in the definition of `Z`: `Z(g^i) = f / g`.
fn wires_permutation_partial_products_and_zs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    beta: F,
    beta_k_is: &[F],
    gamma: F,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.quotient_degree_factor;
    let subgroup = &prover_data.subgroup;
    let num_prods = common_data.num_partial_products;
    debug_assert_eq!(beta_k_is.len(), common_data.config.num_routed_wires);
    let num_routed_wires = common_data.config.num_routed_wires;
    // Only each chunk's product of wire ratios is observable. Form its numerator and denominator
    // products first, reducing the batched inversion width from routed wires to chunks, then write
    // the resulting ratios directly into the retained flat point/chunk layout.
    const INV_BATCH: usize = 128;
    let num_chunks = num_prods + 1;
    debug_assert_eq!(num_chunks, num_routed_wires.div_ceil(degree));
    let mut all_quotient_chunk_products = vec![F::ZERO; subgroup.len() * num_chunks];
    all_quotient_chunk_products
        .par_chunks_mut(INV_BATCH * num_chunks)
        .zip(subgroup.par_chunks(INV_BATCH))
        .enumerate()
        .for_each_init(
            || {
                (
                    vec![F::ZERO; num_chunks * INV_BATCH],
                    vec![F::ZERO; num_chunks * INV_BATCH],
                    Vec::with_capacity(num_chunks * INV_BATCH),
                )
            },
            |(numerator_products, denominator_products, inverse_products),
             (chunk_idx, (products, xs))| {
                let base = chunk_idx * INV_BATCH;
                let product_count = xs.len() * num_chunks;
                debug_assert_eq!(products.len(), product_count);
                for (t, &x) in xs.iter().enumerate() {
                    let i = base + t;
                    let s_sigmas = &prover_data.sigmas[i];
                    let terms = (0..num_routed_wires).map(|j| {
                        let wire_value = witness.get_wire(i, j);
                        let numerator = wire_value + beta_k_is[j] * x + gamma;
                        let denominator = wire_value + beta * s_sigmas[j] + gamma;
                        (numerator, denominator)
                    });
                    let start = t * num_chunks;
                    let end = start + num_chunks;
                    permutation_chunk_product_pairs(
                        terms,
                        degree,
                        &mut numerator_products[start..end],
                        &mut denominator_products[start..end],
                    );
                }

                F::batch_multiplicative_inverse_into(
                    &denominator_products[..product_count],
                    inverse_products,
                );
                for ((product, &denominator_inverse), &numerator_product) in products
                    .iter_mut()
                    .zip(inverse_products.iter())
                    .zip(numerator_products.iter())
                {
                    *product = numerator_product * denominator_inverse;
                }
            },
        );

    // Accumulate the sequential Z chain directly into the column-major output
    // polynomials, deleting the per-point row Vec, the row-major intermediate,
    // and the whole-phase transpose. Values and their order are identical: for
    // each point, column k receives the k-th running product, and the last
    // column receives the previous Z(x) exactly as the swap-based version did.
    let n_points = subgroup.len();
    let mut columns: Vec<Vec<F>> = (0..num_chunks)
        .map(|_| Vec::with_capacity(n_points))
        .collect();
    let mut z_x = F::ONE;
    for quotient_chunk_products in all_quotient_chunk_products.chunks_exact(num_chunks) {
        let mut acc = z_x;
        for (k, &quotient_chunk_product) in quotient_chunk_products.iter().enumerate() {
            acc *= quotient_chunk_product;
            if k == num_prods {
                // The last term is Z(gx), but we store Z(x) in its place,
                // otherwise Z would end up shifted.
                columns[k].push(z_x);
                z_x = acc;
            } else {
                columns[k].push(acc);
            }
        }
    }

    columns.into_iter().map(PolynomialValues::new).collect()
}

#[cfg(test)]
mod permutation_chunk_product_tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::PrimeField64;

    #[test]
    fn chunk_products_before_inversion_match_individual_wire_ratios() {
        type F = GoldilocksField;
        for canonical in [true, false] {
            for width in [1usize, 7, 8, 9, 79, 80, 81] {
                for chunk_size in [2usize, 3, 8, 128] {
                    let make = |seed: u64, i: usize| {
                        let value = seed.wrapping_add((i as u64 + 1) * 0x9e37_79b9);
                        if canonical {
                            F::from_canonical_u64(value)
                        } else {
                            F::from_noncanonical_u64(u64::MAX - value)
                        }
                    };
                    let numerators = (0..width).map(|i| make(0x1111, i)).collect::<Vec<_>>();
                    let denominators = (0..width).map(|i| make(0x2222, i)).collect::<Vec<_>>();

                    let mut wire_ratios = F::batch_multiplicative_inverse(&denominators);
                    for (ratio, &numerator) in wire_ratios.iter_mut().zip(&numerators) {
                        *ratio *= numerator;
                    }
                    let expected = wire_ratios
                        .chunks(chunk_size)
                        .map(|chunk| chunk.iter().copied().product::<F>())
                        .collect::<Vec<_>>();

                    let num_chunks = width.div_ceil(chunk_size);
                    let mut numerator_products = vec![F::ZERO; num_chunks];
                    let mut denominator_products = vec![F::ZERO; num_chunks];
                    permutation_chunk_product_pairs(
                        numerators.iter().copied().zip(denominators.iter().copied()),
                        chunk_size,
                        &mut numerator_products,
                        &mut denominator_products,
                    );
                    let mut actual = F::batch_multiplicative_inverse(&denominator_products);
                    for (ratio, numerator) in actual.iter_mut().zip(numerator_products) {
                        *ratio *= numerator;
                    }

                    assert_eq!(actual, expected, "width={width}, chunk={chunk_size}");
                    assert_eq!(
                        actual.iter().map(|x| x.to_noncanonical_u64()).collect::<Vec<_>>(),
                        expected.iter().map(|x| x.to_noncanonical_u64()).collect::<Vec<_>>(),
                        "raw width={width}, chunk={chunk_size}"
                    );
                }
            }
        }
    }


}

/// Computes lookup polynomials for a given challenge.
/// The polynomials hold the value of RE, Sum and Ldc of the Tip5 paper (<https://eprint.iacr.org/2023/107.pdf>). To reduce their
/// numbers, we batch multiple slots in a single polynomial. Since RE only involves degree one constraints, we can batch
/// all the slots of a row. For Sum and Ldc, batching increases the constraint degree, so we bound the number of
/// partial polynomials according to `max_quotient_degree_factor`.
/// As another optimization, Sum and LDC polynomials are shared (in so called partial SLDC polynomials), and the last value
/// of the last partial polynomial is Sum(end) - LDC(end). If the lookup argument is valid, then it must be equal to 0.
fn compute_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F; 4],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.degree();
    let num_lu_slots = LookupGate::num_slots(&common_data.config);
    let max_lookup_degree = common_data.config.max_quotient_degree_factor - 1;
    let num_partial_lookups = num_lu_slots.div_ceil(max_lookup_degree);
    let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
    let max_lookup_table_degree = num_lut_slots.div_ceil(num_partial_lookups);

    // First poly is RE, the rest are partial SLDCs.
    let mut final_poly_vecs = Vec::with_capacity(num_partial_lookups + 1);
    for _ in 0..num_partial_lookups + 1 {
        final_poly_vecs.push(PolynomialValues::<F>::new(vec![F::ZERO; degree]));
    }

    for LookupWire {
        last_lu_gate: last_lu_row,
        last_lut_gate: last_lut_row,
        first_lut_gate: first_lut_row,
    } in prover_data.lookup_rows.clone()
    {
        // Set values for partial Sums and RE.
        for row in (last_lut_row..(first_lut_row + 1)).rev() {
            // Get combos for Sum.
            let looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeA as usize] * looked_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looked_combos[s])
                .collect();
            // Get 1/(alpha - combo).
            let looked_combo_inverses = F::batch_multiplicative_inverse(&minus_looked_combos);

            // Get lookup combos, used to check the well formation of the LUT.
            let lookup_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s));
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s));

                    looked_inp + deltas[LookupChallenges::ChallengeB as usize] * looked_out
                })
                .collect();

            // Compute next row's first value of RE.
            // If `row == first_lut_row`, then `final_poly_vecs[0].values[row + 1] == 0`.
            let mut new_re = final_poly_vecs[0].values[row + 1];
            for elt in &lookup_combos {
                new_re = new_re * deltas[LookupChallenges::ChallengeDelta as usize] + *elt
            }
            final_poly_vecs[0].values[row] = new_re;

            for slot in 0..num_partial_lookups {
                let prev = if slot != 0 {
                    final_poly_vecs[slot].values[row]
                } else {
                    // If `row == first_lut_row`, then `final_poly_vecs[num_partial_lookups].values[row + 1] == 0`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                };
                let sum = (slot * max_lookup_table_degree
                    ..min((slot + 1) * max_lookup_table_degree, num_lut_slots))
                    .fold(prev, |acc, s| {
                        acc + witness.get_wire(row, LookupTableGate::wire_ith_multiplicity(s))
                            * looked_combo_inverses[s]
                    });
                final_poly_vecs[slot + 1].values[row] = sum;
            }
        }

        // Set values for partial LDCs.
        for row in (last_lu_row..last_lut_row).rev() {
            // Get looking combos.
            let looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| {
                    let looking_in = witness.get_wire(row, LookupGate::wire_ith_looking_inp(s));
                    let looking_out = witness.get_wire(row, LookupGate::wire_ith_looking_out(s));

                    looking_in + deltas[LookupChallenges::ChallengeA as usize] * looking_out
                })
                .collect();
            // Get (alpha - combo).
            let minus_looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looking_combos[s])
                .collect();
            // Get 1 / (alpha - combo).
            let looking_combo_inverses = F::batch_multiplicative_inverse(&minus_looking_combos);

            for slot in 0..num_partial_lookups {
                let prev = if slot == 0 {
                    // Valid at _any_ row, even `first_lu_row`.
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                } else {
                    final_poly_vecs[slot].values[row]
                };
                let sum = (slot * max_lookup_degree
                    ..min((slot + 1) * max_lookup_degree, num_lu_slots))
                    .fold(F::ZERO, |acc, s| acc + looking_combo_inverses[s]);
                final_poly_vecs[slot + 1].values[row] = prev - sum;
            }
        }
    }

    final_poly_vecs
}

/// Computes lookup polynomials for all challenges.
fn compute_all_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    lookup: bool,
) -> Vec<PolynomialValues<F>> {
    if lookup {
        let polys: Vec<Vec<PolynomialValues<F>>> = (0..common_data.config.num_challenges)
            .map(|c| {
                compute_lookup_polys(
                    witness,
                    &deltas[c * NUM_COINS_LOOKUP..(c + 1) * NUM_COINS_LOOKUP]
                        .try_into()
                        .unwrap(),
                    prover_data,
                    common_data,
                )
            })
            .collect();
        polys.into_iter().flatten().collect()
    } else {
        vec![]
    }
}

const BATCH_SIZE: usize = 32;

fn compute_quotient_polys<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash,
    wires_commitment: &'a PolynomialBatch<F, C, D>,
    zs_partial_products_and_lookup_commitment: &'a PolynomialBatch<F, C, D>,
    betas: &[F],
    gammas: &[F],
    beta_k_is: &[F],
    deltas: &[F],
    alphas: &[F],
) -> Vec<PolynomialCoeffs<F>> {
    let num_challenges = common_data.config.num_challenges;

    let has_lookup = common_data.num_lookup_polys != 0;

    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    assert!(
        quotient_degree_bits <= common_data.config.fri_config.rate_bits,
        "Having constraints of degree higher than the rate is not supported yet. \
        If we need this in the future, we can precompute the larger LDE before computing the `PolynomialBatch`s."
    );

    // We reuse the LDE computed in `PolynomialBatch` and extract every `step` points to get
    // an LDE matching `max_filtered_constraint_degree`.
    let step = 1 << (common_data.config.fri_config.rate_bits - quotient_degree_bits);
    // When opening the `Z`s polys at the "next" point in Plonk, need to look at the point `next_step`
    // steps away since we work on an LDE of degree `max_filtered_constraint_degree`.
    let next_step = 1 << quotient_degree_bits;

    let log_lde = common_data.degree_bits() + quotient_degree_bits;
    let lde_size = 1 << log_lde;
    let half_lde_size = lde_size / 2;
    // The retained FFT table's row for this domain stores its first half in the same order as
    // `two_adic_subgroup`; the second half is its field negation. Fresh circuit data always has
    // this row. Keep the allocation fallback for older/deserialized prover data without it.
    let quotient_half_roots = prover_data
        .fft_root_table
        .as_ref()
        .and_then(|table| log_lde.checked_sub(1).and_then(|row| table.get(row)))
        .filter(|row| half_lde_size != 0 && row.len() >= half_lde_size);
    let quotient_points_fallback =
        quotient_half_roots.is_none().then(|| F::two_adic_subgroup(log_lde));

    let z_h_on_coset = ZeroPolyOnCoset::new(common_data.degree_bits(), quotient_degree_bits);

    // Precompute the lookup table evals on the challenges in delta
    // These values are used to produce the final RE constraints for each lut,
    // and are the same each time in check_lookup_constraints_batched.
    // lut_poly_evals[i][j] gives the eval for the i'th challenge and the j'th lookup table
    let lut_re_poly_evals: Vec<Vec<F>> = if has_lookup {
        let num_lut_slots = LookupTableGate::num_slots(&common_data.config);
        (0..num_challenges)
            .map(move |i| {
                let cur_deltas = &deltas[NUM_COINS_LOOKUP * i..NUM_COINS_LOOKUP * (i + 1)];
                let cur_challenge_delta = cur_deltas[LookupChallenges::ChallengeDelta as usize];

                (LookupSelectors::StartEnd as usize..common_data.num_lookup_selectors)
                    .map(|r| {
                        let lut_row_number = common_data.luts
                            [r - LookupSelectors::StartEnd as usize]
                            .len()
                            .div_ceil(num_lut_slots);

                        get_lut_poly(
                            common_data,
                            r - LookupSelectors::StartEnd as usize,
                            cur_deltas,
                            num_lut_slots * lut_row_number,
                        )
                        .eval(cur_challenge_delta)
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };

    let lut_re_poly_evals_refs: Vec<&[F]> =
        lut_re_poly_evals.iter().map(|v| v.as_slice()).collect();

    let num_batches = lde_size.div_ceil(BATCH_SIZE);

    struct QuotientScratch<F: RichField> {
        indices: Vec<usize>,
        indices_next: Vec<usize>,
        shifted_xs: Vec<F>,
        local_constants: Vec<F>,
        local_wires: Vec<F>,
        s_sigmas_flat: Vec<F>,
        zs_local_flat: Vec<F>,
        zs_next_flat: Vec<F>,
        vanishing: VanishingScratch<F>,
    }

    let num_wires = common_data.config.num_wires;
    let zs_row_width = zs_partial_products_and_lookup_commitment.lde_row_width();
    let num_routed_wires = common_data.config.num_routed_wires;

    let mut quotient_values = vec![F::ZERO; lde_size * num_challenges];
    quotient_values
        .par_chunks_mut(BATCH_SIZE * num_challenges)
        .enumerate()
        .for_each_init(
            || QuotientScratch::<F> {
                indices: Vec::with_capacity(BATCH_SIZE),
                indices_next: Vec::with_capacity(BATCH_SIZE),
                shifted_xs: Vec::with_capacity(BATCH_SIZE),
                local_constants: Vec::new(),
                local_wires: Vec::new(),
                s_sigmas_flat: Vec::new(),
                zs_local_flat: Vec::new(),
                zs_next_flat: Vec::new(),
                vanishing: VanishingScratch::default(),
            },
            |scratch, (batch_i, quotient_values_batch)| {
                // Each batch must be the same size, except the last one, which may be smaller.
                let n = quotient_values_batch.len() / num_challenges;
                debug_assert!(
                    n == BATCH_SIZE || (batch_i == num_batches - 1 && n <= BATCH_SIZE)
                );

                scratch.indices.clear();
                scratch
                    .indices
                    .extend(BATCH_SIZE * batch_i..BATCH_SIZE * batch_i + n);
                scratch.indices_next.clear();
                scratch
                    .indices_next
                    .extend(scratch.indices.iter().map(|&i| (i + next_step) % lde_size));

                scratch.shifted_xs.clear();
                scratch.shifted_xs.extend(scratch.indices.iter().map(|&i| {
                    let x = if let Some(half_roots) = quotient_half_roots {
                        if i < half_lde_size {
                            half_roots[i]
                        } else {
                            -half_roots[i - half_lde_size]
                        }
                    } else {
                        quotient_points_fallback
                            .as_ref()
                            .expect("quotient domain fallback must exist")[i]
                    };
                    F::coset_shift() * x
                }));

                prover_data.constants_sigmas_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    common_data.constants_range(),
                    BatchLayout::PolyMajor,
                    &mut scratch.local_constants,
                );
                prover_data.constants_sigmas_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    common_data.sigmas_range(),
                    BatchLayout::PointMajor,
                    &mut scratch.s_sigmas_flat,
                );
                wires_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    0..num_wires,
                    BatchLayout::PolyMajor,
                    &mut scratch.local_wires,
                );
                zs_partial_products_and_lookup_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    0..zs_row_width,
                    BatchLayout::PointMajor,
                    &mut scratch.zs_local_flat,
                );
                zs_partial_products_and_lookup_commitment.fill_lde_batch(
                    &scratch.indices_next,
                    step,
                    0..zs_row_width,
                    BatchLayout::PointMajor,
                    &mut scratch.zs_next_flat,
                );

                let indices_batch = &scratch.indices;
                let local_zs_batch: Vec<&[F]> = (0..n)
                    .map(|k| &scratch.zs_local_flat[k * zs_row_width..][common_data.zs_range()])
                    .collect();
                let next_zs_batch: Vec<&[F]> = (0..n)
                    .map(|k| &scratch.zs_next_flat[k * zs_row_width..][common_data.zs_range()])
                    .collect();
                let partial_products_batch: Vec<&[F]> = (0..n)
                    .map(|k| {
                        &scratch.zs_local_flat[k * zs_row_width..]
                            [common_data.partial_products_range()]
                    })
                    .collect();
                let s_sigmas_batch: Vec<&[F]> = (0..n)
                    .map(|k| {
                        &scratch.s_sigmas_flat[k * num_routed_wires..(k + 1) * num_routed_wires]
                    })
                    .collect();
                let (local_lookup_batch, next_lookup_batch): (Vec<&[F]>, Vec<&[F]>) = if has_lookup
                {
                    (
                        (0..n)
                            .map(|k| {
                                &scratch.zs_local_flat[k * zs_row_width..]
                                    [common_data.lookup_range()]
                            })
                            .collect(),
                        (0..n)
                            .map(|k| {
                                &scratch.zs_next_flat[k * zs_row_width..]
                                    [common_data.lookup_range()]
                            })
                            .collect(),
                    )
                } else {
                    (Vec::new(), Vec::new())
                };

                let vars_batch = EvaluationVarsBaseBatch::new(
                    n,
                    &scratch.local_constants,
                    &scratch.local_wires,
                    public_inputs_hash,
                );

                let quotient_values_batch = &mut quotient_values_batch[..n * num_challenges];
                eval_vanishing_poly_base_batch::<F, D>(
                    common_data,
                    indices_batch,
                    &scratch.shifted_xs,
                    vars_batch,
                    &local_zs_batch,
                    &next_zs_batch,
                    &local_lookup_batch,
                    &next_lookup_batch,
                    &partial_products_batch,
                    &s_sigmas_batch,
                    betas,
                    gammas,
                    beta_k_is,
                    deltas,
                    alphas,
                    &z_h_on_coset,
                    &lut_re_poly_evals_refs,
                    &mut scratch.vanishing,
                    quotient_values_batch,
                );

                for (&i, quotient_values) in indices_batch
                    .iter()
                    .zip(quotient_values_batch.chunks_exact_mut(num_challenges))
                {
                    let denominator_inv = z_h_on_coset.eval_inverse(i);
                    quotient_values
                        .iter_mut()
                        .for_each(|v| *v *= denominator_inv);
                }
            },
        );

    debug_assert_eq!(quotient_values.len(), lde_size * num_challenges);
    (0..num_challenges)
        .into_par_iter()
        .map(|challenge| {
            PolynomialValues::new(
                quotient_values
                    .chunks_exact(num_challenges)
                    .map(|point_values| point_values[challenge])
                    .collect(),
            )
        })
        .map(|values| values.coset_ifft(F::coset_shift()))
        .collect()
}
