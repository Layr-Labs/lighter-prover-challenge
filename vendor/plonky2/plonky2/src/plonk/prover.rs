//! plonky2 prover implementation.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
use core::cmp::min;

use anyhow::{ensure, Result};
use hashbrown::HashMap;
use plonky2_maybe_rayon::*;

use super::circuit_builder::{LookupChallenges, LookupWire};
use crate::field::extension::Extendable;
use crate::field::fft::ifft_borrowed;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::{BatchLayout, PolynomialBatch};
use crate::gates::lookup::LookupGate;
use crate::gates::lookup_table::LookupTableGate;
use crate::gates::poseidon2::Poseidon2Gate;
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
    eval_vanishing_poly_base_batch, get_lut_poly, PermutationBatch, VanishingScratch,
};
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{log2_ceil};

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
    // consumes the matrix). Non-routed columns are moved out and IFFT'd in
    // place; routed columns are IFFT'd from the borrowed witness column
    // (`ifft_borrowed` fuses the former clone with the FFT's initial
    // bit-reversal gather), so no witness column is copied.
    let num_routed_wires = common_data.config.num_routed_wires;
    let wires_coeffs: Vec<PolynomialCoeffs<F>> = timed!(
        timing,
        "compute wire polynomials (IFFT)",
        witness
            .wire_values
            .par_iter_mut()
            .enumerate()
            .map(|(j, column)| {
                if j < num_routed_wires {
                    ifft_borrowed(column)
                } else {
                    PolynomialValues::new(core::mem::take(column)).ifft()
                }
            })
            .collect()
    );

    let wires_commitment = timed!(
        timing,
        "compute wires commitment",
        PolynomialBatch::<F, C, D>::from_coeffs(
            wires_coeffs,
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

    // The permutation argument and lookup polys were the last readers of the
    // witness matrix (non-routed columns were already moved out into
    // `wires_values`). Free the ~80 routed columns now, before the ZS
    // commitment, quotient evaluation, and FRI phases raise memory pressure.
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
            // Layout seam: flat column-major permutation data when the
            // circuit has no lookups; the per-point path otherwise.
            !has_lookup,
            true,
        )
    );

    // Opt-in production validation for the Metal Poseidon2 quotient seam.
    // Reuse the optimized column-major CPU path and only disable the GPU gate,
    // so any mismatch is attributable to the offloaded contribution itself.
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    if !has_lookup && gpu_poseidon_quotient_differential_enabled() {
        let reference = compute_quotient_polys::<F, C, D>(
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
            true,
            false,
        );
        assert_eq!(quotient_polys.len(), reference.len());
        for (p, (actual, expected)) in quotient_polys.iter().zip(reference.iter()).enumerate() {
            assert_eq!(actual.coeffs.len(), expected.coeffs.len());
            for (i, (actual, expected)) in
                actual.coeffs.iter().zip(expected.coeffs.iter()).enumerate()
            {
                assert_eq!(
                    actual.to_canonical_u64(),
                    expected.to_canonical_u64(),
                    "GPU Poseidon2 quotient divergence: poly {p}, coeff {i}"
                );
            }
        }
        let stats = gpu_poseidon_quotient_stats();
        log::info!("Metal Poseidon2 quotient differential passed: {stats:?}");
        if gpu_poseidon_quotient_diagnostics_enabled() {
            eprintln!("[gpu-poseidon-quotient] differential passed {stats:?}");
        }
    }

    // Differential gate for the layout seam: recompute the quotient through
    // the per-point reference path on the same witness, commitments and
    // challenges, and require value-identical polynomials.
    #[cfg(test)]
    if !has_lookup && COMPARE_QUOTIENT_LAYOUTS.load(core::sync::atomic::Ordering::Relaxed) {
        let reference = compute_quotient_polys::<F, C, D>(
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
            false,
            false,
        );
        assert_eq!(quotient_polys.len(), reference.len());
        for (p, (a, b)) in quotient_polys.iter().zip(reference.iter()).enumerate() {
            assert_eq!(a.coeffs.len(), b.coeffs.len());
            for (i, (x, y)) in a.coeffs.iter().zip(b.coeffs.iter()).enumerate() {
                assert_eq!(
                    x.to_canonical_u64(),
                    y.to_canonical_u64(),
                    "quotient layout divergence: poly {p}, coeff {i}"
                );
            }
        }
    }

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

#[inline]
fn divide_chunk_products<F: Field>(
    numerator_products: &mut [F],
    denominator_products: &[F],
    inverse_scratch: &mut Vec<F>,
) {
    debug_assert_eq!(numerator_products.len(), denominator_products.len());
    F::batch_multiplicative_inverse_into(denominator_products, inverse_scratch);
    for (product, &inverse) in numerator_products.iter_mut().zip(inverse_scratch.iter()) {
        *product *= inverse;
    }
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
    let num_chunks = num_prods + 1;
    debug_assert_eq!(num_chunks, num_routed_wires.div_ceil(degree));

    // The permutation argument only consumes one numerator/denominator ratio per quotient-degree
    // chunk. Form those products before Montgomery inversion, shrinking each inversion batch by
    // `degree` and reading every witness wire only once.
    const INV_BATCH: usize = 128;
    // Every slot of this buffer is assigned below before anything reads it —
    // the inner loop writes `quotient_products[t * num_chunks + chunk]` for
    // every `t` in the batch and every `chunk`, which covers each sub-slice
    // exactly, and `divide_chunk_products` only multiplies those cells in
    // place afterwards. So zero-filling it first is dead work: at
    // `num_chunks = 10` and a 2^16 subgroup that is 5.2 MiB of serial stores
    // per challenge, ~10.5 MiB per proof, on the per-proof spine between
    // witness generation and the Zs/partial-products commitment.
    let product_count = subgroup.len() * num_chunks;
    let mut all_quotient_chunk_products: Vec<F> = Vec::with_capacity(product_count);
    let product_slots =
        crate::hash::merkle_tree::capacity_up_to_mut(&mut all_quotient_chunk_products, product_count);
    product_slots
        .par_chunks_mut(INV_BATCH * num_chunks)
        .zip(subgroup.par_chunks(INV_BATCH))
        .enumerate()
        .for_each_init(
            || {
                (
                    Vec::with_capacity(num_chunks * INV_BATCH),
                    Vec::with_capacity(num_chunks * INV_BATCH),
                )
            },
            |scratch, (chunk_idx, (quotient_products, xs))| {
                let base = chunk_idx * INV_BATCH;
                let (denominator_products, denominator_inverses) = scratch;
                denominator_products.clear();
                for (t, &x) in xs.iter().enumerate() {
                    let i = base + t;
                    let s_sigmas = &prover_data.sigmas[i];
                    for chunk in 0..num_chunks {
                        let start = chunk * degree;
                        let end = min(start + degree, num_routed_wires);
                        let mut numerator_product = F::ONE;
                        let mut denominator_product = F::ONE;
                        for j in start..end {
                            let wire_value = witness.get_wire(i, j);
                            numerator_product *= wire_value + beta_k_is[j] * x + gamma;
                            denominator_product *= wire_value + beta * s_sigmas[j] + gamma;
                        }
                        quotient_products[t * num_chunks + chunk].write(numerator_product);
                        denominator_products.push(denominator_product);
                    }
                }
                // SAFETY: the loop above wrote every slot of this sub-slice —
                // `t` covers `0..xs.len()` and `chunk` covers `0..num_chunks`,
                // and the sub-slice length is exactly `xs.len() * num_chunks`
                // (the `zip` pairs each chunk with its own `xs`, so a short
                // final chunk is still covered exactly).
                let quotient_products = unsafe {
                    &mut *(quotient_products as *mut [core::mem::MaybeUninit<F>] as *mut [F])
                };
                divide_chunk_products(
                    quotient_products,
                    denominator_products,
                    denominator_inverses,
                );
            },
        );

    // SAFETY: the parallel pass above wrote and then divided every one of the
    // `product_count` slots; `par_chunks_mut` partitions the buffer exactly, so
    // none is left uninitialized.
    unsafe { all_quotient_chunk_products.set_len(product_count) };

    // Accumulate the sequential Z chain directly into the column-major output
    // polynomials, deleting the per-point row Vec, the row-major intermediate,
    // and the whole-phase transpose. Values and their order are identical: for
    // each point, column k receives the k-th running product, and the last
    // column receives the previous Z(x) exactly as the swap-based version did.
    let n_points = subgroup.len();
    let mut columns: Vec<Vec<F>> = (0..num_prods + 1)
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

/// Process-wide counters for the narrow Metal Poseidon2 quotient path. A
/// successful `started` count proves all of the production guards held: no
/// lookups, two challenges, the 135-wire/123-constraint Poseidon2 gate, and
/// shared Metal-backed wire and constant columns.
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuPoseidonQuotientStats {
    pub attempts: usize,
    pub started: usize,
    pub completed: usize,
    pub fallbacks: usize,
    pub range_attempts: usize,
    pub range_started: usize,
    pub range_completed: usize,
    pub range_fallbacks: usize,
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
static GPU_POSEIDON_QUOTIENT_ATTEMPTS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
static GPU_POSEIDON_QUOTIENT_STARTED: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
static GPU_POSEIDON_QUOTIENT_COMPLETED: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
static GPU_POSEIDON_QUOTIENT_FALLBACKS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
static GPU_RANGE_QUOTIENT_ATTEMPTS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
static GPU_RANGE_QUOTIENT_STARTED: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
static GPU_RANGE_QUOTIENT_COMPLETED: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
static GPU_RANGE_QUOTIENT_FALLBACKS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub fn gpu_poseidon_quotient_stats() -> GpuPoseidonQuotientStats {
    use core::sync::atomic::Ordering;

    GpuPoseidonQuotientStats {
        attempts: GPU_POSEIDON_QUOTIENT_ATTEMPTS.load(Ordering::Relaxed),
        started: GPU_POSEIDON_QUOTIENT_STARTED.load(Ordering::Relaxed),
        completed: GPU_POSEIDON_QUOTIENT_COMPLETED.load(Ordering::Relaxed),
        fallbacks: GPU_POSEIDON_QUOTIENT_FALLBACKS.load(Ordering::Relaxed),
        range_attempts: GPU_RANGE_QUOTIENT_ATTEMPTS.load(Ordering::Relaxed),
        range_started: GPU_RANGE_QUOTIENT_STARTED.load(Ordering::Relaxed),
        range_completed: GPU_RANGE_QUOTIENT_COMPLETED.load(Ordering::Relaxed),
        range_fallbacks: GPU_RANGE_QUOTIENT_FALLBACKS.load(Ordering::Relaxed),
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
fn gpu_poseidon_quotient_diagnostics_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        [
            "PLONKY2_GPU_POSEIDON_DIAGNOSTICS",
            "PLONKY2_GPU_RANGE_DIAGNOSTICS",
        ]
        .iter()
        .any(|name| {
            std::env::var_os(name)
                .map(|value| value != "0")
                .unwrap_or(false)
        })
    })
}

/// A deliberately expensive, opt-in production differential: when enabled,
/// every no-lookup proof recomputes the quotient with the GPU gate disabled
/// and compares canonical coefficients. This is intended for validation runs,
/// never normal benchmarking.
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
fn gpu_poseidon_quotient_differential_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let environment_enabled = *ENABLED.get_or_init(|| {
        [
            "PLONKY2_GPU_POSEIDON_DIFFERENTIAL",
            "PLONKY2_GPU_RANGE_DIFFERENTIAL",
        ]
        .iter()
        .any(|name| {
            std::env::var_os(name)
                .map(|value| value != "0")
                .unwrap_or(false)
        })
    });
    #[cfg(test)]
    {
        environment_enabled || COMPARE_GPU_QUOTIENT.load(core::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        environment_enabled
    }
}

/// Test-only switch for an in-process full GPU/CPU quotient differential.
#[cfg(all(test, feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) static COMPARE_GPU_QUOTIENT: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Test-only switch: when set, `compute_quotient_polys` evaluates the quotient
/// values twice — once through the default column-major (`PolyMajor`)
/// permutation path and once through the per-point (`PointMajor`) reference
/// path — over the same witness, commitments and challenges, and asserts the
/// two are value-identical. (Cross-run proof-byte comparison is not a usable
/// oracle in this fork: unused wire slots carry nondeterministic padding, so
/// two proofs of the same witness legitimately differ byte-wise.)
#[cfg(test)]
pub(crate) static COMPARE_QUOTIENT_LAYOUTS: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Diagnostics-only gate census: prints, once per distinct circuit shape, the
/// full gate list with the wire span / constraint count / degree that decide
/// both the CPU constraint cost and the `cpu_num_wires` gather floor. Gated on
/// the same env switch as the rest of the GPU diagnostics, so it is inert in
/// the ranked sandbox (which clears the environment).
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
fn gate_census_once<F: RichField + Extendable<D>, const D: usize>(
    common_data: &CommonCircuitData<F, D>,
    excluded_gate_indices: &[usize],
    cpu_num_wires: usize,
) {
    use std::collections::HashSet;
    use std::fmt::Write as _;
    use std::sync::Mutex;

    static SEEN: std::sync::OnceLock<Mutex<HashSet<String>>> = std::sync::OnceLock::new();

    let mut report = String::new();
    let _ = writeln!(
        report,
        "[gate-census] degree_bits={} gates={} selectors={} num_gate_constraints={} \
         gather={cpu_num_wires}/{} excluded={excluded_gate_indices:?}",
        common_data.degree_bits(),
        common_data.gates.len(),
        common_data.selectors_info.num_selectors(),
        common_data.num_gate_constraints,
        common_data.config.num_wires,
    );
    for (index, gate) in common_data.gates.iter().enumerate() {
        let _ = writeln!(
            report,
            "[gate-census]   {index:>2} wires={:>3} constraints={:>3} degree={} selector={} \
             off={} id={}",
            gate.0.num_wires(),
            gate.0.num_constraints(),
            gate.0.degree(),
            common_data.selectors_info.selector_indices[index],
            excluded_gate_indices.contains(&index) as u8,
            gate.0.id(),
        );
    }
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut seen = seen.lock().unwrap();
    if seen.insert(report.clone()) {
        eprint!("{report}");
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
fn start_gpu_poseidon_gate_quotient<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    wires_commitment: &PolynomialBatch<F, C, D>,
    quotient_rows: usize,
    step: usize,
    alphas: &[F],
) -> Option<(
    usize,
    crate::hash::poseidon2::metal::PoseidonGateQuotientJob<F>,
)> {
    use core::sync::atomic::Ordering;

    GPU_POSEIDON_QUOTIENT_ATTEMPTS.fetch_add(1, Ordering::Relaxed);

    // Keep the first backend deliberately narrow. Lookup constraints change
    // the alpha prefix, and production ranked circuits use exactly two
    // quotient challenges.
    if common_data.num_lookup_polys != 0 || common_data.config.num_challenges != 2 {
        return None;
    }
    let gate_index = common_data
        .gates
        .iter()
        .position(|gate| gate.0.as_any().is::<Poseidon2Gate<F, D>>())?;
    let gate = &common_data.gates[gate_index];
    if gate.0.num_wires() != 135 || gate.0.num_constraints() != 123 {
        return None;
    }
    let selector_index = common_data.selectors_info.selector_indices[gate_index];
    let group = common_data.selectors_info.groups[selector_index].clone();
    let wires = wires_commitment.merkle_tree.shared_columns()?;
    let constants = prover_data
        .constants_sigmas_commitment
        .merkle_tree
        .shared_columns()?;
    // z_1 contributes one term per challenge and the partial-product path
    // contributes `num_partial_products + 1`; all precede gate constraints.
    let alpha_offset = common_data.config.num_challenges * (common_data.num_partial_products + 2);
    let job = crate::hash::poseidon2::metal::start_poseidon2_gate_quotient(
        wires,
        constants,
        quotient_rows,
        step,
        selector_index,
        gate_index,
        group,
        common_data.selectors_info.num_selectors() > 1,
        alphas,
        alpha_offset,
    )?;
    let started = GPU_POSEIDON_QUOTIENT_STARTED.fetch_add(1, Ordering::Relaxed) + 1;
    log::info!(
        "Metal Poseidon2 gate quotient active: started={started}, gate={gate_index}, \
         selector={selector_index}, rows={quotient_rows}, step={step}, challenges={}, \
         lookups={}, shared_columns=true",
        common_data.config.num_challenges,
        common_data.num_lookup_polys,
    );
    if gpu_poseidon_quotient_diagnostics_enabled() {
        eprintln!(
            "[gpu-poseidon-quotient] active started={started} gate={gate_index} \
             selector={selector_index} rows={quotient_rows} step={step} challenges={} \
             lookups={} shared_columns=true",
            common_data.config.num_challenges, common_data.num_lookup_polys,
        );
    }
    Some((gate_index, job))
}

/// Base-4 result limbs for a `base_bits`-wide U32-family gate. Widths are
/// restricted to the shapes the shader is differentially tested against; any
/// other width leaves the gate on the CPU.
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
fn supported_quotient_result_limbs(base_bits: usize) -> Option<usize> {
    match base_bits {
        16 | 32 | 48 => Some(base_bits / 2),
        _ => None,
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
fn start_gpu_range_check_gate_quotient<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    wires_commitment: &PolynomialBatch<F, C, D>,
    quotient_rows: usize,
    step: usize,
    alphas: &[F],
) -> Option<(
    Vec<usize>,
    crate::hash::poseidon2::metal::RangeCheckGateQuotientJob<F>,
)> {
    use core::sync::atomic::Ordering;
    use crate::gates::equality_base::EqualityGate;
    use crate::gates::exponentiation::ExponentiationGate;
    use crate::gates::gate::U32QuotientGate;
    use crate::gates::reducing::ReducingGate;
    use crate::gates::reducing_extension::ReducingExtensionGate;
    use crate::hash::poseidon2::metal::{
        RangeCheckQuotientSpec, U32QuotientKind, U32QuotientSpec,
    };

    GPU_RANGE_QUOTIENT_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    if common_data.num_lookup_polys != 0 || common_data.config.num_challenges != 2 {
        if gpu_poseidon_quotient_diagnostics_enabled() {
            eprintln!(
                "[gpu-range-quotient] guard rejected: lookups={} challenges={}",
                common_data.num_lookup_polys, common_data.config.num_challenges
            );
        }
        return None;
    }

    let include_unused_selector = common_data.selectors_info.num_selectors() > 1;
    let raw_constant_base = common_data
        .selectors_info
        .num_selectors()
        .checked_add(common_data.num_lookup_selectors)?;
    if raw_constant_base > common_data.num_constants {
        return None;
    }
    let mut gate_indices = Vec::new();
    // Random-access gates are excluded from the CPU quotient only after the
    // combined Metal command has accepted and submitted every metadata record.
    let mut random_access_gate_indices = Vec::new();
    let mut specs = Vec::new();
    let mut u32_specs = Vec::new();
    for (gate_index, gate) in common_data.gates.iter().enumerate() {
        let range = gate.0.range_check_quotient_gate();
        let u32_gate = gate.0.u32_quotient_gate();
        if range.is_some() && u32_gate.is_some() {
            if gpu_poseidon_quotient_diagnostics_enabled() {
                eprintln!(
                    "[gpu-range-quotient] gate {gate_index} advertised conflicting layouts"
                );
            }
            return None;
        }
        if let Some(range) = range {
            if range.bit_size == 0 || range.bit_size > 64 || range.num_ops == 0 {
                if gpu_poseidon_quotient_diagnostics_enabled() {
                    eprintln!("[gpu-range-quotient] invalid gate metadata: {range:?}");
                }
                return None;
            }
            let num_aux = range.bit_size.div_ceil(2);
            let expected = range.num_ops.checked_mul(1 + num_aux)?;
            if gate.0.num_wires() != expected || gate.0.num_constraints() != expected {
                if gpu_poseidon_quotient_diagnostics_enabled() {
                    eprintln!(
                        "[gpu-range-quotient] layout mismatch gate={gate_index} metadata={range:?} \
                         wires={} constraints={} expected={expected}",
                        gate.0.num_wires(),
                        gate.0.num_constraints(),
                    );
                }
                return None;
            }
            let selector_column = common_data.selectors_info.selector_indices[gate_index];
            specs.push(RangeCheckQuotientSpec {
                selector_column,
                gate_index,
                group: common_data.selectors_info.groups[selector_column].clone(),
                include_unused_selector,
                num_ops: range.num_ops,
                bit_size: range.bit_size,
            });
            gate_indices.push(gate_index);
        }
        if let Some(u32_gate) = u32_gate {
            let (kind, num_ops, expected_wires, expected_constraints) = match u32_gate {
                U32QuotientGate::Arithmetic { num_ops } => (
                    U32QuotientKind::Arithmetic,
                    num_ops,
                    num_ops.checked_mul(38)?,
                    num_ops.checked_mul(36)?,
                ),
                // Every width shares one layout: five routed words per
                // operation followed by `base_bits / 2` base-4 result limbs,
                // so the wire and constraint counts are linear in the width.
                U32QuotientGate::Subtraction {
                    num_ops,
                    base_bits,
                } => {
                    let Some(result_limbs) = supported_quotient_result_limbs(base_bits) else {
                        if gpu_poseidon_quotient_diagnostics_enabled() {
                            eprintln!("[gpu-range-quotient] invalid U32 metadata: {u32_gate:?}");
                        }
                        return None;
                    };
                    (
                        U32QuotientKind::Subtraction { result_limbs },
                        num_ops,
                        num_ops.checked_mul(result_limbs.checked_add(5)?)?,
                        num_ops.checked_mul(result_limbs.checked_add(3)?)?,
                    )
                }
                U32QuotientGate::AddMany {
                    num_ops,
                    num_addends,
                    base_bits,
                    num_carry_limbs,
                } => {
                    let Some(result_limbs) = supported_quotient_result_limbs(base_bits) else {
                        if gpu_poseidon_quotient_diagnostics_enabled() {
                            eprintln!("[gpu-range-quotient] invalid U32 metadata: {u32_gate:?}");
                        }
                        return None;
                    };
                    if num_addends == 0 || num_addends > 16 || num_carry_limbs == 0 {
                        if gpu_poseidon_quotient_diagnostics_enabled() {
                            eprintln!(
                                "[gpu-range-quotient] invalid U32 metadata: {u32_gate:?}"
                            );
                        }
                        return None;
                    }
                    let limbs = result_limbs.checked_add(num_carry_limbs)?;
                    (
                        U32QuotientKind::AddMany {
                            num_addends,
                            result_limbs,
                            num_carry_limbs,
                        },
                        num_ops,
                        num_ops.checked_mul(num_addends.checked_add(3)?.checked_add(limbs)?)?,
                        num_ops.checked_mul(limbs.checked_add(3)?)?,
                    )
                }
                U32QuotientGate::ByteDecomposition { num_ops, num_limbs } => {
                    if num_limbs == 0 || num_limbs > 24 {
                        if gpu_poseidon_quotient_diagnostics_enabled() {
                            eprintln!(
                                "[gpu-range-quotient] invalid byte metadata: {u32_gate:?}"
                            );
                        }
                        return None;
                    }
                    let per_op = num_limbs.checked_mul(5)?.checked_add(1)?;
                    let expected = num_ops.checked_mul(per_op)?;
                    (
                        U32QuotientKind::ByteDecomposition { num_limbs },
                        num_ops,
                        expected,
                        expected,
                    )
                }
                U32QuotientGate::QuinticMultiplication { num_ops } => (
                    U32QuotientKind::QuinticMultiplication,
                    num_ops,
                    num_ops.checked_mul(15)?,
                    num_ops.checked_mul(5)?,
                ),
                U32QuotientGate::QuinticSquaring { num_ops } => (
                    U32QuotientKind::QuinticSquaring,
                    num_ops,
                    num_ops.checked_mul(20)?,
                    num_ops.checked_mul(15)?,
                ),
                U32QuotientGate::RandomAccess {
                    bits,
                    num_ops,
                    num_extra_constants,
                } => {
                    if !matches!(
                        (bits, num_ops, num_extra_constants),
                        (3, 8, 0) | (4, 4, 2) | (6, 1, 2)
                    ) {
                        if gpu_poseidon_quotient_diagnostics_enabled() {
                            eprintln!(
                                "[gpu-range-quotient] unaudited random-access metadata: \
                                 {u32_gate:?}"
                            );
                        }
                        return None;
                    }
                    let vec_size = 1usize.checked_shl(u32::try_from(bits).ok()?)?;
                    let routed_per_copy = vec_size.checked_add(2)?;
                    let extra_wire_base = routed_per_copy.checked_mul(num_ops)?;
                    let routed_wires = extra_wire_base.checked_add(num_extra_constants)?;
                    let expected_wires = routed_wires.checked_add(num_ops.checked_mul(bits)?)?;
                    let expected_constraints = num_ops
                        .checked_mul(bits.checked_add(2)?)?
                        .checked_add(num_extra_constants)?;
                    let constant_end = raw_constant_base.checked_add(num_extra_constants)?;
                    let expected_extra_constant_wires = (0..num_extra_constants)
                        .map(|i| extra_wire_base.checked_add(i).map(|wire| (i, wire)))
                        .collect::<Option<Vec<_>>>()?;
                    if gate.0.num_constants() != num_extra_constants
                        || gate.0.extra_constant_wires() != expected_extra_constant_wires
                        || routed_wires > common_data.config.num_routed_wires
                        || expected_wires > common_data.config.num_wires
                        || constant_end > common_data.num_constants
                    {
                        if gpu_poseidon_quotient_diagnostics_enabled() {
                            eprintln!(
                                "[gpu-range-quotient] random-access layout mismatch \
                                 gate={gate_index} metadata={u32_gate:?} constants={} \
                                 expected_constants={num_extra_constants}",
                                gate.0.num_constants(),
                            );
                        }
                        return None;
                    }
                    (
                        U32QuotientKind::RandomAccess {
                            bits,
                            num_extra_constants,
                            constant_base: raw_constant_base,
                        },
                        num_ops,
                        expected_wires,
                        expected_constraints,
                    )
                }
            };
            if num_ops == 0
                || gate.0.num_wires() != expected_wires
                || gate.0.num_constraints() != expected_constraints
            {
                if gpu_poseidon_quotient_diagnostics_enabled() {
                    eprintln!(
                        "[gpu-range-quotient] U32 layout mismatch gate={gate_index} \
                         metadata={u32_gate:?} wires={} constraints={} \
                         expected_wires={expected_wires} expected_constraints={expected_constraints}",
                        gate.0.num_wires(),
                        gate.0.num_constraints(),
                    );
                }
                return None;
            }
            let selector_column = common_data.selectors_info.selector_indices[gate_index];
            u32_specs.push(U32QuotientSpec {
                selector_column,
                gate_index,
                group: common_data.selectors_info.groups[selector_column].clone(),
                include_unused_selector,
                num_ops,
                kind,
            });
            if matches!(u32_gate, U32QuotientGate::RandomAccess { .. }) {
                random_access_gate_indices.push(gate_index);
            } else {
                gate_indices.push(gate_index);
            }
        }
        // These vendored gates sit at the top of the surviving wire span in
        // the production circuits and are pure arithmetic, so they are matched
        // by type here instead of through the downstream-crate trait hooks
        // (those hooks exist only to avoid a `plonky2` -> circuit-crate dep).
        let native = if let Some(exponentiation) =
            gate.0.as_any().downcast_ref::<ExponentiationGate<F, D>>()
        {
            let num_power_bits = exponentiation.num_power_bits;
            Some((
                U32QuotientKind::Exponentiation,
                num_power_bits,
                num_power_bits.checked_mul(2)?.checked_add(2)?,
                num_power_bits.checked_add(1)?,
            ))
        } else if let Some(equality) = gate.0.as_any().downcast_ref::<EqualityGate>() {
            // The gate reads its single constant (the "one" value) as local
            // constant 0, i.e. the raw constant base: the column immediately
            // after every gate selector and lookup selector.
            if equality.num_ops == 0 || gate.0.num_constants() != 1 {
                None
            } else if raw_constant_base.checked_add(1)? > common_data.num_constants {
                None
            } else {
                Some((
                    U32QuotientKind::Equality {
                        constant_column: raw_constant_base,
                    },
                    equality.num_ops,
                    equality.num_ops.checked_mul(6)?,
                    equality.num_ops.checked_mul(4)?,
                ))
            }
        } else if let Some(reducing) = gate.0.as_any().downcast_ref::<ReducingGate<D>>() {
            // The kernel's extension arithmetic is specialised to the
            // quadratic Goldilocks extension.
            if D != 2 || gate.0.num_constants() != 0 {
                None
            } else {
                Some((
                    U32QuotientKind::Reducing {
                        extension_coeffs: false,
                    },
                    reducing.num_coeffs,
                    reducing.num_coeffs.checked_mul(3)?.checked_add(4)?,
                    reducing.num_coeffs.checked_mul(2)?,
                ))
            }
        } else if let Some(reducing) = gate.0.as_any().downcast_ref::<ReducingExtensionGate<D>>() {
            if D != 2 || gate.0.num_constants() != 0 {
                None
            } else {
                Some((
                    U32QuotientKind::Reducing {
                        extension_coeffs: true,
                    },
                    reducing.num_coeffs,
                    reducing.num_coeffs.checked_mul(4)?.checked_add(4)?,
                    reducing.num_coeffs.checked_mul(2)?,
                ))
            }
        } else {
            None
        };
        if let Some((kind, num_ops, expected_wires, expected_constraints)) = native {
            if range.is_some() || u32_gate.is_some() {
                if gpu_poseidon_quotient_diagnostics_enabled() {
                    eprintln!(
                        "[gpu-range-quotient] gate {gate_index} advertised conflicting layouts"
                    );
                }
                return None;
            }
            if num_ops == 0
                || gate.0.num_wires() != expected_wires
                || gate.0.num_constraints() != expected_constraints
            {
                if gpu_poseidon_quotient_diagnostics_enabled() {
                    eprintln!(
                        "[gpu-range-quotient] native layout mismatch gate={gate_index} \
                         kind={kind:?} wires={} constraints={} expected_wires={expected_wires} \
                         expected_constraints={expected_constraints}",
                        gate.0.num_wires(),
                        gate.0.num_constraints(),
                    );
                }
                return None;
            }
            let selector_column = common_data.selectors_info.selector_indices[gate_index];
            u32_specs.push(U32QuotientSpec {
                selector_column,
                gate_index,
                group: common_data.selectors_info.groups[selector_column].clone(),
                include_unused_selector,
                num_ops,
                kind,
            });
            gate_indices.push(gate_index);
        }
    }
    if specs.is_empty() && u32_specs.is_empty() {
        if gpu_poseidon_quotient_diagnostics_enabled() {
            eprintln!("[gpu-range-quotient] no advertised RangeCheck/U32 gates");
        }
        return None;
    }

    let Some(wires) = wires_commitment.merkle_tree.shared_columns() else {
        if gpu_poseidon_quotient_diagnostics_enabled() {
            eprintln!("[gpu-range-quotient] wire commitment is not Metal-backed");
        }
        return None;
    };
    let Some(constants) = prover_data
        .constants_sigmas_commitment
        .merkle_tree
        .shared_columns()
    else {
        if gpu_poseidon_quotient_diagnostics_enabled() {
            eprintln!("[gpu-range-quotient] constants commitment is not Metal-backed");
        }
        return None;
    };
    // These gate rows share the same alpha positions as every other gate.
    // Only the permutation/Z prefix precedes the combined gate-row block.
    let alpha_offset = common_data.config.num_challenges * (common_data.num_partial_products + 2);
    let Some(job) = crate::hash::poseidon2::metal::start_range_check_gate_quotient(
        wires,
        constants,
        quotient_rows,
        step,
        &specs,
        &u32_specs,
        alphas,
        alpha_offset,
    ) else {
        if gpu_poseidon_quotient_diagnostics_enabled() {
            eprintln!(
                "[gpu-range-quotient] Metal launch rejected: gates={gate_indices:?} \
                 wire_shape={}x{} constant_shape={}x{} rows={quotient_rows} step={step}",
                wires.cols(),
                wires.rows(),
                constants.cols(),
                constants.rows(),
            );
        }
        return None;
    };
    gate_indices.extend(random_access_gate_indices);
    let started = GPU_RANGE_QUOTIENT_STARTED.fetch_add(1, Ordering::Relaxed) + 1;
    log::info!(
        "Metal RangeCheck quotient active: started={started}, gates={gate_indices:?}, \
         rows={quotient_rows}, step={step}, shared_columns=true"
    );
    if gpu_poseidon_quotient_diagnostics_enabled() {
        eprintln!(
            "[gpu-range-quotient] active started={started} gates={gate_indices:?} \
             rows={quotient_rows} step={step} shared_columns=true"
        );
    }
    Some((gate_indices, job))
}

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
    col_major_perm: bool,
    allow_gpu_poseidon: bool,
) -> Vec<PolynomialCoeffs<F>> {
    let num_challenges = common_data.config.num_challenges;

    let has_lookup = common_data.num_lookup_polys != 0;

    // The lookup constraint evaluator consumes per-point rows, so the
    // column-major permutation layout is only usable without lookups,
    // whatever the caller asked for.
    let col_major_perm = col_major_perm && !has_lookup;

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

    // Process-global cached subgroup (bit-identical to computing it here): the
    // serial 2^19-length dependent multiply chain runs once per process
    // instead of once per proof.
    let points =
        precomputed::two_adic_subgroup::<F>(common_data.degree_bits() + quotient_degree_bits);
    // Same reasoning applied to the shifted points: `coset_shift * x` over the
    // whole domain is identical for every proof of this size, so compute it
    // once per process and slice per batch instead of re-multiplying every
    // domain element on every proof.
    let shifted_points = precomputed::shifted_two_adic_subgroup::<F>(
        common_data.degree_bits() + quotient_degree_bits,
    );
    let lde_size = points.len();
    debug_assert_eq!(shifted_points.len(), lde_size);

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    let gpu_poseidon = allow_gpu_poseidon
        .then(|| {
            start_gpu_poseidon_gate_quotient(
                common_data,
                prover_data,
                wires_commitment,
                lde_size,
                step,
                alphas,
            )
        })
        .flatten();
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    let gpu_range = allow_gpu_poseidon
        .then(|| {
            start_gpu_range_check_gate_quotient(
                common_data,
                prover_data,
                wires_commitment,
                lde_size,
                step,
                alphas,
            )
        })
        .flatten();
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    let excluded_gate_indices = gpu_poseidon
        .as_ref()
        .map(|(gate_index, _)| core::iter::once(*gate_index))
        .into_iter()
        .flatten()
        .chain(
            gpu_range
                .as_ref()
                .map(|(gate_indices, _)| gate_indices.iter().copied())
                .into_iter()
                .flatten(),
        )
        .collect::<Vec<_>>();
    #[cfg(not(all(feature = "std", target_arch = "aarch64", target_os = "macos")))]
    let excluded_gate_indices = {
        let _ = allow_gpu_poseidon;
        Vec::new()
    };
    // The offload decision is fixed for this quotient proof. Materialize the
    // surviving CPU gates once, in the same ascending order as `gates`, rather
    // than linearly scanning the exclusions for every gate in every batch.
    let cpu_gate_indices = (0..common_data.gates.len())
        .filter(|gate_index| !excluded_gate_indices.contains(gate_index))
        .collect::<Vec<_>>();

    let z_h_on_coset = ZeroPolyOnCoset::new(common_data.degree_bits(), quotient_degree_bits);
    // The `L_0` denominator inverses consumed by `eval_l_0` depend only on
    // `(degree_bits, quotient_degree_bits, coset shift)` — not on any challenge — so they are
    // computed once per circuit shape for the process and shared across proofs. Each cached
    // entry is bit-identical to the per-point inversion it replaces.
    #[cfg(feature = "std")]
    let z_h_on_coset = z_h_on_coset.with_l_0_denominator_inverses(
        l_0_table_cache::l_0_denominator_inverses::<F>(
            common_data.degree_bits(),
            quotient_degree_bits,
        ),
    );

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

    let points_batches = points.par_chunks(BATCH_SIZE);
    let num_batches = points.len().div_ceil(BATCH_SIZE);

    struct QuotientScratch<F: RichField> {
        indices: Vec<usize>,
        indices_next: Vec<usize>,
        local_constants: Vec<F>,
        local_wires: Vec<F>,
        s_sigmas_flat: Vec<F>,
        zs_local_flat: Vec<F>,
        zs_next_flat: Vec<F>,
        vanishing: VanishingScratch<F>,
    }

    let zs_row_width = zs_partial_products_and_lookup_commitment.lde_row_width();
    let num_routed_wires = common_data.config.num_routed_wires;
    // GPU-specialized gates read the retained full-width wire commitment
    // directly. The CPU constraint pass needs only the declared prefix of its
    // remaining gates, while the permutation argument always needs the routed
    // prefix. Do not gather dead high columns for offloaded Poseidon/Range
    // gates into every 32-point CPU scratch batch.
    let cpu_num_wires = cpu_gate_indices
        .iter()
        .map(|&gate_index| common_data.gates[gate_index].0.num_wires())
        .max()
        .unwrap_or(0)
        .max(num_routed_wires);
    debug_assert!(cpu_num_wires <= common_data.config.num_wires);
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    if gpu_poseidon_quotient_diagnostics_enabled() && !excluded_gate_indices.is_empty() {
        eprintln!(
            "[gpu-gate-quotient] CPU wire gather width {cpu_num_wires}/{}; excluded={excluded_gate_indices:?}",
            common_data.config.num_wires,
        );
    }
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    if gpu_poseidon_quotient_diagnostics_enabled() {
        gate_census_once(common_data, &excluded_gate_indices, cpu_num_wires);
    }

    // The zero-fill this used to do existed only to seed the Horner chain in
    // `reduce_gate_constraints_base_batch`, which is the first thing every
    // batch does. That chain now *assigns* its first reversed row instead of
    // accumulating into zeros (a raw-limb-identical change: the old first pass
    // computed `reduce128(term as u128)`, which returns `term` unchanged), so
    // every slot of this buffer is stored before it is read and the memset is
    // dead. `par_chunks_mut` partitions the whole buffer and each batch writes
    // all of its own slice, including a short final batch.
    //
    // `F` has no `IsZero` specialization, so the old `vec![F::ZERO; n]` was a
    // real serial store loop, not `alloc_zeroed`: 8 MiB per d16 tx proof,
    // 2 MiB per chain-step proof, on the per-proof spine between the Zs
    // commitment and the quotient commitment.
    let quotient_len = points.len() * num_challenges;
    let mut quotient_values: Vec<F> = Vec::with_capacity(quotient_len);
    // SAFETY: capacity is exactly `quotient_len`, and the parallel pass below
    // writes every element before any is read (see above). Same idiom as the
    // promoted zero-tail fast path in `fri/oracle.rs`.
    unsafe { quotient_values.set_len(quotient_len) };
    quotient_values
        .par_chunks_mut(BATCH_SIZE * num_challenges)
        .zip(points_batches)
        .enumerate()
        .for_each_init(
            || QuotientScratch::<F> {
                indices: Vec::with_capacity(BATCH_SIZE),
                indices_next: Vec::with_capacity(BATCH_SIZE),
                local_constants: Vec::new(),
                local_wires: Vec::new(),
                s_sigmas_flat: Vec::new(),
                zs_local_flat: Vec::new(),
                zs_next_flat: Vec::new(),
                vanishing: VanishingScratch::default(),
            },
            |scratch, (batch_i, (quotient_values_batch, xs_batch))| {
                // Each batch must be the same size, except the last one, which may be smaller.
                debug_assert!(
                    xs_batch.len() == BATCH_SIZE
                        || (batch_i == num_batches - 1 && xs_batch.len() <= BATCH_SIZE)
                );

                let n = xs_batch.len();
                scratch.indices.clear();
                scratch
                    .indices
                    .extend(BATCH_SIZE * batch_i..BATCH_SIZE * batch_i + n);
                scratch.indices_next.clear();
                scratch
                    .indices_next
                    .extend(scratch.indices.iter().map(|&i| (i + next_step) % lde_size));

                let shifted_xs_batch = &shifted_points[BATCH_SIZE * batch_i..][..n];
                debug_assert!(
                    shifted_xs_batch
                        .iter()
                        .zip(xs_batch)
                        .all(|(&sx, &x)| sx == F::coset_shift() * x)
                );

                // The constants and sigma columns are circuit-fixed, so their
                // quotient-domain values were extracted once at circuit build
                // time; copy them per batch instead of re-walking the strided
                // LDE (which amplifies cache-line traffic 8x at step 8).
                let cache_start = BATCH_SIZE * batch_i;
                // The cache is column-major (`PolyMajor`); the per-point
                // (`PointMajor`) path with lookups keeps the original gathers.
                let constants_cache = if col_major_perm {
                    prover_data.constants_sigmas_quotient_cache.as_ref()
                } else {
                    None
                };
                if let Some(cache) = constants_cache {
                    debug_assert_eq!(
                        prover_data.constants_sigmas_quotient_step, step,
                        "quotient gather step must match the cache extraction step"
                    );
                    let cc = common_data.constants_range().len();
                    let q = prover_data.constants_sigmas_quotient_domain;
                    scratch.local_constants.resize(cc * n, F::ZERO);
                    for ci in 0..cc {
                        scratch.local_constants[ci * n..(ci + 1) * n].copy_from_slice(
                            &cache[ci * q + cache_start..ci * q + cache_start + n],
                        );
                    }
                    let sc = common_data.sigmas_range().len();
                    scratch.s_sigmas_flat.resize(sc * n, F::ZERO);
                    for ci in 0..sc {
                        scratch.s_sigmas_flat[ci * n..(ci + 1) * n].copy_from_slice(
                            &cache[(cc + ci) * q + cache_start..(cc + ci) * q + cache_start + n],
                        );
                    }
                } else {
                    prover_data.constants_sigmas_commitment.fill_lde_batch(
                        &scratch.indices,
                        step,
                        common_data.constants_range(),
                        BatchLayout::PolyMajor,
                        &mut scratch.local_constants,
                    );
                    // Layout seam: the no-lookup column evaluator consumes the
                    // PolyMajor gathers as-is (and the "next" gather narrows to
                    // the Z columns, the only ones it reads); the per-point path
                    // keeps the full-width PointMajor gathers and row views.
                    let (batch_layout, _zs_local_range, _zs_next_range) = if col_major_perm {
                        (BatchLayout::PolyMajor, 0..0, 0..0)
                    } else {
                        (BatchLayout::PointMajor, 0..zs_row_width, 0..zs_row_width)
                    };

                    prover_data.constants_sigmas_commitment.fill_lde_batch(
                        &scratch.indices,
                        step,
                        common_data.sigmas_range(),
                        batch_layout,
                        &mut scratch.s_sigmas_flat,
                    );
                }
                // Layout seam: the no-lookup column evaluator consumes the
                // PolyMajor gathers as-is (and the "next" gather narrows to
                // the Z columns, the only ones it reads); the per-point path
                // keeps the full-width PointMajor gathers and row views.
                let (batch_layout, zs_local_range, zs_next_range) = if col_major_perm {
                    (
                        BatchLayout::PolyMajor,
                        0..common_data.partial_products_range().end,
                        common_data.zs_range(),
                    )
                } else {
                    (BatchLayout::PointMajor, 0..zs_row_width, 0..zs_row_width)
                };
                wires_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    0..cpu_num_wires,
                    BatchLayout::PolyMajor,
                    &mut scratch.local_wires,
                );
                zs_partial_products_and_lookup_commitment.fill_lde_batch(
                    &scratch.indices,
                    step,
                    zs_local_range,
                    batch_layout,
                    &mut scratch.zs_local_flat,
                );
                zs_partial_products_and_lookup_commitment.fill_lde_batch(
                    &scratch.indices_next,
                    step,
                    zs_next_range,
                    batch_layout,
                    &mut scratch.zs_next_flat,
                );

                let indices_batch = &scratch.indices;
                // Per-point row views over the PointMajor gathers, built only
                // for the per-point path; the column path passes the flat
                // buffers straight through, so these four allocations vanish
                // from the hot (no-lookup) path entirely.
                type RowViews<'v, F> = (Vec<&'v [F]>, Vec<&'v [F]>, Vec<&'v [F]>, Vec<&'v [F]>);
                let (local_zs_batch, next_zs_batch, partial_products_batch, s_sigmas_batch): RowViews<'_, F> =
                    if col_major_perm {
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                    } else {
                        (
                            (0..n)
                                .map(|k| {
                                    &scratch.zs_local_flat[k * zs_row_width..]
                                        [common_data.zs_range()]
                                })
                                .collect(),
                            (0..n)
                                .map(|k| {
                                    &scratch.zs_next_flat[k * zs_row_width..]
                                        [common_data.zs_range()]
                                })
                                .collect(),
                            (0..n)
                                .map(|k| {
                                    &scratch.zs_local_flat[k * zs_row_width..]
                                        [common_data.partial_products_range()]
                                })
                                .collect(),
                            (0..n)
                                .map(|k| {
                                    &scratch.s_sigmas_flat
                                        [k * num_routed_wires..(k + 1) * num_routed_wires]
                                })
                                .collect(),
                        )
                    };
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

                let perm = if col_major_perm {
                    PermutationBatch::Cols {
                        zs_partial_products_cols: &scratch.zs_local_flat,
                        zs_next_cols: &scratch.zs_next_flat,
                        s_sigmas_cols: &scratch.s_sigmas_flat,
                    }
                } else {
                    PermutationBatch::Rows {
                        local_zs_batch: &local_zs_batch,
                        next_zs_batch: &next_zs_batch,
                        partial_products_batch: &partial_products_batch,
                        s_sigmas_batch: &s_sigmas_batch,
                    }
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
                    shifted_xs_batch,
                    vars_batch,
                    perm,
                    &local_lookup_batch,
                    &next_lookup_batch,
                    betas,
                    gammas,
                    beta_k_is,
                    deltas,
                    alphas,
                    &cpu_gate_indices,
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

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    if let Some((_, job)) = &gpu_poseidon {
        let gpu_values = match job.finish() {
            Ok(values) => {
                GPU_POSEIDON_QUOTIENT_COMPLETED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                values
            }
            Err(error) => {
                GPU_POSEIDON_QUOTIENT_FALLBACKS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                log::warn!(
                    "Metal Poseidon2 gate quotient failed; recomputing quotient on CPU: {error}"
                );
                if gpu_poseidon_quotient_diagnostics_enabled() {
                    eprintln!(
                        "[gpu-poseidon-quotient] runtime failure; falling back to CPU: {error}"
                    );
                }
                return compute_quotient_polys(
                    common_data,
                    prover_data,
                    public_inputs_hash,
                    wires_commitment,
                    zs_partial_products_and_lookup_commitment,
                    betas,
                    gammas,
                    beta_k_is,
                    deltas,
                    alphas,
                    col_major_perm,
                    false,
                );
            }
        };
        debug_assert_eq!(gpu_values.len(), quotient_values.len());
        quotient_values
            .par_chunks_exact_mut(num_challenges)
            .zip(gpu_values.par_chunks_exact(num_challenges))
            .enumerate()
            .for_each(|(i, (cpu_values, gpu_values))| {
                let denominator_inv = z_h_on_coset.eval_inverse(i);
                for (cpu, &gpu) in cpu_values.iter_mut().zip(gpu_values) {
                    *cpu += gpu * denominator_inv;
                }
            });
    }

    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    if let Some((_, job)) = &gpu_range {
        let gpu_values = match job.finish() {
            Ok(values) => {
                GPU_RANGE_QUOTIENT_COMPLETED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                values
            }
            Err(error) => {
                GPU_RANGE_QUOTIENT_FALLBACKS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                log::warn!(
                    "Metal RangeCheck gate quotient failed; recomputing quotient on CPU: {error}"
                );
                if gpu_poseidon_quotient_diagnostics_enabled() {
                    eprintln!(
                        "[gpu-range-quotient] runtime failure; falling back to CPU: {error}"
                    );
                }
                return compute_quotient_polys(
                    common_data,
                    prover_data,
                    public_inputs_hash,
                    wires_commitment,
                    zs_partial_products_and_lookup_commitment,
                    betas,
                    gammas,
                    beta_k_is,
                    deltas,
                    alphas,
                    col_major_perm,
                    false,
                );
            }
        };
        debug_assert_eq!(gpu_values.len(), quotient_values.len());
        quotient_values
            .par_chunks_exact_mut(num_challenges)
            .zip(gpu_values.par_chunks_exact(num_challenges))
            .enumerate()
            .for_each(|(i, (cpu_values, gpu_values))| {
                let denominator_inv = z_h_on_coset.eval_inverse(i);
                for (cpu, &gpu) in cpu_values.iter_mut().zip(gpu_values) {
                    *cpu += gpu * denominator_inv;
                }
            });
    }

    debug_assert_eq!(quotient_values.len(), points.len() * num_challenges);
    // One streaming pass splits the interleaved point-major buffer into the
    // per-challenge columns, instead of `num_challenges` parallel passes each
    // stride-reading the whole buffer. Same values in the same order; only
    // which pass writes them changes.
    let mut challenge_columns: Vec<Vec<F>> = (0..num_challenges)
        .map(|_| Vec::with_capacity(points.len()))
        .collect();
    for point_values in quotient_values.chunks_exact(num_challenges) {
        for (column, &value) in challenge_columns.iter_mut().zip(point_values) {
            column.push(value);
        }
    }
    let inverse_coset_shift_powers = precomputed::inverse_coset_shift_powers::<F>(points.len());
    challenge_columns
        .into_par_iter()
        .map(|column| {
            // Fuse the coset post-scaling into the IFFT instead of walking the
            // whole coefficient vector again afterwards, reusing a
            // process-global inverse-shift power chain.
            PolynomialValues::new(column)
                .coset_ifft_with_powers(inverse_coset_shift_powers.as_slice())
        })
        .collect()
}

/// Process-global caches for deterministic per-degree precomputations that
/// were being redone per proof: the quotient-domain two-adic subgroup (a
/// serial dependent multiply chain over 2^19 points) and the coset-shift power
/// table used by every `PolynomialBatch` LDE. Entries are keyed by field type
/// and size; the stored vectors are exactly what the direct computation
/// returns, computed once, so every lookup is bit-identical to computing in
/// place. (Kept here rather than in `plonky2_field` so the file set stays
/// disjoint from pending `fft.rs` work.)
pub(crate) mod precomputed {
    #[cfg(feature = "std")]
    mod imp {
        use core::any::{Any, TypeId};
        use std::collections::HashMap;
        use std::sync::{Arc, OnceLock, RwLock};

        use crate::field::types::Field;

        type Map = RwLock<HashMap<(TypeId, usize), Arc<dyn Any + Send + Sync>>>;

        static SUBGROUPS: OnceLock<Map> = OnceLock::new();
        static COSET_POWERS: OnceLock<Map> = OnceLock::new();
        static SHIFTED_SUBGROUPS: OnceLock<Map> = OnceLock::new();
        static INVERSE_COSET_POWERS: OnceLock<Map> = OnceLock::new();

        fn get_or_compute<F: Field>(
            cache: &'static OnceLock<Map>,
            len_key: usize,
            compute: impl FnOnce() -> Vec<F>,
        ) -> Arc<Vec<F>> {
            let key = (TypeId::of::<F>(), len_key);
            let map = cache.get_or_init(|| RwLock::new(HashMap::new()));
            if let Some(hit) = map.read().unwrap().get(&key) {
                return Arc::clone(hit)
                    .downcast::<Vec<F>>()
                    .ok()
                    .expect("type-keyed cache entry has the keyed type");
            }
            let computed: Arc<Vec<F>> = Arc::new(compute());
            let mut map = map.write().unwrap();
            // If another thread inserted concurrently, keep its (identical)
            // table so all callers share one allocation.
            let entry = map
                .entry(key)
                .or_insert_with(|| computed as Arc<dyn Any + Send + Sync>);
            Arc::clone(entry)
                .downcast::<Vec<F>>()
                .ok()
                .expect("type-keyed cache entry has the keyed type")
        }

        /// Cached `F::two_adic_subgroup(n_log)`.
        pub(crate) fn two_adic_subgroup<F: Field>(n_log: usize) -> Arc<Vec<F>> {
            get_or_compute(&SUBGROUPS, n_log, || F::two_adic_subgroup(n_log))
        }

        /// Cached `F::coset_shift().powers().take(degree)`.
        pub(crate) fn coset_shift_powers<F: Field>(degree: usize) -> Arc<Vec<F>> {
            get_or_compute(&COSET_POWERS, degree, || {
                F::coset_shift().powers().take(degree).collect()
            })
        }

        /// Cached `x * F::coset_shift()` over the whole two-adic subgroup of
        /// size `1 << n_log`. The quotient evaluator needs the shifted point
        /// for every domain element of every proof, and the domain depends
        /// only on its size, so this full multiply-and-write traversal runs
        /// once per process instead of once per proof.
        pub(crate) fn shifted_two_adic_subgroup<F: Field>(n_log: usize) -> Arc<Vec<F>> {
            get_or_compute(&SHIFTED_SUBGROUPS, n_log, || {
                let shift = F::coset_shift();
                two_adic_subgroup::<F>(n_log).iter().map(|&x| shift * x).collect()
            })
        }

        /// Cached `F::coset_shift().inverse().powers().take(degree)`. The
        /// quotient columns' coset IFFT post-scaling uses the same power chain
        /// on every proof of a given size, so build it once per process.
        pub(crate) fn inverse_coset_shift_powers<F: Field>(degree: usize) -> Arc<Vec<F>> {
            get_or_compute(&INVERSE_COSET_POWERS, degree, || {
                F::coset_shift().inverse().powers().take(degree).collect()
            })
        }
    }

    /// Without `std` there is no process-global synchronization; fall back to
    /// direct (uncached) computation, which is what the callers did before.
    #[cfg(not(feature = "std"))]
    mod imp {
        use alloc::sync::Arc;
        use alloc::vec::Vec;

        use crate::field::types::Field;

        pub(crate) fn two_adic_subgroup<F: Field>(n_log: usize) -> Arc<Vec<F>> {
            Arc::new(F::two_adic_subgroup(n_log))
        }

        pub(crate) fn coset_shift_powers<F: Field>(degree: usize) -> Arc<Vec<F>> {
            Arc::new(F::coset_shift().powers().take(degree).collect::<Vec<F>>())
        }

        pub(crate) fn shifted_two_adic_subgroup<F: Field>(n_log: usize) -> Arc<Vec<F>> {
            let shift = F::coset_shift();
            Arc::new(F::two_adic_subgroup(n_log).into_iter().map(|x| shift * x).collect::<Vec<F>>())
        }

        pub(crate) fn inverse_coset_shift_powers<F: Field>(degree: usize) -> Arc<Vec<F>> {
            Arc::new(
                F::coset_shift()
                    .inverse()
                    .powers()
                    .take(degree)
                    .collect::<Vec<F>>(),
            )
        }
    }

    pub(crate) use imp::{
        coset_shift_powers, inverse_coset_shift_powers, shifted_two_adic_subgroup,
        two_adic_subgroup,
    };
}

#[cfg(test)]
mod quotient_layout_tests {
    use core::sync::atomic::Ordering;

    use anyhow::Result;

    use super::{precomputed, BatchLayout, COMPARE_QUOTIENT_LAYOUTS};
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    use super::{gpu_poseidon_quotient_stats, COMPARE_GPU_QUOTIENT};
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    use crate::field::extension::Extendable;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, Field64};
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    use crate::gates::gate::U32QuotientGate;
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    use crate::gates::noop::NoopGate;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    use crate::util::reducing::ReducingFactorTarget;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    use crate::plonk::config::Poseidon2GoldilocksConfig;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    fn small_circuit() -> (CircuitData<F, C, D>, PartialWitness<F>) {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let x = builder.add_virtual_target();
        let mut cur = x;
        for i in 0..64 {
            cur = builder.mul_add(cur, cur, x);
            let c = builder.constant(F::from_canonical_usize(i + 1));
            cur = builder.add(cur, c);
        }
        builder.register_public_input(cur);
        let data = builder.build::<C>();
        let mut pw = PartialWitness::new();
        pw.set_target(x, F::from_canonical_u64(3)).unwrap();
        (data, pw)
    }

    /// B1/B2/D1 differential gate: within a single prove call — same witness,
    /// commitments and challenges — the default column-major (`PolyMajor`)
    /// quotient path and the per-point (`PointMajor`) reference path must
    /// produce value-identical quotient polynomials. The element-wise
    /// comparison itself runs inside `prove` (see `COMPARE_QUOTIENT_LAYOUTS`);
    /// the proof must also verify.
    #[test]
    fn quotient_layout_paths_agree() -> Result<()> {
        let (data, pw) = small_circuit();
        assert!(data.common.luts.is_empty());

        COMPARE_QUOTIENT_LAYOUTS.store(true, Ordering::SeqCst);
        let proof = data.prove(pw);
        COMPARE_QUOTIENT_LAYOUTS.store(false, Ordering::SeqCst);

        data.verify(proof?)?;
        Ok(())
    }

    /// End-to-end retained-column differential for every audited random-access
    /// layout. The same prove call compares the combined Metal quotient with a
    /// full CPU recomputation over identical LDE columns and challenges, then
    /// verifies the resulting proof.
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn metal_random_access_quotient_matches_cpu_and_verifies() -> Result<()> {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        for (bits, copies) in [(3usize, 8usize), (4, 4), (6, 1)] {
            let vec_size = 1usize << bits;
            for copy in 0..copies {
                let index = copy % vec_size;
                let index_target = builder.constant(F::from_canonical_usize(index));
                let items = (0..vec_size)
                    .map(|i| {
                        builder.constant(F::from_canonical_usize(
                            1 + i + copy * vec_size + bits * 10_000,
                        ))
                    })
                    .collect::<Vec<_>>();
                let expected = items[index];
                let selected = builder.random_access(index_target, items);
                builder.connect(selected, expected);
                builder.register_public_input(selected);
            }
        }
        // A 2^16-point LDE retains both wire and constants/sigmas commitments
        // in shared Metal columns, exercising the production full-domain seam.
        while builder.num_gates() <= (1 << 12) {
            builder.add_gate(NoopGate, vec![]);
        }
        let data = builder.build::<Poseidon2GoldilocksConfig>();
        assert!(data.common.luts.is_empty());
        let mut advertised = data
            .common
            .gates
            .iter()
            .filter_map(|gate| match gate.0.u32_quotient_gate() {
                Some(U32QuotientGate::RandomAccess {
                    bits,
                    num_ops,
                    num_extra_constants,
                }) => Some((bits, num_ops, num_extra_constants)),
                _ => None,
            })
            .collect::<Vec<_>>();
        advertised.sort_unstable();
        assert_eq!(advertised, vec![(3, 8, 0), (4, 4, 2), (6, 1, 2)]);

        let before = gpu_poseidon_quotient_stats();
        COMPARE_GPU_QUOTIENT.store(true, Ordering::SeqCst);
        let proof = data.prove(PartialWitness::new());
        COMPARE_GPU_QUOTIENT.store(false, Ordering::SeqCst);
        let proof = proof?;
        let after = gpu_poseidon_quotient_stats();
        assert!(after.range_started > before.range_started);
        assert!(after.range_completed > before.range_completed);
        data.verify(proof)?;
        Ok(())
    }

    /// End-to-end retained-column differential for the four gates matched by
    /// type in the spec builder: `ExponentiationGate`, `EqualityGate`,
    /// `ReducingGate` and `ReducingExtensionGate`. The same prove call compares
    /// the combined Metal quotient with a full CPU recomputation over identical
    /// LDE columns and challenges, then verifies the resulting proof.
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn metal_native_gate_quotient_matches_cpu_and_verifies() -> Result<()> {
        // The ranked circuit shape: 136 wires / 80 routed / 2 constants, with
        // the equality-gate optimization flag that production sets.
        let config = CircuitConfig {
            num_wires: 136,
            optimization_flags: 1 << 3,
            ..CircuitConfig::standard_recursion_config()
        };
        assert!(config.equality_gate_enable());
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // EqualityGate: `is_equal` packs these into 22-operation gates.
        for i in 0..48usize {
            let x = builder.constant(F::from_canonical_usize(i));
            let y = builder.constant(F::from_canonical_usize(i % 5));
            let equal = builder.is_equal(x, y);
            builder.register_public_input(equal.target);
        }

        // ExponentiationGate: one gate per `exp_from_bits` call.
        for round in 0..3usize {
            let base = builder.constant(F::from_canonical_usize(3 + round));
            let bits = (0..12)
                .map(|bit| builder.constant_bool((round + bit) % 3 == 0))
                .collect::<Vec<_>>();
            let power = builder.exp_from_bits(base, bits.iter());
            builder.register_public_input(power);
        }

        // ReducingGate (base coefficients) and ReducingExtensionGate
        // (extension coefficients), both well past the arithmetic-gate
        // shortcut so the dedicated gates are emitted.
        let alpha = builder.constant_extension(
            <F as Extendable<D>>::Extension::from_canonical_u64(0x1234_5678_9abc),
        );
        let base_terms = (0..96usize)
            .map(|i| builder.constant(F::from_canonical_usize(7 * i + 1)))
            .collect::<Vec<_>>();
        let reduced_base = ReducingFactorTarget::new(alpha).reduce_base(&base_terms, &mut builder);
        for limb in reduced_base.0 {
            builder.register_public_input(limb);
        }
        let ext_terms = (0..96usize)
            .map(|i| {
                builder.constant_extension(<F as Extendable<D>>::Extension::from_canonical_u64(
                    (11 * i + 5) as u64,
                ))
            })
            .collect::<Vec<_>>();
        let reduced_ext = ReducingFactorTarget::new(alpha).reduce(&ext_terms, &mut builder);
        for limb in reduced_ext.0 {
            builder.register_public_input(limb);
        }

        // A 2^16-point LDE retains both wire and constants/sigmas commitments
        // in shared Metal columns, exercising the production full-domain seam.
        while builder.num_gates() <= (1 << 12) {
            builder.add_gate(NoopGate, vec![]);
        }
        let data = builder.build::<Poseidon2GoldilocksConfig>();
        assert!(data.common.luts.is_empty());
        assert_eq!(data.common.num_lookup_selectors, 0);

        // All four gate families must actually be present, or the differential
        // below would be vacuous for them.
        let mut present = [false; 4];
        for gate in &data.common.gates {
            let any = gate.0.as_any();
            present[0] |= any
                .downcast_ref::<crate::gates::exponentiation::ExponentiationGate<F, D>>()
                .is_some();
            present[1] |= any
                .downcast_ref::<crate::gates::equality_base::EqualityGate>()
                .is_some();
            present[2] |= any
                .downcast_ref::<crate::gates::reducing::ReducingGate<D>>()
                .is_some();
            present[3] |= any
                .downcast_ref::<crate::gates::reducing_extension::ReducingExtensionGate<D>>()
                .is_some();
        }
        assert_eq!(
            present,
            [true; 4],
            "circuit must contain all four natively matched gates"
        );

        let before = gpu_poseidon_quotient_stats();
        COMPARE_GPU_QUOTIENT.store(true, Ordering::SeqCst);
        let proof = data.prove(PartialWitness::new());
        COMPARE_GPU_QUOTIENT.store(false, Ordering::SeqCst);
        let proof = proof?;
        let after = gpu_poseidon_quotient_stats();
        assert!(after.range_started > before.range_started);
        assert!(after.range_completed > before.range_completed);
        data.verify(proof)?;
        Ok(())
    }

    /// Layout seam: `PolyMajor` output is exactly the transpose of
    /// `PointMajor` output, element for element (raw u64 compare).
    #[test]
    fn fill_lde_batch_layouts_agree() {
        let (data, _) = small_circuit();
        let commitment = &data.prover_only.constants_sigmas_commitment;
        let range = data.common.sigmas_range();
        let w = range.len();
        let indices = [0usize, 1, 2, 5, 7, 11, 30];
        let n = indices.len();
        let mut point_major = Vec::new();
        let mut poly_major = Vec::new();
        commitment.fill_lde_batch(
            &indices,
            2,
            range.clone(),
            BatchLayout::PointMajor,
            &mut point_major,
        );
        commitment.fill_lde_batch(&indices, 2, range, BatchLayout::PolyMajor, &mut poly_major);
        assert_eq!(point_major.len(), n * w);
        assert_eq!(poly_major.len(), n * w);
        for k in 0..n {
            for c in 0..w {
                assert_eq!(point_major[k * w + c].0, poly_major[c * n + k].0);
            }
        }
    }

    /// A contiguous PolyMajor gather must produce the same column slices as
    /// the generic indexed gather. This catches off-by-one source ranges and
    /// accidental point-major writes in the quotient fast path.
    #[test]
    fn contiguous_lde_batch_matches_indexed_gather() {
        let (data, _) = small_circuit();
        let commitment = &data.prover_only.constants_sigmas_commitment;
        let range = data.common.sigmas_range();
        let indices = [3usize, 4, 5, 6, 7, 8, 9];
        let mut indexed = Vec::new();
        let mut contiguous = Vec::new();

        commitment.fill_lde_batch(
            &indices,
            1,
            range.clone(),
            BatchLayout::PolyMajor,
            &mut indexed,
        );
        commitment.fill_lde_batch_contiguous(indices[0], indices.len(), range, &mut contiguous);

        assert_eq!(contiguous, indexed);
    }

    /// Scratch reuse: `fill_lde_batch` writes every cell of `out` before any
    /// is read, so dropping the zero-fill of an already correctly sized buffer
    /// must be invisible. A poisoned reused buffer has to gather exactly what
    /// a freshly allocated one does, for both layouts, across the full batch
    /// and the short final batch (which shrinks the buffer).
    #[test]
    fn fill_lde_batch_overwrites_dirty_scratch() {
        let (data, _) = small_circuit();
        let commitment = &data.prover_only.constants_sigmas_commitment;
        let range = data.common.sigmas_range();
        let indices = [0usize, 1, 2, 5, 7, 11, 30];
        let poison = F::from_canonical_u64(0x1234_5678_9abc_def0);
        for layout in [BatchLayout::PointMajor, BatchLayout::PolyMajor] {
            // One buffer reused across a full batch then a short one, exactly
            // as the quotient loop's `for_each_init` scratch is.
            let mut scratch = vec![poison; 3];
            for n in [indices.len(), 3, 3] {
                let mut fresh = Vec::new();
                commitment.fill_lde_batch(&indices[..n], 2, range.clone(), layout, &mut fresh);
                commitment.fill_lde_batch(&indices[..n], 2, range.clone(), layout, &mut scratch);
                assert_eq!(scratch.len(), fresh.len());
                for (actual, expected) in scratch.iter().zip(&fresh) {
                    assert_eq!(actual.0, expected.0);
                }
                // Poison every cell so the next iteration starts from a dirty
                // buffer of the right length (the reuse case being deleted).
                scratch.fill(poison);
            }
        }
    }

    /// C1/C2: cached tables must be bit-identical to direct computation, on
    /// both the miss and hit paths.
    #[test]
    fn precomputed_tables_match_direct() {
        for n_log in [1usize, 4, 9] {
            assert_eq!(
                *precomputed::two_adic_subgroup::<F>(n_log),
                F::two_adic_subgroup(n_log)
            );
            assert_eq!(
                *precomputed::two_adic_subgroup::<F>(n_log),
                F::two_adic_subgroup(n_log)
            );
        }
        for degree in [8usize, 64, 512] {
            let direct: Vec<F> = F::coset_shift().powers().take(degree).collect();
            assert_eq!(*precomputed::coset_shift_powers::<F>(degree), direct);
            assert_eq!(*precomputed::coset_shift_powers::<F>(degree), direct);
        }
    }

    /// D1: `ONE * a == a` bitwise for Goldilocks, canonical or not, so peeling
    /// the first factor of each chunk product into a direct assignment is
    /// value-exact.
    #[test]
    fn mul_by_one_is_bitwise_identity() {
        let order = GoldilocksField::ORDER;
        for raw in [0u64, 1, 1234567, order - 1, order, order + 12345, u64::MAX] {
            let x = GoldilocksField::from_noncanonical_u64(raw);
            assert_eq!((GoldilocksField::ONE * x).0, x.0);
        }
    }
}

/// Process-global cache of the `L_0(x)` denominator inverses `(n * (x - 1))^-1` consumed by
/// `ZeroPolyOnCoset::eval_l_0` in the quotient pass: one entry per LDE point `x = g * w^i`,
/// `2^(degree_bits + quotient_degree_bits)` per circuit shape. The values depend only on
/// `(degree_bits, quotient_degree_bits)` and the field's coset shift, so they are built once
/// per process and shared across proofs, mirroring the precomputed-table style of
/// `field::fft::fft_root_table`.
#[cfg(feature = "std")]
mod l_0_table_cache {
    use core::any::{Any, TypeId};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use plonky2_maybe_rayon::*;

    use crate::field::types::Field;

    /// Keyed by field type and `(degree_bits, quotient_degree_bits)`; the coset shift is a
    /// constant of the field type.
    static CACHE: OnceLock<Mutex<HashMap<(TypeId, usize, usize), Arc<dyn Any + Send + Sync>>>> =
        OnceLock::new();

    /// Builds the table with, per entry, exactly the operations of the uncached
    /// `eval_l_0(i, g * w^i)` path: `x = g * w^i` from the same `two_adic_subgroup` points the
    /// prover feeds it, then `(n * (x - ONE)).inverse()` — the same inverse of the same
    /// product, so every entry is bit-identical to the value it replaces. Entries are
    /// independent, so the parallel map changes nothing.
    fn build<F: Field>(degree_bits: usize, quotient_degree_bits: usize) -> Vec<F> {
        let n = F::from_canonical_usize(1 << degree_bits);
        F::two_adic_subgroup(degree_bits + quotient_degree_bits)
            .into_par_iter()
            .map(|x| (n * (F::coset_shift() * x - F::ONE)).inverse())
            .collect()
    }

    pub(super) fn l_0_denominator_inverses<F: Field>(
        degree_bits: usize,
        quotient_degree_bits: usize,
    ) -> Arc<Vec<F>> {
        let key = (TypeId::of::<F>(), degree_bits, quotient_degree_bits);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(entry) = cache.lock().unwrap().get(&key) {
            return Arc::clone(entry).downcast::<Vec<F>>().unwrap();
        }
        // Built outside the lock so a slow build never serializes other keys; concurrent
        // builders of the same key produce identical tables and the first insert wins.
        let table: Arc<Vec<F>> = Arc::new(build::<F>(degree_bits, quotient_degree_bits));
        let mut guard = cache.lock().unwrap();
        let entry = guard
            .entry(key)
            .or_insert_with(|| table as Arc<dyn Any + Send + Sync>);
        Arc::clone(entry).downcast::<Vec<F>>().unwrap()
    }
}

#[cfg(test)]
mod flat_chunk_products_tests {
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::{Field, PrimeField64};

    use crate::util::partial_products::quotient_chunk_products_into;

    use super::divide_chunk_products;

    type F = GoldilocksField;

    #[test]
    fn chunk_before_inversion_matches_individual_ratios() {
        for (width, chunk_size) in [(1, 2), (7, 3), (8, 8), (9, 8), (79, 8), (80, 8), (81, 8)] {
            let numerators = (0..width)
                .map(|i| F::from_canonical_usize(17 * i + 3))
                .collect::<Vec<_>>();
            let denominators = (0..width)
                .map(|i| F::from_canonical_usize(29 * i + 5))
                .collect::<Vec<_>>();

            let expected = numerators
                .chunks(chunk_size)
                .zip(denominators.chunks(chunk_size))
                .map(|(ns, ds)| {
                    ns.iter()
                        .zip(ds)
                        .map(|(&n, &d)| n * d.inverse())
                        .product::<F>()
                })
                .collect::<Vec<_>>();
            let mut actual = numerators
                .chunks(chunk_size)
                .map(|chunk| chunk.iter().copied().product())
                .collect::<Vec<_>>();
            let denominator_products = denominators
                .chunks(chunk_size)
                .map(|chunk| chunk.iter().copied().product())
                .collect::<Vec<_>>();
            let mut scratch = vec![F::ONE; width + 3];

            divide_chunk_products(&mut actual, &denominator_products, &mut scratch);
            assert_eq!(actual, expected, "width={width}, chunk_size={chunk_size}");
            assert_eq!(scratch.len(), actual.len());
        }
    }

    /// Deterministic, mostly-noncanonical values so raw-representation comparisons are
    /// meaningful.
    fn noncanonical_vec(len: usize, seed: u64) -> Vec<F> {
        (0..len as u64)
            .map(|i| {
                F::from_noncanonical_u64(
                    u64::MAX - seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(3 * i),
                )
            })
            .collect()
    }

    fn raw(values: &[F]) -> Vec<u64> {
        values.iter().map(|x| x.to_noncanonical_u64()).collect()
    }

    /// The Z accumulation exactly as `wires_permutation_partial_products_and_zs` performs it,
    /// over one point's chunk products.
    fn accumulate_point(columns: &mut [Vec<F>], z_x: &mut F, chunk_products: &[F]) {
        let num_prods = columns.len() - 1;
        let mut acc = *z_x;
        for (k, &quotient_chunk_product) in chunk_products.iter().enumerate() {
            acc *= quotient_chunk_product;
            if k == num_prods {
                columns[k].push(*z_x);
                *z_x = acc;
            } else {
                columns[k].push(acc);
            }
        }
    }

    /// Differential test for the flat chunk-products refactor: the legacy pipeline (a fresh
    /// per-point `collect()` of chunk products into a `Vec<Vec<F>>`, then the Z chain over
    /// those rows) against the shipping pipeline (`quotient_chunk_products_into` writing
    /// batch-aligned slices of one flat buffer, then the Z chain over `chunks_exact`). Every
    /// column entry must match in raw representation. Uses the production shape (80 routed
    /// wires, quotient degree 8) and point counts spanning partial and multiple inversion
    /// batches.
    #[test]
    fn flat_chunk_products_and_z_chain_match_legacy() {
        const INV_BATCH: usize = 128;
        let num_routed_wires = 80usize;
        let degree = 8usize;
        let num_chunks = num_routed_wires.div_ceil(degree);
        let num_prods = num_chunks - 1;

        for &n_points in &[1usize, 5, 128, 300] {
            let points: Vec<Vec<F>> = (0..n_points)
                .map(|i| noncanonical_vec(num_routed_wires, i as u64 + 1))
                .collect();

            // Legacy pipeline.
            let legacy_products: Vec<Vec<F>> = points
                .iter()
                .map(|quotient_values| {
                    quotient_values
                        .chunks(degree)
                        .map(|chunk| chunk.iter().copied().product())
                        .collect()
                })
                .collect();
            let mut legacy_columns: Vec<Vec<F>> = (0..num_chunks)
                .map(|_| Vec::with_capacity(n_points))
                .collect();
            let mut z_x = F::ONE;
            for chunk_products in &legacy_products {
                assert_eq!(chunk_products.len(), num_chunks);
                accumulate_point(&mut legacy_columns, &mut z_x, chunk_products);
            }

            // Shipping pipeline: batch-aligned writes into one flat buffer, exactly as the
            // prover slices it.
            let mut flat = vec![F::ZERO; n_points * num_chunks];
            for (xs, out_chunk) in points
                .chunks(INV_BATCH)
                .zip(flat.chunks_mut(INV_BATCH * num_chunks))
            {
                for (t, quotient_values) in xs.iter().enumerate() {
                    quotient_chunk_products_into(
                        quotient_values,
                        degree,
                        &mut out_chunk[t * num_chunks..(t + 1) * num_chunks],
                    );
                }
            }
            let mut flat_columns: Vec<Vec<F>> = (0..num_chunks)
                .map(|_| Vec::with_capacity(n_points))
                .collect();
            let mut z_x = F::ONE;
            for chunk_products in flat.chunks_exact(num_chunks) {
                accumulate_point(&mut flat_columns, &mut z_x, chunk_products);
            }

            for (k, (flat_column, legacy_column)) in
                flat_columns.iter().zip(&legacy_columns).enumerate()
            {
                assert_eq!(
                    raw(flat_column),
                    raw(legacy_column),
                    "column {k} mismatch for {n_points} points"
                );
            }
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod l_0_table_tests {
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::Field;
    use plonky2_field::zero_poly_coset::ZeroPolyOnCoset;

    use super::l_0_table_cache::l_0_denominator_inverses;

    type F = GoldilocksField;

    const COMBOS: [(usize, usize); 5] = [(1, 1), (3, 2), (4, 3), (6, 2), (8, 3)];

    /// Every cached entry must equal the legacy per-point computation
    /// `(n * (g * w^i - 1)).inverse()` bit-for-bit (raw u64 representation, not just field
    /// value), for several (degree_bits, quotient_degree_bits) combos.
    #[test]
    fn table_entries_match_legacy_per_point_inversion() {
        for (degree_bits, quotient_degree_bits) in COMBOS {
            let table = l_0_denominator_inverses::<F>(degree_bits, quotient_degree_bits);
            let points = F::two_adic_subgroup(degree_bits + quotient_degree_bits);
            assert_eq!(table.len(), points.len());
            let n = F::from_canonical_usize(1 << degree_bits);
            for (i, &point) in points.iter().enumerate() {
                // The prover's shifted point, computed exactly as `compute_quotient_polys`
                // computes `shifted_xs`.
                let x = F::coset_shift() * point;
                let legacy = (n * (x - F::ONE)).inverse();
                assert_eq!(
                    table[i].0, legacy.0,
                    "entry {i} of table ({degree_bits}, {quotient_degree_bits})"
                );
            }
        }
    }

    /// `eval_l_0` with the table attached must return raw-identical values to the uncached
    /// path at every LDE point.
    #[test]
    fn eval_l_0_with_table_matches_uncached() {
        for (degree_bits, quotient_degree_bits) in COMBOS {
            let plain = ZeroPolyOnCoset::<F>::new(degree_bits, quotient_degree_bits);
            let cached = ZeroPolyOnCoset::<F>::new(degree_bits, quotient_degree_bits)
                .with_l_0_denominator_inverses(l_0_denominator_inverses::<F>(
                    degree_bits,
                    quotient_degree_bits,
                ));
            for (i, point) in F::two_adic_subgroup(degree_bits + quotient_degree_bits)
                .into_iter()
                .enumerate()
            {
                let x = F::coset_shift() * point;
                assert_eq!(
                    cached.eval_l_0(i, x).0,
                    plain.eval_l_0(i, x).0,
                    "eval_l_0({i}) for ({degree_bits}, {quotient_degree_bits})"
                );
            }
        }
    }
}
