#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::{unflatten, Extendable, FieldExtension};
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::oracle::coset_fft_zero_tail;
use crate::fri::proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use crate::fri::{FriConfig, FriParams};
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::merkle_tree::{ColumnStore, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::reduce_with_powers;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits};

/// Builds a FRI proof.
pub fn fri_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    // Coefficients of the polynomial on which the LDT is performed. Only the first `1/rate` coefficients are non-zero.
    lde_polynomial_coeffs: PolynomialCoeffs<F::Extension>,
    // Evaluation of the polynomial on the large domain.
    lde_polynomial_values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
    timing: &mut TimingTree,
) -> FriProof<F, C::Hasher, D> {
    let n = lde_polynomial_values.len();
    assert_eq!(lde_polynomial_coeffs.len(), n);

    // Commit phase
    let (trees, final_coeffs) = timed!(
        timing,
        "fold codewords in the commitment phase",
        fri_committed_trees::<F, C, D>(
            lde_polynomial_coeffs,
            lde_polynomial_values,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
        )
    );

    // PoW phase
    let pow_witness = timed!(
        timing,
        "find proof-of-work witness",
        fri_proof_of_work::<F, C, D>(challenger, &fri_params.config)
    );

    // Query phase
    let query_round_proofs =
        fri_prover_query_rounds::<F, C, D>(initial_merkle_trees, &trees, challenger, n, fri_params);

    FriProof {
        commit_phase_merkle_caps: trees.iter().map(|t| t.cap.clone()).collect(),
        query_round_proofs,
        final_poly: final_coeffs,
        pow_witness,
    }
}

pub(crate) type FriCommitedTrees<F, C, const D: usize> = (
    Vec<MerkleTree<F, <C as GenericConfig<D>>::Hasher>>,
    PolynomialCoeffs<<F as Extendable<D>>::Extension>,
);

pub fn final_poly_coeff_len(mut degree_bits: usize, reduction_arity_bits: &Vec<usize>) -> usize {
    for arity_bits in reduction_arity_bits {
        degree_bits -= *arity_bits;
    }
    1 << degree_bits
}

/// Bit-reversal + flatten in one gather pass: output leaf `i` is the base-field
/// limb array of `values[reverse_bits(i, log2(values.len()))]`, so the returned
/// flat buffer is the bit-reversed codeword laid out row-major, ready for
/// [`MerkleTree::new_flat`].
///
/// The gather is bandwidth- and latency-bound rather than arithmetic-bound:
/// `reverse_bits` scatters consecutive outputs across the whole codeword, so
/// essentially every read is a cache miss and a single thread can only keep a
/// handful of them in flight. Splitting the *output* range into blocks lets one
/// worker per core drive its own independent miss stream. Block `b` owns
/// outputs `b * FLATTEN_BLOCK .. (b + 1) * FLATTEN_BLOCK`, a partition of
/// `0..n`, so every slot is written exactly once and the source is only read —
/// the result is index-for-index identical to the serial fill.
fn bitrev_flatten<F: RichField + Extendable<D>, const D: usize>(values: &[F::Extension]) -> Vec<F> {
    const FLATTEN_BLOCK: usize = 1 << 10;

    let n = values.len();
    let log_n = log2_strict(n);
    let mut flat: Vec<F> = Vec::with_capacity(n * D);
    {
        let spare = &mut flat.spare_capacity_mut()[..n * D];
        spare
            .par_chunks_mut(FLATTEN_BLOCK * D)
            .enumerate()
            .for_each(|(block, out)| {
                let base = block * FLATTEN_BLOCK;
                for (j, slot) in out.chunks_exact_mut(D).enumerate() {
                    let limbs = values[reverse_bits(base + j, log_n)].to_basefield_array();
                    for k in 0..D {
                        slot[k].write(limbs[k]);
                    }
                }
            });
    }
    // SAFETY: the loop above wrote every one of the `n * D` slots of spare
    // capacity exactly once, so the whole prefix is initialized.
    unsafe { flat.set_len(n * D) };
    flat
}

/// Fills the natural-order columns whose bit-reversed rows are exactly the
/// arity-sized leaves produced by [`bitrev_flatten`]. If `A = 2^arity_bits`
/// and `R = values.len() / A`, column `j * D + k` holds limb `k` of
/// `values[reverse_bits(j, arity_bits) * R + r]` at row `r`.
///
/// A column-backed Merkle tree reverses the `log2(R)` row bits when reading a
/// leaf. Thus leaf position `i * A + j` addresses
/// `reverse_bits(j, arity_bits) * R + reverse_bits(i, log2(R))`, equal to the
/// full-width `reverse_bits(i * A + j, log2(values.len()))` used by the flat
/// path. Grouping the `D` limb columns also decomposes each extension value
/// only once.
fn fill_fri_columns<F: RichField + Extendable<D>, const D: usize>(
    values: &[F::Extension],
    arity_bits: usize,
    columns: &mut ColumnStore<F>,
) -> bool {
    let arity = 1 << arity_bits;
    assert_eq!(values.len() % arity, 0);
    let rows = values.len() / arity;
    let Some(mut destinations) = columns.columns_mut() else {
        return false;
    };
    assert_eq!(destinations.len(), arity * D);
    assert!(destinations.iter().all(|column| column.len() == rows));

    destinations
        .par_chunks_mut(D)
        .enumerate()
        .for_each(|(j, limb_columns)| {
            let source_start = reverse_bits(j, arity_bits) * rows;
            for (r, value) in values[source_start..source_start + rows]
                .iter()
                .enumerate()
            {
                let limbs = value.to_basefield_array();
                for (column, limb) in limb_columns.iter_mut().zip(limbs) {
                    column[r] = limb;
                }
            }
        });
    true
}

/// Tries the retained shared-column Merkle route used by the Metal Poseidon2
/// backend. Every declined step returns `None`, allowing the caller to execute
/// the historical flat construction unchanged.
fn try_fri_column_merkle_tree<
    F: RichField + Extendable<D>,
    H: Hasher<F>,
    const D: usize,
>(
    values: &[F::Extension],
    arity_bits: usize,
    cap_height: usize,
) -> Option<MerkleTree<F, H>> {
    let arity = 1 << arity_bits;
    assert_eq!(values.len() % arity, 0);
    let rows = values.len() / arity;
    let mut columns = H::try_allocate_merkle_tree_columns(arity * D, rows, cap_height)?;
    if !fill_fri_columns::<F, D>(values, arity_bits, &mut columns) {
        return None;
    }
    let (level_digests, cap) = H::try_build_merkle_tree_column_store(&columns, cap_height)?;
    Some(MerkleTree::from_prebuilt_columns(
        columns,
        level_digests,
        cap,
    ))
}

fn fri_committed_trees<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    mut coeffs: PolynomialCoeffs<F::Extension>,
    mut values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
) -> FriCommitedTrees<F, C, D> {
    let mut trees = Vec::with_capacity(fri_params.reduction_arity_bits.len());

    let mut shift = F::MULTIPLICATIVE_GROUP_GENERATOR;
    let num_rounds = fri_params.reduction_arity_bits.len();
    for (round, arity_bits) in fri_params.reduction_arity_bits.iter().enumerate() {
        let arity = 1 << arity_bits;

        // GPU-qualified rounds are written directly into the retained shared
        // column store consumed by Metal. If allocation or the specialized
        // Merkle build declines, execute the historical flat path exactly.
        let tree = try_fri_column_merkle_tree::<F, C::Hasher, D>(
            &values.values,
            *arity_bits,
            fri_params.config.cap_height,
        )
        .unwrap_or_else(|| {
            let flat_values = bitrev_flatten::<F, D>(&values.values);
            MerkleTree::<F, C::Hasher>::new_flat(
                flat_values,
                arity * D,
                fri_params.config.cap_height,
            )
        });

        challenger.observe_cap(&tree.cap);
        trees.push(tree);

        let beta = challenger.get_extension_challenge::<D>();
        // P(x) = sum_{i<r} x^i * P_i(x^r) becomes sum_{i<r} beta^i * P_i(x).
        // Only `1/2^rate_bits` of the coefficients are nonzero every round
        // (the zero-tail invariant asserted by the final truncation), and the
        // Horner fold of an all-zero chunk is exactly zero, so fold only the
        // live prefix and extend with the zeros those chunks would produce.
        let n_chunks = coeffs.coeffs.len() / arity;
        let support = coeffs.coeffs.len() >> fri_params.config.rate_bits;
        let live_chunks = support.div_ceil(arity).min(n_chunks);
        let beta_powers_16 = if arity == 16 {
            let mut powers = [F::Extension::ONE; 16];
            for i in 1..16 {
                powers[i] = powers[i - 1] * beta;
            }
            Some(powers)
        } else {
            None
        };
        let mut folded = coeffs.coeffs[..live_chunks * arity]
            .par_chunks_exact(arity)
            .map(|chunk| match &beta_powers_16 {
                Some(beta_powers) => {
                    let terms: &[F::Extension; 16] = chunk
                        .try_into()
                        .expect("arity-16 FRI chunk must contain 16 terms");
                    F::fri_fold_arity16(terms, beta, beta_powers)
                }
                None => reduce_with_powers(chunk, beta),
            })
            .collect::<Vec<_>>();
        // The historical `resize(n_chunks, ZERO)` zero-filled the whole dead
        // tail. Zeros are actually *read as values* only where the next
        // round's exact-`arity` chunking can reach past the live support —
        // at most `arity_next - 1` slots past `live` — because every other
        // tail consumer (the zero-tail coset FFT and the final truncation +
        // transcript observation) reads only the live prefix. Extend the
        // length without storing the rest.
        let live = folded.len();
        folded.reserve(n_chunks - live);
        // SAFETY: length equals capacity; the slots beyond `pad_end` are
        // never read (see above), and `F::Extension` is plain data.
        unsafe { folded.set_len(n_chunks) };
        let pad_end = if round + 1 < num_rounds {
            n_chunks.min(live + (1 << fri_params.reduction_arity_bits[round + 1]))
        } else {
            live
        };
        for value in folded[live..pad_end].iter_mut() {
            *value = F::Extension::ZERO;
        }
        coeffs = PolynomialCoeffs::new(folded);
        shift = shift.exp_u64(arity as u64);
        // Chunk-wise folding preserves the zero tail: the coefficient vector
        // keeps `1/2^rate_bits` support every round (asserted by the
        // truncation below), so the FFT's zero-run shortcut always applies.
        // The coefficients from `live_chunks` on are the zeros the `resize`
        // above just wrote, and `shift^i * 0 == 0`, so the coset scaling is
        // dead work over that tail: scale only the folded prefix.
        //
        // `values` is read by exactly one thing: the *next* round's leaf
        // gather at the top of this loop. After the final round it is dropped
        // unread — everything below this loop uses only `coeffs` — so the
        // last round's transform is entirely dead work. Skip it.
        if round + 1 < num_rounds {
            values = coset_fft_zero_tail(
                &coeffs,
                shift.into(),
                live_chunks,
                Some(fri_params.config.rate_bits),
                None,
            );
        }
    }

    // When verifying this proof in a circuit with a different number of query steps,
    // we need the challenger to stay in sync with the verifier. Therefore, the challenger
    // must observe the additional hash caps and generate dummy challenges.
    if let Some(step_count) = max_num_query_steps {
        let cap_len = (1 << fri_params.config.cap_height) * NUM_HASH_OUT_ELTS;
        let zero_cap = vec![F::ZERO; cap_len];
        for _ in fri_params.reduction_arity_bits.len()..step_count {
            challenger.observe_elements(&zero_cap);
            challenger.get_extension_challenge::<D>();
        }
    }

    // The coefficients being removed here should always be zero.
    coeffs
        .coeffs
        .truncate(coeffs.len() >> fri_params.config.rate_bits);

    challenger.observe_extension_elements(&coeffs.coeffs);
    // When verifying this proof in a circuit with a different final polynomial length,
    // the challenger needs to observe the full length of the final polynomial.
    if let Some(len) = final_poly_coeff_len {
        let current_len = coeffs.coeffs.len();
        for _ in current_len..len {
            challenger.observe_extension_element(&F::Extension::ZERO);
        }
    }

    (trees, coeffs)
}

/// Performs the proof-of-work (a.k.a. grinding) step of the FRI protocol. Returns the PoW witness.
pub(crate) fn fri_proof_of_work<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    challenger: &mut Challenger<F, C::Hasher>,
    config: &FriConfig,
) -> F {
    let min_leading_zeros = config.proof_of_work_bits + (64 - F::order().bits()) as u32;

    // The easiest implementation would be repeatedly clone our Challenger. With each clone, we'd
    // observe an incrementing PoW witness, then get the PoW response. If it contained sufficient
    // leading zeros, we'd end the search, and store this clone as our new challenger.
    //
    // However, performance is critical here. We want to avoid cloning Challenger, particularly
    // since it stores vectors, which means allocations. We'd like a more compact state to clone.
    //
    // We know that a duplex will be performed right after we send the PoW witness, so we can ignore
    // any output_buffer, which will be invalidated. We also know
    // input_buffer.len() < H::Permutation::WIDTH, an invariant of Challenger.
    //
    // We separate the duplex operation into two steps, one which can be performed now, and the
    // other which depends on the PoW witness candidate. The first step is the overwrite our sponge
    // state with any inputs (excluding the PoW witness candidate). The second step is to overwrite
    // one more element of our sponge state with the candidate, then apply the permutation,
    // obtaining our duplex's post-state which contains the PoW response.
    let mut duplex_intermediate_state = challenger.sponge_state;
    let witness_input_pos = challenger.input_buffer.len();
    duplex_intermediate_state.set_from_iter(challenger.input_buffer.clone(), 0);

    let pow_witness = (0..=F::NEG_ONE.to_canonical_u64())
        .into_par_iter()
        .find_any(|&candidate| {
            let mut duplex_state = duplex_intermediate_state;
            duplex_state.set_elt(F::from_canonical_u64(candidate), witness_input_pos);
            duplex_state.permute();
            let pow_response = duplex_state.squeeze().iter().last().unwrap();
            let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
            leading_zeros >= min_leading_zeros
        })
        .map(F::from_canonical_u64)
        .expect("Proof of work failed. This is highly unlikely!");

    // Recompute pow_response using our normal Challenger code, and make sure it matches.
    challenger.observe_element(pow_witness);
    let pow_response = challenger.get_challenge();
    let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
    assert!(leading_zeros >= min_leading_zeros);
    pow_witness
}

fn fri_prover_query_rounds<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    trees: &[MerkleTree<F, C::Hasher>],
    challenger: &mut Challenger<F, C::Hasher>,
    n: usize,
    fri_params: &FriParams,
) -> Vec<FriQueryRound<F, C::Hasher, D>> {
    challenger
        .get_n_challenges(fri_params.config.num_query_rounds)
        .into_par_iter()
        .map(|rand| {
            let x_index = rand.to_canonical_u64() as usize % n;
            fri_prover_query_round::<F, C, D>(initial_merkle_trees, trees, x_index, fri_params)
        })
        .collect()
}

fn fri_prover_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    trees: &[MerkleTree<F, C::Hasher>],
    mut x_index: usize,
    fri_params: &FriParams,
) -> FriQueryRound<F, C::Hasher, D> {
    let mut query_steps = Vec::new();
    let initial_proof = initial_merkle_trees
        .iter()
        .map(|t| (t.leaf_vec(x_index), t.prove(x_index)))
        .collect::<Vec<_>>();
    for (i, tree) in trees.iter().enumerate() {
        let arity_bits = fri_params.reduction_arity_bits[i];
        let leaf = tree.leaf_vec(x_index >> arity_bits);
        let evals = unflatten(&leaf);
        let merkle_proof = tree.prove(x_index >> arity_bits);

        query_steps.push(FriQueryStep {
            evals,
            merkle_proof,
        });

        x_index >>= arity_bits;
    }
    FriQueryRound {
        initial_trees_proof: FriInitialTreeProof {
            evals_proofs: initial_proof,
        },
        steps: query_steps,
    }
}

#[cfg(test)]
mod tests {
    use plonky2_field::types::Sample;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::poseidon::PoseidonHash;
    use crate::plonk::config::GenericHashOut;

    /// `bitrev_flatten` must be raw-`u64`-identical to the serial
    /// gather-and-extend loop it replaced, for every leaf and every limb.
    #[test]
    fn bitrev_flatten_matches_serial_gather() {
        const D: usize = 2;
        type F = GoldilocksField;
        type FE = <F as Extendable<D>>::Extension;

        // Sizes on both sides of the `FLATTEN_BLOCK = 1 << 10` grain: below it
        // (a single partial chunk), exactly on it, and several blocks past it.
        for log_n in [0usize, 1, 5, 10, 11, 13] {
            let n = 1usize << log_n;
            let values: Vec<FE> = (0..n).map(|_| FE::rand()).collect();

            // Reference: the original serial fill.
            let mut expected: Vec<F> = Vec::with_capacity(n * D);
            for i in 0..n {
                let x: [F; D] = values[reverse_bits(i, log_n)].to_basefield_array();
                expected.extend_from_slice(&x);
            }

            let actual = bitrev_flatten::<F, D>(&values);
            assert_eq!(actual.len(), expected.len(), "length for n = {n}");
            for (k, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_eq!(a.0, e.0, "limb {k} of {n}");
            }
        }
    }

    /// Direct FRI columns must expose the exact old row bytes and therefore
    /// produce identical caps, paths, and cap-derived transcript challenges.
    /// PoseidonHash has no specialized column backend, keeping this a pure CPU
    /// differential independent of Metal availability.
    #[test]
    fn fri_columns_match_flat_leaves_caps_proofs_and_transcript() {
        const D: usize = 2;
        type F = GoldilocksField;
        type FE = <F as Extendable<D>>::Extension;

        for (log_n, arity_bits, cap_height) in
            [(4usize, 1usize, 1usize), (6, 2, 2), (8, 4, 2)]
        {
            let n = 1 << log_n;
            let arity = 1 << arity_bits;
            let rows = n / arity;
            let values: Vec<FE> = (0..n).map(|_| FE::rand()).collect();

            let flat = bitrev_flatten::<F, D>(&values);
            let flat_tree = MerkleTree::<F, PoseidonHash>::new_flat(
                flat.clone(),
                arity * D,
                cap_height,
            );
            let mut columns = ColumnStore::Owned(vec![vec![F::ZERO; rows]; arity * D]);
            assert!(fill_fri_columns::<F, D>(
                &values,
                arity_bits,
                &mut columns
            ));
            let column_tree =
                MerkleTree::<F, PoseidonHash>::new_column_store(columns, cap_height);

            for leaf in 0..rows {
                let expected = &flat[leaf * arity * D..(leaf + 1) * arity * D];
                let actual = column_tree.leaf_vec(leaf);
                for (limb, (a, e)) in actual.iter().zip(expected).enumerate() {
                    assert_eq!(a.0, e.0, "leaf {leaf}, limb {limb}, arity {arity}");
                }

                let flat_proof = flat_tree.prove(leaf);
                let column_proof = column_tree.prove(leaf);
                assert_eq!(column_proof.siblings.len(), flat_proof.siblings.len());
                for (level, (a, e)) in column_proof
                    .siblings
                    .iter()
                    .zip(&flat_proof.siblings)
                    .enumerate()
                {
                    for (limb, (a, e)) in a.to_vec().iter().zip(e.to_vec()).enumerate() {
                        assert_eq!(
                            a.0, e.0,
                            "proof level {level}, limb {limb}, leaf {leaf}, arity {arity}"
                        );
                    }
                }
            }

            for (i, (a, e)) in column_tree
                .cap
                .flatten()
                .iter()
                .zip(flat_tree.cap.flatten())
                .enumerate()
            {
                assert_eq!(a.0, e.0, "cap limb {i}, arity {arity}");
            }

            let mut flat_challenger = Challenger::<F, PoseidonHash>::new();
            flat_challenger.observe_cap(&flat_tree.cap);
            let flat_beta = flat_challenger.get_extension_challenge::<D>();
            let mut column_challenger = Challenger::<F, PoseidonHash>::new();
            column_challenger.observe_cap(&column_tree.cap);
            let column_beta = column_challenger.get_extension_challenge::<D>();
            let column_beta_limbs: [F; D] = column_beta.to_basefield_array();
            let flat_beta_limbs: [F; D] = flat_beta.to_basefield_array();
            for (limb, (a, e)) in column_beta_limbs.iter().zip(flat_beta_limbs).enumerate() {
                assert_eq!(a.0, e.0, "transcript limb {limb}, arity {arity}");
            }
        }
    }
}
