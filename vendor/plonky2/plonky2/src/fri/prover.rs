#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::{unflatten, Extendable, FieldExtension};
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use crate::fri::{FriConfig, FriParams};
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::merkle_tree::{ColumnStore, MerkleTree};
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::plonk::plonk_common::reduce_with_powers;
use crate::timed;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits};

/// Where the LDE values of the FRI codeword live. The commit phase reads them
/// once (the bit-reversed flat gather), so a specialized backend's retained
/// columns (e.g. GPU-shared NTT output) can be consumed directly instead of
/// materializing an extension-value buffer first.
#[derive(Debug)]
pub enum FriLdeSource<F: RichField + Extendable<D>, const D: usize> {
    /// Extension values in natural order.
    Owned(PolynomialValues<F::Extension>),
    /// One natural-order base column per extension limb.
    Columns(ColumnStore<F>),
}

impl<F: RichField + Extendable<D>, const D: usize> FriLdeSource<F, D> {
    pub(crate) fn len(&self) -> usize {
        match self {
            FriLdeSource::Owned(values) => values.len(),
            FriLdeSource::Columns(columns) => columns.num_rows(),
        }
    }

    #[inline]
    pub(crate) fn extension_at(&self, i: usize) -> F::Extension {
        match self {
            FriLdeSource::Owned(values) => values.values[i],
            FriLdeSource::Columns(columns) => {
                let mut arr = [F::ZERO; D];
                for (b, elt) in arr.iter_mut().enumerate() {
                    *elt = columns.col(b)[i];
                }
                F::Extension::from_basefield_array(arr)
            }
        }
    }
}

/// Builds a FRI proof.
pub fn fri_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    // Coefficients of the polynomial on which the LDT is performed. Only the first `1/rate` coefficients are non-zero.
    lde_polynomial_coeffs: PolynomialCoeffs<F::Extension>,
    // Evaluation of the polynomial on the large domain.
    lde_polynomial_values: FriLdeSource<F, D>,
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

fn fri_committed_trees<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    mut coeffs: PolynomialCoeffs<F::Extension>,
    mut values: FriLdeSource<F, D>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
) -> FriCommitedTrees<F, C, D> {
    let mut trees = Vec::with_capacity(fri_params.reduction_arity_bits.len());

    let mut shift = F::MULTIPLICATIVE_GROUP_GENERATOR;
    for arity_bits in &fri_params.reduction_arity_bits {
        let arity = 1 << arity_bits;

        // Fused bit-reversal + flatten: one parallel gather pass writes the
        // flat leaf buffer directly (leaf `i` is the `arity`-chunk of the
        // bit-reversed codeword starting at `i * arity`), instead of a
        // random-access in-place permutation followed by a separate
        // flattening pass with a heap allocation per element. Output layout
        // is identical to the sequential gather (same per-element order), so
        // the tree and every subsequent artifact are unchanged.
        let n = values.len();
        let log_n = log2_strict(n);
        let mut flat_values = vec![F::ZERO; n * D];
        flat_values.par_chunks_mut(D).enumerate().for_each(|(i, chunk)| {
            let x = values.extension_at(reverse_bits(i, log_n));
            chunk.copy_from_slice(&x.to_basefield_array());
        });
        let tree = MerkleTree::<F, C::Hasher>::new_flat(
            flat_values,
            arity * D,
            fri_params.config.cap_height,
        );

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
        let mut folded = coeffs.coeffs[..live_chunks * arity]
            .par_chunks_exact(arity)
            .map(|chunk| reduce_with_powers(chunk, beta))
            .collect::<Vec<_>>();
        folded.resize(n_chunks, F::Extension::ZERO);
        coeffs = PolynomialCoeffs::new(folded);
        shift = shift.exp_u64(arity as u64);
        // Chunk-wise folding preserves the zero tail: the coefficient vector
        // keeps `1/2^rate_bits` support every round (asserted by the
        // truncation below), so the FFT's zero-run shortcut always applies.
        values = FriLdeSource::Owned(coeffs.coset_fft_with_options(
            shift.into(),
            Some(fri_params.config.rate_bits),
            None,
        ))
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
        let evals = unflatten(tree.get(x_index >> arity_bits));
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
    use std::time::Instant;

    use crate::field::extension::{Extendable, FieldExtension};
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::polynomial::PolynomialCoeffs;
    use crate::field::types::{Field, PrimeField64};
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::fri::prover::{FriLdeSource, fri_committed_trees, fri_proof_of_work, fri_prover_query_rounds};
    use crate::fri::{FriConfig, FriParams};
    use crate::hash::merkle_tree::MerkleTree;
    use crate::iop::challenger::Challenger;
    use crate::plonk::config::{GenericConfig, Hasher, Poseidon2GoldilocksConfig};
    use crate::util::timing::TimingTree;
    use plonky2_field::types::Sample;

    type F = GoldilocksField;
    type C = Poseidon2GoldilocksConfig;
    const D: usize = 2;

    fn ms(t: Instant) -> f64 {
        t.elapsed().as_secs_f64() * 1000.0
    }

    fn chain_step_params() -> (FriConfig, FriParams) {
        let config = FriConfig {
            rate_bits: 3,
            cap_height: 4,
            proof_of_work_bits: 16,
            reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
            num_query_rounds: 28,
        };
        let params = config.fri_params(14, false);
        (config, params)
    }

    /// The chain-step final FRI poly shape: degree 2^14 padded to the 2^17 LDE
    /// size with a zero tail (exactly what prove_openings produces).
    fn chain_step_final_poly() -> PolynomialCoeffs<<F as Extendable<D>>::Extension> {
        let degree = 1usize << 14;
        let mut coeffs = PolynomialCoeffs::new(
            (0..degree)
                .map(|_| <F as Extendable<D>>::Extension::rand())
                .collect::<Vec<_>>(),
        );
        coeffs.coeffs.resize(degree << 3, <F as Extendable<D>>::Extension::ZERO);
        coeffs
    }

    /// Differential: the specialized backend's LDE must be bit-identical to
    /// the CPU coset FFT for the chain-step final-poly shape (this is what
    /// keeps every committed cap and hence the proof bytes unchanged).
    #[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn test_gpu_lde_bit_exact() {
        // The bench only routes mid-size GPU work while a serial proving
        // phase holds the GPU stream exclusively; mirror that here.
        crate::hash::poseidon2::set_exclusive_gpu_phase(true);
        let coeffs = chain_step_final_poly();
        let nonzero = coeffs.len() >> 3;
        let mut coeff_columns: Vec<Vec<F>> = vec![Vec::with_capacity(nonzero); D];
        for c in &coeffs.coeffs[..nonzero] {
            let arr: [_; D] = c.to_basefield_array();
            for (b, elt) in arr.into_iter().enumerate() {
                coeff_columns[b].push(elt);
            }
        }
        let coeff_refs: Vec<&[F]> = coeff_columns.iter().map(|c| c.as_slice()).collect();
        let columns = match <C as GenericConfig<D>>::Hasher::try_lde_from_coeffs(&coeff_refs, 3) {
            Some(columns) => columns,
            None => {
                eprintln!("GPU LDE unavailable; skipping");
                return;
            }
        };
        let cpu = coeffs.coset_fft_with_options(F::coset_shift().into(), Some(3), None);
        crate::hash::poseidon2::set_exclusive_gpu_phase(false);
        for i in 0..cpu.len() {
            let expected: [_; D] = cpu.values[i].to_basefield_array();
            for b in 0..D {
                assert_eq!(
                    columns.col(b)[i],
                    expected[b],
                    "LDE mismatch at {i} limb {b}"
                );
            }
        }
    }

    /// Differential: feeding the fold the same values as retained GPU columns
    /// vs. as an owned extension buffer must produce identical fold trees and
    /// final polynomial (hence identical proof bytes).
    #[test]
    fn test_fri_committed_trees_source_equivalence() {
        let (_, params) = chain_step_params();
        let coeffs = chain_step_final_poly();
        let lde = coeffs.coset_fft_with_options(F::coset_shift().into(), Some(3), None);

        let mut challenger_a = Challenger::<F, <C as GenericConfig<D>>::Hasher>::new();
        let mut challenger_b = Challenger::<F, <C as GenericConfig<D>>::Hasher>::new();
        let mut _timing = TimingTree::new("t", log::Level::Debug);

        let coeffs_b = coeffs.clone();
        let (trees_a, final_a) = fri_committed_trees::<F, C, D>(
            coeffs,
            FriLdeSource::Owned(lde.clone()),
            &mut challenger_a,
            &params,
            None,
            None,
        );

        // Same LDE values, rebuilt as natural-order base columns.
        let lde_len = lde.len();
        let mut value_columns: Vec<Vec<F>> = vec![Vec::with_capacity(lde_len); D];
        for v in &lde.values {
            let arr: [_; D] = v.to_basefield_array();
            for (b, elt) in arr.into_iter().enumerate() {
                value_columns[b].push(elt);
            }
        }
        let columns = crate::hash::merkle_tree::ColumnStore::Owned(value_columns);
        let (trees_b, final_b) = fri_committed_trees::<F, C, D>(
            coeffs_b,
            FriLdeSource::Columns(columns),
            &mut challenger_b,
            &params,
            None,
            None,
        );

        assert_eq!(trees_a.len(), trees_b.len());
        for (ta, tb) in trees_a.iter().zip(trees_b.iter()) {
            assert_eq!(ta.leaves, tb.leaves);
            assert_eq!(ta.digests, tb.digests);
            assert_eq!(ta.cap, tb.cap);
        }
        assert_eq!(final_a.coeffs, final_b.coeffs);
        // The challenger transcript (caps + final coeffs observed) must match too.
        assert_eq!(challenger_a.compact(), challenger_b.compact());
        assert_eq!(lde_len, trees_a[0].num_leaves << 4);
    }

    /// Determinism of the (now parallel) bit-reversal gather: two identical
    /// runs must produce byte-identical fold trees and transcripts. Also
    /// times the phases a serial chain step pays.
    #[test]
    fn test_fri_committed_trees_deterministic_and_timing() {
        let (config, params) = chain_step_params();
        let coeffs = chain_step_final_poly();

        let t = Instant::now();
        let lde = coeffs.coset_fft_with_options(F::coset_shift().into(), Some(3), None);
        eprintln!("final-poly LDE FFT (2^17 ext): {:.2} ms", ms(t));

        let run = |coeffs: PolynomialCoeffs<<F as Extendable<D>>::Extension>| {
            let mut challenger = Challenger::<F, <C as GenericConfig<D>>::Hasher>::new();
            let mut _timing = TimingTree::new("t", log::Level::Debug);
            let t = Instant::now();
            let (trees, final_coeffs) = fri_committed_trees::<F, C, D>(
                coeffs,
                FriLdeSource::Owned(lde.clone()),
                &mut challenger,
                &params,
                None,
                None,
            );
            let fold_ms = ms(t);
            (trees, final_coeffs, challenger, fold_ms)
        };

        let (trees_a, final_a, mut challenger_a, fold_ms) = run(coeffs.clone());
        eprintln!("fri_committed_trees (3 folds, parallel gather): {:.2} ms", fold_ms);
        let (trees_b, final_b, mut challenger_b, _) = run(coeffs.clone());

        for (ta, tb) in trees_a.iter().zip(trees_b.iter()) {
            assert_eq!(ta.leaves, tb.leaves);
            assert_eq!(ta.digests, tb.digests);
            assert_eq!(ta.cap, tb.cap);
        }
        assert_eq!(final_a.coeffs, final_b.coeffs);
        assert_eq!(challenger_a.compact(), challenger_b.compact());

        // PoW: parallel find_any still yields a valid witness.
        let mut challenger = Challenger::<F, <C as GenericConfig<D>>::Hasher>::new();
        let mut _timing = TimingTree::new("t", log::Level::Debug);
        let (trees, _) = fri_committed_trees::<F, C, D>(
            coeffs,
            FriLdeSource::Owned(lde),
            &mut challenger,
            &params,
            None,
            None,
        );
        let min_leading_zeros = config.proof_of_work_bits + (64 - F::order().bits()) as u32;
        // Verify the witness exactly as the protocol does, against a clone of
        // the pre-PoW challenger (the live challenger advances past the PoW).
        let mut pow_check = challenger.clone();
        let t = Instant::now();
        let pow = fri_proof_of_work::<F, C, D>(&mut challenger, &config);
        eprintln!("proof of work (16 bits): {:.2} ms", ms(t));
        pow_check.observe_element(pow);
        let response = pow_check.get_challenge();
        assert!(
            response.to_canonical_u64().leading_zeros() >= min_leading_zeros,
            "PoW witness must satisfy the leading-zero bound"
        );

        // Query rounds against 4 simulated initial trees + the 3 fold trees.
        let mut initial = Vec::new();
        for _ in 0..4 {
            let columns: Vec<Vec<F>> = (0..8)
                .map(|_| (0..(1usize << 17)).map(|_| F::rand()).collect())
                .collect();
            initial.push(
                MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new_columns(columns, 4),
            );
        }
        let t = Instant::now();
        let queries = fri_prover_query_rounds::<F, C, D>(
            &initial.iter().collect::<Vec<_>>(),
            &trees,
            &mut challenger,
            1usize << 17,
            &params,
        );
        eprintln!("query rounds (28, 4+3 trees): {:.2} ms", ms(t));
        assert_eq!(queries.len(), 28);
        for q in &queries {
            assert_eq!(q.steps.len(), 3);
        }
    }

    /// End-to-end roundtrip through the classic single-instance FRI path the
    /// bench uses (`PolynomialBatch::prove_openings` + `verify_fri_proof`),
    /// under both the GPU LDE backend (exclusive phase) and the CPU FFT
    /// fallback: both proofs must verify and be byte-identical.
    #[test]
    fn test_fri_roundtrip_gpu_cpu_identical() -> anyhow::Result<()> {
        use crate::fri::oracle::PolynomialBatch;
        use crate::fri::proof::{FriChallenges, FriProof};
        use crate::fri::structure::{
            FriBatchInfo, FriInstanceInfo, FriOpeningBatch, FriOpenings, FriOracleInfo,
            FriPolynomialInfo,
        };
        use crate::fri::verifier::verify_fri_proof;

        let k = 14;
        let mut timing = TimingTree::default();
        let reduction_arity_bits = vec![4, 4, 4];
        let fri_params = FriParams {
            config: FriConfig {
                rate_bits: 3,
                cap_height: 4,
                proof_of_work_bits: 0,
                reduction_strategy: FriReductionStrategy::Fixed(reduction_arity_bits.clone()),
                num_query_rounds: 8,
            },
            hiding: false,
            degree_bits: k,
            reduction_arity_bits,
        };

        let polys: Vec<PolynomialCoeffs<F>> = (0..2)
            .map(|_| PolynomialCoeffs::new((0..(1usize << k)).map(|_| F::rand()).collect()))
            .collect();
        let batch = PolynomialBatch::<F, C, D>::from_coeffs(
            polys.clone(),
            fri_params.config.rate_bits,
            false,
            fri_params.config.cap_height,
            &mut timing,
            None,
        );

        // Common transcript: observe the cap, derive the opening point, and
        // observe the opening evaluations (the plonk prover's sequence around
        // `prove_openings`).
        let mut base_challenger = Challenger::<F, <C as GenericConfig<D>>::Hasher>::new();
        base_challenger.observe_cap::<<C as GenericConfig<D>>::Hasher>(&batch.merkle_tree.cap);
        let zeta = base_challenger.get_extension_challenge::<D>();
        let evals: Vec<<F as Extendable<D>>::Extension> = polys
            .iter()
            .map(|p| p.to_extension::<D>().eval(zeta))
            .collect();
        let mut prover_challenger = base_challenger.clone();
        prover_challenger.observe_extension_elements::<D>(&evals);
        let mut verifier_challenger = base_challenger.clone();
        verifier_challenger.observe_extension_elements::<D>(&evals);

        let fri_instance = FriInstanceInfo {
            oracles: vec![FriOracleInfo {
                num_polys: 2,
                blinding: false,
            }],
            batches: vec![FriBatchInfo {
                point: zeta,
                polynomials: (0..2)
                    .map(|i| FriPolynomialInfo {
                        oracle_index: 0,
                        polynomial_index: i,
                    })
                    .collect(),
            }],
        };

        let run = |gpu: bool| -> anyhow::Result<(FriProof<F, <C as GenericConfig<D>>::Hasher, D>, FriChallenges<F, D>)> {
            crate::hash::poseidon2::set_exclusive_gpu_phase(gpu);
            let mut challenger = prover_challenger.clone();
            let mut timing = TimingTree::default();
            let proof = PolynomialBatch::<F, C, D>::prove_openings(
                &fri_instance,
                &[&batch],
                &mut challenger,
                &fri_params,
                None,
                None,
                &mut timing,
            );
            crate::hash::poseidon2::set_exclusive_gpu_phase(false);
            let mut vc = verifier_challenger.clone();
            let challenges = vc.fri_challenges::<C, D>(
                &proof.commit_phase_merkle_caps,
                &proof.final_poly,
                proof.pow_witness,
                k,
                &fri_params.config,
                None,
                None,
            );
            Ok((proof, challenges))
        };

        let (proof_gpu, challenges_gpu) = run(true)?;
        let (proof_cpu, challenges_cpu) = run(false)?;

        // Both variants must verify against the same instance and openings.
        verify_fri_proof::<F, C, D>(
            &fri_instance,
            &FriOpenings {
                batches: vec![FriOpeningBatch {
                    values: evals.clone(),
                }],
            },
            &challenges_gpu,
            &[batch.merkle_tree.cap.clone()],
            &proof_gpu,
            &fri_params,
        )?;
        verify_fri_proof::<F, C, D>(
            &fri_instance,
            &FriOpenings {
                batches: vec![FriOpeningBatch { values: evals }],
            },
            &challenges_cpu,
            &[batch.merkle_tree.cap.clone()],
            &proof_cpu,
            &fri_params,
        )?;

        // The GPU LDE is bit-identical to the CPU FFT, so every proof field
        // must match exactly between the two backends.
        assert_eq!(
            proof_gpu.commit_phase_merkle_caps,
            proof_cpu.commit_phase_merkle_caps
        );
        assert_eq!(proof_gpu.final_poly, proof_cpu.final_poly);
        assert_eq!(proof_gpu.query_round_proofs, proof_cpu.query_round_proofs);
        Ok(())
    }
}
