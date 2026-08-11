#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;

use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::{unflatten, Extendable, FieldExtension};
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::oracle::coset_fft_zero_tail;
use crate::fri::proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use crate::fri::{FriConfig, FriParams};
use crate::hash::hash_types::{RichField, NUM_HASH_OUT_ELTS};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::merkle_proofs::MerkleProof;
use crate::hash::merkle_tree::{
    capacity_up_to_mut, fill_subtree_flat, merkle_tree_prove, MerkleCap, MerkleTree,
};
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
        commit_phase_merkle_caps: trees.iter().map(CommittedFriTree::cap).cloned().collect(),
        query_round_proofs,
        final_poly: final_coeffs,
        pow_witness,
    }
}

pub(crate) type FriCommitedTrees<F, C, const D: usize> = (
    Vec<MerkleTree<F, <C as GenericConfig<D>>::Hasher>>,
    PolynomialCoeffs<<F as Extendable<D>>::Extension>,
);

/// A FRI commitment tree which either owns the historical materialized,
/// bit-reversed flat leaves or retains the natural-order extension codeword
/// and gathers one small CPU leaf tile at a time while hashing.
///
/// The direct form is deliberately private to the ordinary FRI prover. Batch
/// FRI keeps its existing `MerkleTree` ABI, while the proof emitted here is
/// unchanged: both variants expose the same cap, leaves and sibling paths.
enum CommittedFriTree<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    Materialized(MerkleTree<F, H>),
    Direct(DirectBitrevFriTree<F, H, D>),
}

impl<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>
    CommittedFriTree<F, H, D>
{
    fn cap(&self) -> &MerkleCap<F, H> {
        match self {
            Self::Materialized(tree) => &tree.cap,
            Self::Direct(tree) => &tree.cap,
        }
    }

    fn evals(&self, leaf_index: usize) -> Vec<F::Extension> {
        match self {
            Self::Materialized(tree) => unflatten(tree.get(leaf_index)),
            Self::Direct(tree) => tree.evals(leaf_index),
        }
    }

    fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        match self {
            Self::Materialized(tree) => tree.prove(leaf_index),
            Self::Direct(tree) => tree.prove(leaf_index),
        }
    }
}

const DIRECT_FRI_TILE_LEAVES: usize = 64;
const DIRECT_FRI_MAX_LEAF_WIDTH: usize = 32;
const NORMAL_MIN_GPU_PERMUTATIONS: usize = 1 << 19;
const EXCLUSIVE_MIN_GPU_PERMUTATIONS: usize = 1 << 16;

/// Mirrors the deterministic portion of the Metal Merkle routing rule. The
/// direct tree is used only when the historical tree is certain to stay on
/// the CPU. The special 2^17-leaf shape also depends on live GPU occupancy, so
/// it conservatively keeps the materialized path.
fn direct_fri_tree_worthwhile<const D: usize>(
    value_count: usize,
    arity_bits: usize,
    cap_height: usize,
    exclusive_gpu_phase: bool,
) -> bool {
    let arity = 1usize << arity_bits;
    let leaf_width = arity * D;
    let leaf_count = value_count >> arity_bits;
    if leaf_width == 0
        || leaf_width > DIRECT_FRI_MAX_LEAF_WIDTH
        || leaf_count == 0
        || leaf_count == 1 << 17
    {
        return false;
    }
    let leaf_permutations = if leaf_width <= 4 {
        0
    } else {
        leaf_width.div_ceil(8) * leaf_count
    };
    let parent_permutations = leaf_count - (1usize << cap_height);
    let threshold = if exclusive_gpu_phase {
        EXCLUSIVE_MIN_GPU_PERMUTATIONS
    } else {
        NORMAL_MIN_GPU_PERMUTATIONS
    };
    leaf_permutations + parent_permutations < threshold
}

/// Natural-order extension values plus the digest tree built over their
/// bit-reversed arity-sized groups. This removes the separate `n * D` flat
/// leaf allocation and its write/read pass for CPU FRI trees.
struct DirectBitrevFriTree<
    F: RichField + Extendable<D>,
    H: Hasher<F>,
    const D: usize,
> {
    values: Vec<F::Extension>,
    arity_bits: usize,
    log_values: usize,
    num_leaves: usize,
    digests: Vec<H::Hash>,
    cap: MerkleCap<F, H>,
}

impl<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>
    DirectBitrevFriTree<F, H, D>
{
    fn new(values: Vec<F::Extension>, arity_bits: usize, cap_height: usize) -> Self {
        let log_values = log2_strict(values.len());
        let num_leaves = values.len() >> arity_bits;
        let log_leaves = log2_strict(num_leaves);
        assert!(cap_height <= log_leaves);
        assert!((1usize << arity_bits) * D <= DIRECT_FRI_MAX_LEAF_WIDTH);

        let cap_count = 1usize << cap_height;
        let num_digests = 2 * (num_leaves - cap_count);
        let mut digests = Vec::with_capacity(num_digests);
        let mut cap = Vec::with_capacity(cap_count);
        let cap_slots = capacity_up_to_mut(&mut cap, cap_count);
        let subtree_leaves = num_leaves >> cap_height;
        if num_digests == 0 {
            cap_slots.par_iter_mut().enumerate().for_each(|(leaf, cap_slot)| {
                cap_slot.write(fill_direct_fri_subtree::<F, H, D>(
                    &mut [],
                    &values,
                    log_values,
                    arity_bits,
                    leaf,
                    1,
                ));
            });
        } else {
            let digest_chunks = capacity_up_to_mut(&mut digests, num_digests)
                .par_chunks_exact_mut(num_digests >> cap_height);
            digest_chunks
                .zip(cap_slots)
                .enumerate()
                .for_each(|(subtree, (digest_chunk, cap_slot))| {
                    cap_slot.write(fill_direct_fri_subtree::<F, H, D>(
                        digest_chunk,
                        &values,
                        log_values,
                        arity_bits,
                        subtree * subtree_leaves,
                        subtree_leaves,
                    ));
                });
        }
        // SAFETY: every digest and cap slot was initialized by the disjoint
        // subtree traversal above.
        unsafe {
            digests.set_len(num_digests);
            cap.set_len(cap_count);
        }
        Self {
            values,
            arity_bits,
            log_values,
            num_leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    #[cfg(test)]
    fn leaf_vec(&self, leaf_index: usize) -> Vec<F> {
        assert!(leaf_index < self.num_leaves);
        direct_fri_leaf::<F, D>(
            &self.values,
            self.log_values,
            self.arity_bits,
            leaf_index,
        )
    }

    fn evals(&self, leaf_index: usize) -> Vec<F::Extension> {
        assert!(leaf_index < self.num_leaves);
        let arity = 1usize << self.arity_bits;
        (0..arity)
            .map(|lane| {
                let bitreversed = (leaf_index << self.arity_bits) + lane;
                self.values[reverse_bits(bitreversed, self.log_values)]
            })
            .collect()
    }

    fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        MerkleProof {
            siblings: merkle_tree_prove::<F, H>(
                leaf_index,
                self.num_leaves,
                self.cap.height(),
                &self.digests,
            ),
        }
    }
}

#[cfg(test)]
fn direct_fri_leaf<F: RichField + Extendable<D>, const D: usize>(
    values: &[F::Extension],
    log_values: usize,
    arity_bits: usize,
    leaf_index: usize,
) -> Vec<F> {
    let arity = 1usize << arity_bits;
    let mut leaf = Vec::with_capacity(arity * D);
    for lane in 0..arity {
        let bitreversed = (leaf_index << arity_bits) + lane;
        let natural = reverse_bits(bitreversed, log_values);
        leaf.extend_from_slice(&values[natural].to_basefield_array());
    }
    leaf
}

fn fill_direct_fri_subtree<
    F: RichField + Extendable<D>,
    H: Hasher<F>,
    const D: usize,
>(
    digests: &mut [MaybeUninit<H::Hash>],
    values: &[F::Extension],
    log_values: usize,
    arity_bits: usize,
    start_leaf: usize,
    num_leaves: usize,
) -> H::Hash {
    debug_assert_eq!(num_leaves, digests.len() / 2 + 1);
    if num_leaves <= DIRECT_FRI_TILE_LEAVES {
        return hash_direct_fri_tile::<F, H, D>(
            digests,
            values,
            log_values,
            arity_bits,
            start_leaf,
            num_leaves,
        );
    }

    let (left_digests, right_digests) = digests.split_at_mut(digests.len() / 2);
    let (left_root_slot, left_digests) = left_digests.split_last_mut().unwrap();
    let (right_root_slot, right_digests) = right_digests.split_first_mut().unwrap();
    let half = num_leaves / 2;
    let mut recurse_left = || {
        fill_direct_fri_subtree::<F, H, D>(
            left_digests,
            values,
            log_values,
            arity_bits,
            start_leaf,
            half,
        )
    };
    let mut recurse_right = || {
        fill_direct_fri_subtree::<F, H, D>(
            right_digests,
            values,
            log_values,
            arity_bits,
            start_leaf + half,
            half,
        )
    };
    let (left, right) = if num_leaves > 64 {
        plonky2_maybe_rayon::join(recurse_left, recurse_right)
    } else {
        (recurse_left(), recurse_right())
    };
    left_root_slot.write(left);
    right_root_slot.write(right);
    H::two_to_one(left, right)
}

/// Kept out of the recursive frame so production's 16 KiB Goldilocks scratch
/// tile is live only at a leaf task, rather than once per recursion level.
#[inline(never)]
fn hash_direct_fri_tile<
    F: RichField + Extendable<D>,
    H: Hasher<F>,
    const D: usize,
>(
    digests: &mut [MaybeUninit<H::Hash>],
    values: &[F::Extension],
    log_values: usize,
    arity_bits: usize,
    start_leaf: usize,
    num_leaves: usize,
) -> H::Hash {
    let leaf_width = (1usize << arity_bits) * D;
    let mut tile: [MaybeUninit<F>; DIRECT_FRI_TILE_LEAVES * DIRECT_FRI_MAX_LEAF_WIDTH] =
        // SAFETY: `MaybeUninit` has no initialization requirement.
        unsafe { MaybeUninit::uninit().assume_init() };
    for local_leaf in 0..num_leaves {
        let leaf_index = start_leaf + local_leaf;
        for lane in 0..1usize << arity_bits {
            let bitreversed = (leaf_index << arity_bits) + lane;
            let natural = reverse_bits(bitreversed, log_values);
            let limbs = values[natural].to_basefield_array();
            let offset = local_leaf * leaf_width + lane * D;
            for (slot, limb) in tile[offset..offset + D].iter_mut().zip(limbs) {
                slot.write(limb);
            }
        }
    }
    // SAFETY: the loops initialized the exact `num_leaves * leaf_width`
    // prefix and `MaybeUninit<F>` has the same layout as `F`.
    let flat = unsafe {
        slice::from_raw_parts(tile.as_ptr().cast::<F>(), num_leaves * leaf_width)
    };
    fill_subtree_flat::<F, H>(digests, flat, leaf_width, num_leaves)
}

fn build_committed_fri_tree<
    F: RichField + Extendable<D>,
    H: Hasher<F>,
    const D: usize,
>(
    values: Vec<F::Extension>,
    arity_bits: usize,
    cap_height: usize,
) -> CommittedFriTree<F, H, D> {
    let exclusive = crate::hash::poseidon2::is_exclusive_gpu_phase();
    if direct_fri_tree_worthwhile::<D>(values.len(), arity_bits, cap_height, exclusive) {
        CommittedFriTree::Direct(DirectBitrevFriTree::new(
            values,
            arity_bits,
            cap_height,
        ))
    } else {
        let flat = bitrev_flatten::<F, D>(&values);
        CommittedFriTree::Materialized(MerkleTree::new_flat(
            flat,
            (1usize << arity_bits) * D,
            cap_height,
        ))
    }
}

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

fn fri_committed_trees<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    mut coeffs: PolynomialCoeffs<F::Extension>,
    mut values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    final_poly_coeff_len: Option<usize>,
    max_num_query_steps: Option<usize>,
) -> (
    Vec<CommittedFriTree<F, C::Hasher, D>>,
    PolynomialCoeffs<F::Extension>,
) {
    let mut trees = Vec::with_capacity(fri_params.reduction_arity_bits.len());

    let mut shift = F::MULTIPLICATIVE_GROUP_GENERATOR;
    let num_rounds = fri_params.reduction_arity_bits.len();
    for (round, arity_bits) in fri_params.reduction_arity_bits.iter().enumerate() {
        let arity = 1 << arity_bits;

        // Fused bit-reversal + flatten: one gather pass writes the flat leaf
        // buffer directly (leaf `i` is the `arity`-chunk of the bit-reversed
        // codeword starting at `i * arity`), instead of a random-access
        // in-place permutation followed by a separate flattening pass with a
        // heap allocation per element.
        // The current codeword is never read after its commitment. Move it
        // into a direct CPU tree when the Metal router is guaranteed to
        // decline this shape; otherwise preserve the materialized path.
        let current_values = core::mem::take(&mut values.values);
        let tree = build_committed_fri_tree::<F, C::Hasher, D>(
            current_values,
            *arity_bits,
            fri_params.config.cap_height,
        );

        challenger.observe_cap(tree.cap());
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
    trees: &[CommittedFriTree<F, C::Hasher, D>],
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
    trees: &[CommittedFriTree<F, C::Hasher, D>],
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
        let evals = tree.evals(x_index >> arity_bits);
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

    use plonky2_field::types::Sample;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::hash_types::HashOut;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::hash::poseidon2::hash::Poseidon2Hash;

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

    #[test]
    fn direct_bitrev_fri_tree_matches_materialized_tree() {
        const D: usize = 2;
        type F = GoldilocksField;
        type FE = <F as Extendable<D>>::Extension;

        for log_values in [8usize, 12, 17] {
            let values: Vec<FE> = (0..1usize << log_values).map(|_| FE::rand()).collect();
            let flat = bitrev_flatten::<F, D>(&values);
            let expected = MerkleTree::<F, Poseidon2Hash>::new_flat(flat, 16 * D, 4);
            let actual = DirectBitrevFriTree::<F, Poseidon2Hash, D>::new(values, 4, 4);

            let raw_hashes = |hashes: &[HashOut<F>]| {
                hashes
                    .iter()
                    .map(|hash| hash.elements.map(|limb| limb.0))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                raw_hashes(&actual.cap.0),
                raw_hashes(&expected.cap.0),
                "raw cap limbs at log_values={log_values}"
            );
            assert_eq!(
                raw_hashes(&actual.digests),
                raw_hashes(&expected.digests),
                "raw digest limbs at log_values={log_values}"
            );
            let leaves = actual.num_leaves;
            for leaf_index in [0, 1, leaves / 3, leaves / 2, leaves - 1] {
                let leaf = actual.leaf_vec(leaf_index);
                assert_eq!(
                    leaf.iter().map(|limb| limb.0).collect::<Vec<_>>(),
                    expected
                        .leaf_vec(leaf_index)
                        .iter()
                        .map(|limb| limb.0)
                        .collect::<Vec<_>>()
                );
                let proof = actual.prove(leaf_index);
                assert_eq!(
                    raw_hashes(&proof.siblings),
                    raw_hashes(&expected.prove(leaf_index).siblings)
                );
                verify_merkle_proof_to_cap(leaf, leaf_index, &actual.cap, &proof).unwrap();
            }
        }
    }

    #[test]
    fn direct_fri_routing_preserves_gpu_shapes() {
        const D: usize = 2;
        // d14 chain/pre proof after rate-3 expansion, arity-16 reduction:
        // 2^13 leaves x width 32 stays below both GPU thresholds.
        assert!(direct_fri_tree_worthwhile::<D>(1 << 17, 4, 4, false));
        assert!(direct_fri_tree_worthwhile::<D>(1 << 17, 4, 4, true));
        // d16 transaction proof stays CPU during the normal pipeline, but
        // retains the old GPU path if it finishes inside an exclusive drain.
        assert!(direct_fri_tree_worthwhile::<D>(1 << 19, 4, 4, false));
        assert!(!direct_fri_tree_worthwhile::<D>(1 << 19, 4, 4, true));
        // d18 final proof has 2^17 leaves and occupancy-sensitive routing.
        assert!(!direct_fri_tree_worthwhile::<D>(1 << 21, 4, 4, false));
        assert!(!direct_fri_tree_worthwhile::<D>(1 << 21, 4, 4, true));
    }

    #[test]
    fn materialized_fri_fallback_remains_exact() {
        const D: usize = 2;
        type F = GoldilocksField;
        type FE = <F as Extendable<D>>::Extension;

        // Width 64 is outside the direct route, independent of GPU state.
        let values: Vec<FE> = (0..1usize << 8).map(|_| FE::rand()).collect();
        let expected = MerkleTree::<F, Poseidon2Hash>::new_flat(
            bitrev_flatten::<F, D>(&values),
            32 * D,
            3,
        );
        let actual = build_committed_fri_tree::<F, Poseidon2Hash, D>(values, 5, 3);
        match actual {
            CommittedFriTree::Materialized(tree) => {
                assert_eq!(
                    tree.cap
                        .0
                        .iter()
                        .map(|hash| hash.elements.map(|limb| limb.0))
                        .collect::<Vec<_>>(),
                    expected
                        .cap
                        .0
                        .iter()
                        .map(|hash| hash.elements.map(|limb| limb.0))
                        .collect::<Vec<_>>()
                );
                assert_eq!(tree.digests, expected.digests);
            }
            CommittedFriTree::Direct(_) => panic!("width-64 tree bypassed fallback"),
        }
    }

    #[test]
    #[ignore = "focused release microbenchmark; not part of the default test suite"]
    fn direct_bitrev_fri_tree_microbench() {
        const D: usize = 2;
        type F = GoldilocksField;
        type FE = <F as Extendable<D>>::Extension;
        const RUNS: usize = 7;

        for log_values in [17usize, 19] {
            let values: Vec<FE> = (0..1usize << log_values).map(|_| FE::rand()).collect();
            let mut materialized_times = Vec::with_capacity(RUNS);
            let mut direct_times = Vec::with_capacity(RUNS);
            for run in 0..RUNS {
                let direct_input = values.clone();
                if run % 2 == 0 {
                    let start = Instant::now();
                    let direct =
                        DirectBitrevFriTree::<F, Poseidon2Hash, D>::new(direct_input, 4, 4);
                    direct_times.push(start.elapsed());
                    let start = Instant::now();
                    let materialized = MerkleTree::<F, Poseidon2Hash>::new_flat(
                        bitrev_flatten::<F, D>(&values),
                        16 * D,
                        4,
                    );
                    materialized_times.push(start.elapsed());
                    assert_eq!(direct.cap, materialized.cap, "run {run}");
                } else {
                    let start = Instant::now();
                    let materialized = MerkleTree::<F, Poseidon2Hash>::new_flat(
                        bitrev_flatten::<F, D>(&values),
                        16 * D,
                        4,
                    );
                    materialized_times.push(start.elapsed());
                    let start = Instant::now();
                    let direct =
                        DirectBitrevFriTree::<F, Poseidon2Hash, D>::new(direct_input, 4, 4);
                    direct_times.push(start.elapsed());
                    assert_eq!(direct.cap, materialized.cap, "run {run}");
                }
            }
            direct_times.sort();
            materialized_times.sort();
            eprintln!(
                "log_values={log_values}: direct {:?}, materialized {:?}, ratio {:.4}",
                direct_times[RUNS / 2],
                materialized_times[RUNS / 2],
                direct_times[RUNS / 2].as_secs_f64()
                    / materialized_times[RUNS / 2].as_secs_f64()
            );
        }
    }
}
