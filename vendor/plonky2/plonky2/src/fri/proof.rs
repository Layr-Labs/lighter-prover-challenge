#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use hashbrown::{hash_map::Entry, HashMap};
use itertools::izip;
use serde::{Deserialize, Serialize};

use crate::field::extension::{flatten, unflatten, Extendable};
use crate::field::polynomial::PolynomialCoeffs;
use crate::fri::FriParams;
use crate::gadgets::polynomial::PolynomialCoeffsExtTarget;
use crate::hash::hash_types::{MerkleCapTarget, RichField};
use crate::hash::merkle_proofs::{MerkleProof, MerkleProofTarget};
use crate::hash::merkle_tree::MerkleCap;
use crate::hash::path_compression::{compress_merkle_proofs, decompress_merkle_proofs};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::config::Hasher;
use crate::plonk::plonk_common::salt_size;
use crate::plonk::proof::{FriInferredElements, ProofChallenges};

/// Evaluations and Merkle proof produced by the prover in a FRI query step.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriQueryStep<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    pub evals: Vec<F::Extension>,
    pub merkle_proof: MerkleProof<F, H>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FriQueryStepTarget<const D: usize> {
    pub evals: Vec<ExtensionTarget<D>>,
    pub merkle_proof: MerkleProofTarget,
}

/// Evaluations and Merkle proofs of the original set of polynomials,
/// before they are combined into a composition polynomial.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriInitialTreeProof<F: RichField, H: Hasher<F>> {
    pub evals_proofs: Vec<(Vec<F>, MerkleProof<F, H>)>,
}

impl<F: RichField, H: Hasher<F>> FriInitialTreeProof<F, H> {
    pub(crate) fn unsalted_eval(&self, oracle_index: usize, poly_index: usize, salted: bool) -> F {
        self.unsalted_evals(oracle_index, salted)[poly_index]
    }

    fn unsalted_evals(&self, oracle_index: usize, salted: bool) -> &[F] {
        let evals = &self.evals_proofs[oracle_index].0;
        &evals[..evals.len() - salt_size(salted)]
    }
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FriInitialTreeProofTarget {
    pub evals_proofs: Vec<(Vec<Target>, MerkleProofTarget)>,
}

impl FriInitialTreeProofTarget {
    pub(crate) fn unsalted_eval(
        &self,
        oracle_index: usize,
        poly_index: usize,
        salted: bool,
    ) -> Target {
        self.unsalted_evals(oracle_index, salted)[poly_index]
    }

    fn unsalted_evals(&self, oracle_index: usize, salted: bool) -> &[Target] {
        let evals = &self.evals_proofs[oracle_index].0;
        &evals[..evals.len() - salt_size(salted)]
    }
}

/// Proof for a FRI query round.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriQueryRound<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    pub initial_trees_proof: FriInitialTreeProof<F, H>,
    pub steps: Vec<FriQueryStep<F, H, D>>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FriQueryRoundTarget<const D: usize> {
    pub initial_trees_proof: FriInitialTreeProofTarget,
    pub steps: Vec<FriQueryStepTarget<D>>,
}

/// Compressed proof of the FRI query rounds.
#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedFriQueryRounds<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    /// Query indices.
    pub indices: Vec<usize>,
    /// Map from initial indices `i` to the `FriInitialProof` for the `i`th leaf.
    pub initial_trees_proofs: HashMap<usize, FriInitialTreeProof<F, H>>,
    /// For each FRI query step, a map from indices `i` to the `FriQueryStep` for the `i`th leaf.
    pub steps: Vec<HashMap<usize, FriQueryStep<F, H, D>>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct FriProof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    /// A Merkle cap for each reduced polynomial in the commit phase.
    pub commit_phase_merkle_caps: Vec<MerkleCap<F, H>>,
    /// Query rounds proofs
    pub query_round_proofs: Vec<FriQueryRound<F, H, D>>,
    /// The final polynomial in coefficient form.
    pub final_poly: PolynomialCoeffs<F::Extension>,
    /// Witness showing that the prover did PoW.
    pub pow_witness: F,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FriProofTarget<const D: usize> {
    pub commit_phase_merkle_caps: Vec<MerkleCapTarget>,
    pub query_round_proofs: Vec<FriQueryRoundTarget<D>>,
    pub final_poly: PolynomialCoeffsExtTarget<D>,
    pub pow_witness: Target,
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedFriProof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> {
    /// A Merkle cap for each reduced polynomial in the commit phase.
    pub commit_phase_merkle_caps: Vec<MerkleCap<F, H>>,
    /// Compressed query rounds proof.
    pub query_round_proofs: CompressedFriQueryRounds<F, H, D>,
    /// The final polynomial in coefficient form.
    pub final_poly: PolynomialCoeffs<F::Extension>,
    /// Witness showing that the prover did PoW.
    pub pow_witness: F,
}

impl<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> FriProof<F, H, D> {
    /// Compress all the Merkle paths in the FRI proof and remove duplicate indices.
    pub fn compress(self, indices: Vec<usize>, params: &FriParams) -> CompressedFriProof<F, H, D> {
        let FriProof {
            commit_phase_merkle_caps,
            query_round_proofs,
            final_poly,
            pow_witness,
            ..
        } = self;
        let cap_height = params.config.cap_height;
        let reduction_arity_bits = &params.reduction_arity_bits;
        let num_reductions = reduction_arity_bits.len();
        let num_initial_trees = query_round_proofs[0].initial_trees_proof.evals_proofs.len();
        let num_queries = indices.len();

        // "Transpose" the query round proofs, so that information for each Merkle tree is collected together.
        let mut initial_trees_indices = (0..num_initial_trees)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut initial_trees_leaves = (0..num_initial_trees)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut initial_trees_proofs = (0..num_initial_trees)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut steps_indices = (0..num_reductions)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut steps_evals = (0..num_reductions)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut steps_proofs = (0..num_reductions)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();

        for (mut index, qrp) in indices.iter().copied().zip(query_round_proofs) {
            let FriQueryRound {
                initial_trees_proof,
                steps,
            } = qrp;
            for (i, (leaves_data, proof)) in
                initial_trees_proof.evals_proofs.into_iter().enumerate()
            {
                initial_trees_indices[i].push(index);
                initial_trees_leaves[i].push(leaves_data);
                initial_trees_proofs[i].push(proof);
            }
            for (i, query_step) in steps.into_iter().enumerate() {
                let index_within_coset = index & ((1 << reduction_arity_bits[i]) - 1);
                index >>= reduction_arity_bits[i];
                steps_indices[i].push(index);
                let mut evals = query_step.evals;
                // Remove the element that can be inferred.
                evals.remove(index_within_coset);
                steps_evals[i].push(evals);
                steps_proofs[i].push(query_step.merkle_proof);
            }
        }

        // Compress all Merkle proofs.
        let mut initial_trees_proofs = initial_trees_indices
            .iter()
            .zip(initial_trees_proofs)
            .map(|(is, ps)| compress_merkle_proofs(cap_height, is, &ps))
            .collect::<Vec<_>>();
        let mut steps_proofs = steps_indices
            .iter()
            .zip(steps_proofs)
            .map(|(is, ps)| compress_merkle_proofs(cap_height, is, &ps))
            .collect::<Vec<_>>();

        let mut compressed_query_proofs = CompressedFriQueryRounds {
            indices: Vec::new(),
            initial_trees_proofs: HashMap::with_capacity(num_queries),
            steps: (0..num_reductions)
                .map(|_| HashMap::with_capacity(num_queries))
                .collect(),
        };

        // Replace the query round proofs with the compressed versions.
        for (i, mut index) in indices.iter().copied().enumerate() {
            if let Entry::Vacant(entry) = compressed_query_proofs.initial_trees_proofs.entry(index)
            {
                entry.insert(FriInitialTreeProof {
                    evals_proofs: (0..num_initial_trees)
                        .map(|j| {
                            (
                                core::mem::take(&mut initial_trees_leaves[j][i]),
                                MerkleProof {
                                    siblings: core::mem::take(
                                        &mut initial_trees_proofs[j][i].siblings,
                                    ),
                                },
                            )
                        })
                        .collect(),
                });
            }
            for j in 0..num_reductions {
                index >>= reduction_arity_bits[j];
                if let Entry::Vacant(entry) = compressed_query_proofs.steps[j].entry(index) {
                    entry.insert(FriQueryStep {
                        evals: core::mem::take(&mut steps_evals[j][i]),
                        merkle_proof: MerkleProof {
                            siblings: core::mem::take(&mut steps_proofs[j][i].siblings),
                        },
                    });
                }
            }
        }
        compressed_query_proofs.indices = indices;

        CompressedFriProof {
            commit_phase_merkle_caps,
            query_round_proofs: compressed_query_proofs,
            final_poly,
            pow_witness,
        }
    }
}

impl<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize> CompressedFriProof<F, H, D> {
    /// Decompress all the Merkle paths in the FRI proof and reinsert duplicate indices.
    pub(crate) fn decompress(
        self,
        challenges: &ProofChallenges<F, D>,
        fri_inferred_elements: FriInferredElements<F, D>,
        params: &FriParams,
    ) -> FriProof<F, H, D> {
        let CompressedFriProof {
            commit_phase_merkle_caps,
            mut query_round_proofs,
            final_poly,
            pow_witness,
            ..
        } = self;
        let FriChallenges {
            fri_query_indices: indices,
            ..
        } = &challenges.fri_challenges;
        let mut fri_inferred_elements = fri_inferred_elements.0.into_iter();
        let cap_height = params.config.cap_height;
        let reduction_arity_bits = &params.reduction_arity_bits;
        let num_reductions = reduction_arity_bits.len();
        let num_initial_trees = query_round_proofs
            .initial_trees_proofs
            .values()
            .next()
            .unwrap()
            .evals_proofs
            .len();
        let num_queries = indices.len();

        // "Transpose" the query round proofs, so that information for each Merkle tree is collected together.
        let mut initial_trees_indices = (0..num_initial_trees)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut initial_trees_leaves = (0..num_initial_trees)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut initial_trees_proofs = (0..num_initial_trees)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut steps_indices = (0..num_reductions)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut steps_evals = (0..num_reductions)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let mut steps_proofs = (0..num_reductions)
            .map(|_| Vec::with_capacity(num_queries))
            .collect::<Vec<_>>();
        let height = params.degree_bits + params.config.rate_bits;
        let heights = reduction_arity_bits
            .iter()
            .scan(height, |acc, &bits| {
                *acc -= bits;
                Some(*acc)
            })
            .collect::<Vec<_>>();

        // Holds the `evals` vectors that have already been reconstructed at each reduction depth.
        let mut evals_by_depth = (0..num_reductions)
            .map(|_| HashMap::<usize, Vec<_>>::with_capacity(num_queries))
            .collect::<Vec<_>>();
        for (query_pos, &(mut index)) in indices.iter().enumerate() {
            let later_indices = &indices[query_pos + 1..];
            let initial_trees_proof = if later_indices.contains(&index) {
                query_round_proofs.initial_trees_proofs[&index].clone()
            } else {
                query_round_proofs
                    .initial_trees_proofs
                    .remove(&index)
                    .unwrap()
            };
            for (i, (leaves_data, proof)) in
                initial_trees_proof.evals_proofs.into_iter().enumerate()
            {
                initial_trees_indices[i].push(index);
                initial_trees_leaves[i].push(leaves_data);
                initial_trees_proofs[i].push(proof);
            }
            let mut reduced_bits = 0;
            for i in 0..num_reductions {
                let index_within_coset = index & ((1 << reduction_arity_bits[i]) - 1);
                reduced_bits += reduction_arity_bits[i];
                index >>= reduction_arity_bits[i];
                let reused_later = later_indices
                    .iter()
                    .any(|&later_index| later_index >> reduced_bits == index);
                let FriQueryStep {
                    mut evals,
                    merkle_proof,
                } = if reused_later {
                    query_round_proofs.steps[i][&index].clone()
                } else {
                    query_round_proofs.steps[i].remove(&index).unwrap()
                };
                steps_indices[i].push(index);
                if evals_by_depth[i].contains_key(&index) {
                    // If this index has already been seen, get `evals` from the `HashMap`.
                    evals = if reused_later {
                        evals_by_depth[i][&index].clone()
                    } else {
                        evals_by_depth[i].remove(&index).unwrap()
                    };
                } else {
                    // Otherwise insert the next inferred element.
                    evals.insert(index_within_coset, fri_inferred_elements.next().unwrap());
                    if reused_later {
                        evals_by_depth[i].insert(index, evals.clone());
                    }
                }
                steps_evals[i].push(flatten(&evals));
                steps_proofs[i].push(merkle_proof);
            }
        }

        // Decompress all Merkle proofs.
        let initial_trees_proofs = izip!(
            &initial_trees_leaves,
            &initial_trees_indices,
            initial_trees_proofs
        )
        .map(|(ls, is, ps)| decompress_merkle_proofs(ls, is, &ps, height, cap_height))
        .collect::<Vec<_>>();
        let steps_proofs = izip!(&steps_evals, &steps_indices, steps_proofs, heights)
            .map(|(ls, is, ps, h)| decompress_merkle_proofs(ls, is, &ps, h, cap_height))
            .collect::<Vec<_>>();

        // Turn the transposed buffers into iterators so each query can take ownership of its
        // leaves and Merkle paths. These values are no longer needed by any other query.
        let mut initial_trees_leaves = initial_trees_leaves
            .into_iter()
            .map(Vec::into_iter)
            .collect::<Vec<_>>();
        let mut initial_trees_proofs = initial_trees_proofs
            .into_iter()
            .map(Vec::into_iter)
            .collect::<Vec<_>>();
        let mut steps_evals = steps_evals
            .into_iter()
            .map(Vec::into_iter)
            .collect::<Vec<_>>();
        let mut steps_proofs = steps_proofs
            .into_iter()
            .map(Vec::into_iter)
            .collect::<Vec<_>>();
        let mut decompressed_query_proofs = Vec::with_capacity(num_queries);
        for _ in 0..num_queries {
            let initial_trees_proof = FriInitialTreeProof {
                evals_proofs: izip!(&mut initial_trees_leaves, &mut initial_trees_proofs)
                    .map(|(leaves, proof)| (leaves.next().unwrap(), proof.next().unwrap()))
                    .collect(),
            };
            let steps = izip!(&mut steps_evals, &mut steps_proofs)
                .map(|(evals, proof)| FriQueryStep {
                    evals: unflatten(&evals.next().unwrap()),
                    merkle_proof: proof.next().unwrap(),
                })
                .collect();
            decompressed_query_proofs.push(FriQueryRound {
                initial_trees_proof,
                steps,
            })
        }

        FriProof {
            commit_phase_merkle_caps,
            query_round_proofs: decompressed_query_proofs,
            final_poly,
            pow_witness,
        }
    }
}

#[derive(Debug)]
pub struct FriChallenges<F: RichField + Extendable<D>, const D: usize> {
    // Scaling factor to combine polynomials.
    pub fri_alpha: F::Extension,

    // Betas used in the FRI commit phase reductions.
    pub fri_betas: Vec<F::Extension>,

    pub fri_pow_response: F,

    // Indices at which the oracle is queried in FRI.
    pub fri_query_indices: Vec<usize>,
}

#[derive(Debug)]
pub struct FriChallengesTarget<const D: usize> {
    pub fri_alpha: ExtensionTarget<D>,
    pub fri_betas: Vec<ExtensionTarget<D>>,
    pub fri_pow_response: Target,
    pub fri_query_indices: Vec<Target>,
}
