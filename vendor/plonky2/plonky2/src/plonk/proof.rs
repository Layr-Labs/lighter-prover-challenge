//! plonky2 proof definition.
//!
//! Proofs can be later compressed to reduce their size, into either
//! [`CompressedProof`] or [`CompressedProofWithPublicInputs`] formats.
//! The latter can be directly passed to a verifier to assert its correctness.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::ensure;
use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::polynomial::PolynomialCoeffs;
use crate::field::types::Field;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::proof::{
    CompressedFriProof, FriChallenges, FriChallengesTarget, FriProof, FriProofTarget,
};
use crate::fri::structure::{
    FriOpeningBatch, FriOpeningBatchTarget, FriOpenings, FriOpeningsTarget,
};
use crate::fri::FriParams;
use crate::hash::hash_types::{MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_data::{CommonCircuitData, VerifierOnlyCircuitData};
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::verifier::verify_with_challenges;
use crate::util::serialization::{Buffer, Read, Write};

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct Proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> {
    /// Merkle cap of LDEs of wire values.
    pub wires_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of Z, in the context of Plonk's permutation argument.
    pub plonk_zs_partial_products_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of the quotient polynomial components.
    pub quotient_polys_cap: MerkleCap<F, C::Hasher>,
    /// Purported values of each polynomial at the challenge point.
    pub openings: OpeningSet<F, D>,
    /// A batch FRI argument for all openings.
    pub opening_proof: FriProof<F, C::Hasher, D>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProofTarget<const D: usize> {
    pub wires_cap: MerkleCapTarget,
    pub plonk_zs_partial_products_cap: MerkleCapTarget,
    pub quotient_polys_cap: MerkleCapTarget,
    pub openings: OpeningSetTarget<D>,
    pub opening_proof: FriProofTarget<D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Proof<F, C, D> {
    /// Compress the proof.
    pub fn compress(self, indices: &[usize], params: &FriParams) -> CompressedProof<F, C, D> {
        let Proof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof,
        } = self;

        CompressedProof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof: opening_proof.compress(indices, params),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct ProofWithPublicInputs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub proof: Proof<F, C, D>,
    pub public_inputs: Vec<F>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    ProofWithPublicInputs<F, C, D>
{
    pub fn compress(
        self,
        circuit_digest: &<<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<CompressedProofWithPublicInputs<F, C, D>> {
        let indices = self.fri_query_indices(circuit_digest, common_data)?;
        let compressed_proof = self.proof.compress(&indices, &common_data.fri_params);
        Ok(CompressedProofWithPublicInputs {
            public_inputs: self.public_inputs,
            proof: compressed_proof,
        })
    }

    pub fn get_public_inputs_hash(
        &self,
    ) -> <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash {
        C::InnerHasher::hash_no_pad(&self.public_inputs)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer
            .write_proof_with_public_inputs(self)
            .expect("Writing to a byte-vector cannot fail.");
        buffer
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<Self> {
        let mut buffer = Buffer::new(&bytes);
        let proof = buffer
            .read_proof_with_public_inputs(common_data)
            .map_err(anyhow::Error::msg)?;
        Ok(proof)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedProof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    /// Merkle cap of LDEs of wire values.
    pub wires_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of Z, in the context of Plonk's permutation argument.
    pub plonk_zs_partial_products_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of the quotient polynomial components.
    pub quotient_polys_cap: MerkleCap<F, C::Hasher>,
    /// Purported values of each polynomial at the challenge point.
    pub openings: OpeningSet<F, D>,
    /// A compressed batch FRI argument for all openings.
    pub opening_proof: CompressedFriProof<F, C::Hasher, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    CompressedProof<F, C, D>
{
    /// Decompress the proof.
    pub(crate) fn decompress(
        self,
        challenges: &ProofChallenges<F, D>,
        fri_inferred_elements: FriInferredElements<F, D>,
        params: &FriParams,
    ) -> Proof<F, C, D> {
        let CompressedProof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof,
        } = self;

        Proof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof: opening_proof.decompress(challenges, fri_inferred_elements, params),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedProofWithPublicInputs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub proof: CompressedProof<F, C, D>,
    pub public_inputs: Vec<F>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    CompressedProofWithPublicInputs<F, C, D>
{
    pub fn decompress(
        self,
        circuit_digest: &<<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<ProofWithPublicInputs<F, C, D>> {
        let challenges =
            self.get_challenges(self.get_public_inputs_hash(), circuit_digest, common_data)?;
        let fri_inferred_elements = self.get_inferred_elements(&challenges, common_data);
        let decompressed_proof =
            self.proof
                .decompress(&challenges, fri_inferred_elements, &common_data.fri_params);
        Ok(ProofWithPublicInputs {
            public_inputs: self.public_inputs,
            proof: decompressed_proof,
        })
    }

    pub(crate) fn verify(
        self,
        verifier_data: &VerifierOnlyCircuitData<C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<()> {
        ensure!(
            self.public_inputs.len() == common_data.num_public_inputs,
            "Number of public inputs doesn't match circuit data."
        );
        let public_inputs_hash = self.get_public_inputs_hash();
        let challenges = self.get_challenges(
            public_inputs_hash,
            &verifier_data.circuit_digest,
            common_data,
        )?;
        let fri_inferred_elements = self.get_inferred_elements(&challenges, common_data);
        let decompressed_proof =
            self.proof
                .decompress(&challenges, fri_inferred_elements, &common_data.fri_params);
        verify_with_challenges::<F, C, D>(
            decompressed_proof,
            public_inputs_hash,
            challenges,
            verifier_data,
            common_data,
        )
    }

    pub(crate) fn get_public_inputs_hash(
        &self,
    ) -> <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash {
        C::InnerHasher::hash_no_pad(&self.public_inputs)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer
            .write_compressed_proof_with_public_inputs(self)
            .expect("Writing to a byte-vector cannot fail.");
        buffer
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<Self> {
        let mut buffer = Buffer::new(&bytes);
        let proof = buffer
            .read_compressed_proof_with_public_inputs(common_data)
            .map_err(anyhow::Error::msg)?;
        Ok(proof)
    }
}

#[derive(Debug)]
pub struct ProofChallenges<F: RichField + Extendable<D>, const D: usize> {
    /// Random values used in Plonk's permutation argument.
    pub plonk_betas: Vec<F>,

    /// Random values used in Plonk's permutation argument.
    pub plonk_gammas: Vec<F>,

    /// Random values used to combine PLONK constraints.
    pub plonk_alphas: Vec<F>,

    /// Lookup challenges.
    pub plonk_deltas: Vec<F>,

    /// Point at which the PLONK polynomials are opened.
    pub plonk_zeta: F::Extension,

    pub fri_challenges: FriChallenges<F, D>,
}

pub(crate) struct ProofChallengesTarget<const D: usize> {
    pub plonk_betas: Vec<Target>,
    pub plonk_gammas: Vec<Target>,
    pub plonk_alphas: Vec<Target>,
    pub plonk_deltas: Vec<Target>,
    pub plonk_zeta: ExtensionTarget<D>,
    pub fri_challenges: FriChallengesTarget<D>,
}

/// Coset elements that can be inferred in the FRI reduction steps.
pub(crate) struct FriInferredElements<F: RichField + Extendable<D>, const D: usize>(
    pub Vec<F::Extension>,
);

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProofWithPublicInputsTarget<const D: usize> {
    pub proof: ProofTarget<D>,
    pub public_inputs: Vec<Target>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, Eq, PartialEq)]
/// The purported values of each polynomial at a single point.
pub struct OpeningSet<F: RichField + Extendable<D>, const D: usize> {
    pub constants: Vec<F::Extension>,
    pub plonk_sigmas: Vec<F::Extension>,
    pub wires: Vec<F::Extension>,
    pub plonk_zs: Vec<F::Extension>,
    pub plonk_zs_next: Vec<F::Extension>,
    pub partial_products: Vec<F::Extension>,
    pub quotient_polys: Vec<F::Extension>,
    pub lookup_zs: Vec<F::Extension>,
    pub lookup_zs_next: Vec<F::Extension>,
}

/// One opening-evaluation task: a single polynomial, or two polynomials
/// sharing one powers-table read. Lifetime is tied to the coefficient slices.
enum OpeningDotTask<'a, F: RichField + Extendable<D>, const D: usize> {
    Single {
        pows: &'a [F::Extension],
        coeffs: &'a [F],
    },
    Pair {
        pows: &'a [F::Extension],
        a: &'a [F],
        b: &'a [F],
    },
}

impl<'a, F: RichField + Extendable<D>, const D: usize> OpeningDotTask<'a, F, D> {
    fn eval(&self) -> ([F::Extension; 2], usize) {
        match self {
            Self::Single { pows, coeffs } => (
                [F::extension_base_dot_product(pows, coeffs), F::Extension::ZERO],
                1,
            ),
            Self::Pair { pows, a, b } => (F::extension_base_dot_products_2(pows, [*a, *b]), 2),
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> OpeningSet<F, D> {
    pub fn new<C: GenericConfig<D, F = F>>(
        zeta: F::Extension,
        g: F::Extension,
        constants_sigmas_commitment: &PolynomialBatch<F, C, D>,
        wires_commitment: &PolynomialBatch<F, C, D>,
        zs_partial_products_lookup_commitment: &PolynomialBatch<F, C, D>,
        quotient_polys_commitment: &PolynomialBatch<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Self {
        // Every committed polynomial in these batches has at most `degree`
        // coefficients, so one powers table per opening point covers all of
        // them. Evaluating as `sum_i c_i * z^i` against the table replaces the
        // previous `p.to_extension().eval(z)`, which cloned each polynomial
        // into a degree-sized extension vector (one allocation plus a full
        // conversion pass per polynomial) before running extension-by-
        // extension Horner. The dot product reads the base coefficients in
        // place; `Extendable::extension_base_dot_product` defaults to the old
        // scalar-multiply-and-sum loop and lets concrete fields delay
        // reductions where their representation makes that safe.
        // Value-exactness: the field is exact, `powers()` produces exactly
        // `z^i`, and `sum c_i z^i` under any association equals Horner's
        // `(..(c_{n-1} z + c_{n-2}) z + ..)`, so every opening is the
        // identical field element and the transcript is unchanged.
        let degree = common_data.degree();
        let table = |z: F::Extension| -> Vec<F::Extension> { z.powers().take(degree).collect() };
        let zeta_pows = table(zeta);
        // `g` is the order-`degree` subgroup generator, so `g^i` is exactly
        // the process-cached natural-order two-adic subgroup, and
        // `(g·ζ)^i = g^i · ζ^i` in the exact field. Deriving the shifted
        // table from the `ζ` table by one elementwise base-field scalar
        // product deletes the second serial `powers()` chain (a
        // `degree`-long dependent extension-multiply chain in this serial
        // opening phase) per proof. Representative-exactness is checkable at
        // runtime: `LIGHTER_GZETA_TABLE_ASSERT=1` recomputes the old chain
        // and compares every entry by raw noncanonical limbs.
        let g_subgroup =
            crate::plonk::prover::precomputed::two_adic_subgroup::<F>(common_data.degree_bits());
        let g_zeta_pows: Vec<F::Extension> = zeta_pows
            .iter()
            .zip(g_subgroup.iter())
            .map(|(&zeta_pow, &g_pow)| zeta_pow.scalar_mul(g_pow))
            .collect();
        if std::env::var_os("LIGHTER_GZETA_TABLE_ASSERT").is_some() {
            let reference = table(g * zeta);
            assert_eq!(reference.len(), g_zeta_pows.len());
            for (i, (a, b)) in reference.iter().zip(&g_zeta_pows).enumerate() {
                let a_raw: Vec<u64> = a
                    .to_basefield_array()
                    .iter()
                    .map(|c| c.to_noncanonical_u64())
                    .collect();
                let b_raw: Vec<u64> = b
                    .to_basefield_array()
                    .iter()
                    .map(|c| c.to_noncanonical_u64())
                    .collect();
                assert_eq!(a_raw, b_raw, "g-zeta table representative mismatch at {i}");
            }
        }
        let eval_polynomials = |pows: &[F::Extension], polynomials: &[PolynomialCoeffs<F>]| {
            polynomials
                .par_iter()
                .map(|p| F::extension_base_dot_product(pows, &p.coeffs))
                .collect::<Vec<_>>()
        };
        let _eval_polynomials = eval_polynomials;
        let shifted_polynomials = &zs_partial_products_lookup_commitment.polynomials;

        // Build one ordered work plan over every opening evaluation. Each task
        // evaluates one or two coefficient polynomials against one powers
        // table; consecutive polynomials that share the same powers slice
        // (always true within a point, since all zeta groups use `zeta_pows`
        // and the shifted groups use `g_zeta_pows`) are paired so the single
        // powers-table load is shared across two accumulators. This replaces
        // the separate short Rayon waves per logical group with one indexed
        // wave, deleting the per-group dispatch and load-balance overhead.
        //
        // Flattened output order is historical: constants, sigmas, wires, Z,
        // partial products, quotient, lookup (all at `zeta`), then shifted Z
        // and lookup (at `g * zeta`). Ranges over the flat output slice below
        // reproduce exactly the slices the per-commitment evaluations made.
        let zeta_pows_ref: &[F::Extension] = &zeta_pows;
        let g_zeta_pows_ref: &[F::Extension] = &g_zeta_pows;

        let mut entries: Vec<(&[F::Extension], &[F])> = Vec::new();
        let mut ranges = [0usize; 9];
        let mut zppl_end = 0usize;
        let mut wires_len = 0usize;
        let mut quotient_len = 0usize;

        let cs_start = entries.len();
        for p in &constants_sigmas_commitment.polynomials {
            entries.push((zeta_pows_ref, &p.coeffs));
        }
        let cr = common_data.constants_range();
        ranges[0] = cs_start + cr.start;
        let sr = common_data.sigmas_range();
        ranges[1] = cs_start + sr.start;

        let wires_start = entries.len();
        for p in &wires_commitment.polynomials {
            entries.push((zeta_pows_ref, &p.coeffs));
        }
        ranges[2] = wires_start;
        wires_len = entries.len() - wires_start;

        let zppl_start = entries.len();
        for p in &zs_partial_products_lookup_commitment.polynomials {
            entries.push((zeta_pows_ref, &p.coeffs));
        }
        zppl_end = entries.len();
        ranges[3] = zppl_start + common_data.zs_range().start;
        ranges[4] = zppl_start + common_data.partial_products_range().start;
        ranges[5] = zppl_start + common_data.lookup_range().start;

        let q_start = entries.len();
        for p in &quotient_polys_commitment.polynomials {
            entries.push((zeta_pows_ref, &p.coeffs));
        }
        ranges[6] = q_start;
        quotient_len = entries.len() - q_start;

        // Shifted groups at `g * zeta`: Z and lookup only (partial products
        // stay single-point at `zeta`).
        let gz_start = entries.len();
        for p in &shifted_polynomials[common_data.zs_range()] {
            entries.push((g_zeta_pows_ref, &p.coeffs));
        }
        ranges[7] = gz_start;
        let gzl_start = entries.len();
        for p in &shifted_polynomials[common_data.lookup_range()] {
            entries.push((g_zeta_pows_ref, &p.coeffs));
        }
        ranges[8] = gzl_start;

        // Pair consecutive entries sharing a pointer-identical powers slice.
        let mut tasks: Vec<OpeningDotTask<'_, F, D>> = Vec::with_capacity(entries.len().div_ceil(2));
        let mut i = 0;
        while i < entries.len() {
            if i + 1 < entries.len()
                && core::ptr::eq(entries[i].0, entries[i + 1].0)
            {
                tasks.push(OpeningDotTask::Pair {
                    pows: entries[i].0,
                    a: entries[i].1,
                    b: entries[i + 1].1,
                });
                i += 2;
            } else {
                tasks.push(OpeningDotTask::Single {
                    pows: entries[i].0,
                    coeffs: entries[i].1,
                });
                i += 1;
            }
        }

        // One indexed Rayon wave over the tasks; each returns fixed storage
        // plus a valid-lane count, and the only output collection is here.
        // `par_map_init` preserves task order, so the flattened result is the
        // historical entry order.
        let mut outputs: Vec<F::Extension> = vec![F::Extension::ZERO; entries.len()];
        let task_outputs: Vec<([F::Extension; 2], usize)> = tasks
            .par_iter()
            .map(OpeningDotTask::eval)
            .collect();
        let mut cursor = 0;
        for ([a, b], count) in task_outputs {
            outputs[cursor] = a;
            if count == 2 {
                outputs[cursor + 1] = b;
            }
            cursor += count;
        }

        let sl = |start: usize, end: usize| -> Vec<F::Extension> {
            outputs[start..end].to_vec()
        };
        let cr = common_data.constants_range();
        let sr = common_data.sigmas_range();
        let zr = common_data.zs_range();
        let pr = common_data.partial_products_range();
        let lookup_len = zppl_end.saturating_sub(zppl_start + common_data.lookup_range().start);
        let shifted_zr = ranges[7]..ranges[7] + zr.end - zr.start;
        let shifted_lr = ranges[8]..ranges[8] + lookup_len;

        Self {
            constants: sl(ranges[0], ranges[0] + cr.end - cr.start),
            plonk_sigmas: sl(ranges[1], ranges[1] + sr.end - sr.start),
            wires: sl(ranges[2], ranges[2] + wires_len),
            plonk_zs: sl(ranges[3], ranges[3] + zr.end - zr.start),
            // Partial-product polynomials are opened only at `zeta`, never at
            // `g * zeta`; evaluate only the shifted Z polynomials consumed by
            // the FRI next batch.
            plonk_zs_next: sl(shifted_zr.start, shifted_zr.end),
            partial_products: sl(ranges[4], ranges[4] + pr.end - pr.start),
            quotient_polys: sl(ranges[6], ranges[6] + quotient_len),
            lookup_zs: sl(ranges[5], ranges[5] + lookup_len),
            lookup_zs_next: sl(shifted_lr.start, shifted_lr.end),
        }
    }
    pub(crate) fn to_fri_openings(&self) -> FriOpenings<F, D> {
        let has_lookup = !self.lookup_zs.is_empty();
        let zeta_batch = if has_lookup {
            FriOpeningBatch {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                    self.lookup_zs.as_slice(),
                ]
                .concat(),
            }
        } else {
            FriOpeningBatch {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                ]
                .concat(),
            }
        };
        let zeta_next_batch = if has_lookup {
            FriOpeningBatch {
                values: [self.plonk_zs_next.clone(), self.lookup_zs_next.clone()].concat(),
            }
        } else {
            FriOpeningBatch {
                values: self.plonk_zs_next.clone(),
            }
        };
        FriOpenings {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}

/// The purported values of each polynomial at a single point.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OpeningSetTarget<const D: usize> {
    pub constants: Vec<ExtensionTarget<D>>,
    pub plonk_sigmas: Vec<ExtensionTarget<D>>,
    pub wires: Vec<ExtensionTarget<D>>,
    pub plonk_zs: Vec<ExtensionTarget<D>>,
    pub plonk_zs_next: Vec<ExtensionTarget<D>>,
    pub lookup_zs: Vec<ExtensionTarget<D>>,
    pub next_lookup_zs: Vec<ExtensionTarget<D>>,
    pub partial_products: Vec<ExtensionTarget<D>>,
    pub quotient_polys: Vec<ExtensionTarget<D>>,
}

impl<const D: usize> OpeningSetTarget<D> {
    pub(crate) fn to_fri_openings(&self) -> FriOpeningsTarget<D> {
        let has_lookup = !self.lookup_zs.is_empty();
        let zeta_batch = if has_lookup {
            FriOpeningBatchTarget {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                    self.lookup_zs.as_slice(),
                ]
                .concat(),
            }
        } else {
            FriOpeningBatchTarget {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                ]
                .concat(),
            }
        };
        let zeta_next_batch = if has_lookup {
            FriOpeningBatchTarget {
                values: [self.plonk_zs_next.clone(), self.next_lookup_zs.clone()].concat(),
            }
        } else {
            FriOpeningBatchTarget {
                values: self.plonk_zs_next.clone(),
            }
        };
        FriOpeningsTarget {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::sync::Arc;
    #[cfg(feature = "std")]
    use std::sync::Arc;

    use anyhow::Result;
    use itertools::Itertools;
    use plonky2_field::types::Sample;

    use super::*;
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::gates::lookup_table::LookupTable;
    use crate::gates::noop::NoopGate;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::plonk::verifier::verify;

    #[test]
    fn test_proof_compression() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![1, 1]);
        config.fri_config.num_query_rounds = 50;

        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // Build dummy circuit to get a valid proof.
        let x = F::rand();
        let y = F::rand();
        let z = x * y;
        let xt = builder.constant(x);
        let yt = builder.constant(y);
        let zt = builder.constant(z);
        let comp_zt = builder.mul(xt, yt);
        builder.connect(zt, comp_zt);
        for _ in 0..100 {
            builder.add_gate(NoopGate, vec![]);
        }
        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        verify(proof.clone(), &data.verifier_only, &data.common)?;

        // Verify that `decompress ∘ compress = identity`.
        let compressed_proof = data.compress(proof.clone())?;
        let decompressed_compressed_proof = data.decompress(compressed_proof.clone())?;
        assert_eq!(proof, decompressed_compressed_proof);

        verify(proof, &data.verifier_only, &data.common)?;
        data.verify_compressed(compressed_proof)
    }

    #[test]
    fn test_proof_compression_lookup() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        use plonky2_field::types::Field;
        type F = <C as GenericConfig<D>>::F;

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![1, 1]);
        config.fri_config.num_query_rounds = 50;

        let pw = PartialWitness::new();
        let tip5_table = vec![
            0, 7, 26, 63, 124, 215, 85, 254, 214, 228, 45, 185, 140, 173, 33, 240, 29, 177, 176,
            32, 8, 110, 87, 202, 204, 99, 150, 106, 230, 14, 235, 128, 213, 239, 212, 138, 23, 130,
            208, 6, 44, 71, 93, 116, 146, 189, 251, 81, 199, 97, 38, 28, 73, 179, 95, 84, 152, 48,
            35, 119, 49, 88, 242, 3, 148, 169, 72, 120, 62, 161, 166, 83, 175, 191, 137, 19, 100,
            129, 112, 55, 221, 102, 218, 61, 151, 237, 68, 164, 17, 147, 46, 234, 203, 216, 22,
            141, 65, 57, 123, 12, 244, 54, 219, 231, 96, 77, 180, 154, 5, 253, 133, 165, 98, 195,
            205, 134, 245, 30, 9, 188, 59, 142, 186, 197, 181, 144, 92, 31, 224, 163, 111, 74, 58,
            69, 113, 196, 67, 246, 225, 10, 121, 50, 60, 157, 90, 122, 2, 250, 101, 75, 178, 159,
            24, 36, 201, 11, 243, 132, 198, 190, 114, 233, 39, 52, 21, 209, 108, 238, 91, 187, 18,
            104, 194, 37, 153, 34, 200, 143, 126, 155, 236, 118, 64, 80, 172, 89, 94, 193, 135,
            183, 86, 107, 252, 13, 167, 206, 136, 220, 207, 103, 171, 160, 76, 182, 227, 217, 158,
            56, 174, 4, 66, 109, 139, 162, 184, 211, 249, 47, 125, 232, 117, 43, 16, 42, 127, 20,
            241, 25, 149, 105, 156, 51, 53, 168, 145, 247, 223, 79, 78, 226, 15, 222, 82, 115, 70,
            210, 27, 41, 1, 170, 40, 131, 192, 229, 248, 255,
        ];
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let lut_index = builder.add_lookup_table_from_pairs(table);

        // Build dummy circuit with a lookup to get a valid proof.
        let x = F::TWO;
        let out = builder.constant(F::from_canonical_usize(26));

        let xt = builder.constant(x);
        let look_out = builder.add_lookup_from_index(xt, lut_index);
        builder.connect(look_out, out);
        for _ in 0..100 {
            builder.add_gate(NoopGate, vec![]);
        }
        let data = builder.build::<C>();

        let proof = data.prove(pw)?;
        verify(proof.clone(), &data.verifier_only, &data.common)?;

        // Verify that `decompress ∘ compress = identity`.
        let compressed_proof = data.compress(proof.clone())?;
        let decompressed_compressed_proof = data.decompress(compressed_proof.clone())?;
        assert_eq!(proof, decompressed_compressed_proof);

        verify(proof, &data.verifier_only, &data.common)?;
        data.verify_compressed(compressed_proof)
    }
}
