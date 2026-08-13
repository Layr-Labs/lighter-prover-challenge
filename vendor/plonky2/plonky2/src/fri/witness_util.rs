use anyhow::{anyhow, Result};
use itertools::Itertools;
use plonky2_field::types::Field;

use crate::field::extension::Extendable;
use crate::field::polynomial::PolynomialCoeffs;
use crate::fri::proof::{FriProof, FriProofTarget, FriQueryRound};
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::witness::WitnessWrite;
use crate::plonk::config::AlgebraicHasher;

/// Set the targets in a `FriProofTarget` to their corresponding values in a `FriProof`.
pub fn set_fri_proof_target<F, W, H, const D: usize>(
    witness: &mut W,
    fri_proof_target: &FriProofTarget<D>,
    fri_proof: &FriProof<F, H, D>,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    W: WitnessWrite<F> + ?Sized,
    H: AlgebraicHasher<F>,
{
    set_fri_pow_witness_target(witness, fri_proof_target, fri_proof.pow_witness)?;
    set_fri_commit_phase_target(
        witness,
        fri_proof_target,
        &fri_proof.commit_phase_merkle_caps,
        &fri_proof.final_poly,
    )?;
    set_fri_query_rounds_target(
        witness,
        fri_proof_target,
        &fri_proof.query_round_proofs,
    )
}

/// Set the FRI commitment caps and final polynomial without requiring the PoW
/// witness or query rounds to exist yet.
pub fn set_fri_commit_phase_target<F, W, H, const D: usize>(
    witness: &mut W,
    fri_proof_target: &FriProofTarget<D>,
    commit_phase_merkle_caps: &[MerkleCap<F, H>],
    final_poly: &PolynomialCoeffs<F::Extension>,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    W: WitnessWrite<F> + ?Sized,
    H: AlgebraicHasher<F>,
{

    let target_len = fri_proof_target.final_poly.0.len();
    let coeffs_len = final_poly.coeffs.len();

    if target_len < coeffs_len {
        return Err(anyhow!(
            "fri_proof->final_poly's target length is less than the proof length"
        ));
    }

    // Set overlapping elements
    for i in 0..coeffs_len {
        witness.set_extension_target(
            fri_proof_target.final_poly.0[i],
            final_poly.coeffs[i],
        )?;
    }

    // Set remaining elements in target to ZERO if target is longer
    for i in coeffs_len..target_len {
        witness.set_extension_target(fri_proof_target.final_poly.0[i], F::Extension::ZERO)?;
    }

    let target_caps = &fri_proof_target.commit_phase_merkle_caps;
    let proof_caps = commit_phase_merkle_caps;

    if target_caps.len() < proof_caps.len() {
        return Err(anyhow!(
            "fri_proof->commit_phase_merkle_caps's target length is less than the proof length"
        ));
    }

    // Set matching elements in both proof and target caps
    for (target_cap, proof_cap) in target_caps.iter().zip(proof_caps) {
        witness.set_cap_target(target_cap, proof_cap)?;
    }

    // Set remaining elements in target caps to ZERO if target is longer
    for target_cap in target_caps.iter().skip(proof_caps.len()) {
        for hash in target_cap.0.iter() {
            witness.set_hash_target(*hash, HashOut::ZERO)?;
        }
    }

    Ok(())
}

/// Set only the FRI proof-of-work witness.
pub fn set_fri_pow_witness_target<F, W, const D: usize>(
    witness: &mut W,
    fri_proof_target: &FriProofTarget<D>,
    pow_witness: F,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    W: WitnessWrite<F> + ?Sized,
{
    witness.set_target(fri_proof_target.pow_witness, pow_witness)
}

/// Set the complete ordered FRI query-round batch.
pub fn set_fri_query_rounds_target<F, W, H, const D: usize>(
    witness: &mut W,
    fri_proof_target: &FriProofTarget<D>,
    query_round_proofs: &[FriQueryRound<F, H, D>],
) -> Result<()>
where
    F: RichField + Extendable<D>,
    W: WitnessWrite<F> + ?Sized,
    H: AlgebraicHasher<F>,
{
    if fri_proof_target.query_round_proofs.len() != query_round_proofs.len() {
        return Err(anyhow!(
            "FRI query-round target/proof length mismatch: {} targets, {} rounds",
            fri_proof_target.query_round_proofs.len(),
            query_round_proofs.len(),
        ));
    }
    for (ordinal, query_round) in query_round_proofs.iter().enumerate() {
        set_fri_query_round_target_at(witness, fri_proof_target, ordinal, query_round)?;
    }

    Ok(())
}

/// Set one authoritative FRI query round at its challenger ordinal.
pub fn set_fri_query_round_target_at<F, W, H, const D: usize>(
    witness: &mut W,
    fri_proof_target: &FriProofTarget<D>,
    ordinal: usize,
    query_round: &FriQueryRound<F, H, D>,
) -> Result<()>
where
    F: RichField + Extendable<D>,
    W: WitnessWrite<F> + ?Sized,
    H: AlgebraicHasher<F>,
{
    let query_target = fri_proof_target
        .query_round_proofs
        .get(ordinal)
        .ok_or_else(|| anyhow!("FRI query ordinal {ordinal} is out of range"))?;

    for (initial_target, initial_proof) in query_target
        .initial_trees_proof
        .evals_proofs
        .iter()
        .zip_eq(&query_round.initial_trees_proof.evals_proofs)
    {
        for (&target, &value) in initial_target.0.iter().zip_eq(&initial_proof.0) {
            witness.set_target(target, value)?;
        }
        let target_len = initial_target.1.siblings.len();
        let siblings_len = initial_proof.1.siblings.len();
        if target_len < siblings_len {
            return Err(anyhow!("fri_proof->query_round_proofs->initial_trees_proof->evals_proofs->siblings' target length is less than the proof length"));
        }
        for i in 0..siblings_len {
            witness.set_hash_target(initial_target.1.siblings[i], initial_proof.1.siblings[i])?;
        }
        for i in siblings_len..target_len {
            witness.set_hash_target(initial_target.1.siblings[i], HashOut::ZERO)?;
        }
    }

    for (step_target, step) in query_target.steps.iter().zip(&query_round.steps) {
        for (&target, &value) in step_target.evals.iter().zip_eq(&step.evals) {
            witness.set_extension_target(target, value)?;
        }
        let target_len = step_target.merkle_proof.siblings.len();
        let siblings_len = step.merkle_proof.siblings.len();
        if target_len < siblings_len {
            return Err(anyhow!("fri_proof->query_round_proofs->steps->merkle_proof->siblings' target length is less than the proof length"));
        }
        for i in 0..siblings_len {
            witness.set_hash_target(
                step_target.merkle_proof.siblings[i],
                step.merkle_proof.siblings[i],
            )?;
        }
        for i in siblings_len..target_len {
            witness.set_hash_target(step_target.merkle_proof.siblings[i], HashOut::ZERO)?;
        }
    }

    for step_target in query_target.steps.iter().skip(query_round.steps.len()) {
        for &eval in &step_target.evals {
            witness.set_extension_target(eval, F::Extension::ZERO)?;
        }
        for &sibling in &step_target.merkle_proof.siblings {
            witness.set_hash_target(sibling, HashOut::ZERO)?;
        }
    }

    Ok(())
}
