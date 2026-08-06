#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::iter::zip;

use anyhow::{anyhow, Result};
use hashbrown::HashMap;
use itertools::{zip_eq, Itertools};

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::fri::structure::{FriOpenings, FriOpeningsTarget};
use crate::fri::witness_util::set_fri_proof_target;
use crate::hash::hash_types::{HashOut, HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::{BoolTarget, Target};
use crate::iop::wire::Wire;
use crate::plonk::circuit_data::{VerifierCircuitTarget, VerifierOnlyCircuitData};
use crate::plonk::config::{AlgebraicHasher, GenericConfig};
use crate::plonk::proof::{Proof, ProofTarget, ProofWithPublicInputs, ProofWithPublicInputsTarget};

pub trait WitnessWrite<F: Field> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()>;

    fn set_hash_target(&mut self, ht: HashOutTarget, value: HashOut<F>) -> Result<()> {
        for (t, x) in zip(ht.elements, value.elements) {
            self.set_target(t, x)?;
        }

        Ok(())
    }

    fn set_cap_target<H: AlgebraicHasher<F>>(
        &mut self,
        ct: &MerkleCapTarget,
        value: &MerkleCap<F, H>,
    ) -> Result<()>
    where
        F: RichField,
    {
        for (ht, h) in ct.0.iter().zip(&value.0) {
            self.set_hash_target(*ht, *h)?;
        }

        Ok(())
    }

    fn set_extension_target<const D: usize>(
        &mut self,
        et: ExtensionTarget<D>,
        value: F::Extension,
    ) -> Result<()>
    where
        F: RichField + Extendable<D>,
    {
        self.set_target_arr(&et.0, &value.to_basefield_array())
    }

    fn set_target_arr(&mut self, targets: &[Target], values: &[F]) -> Result<()> {
        for (&target, &value) in zip_eq(targets, values) {
            self.set_target(target, value)?;
        }

        Ok(())
    }

    fn set_extension_targets<const D: usize>(
        &mut self,
        ets: &[ExtensionTarget<D>],
        values: &[F::Extension],
    ) -> Result<()>
    where
        F: RichField + Extendable<D>,
    {
        debug_assert_eq!(ets.len(), values.len());
        for (&et, &v) in zip(ets, values) {
            self.set_extension_target(et, v)?;
        }

        Ok(())
    }

    fn set_bool_target(&mut self, target: BoolTarget, value: bool) -> Result<()> {
        self.set_target(target.target, F::from_bool(value))
    }

    /// Set the targets in a `ProofWithPublicInputsTarget` to their corresponding values in a
    /// `ProofWithPublicInputs`.
    fn set_proof_with_pis_target<C: GenericConfig<D, F = F>, const D: usize>(
        &mut self,
        proof_with_pis_target: &ProofWithPublicInputsTarget<D>,
        proof_with_pis: &ProofWithPublicInputs<F, C, D>,
    ) -> Result<()>
    where
        F: RichField + Extendable<D>,
        C::Hasher: AlgebraicHasher<F>,
    {
        let ProofWithPublicInputs {
            proof,
            public_inputs,
        } = proof_with_pis;
        let ProofWithPublicInputsTarget {
            proof: pt,
            public_inputs: pi_targets,
        } = proof_with_pis_target;

        // Set public inputs.
        for (&pi_t, &pi) in pi_targets.iter().zip_eq(public_inputs) {
            self.set_target(pi_t, pi)?;
        }

        self.set_proof_target(pt, proof)
    }

    /// Set the targets in a `ProofTarget` to their corresponding values in a `Proof`.
    fn set_proof_target<C: GenericConfig<D, F = F>, const D: usize>(
        &mut self,
        proof_target: &ProofTarget<D>,
        proof: &Proof<F, C, D>,
    ) -> Result<()>
    where
        F: RichField + Extendable<D>,
        C::Hasher: AlgebraicHasher<F>,
    {
        self.set_cap_target(&proof_target.wires_cap, &proof.wires_cap)?;
        self.set_cap_target(
            &proof_target.plonk_zs_partial_products_cap,
            &proof.plonk_zs_partial_products_cap,
        )?;
        self.set_cap_target(&proof_target.quotient_polys_cap, &proof.quotient_polys_cap)?;

        self.set_fri_openings(
            &proof_target.openings.to_fri_openings(),
            &proof.openings.to_fri_openings(),
        )?;

        set_fri_proof_target(self, &proof_target.opening_proof, &proof.opening_proof)
    }

    fn set_fri_openings<const D: usize>(
        &mut self,
        fri_openings_target: &FriOpeningsTarget<D>,
        fri_openings: &FriOpenings<F, D>,
    ) -> Result<()>
    where
        F: RichField + Extendable<D>,
    {
        for (batch_target, batch) in fri_openings_target
            .batches
            .iter()
            .zip_eq(&fri_openings.batches)
        {
            self.set_extension_targets(&batch_target.values, &batch.values)?;
        }

        Ok(())
    }

    fn set_verifier_data_target<C: GenericConfig<D, F = F>, const D: usize>(
        &mut self,
        vdt: &VerifierCircuitTarget,
        vd: &VerifierOnlyCircuitData<C, D>,
    ) -> Result<()>
    where
        F: RichField + Extendable<D>,
        C::Hasher: AlgebraicHasher<F>,
    {
        self.set_cap_target(&vdt.constants_sigmas_cap, &vd.constants_sigmas_cap)?;
        self.set_hash_target(vdt.circuit_digest, vd.circuit_digest)
    }

    fn set_wire(&mut self, wire: Wire, value: F) -> Result<()> {
        self.set_target(Target::Wire(wire), value)
    }

    fn set_wires<W>(&mut self, wires: W, values: &[F]) -> Result<()>
    where
        W: IntoIterator<Item = Wire>,
    {
        // If we used itertools, we could use zip_eq for extra safety.
        for (wire, &value) in wires.into_iter().zip(values) {
            self.set_wire(wire, value)?;
        }

        Ok(())
    }

    fn set_ext_wires<W, const D: usize>(&mut self, wires: W, value: F::Extension) -> Result<()>
    where
        F: RichField + Extendable<D>,
        W: IntoIterator<Item = Wire>,
    {
        self.set_wires(wires, &value.to_basefield_array())
    }

    fn extend<I: Iterator<Item = (Target, F)>>(&mut self, pairs: I) -> Result<()> {
        for (t, v) in pairs {
            self.set_target(t, v)?;
        }

        Ok(())
    }
}

/// A witness holds information on the values of targets in a circuit.
pub trait Witness<F: Field>: WitnessWrite<F> {
    fn try_get_target(&self, target: Target) -> Option<F>;

    fn get_target(&self, target: Target) -> F {
        self.try_get_target(target).unwrap()
    }

    fn get_targets(&self, targets: &[Target]) -> Vec<F> {
        targets.iter().map(|&t| self.get_target(t)).collect()
    }

    fn get_extension_target<const D: usize>(&self, et: ExtensionTarget<D>) -> F::Extension
    where
        F: RichField + Extendable<D>,
    {
        F::Extension::from_basefield_array(
            self.get_targets(&et.to_target_array()).try_into().unwrap(),
        )
    }

    fn get_extension_targets<const D: usize>(&self, ets: &[ExtensionTarget<D>]) -> Vec<F::Extension>
    where
        F: RichField + Extendable<D>,
    {
        ets.iter()
            .map(|&et| self.get_extension_target(et))
            .collect()
    }

    fn get_bool_target(&self, target: BoolTarget) -> bool {
        let value = self.get_target(target.target);
        if value.is_zero() {
            return false;
        }
        if value.is_one() {
            return true;
        }
        panic!("not a bool")
    }

    fn get_hash_target(&self, ht: HashOutTarget) -> HashOut<F> {
        HashOut {
            elements: self.get_targets(&ht.elements).try_into().unwrap(),
        }
    }

    fn get_merkle_cap_target<H>(&self, cap_target: MerkleCapTarget) -> MerkleCap<F, H>
    where
        F: RichField,
        H: AlgebraicHasher<F>,
    {
        let cap = cap_target
            .0
            .iter()
            .map(|hash_target| self.get_hash_target(*hash_target))
            .collect();
        MerkleCap(cap)
    }

    fn get_wire(&self, wire: Wire) -> F {
        self.get_target(Target::Wire(wire))
    }

    fn try_get_wire(&self, wire: Wire) -> Option<F> {
        self.try_get_target(Target::Wire(wire))
    }

    fn contains(&self, target: Target) -> bool {
        self.try_get_target(target).is_some()
    }

    fn contains_all(&self, targets: &[Target]) -> bool {
        targets.iter().all(|&t| self.contains(t))
    }
}

#[derive(Clone, Debug)]
pub struct MatrixWitness<F: Field> {
    pub(crate) wire_values: Vec<Vec<F>>,
}

impl<F: Field> MatrixWitness<F> {
    pub fn get_wire(&self, gate: usize, input: usize) -> F {
        self.wire_values[input][gate]
    }
}

#[derive(Clone, Debug, Default)]
pub struct PartialWitness<F: Field> {
    pub target_values: HashMap<Target, F>,
}

impl<F: Field> PartialWitness<F> {
    pub fn new() -> Self {
        Self {
            target_values: HashMap::new(),
        }
    }
}

fn materialize_wire_values<F: Field>(
    values: &[Option<F>],
    representative_map: &[usize],
    num_wires: usize,
    degree: usize,
) -> Vec<Vec<F>> {
    let mut wire_values = (0..num_wires)
        .map(|_| Vec::with_capacity(degree))
        .collect::<Vec<_>>();
    for row in 0..degree {
        for column in 0..num_wires {
            let representative = representative_map[row * num_wires + column];
            wire_values[column].push(values[representative].unwrap_or(F::ZERO));
        }
    }
    wire_values
}

impl<F: Field> WitnessWrite<F> for PartialWitness<F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        let opt_old_value = self.target_values.insert(target, value);
        if let Some(old_value) = opt_old_value {
            if value != old_value {
                return Err(anyhow!(
                    "Target {:?} was set twice with different values: {} != {}",
                    target,
                    old_value,
                    value
                ));
            }
        }

        Ok(())
    }
}

impl<F: Field> Witness<F> for PartialWitness<F> {
    fn try_get_target(&self, target: Target) -> Option<F> {
        self.target_values.get(&target).copied()
    }
}

/// `PartitionWitness` holds a disjoint-set forest of the targets respecting a circuit's copy constraints.
/// The value of a target is defined to be the value of its root in the forest.
#[derive(Clone, Debug)]
pub struct PartitionWitness<'a, F: Field> {
    pub values: Vec<Option<F>>,
    pub representative_map: &'a [usize],
    pub num_wires: usize,
    pub degree: usize,
}

impl<'a, F: Field> PartitionWitness<'a, F> {
    pub fn new(num_wires: usize, degree: usize, representative_map: &'a [usize]) -> Self {
        Self {
            values: vec![None; representative_map.len()],
            representative_map,
            num_wires,
            degree,
        }
    }

    /// Set a `Target`. On success, returns the representative index of the newly-set target. If the
    /// target was already set, returns `None`.
    pub fn set_target_returning_rep(&mut self, target: Target, value: F) -> Result<Option<usize>> {
        let rep_index = self.representative_map[self.target_index(target)];
        let rep_value = &mut self.values[rep_index];
        if let Some(old_value) = *rep_value {
            if value != old_value {
                return Err(anyhow!(
                    "Partition containing {:?} was set twice with different values: {} != {}",
                    target,
                    old_value,
                    value
                ));
            }

            Ok(None)
        } else {
            *rep_value = Some(value);
            Ok(Some(rep_index))
        }
    }

    pub(crate) fn target_index(&self, target: Target) -> usize {
        target.index(self.num_wires, self.degree)
    }

    pub fn full_witness(self) -> MatrixWitness<F> {
        MatrixWitness {
            wire_values: materialize_wire_values(
                &self.values,
                self.representative_map,
                self.num_wires,
                self.degree,
            ),
        }
    }
}

impl<F: Field> WitnessWrite<F> for PartitionWitness<'_, F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        self.set_target_returning_rep(target, value).map(|_| ())
    }
}

impl<F: Field> Witness<F> for PartitionWitness<'_, F> {
    fn try_get_target(&self, target: Target) -> Option<F> {
        let rep_index = self.representative_map[self.target_index(target)];
        self.values[rep_index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;

    #[test]
    fn materialized_wire_values_match_legacy_for_dense_aliased_and_missing_representatives() {
        type F = GoldilocksField;

        for (num_wires, degree) in [(1, 1), (4, 3), (2, 7)] {
            let wire_count = num_wires * degree;
            let value_count = wire_count + 3;

            let dense_map = (0..value_count).collect::<Vec<_>>();
            let dense_values = (0..value_count)
                .map(|i| Some(F::from_canonical_usize(i + 1)))
                .collect::<Vec<_>>();

            let mut aliased_map = (0..value_count).collect::<Vec<_>>();
            for (i, representative) in aliased_map[..wire_count].iter_mut().enumerate() {
                *representative = wire_count + i % 2;
            }
            let mut aliased_values = vec![None; value_count];
            aliased_values[wire_count] = Some(F::from_canonical_usize(101));
            aliased_values[wire_count + 1] = Some(F::from_canonical_usize(202));

            let mut missing_map = aliased_map.clone();
            for (i, representative) in missing_map[..wire_count].iter_mut().enumerate() {
                *representative = wire_count + i % 3;
            }
            let mut missing_values = vec![None; value_count];
            missing_values[wire_count] = Some(F::from_canonical_usize(303));
            missing_values[wire_count + 2] = Some(F::from_canonical_usize(505));

            for (representative_map, values) in [
                (dense_map, dense_values),
                (aliased_map, aliased_values),
                (missing_map, missing_values),
            ] {
                let witness = PartitionWitness {
                    values,
                    representative_map: &representative_map,
                    num_wires,
                    degree,
                };
                let expected = legacy_wire_values(&witness);
                let actual = materialize_wire_values(
                    &witness.values,
                    witness.representative_map,
                    num_wires,
                    degree,
                );
                let full_witness = witness.full_witness();

                assert_eq!(actual, expected);
                assert_eq!(full_witness.wire_values, expected);
            }
        }
    }

    fn legacy_wire_values<F: Field>(witness: &PartitionWitness<F>) -> Vec<Vec<F>> {
        let mut wire_values = vec![vec![F::ZERO; witness.degree]; witness.num_wires];
        for row in 0..witness.degree {
            for column in 0..witness.num_wires {
                let target = Target::Wire(Wire { row, column });
                if let Some(value) = witness.try_get_target(target) {
                    wire_values[column][row] = value;
                }
            }
        }
        wire_values
    }
}
