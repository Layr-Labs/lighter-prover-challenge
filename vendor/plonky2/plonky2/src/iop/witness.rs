#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::fmt;
use core::iter::zip;
use core::mem::MaybeUninit;

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
pub struct PartitionWitness<'a, F: Field> {
    values: Vec<MaybeUninit<F>>,
    initialized: Vec<u64>,
    pub representative_map: &'a [usize],
    pub num_wires: usize,
    pub degree: usize,
}

impl<'a, F: Field> Clone for PartitionWitness<'a, F> {
    fn clone(&self) -> Self {
        let mut cloned = Self::new(self.num_wires, self.degree, self.representative_map);
        for (word_index, &word) in self.initialized.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit_index = remaining.trailing_zeros() as usize;
                let rep_index = word_index * u64::BITS as usize + bit_index;
                let value = self
                    .representative_value(rep_index)
                    .expect("set bitmap bit must have a value");
                cloned.initialize_representative(rep_index, value);
                remaining &= remaining - 1;
            }
        }
        cloned
    }
}

struct PartitionValuesDebug<'w, 'map, F: Field>(&'w PartitionWitness<'map, F>);

impl<F: Field> fmt::Debug for PartitionValuesDebug<'_, '_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        for rep_index in 0..self.0.values.len() {
            list.entry(&self.0.representative_value(rep_index));
        }
        list.finish()
    }
}

impl<F: Field> fmt::Debug for PartitionWitness<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartitionWitness")
            .field("values", &PartitionValuesDebug(self))
            .field("representative_map", &self.representative_map)
            .field("num_wires", &self.num_wires)
            .field("degree", &self.degree)
            .finish()
    }
}

impl<'a, F: Field> PartitionWitness<'a, F> {
    pub fn new(num_wires: usize, degree: usize, representative_map: &'a [usize]) -> Self {
        let mut values = Vec::with_capacity(representative_map.len());
        unsafe {
            // SAFETY: `MaybeUninit<F>` may hold uninitialized storage, and its drop implementation
            // never reads the contained `F`. Every read is guarded by `initialized` below.
            values.set_len(representative_map.len());
        }
        Self {
            values,
            initialized: vec![0; representative_map.len().div_ceil(u64::BITS as usize)],
            representative_map,
            num_wires,
            degree,
        }
    }

    /// Set a `Target`. On success, returns the representative index of the newly-set target. If the
    /// target was already set, returns `None`.
    pub fn set_target_returning_rep(&mut self, target: Target, value: F) -> Result<Option<usize>> {
        let rep_index = self.representative_map[self.target_index(target)];
        if let Some(old_value) = self.representative_value(rep_index) {
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
            self.initialize_representative(rep_index, value);
            Ok(Some(rep_index))
        }
    }

    #[inline]
    fn initialize_representative(&mut self, rep_index: usize, value: F) {
        debug_assert!(!self.is_representative_set(rep_index));
        self.values[rep_index].write(value);
        self.initialized[rep_index / u64::BITS as usize] |= 1 << (rep_index % u64::BITS as usize);
    }

    #[inline]
    pub(crate) fn is_representative_set(&self, rep_index: usize) -> bool {
        self.initialized[rep_index / u64::BITS as usize] & (1 << (rep_index % u64::BITS as usize))
            != 0
    }

    #[inline]
    fn representative_value(&self, rep_index: usize) -> Option<F> {
        if !self.is_representative_set(rep_index) {
            return None;
        }

        unsafe {
            // SAFETY: A bit is set only after `values[rep_index]` has been initialized, and no
            // initialized slot is ever deinitialized. `F: Field` is `Copy`, so returning it leaves
            // the initialized slot intact.
            Some(*self.values[rep_index].assume_init_ref())
        }
    }

    pub(crate) fn target_index(&self, target: Target) -> usize {
        target.index(self.num_wires, self.degree)
    }

    pub fn full_witness(self) -> MatrixWitness<F> {
        let mut wire_values = vec![vec![F::ZERO; self.degree]; self.num_wires];
        for i in 0..self.degree {
            for j in 0..self.num_wires {
                let t = Target::Wire(Wire { row: i, column: j });
                if let Some(x) = self.try_get_target(t) {
                    wire_values[j][i] = x;
                }
            }
        }

        MatrixWitness { wire_values }
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
        self.representative_value(rep_index)
    }
}

#[cfg(test)]
mod tests {
    use core::mem::{size_of, size_of_val};

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;

    #[test]
    fn partition_witness_allocates_uninitialized_field_slots_and_a_compact_bitmap() {
        let representative_map = (0..130).collect::<Vec<_>>();
        let witness = PartitionWitness::<GoldilocksField>::new(2, 2, &representative_map);

        assert_eq!(
            size_of_val(witness.values.as_slice()),
            representative_map.len() * size_of::<GoldilocksField>()
        );
        assert_eq!(witness.initialized.len(), 3);
        assert!(witness.initialized.iter().all(|&word| word == 0));
    }

    #[test]
    fn partition_witness_reads_initialized_representatives_through_aliases() {
        let representative_map = [0, 0, 2, 3];
        let mut witness = PartitionWitness::<GoldilocksField>::new(2, 2, &representative_map);
        let original = Target::wire(0, 0);
        let alias = Target::wire(0, 1);
        let unset = Target::wire(1, 0);
        let value = GoldilocksField::from_canonical_u64(17);

        assert_eq!(witness.try_get_target(original), None);
        assert_eq!(witness.try_get_target(alias), None);
        assert_eq!(
            witness.set_target_returning_rep(original, value).unwrap(),
            Some(0)
        );
        assert_eq!(witness.try_get_target(original), Some(value));
        assert_eq!(witness.try_get_target(alias), Some(value));
        assert_eq!(witness.try_get_target(unset), None);
    }

    #[test]
    fn partition_witness_rejects_conflicting_alias_without_overwriting_value() {
        let representative_map = [0, 0];
        let mut witness = PartitionWitness::<GoldilocksField>::new(2, 1, &representative_map);
        let original = Target::wire(0, 0);
        let alias = Target::wire(0, 1);
        let original_value = GoldilocksField::from_canonical_u64(17);

        assert_eq!(
            witness
                .set_target_returning_rep(original, original_value)
                .unwrap(),
            Some(0)
        );
        assert_eq!(
            witness
                .set_target_returning_rep(alias, original_value)
                .unwrap(),
            None
        );
        let error = witness
            .set_target(alias, GoldilocksField::from_canonical_u64(18))
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("set twice with different values"));
        assert_eq!(witness.get_target(original), original_value);
        assert_eq!(witness.get_target(alias), original_value);
        assert_eq!(witness.initialized[0].count_ones(), 1);
    }

    #[test]
    fn partition_witness_clone_and_debug_expose_only_logical_values() {
        let representative_map = [0, 1];
        let mut witness = PartitionWitness::<GoldilocksField>::new(2, 1, &representative_map);
        let initialized = Target::wire(0, 0);
        let unset = Target::wire(0, 1);
        let value = GoldilocksField::from_canonical_u64(17);
        witness.set_target(initialized, value).unwrap();

        let mut cloned = witness.clone();
        assert_eq!(cloned.try_get_target(initialized), Some(value));
        assert_eq!(cloned.try_get_target(unset), None);
        let rendered = format!("{cloned:?}");
        assert!(rendered.contains("values: [Some(17), None]"));
        assert!(!rendered.contains("MaybeUninit"));

        cloned
            .set_target(unset, GoldilocksField::from_canonical_u64(23))
            .unwrap();
        assert_eq!(witness.try_get_target(unset), None);
    }

    #[test]
    fn partition_witness_bitmap_handles_word_boundaries() {
        let representative_map = (0..130).collect::<Vec<_>>();
        let mut witness = PartitionWitness::<GoldilocksField>::new(1, 130, &representative_map);

        for row in [0, 63, 64, 127, 128, 129] {
            witness
                .set_target(
                    Target::wire(row, 0),
                    GoldilocksField::from_canonical_usize(row + 1),
                )
                .unwrap();
        }

        assert_eq!(witness.initialized, vec![1 | (1 << 63), 1 | (1 << 63), 3]);
        for row in 0..130 {
            let expected = [0, 63, 64, 127, 128, 129]
                .contains(&row)
                .then(|| GoldilocksField::from_canonical_usize(row + 1));
            assert_eq!(witness.try_get_target(Target::wire(row, 0)), expected);
        }
    }

    #[test]
    fn partition_witness_full_witness_preserves_aliases_and_zero_fills_unset_wires() {
        let representative_map = [0, 0, 2, 3];
        let mut witness = PartitionWitness::<GoldilocksField>::new(2, 2, &representative_map);
        let aliased_value = GoldilocksField::from_canonical_u64(17);
        let last_value = GoldilocksField::from_canonical_u64(23);
        witness
            .set_target(Target::wire(0, 0), aliased_value)
            .unwrap();
        witness.set_target(Target::wire(1, 1), last_value).unwrap();

        let full = witness.full_witness();

        assert_eq!(
            full.wire_values,
            vec![
                vec![aliased_value, GoldilocksField::ZERO],
                vec![aliased_value, last_value]
            ]
        );
    }

    #[test]
    fn partition_witness_sparse_storage_produces_a_valid_proof() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;

        let mut builder =
            CircuitBuilder::<GoldilocksField, D>::new(CircuitConfig::standard_recursion_config());
        let input = builder.add_virtual_target();
        let square = builder.square(input);
        builder.register_public_input(square);
        let circuit = builder.build::<C>();
        let mut partial_witness = PartialWitness::new();
        partial_witness
            .set_target(input, GoldilocksField::from_canonical_u64(7))
            .unwrap();

        let proof = circuit.prove(partial_witness).unwrap();

        assert_eq!(
            proof.public_inputs,
            vec![GoldilocksField::from_canonical_u64(49)]
        );
        circuit.verify(proof).unwrap();
    }
}
