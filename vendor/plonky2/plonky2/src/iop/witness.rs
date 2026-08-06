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
    /// Value of each representative slot. Unset slots hold `F::ZERO`; whether a slot has actually
    /// been set is tracked by the 1-bit-per-slot `set_bitmap`. Storing the values densely (8 bytes
    /// per slot instead of 16 for `Option<F>`) halves memory traffic during witness generation.
    pub values: Vec<F>,
    /// Bitmap with one bit per slot of `values`; bit `i` of word `i / 64` is set iff slot `i` has
    /// been assigned a value.
    pub set_bitmap: Vec<u64>,
    pub representative_map: &'a [usize],
    pub num_wires: usize,
    pub degree: usize,
}

impl<'a, F: Field> PartitionWitness<'a, F> {
    pub fn new(num_wires: usize, degree: usize, representative_map: &'a [usize]) -> Self {
        let len = representative_map.len();
        Self {
            values: vec![F::ZERO; len],
            set_bitmap: vec![0u64; len.div_ceil(64)],
            representative_map,
            num_wires,
            degree,
        }
    }

    /// Returns whether the slot at the given representative index has been set.
    #[inline]
    pub fn is_set_by_rep_index(&self, rep_index: usize) -> bool {
        (self.set_bitmap[rep_index >> 6] >> (rep_index & 63)) & 1 != 0
    }

    #[inline]
    fn mark_set(&mut self, rep_index: usize) {
        self.set_bitmap[rep_index >> 6] |= 1u64 << (rep_index & 63);
    }

    /// Set a `Target`. On success, returns the representative index of the newly-set target. If the
    /// target was already set, returns `None`.
    pub fn set_target_returning_rep(&mut self, target: Target, value: F) -> Result<Option<usize>> {
        let rep_index = self.representative_map[self.target_index(target)];
        if self.is_set_by_rep_index(rep_index) {
            let old_value = self.values[rep_index];
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
            self.values[rep_index] = value;
            self.mark_set(rep_index);
            Ok(Some(rep_index))
        }
    }

    pub(crate) fn target_index(&self, target: Target) -> usize {
        target.index(self.num_wires, self.degree)
    }

    pub fn full_witness(self) -> MatrixWitness<F> {
        // Split the fused row-major read / column-major write across disjoint row chunks.
        // Every task owns one row range in every column; the vectors keep length zero until all
        // tasks finish, so unwinding before `set_len` drops no uninitialized elements.
        let num_wires = self.num_wires;
        let degree = self.degree;
        let mut wire_values: Vec<Vec<F>> =
            (0..num_wires).map(|_| Vec::with_capacity(degree)).collect();

        #[cfg(feature = "parallel")]
        let use_parallel = degree.saturating_mul(num_wires) >= 1 << 15
            && plonky2_maybe_rayon::rayon::current_num_threads() > 1;
        #[cfg(not(feature = "parallel"))]
        let use_parallel = false;
        if !use_parallel {
            let mut wire_index = 0;
            for _ in 0..degree {
                for column in &mut wire_values {
                    column.push(self.values[self.representative_map[wire_index]]);
                    wire_index += 1;
                }
            }
            return MatrixWitness { wire_values };
        }

        let num_chunks = 16.min(degree.max(1));
        let chunk_rows = degree.div_ceil(num_chunks);
        {
            let mut segments: Vec<Vec<&mut [core::mem::MaybeUninit<F>]>> = (0..num_chunks)
                .map(|_| Vec::with_capacity(num_wires))
                .collect();
            for column in &mut wire_values {
                let mut rest = &mut column.spare_capacity_mut()[..degree];
                for chunk_columns in &mut segments {
                    let take = chunk_rows.min(rest.len());
                    let (head, tail) = rest.split_at_mut(take);
                    chunk_columns.push(head);
                    rest = tail;
                }
                debug_assert!(rest.is_empty());
            }

            use plonky2_maybe_rayon::*;
            segments
                .par_iter_mut()
                .enumerate()
                .for_each(|(chunk, columns)| {
                    let rows = columns.first().map_or(0, |column| column.len());
                    debug_assert!(columns.iter().all(|column| column.len() == rows));
                    let mut wire_index = chunk * chunk_rows * num_wires;
                    for row in 0..rows {
                        for column in columns.iter_mut() {
                            column[row].write(self.values[self.representative_map[wire_index]]);
                            wire_index += 1;
                        }
                    }
                    debug_assert_eq!(wire_index, (chunk * chunk_rows + rows) * num_wires);
                });
        }

        for column in &mut wire_values {
            // SAFETY: the disjoint segments cover exactly `degree` slots in every column, and the
            // completed parallel traversal writes every slot exactly once before this point.
            unsafe {
                column.set_len(degree);
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
        if self.is_set_by_rep_index(rep_index) {
            Some(self.values[rep_index])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;

    fn serial_full_witness_reference<F: Field>(witness: &PartitionWitness<F>) -> Vec<Vec<F>> {
        let mut wire_values: Vec<Vec<F>> = (0..witness.num_wires)
            .map(|_| Vec::with_capacity(witness.degree))
            .collect();
        let mut wire_index = 0;
        for _ in 0..witness.degree {
            for column in &mut wire_values {
                column.push(witness.values[witness.representative_map[wire_index]]);
                wire_index += 1;
            }
        }
        wire_values
    }

    #[test]
    fn full_witness_matches_serial_reference_across_chunk_boundaries() {
        type F = GoldilocksField;

        let run_cases = || {
            for &degree in &[0usize, 1, 2, 15, 16, 17, 31, 32, 33, 257, 8_192, 8_193] {
                for &num_wires in &[0usize, 1, 2, 5] {
                    let num_wire_targets = degree * num_wires;
                    let num_targets = num_wire_targets + usize::from(num_wire_targets != 0) * 3;
                    let num_representatives = num_targets.div_ceil(2).max(1);
                    let representative_map = (0..num_targets)
                        .map(|i| (i.wrapping_mul(7).wrapping_add(3)) % num_representatives)
                        .collect::<Vec<_>>();
                    let mut witness =
                        PartitionWitness::<F>::new(num_wires, degree, &representative_map);
                    for (i, value) in witness.values.iter_mut().enumerate() {
                        // Leave every third representative at its unset zero value. The remaining
                        // representatives deliberately alias many wire cells through the map.
                        if i % 3 != 0 {
                            *value = F::from_canonical_usize(i.wrapping_mul(11).wrapping_add(5));
                        }
                    }

                    let expected = serial_full_witness_reference(&witness);
                    let actual = witness.full_witness().wire_values;

                    assert_eq!(actual, expected, "degree={degree}, num_wires={num_wires}");
                    assert_eq!(actual.len(), num_wires);
                    assert!(actual.iter().all(|column| column.len() == degree));
                }
            }
        };

        #[cfg(feature = "parallel")]
        for threads in [1, 2, 4, 16] {
            plonky2_maybe_rayon::rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(run_cases);
        }
        #[cfg(not(feature = "parallel"))]
        run_cases();
    }

    #[test]
    fn full_witness_panic_paths_drop_partially_initialized_storage_safely() {
        type F = GoldilocksField;

        let run_cases = || {
            for malformed_map in ["short", "out-of-range"] {
                let degree = 8_192;
                let num_wires = 5;
                let required = degree * num_wires;
                let mut representative_map = (0..required).collect::<Vec<_>>();
                if malformed_map == "short" {
                    representative_map.pop();
                } else {
                    representative_map[required - 1] = required;
                }
                let mut witness =
                    PartitionWitness::<F>::new(num_wires, degree, &representative_map);
                witness.values.fill(F::ONE);

                let result = std::panic::catch_unwind(|| witness.full_witness());
                assert!(result.is_err(), "malformed_map={malformed_map}");
            }
        };

        #[cfg(feature = "parallel")]
        for threads in [1, 2, 4, 16] {
            plonky2_maybe_rayon::rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(run_cases);
        }
        #[cfg(not(feature = "parallel"))]
        run_cases();
    }
}
