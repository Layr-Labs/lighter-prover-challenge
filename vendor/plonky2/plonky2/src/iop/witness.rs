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
///
/// Representative values are stored sparsely: one *uninitialized* 8-byte value slot per
/// representative plus a compact bitmap (one bit per representative, packed into `u64` words)
/// recording which slots hold initialized values. Compared to the former eagerly-initialized
/// `Vec<Option<F>>` (16 bytes per slot for Goldilocks, all written to `None` up front), the
/// constructor writes only one zeroed bit per representative and value pages are touched only
/// when a representative is actually assigned.
///
/// The central safety invariant is:
///
/// > If bitmap bit `i` is set, value slot `i` contains a fully initialized `F`.
///
/// It is upheld because (1) both storage fields are private to this module, (2) the only
/// initializer, [`Self::init_representative_slot`], writes `values[i]` first and (3) sets bitmap
/// bit `i` only after that write completes, (4) bits are never cleared and initialized slots are
/// never deinitialized, (5) every value read first checks the bit through the guarded
/// representative accessor, and (6) all mutation requires `&mut self`, so in safe Rust a reader
/// cannot observe a partially completed write.
pub struct PartitionWitness<'a, F: Field> {
    /// One value slot per representative. Slot `i` is initialized if and only if bit `i` of
    /// `initialized` is set. Private: all access goes through the guarded accessors below.
    values: Vec<MaybeUninit<F>>,
    /// Initialization bitmap: bit `i` (word `i / 64`, bit `i % 64`) records whether `values[i]`
    /// holds an initialized value. Bits are set only after the slot write completes and are
    /// never cleared. Private for the same reason as `values`.
    initialized: Vec<u64>,
    pub representative_map: &'a [usize],
    pub num_wires: usize,
    pub degree: usize,
}

impl<'a, F: Field> PartitionWitness<'a, F> {
    pub fn new(num_wires: usize, degree: usize, representative_map: &'a [usize]) -> Self {
        let len = representative_map.len();
        let mut values = Vec::with_capacity(len);
        // SAFETY (unsafe site 1 of 2): the reserved capacity is `len`, and `MaybeUninit<F>` is
        // valid for any bit pattern — including uninitialized memory — while dropping it never
        // reads the underlying `F`. No slot is read until the matching bit of `initialized`
        // (allocated all-zero below) is set, which `init_representative_slot` does only after
        // fully writing the slot.
        unsafe {
            values.set_len(len);
        }
        Self {
            values,
            initialized: vec![0u64; len.div_ceil(64)],
            representative_map,
            num_wires,
            degree,
        }
    }

    /// Returns whether the value slot for representative `rep_index` has been initialized.
    ///
    /// This is the guarded representative-set query: presence needs only a bitmap word read,
    /// never the value storage.
    #[inline]
    pub fn is_representative_set(&self, rep_index: usize) -> bool {
        self.initialized[rep_index / 64] & (1u64 << (rep_index % 64)) != 0
    }

    /// Guarded read of a representative's value. Checks the initialization bitmap before
    /// touching the value slot; an unset representative reads as `None`.
    ///
    /// Out-of-range indices are handled by the safe bitmap/vector indexing, which panics before
    /// any unsafe read; tail bitmap bits (past `values.len()`) are never set, so they cannot
    /// manufacture a read.
    #[inline]
    pub(crate) fn representative_value(&self, rep_index: usize) -> Option<F> {
        if self.is_representative_set(rep_index) {
            // SAFETY (unsafe site 2 of 2): bit `rep_index` was observed set. Bits are set only
            // by `init_representative_slot` after the slot is fully written, are never cleared,
            // and slots are never deinitialized, so the slot holds a valid `F`. All mutation
            // requires `&mut self`, so this shared read cannot race a partial write. `F: Field`
            // is `Copy`, so copying the value out leaves the slot initialized.
            Some(*unsafe { self.values[rep_index].assume_init_ref() })
        } else {
            None
        }
    }

    /// The only initializer of value slots and the only setter of bitmap bits.
    ///
    /// Writes the value slot first and publishes the bit only after the write completes, so a
    /// set bit always denotes a fully initialized slot. Bits are never cleared. The safe
    /// (bounds-checked) slot write panics on an out-of-range representative before its bit
    /// could be set, so tail bitmap bits stay clear.
    #[inline]
    fn init_representative_slot(&mut self, rep_index: usize, value: F) {
        self.values[rep_index] = MaybeUninit::new(value);
        self.initialized[rep_index / 64] |= 1u64 << (rep_index % 64);
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
            self.init_representative_slot(rep_index, value);
            Ok(Some(rep_index))
        }
    }

    pub(crate) fn target_index(&self, target: Target) -> usize {
        target.index(self.num_wires, self.degree)
    }

    pub fn full_witness(self) -> MatrixWitness<F> {
        // Redraw ticket 5.
        // Single fused pass. Cell (column j, row i) is the value of representative
        // `representative_map[i * num_wires + j]` or zero — the same lookup
        // `try_get_target(Target::Wire { row: i, column: j })` resolved to, with
        // `Target::index`'s `row * num_wires + column` inlined as a running cursor. This
        // deletes the full zero-prefill pass over the matrix and the per-cell Target
        // construction and index arithmetic of the second pass, while reading
        // `representative_map` (and, through the guarded accessor, the compact
        // initialization bitmap) sequentially.
        let mut wire_values: Vec<Vec<F>> = (0..self.num_wires)
            .map(|_| Vec::with_capacity(self.degree))
            .collect();
        let mut wire_index = 0;
        for _ in 0..self.degree {
            for column in wire_values.iter_mut() {
                column.push(
                    self.representative_value(self.representative_map[wire_index])
                        .unwrap_or(F::ZERO),
                );
                wire_index += 1;
            }
        }

        MatrixWitness { wire_values }
    }
}

impl<F: Field> Clone for PartitionWitness<'_, F> {
    fn clone(&self) -> Self {
        // Allocate a fresh sparse witness and copy only initialized slots: visit set bitmap
        // bits, read each through the guarded accessor, and initialize the cloned slot through
        // the same write-before-bit helper. Uninitialized slots are never read.
        let mut clone = Self::new(self.num_wires, self.degree, self.representative_map);
        for (word_index, &word) in self.initialized.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let rep_index = word_index * 64 + remaining.trailing_zeros() as usize;
                let value = self
                    .representative_value(rep_index)
                    .expect("set bitmap bit must denote an initialized slot");
                clone.init_representative_slot(rep_index, value);
                remaining &= remaining - 1;
            }
        }
        clone
    }
}

impl<F: Field> fmt::Debug for PartitionWitness<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Present the prior logical `Option`-style value list: every representative index is
        // formatted as `Some(value)` or `None` through the guarded accessor. The raw
        // `MaybeUninit` storage is never exposed or read directly.
        struct OptionStyleValues<'s, 'a, F: Field>(&'s PartitionWitness<'a, F>);

        impl<F: Field> fmt::Debug for OptionStyleValues<'_, '_, F> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_list()
                    .entries((0..self.0.values.len()).map(|i| self.0.representative_value(i)))
                    .finish()
            }
        }

        f.debug_struct("PartitionWitness")
            .field("values", &OptionStyleValues(self))
            .field("representative_map", &self.representative_map)
            .field("num_wires", &self.num_wires)
            .field("degree", &self.degree)
            .finish()
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
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use anyhow::Result;

    use super::*;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    /// Deterministic xorshift64 PRNG so the differential test is reproducible.
    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Dense reference oracle: the former `Vec<Option<F>>` storage with the verbatim legacy
    /// write-once semantics, used to differentially validate the sparse bitmap witness.
    struct DenseShadow<'a, F: Field> {
        values: Vec<Option<F>>,
        representative_map: &'a [usize],
        num_wires: usize,
        degree: usize,
    }

    impl<'a, F: Field> DenseShadow<'a, F> {
        fn new(num_wires: usize, degree: usize, representative_map: &'a [usize]) -> Self {
            Self {
                values: vec![None; representative_map.len()],
                representative_map,
                num_wires,
                degree,
            }
        }

        fn set_target_returning_rep(
            &mut self,
            target: Target,
            value: F,
        ) -> Result<Option<usize>> {
            let rep_index = self.representative_map[target.index(self.num_wires, self.degree)];
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

        fn try_get_target(&self, target: Target) -> Option<F> {
            let rep_index = self.representative_map[target.index(self.num_wires, self.degree)];
            self.values[rep_index]
        }
    }

    #[test]
    fn sparse_allocation_shape() {
        let rep_map: Vec<usize> = (0..130).collect();
        let w = PartitionWitness::<F>::new(130, 1, &rep_map);
        assert_eq!(w.values.len(), 130);
        // One bit per representative, packed into u64 words: ceil(130 / 64) = 3.
        assert_eq!(w.initialized.len(), 3);
        assert!(w.initialized.iter().all(|&word| word == 0));
        // A zero-length map still allocates a consistent (empty) shape.
        let empty_map: [usize; 0] = [];
        let empty = PartitionWitness::<F>::new(0, 0, &empty_map);
        assert_eq!(empty.values.len(), 0);
        assert_eq!(empty.initialized.len(), 0);
    }

    #[test]
    fn unset_and_initialized_reads_through_aliases() {
        // 2 wires x 2 rows = wire indices 0..4, then 2 virtual targets (indices 4, 5).
        // Virtual target 0 (index 4) aliases wire (0, 0) (index 0).
        let rep_map = [0usize, 1, 2, 3, 0, 5];
        let mut w = PartitionWitness::<F>::new(2, 2, &rep_map);
        let wire00 = Target::wire(0, 0);
        let virt0 = Target::VirtualTarget { index: 0 };

        assert_eq!(w.try_get_target(wire00), None);
        assert_eq!(w.try_get_target(virt0), None);
        assert!(!w.contains(wire00));
        assert!(!w.is_representative_set(0));

        let seven = F::from_canonical_u64(7);
        assert_eq!(w.set_target_returning_rep(virt0, seven).unwrap(), Some(0));

        // Both aliases resolve through the representative map to the same guarded slot.
        assert!(w.is_representative_set(0));
        assert_eq!(w.try_get_target(wire00), Some(seven));
        assert_eq!(w.try_get_target(virt0), Some(seven));
        assert_eq!(w.get_target(virt0), seven);
        // Unrelated targets remain unset.
        assert_eq!(w.try_get_target(Target::wire(0, 1)), None);
        assert_eq!(w.try_get_target(Target::VirtualTarget { index: 1 }), None);
    }

    #[test]
    fn double_set_same_and_conflicting_values() {
        let rep_map = [0usize, 1, 2, 3, 0, 5];
        let mut w = PartitionWitness::<F>::new(2, 2, &rep_map);
        let wire00 = Target::wire(0, 0);
        let virt0 = Target::VirtualTarget { index: 0 }; // aliases wire00

        let seven = F::from_canonical_u64(7);
        let eight = F::from_canonical_u64(8);

        // First assignment returns the representative index.
        assert_eq!(w.set_target_returning_rep(wire00, seven).unwrap(), Some(0));
        // Reassigning the identical value (through an alias) returns None.
        assert_eq!(w.set_target_returning_rep(virt0, seven).unwrap(), None);
        // Invariant: bits are never cleared.
        assert!(w.is_representative_set(0));

        // A conflicting reassignment fails with the exact legacy message and must not
        // overwrite the original value.
        let err = w.set_target_returning_rep(virt0, eight).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "Partition containing {:?} was set twice with different values: {} != {}",
                virt0, seven, eight
            )
        );
        assert!(w.is_representative_set(0));
        assert_eq!(w.try_get_target(wire00), Some(seven));

        // WitnessWrite::set_target surfaces the identical error.
        let err2 = w.set_target(wire00, eight).unwrap_err();
        assert_eq!(
            err2.to_string(),
            format!(
                "Partition containing {:?} was set twice with different values: {} != {}",
                wire00, seven, eight
            )
        );
        assert_eq!(w.try_get_target(virt0), Some(seven));
    }

    #[test]
    fn clone_is_independent_and_sparse() {
        let rep_map: Vec<usize> = (0..100).collect();
        let mut w = PartitionWitness::<F>::new(100, 1, &rep_map);
        for i in (0..100).step_by(7) {
            w.set_target(Target::wire(0, i), F::from_canonical_usize(i))
                .unwrap();
        }

        let mut c = w.clone();
        // The clone reproduces exactly the set bits and their values, nothing else.
        assert_eq!(c.initialized, w.initialized);
        for i in 0..100 {
            assert_eq!(c.representative_value(i), w.representative_value(i));
        }

        // Mutating the clone does not affect the original, and vice versa.
        c.set_target(Target::wire(0, 1), F::from_canonical_u64(41))
            .unwrap();
        assert!(c.is_representative_set(1));
        assert!(!w.is_representative_set(1));
        w.set_target(Target::wire(0, 2), F::from_canonical_u64(42))
            .unwrap();
        assert!(!c.is_representative_set(2));
    }

    #[test]
    fn debug_presents_option_style_values() {
        let rep_map = [0usize, 1, 2];
        let mut w = PartitionWitness::<F>::new(3, 1, &rep_map);
        w.set_target(Target::wire(0, 1), F::from_canonical_u64(5))
            .unwrap();
        let debug = format!("{w:?}");
        // The logical Option-style value list of the former derived Debug is preserved.
        assert!(debug.contains("values: [None, Some(5), None]"), "{debug}");
        assert!(debug.contains("num_wires: 3"), "{debug}");
        assert!(debug.contains("degree: 1"), "{debug}");
        // The raw storage representation must never leak.
        assert!(!debug.contains("MaybeUninit"), "{debug}");
        assert!(!debug.contains("initialized"), "{debug}");
    }

    #[test]
    fn bitmap_word_boundaries() {
        let rep_map: Vec<usize> = (0..130).collect();
        // num_wires = 130, degree = 1: Target::wire(0, c) has index c.
        let mut w = PartitionWitness::<F>::new(130, 1, &rep_map);
        for &i in &[63usize, 64, 127, 128] {
            assert!(!w.is_representative_set(i));
            w.set_target(Target::wire(0, i), F::from_canonical_usize(i))
                .unwrap();
        }
        assert_eq!(w.initialized[0], 1u64 << 63);
        assert_eq!(w.initialized[1], 1u64 | (1u64 << 63));
        assert_eq!(w.initialized[2], 1u64);
        for &i in &[63usize, 64, 127, 128] {
            assert!(w.is_representative_set(i));
            assert_eq!(w.representative_value(i), Some(F::from_canonical_usize(i)));
        }
        for &i in &[0usize, 62, 65, 126, 129] {
            assert!(!w.is_representative_set(i));
            assert_eq!(w.representative_value(i), None);
        }
    }

    #[test]
    fn tail_bitmap_bits_never_set() {
        let rep_map: Vec<usize> = (0..70).collect();
        let mut w = PartitionWitness::<F>::new(70, 1, &rep_map);
        for i in 0..70 {
            w.set_target(Target::wire(0, i), F::from_canonical_usize(i))
                .unwrap();
        }
        assert_eq!(w.initialized[0], u64::MAX);
        // Only bits 64..=69 of the last word: the tail bits 70..127 stay clear even with
        // every representative assigned.
        assert_eq!(w.initialized[1], (1u64 << 6) - 1);
        // Cloning a fully-set witness visits exactly the set bits and reproduces the bitmap.
        let c = w.clone();
        assert_eq!(c.initialized, w.initialized);
        for i in 0..70 {
            assert_eq!(c.representative_value(i), Some(F::from_canonical_usize(i)));
        }
    }

    #[test]
    fn out_of_range_set_panics_before_publishing_a_bit() {
        // The single target maps to representative 10, which is inside the (one-word) bitmap
        // range but outside the value storage. The guarded presence probe reads bit 10 as
        // unset, then the safe bounds-checked slot write panics before the bit publish.
        let rep_map = [10usize];
        let mut w = PartitionWitness::<F>::new(1, 1, &rep_map);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = w.set_target_returning_rep(Target::wire(0, 0), F::ONE);
        }));
        assert!(
            result.is_err(),
            "out-of-range representative must panic on the safe slot write"
        );
        // The write-before-bit ordering means the failed write published nothing.
        assert!(w.initialized.iter().all(|&word| word == 0));
        assert_eq!(w.representative_value(0), None);
    }

    #[test]
    #[should_panic]
    fn out_of_range_read_panics_on_safe_bitmap_indexing() {
        let rep_map: Vec<usize> = (0..64).collect();
        let w = PartitionWitness::<F>::new(64, 1, &rep_map);
        // Representative 64 lies past the single-word bitmap: safe indexing panics before any
        // unsafe read could be attempted.
        let _ = w.is_representative_set(64);
    }

    #[test]
    fn full_witness_alias_expansion_and_zero_fill() {
        // 2 wires x 2 rows. Wire (1, 0) (index 2) aliases wire (0, 0) (index 0).
        let rep_map = [0usize, 1, 0, 3];
        let mut w = PartitionWitness::<F>::new(2, 2, &rep_map);
        let five = F::from_canonical_u64(5);
        let nine = F::from_canonical_u64(9);
        w.set_target(Target::wire(0, 0), five).unwrap();
        w.set_target(Target::wire(1, 1), nine).unwrap();

        let matrix = w.full_witness();
        assert_eq!(matrix.get_wire(0, 0), five);
        // Unset wires are zero-filled.
        assert_eq!(matrix.get_wire(0, 1), F::ZERO);
        // Aliased wires expand to their representative's value.
        assert_eq!(matrix.get_wire(1, 0), five);
        assert_eq!(matrix.get_wire(1, 1), nine);
    }

    #[test]
    fn differential_against_dense_option_shadow() {
        const NUM_WIRES: usize = 8;
        const DEGREE: usize = 32;
        const NUM_VIRTUAL: usize = 256;
        let len = NUM_WIRES * DEGREE + NUM_VIRTUAL;

        let mut rng = 0x243F_6A88_85A3_08D3u64;

        // A random disjoint-set forest: roots map to themselves; about a quarter of the
        // indices alias the representative of an earlier index.
        let mut rep_map: Vec<usize> = Vec::with_capacity(len);
        for i in 0..len {
            if i > 0 && xorshift(&mut rng) % 4 == 0 {
                let j = (xorshift(&mut rng) % i as u64) as usize;
                rep_map.push(rep_map[j]);
            } else {
                rep_map.push(i);
            }
        }

        let mut sparse = PartitionWitness::<F>::new(NUM_WIRES, DEGREE, &rep_map);
        let mut shadow = DenseShadow::<F>::new(NUM_WIRES, DEGREE, &rep_map);

        let (mut saw_fresh, mut saw_repeat, mut saw_conflict) = (false, false, false);
        for _ in 0..4000 {
            let target = if xorshift(&mut rng) % 2 == 0 {
                Target::wire(
                    (xorshift(&mut rng) % DEGREE as u64) as usize,
                    (xorshift(&mut rng) % NUM_WIRES as u64) as usize,
                )
            } else {
                Target::VirtualTarget {
                    index: (xorshift(&mut rng) % NUM_VIRTUAL as u64) as usize,
                }
            };
            // A tiny value space forces plenty of identical re-sets and conflicts.
            let value = F::from_canonical_u64(xorshift(&mut rng) % 3);

            let sparse_result = sparse.set_target_returning_rep(target, value);
            let shadow_result = shadow.set_target_returning_rep(target, value);
            match (sparse_result, shadow_result) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a, b);
                    match a {
                        Some(_) => saw_fresh = true,
                        None => saw_repeat = true,
                    }
                }
                (Err(a), Err(b)) => {
                    // Identical conflict errors, message included.
                    assert_eq!(a.to_string(), b.to_string());
                    saw_conflict = true;
                }
                (a, b) => panic!("divergent outcomes: {a:?} vs {b:?}"),
            }
        }
        assert!(
            saw_fresh && saw_repeat && saw_conflict,
            "differential run must exercise fresh sets, repeats, and conflicts"
        );

        // Identical None/Some population and identical values for every representative.
        for rep_index in 0..len {
            assert_eq!(
                sparse.is_representative_set(rep_index),
                shadow.values[rep_index].is_some()
            );
            assert_eq!(
                sparse.representative_value(rep_index),
                shadow.values[rep_index]
            );
        }

        // Identical reads for every target, wire and virtual alike.
        for row in 0..DEGREE {
            for column in 0..NUM_WIRES {
                let t = Target::wire(row, column);
                assert_eq!(sparse.try_get_target(t), shadow.try_get_target(t));
            }
        }
        for index in 0..NUM_VIRTUAL {
            let t = Target::VirtualTarget { index };
            assert_eq!(sparse.try_get_target(t), shadow.try_get_target(t));
        }

        // The fused full-witness pass matches the legacy per-cell resolution.
        let matrix = sparse.clone().full_witness();
        for row in 0..DEGREE {
            for column in 0..NUM_WIRES {
                let expected = shadow
                    .try_get_target(Target::wire(row, column))
                    .unwrap_or(F::ZERO);
                assert_eq!(matrix.get_wire(row, column), expected);
            }
        }
    }

    #[test]
    fn sparse_witness_end_to_end_proof() -> Result<()> {
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let x = builder.add_virtual_target();
        let x_squared = builder.mul(x, x);
        builder.register_public_input(x_squared);
        let data = builder.build::<C>();

        let mut pw = PartialWitness::new();
        pw.set_target(x, F::from_canonical_u64(3))?;
        let proof = data.prove(pw)?;
        assert_eq!(proof.public_inputs[0], F::from_canonical_u64(9));
        verify(proof, &data.verifier_only, &data.common)
    }
}
