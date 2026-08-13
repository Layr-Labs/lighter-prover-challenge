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

/// Largest circuit for which the prover retains the runtime-only direct-IFFT
/// representative plan. The dynamic final block is degree 2^18 and remains on
/// the matrix fallback, bounding plan residency to the historical H41 shapes.
pub const MAX_DIRECT_WIRE_IFFT_DEGREE: usize = 1 << 16;

/// Whether a circuit has the exact production shape admitted to full H41.
/// Plan construction and proof dispatch share this predicate so no fallback
/// circuit allocates or consumes a direct-IFFT plan.
pub fn direct_wire_ifft_shape_is_eligible(
    num_wires: usize,
    num_routed_wires: usize,
    num_challenges: usize,
    degree: usize,
    has_lookup: bool,
) -> bool {
    !has_lookup
        && num_wires == 136
        && num_routed_wires == 80
        && num_challenges == 2
        && (degree == 1 << 14 || degree == 1 << 16)
}

/// Only the full-H41 experiment arm owns the historical circuit-fixed plan.
/// All three arms parse the same process-fixed selector during circuit load;
/// invalid values fail closed before any proof can be labeled.
#[cfg(feature = "std")]
pub fn direct_wire_ifft_plan_is_requested() -> bool {
    match std::env::var("LIGHTER_H41_ARM").as_deref() {
        Ok("full-h41") => true,
        Err(std::env::VarError::NotPresent) | Ok("control") | Ok("hybrid") => false,
        Ok(other) => panic!(
            "invalid LIGHTER_H41_ARM={other:?}; expected control, full-h41, or hybrid"
        ),
        Err(error) => panic!("invalid LIGHTER_H41_ARM: {error}"),
    }
}

#[cfg(not(feature = "std"))]
pub fn direct_wire_ifft_plan_is_requested() -> bool {
    false
}

/// Build the circuit-fixed column-major, bit-reversed representative plan.
/// The plan is a pure derivation of `representative_map`, built once during
/// circuit construction/deserialization and borrowed lock-free by each proof.
pub fn build_wire_ifft_representative_plan(
    representative_map: &[u32],
    num_wires: usize,
    degree: usize,
) -> Option<Vec<u32>> {
    if degree > MAX_DIRECT_WIRE_IFFT_DEGREE {
        return None;
    }
    assert!(degree.is_power_of_two());
    let wire_cells = num_wires
        .checked_mul(degree)
        .expect("wire IFFT plan length overflow");
    assert!(representative_map.len() >= wire_cells);
    let degree_bits = degree.trailing_zeros() as usize;
    let mut plan = Vec::with_capacity(wire_cells);
    for column in 0..num_wires {
        for destination in 0..degree {
            let source_row =
                destination.reverse_bits() >> (usize::BITS as usize - degree_bits);
            plan.push(representative_map[source_row * num_wires + column]);
        }
    }
    Some(plan)
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
    /// Representative index of every target, as `u32`. Halving the entry width halves the bytes
    /// this table costs per read; every value is zero-extended at the indexing site, so the slots
    /// selected in `values`/`set_bitmap` are unchanged.
    pub representative_map: &'a [u32],
    pub num_wires: usize,
    pub degree: usize,
}

impl<'a, F: Field> PartitionWitness<'a, F> {
    pub fn new(num_wires: usize, degree: usize, representative_map: &'a [u32]) -> Self {
        let len = representative_map.len();
        // `values` is left uninitialized: `F` has no `IsZero` specialization,
        // so `vec![F::ZERO; len]` is a real serial store pass (~70-100 MB per
        // transaction proof, ~285 MB for the final block) at the head of the
        // serial witness lane. Every reader is `set_bitmap`-guarded —
        // `set_target_returning_rep`, `try_get_target`, and (since this
        // change) `full_witness` — so an unset slot's storage is never read;
        // the bitmap itself IS zeroed (u64 hits the `alloc_zeroed` path).
        // Unset slots still yield exactly `F::ZERO` at every observation
        // point, so proof bytes are unchanged.
        let mut values = Vec::with_capacity(len);
        // SAFETY: `F` is a plain field element (`Copy`, no drop); all reads
        // are bitmap-guarded per the invariant above.
        unsafe { values.set_len(len) };
        Self {
            values,
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
        let rep_index = self.representative_map[self.target_index(target)] as usize;
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

    /// Split the production wire representation at the permutation boundary.
    /// Routed columns retain the exact natural, column-major layout consumed
    /// by the permutation argument; non-routed columns are gathered directly
    /// into the bit-reversed input order consumed by `ifft_from_bit_reversed`.
    /// Consuming `self` here drops the Partition storage at the same boundary
    /// as `full_witness`, before either wire IFFT or commitment starts.
    pub fn routed_witness_and_direct_ifft_inputs(
        self,
        num_routed_wires: usize,
    ) -> (MatrixWitness<F>, Vec<Vec<F>>) {
        assert!(num_routed_wires <= self.num_wires);
        assert!(self.degree.is_power_of_two());

        let degree = self.degree;
        let num_wires = self.num_wires;
        let mut routed_values: Vec<Vec<F>> = (0..num_routed_wires)
            .map(|_| Vec::with_capacity(degree))
            .collect();
        let num_chunks = 16.min(degree.max(1));
        let chunk_rows = degree.div_ceil(num_chunks);
        {
            let mut segments: Vec<Vec<&mut [core::mem::MaybeUninit<F>]>> = (0..num_chunks)
                .map(|_| Vec::with_capacity(num_routed_wires))
                .collect();
            for column in routed_values.iter_mut() {
                let mut rest = crate::hash::merkle_tree::capacity_up_to_mut(column, degree);
                for segment_columns in segments.iter_mut() {
                    let take = chunk_rows.min(rest.len());
                    let (head, tail) = rest.split_at_mut(take);
                    segment_columns.push(head);
                    rest = tail;
                }
            }
            use plonky2_maybe_rayon::*;
            segments
                .par_iter_mut()
                .enumerate()
                .for_each(|(chunk, columns)| {
                    let rows = columns.first().map_or(0, |column| column.len());
                    let first_row = chunk * chunk_rows;
                    for i in 0..rows {
                        let row_base = (first_row + i) * num_wires;
                        for (j, column) in columns.iter_mut().enumerate() {
                            let rep = self.representative_map[row_base + j] as usize;
                            let value = if self.is_set_by_rep_index(rep) {
                                self.values[rep]
                            } else {
                                F::ZERO
                            };
                            column[i].write(value);
                        }
                    }
                });
        }
        for column in routed_values.iter_mut() {
            // SAFETY: every slot was initialized exactly once by the disjoint
            // row segments above.
            unsafe { column.set_len(degree) };
        }

        use plonky2_maybe_rayon::*;
        let degree_bits = degree.trailing_zeros() as usize;
        let direct_inputs = (num_routed_wires..num_wires)
            .into_par_iter()
            .map(|column| {
                (0..degree)
                    .map(|destination| {
                        let source_row = destination.reverse_bits()
                            >> (usize::BITS as usize - degree_bits);
                        let rep = self.representative_map[source_row * num_wires + column] as usize;
                        if self.is_set_by_rep_index(rep) {
                            self.values[rep]
                        } else {
                            F::ZERO
                        }
                    })
                    .collect()
            })
            .collect();

        (
            MatrixWitness {
                wire_values: routed_values,
            },
            direct_inputs,
        )
    }

    pub fn full_witness(self) -> MatrixWitness<F> {
        // Single fused pass, parallel over row chunks. Cell (column j, row i)
        // is `values[representative_map[i * num_wires + j]]` (unset slots hold
        // `F::ZERO` in the dense `values` vector), with `Target::index`'s
        // `row * num_wires + column` inlined as a running cursor. Each
        // parallel task owns one row range of every column — disjoint
        // `MaybeUninit` segments carved off the pre-sized column buffers up
        // front — reads `representative_map` sequentially within its range,
        // and initializes every cell exactly once before the final `set_len`.
        let num_wires = self.num_wires;
        let degree = self.degree;
        let mut wire_values: Vec<Vec<F>> = (0..num_wires)
            .map(|_| Vec::with_capacity(degree))
            .collect();
        let num_chunks = 16.min(degree.max(1));
        let chunk_rows = degree.div_ceil(num_chunks);
        {
            let mut segments: Vec<Vec<&mut [core::mem::MaybeUninit<F>]>> = (0..num_chunks)
                .map(|_| Vec::with_capacity(num_wires))
                .collect();
            for column in wire_values.iter_mut() {
                let mut rest =
                    crate::hash::merkle_tree::capacity_up_to_mut(column, degree);
                for segment_columns in segments.iter_mut() {
                    let take = chunk_rows.min(rest.len());
                    let (head, tail) = rest.split_at_mut(take);
                    segment_columns.push(head);
                    rest = tail;
                }
            }
            use plonky2_maybe_rayon::*;
            segments
                .par_iter_mut()
                .enumerate()
                .for_each(|(chunk, columns)| {
                    let rows = columns.first().map_or(0, |column| column.len());
                    let mut wire_index = chunk * chunk_rows * num_wires;
                    for i in 0..rows {
                        for column in columns.iter_mut() {
                            let rep = self.representative_map[wire_index] as usize;
                            // Bitmap-guarded: unset slots are uninitialized
                            // storage and must read as F::ZERO (identical to
                            // the dense-zero representation this replaces).
                            let value = if self.is_set_by_rep_index(rep) {
                                self.values[rep]
                            } else {
                                F::ZERO
                            };
                            column[i].write(value);
                            wire_index += 1;
                        }
                    }
                });
        }
        for column in wire_values.iter_mut() {
            // SAFETY: every one of the `degree` slots of every column was
            // initialized exactly once by the disjoint segment writes above.
            unsafe {
                column.set_len(degree);
            }
        }

        MatrixWitness { wire_values }
    }
    #[allow(dead_code)]
    fn full_witness_serial_reference(self) -> MatrixWitness<F> {
        let mut wire_values: Vec<Vec<F>> = (0..self.num_wires)
            .map(|_| Vec::with_capacity(self.degree))
            .collect();
        let mut wire_index = 0;
        for _ in 0..self.degree {
            for column in wire_values.iter_mut() {
                // Bitmap-guarded like the parallel path: unset slots are
                // uninitialized storage and read as `F::ZERO`.
                let rep = self.representative_map[wire_index] as usize;
                let value = if self.is_set_by_rep_index(rep) {
                    self.values[rep]
                } else {
                    F::ZERO
                };
                column.push(value);
                wire_index += 1;
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
        let rep_index = self.representative_map[self.target_index(target)] as usize;
        if self.is_set_by_rep_index(rep_index) {
            Some(self.values[rep_index])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod wire_representation_tests {
    use plonky2_util::reverse_index_bits;

    use super::{PartitionWitness, build_wire_ifft_representative_plan};
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;

    #[test]
    fn routed_split_matches_full_matrix_and_direct_bitrev_raw_limbs() {
        type F = GoldilocksField;
        let num_wires = 5;
        let num_routed_wires = 3;
        let degree = 8;
        let representative_map: Vec<u32> = (0..num_wires * degree)
            .map(|index| ((index * 7 + 5) % (num_wires * degree)) as u32)
            .collect();
        let mut partition = PartitionWitness::<F>::new(num_wires, degree, &representative_map);
        for (index, value) in partition.values.iter_mut().enumerate() {
            *value = GoldilocksField(match index % 4 {
                0 => 0xffff_ffff_0000_0001,
                1 => 0xffff_ffff_0000_0002,
                2 => u64::MAX,
                _ => (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            });
        }
        partition.set_bitmap.fill(u64::MAX);

        let full = partition.clone().full_witness();
        let (routed, direct) =
            partition.routed_witness_and_direct_ifft_inputs(num_routed_wires);
        assert_eq!(routed.wire_values.len(), num_routed_wires);
        for column in 0..num_routed_wires {
            let actual: Vec<_> = routed.wire_values[column]
                .iter()
                .map(|value| value.0)
                .collect();
            let expected: Vec<_> = full.wire_values[column]
                .iter()
                .map(|value| value.0)
                .collect();
            assert_eq!(actual, expected, "routed column {column}");
        }
        for (offset, actual) in direct.iter().enumerate() {
            let column = num_routed_wires + offset;
            let expected = reverse_index_bits(&full.wire_values[column]);
            assert_eq!(
                actual.iter().map(|value| value.0).collect::<Vec<_>>(),
                expected.iter().map(|value| value.0).collect::<Vec<_>>(),
                "direct column {column}"
            );
        }
    }

    #[test]
    fn staged_wire_representations_match_sparse_unset_full_matrix_raw_limbs() {
        type F = GoldilocksField;
        let num_wires = 5;
        let num_routed_wires = 3;
        let degree = 8;
        let representative_map: Vec<u32> = (0..num_wires * degree)
            .map(|index| ((index * 7 + 5) % (num_wires * degree)) as u32)
            .collect();
        let mut partition = PartitionWitness::<F>::new(num_wires, degree, &representative_map);
        for (index, value) in partition.values.iter_mut().enumerate() {
            *value = GoldilocksField(match index % 4 {
                0 => 0xffff_ffff_0000_0001,
                1 => 0xffff_ffff_0000_0002,
                2 => u64::MAX,
                _ => (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            });
            if index % 3 != 0 {
                partition.set_bitmap[index >> 6] |= 1u64 << (index & 63);
            }
        }

        let full = partition.clone().full_witness();
        let plan = build_wire_ifft_representative_plan(
            &representative_map,
            num_wires,
            degree,
        )
        .unwrap();
        for (column, column_plan) in plan.chunks_exact(degree).enumerate() {
            let actual: Vec<_> = column_plan
                .iter()
                .map(|&rep| {
                    let rep = rep as usize;
                    if partition.is_set_by_rep_index(rep) {
                        partition.values[rep].0
                    } else {
                        F::ZERO.0
                    }
                })
                .collect();
            let expected = reverse_index_bits(&full.wire_values[column]);
            assert_eq!(
                actual,
                expected.iter().map(|value| value.0).collect::<Vec<_>>(),
                "full-H41 sparse column {column}"
            );
        }

        let (routed, direct) =
            partition.routed_witness_and_direct_ifft_inputs(num_routed_wires);
        for column in 0..num_routed_wires {
            assert_eq!(
                routed.wire_values[column]
                    .iter()
                    .map(|value| value.0)
                    .collect::<Vec<_>>(),
                full.wire_values[column]
                    .iter()
                    .map(|value| value.0)
                    .collect::<Vec<_>>(),
                "hybrid routed sparse column {column}"
            );
        }
        for (offset, actual) in direct.iter().enumerate() {
            let column = num_routed_wires + offset;
            let expected = reverse_index_bits(&full.wire_values[column]);
            assert_eq!(
                actual.iter().map(|value| value.0).collect::<Vec<_>>(),
                expected.iter().map(|value| value.0).collect::<Vec<_>>(),
                "hybrid direct sparse column {column}"
            );
        }
    }
}
