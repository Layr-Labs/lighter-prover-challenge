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
use crate::plonk::proof::{
    OpeningSet, OpeningSetTarget, Proof, ProofTarget, ProofWithPublicInputs,
    ProofWithPublicInputsTarget,
};

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

        set_opening_set_target(self, &proof_target.openings, &proof.openings)?;

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

/// Writes a structured PLONK opening set without materializing the two owned
/// [`FriOpeningsTarget`] / [`FriOpenings`] transport graphs used by the generic
/// FRI API. The component order is exactly the order of
/// `OpeningSet{,Target}::to_fri_openings`: the zeta components first, followed
/// by the next-point components.
fn set_opening_set_target<W, F, const D: usize>(
    writer: &mut W,
    targets: &OpeningSetTarget<D>,
    values: &OpeningSet<F, D>,
) -> Result<()>
where
    W: WitnessWrite<F> + ?Sized,
    F: RichField + Extendable<D>,
{
    // Validate the complete shape before mutating the witness. Besides making
    // every component mismatch an error in release builds, this avoids a
    // partially written witness when a later (next-point) component is wrong.
    for (component, target_len, value_len) in [
        ("constants", targets.constants.len(), values.constants.len()),
        (
            "plonk_sigmas",
            targets.plonk_sigmas.len(),
            values.plonk_sigmas.len(),
        ),
        ("wires", targets.wires.len(), values.wires.len()),
        ("plonk_zs", targets.plonk_zs.len(), values.plonk_zs.len()),
        (
            "partial_products",
            targets.partial_products.len(),
            values.partial_products.len(),
        ),
        (
            "quotient_polys",
            targets.quotient_polys.len(),
            values.quotient_polys.len(),
        ),
        ("lookup_zs", targets.lookup_zs.len(), values.lookup_zs.len()),
        (
            "plonk_zs_next",
            targets.plonk_zs_next.len(),
            values.plonk_zs_next.len(),
        ),
        (
            "lookup_zs_next",
            targets.next_lookup_zs.len(),
            values.lookup_zs_next.len(),
        ),
    ] {
        if target_len != value_len {
            return Err(anyhow!(
                "Opening set component {component} length mismatch: {target_len} targets != {value_len} values"
            ));
        }
    }

    writer.set_extension_targets(&targets.constants, &values.constants)?;
    writer.set_extension_targets(&targets.plonk_sigmas, &values.plonk_sigmas)?;
    writer.set_extension_targets(&targets.wires, &values.wires)?;
    writer.set_extension_targets(&targets.plonk_zs, &values.plonk_zs)?;
    writer.set_extension_targets(&targets.partial_products, &values.partial_products)?;
    writer.set_extension_targets(&targets.quotient_polys, &values.quotient_polys)?;
    writer.set_extension_targets(&targets.lookup_zs, &values.lookup_zs)?;
    writer.set_extension_targets(&targets.plonk_zs_next, &values.plonk_zs_next)?;
    writer.set_extension_targets(&targets.next_lookup_zs, &values.lookup_zs_next)
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

    /// [`Self::set_target_returning_rep`] with the representative supplied by
    /// the caller instead of gathered from `representative_map`.
    ///
    /// The representative must be the one `representative_map` holds for
    /// `target`; the caller is responsible for that (see
    /// [`crate::iop::generator::PartitionSeedLayout`], which records it from
    /// this very map under a pointer-equality binding to the owning prover
    /// data). Everything after the gather -- the set-bitmap probe, the
    /// contradiction check, the store and the mark -- is the identical
    /// sequence on the identical slot, so the resulting witness is bit-for-bit
    /// what `set_target_returning_rep` would have produced.
    pub fn set_rep_index_returning_new(
        &mut self,
        rep_index: usize,
        target: Target,
        value: F,
    ) -> Result<Option<usize>> {
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
mod tests {
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;

    const D: usize = 2;
    type F = GoldilocksField;

    #[derive(Default)]
    struct RecordingWitness {
        writes: Vec<(Target, F)>,
    }

    impl WitnessWrite<F> for RecordingWitness {
        fn set_target(&mut self, target: Target, value: F) -> Result<()> {
            self.writes.push((target, value));
            Ok(())
        }
    }

    fn extension_targets(next_index: &mut usize, len: usize) -> Vec<ExtensionTarget<D>> {
        (0..len)
            .map(|_| {
                ExtensionTarget(core::array::from_fn(|_| {
                    let target = Target::VirtualTarget { index: *next_index };
                    *next_index += 1;
                    target
                }))
            })
            .collect()
    }

    fn extension_values(
        next_value: &mut u64,
        len: usize,
    ) -> Vec<<F as Extendable<D>>::Extension> {
        (0..len)
            .map(|_| {
                <<F as Extendable<D>>::Extension as FieldExtension<D>>::from_basefield_array(
                    core::array::from_fn(|_| {
                        let value = F::from_canonical_u64(*next_value);
                        *next_value += 1;
                        value
                    }),
                )
            })
            .collect()
    }

    fn opening_fixture(with_lookup: bool) -> (OpeningSetTarget<D>, OpeningSet<F, D>) {
        let lookup_len = usize::from(with_lookup) * 2;

        // Construct in legacy FRI write order so monotonically increasing
        // target/value IDs independently pin that order in the assertions.
        let mut next_target = 0;
        let constants = extension_targets(&mut next_target, 2);
        let plonk_sigmas = extension_targets(&mut next_target, 3);
        let wires = extension_targets(&mut next_target, 4);
        let plonk_zs = extension_targets(&mut next_target, 2);
        let partial_products = extension_targets(&mut next_target, 3);
        let quotient_polys = extension_targets(&mut next_target, 2);
        let lookup_zs = extension_targets(&mut next_target, lookup_len);
        let plonk_zs_next = extension_targets(&mut next_target, 2);
        let next_lookup_zs = extension_targets(&mut next_target, lookup_len);

        let mut next_value = 1;
        let constant_values = extension_values(&mut next_value, 2);
        let plonk_sigma_values = extension_values(&mut next_value, 3);
        let wire_values = extension_values(&mut next_value, 4);
        let plonk_z_values = extension_values(&mut next_value, 2);
        let partial_product_values = extension_values(&mut next_value, 3);
        let quotient_poly_values = extension_values(&mut next_value, 2);
        let lookup_z_values = extension_values(&mut next_value, lookup_len);
        let plonk_z_next_values = extension_values(&mut next_value, 2);
        let lookup_z_next_values = extension_values(&mut next_value, lookup_len);

        (
            OpeningSetTarget {
                constants,
                plonk_sigmas,
                wires,
                plonk_zs,
                plonk_zs_next,
                lookup_zs,
                next_lookup_zs,
                partial_products,
                quotient_polys,
            },
            OpeningSet {
                constants: constant_values,
                plonk_sigmas: plonk_sigma_values,
                wires: wire_values,
                plonk_zs: plonk_z_values,
                plonk_zs_next: plonk_z_next_values,
                partial_products: partial_product_values,
                quotient_polys: quotient_poly_values,
                lookup_zs: lookup_z_values,
                lookup_zs_next: lookup_z_next_values,
            },
        )
    }

    fn assert_direct_opening_writes_match_legacy(with_lookup: bool) -> Result<()> {
        let (targets, values) = opening_fixture(with_lookup);
        let mut direct = RecordingWitness::default();
        set_opening_set_target(&mut direct, &targets, &values)?;

        let mut legacy = RecordingWitness::default();
        legacy.set_fri_openings(&targets.to_fri_openings(), &values.to_fri_openings())?;

        assert_eq!(direct.writes, legacy.writes);
        for (expected_index, &(target, value)) in direct.writes.iter().enumerate() {
            assert_eq!(target, Target::VirtualTarget { index: expected_index });
            assert_eq!(value, F::from_canonical_u64(expected_index as u64 + 1));
        }
        Ok(())
    }

    #[test]
    fn direct_opening_writes_match_legacy_without_lookups() -> Result<()> {
        assert_direct_opening_writes_match_legacy(false)
    }

    #[test]
    fn direct_opening_writes_match_legacy_with_lookups() -> Result<()> {
        assert_direct_opening_writes_match_legacy(true)
    }

    #[test]
    fn direct_opening_writes_reject_every_component_length_mismatch() {
        let (targets, values) = opening_fixture(true);
        let cases: [(&str, fn(&mut OpeningSetTarget<D>)); 9] = [
            ("constants", |t| {
                t.constants.pop();
            }),
            ("plonk_sigmas", |t| {
                t.plonk_sigmas.pop();
            }),
            ("wires", |t| {
                t.wires.pop();
            }),
            ("plonk_zs", |t| {
                t.plonk_zs.pop();
            }),
            ("partial_products", |t| {
                t.partial_products.pop();
            }),
            ("quotient_polys", |t| {
                t.quotient_polys.pop();
            }),
            ("lookup_zs", |t| {
                t.lookup_zs.pop();
            }),
            ("plonk_zs_next", |t| {
                t.plonk_zs_next.pop();
            }),
            ("lookup_zs_next", |t| {
                t.next_lookup_zs.pop();
            }),
        ];

        for (component, mutate) in cases {
            let mut malformed = targets.clone();
            mutate(&mut malformed);
            let mut writer = RecordingWitness::default();
            let error = set_opening_set_target(&mut writer, &malformed, &values)
                .expect_err("every component mismatch must fail closed");
            assert!(
                error.to_string().contains(component),
                "wrong error for {component}: {error:#}"
            );
            assert!(
                writer.writes.is_empty(),
                "{component} mismatch wrote a partial witness before failing"
            );
        }
    }
}
