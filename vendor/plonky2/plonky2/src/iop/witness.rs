#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::iter::zip;

use anyhow::{anyhow, Result};
use hashbrown::HashMap;
use itertools::{zip_eq, Itertools};

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::fft::ifft_bitrev_buffer;
use crate::field::polynomial::PolynomialCoeffs;
use crate::field::types::Field;
use crate::util::{log2_strict, reverse_bits};
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

    /// Expand only columns `0..num_routed_wires` into a `MatrixWitness`,
    /// bitmap-guarded the same way as [`Self::full_witness`]. Does not
    /// consume `self`, so the partition stays borrowed for a subsequent
    /// non-routed gather IFFT.
    pub fn routed_witness(&self, num_routed_wires: usize) -> MatrixWitness<F> {
        debug_assert!(num_routed_wires <= self.num_wires);
        let num_wires = self.num_wires;
        let degree = self.degree;
        let mut wire_values: Vec<Vec<F>> = (0..num_routed_wires)
            .map(|_| Vec::with_capacity(degree))
            .collect();
        let num_chunks = 16.min(degree.max(1));
        let chunk_rows = degree.div_ceil(num_chunks);
        {
            let mut segments: Vec<Vec<&mut [core::mem::MaybeUninit<F>]>> = (0..num_chunks)
                .map(|_| Vec::with_capacity(num_routed_wires))
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
                    let row_base = chunk * chunk_rows;
                    for i in 0..rows {
                        // Row-major map is `i * num_wires + j`. Restart each
                        // row so we skip the non-routed tail of the row.
                        let mut wire_index = (row_base + i) * num_wires;
                        for column in columns.iter_mut() {
                            let rep = self.representative_map[wire_index] as usize;
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
            // SAFETY: every one of the `degree` slots of every routed
            // column was initialized exactly once by the disjoint segment
            // writes above.
            unsafe {
                column.set_len(degree);
            }
        }

        MatrixWitness { wire_values }
    }

    /// Gather wire column `col` from the partition in bit-reversed order
    /// (the same permutation `ifft_borrowed` applies via
    /// `reverse_index_bits`) and run that IFFT. For each natural row `i`:
    /// look up `representative_map[i * num_wires + col]`; if
    /// `is_set_by_rep_index(rep)` then `values[rep]`, else `F::ZERO`;
    /// write that into the FFT buffer at `reverse_bits(i, log_n)`.
    /// Unset slots therefore stay ZERO the same way as `full_witness`.
    /// Does not consume `self`.
    pub fn ifft_non_routed_column(&self, col: usize) -> PolynomialCoeffs<F> {
        debug_assert!(col < self.num_wires);
        let n = self.degree;
        let lg_n = log2_strict(n);
        let num_wires = self.num_wires;
        let mut buffer = Vec::with_capacity(n);
        // SAFETY: `reverse_bits(_, lg_n)` is a permutation of `0..n`, and
        // every natural row writes exactly one slot, so the buffer is
        // fully initialized before `ifft_bitrev_buffer` reads it.
        unsafe {
            buffer.set_len(n);
        }
        for i in 0..n {
            let bitrev = reverse_bits(i, lg_n);
            let rep = self.representative_map[i * num_wires + col] as usize;
            buffer[bitrev] = if self.is_set_by_rep_index(rep) {
                self.values[rep]
            } else {
                F::ZERO
            };
        }
        ifft_bitrev_buffer(buffer)
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
    use crate::field::polynomial::PolynomialValues;
    use crate::field::types::PrimeField64;

    type F = GoldilocksField;

    /// Oracle: partition-direct non-routed IFFT matches today's
    /// bitmap-guarded expand + `ifft` on the same column, compared by
    /// canonical *and* raw noncanonical limbs (not Goldilocks `PartialEq`
    /// alone). Unset representative slots must stay `F::ZERO`.
    #[test]
    fn ifft_non_routed_column_matches_expand_ifft_limbs() {
        let num_wires = 4usize;
        let num_routed_wires = 2usize;
        let degree = 8usize;
        let mut representative_map: Vec<u32> = (0..(num_wires * degree) as u32).collect();
        // Shared rep: row 1, col 2 aliases row 0, col 2.
        representative_map[1 * num_wires + 2] = representative_map[0 * num_wires + 2];

        let mut pw = PartitionWitness::new(num_wires, degree, &representative_map);
        pw.set_wire(Wire { row: 0, column: 2 }, F::from_canonical_u64(7))
            .unwrap();
        pw.set_wire(Wire { row: 3, column: 2 }, F::from_canonical_u64(11))
            .unwrap();
        pw.set_wire(Wire { row: 5, column: 2 }, F::ZERO).unwrap();
        pw.set_wire(Wire { row: 1, column: 3 }, F::from_canonical_u64(13))
            .unwrap();
        pw.set_wire(Wire { row: 2, column: 0 }, F::from_canonical_u64(3))
            .unwrap();

        // NEW gather while the partition is still borrowed.
        let new_col2 = pw.ifft_non_routed_column(2);
        let new_col3 = pw.ifft_non_routed_column(3);
        let routed = pw.routed_witness(num_routed_wires);

        // OLD expand+ifft (consumes a clone; prove path no longer does this).
        let matrix = pw.clone().full_witness();
        let old_col2 = PolynomialValues::new(matrix.wire_values[2].clone()).ifft();
        let old_col3 = PolynomialValues::new(matrix.wire_values[3].clone()).ifft();

        fn assert_limbs_eq(actual: &[F], expected: &[F], label: &str) {
            assert_eq!(actual.len(), expected.len(), "{label} length");
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    a.to_canonical_u64(),
                    e.to_canonical_u64(),
                    "{label} canonical limb mismatch at {i}"
                );
                assert_eq!(
                    a.to_noncanonical_u64(),
                    e.to_noncanonical_u64(),
                    "{label} raw noncanonical limb mismatch at {i}"
                );
            }
        }

        assert_limbs_eq(&new_col2.coeffs, &old_col2.coeffs, "col 2");
        assert_limbs_eq(&new_col3.coeffs, &old_col3.coeffs, "col 3");

        assert_eq!(routed.wire_values.len(), num_routed_wires);
        for j in 0..num_routed_wires {
            for i in 0..degree {
                assert_eq!(
                    routed.wire_values[j][i].to_canonical_u64(),
                    matrix.wire_values[j][i].to_canonical_u64(),
                    "routed expand canonical mismatch at ({i},{j})"
                );
                assert_eq!(
                    routed.wire_values[j][i].to_noncanonical_u64(),
                    matrix.wire_values[j][i].to_noncanonical_u64(),
                    "routed expand raw limb mismatch at ({i},{j})"
                );
            }
        }

        // BITMAP_ZERO: an unset natural row must have expanded as ZERO.
        assert_eq!(matrix.wire_values[2][2].to_canonical_u64(), 0);
        assert_eq!(matrix.wire_values[2][2].to_noncanonical_u64(), 0);
        // Shared-rep row picked up the set value, not uninit storage.
        assert_eq!(matrix.wire_values[2][1].to_canonical_u64(), 7);
    }
}
