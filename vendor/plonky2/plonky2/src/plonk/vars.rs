//! Logic for evaluating constraints.

use core::ops::Range;

use crate::field::extension::algebra::ExtensionAlgebra;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::packed::PackedField;
use crate::field::types::Field;
use crate::hash::hash_types::{HashOut, HashOutTarget, RichField};
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::util::strided_view::PackedStridedView;

#[derive(Debug, Copy, Clone)]
pub struct EvaluationVars<'a, F: RichField + Extendable<D>, const D: usize> {
    pub local_constants: &'a [F::Extension],
    pub local_wires: &'a [F::Extension],
    pub public_inputs_hash: &'a HashOut<F>,
}

/// A batch of evaluation vars, in the base field.
/// Wires and constants are stored in an evaluation point-major order (that is, wire 0 for all
/// evaluation points, then wire 1 for all points, and so on).
#[derive(Debug, Copy, Clone)]
pub struct EvaluationVarsBaseBatch<'a, F: Field> {
    batch_size: usize,
    local_constants_len: usize,
    local_wires_len: usize,
    pub local_constants: &'a [F],
    pub local_wires: &'a [F],
    pub public_inputs_hash: &'a HashOut<F>,
}

/// A view into `EvaluationVarsBaseBatch` for a particular evaluation point. Does not copy the data.
#[derive(Debug, Copy, Clone)]
pub struct EvaluationVarsBase<'a, F: Field> {
    pub local_constants: PackedStridedView<'a, F>,
    pub local_wires: PackedStridedView<'a, F>,
    pub public_inputs_hash: &'a HashOut<F>,
}

/// Like `EvaluationVarsBase`, but packed.
// It's a separate struct because `EvaluationVarsBase` implements `get_local_ext` and we do not yet
// have packed extension fields.
#[derive(Debug, Copy, Clone)]
pub struct EvaluationVarsBasePacked<'a, P: PackedField> {
    pub local_constants: PackedStridedView<'a, P>,
    pub local_wires: PackedStridedView<'a, P>,
    pub public_inputs_hash: &'a HashOut<P::Scalar>,
}

impl<F: RichField + Extendable<D>, const D: usize> EvaluationVars<'_, F, D> {
    pub fn get_local_ext_algebra(
        &self,
        wire_range: Range<usize>,
    ) -> ExtensionAlgebra<F::Extension, D> {
        debug_assert_eq!(wire_range.len(), D);
        let arr = self.local_wires[wire_range].try_into().unwrap();
        ExtensionAlgebra::from_basefield_array(arr)
    }

    pub fn remove_prefix(&mut self, num_selectors: usize) {
        self.local_constants = &self.local_constants[num_selectors..];
    }
}

impl<'a, F: Field> EvaluationVarsBaseBatch<'a, F> {
    pub fn new(
        batch_size: usize,
        local_constants: &'a [F],
        local_wires: &'a [F],
        public_inputs_hash: &'a HashOut<F>,
    ) -> Self {
        assert_eq!(local_constants.len() % batch_size, 0);
        assert_eq!(local_wires.len() % batch_size, 0);
        Self {
            batch_size,
            local_constants_len: local_constants.len() / batch_size,
            local_wires_len: local_wires.len() / batch_size,
            local_constants,
            local_wires,
            public_inputs_hash,
        }
    }

    pub fn remove_prefix(&mut self, num_selectors: usize) {
        let remaining = self
            .local_constants_len
            .checked_sub(num_selectors)
            .expect("cannot remove more constant columns than remain");
        self.local_constants = &self.local_constants[num_selectors * self.len()..];
        self.local_constants_len = remaining;
        debug_assert_eq!(
            self.local_constants_len,
            self.local_constants.len() / self.batch_size
        );
    }

    pub const fn len(&self) -> usize {
        self.batch_size
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn view(&self, index: usize) -> EvaluationVarsBase<'a, F> {
        // We cannot implement `Index` as `EvaluationVarsBase` is a struct, not a reference.
        assert!(index < self.len());
        let local_constants = PackedStridedView::new(self.local_constants, self.len(), index);
        let local_wires = PackedStridedView::new(self.local_wires, self.len(), index);
        EvaluationVarsBase {
            local_constants,
            local_wires,
            public_inputs_hash: self.public_inputs_hash,
        }
    }

    pub const fn iter(&self) -> EvaluationVarsBaseBatchIter<'a, F> {
        EvaluationVarsBaseBatchIter::new(*self)
    }

    pub fn pack<P: PackedField<Scalar = F>>(
        &self,
    ) -> (
        EvaluationVarsBaseBatchIterPacked<'a, P>,
        EvaluationVarsBaseBatchIterPacked<'a, F>,
    ) {
        let n_leftovers = self.len() % P::WIDTH;
        (
            EvaluationVarsBaseBatchIterPacked::new_with_start(*self, 0),
            EvaluationVarsBaseBatchIterPacked::new_with_start(*self, self.len() - n_leftovers),
        )
    }
}

impl<F: Field> EvaluationVarsBase<'_, F> {
    pub fn get_local_ext<const D: usize>(&self, wire_range: Range<usize>) -> F::Extension
    where
        F: RichField + Extendable<D>,
    {
        debug_assert_eq!(wire_range.len(), D);
        let arr = self.local_wires.view(wire_range).try_into().unwrap();
        F::Extension::from_basefield_array(arr)
    }
}

/// Iterator of views (`EvaluationVarsBase`) into a `EvaluationVarsBaseBatch`.
#[derive(Debug)]
pub struct EvaluationVarsBaseBatchIter<'a, F: Field> {
    i: usize,
    vars_batch: EvaluationVarsBaseBatch<'a, F>,
}

impl<'a, F: Field> EvaluationVarsBaseBatchIter<'a, F> {
    pub const fn new(vars_batch: EvaluationVarsBaseBatch<'a, F>) -> Self {
        EvaluationVarsBaseBatchIter { i: 0, vars_batch }
    }
}

impl<'a, F: Field> Iterator for EvaluationVarsBaseBatchIter<'a, F> {
    type Item = EvaluationVarsBase<'a, F>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.vars_batch.len() {
            let res = self.vars_batch.view(self.i);
            self.i += 1;
            Some(res)
        } else {
            None
        }
    }
}

/// Iterator of packed views (`EvaluationVarsBasePacked`) into a `EvaluationVarsBaseBatch`.
/// Note: if the length of `EvaluationVarsBaseBatch` is not a multiple of `P::WIDTH`, then the
/// leftovers at the end are ignored.
#[derive(Debug)]
pub struct EvaluationVarsBaseBatchIterPacked<'a, P: PackedField> {
    /// Index to yield next, in units of `P::Scalar`. E.g. if `P::WIDTH == 4`, then we will yield
    /// the vars for points `i`, `i + 1`, `i + 2`, and `i + 3`, packed.
    i: usize,
    vars_batch: EvaluationVarsBaseBatch<'a, P::Scalar>,
}

impl<'a, P: PackedField> EvaluationVarsBaseBatchIterPacked<'a, P> {
    pub fn new_with_start(
        vars_batch: EvaluationVarsBaseBatch<'a, P::Scalar>,
        start: usize,
    ) -> Self {
        assert!(start <= vars_batch.len());
        EvaluationVarsBaseBatchIterPacked {
            i: start,
            vars_batch,
        }
    }
}

impl<'a, P: PackedField> Iterator for EvaluationVarsBaseBatchIterPacked<'a, P> {
    type Item = EvaluationVarsBasePacked<'a, P>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i + P::WIDTH <= self.vars_batch.len() {
            // SAFETY: `EvaluationVarsBaseBatch::new` checks that both slices
            // are divisible by `batch_size` and records their exact logical
            // lengths. `remove_prefix` removes whole constant columns and
            // updates that length. The branch above proves the packed offset
            // fits within the stride. The original borrowed slices retain the
            // lifetime brand of both returned views.
            let local_constants = unsafe {
                PackedStridedView::new_derived(
                    self.vars_batch.local_constants,
                    self.vars_batch.len(),
                    self.i,
                    self.vars_batch.local_constants_len,
                )
            };
            let local_wires = unsafe {
                PackedStridedView::new_derived(
                    self.vars_batch.local_wires,
                    self.vars_batch.len(),
                    self.i,
                    self.vars_batch.local_wires_len,
                )
            };
            let res = EvaluationVarsBasePacked {
                local_constants,
                local_wires,
                public_inputs_hash: self.vars_batch.public_inputs_hash,
            };
            self.i += P::WIDTH;
            Some(res)
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<P: PackedField> ExactSizeIterator for EvaluationVarsBaseBatchIterPacked<'_, P> {
    fn len(&self) -> usize {
        (self.vars_batch.len() - self.i) / P::WIDTH
    }
}

impl<const D: usize> EvaluationTargets<'_, D> {
    pub fn remove_prefix(&mut self, num_selectors: usize) {
        self.local_constants = &self.local_constants[num_selectors..];
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::packable::Packable;
    use crate::field::types::Field64;
    use crate::gates::arithmetic_base::ArithmeticGate;
    use crate::gates::gate::Gate;
    use crate::gates::packed_util::PackedEvaluableBase;
    use crate::gates::public_input::PublicInputGate;
    use crate::gates::util::StridedConstraintConsumer;

    type F = GoldilocksField;
    const D: usize = 2;

    fn value(i: usize) -> F {
        match i % 7 {
            0 => GoldilocksField(0),
            1 => GoldilocksField(F::ORDER),
            2 => GoldilocksField(F::ORDER + 1),
            3 => GoldilocksField(u64::MAX),
            4 => GoldilocksField((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            5 => GoldilocksField(17),
            _ => GoldilocksField((i as u64).rotate_left(23)),
        }
    }

    fn hash() -> HashOut<F> {
        HashOut {
            elements: [value(101), value(102), value(103), value(104)],
        }
    }

    fn assert_view_raw_eq<P: PackedField<Scalar = F>>(
        actual: PackedStridedView<'_, P>,
        checked: PackedStridedView<'_, P>,
    ) {
        assert_eq!(actual.len(), checked.len());
        assert_eq!(actual.is_empty(), checked.is_empty());
        for i in 0..actual.len() {
            let actual_raw = actual[i].as_slice().iter().map(|x| x.0).collect::<Vec<_>>();
            let checked_raw = checked[i].as_slice().iter().map(|x| x.0).collect::<Vec<_>>();
            assert_eq!(actual_raw, checked_raw, "logical column {i}");
        }
        let actual_raw = actual
            .into_iter()
            .flat_map(|x| x.as_slice().iter().map(|x| x.0))
            .collect::<Vec<_>>();
        let checked_raw = checked
            .into_iter()
            .flat_map(|x| x.as_slice().iter().map(|x| x.0))
            .collect::<Vec<_>>();
        assert_eq!(actual_raw, checked_raw);
    }

    /// Release-codegen probe for exactly one production packed-iterator step.
    /// Keeping the symbol uninlined lets the component gate verify that the
    /// derived view no longer carries a remainder, division, or release panic
    /// edge. This function is test-only and does not alter production code.
    #[inline(never)]
    fn derived_view_codegen_probe(
        vars_batch: EvaluationVarsBaseBatch<'_, F>,
    ) -> Option<(usize, usize)> {
        let mut iter = EvaluationVarsBaseBatchIterPacked::<<F as Packable>::Packing> {
            i: 0,
            vars_batch,
        };
        iter.next()
            .map(|vars| (vars.local_constants.len(), vars.local_wires.len()))
    }

    #[test]
    fn packed_derived_views_match_checked_after_every_prefix_state_raw() {
        type Packing = <F as Packable>::Packing;
        let public_inputs_hash = hash();

        for batch_size in [1usize, 3, 4, 5, 7, 8, 31, 32] {
            for constants_len in 0usize..=6 {
                let constants = (0..constants_len * batch_size)
                    .map(|i| value(i + 1000))
                    .collect::<Vec<_>>();
                let wires_len = 5;
                let wires = (0..wires_len * batch_size)
                    .map(|i| value(i + 2000))
                    .collect::<Vec<_>>();
                let original = EvaluationVarsBaseBatch::new(
                    batch_size,
                    &constants,
                    &wires,
                    &public_inputs_hash,
                );
                let probe = derived_view_codegen_probe(original);
                if batch_size >= <F as Packable>::Packing::WIDTH {
                    assert_eq!(probe, Some((constants_len, wires_len)));
                } else {
                    assert_eq!(probe, None);
                }

                for first_prefix in 0..=constants_len {
                    for second_prefix in 0..=constants_len - first_prefix {
                        let mut vars = original;
                        vars.remove_prefix(first_prefix);
                        vars.remove_prefix(second_prefix);
                        let remaining = constants_len - first_prefix - second_prefix;
                        assert_eq!(vars.local_constants_len, remaining);
                        assert_eq!(vars.local_wires_len, wires_len);
                        assert_eq!(
                            vars.local_constants_len,
                            vars.local_constants.len() / batch_size
                        );
                        assert_eq!(vars.local_wires_len, vars.local_wires.len() / batch_size);

                        let (packed, leftovers) = vars.pack::<Packing>();
                        for (group, actual) in packed.enumerate() {
                            let offset = group * Packing::WIDTH;
                            let constants_checked = PackedStridedView::<Packing>::new(
                                vars.local_constants,
                                batch_size,
                                offset,
                            );
                            let wires_checked = PackedStridedView::<Packing>::new(
                                vars.local_wires,
                                batch_size,
                                offset,
                            );
                            assert_view_raw_eq(actual.local_constants, constants_checked);
                            assert_view_raw_eq(actual.local_wires, wires_checked);
                        }

                        let leftovers_start = batch_size - batch_size % Packing::WIDTH;
                        for (i, actual) in leftovers.enumerate() {
                            let offset = leftovers_start + i;
                            let constants_checked = PackedStridedView::<F>::new(
                                vars.local_constants,
                                batch_size,
                                offset,
                            );
                            let wires_checked = PackedStridedView::<F>::new(
                                vars.local_wires,
                                batch_size,
                                offset,
                            );
                            assert_view_raw_eq(actual.local_constants, constants_checked);
                            assert_view_raw_eq(actual.local_wires, wires_checked);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn checked_batch_and_prefix_reject_malformed_shapes() {
        let public_inputs_hash = hash();
        let malformed_constants = [F::ZERO; 3];
        let wires = [F::ZERO; 8];
        assert!(catch_unwind(|| {
            EvaluationVarsBaseBatch::new(
                4,
                &malformed_constants,
                &wires,
                &public_inputs_hash,
            )
        })
        .is_err());
        assert!(catch_unwind(|| {
            EvaluationVarsBaseBatch::new(0, &[], &[], &public_inputs_hash)
        })
        .is_err());

        let constants = [F::ZERO; 8];
        let mut vars = EvaluationVarsBaseBatch::new(4, &constants, &wires, &public_inputs_hash);
        assert!(catch_unwind(AssertUnwindSafe(|| vars.remove_prefix(3))).is_err());
    }

    fn checked_packed_gate_output<G>(
        gate: &G,
        vars: EvaluationVarsBaseBatch<'_, F>,
    ) -> Vec<F>
    where
        G: PackedEvaluableBase<F, D>,
    {
        type Packing = <F as Packable>::Packing;
        let n = vars.len();
        let num_constraints = <G as Gate<F, D>>::num_constraints(gate);
        let mut result = vec![F::ZERO; n * num_constraints];
        let width = Packing::WIDTH;
        let packed_end = n - n % width;

        for offset in (0..packed_end).step_by(width) {
            let checked = EvaluationVarsBasePacked {
                local_constants: PackedStridedView::<Packing>::new(
                    vars.local_constants,
                    n,
                    offset,
                ),
                local_wires: PackedStridedView::<Packing>::new(vars.local_wires, n, offset),
                public_inputs_hash: vars.public_inputs_hash,
            };
            <G as PackedEvaluableBase<F, D>>::eval_unfiltered_base_packed(
                gate,
                checked,
                StridedConstraintConsumer::new(&mut result, n, offset),
            );
        }
        for offset in packed_end..n {
            let checked = EvaluationVarsBasePacked {
                local_constants: PackedStridedView::<F>::new(
                    vars.local_constants,
                    n,
                    offset,
                ),
                local_wires: PackedStridedView::<F>::new(vars.local_wires, n, offset),
                public_inputs_hash: vars.public_inputs_hash,
            };
            <G as PackedEvaluableBase<F, D>>::eval_unfiltered_base_packed(
                gate,
                checked,
                StridedConstraintConsumer::new(&mut result, n, offset),
            );
        }
        result
    }

    fn assert_gate_raw_eq<G>(
        gate: &G,
        vars: EvaluationVarsBaseBatch<'_, F>,
    ) where
        G: PackedEvaluableBase<F, D>,
    {
        let checked = checked_packed_gate_output(gate, vars);
        let derived =
            <G as PackedEvaluableBase<F, D>>::eval_unfiltered_base_batch_packed(gate, vars);
        assert_eq!(checked.len(), derived.len());
        assert_eq!(
            checked.iter().map(|x| x.0).collect::<Vec<_>>(),
            derived.iter().map(|x| x.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn production_packed_gates_match_checked_view_reference_after_prefix_raw() {
        let public_inputs_hash = hash();
        for n in [1usize, 3, 4, 5, 31, 32] {
            for (first_prefix, second_prefix) in [(0usize, 0usize), (1, 0), (1, 2), (2, 2)] {
                let prefix = first_prefix + second_prefix;

                let public_constants = (0..prefix * n)
                    .map(|i| value(i + 3000))
                    .collect::<Vec<_>>();
                let public_wires = (0..4 * n)
                    .map(|i| value(i + 4000))
                    .collect::<Vec<_>>();
                let mut public_vars = EvaluationVarsBaseBatch::new(
                    n,
                    &public_constants,
                    &public_wires,
                    &public_inputs_hash,
                );
                public_vars.remove_prefix(first_prefix);
                public_vars.remove_prefix(second_prefix);
                assert_eq!(public_vars.local_constants_len, 0);
                assert_gate_raw_eq(&PublicInputGate, public_vars);

                let arithmetic_constants = (0..(prefix + 2) * n)
                    .map(|i| value(i + 5000))
                    .collect::<Vec<_>>();
                let arithmetic_wires = (0..12 * n)
                    .map(|i| value(i + 6000))
                    .collect::<Vec<_>>();
                let mut arithmetic_vars = EvaluationVarsBaseBatch::new(
                    n,
                    &arithmetic_constants,
                    &arithmetic_wires,
                    &public_inputs_hash,
                );
                arithmetic_vars.remove_prefix(first_prefix);
                arithmetic_vars.remove_prefix(second_prefix);
                assert_eq!(arithmetic_vars.local_constants_len, 2);
                assert_gate_raw_eq(&ArithmeticGate { num_ops: 3 }, arithmetic_vars);
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EvaluationTargets<'a, const D: usize> {
    pub local_constants: &'a [ExtensionTarget<D>],
    pub local_wires: &'a [ExtensionTarget<D>],
    pub public_inputs_hash: &'a HashOutTarget,
}

impl<const D: usize> EvaluationTargets<'_, D> {
    pub fn get_local_ext_algebra(&self, wire_range: Range<usize>) -> ExtensionAlgebraTarget<D> {
        debug_assert_eq!(wire_range.len(), D);
        let arr = self.local_wires[wire_range].try_into().unwrap();
        ExtensionAlgebraTarget(arr)
    }
}
