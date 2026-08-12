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
    // Keep the raw backing private to this crate. Unlike the compact default
    // representation, an optimized batch may retain a full-domain column
    // stride; external gates must use `view`, `pack`, or `constant_column`
    // instead of silently assuming `column * batch_size` indexing.
    pub(crate) local_constants: &'a [F],
    /// Distance, in field elements, between adjacent constant columns.
    /// Usually this is `batch_size`; quotient evaluation may borrow a window
    /// from a retained full-domain column store instead.
    constant_stride: usize,
    /// First point of this batch inside every constant column.
    constant_offset: usize,
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
            local_constants,
            constant_stride: batch_size,
            constant_offset: 0,
            local_wires,
            public_inputs_hash,
        }
    }

    /// Builds a batch whose constants are a window into a larger column-major
    /// backing store. Wires retain the ordinary compact batch layout.
    pub fn new_with_constant_stride(
        batch_size: usize,
        local_constants: &'a [F],
        constant_stride: usize,
        constant_offset: usize,
        local_wires: &'a [F],
        public_inputs_hash: &'a HashOut<F>,
    ) -> Self {
        assert!(constant_stride >= batch_size);
        assert_eq!(local_constants.len() % constant_stride, 0);
        assert!(
            constant_offset
                .checked_add(batch_size)
                .is_some_and(|end| end <= constant_stride),
            "constant batch window must fit in every column"
        );
        assert_eq!(local_wires.len() % batch_size, 0);
        Self {
            batch_size,
            local_constants,
            constant_stride,
            constant_offset,
            local_wires,
            public_inputs_hash,
        }
    }

    pub fn remove_prefix(&mut self, num_selectors: usize) {
        let skip = num_selectors
            .checked_mul(self.constant_stride)
            .expect("constant column offset overflow");
        self.local_constants = &self.local_constants[skip..];
    }

    /// Returns one logical constant column over this batch's points.
    #[inline]
    pub fn constant_column(&self, column: usize) -> &'a [F] {
        let start = self
            .constant_offset
            .checked_add(
                column
                    .checked_mul(self.constant_stride)
                    .expect("constant column stride overflow"),
            )
            .expect("constant column offset overflow");
        let end = start
            .checked_add(self.batch_size)
            .expect("constant column end overflow");
        &self.local_constants[start..end]
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
        let local_constants = PackedStridedView::new(
            self.local_constants,
            self.constant_stride,
            self.constant_offset + index,
        );
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
            let local_constants = PackedStridedView::new(
                self.vars_batch.local_constants,
                self.vars_batch.constant_stride,
                self.vars_batch.constant_offset + self.i,
            );
            let local_wires =
                PackedStridedView::new(self.vars_batch.local_wires, self.vars_batch.len(), self.i);
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
    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;

    #[test]
    fn strided_constants_preserve_window_and_prefix_semantics() {
        type F = GoldilocksField;
        const BATCH: usize = 3;
        const DOMAIN: usize = 8;
        const COLUMNS: usize = 3;
        const OFFSET: usize = 5;

        let constants = (0..COLUMNS * DOMAIN)
            .map(F::from_canonical_usize)
            .collect::<Vec<_>>();
        let wires = vec![F::ZERO; BATCH];
        let public_inputs_hash = HashOut {
            elements: [F::ZERO; 4],
        };
        let vars = EvaluationVarsBaseBatch::new_with_constant_stride(
            BATCH,
            &constants,
            DOMAIN,
            OFFSET,
            &wires,
            &public_inputs_hash,
        );

        for column in 0..COLUMNS {
            assert_eq!(
                vars.constant_column(column),
                &constants[column * DOMAIN + OFFSET..column * DOMAIN + OFFSET + BATCH]
            );
        }
        for point in 0..BATCH {
            let view = vars.view(point);
            for column in 0..COLUMNS {
                assert_eq!(
                    view.local_constants[column],
                    constants[column * DOMAIN + OFFSET + point]
                );
            }
        }

        let mut after_prefix = vars;
        after_prefix.remove_prefix(1);
        assert_eq!(
            after_prefix.constant_column(0),
            &constants[DOMAIN + OFFSET..DOMAIN + OFFSET + BATCH]
        );
        assert_eq!(
            after_prefix.constant_column(1),
            &constants[2 * DOMAIN + OFFSET..2 * DOMAIN + OFFSET + BATCH]
        );
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
