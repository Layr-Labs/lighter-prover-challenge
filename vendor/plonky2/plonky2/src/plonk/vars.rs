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
    pub local_constants: &'a [F],
    pub local_wires: &'a [F],
    wire_stride: usize,
    wire_offset: usize,
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
            local_wires,
            wire_stride: batch_size,
            wire_offset: 0,
            public_inputs_hash,
        }
    }

    /// Constructs a batch whose wire columns remain in their retained LDE
    /// storage. Column `c`, point `p` is read at
    /// `local_wires[c * wire_stride + wire_offset + p]`.
    pub fn new_strided_wires(
        batch_size: usize,
        local_constants: &'a [F],
        local_wires: &'a [F],
        wire_stride: usize,
        wire_offset: usize,
        public_inputs_hash: &'a HashOut<F>,
    ) -> Self {
        assert_eq!(local_constants.len() % batch_size, 0);
        assert!(wire_stride >= batch_size);
        assert!(wire_offset + batch_size <= wire_stride);
        assert_eq!(local_wires.len() % wire_stride, 0);
        Self {
            batch_size,
            local_constants,
            local_wires,
            wire_stride,
            wire_offset,
            public_inputs_hash,
        }
    }

    pub fn remove_prefix(&mut self, num_selectors: usize) {
        self.local_constants = &self.local_constants[num_selectors * self.len()..];
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
        let local_wires =
            PackedStridedView::new(self.local_wires, self.wire_stride, self.wire_offset + index);
        EvaluationVarsBase {
            local_constants,
            local_wires,
            public_inputs_hash: self.public_inputs_hash,
        }
    }

    #[inline(always)]
    pub fn wire_column(&self, index: usize) -> &'a [F] {
        let start = index
            .checked_mul(self.wire_stride)
            .and_then(|start| start.checked_add(self.wire_offset))
            .expect("wire column offset overflow");
        let end = start
            .checked_add(self.batch_size)
            .expect("wire column end overflow");
        assert!(end <= self.local_wires.len(), "wire column out of bounds");
        &self.local_wires[start..end]
    }

    #[inline]
    pub fn num_wire_columns(&self) -> usize {
        self.local_wires.len() / self.wire_stride
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
                self.vars_batch.len(),
                self.i,
            );
            let local_wires = PackedStridedView::new(
                self.vars_batch.local_wires,
                self.vars_batch.wire_stride,
                self.vars_batch.wire_offset + self.i,
            );
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
    use super::EvaluationVarsBaseBatch;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::packable::Packable;
    use crate::field::packed::PackedField;
    use crate::field::types::{Field, Field64, PrimeField64};
    use crate::hash::hash_types::HashOut;

    type F = GoldilocksField;
    type P = <F as Packable>::Packing;

    fn raw_words(values: &[F]) -> Vec<u64> {
        values
            .iter()
            .map(|value| value.to_noncanonical_u64())
            .collect()
    }

    #[test]
    fn compact_and_retained_strided_wire_views_are_raw_identical() {
        let mut sizes = vec![1, P::WIDTH.saturating_sub(1).max(1), P::WIDTH, P::WIDTH + 1, 31, 32];
        sizes.sort_unstable();
        sizes.dedup();

        const COLS: usize = 7;
        const STRIDE: usize = 64;
        let hash = HashOut::<F>::default();
        let retained = (0..COLS * STRIDE)
            .map(|i| {
                let limb = i as u64 + 17;
                GoldilocksField(if i % 3 == 0 { F::ORDER + limb } else { limb })
            })
            .collect::<Vec<_>>();

        for n in sizes {
            let offset = STRIDE - n;
            let compact = (0..COLS)
                .flat_map(|column| {
                    retained[column * STRIDE + offset..column * STRIDE + offset + n]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let constants = (0..2 * n)
                .map(|i| F::from_canonical_usize(10_000 + i))
                .collect::<Vec<_>>();
            let compact_vars = EvaluationVarsBaseBatch::new(n, &constants, &compact, &hash);
            let strided_vars = EvaluationVarsBaseBatch::new_strided_wires(
                n,
                &constants,
                &retained,
                STRIDE,
                offset,
                &hash,
            );

            assert_eq!(compact_vars.num_wire_columns(), COLS);
            assert_eq!(strided_vars.num_wire_columns(), COLS);
            for column in 0..COLS {
                assert_eq!(
                    raw_words(compact_vars.wire_column(column)),
                    raw_words(strided_vars.wire_column(column)),
                    "wire column mismatch for n={n}, column={column}"
                );
            }
            for point in 0..n {
                let compact_view = compact_vars.view(point);
                let strided_view = strided_vars.view(point);
                let compact_raw = compact_view
                    .local_wires
                    .iter()
                    .map(|value| value.to_noncanonical_u64())
                    .collect::<Vec<_>>();
                let strided_raw = strided_view
                    .local_wires
                    .iter()
                    .map(|value| value.to_noncanonical_u64())
                    .collect::<Vec<_>>();
                assert_eq!(compact_raw, strided_raw, "scalar view mismatch for n={n}, point={point}");
            }

            let (compact_packed, _) = compact_vars.pack::<P>();
            let (strided_packed, _) = strided_vars.pack::<P>();
            for (pack_index, (compact_pack, strided_pack)) in
                compact_packed.zip(strided_packed).enumerate()
            {
                for column in 0..COLS {
                    assert_eq!(
                        raw_words(compact_pack.local_wires[column].as_slice()),
                        raw_words(strided_pack.local_wires[column].as_slice()),
                        "packed view mismatch for n={n}, pack={pack_index}, column={column}"
                    );
                }
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
