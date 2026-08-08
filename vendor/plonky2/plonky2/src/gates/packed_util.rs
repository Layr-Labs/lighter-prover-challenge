#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::field::extension::Extendable;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::plonk::vars::{EvaluationVarsBaseBatch, EvaluationVarsBasePacked};

pub trait PackedEvaluableBase<F: RichField + Extendable<D>, const D: usize>: Gate<F, D> {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars_base: EvaluationVarsBasePacked<P>,
        yield_constr: StridedConstraintConsumer<P>,
    );

    /// Evaluates entire batch of points. Returns a matrix of constraints. Constraint `j` for point
    /// `i` is at `index j * batch_size + i`.
    fn eval_unfiltered_base_batch_packed(&self, vars_batch: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        let mut res = vec![F::ZERO; vars_batch.len() * self.num_constraints()];
        let (vars_packed_iter, vars_leftovers_iter) = vars_batch.pack::<<F as Packable>::Packing>();
        let leftovers_start = vars_batch.len() - vars_leftovers_iter.len();
        for (i, vars_packed) in vars_packed_iter.enumerate() {
            self.eval_unfiltered_base_packed(
                vars_packed,
                StridedConstraintConsumer::new(
                    &mut res[..],
                    vars_batch.len(),
                    <F as Packable>::Packing::WIDTH * i,
                ),
            );
        }
        for (i, vars_leftovers) in vars_leftovers_iter.enumerate() {
            self.eval_unfiltered_base_packed(
                vars_leftovers,
                StridedConstraintConsumer::new(&mut res[..], vars_batch.len(), leftovers_start + i),
            );
        }
        res
    }

    /// Filter-fused variant of [`Self::eval_unfiltered_base_batch_packed`]: evaluates one packed
    /// lane group at a time and multiply-accumulates each emitted row directly into the shared
    /// quotient buffer, instead of materializing and rereading a temporary constraint block. Per
    /// point `p` and constraint `j` this performs the same
    /// `combined[j * n + p] += constraint * filters[p]` product-accumulate the materialize-then-add
    /// default performs, with the same packed evaluation producing each constraint value, so the
    /// result is value-identical; only the memory traffic changes.
    fn eval_unfiltered_base_batch_accumulate_packed(
        &self,
        vars_batch: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_batch.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= self.num_constraints() * n);
        let width = <<F as Packable>::Packing as PackedField>::WIDTH;

        let (vars_packed_iter, vars_leftovers_iter) = vars_batch.pack::<<F as Packable>::Packing>();
        let leftovers_start = n - vars_leftovers_iter.len();
        for (i, vars_packed) in vars_packed_iter.enumerate() {
            let offset = width * i;
            let filter = *<F as Packable>::Packing::from_slice(
                &filters[offset..offset + width],
            );
            self.eval_unfiltered_base_packed(
                vars_packed,
                StridedConstraintConsumer::new_accumulating(
                    combined_gate_constraints,
                    n,
                    offset,
                    filter,
                ),
            );
        }
        for (i, vars_leftovers) in vars_leftovers_iter.enumerate() {
            let point = leftovers_start + i;
            self.eval_unfiltered_base_packed(
                vars_leftovers,
                StridedConstraintConsumer::new_accumulating(
                    combined_gate_constraints,
                    n,
                    point,
                    filters[point],
                ),
            );
        }
    }
}
