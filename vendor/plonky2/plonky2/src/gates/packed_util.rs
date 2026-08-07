#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::field::extension::Extendable;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::gates::gate::{Gate, resize_constraint_scratch};
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
        let mut res = Vec::new();
        self.eval_unfiltered_base_batch_packed_into(vars_batch, &mut res);
        res
    }

    /// Like [`PackedEvaluableBase::eval_unfiltered_base_batch_packed`], but
    /// writes into a caller-owned buffer so the quotient loop reuses one
    /// allocation across every gate and every batch. Every cell is written
    /// before it is read, so `out`'s prior contents do not matter.
    fn eval_unfiltered_base_batch_packed_into(
        &self,
        vars_batch: EvaluationVarsBaseBatch<F>,
        out: &mut Vec<F>,
    ) {
        let batch_size = vars_batch.len();
        resize_constraint_scratch(out, batch_size * self.num_constraints());
        let (vars_packed_iter, vars_leftovers_iter) = vars_batch.pack::<<F as Packable>::Packing>();
        let leftovers_start = batch_size - vars_leftovers_iter.len();
        for (i, vars_packed) in vars_packed_iter.enumerate() {
            self.eval_unfiltered_base_packed(
                vars_packed,
                StridedConstraintConsumer::new(
                    &mut out[..],
                    batch_size,
                    <F as Packable>::Packing::WIDTH * i,
                ),
            );
        }
        for (i, vars_leftovers) in vars_leftovers_iter.enumerate() {
            self.eval_unfiltered_base_packed(
                vars_leftovers,
                StridedConstraintConsumer::new(&mut out[..], batch_size, leftovers_start + i),
            );
        }
    }
}
