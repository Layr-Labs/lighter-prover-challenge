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

    /// Evaluates, filters, and adds an entire batch without materializing a constraint matrix.
    fn eval_unfiltered_base_batch_packed_accumulate(
        &self,
        vars_batch: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let batch_size = vars_batch.len();
        let required_len = batch_size * self.num_constraints();
        assert_eq!(filters.len(), batch_size);
        assert!(combined_gate_constraints.len() >= required_len);
        let combined_gate_constraints = &mut combined_gate_constraints[..required_len];

        let (vars_packed_iter, vars_leftovers_iter) = vars_batch.pack::<<F as Packable>::Packing>();
        let leftovers_start = batch_size - vars_leftovers_iter.len();
        for (i, vars_packed) in vars_packed_iter.enumerate() {
            let offset = <F as Packable>::Packing::WIDTH * i;
            let filter = *<F as Packable>::Packing::from_slice(
                &filters[offset..offset + <F as Packable>::Packing::WIDTH],
            );
            self.eval_unfiltered_base_packed(
                vars_packed,
                StridedConstraintConsumer::new_accumulating(
                    combined_gate_constraints,
                    batch_size,
                    offset,
                    filter,
                ),
            );
        }
        for (i, vars_leftovers) in vars_leftovers_iter.enumerate() {
            let offset = leftovers_start + i;
            self.eval_unfiltered_base_packed(
                vars_leftovers,
                StridedConstraintConsumer::new_accumulating(
                    combined_gate_constraints,
                    batch_size,
                    offset,
                    filters[offset],
                ),
            );
        }
    }
}
