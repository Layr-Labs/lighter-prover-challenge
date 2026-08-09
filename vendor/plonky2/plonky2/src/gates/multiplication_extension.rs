#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::mem::MaybeUninit;
use core::ops::Range;

use anyhow::Result;

use crate::field::batch_util::batch_multiply_add_inplace;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate which can perform a weighted multiplication, i.e. `result = c0.x.y` on [`ExtensionTarget`].
/// If the config has enough routed wires, it can support several such operations in one gate.
#[derive(Debug, Clone)]
pub struct MulExtensionGate<const D: usize> {
    /// Number of multiplications performed by the gate.
    pub num_ops: usize,
}

impl<const D: usize> MulExtensionGate<D> {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }

    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let wires_per_op = 3 * D;
        config.num_routed_wires / wires_per_op
    }

    pub(crate) const fn wires_ith_multiplicand_0(i: usize) -> Range<usize> {
        3 * D * i..3 * D * i + D
    }
    pub(crate) const fn wires_ith_multiplicand_1(i: usize) -> Range<usize> {
        3 * D * i + D..3 * D * i + 2 * D
    }
    pub(crate) const fn wires_ith_output(i: usize) -> Range<usize> {
        3 * D * i + 2 * D..3 * D * i + 3 * D
    }

    /// Scalar contiguous-column fused evaluation, kept for extension degrees
    /// other than 2 (the packed path only implements the D=2 quadratic case).
    #[allow(clippy::too_many_arguments)]
    fn eval_unfiltered_base_batch_accumulate_scalar<F: RichField + Extendable<D>>(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= <Self as Gate<F, D>>::num_constraints(self) * n);

        let wires = vars_base.local_wires;
        let const_0 = &vars_base.local_constants[..n];
        let ext = |start: usize, p: usize| {
            let mut arr = [F::ZERO; D];
            for (d, a) in arr.iter_mut().enumerate() {
                *a = wires[(start + d) * n + p];
            }
            F::Extension::from_basefield_array(arr)
        };

        // One `D x n` constraint block, reused across the `num_ops` operations.
        // Every slot is assigned by the point loop below before the
        // `batch_multiply_add_inplace` read: `to_basefield_array()` yields exactly
        // `D` values and `p` covers `0..n`, so `(d, p)` sweeps the whole block —
        // and the loop over operations already depends on that, since it never
        // re-zeroes between iterations. The buffer therefore needed neither its
        // zero-fill nor a heap allocation of its own. This accumulate impl runs
        // once per 32-point quotient batch for a gate that stays on the CPU
        // quotient path of every proof, i.e. `quotient_domain / 32` times per
        // proof, and each call was paying a malloc, a free, and a `D * n` memset
        // for a block it immediately overwrote.
        const STACK_SCRATCH: usize = 128;
        let scratch_len = D * n;
        let mut scratch_stack = [MaybeUninit::<F>::uninit(); STACK_SCRATCH];
        let mut scratch_heap;
        let scratch: &mut [F] = if scratch_len <= STACK_SCRATCH {
            // SAFETY: `MaybeUninit<F>` has the same layout and alignment as `F`,
            // and every element of `[..scratch_len]` is written by the point loop
            // below before any is read. Same idiom as `RandomAccessGate`'s
            // stack-or-heap scratch.
            unsafe {
                core::slice::from_raw_parts_mut(
                    scratch_stack[..scratch_len].as_mut_ptr().cast::<F>(),
                    scratch_len,
                )
            }
        } else {
            scratch_heap = vec![F::ZERO; scratch_len];
            &mut scratch_heap
        };
        for i in 0..self.num_ops {
            let m0_start = Self::wires_ith_multiplicand_0(i).start;
            let m1_start = Self::wires_ith_multiplicand_1(i).start;
            let output_start = Self::wires_ith_output(i).start;
            for p in 0..n {
                let multiplicand_0 = ext(m0_start, p);
                let multiplicand_1 = ext(m1_start, p);
                let output = ext(output_start, p);
                let computed_output = (multiplicand_0 * multiplicand_1).scalar_mul(const_0[p]);
                let arr = (output - computed_output).to_basefield_array();
                for (d, a) in arr.iter().enumerate() {
                    scratch[d * n + p] = *a;
                }
            }
            for d in 0..D {
                batch_multiply_add_inplace(
                    &mut combined_gate_constraints[(i * D + d) * n..][..n],
                    &scratch[d * n..][..n],
                    filters,
                );
            }
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for MulExtensionGate<D> {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self { num_ops })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let const_0 = vars.local_constants[0];

        let mut constraints = Vec::with_capacity(self.num_ops * D);
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.get_local_ext_algebra(Self::wires_ith_multiplicand_0(i));
            let multiplicand_1 = vars.get_local_ext_algebra(Self::wires_ith_multiplicand_1(i));
            let output = vars.get_local_ext_algebra(Self::wires_ith_output(i));
            let computed_output = (multiplicand_0 * multiplicand_1).scalar_mul(const_0);

            constraints.extend((output - computed_output).to_basefield_array());
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let const_0 = vars.local_constants[0];

        for i in 0..self.num_ops {
            let multiplicand_0 = vars.get_local_ext(Self::wires_ith_multiplicand_0(i));
            let multiplicand_1 = vars.get_local_ext(Self::wires_ith_multiplicand_1(i));
            let output = vars.get_local_ext(Self::wires_ith_output(i));
            let computed_output = (multiplicand_0 * multiplicand_1).scalar_mul(const_0);

            yield_constr.many((output - computed_output).to_basefield_array());
        }
    }

    /// Contiguous-column fused evaluation: reads each wire as a contiguous
    /// `n`-point column and multiply-adds the filtered constraint rows
    /// straight into the shared buffer, avoiding the per-point strided writes
    /// of the default path. For the quadratic extension (D=2, the only degree
    /// the ranked circuits instantiate) this routes to the 4-lane packed
    /// implementation in [`PackedEvaluableBase`].
    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        if D == 2 {
            self.eval_unfiltered_base_batch_accumulate_packed(
                vars_base,
                filters,
                combined_gate_constraints,
            );
        } else {
            self.eval_unfiltered_base_batch_accumulate_scalar(
                vars_base,
                filters,
                combined_gate_constraints,
            );
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = vars.local_constants[0];

        let mut constraints = Vec::with_capacity(self.num_ops * D);
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.get_local_ext_algebra(Self::wires_ith_multiplicand_0(i));
            let multiplicand_1 = vars.get_local_ext_algebra(Self::wires_ith_multiplicand_1(i));
            let output = vars.get_local_ext_algebra(Self::wires_ith_output(i));
            let computed_output = {
                let mul = builder.mul_ext_algebra(multiplicand_0, multiplicand_1);
                builder.scalar_mul_ext_algebra(const_0, mul)
            };

            let diff = builder.sub_ext_algebra(output, computed_output);
            constraints.extend(diff.to_ext_target_array());
        }

        constraints
    }

    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    MulExtensionGenerator {
                        row,
                        const_0: local_constants[0],
                        i,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * 3 * D
    }

    fn num_constants(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * D
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for MulExtensionGate<D>
{
    /// 4-lane packed evaluation for the quadratic extension (D=2, W=7), the only
    /// degree the ranked circuits instantiate. The dispatch in
    /// `eval_unfiltered_base_batch_accumulate` routes D != 2 to the scalar
    /// contiguous-column path, so this is only ever called with D == 2.
    ///
    /// Constraint value per op, matching `eval_unfiltered` exactly:
    ///   computed = (multiplicand_0 * multiplicand_1).scalar_mul(const_0)
    /// with ext2_mul formula c0 = a0*b0 + W*a1*b1, c1 = a0*b1 + a1*b0 (W = 7),
    /// yielded as (output - computed).to_basefield_array() = [c0, c1] per op.
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        debug_assert_eq!(D, 2);
        let const_0 = vars.local_constants[0];
        let seven = P::from(F::from_canonical_u64(7));
        for i in 0..self.num_ops {
            let m0 = Self::wires_ith_multiplicand_0(i).start;
            let m1 = Self::wires_ith_multiplicand_1(i).start;
            let out = Self::wires_ith_output(i).start;
            let a0 = vars.local_wires[m0];
            let a1 = vars.local_wires[m0 + 1];
            let b0 = vars.local_wires[m1];
            let b1 = vars.local_wires[m1 + 1];
            let o0 = vars.local_wires[out];
            let o1 = vars.local_wires[out + 1];
            let c0 = a0 * b0 + (a1 * b1) * seven;
            let c1 = a0 * b1 + a1 * b0;
            let s0 = c0 * const_0;
            let s1 = c1 * const_0;
            yield_constr.one(o0 - s0);
            yield_constr.one(o1 - s1);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MulExtensionGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    const_0: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for MulExtensionGenerator<F, D>
{
    fn id(&self) -> String {
        "MulExtensionGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        MulExtensionGate::<D>::wires_ith_multiplicand_0(self.i)
            .chain(MulExtensionGate::<D>::wires_ith_multiplicand_1(self.i))
            .map(|i| Target::wire(self.row, i))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let extract_extension = |range: Range<usize>| -> F::Extension {
            let t = ExtensionTarget::from_range(self.row, range);
            witness.get_extension_target(t)
        };

        let multiplicand_0 =
            extract_extension(MulExtensionGate::<D>::wires_ith_multiplicand_0(self.i));
        let multiplicand_1 =
            extract_extension(MulExtensionGate::<D>::wires_ith_multiplicand_1(self.i));

        let output_target =
            ExtensionTarget::from_range(self.row, MulExtensionGate::<D>::wires_ith_output(self.i));

        let computed_output = (multiplicand_0 * multiplicand_1).scalar_mul(self.const_0);

        out_buffer.set_extension_target(output_target, computed_output)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_field(self.const_0)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let const_0 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self { row, const_0, i })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate = MulExtensionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = MulExtensionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }
}
