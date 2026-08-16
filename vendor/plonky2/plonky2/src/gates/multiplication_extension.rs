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
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
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
}

/// Exact ranked D=2/n=32 packed coefficient evaluator. Generic dimensions, shapes, widths, and
/// scalar tails retain the established extension-field path below.
fn eval_quadratic_packed_direct_n32<F: RichField + Extendable<D>, const D: usize>(
    vars: EvaluationVarsBaseBatch<F>,
    filters: &[F],
    combined_gate_constraints: &mut [F],
) {
    type Packing<T> = <T as Packable>::Packing;
    debug_assert_eq!(D, 2);
    debug_assert_eq!(vars.len(), 32);
    debug_assert_eq!(Packing::<F>::WIDTH, 4);
    debug_assert_eq!(filters.len(), 32);
    debug_assert!(combined_gate_constraints.len() >= 26 * 32);

    let wires = vars.local_wires;
    let constants = vars.local_constants;
    let w = Packing::<F>::from(<F as Extendable<D>>::W);
    for op in 0..13 {
        // With D=2, each operation occupies [a0,a1,b0,b1,out0,out1].
        let m0 = 6 * op;
        let m1 = m0 + 2;
        let out = m0 + 4;
        for group in 0..8 {
            let offset = 4 * group;
            let wire = |column| {
                let start = 32 * column + offset;
                *Packing::<F>::from_slice(&wires[start..start + 4])
            };
            let constant = *Packing::<F>::from_slice(&constants[offset..offset + 4]);
            let a0 = wire(m0);
            let a1 = wire(m0 + 1);
            let b0 = wire(m1);
            let b1 = wire(m1 + 1);

            // Direct coefficients of (a0 + a1 X)(b0 + b1 X), X^2 = W. The first
            // product in each sum remains the accumulator; the second is the existing packed
            // multiply-accumulate primitive.
            let prod0 = (a0 * b0)
                .multiply_accumulate(w * a1, b1);
            let prod1 = (a0 * b1).multiply_accumulate(a1, b0);
            let constraint0 = wire(out) - prod0 * constant;
            let constraint1 = wire(out + 1) - prod1 * constant;
            let filter = *Packing::<F>::from_slice(&filters[offset..offset + 4]);

            let row0 = (2 * op) * 32 + offset;
            let slot0 = Packing::<F>::from_slice_mut(
                &mut combined_gate_constraints[row0..row0 + 4],
            );
            *slot0 = slot0.multiply_accumulate(constraint0, filter);
            let row1 = (2 * op + 1) * 32 + offset;
            let slot1 = Packing::<F>::from_slice_mut(
                &mut combined_gate_constraints[row1..row1 + 4],
            );
            *slot1 = slot1.multiply_accumulate(constraint1, filter);
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
    /// of the default path.
    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= <Self as Gate<F, D>>::num_constraints(self) * n);

        if D == 2
            && self.num_ops == 13
            && n == 32
            && <<F as Packable>::Packing as PackedField>::WIDTH == 4
        {
            eval_quadratic_packed_direct_n32::<F, D>(
                vars_base,
                filters,
                combined_gate_constraints,
            );
            return;
        }

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
    use crate::field::types::{Field, Field64, PrimeField64};
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::HashOut;
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

    fn scalar_extension_accumulate_reference(
        gate: &MulExtensionGate<2>,
        vars: EvaluationVarsBaseBatch<GoldilocksField>,
        filters: &[GoldilocksField],
        combined: &mut [GoldilocksField],
    ) {
        type F = GoldilocksField;
        let n = vars.len();
        let wires = vars.local_wires;
        let constants = &vars.local_constants[..n];
        let ext = |start: usize, point: usize| {
            <F as Extendable<2>>::Extension::from_basefield_array([
                wires[start * n + point],
                wires[(start + 1) * n + point],
            ])
        };
        let mut scratch = vec![F::ZERO; 2 * n];
        for op in 0..gate.num_ops {
            let m0 = MulExtensionGate::<2>::wires_ith_multiplicand_0(op).start;
            let m1 = MulExtensionGate::<2>::wires_ith_multiplicand_1(op).start;
            let out = MulExtensionGate::<2>::wires_ith_output(op).start;
            for point in 0..n {
                let product = ext(m0, point) * ext(m1, point);
                let computed = <<F as Extendable<2>>::Extension as FieldExtension<2>>::scalar_mul(
                    &product,
                    constants[point],
                );
                let difference = ext(out, point) - computed;
                let coefficients =
                    <<F as Extendable<2>>::Extension as FieldExtension<2>>::to_basefield_array(
                        &difference,
                    );
                scratch[point] = coefficients[0];
                scratch[n + point] = coefficients[1];
            }
            batch_multiply_add_inplace(
                &mut combined[(2 * op) * n..][..n],
                &scratch[..n],
                filters,
            );
            batch_multiply_add_inplace(
                &mut combined[(2 * op + 1) * n..][..n],
                &scratch[n..],
                filters,
            );
        }
    }

    fn quadratic_raw_value(i: usize) -> GoldilocksField {
        type F = GoldilocksField;
        const EDGES: [u64; 9] = [
            0,
            1,
            2,
            (1u64 << 32) - 1,
            1u64 << 32,
            F::ORDER - 1,
            F::ORDER,
            F::ORDER + 1,
            u64::MAX,
        ];
        let raw = if i < EDGES.len() {
            EDGES[i]
        } else {
            let mut x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            x ^= x >> 29;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x ^ (x >> 31)
        };
        F::from_noncanonical_u64(raw)
    }

    fn raw_words(values: &[GoldilocksField]) -> Vec<u64> {
        values.iter().map(|x| x.to_noncanonical_u64()).collect()
    }

    /// Production n=32 exercises the packed quadratic specialization. Width and batch boundaries
    /// exercise the untouched scalar/extension fallback. The comparison is raw-word strict.
    #[test]
    fn quadratic_packed_accumulate_matches_scalar_extension_raw() {
        let gate = MulExtensionGate::<2> { num_ops: 13 };
        for n in [1usize, 3, 4, 5, 31, 32, 33] {
            let wires = (0..78 * n)
                .map(|i| quadratic_raw_value(i + 0x1000 + n))
                .collect::<Vec<_>>();
            let constants = (0..n)
                .map(|i| quadratic_raw_value(5 * i + 0x2000 + n))
                .collect::<Vec<_>>();
            let filters = (0..n)
                .map(|i| quadratic_raw_value(7 * i + 0x3000 + n))
                .collect::<Vec<_>>();
            let initial = (0..136 * n)
                .map(|i| quadratic_raw_value(13 * i + 0x4000 + n))
                .collect::<Vec<_>>();
            let hash = HashOut {
                elements: core::array::from_fn(|i| quadratic_raw_value(0x5000 + 17 * i + n)),
            };
            let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);
            let mut reference = initial.clone();
            scalar_extension_accumulate_reference(&gate, vars, &filters, &mut reference);
            let mut candidate = initial;
            gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut candidate);
            assert_eq!(raw_words(&candidate), raw_words(&reference), "raw mismatch at n={n}");
        }
    }
}
