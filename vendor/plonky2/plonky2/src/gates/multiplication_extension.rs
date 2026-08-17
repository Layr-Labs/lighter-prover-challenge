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
        // Width-divisible batches (every production batch, n = 32) take the
        // packed lane path; any other length keeps the scalar path verbatim.
        // The packed path is only written for the quadratic extension
        // (D = 2), where the product formula is two lane-wise base products;
        // other degrees keep the generic scalar body.
        let width = <F as Packable>::Packing::WIDTH;
        if D == 2 && vars_base.len() % width == 0 {
            self.eval_accumulate_packed(vars_base, filters, combined_gate_constraints);
        } else {
            self.eval_accumulate_scalar(vars_base, filters, combined_gate_constraints);
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

/// Inherent helpers for the accumulate path. The trait method above dispatches
/// width-divisible batches (every production batch, n = 32) to the packed lane
/// implementation; anything else runs the original scalar body verbatim.
impl<const D: usize> MulExtensionGate<D> {
    /// The original contiguous-column scalar accumulate, unchanged. Kept as
    /// the reference for non-width-divisible batches and as the oracle for
    /// the packed/scalar differential test.
    fn eval_accumulate_scalar<F: RichField + Extendable<D>>(
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

        const STACK_SCRATCH: usize = 128;
        let scratch_len = D * n;
        let mut scratch_stack = [MaybeUninit::<F>::uninit(); STACK_SCRATCH];
        let mut scratch_heap;
        let scratch: &mut [F] = if scratch_len <= STACK_SCRATCH {
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

    /// Packed-lane accumulate for the quadratic-extension gate (D = 2).
    ///
    /// The constraint for operation `i` and point `p` is
    /// `output - c0 * m0 * m1 = 0` over the extension field, which for D = 2
    /// splits into two independent base-field equations:
    /// `out0 - c0*(m00*m10 + W*m01*m11)` and `out1 - c0*(m00*m11 + m01*m10)`
    /// with `W = 7` for Goldilocks. Each equation is a point-wise independent
    /// combination of the wire columns, so `WIDTH` points process together in
    /// packed lanes; only the final multiply-add into the shared buffer stays
    /// the (already packed) `batch_multiply_add_inplace`.
    ///
    /// Raw representatives can differ from the scalar path's fused u160
    /// products, never the field values; every consumer of the
    /// combined-constraint buffer is congruence-preserving (the same contract
    /// as the packed vanishing rows).
    fn eval_accumulate_packed<F: RichField + Extendable<D> + Packable>(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        use crate::field::packable::Packable;
        use crate::field::packed::PackedField;

        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= <Self as Gate<F, D>>::num_constraints(self) * n);
        debug_assert_eq!(D, 2);
        debug_assert_eq!(n % <F as Packable>::Packing::WIDTH, 0);

        type Packing<F> = <F as Packable>::Packing;
        let width = Packing::<F>::WIDTH;
        let n_groups = n / width;
        // Broadcast W = <F as Extendable<D>>::W into every lane: ONES * scalar
        // uses the trait's scalar-add, so each lane holds the field element W.
        let w = Packing::<F>::ONES * <F as Extendable<D>>::W;

        let wires = vars_base.local_wires;
        let const_0 = &vars_base.local_constants[..n];
        let col = |start: usize, d: usize| &wires[(start + d) * n..][..n];

        const STACK_SCRATCH: usize = 128;
        let scratch_len = D * n;
        let mut scratch_stack = [MaybeUninit::<F>::uninit(); STACK_SCRATCH];
        let mut scratch_heap;
        let scratch: &mut [F] = if scratch_len <= STACK_SCRATCH {
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

            // For each group of `width` points, form the extension operands as
            // packed lanes and evaluate both base components lane-wise.
            for g in 0..n_groups {
                let m00 = *Packing::<F>::from_slice(&col(m0_start, 0)[g * width..][..width]);
                let m01 = *Packing::<F>::from_slice(&col(m0_start, 1)[g * width..][..width]);
                let m10 = *Packing::<F>::from_slice(&col(m1_start, 0)[g * width..][..width]);
                let m11 = *Packing::<F>::from_slice(&col(m1_start, 1)[g * width..][..width]);
                let o0 = *Packing::<F>::from_slice(&col(output_start, 0)[g * width..][..width]);
                let o1 = *Packing::<F>::from_slice(&col(output_start, 1)[g * width..][..width]);
                let c = *Packing::<F>::from_slice(&const_0[g * width..][..width]);

                // (m00 + m01*u) * (m10 + m11*u) over X^2 - W.
                let t0 = m00 * m10 + w * (m01 * m11);
                let t1 = m00 * m11 + m01 * m10;

                let (d0, d1) = (o0 - t0 * c, o1 - t1 * c);
                scratch[0 * n + g * width..][..width].copy_from_slice(d0.as_slice());
                scratch[1 * n + g * width..][..width].copy_from_slice(d1.as_slice());
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

    /// Packed accumulate must agree with the scalar accumulate on the field
    /// values of every combined-constraint slot, across all batch sizes
    /// (including non-width-divisible ones, which stay on the scalar path).
    /// Raw representatives may differ (packed lanes fuse products), so the
    /// comparison canonicalizes.
    #[test]
    fn packed_accumulate_matches_scalar_across_batch_sizes() {
        use crate::field::packable::Packable;
        use crate::field::packed::PackedField;
        use crate::field::types::{Field, Field64, PrimeField64, Sample};

        const D: usize = 2;
        type F = GoldilocksField;
        let gate: MulExtensionGate<D> =
            MulExtensionGate::<D>::new_from_config(&CircuitConfig::standard_recursion_config());

        let width = <F as Packable>::Packing::WIDTH;
        let mut batch_sizes = vec![1usize, 3, 5, 7, 11, 31, 32, 33];
        batch_sizes.extend((0..width).map(|r| width + r));
        batch_sizes.extend((0..width).map(|r| 2 * width + r));
        batch_sizes.sort_unstable();
        batch_sizes.dedup();

        for &n in &batch_sizes {
            let num_wires = <MulExtensionGate<D> as Gate<F, D>>::num_wires(&gate);
            let num_constants = <MulExtensionGate<D> as Gate<F, D>>::num_constants(&gate);
            let mut wires = (0..num_wires * n)
                .map(|i| {
                    let v = ((i as u64).wrapping_mul(0x9e37_79b9) ^ 0x5a5a_a5a5) & 0xffff;
                    if i % 3 == 0 {
                        F::from_canonical_u64(F::ORDER + v)
                    } else {
                        F::from_canonical_u64(v)
                    }
                })
                .collect::<Vec<_>>();
            let constants = (0..num_constants * n)
                .map(|i| F::from_canonical_u64(((i as u64).wrapping_mul(0x1234_5678) ^ 0xdead_beef) & 0xffff))
                .collect::<Vec<_>>();
            let filters = (0..n)
                .map(|i| match i % 7 {
                    0 => F::ZERO,
                    1 => F::from_canonical_u64(F::ORDER), // noncanonical zero
                    _ => F::from_canonical_u64(((i as u64).wrapping_mul(0xabcd_ef01) ^ 0x1234_5678) & 0xffff),
                })
                .collect::<Vec<_>>();

            let public_inputs_hash = crate::hash::hash_types::HashOut::rand();

            let mut reference = vec![F::ZERO; <MulExtensionGate<D> as Gate<F, D>>::num_constraints(&gate) * n];
            let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &public_inputs_hash);
            gate.eval_accumulate_scalar(vars.clone(), &filters, &mut reference);

            let mut actual = vec![F::ZERO; <MulExtensionGate<D> as Gate<F, D>>::num_constraints(&gate) * n];
            let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &public_inputs_hash);
            if D == 2 && n % width == 0 {
                gate.eval_accumulate_packed(vars.clone(), &filters, &mut actual);
            } else {
                gate.eval_accumulate_scalar(vars.clone(), &filters, &mut actual);
            }

            for (slot, (r, a)) in reference.iter().zip(&actual).enumerate() {
                assert_eq!(
                    r.to_canonical_u64(),
                    a.to_canonical_u64(),
                    "n={n} slot={slot}: scalar {r:?} != packed {a:?}"
                );
            }
        }
    }
}
