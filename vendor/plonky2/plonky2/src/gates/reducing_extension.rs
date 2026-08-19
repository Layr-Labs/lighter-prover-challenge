#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
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
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Computes `sum alpha^i c_i` for a vector `c_i` of `num_coeffs` elements of the extension field.
#[derive(Debug, Clone, Default)]
pub struct ReducingExtensionGate<const D: usize> {
    pub num_coeffs: usize,
}

impl<const D: usize> ReducingExtensionGate<D> {
    pub const fn new(num_coeffs: usize) -> Self {
        Self { num_coeffs }
    }

    pub fn max_coeffs_len(num_wires: usize, num_routed_wires: usize) -> usize {
        // `3*D` routed wires are used for the output, alpha and old accumulator.
        // Need `num_coeffs*D` routed wires for coeffs, and `(num_coeffs-1)*D` wires for accumulators.
        ((num_routed_wires - 3 * D) / D).min((num_wires - 2 * D) / (D * 2))
    }

    pub(crate) const fn wires_output() -> Range<usize> {
        0..D
    }
    pub(crate) const fn wires_alpha() -> Range<usize> {
        D..2 * D
    }
    pub(crate) const fn wires_old_acc() -> Range<usize> {
        2 * D..3 * D
    }
    const START_COEFFS: usize = 3 * D;
    pub(crate) const fn wires_coeff(i: usize) -> Range<usize> {
        Self::START_COEFFS + i * D..Self::START_COEFFS + (i + 1) * D
    }
    const fn start_accs(&self) -> usize {
        Self::START_COEFFS + self.num_coeffs * D
    }
    const fn wires_accs(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_coeffs);
        if i == self.num_coeffs - 1 {
            // The last accumulator is the output.
            return Self::wires_output();
        }
        self.start_accs() + D * i..self.start_accs() + D * (i + 1)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ReducingExtensionGate<D> {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_coeffs)?;
        Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized,
    {
        let num_coeffs = src.read_usize()?;
        Ok(Self::new(num_coeffs))
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let alpha = vars.get_local_ext_algebra(Self::wires_alpha());
        let old_acc = vars.get_local_ext_algebra(Self::wires_old_acc());
        let coeffs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(Self::wires_coeff(i)))
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut constraints = Vec::with_capacity(<Self as Gate<F, D>>::num_constraints(self));
        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            constraints.push(acc * alpha + coeffs[i] - accs[i]);
            acc = accs[i];
        }

        constraints
            .into_iter()
            .flat_map(|alg| alg.to_basefield_array())
            .collect()
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let alpha = vars.get_local_ext(Self::wires_alpha());
        let old_acc = vars.get_local_ext(Self::wires_old_acc());
        let coeffs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext(Self::wires_coeff(i)))
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            yield_constr.many((acc * alpha + coeffs[i] - accs[i]).to_basefield_array());
            acc = accs[i];
        }
    }

    /// Dispatcher: width-divisible D=2 batches take the packed lane
    /// path, everything else falls back to the verbatim scalar oracle.
    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
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
        let alpha = vars.get_local_ext_algebra(Self::wires_alpha());
        let old_acc = vars.get_local_ext_algebra(Self::wires_old_acc());
        let coeffs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(Self::wires_coeff(i)))
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut constraints = Vec::with_capacity(<Self as Gate<F, D>>::num_constraints(self));
        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            let coeff = coeffs[i];
            let mut tmp = builder.mul_add_ext_algebra(acc, alpha, coeff);
            tmp = builder.sub_ext_algebra(tmp, accs[i]);
            constraints.push(tmp);
            acc = accs[i];
        }

        constraints
            .into_iter()
            .flat_map(|alg| alg.to_ext_target_array())
            .collect()
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            ReducingGenerator {
                row,
                gate: self.clone(),
            }
            .adapter(),
        )]
    }

    fn num_wires(&self) -> usize {
        2 * D + 2 * D * self.num_coeffs
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        D * self.num_coeffs
    }
}

impl<const D: usize> ReducingExtensionGate<D> {
    /// Verbatim scalar body — differential oracle for the packed path.
    fn eval_accumulate_scalar<F: RichField + Extendable<D> + Packable>(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= <Self as Gate<F, D>>::num_constraints(self) * n);
        let wires = vars_base.local_wires;
        let ext = |start: usize, p: usize| {
            let mut arr = [F::ZERO; D];
            for (d, a) in arr.iter_mut().enumerate() {
                *a = wires[(start + d) * n + p];
            }
            F::Extension::from_basefield_array(arr)
        };
        let alphas: Vec<F::Extension> = (0..n).map(|p| ext(Self::wires_alpha().start, p)).collect();
        let mut accs: Vec<F::Extension> = (0..n).map(|p| ext(Self::wires_old_acc().start, p)).collect();
        let mut scratch = vec![F::ZERO; D * n];
        for i in 0..self.num_coeffs {
            let coeff_start = Self::wires_coeff(i).start;
            let acc_start = self.wires_accs(i).start;
            for p in 0..n {
                let next_acc = ext(acc_start, p);
                let constraint = accs[p] * alphas[p] + ext(coeff_start, p) - next_acc;
                let arr = constraint.to_basefield_array();
                for (d, a) in arr.iter().enumerate() {
                    scratch[d * n + p] = *a;
                }
                accs[p] = next_acc;
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

    /// Packed-lane accumulate for D=2 production batches (n % WIDTH == 0, WIDTH=4 on aarch64).
    fn eval_accumulate_packed<F: RichField + Extendable<D> + Packable>(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        debug_assert_eq!(D, 2);
        let n = vars_base.len();
        let width = <F as Packable>::Packing::WIDTH;
        debug_assert_eq!(n % width, 0);
        let num_groups = n / width;
        let wires = vars_base.local_wires;
        let w = <F as Extendable<D>>::W;
        let packing = <F as Packable>::Packing::from_slice;
        // per-group packed carries
        let mut acc0: Vec<F::Packing> = Vec::with_capacity(num_groups);
        let mut acc1: Vec<F::Packing> = Vec::with_capacity(num_groups);
        let mut alpha0: Vec<F::Packing> = Vec::with_capacity(num_groups);
        let mut alpha1: Vec<F::Packing> = Vec::with_capacity(num_groups);
        let old_acc_start = Self::wires_old_acc().start;
        let alpha_start = Self::wires_alpha().start;
        for g in 0..num_groups {
            let off = g * width;
            acc0.push(*packing(&wires[old_acc_start * n + off..][..width]));
            acc1.push(*packing(&wires[(old_acc_start + 1) * n + off..][..width]));
            alpha0.push(*packing(&wires[alpha_start * n + off..][..width]));
            alpha1.push(*packing(&wires[(alpha_start + 1) * n + off..][..width]));
        }
        use core::mem::MaybeUninit;
        const STACK_SCRATCH: usize = 32;
        let mut scratch_stack = [MaybeUninit::<F>::uninit(); STACK_SCRATCH];
        let scratch_len = 2 * width;
        let mut scratch_heap;
        let scratch: &mut [F] = if scratch_len <= STACK_SCRATCH {
            unsafe { core::slice::from_raw_parts_mut(scratch_stack[..scratch_len].as_mut_ptr().cast::<F>(), scratch_len) }
        } else {
            scratch_heap = vec![F::ZERO; scratch_len];
            &mut scratch_heap
        };
        for i in 0..self.num_coeffs {
            let coeff_start = Self::wires_coeff(i).start;
            let acc_start = self.wires_accs(i).start;
            for g in 0..num_groups {
                let off = g * width;
                let c0 = *packing(&wires[coeff_start * n + off..][..width]);
                let c1 = *packing(&wires[(coeff_start + 1) * n + off..][..width]);
                let n0 = *packing(&wires[acc_start * n + off..][..width]);
                let n1 = *packing(&wires[(acc_start + 1) * n + off..][..width]);
                let cur0 = acc0[g];
                let cur1 = acc1[g];
                let al0 = alpha0[g];
                let al1 = alpha1[g];
                let prod0 = cur0 * al0 + (cur1 * al1) * w;
                let prod1 = cur0 * al1 + cur1 * al0;
                let term0 = (prod0 + c0) - n0;
                let term1 = (prod1 + c1) - n1;
                scratch[..width].copy_from_slice(term0.as_slice());
                scratch[width..2*width].copy_from_slice(term1.as_slice());
                acc0[g] = n0;
                acc1[g] = n1;
                batch_multiply_add_inplace(&mut combined_gate_constraints[(i * 2) * n + off..][..width], &scratch[..width], &filters[off..off+width]);
                batch_multiply_add_inplace(&mut combined_gate_constraints[(i * 2 + 1) * n + off..][..width], &scratch[width..2*width], &filters[off..off+width]);
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct ReducingGenerator<const D: usize> {
    row: usize,
    gate: ReducingExtensionGate<D>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for ReducingGenerator<D> {
    fn id(&self) -> String {
        "ReducingExtensionGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        ReducingExtensionGate::<D>::wires_alpha()
            .chain(ReducingExtensionGate::<D>::wires_old_acc())
            .chain((0..self.gate.num_coeffs).flat_map(ReducingExtensionGate::<D>::wires_coeff))
            .map(|i| Target::wire(self.row, i))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_extension = |range: Range<usize>| -> F::Extension {
            let t = ExtensionTarget::from_range(self.row, range);
            witness.get_extension_target(t)
        };

        let alpha = local_extension(ReducingExtensionGate::<D>::wires_alpha());
        let old_acc = local_extension(ReducingExtensionGate::<D>::wires_old_acc());
        let coeffs = (0..self.gate.num_coeffs)
            .map(|i| local_extension(ReducingExtensionGate::<D>::wires_coeff(i)))
            .collect::<Vec<_>>();
        let accs = (0..self.gate.num_coeffs)
            .map(|i| ExtensionTarget::from_range(self.row, self.gate.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut acc = old_acc;
        for i in 0..self.gate.num_coeffs {
            let computed_acc = acc * alpha + coeffs[i];
            out_buffer.set_extension_target(accs[i], computed_acc)?;
            acc = computed_acc;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        <ReducingExtensionGate<D> as Gate<F, D>>::serialize(&self.gate, dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let gate = <ReducingExtensionGate<D> as Gate<F, D>>::deserialize(src, _common_data)?;
        Ok(Self { row, gate })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::reducing_extension::ReducingExtensionGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(ReducingExtensionGate::new(22));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(ReducingExtensionGate::new(22))
    }

    #[test]
    fn packed_accumulate_matches_scalar_across_batch_sizes() {
        use crate::field::packable::Packable;
        use crate::field::packed::PackedField;
        use crate::field::types::{Field, Field64, PrimeField64};
        use crate::gates::gate::Gate;
        use crate::hash::hash_types::HashOut;
        use crate::plonk::vars::EvaluationVarsBaseBatch;
        const D: usize = 2;
        type F = GoldilocksField;
        let gate = ReducingExtensionGate::<D>::new(22);
        fn value(i: usize) -> F {
            let small = ((i as u64).wrapping_mul(0x9e37_79b9) ^ 0x5a5a_a5a5) & 0xffff;
            if i % 3 == 0 { GoldilocksField(F::ORDER + small) } else { F::from_canonical_u64(small) }
        }
        let packing_width = <<F as Packable>::Packing as PackedField>::WIDTH;
        let mut batch_sizes = vec![1,3,5,7,11,31,32,33, packing_width.saturating_sub(1).max(1), packing_width, packing_width+1, packing_width+2, 2*packing_width-1, 2*packing_width, 2*packing_width+1];
        batch_sizes.extend((0..packing_width).map(|r| packing_width + r));
        batch_sizes.extend((0..packing_width).map(|r| 2*packing_width + r));
        batch_sizes.sort_unstable(); batch_sizes.dedup();
        for &n in &batch_sizes {
            let num_wires = <ReducingExtensionGate<D> as Gate<F, D>>::num_wires(&gate);
            let num_constraints = <ReducingExtensionGate<D> as Gate<F, D>>::num_constraints(&gate);
            let wires = (0..num_wires*n).map(|i| value(i+1)).collect::<Vec<_>>();
            let filters = (0..n).map(|i| match i%7 {0=>F::ZERO, 1=>GoldilocksField(F::ORDER), _=>value(i+20001)}).collect::<Vec<_>>();
            let hash = HashOut::ZERO;
            let vars = EvaluationVarsBaseBatch::new(n, &[], &wires, &hash);
            let initial = (0..num_constraints*n).map(|i| match i%11 {0=>F::ZERO, 1=>GoldilocksField(F::ORDER), _=>value(i+30001)}).collect::<Vec<_>>();
            let mut expected = initial.clone();
            gate.eval_accumulate_scalar(vars, &filters, &mut expected);
            let mut actual = initial;
            {
                let vars2 = EvaluationVarsBaseBatch::new(n, &[], &wires, &hash);
                gate.eval_unfiltered_base_batch_accumulate(vars2, &filters, &mut actual);
            }
            for (i, (&e,&a)) in expected.iter().zip(&actual).enumerate() {
                assert_eq!(a.to_canonical_u64(), e.to_canonical_u64(), "n={n} idx={i}");
            }
        }
    }
}
