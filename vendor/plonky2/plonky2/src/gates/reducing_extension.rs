#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::mem::MaybeUninit;
use core::ops::Range;

use anyhow::Result;

use crate::field::batch_util::batch_multiply_add_inplace;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
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

#[cfg(feature = "std")]
fn reducing_extension_stack_scratch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var_os("LIGHTER_REDUCING_STACK").is_some_and(|value| value == "0")
    })
}

#[cfg(not(feature = "std"))]
fn reducing_extension_stack_scratch_enabled() -> bool {
    true
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

    /// Contiguous-column fused evaluation: reads each wire as a contiguous
    /// `n`-point column, evaluates one accumulator step at a time and
    /// multiply-adds the filtered constraint rows straight into the shared
    /// buffer, avoiding the per-point strided writes and per-point `Vec`
    /// allocations of the default path.
    fn eval_unfiltered_base_batch_accumulate(
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

        const STACK_POINTS: usize = 32;
        const STACK_BASE_VALUES: usize = 128;
        let scratch_len = D
            .checked_mul(n)
            .expect("reducing extension gate scratch length overflow");
        let use_stack = reducing_extension_stack_scratch_enabled()
            && n <= STACK_POINTS
            && scratch_len <= STACK_BASE_VALUES;
        let mut alpha_stack = [MaybeUninit::<F::Extension>::uninit(); STACK_POINTS];
        let mut acc_stack = [MaybeUninit::<F::Extension>::uninit(); STACK_POINTS];
        let mut scratch_stack = [MaybeUninit::<F>::uninit(); STACK_BASE_VALUES];
        let mut alpha_heap;
        let mut acc_heap;
        let mut scratch_heap;
        let (alpha_slots, acc_slots, scratch_slots): (
            &mut [MaybeUninit<F::Extension>],
            &mut [MaybeUninit<F::Extension>],
            &mut [MaybeUninit<F>],
        ) = if use_stack {
            (
                &mut alpha_stack[..n],
                &mut acc_stack[..n],
                &mut scratch_stack[..scratch_len],
            )
        } else {
            // Preserve the heap fallback's lengths and valid ZERO initialization.
            alpha_heap = vec![MaybeUninit::new(F::Extension::ZERO); n];
            acc_heap = vec![MaybeUninit::new(F::Extension::ZERO); n];
            scratch_heap = vec![MaybeUninit::new(F::ZERO); scratch_len];
            (&mut alpha_heap, &mut acc_heap, &mut scratch_heap)
        };
        for p in 0..n {
            alpha_slots[p].write(ext(Self::wires_alpha().start, p));
            acc_slots[p].write(ext(Self::wires_old_acc().start, p));
        }

        // SAFETY: both slot prefixes have length n, and the preceding loop
        // initialized every element with a valid extension-field value.
        // MaybeUninit has the same layout and alignment as its element type;
        // the backing buffers are distinct and their slot views are not used
        // while these typed views are live.
        let alphas: &[F::Extension] =
            unsafe { core::slice::from_raw_parts(alpha_slots.as_ptr().cast::<F::Extension>(), n) };
        let accs: &mut [F::Extension] = unsafe {
            core::slice::from_raw_parts_mut(acc_slots.as_mut_ptr().cast::<F::Extension>(), n)
        };

        for i in 0..self.num_coeffs {
            let coeff_start = Self::wires_coeff(i).start;
            let acc_start = self.wires_accs(i).start;
            for p in 0..n {
                let next_acc = ext(acc_start, p);
                let constraint = accs[p] * alphas[p] + ext(coeff_start, p) - next_acc;
                let arr = constraint.to_basefield_array();
                for (d, a) in arr.iter().enumerate() {
                    scratch_slots[d * n + p].write(*a);
                }
                accs[p] = next_acc;
            }
            for d in 0..D {
                let row_slots = &scratch_slots[d * n..][..n];
                // SAFETY: this coefficient's point loop wrote every slot in
                // every active row before any row is viewed. This temporary
                // shared view ends with the batch read, before the next write.
                let scratch_row: &[F] =
                    unsafe { core::slice::from_raw_parts(row_slots.as_ptr().cast::<F>(), n) };
                batch_multiply_add_inplace(
                    &mut combined_gate_constraints[(i * D + d) * n..][..n],
                    scratch_row,
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
    use crate::field::types::Sample;
    use crate::gates::gate::Gate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::reducing_extension::ReducingExtensionGate;
    use crate::hash::hash_types::HashOut;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::vars::EvaluationVarsBaseBatch;

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(ReducingExtensionGate::new(22));
    }

    fn check_batch_accumulate_at_size(gate: ReducingExtensionGate<D>, n: usize) {
        let num_wires = <ReducingExtensionGate<D> as Gate<F, D>>::num_wires(&gate);
        let num_constants = <ReducingExtensionGate<D> as Gate<F, D>>::num_constants(&gate);
        let num_constraints = <ReducingExtensionGate<D> as Gate<F, D>>::num_constraints(&gate);
        let wires = F::rand_vec(num_wires * n);
        let constants = F::rand_vec(num_constants * n);
        let public_inputs_hash = HashOut::rand();
        let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &public_inputs_hash);
        let reference = gate.eval_unfiltered_base_batch(vars);
        let filters = F::rand_vec(n);
        let mut expected = F::rand_vec(num_constraints * n);
        let mut actual = expected.clone();
        for (expected_row, reference_row) in
            expected.chunks_exact_mut(n).zip(reference.chunks_exact(n))
        {
            for ((value, &constraint), &filter) in
                expected_row.iter_mut().zip(reference_row).zip(&filters)
            {
                *value += constraint * filter;
            }
        }
        gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);
        assert_eq!(actual, expected, "batch size {n}");
    }

    #[test]
    fn batch_accumulate_stack_boundary() {
        for n in [1, 31, 32, 33] {
            check_batch_accumulate_at_size(ReducingExtensionGate::new(22), n);
        }
    }

    #[test]
    fn eval_fns() -> Result<()> {
        test_eval_fns::<F, C, _, D>(ReducingExtensionGate::new(22))
    }
}
