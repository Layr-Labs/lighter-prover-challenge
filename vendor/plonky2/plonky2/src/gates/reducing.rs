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

/// Computes `sum alpha^i c_i` for a vector `c_i` of `num_coeffs` elements of the base field.
#[derive(Debug, Default, Clone)]
pub struct ReducingGate<const D: usize> {
    pub num_coeffs: usize,
}

impl<const D: usize> ReducingGate<D> {
    pub const fn new(num_coeffs: usize) -> Self {
        Self { num_coeffs }
    }

    pub fn max_coeffs_len(num_wires: usize, num_routed_wires: usize) -> usize {
        (num_routed_wires - 3 * D).min((num_wires - 2 * D) / (D + 1))
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
    pub(crate) const fn wires_coeffs(&self) -> Range<usize> {
        Self::START_COEFFS..Self::START_COEFFS + self.num_coeffs
    }
    const fn start_accs(&self) -> usize {
        Self::START_COEFFS + self.num_coeffs
    }
    const fn wires_accs(&self, i: usize) -> Range<usize> {
        if i == self.num_coeffs - 1 {
            // The last accumulator is the output.
            return Self::wires_output();
        }
        self.start_accs() + D * i..self.start_accs() + D * (i + 1)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ReducingGate<D> {
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
        let coeffs = self
            .wires_coeffs()
            .map(|i| vars.local_wires[i])
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut constraints = Vec::with_capacity(<Self as Gate<F, D>>::num_constraints(self));
        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            constraints.push(acc * alpha + coeffs[i].into() - accs[i]);
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
        let coeffs = self
            .wires_coeffs()
            .map(|i| vars.local_wires[i])
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            yield_constr.many((acc * alpha + coeffs[i].into() - accs[i]).to_basefield_array());
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

        // Same stack-or-heap scratch / running-acc contract as
        // `ReducingExtensionGate`: production batches are 32 points, the
        // scratch is fully overwritten before it is read, and the two
        // extension vectors were a malloc+free per quotient batch.
        const STACK_SCRATCH: usize = 128;
        const STACK_POINTS: usize = 64;
        let scratch_len = D * n;
        let mut scratch_stack = [MaybeUninit::<F>::uninit(); STACK_SCRATCH];
        let mut scratch_heap;
        let scratch: &mut [F] = if scratch_len <= STACK_SCRATCH {
            // SAFETY: `MaybeUninit<F>` matches `F` layout/alignment. The
            // point loop writes `[..scratch_len]` before any read.
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

        let mut alpha_stack = [MaybeUninit::<F::Extension>::uninit(); STACK_POINTS];
        let mut acc_stack = [MaybeUninit::<F::Extension>::uninit(); STACK_POINTS];
        let mut alpha_heap;
        let mut acc_heap;
        let (alphas, accs): (&mut [F::Extension], &mut [F::Extension]) = if n <= STACK_POINTS {
            // SAFETY: both slices are written in the fill loop below before
            // the Horner body reads them.
            unsafe {
                (
                    core::slice::from_raw_parts_mut(
                        alpha_stack[..n].as_mut_ptr().cast::<F::Extension>(),
                        n,
                    ),
                    core::slice::from_raw_parts_mut(
                        acc_stack[..n].as_mut_ptr().cast::<F::Extension>(),
                        n,
                    ),
                )
            }
        } else {
            alpha_heap = (0..n).map(|p| ext(Self::wires_alpha().start, p)).collect();
            acc_heap = (0..n)
                .map(|p| ext(Self::wires_old_acc().start, p))
                .collect();
            (alpha_heap.as_mut_slice(), acc_heap.as_mut_slice())
        };
        for p in 0..n {
            alphas[p] = ext(Self::wires_alpha().start, p);
            accs[p] = ext(Self::wires_old_acc().start, p);
        }

        for i in 0..self.num_coeffs {
            let coeff = &wires[(Self::START_COEFFS + i) * n..][..n];
            let acc_start = self.wires_accs(i).start;
            for p in 0..n {
                let next_acc = ext(acc_start, p);
                let constraint = accs[p] * alphas[p] + coeff[p].into() - next_acc;
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

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let alpha = vars.get_local_ext_algebra(Self::wires_alpha());
        let old_acc = vars.get_local_ext_algebra(Self::wires_old_acc());
        let coeffs = self
            .wires_coeffs()
            .map(|i| vars.local_wires[i])
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut constraints = Vec::with_capacity(<Self as Gate<F, D>>::num_constraints(self));
        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            let coeff = builder.convert_to_ext_algebra(coeffs[i]);
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
        2 * D + self.num_coeffs * (D + 1)
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
    gate: ReducingGate<D>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for ReducingGenerator<D> {
    fn id(&self) -> String {
        "ReducingGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        ReducingGate::<D>::wires_alpha()
            .chain(ReducingGate::<D>::wires_old_acc())
            .chain(self.gate.wires_coeffs())
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

        let alpha = extract_extension(ReducingGate::<D>::wires_alpha());
        let old_acc = extract_extension(ReducingGate::<D>::wires_old_acc());
        let coeffs = witness.get_targets(
            &self
                .gate
                .wires_coeffs()
                .map(|i| Target::wire(self.row, i))
                .collect::<Vec<_>>(),
        );
        let accs = (0..self.gate.num_coeffs)
            .map(|i| ExtensionTarget::from_range(self.row, self.gate.wires_accs(i)))
            .collect::<Vec<_>>();
        let output = ExtensionTarget::from_range(self.row, ReducingGate::<D>::wires_output());

        let mut acc = old_acc;
        for i in 0..self.gate.num_coeffs {
            let computed_acc = acc * alpha + coeffs[i].into();
            out_buffer.set_extension_target(accs[i], computed_acc)?;
            acc = computed_acc;
        }

        out_buffer.set_extension_target(output, acc)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        <ReducingGate<D> as Gate<F, D>>::serialize(&self.gate, dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let gate = <ReducingGate<D> as Gate<F, D>>::deserialize(src, _common_data)?;
        Ok(Self { row, gate })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::reducing::ReducingGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(ReducingGate::new(22));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(ReducingGate::new(22))
    }

    /// Raw-limb oracle for the stack-or-heap reducing accumulate.
    #[test]
    fn accumulate_matches_materialized_raw_limbs() {
        use crate::field::batch_util::batch_multiply_add_inplace;
        use crate::field::types::{Field, Field64, PrimeField64};
        use crate::gates::gate::Gate;
        use crate::hash::hash_types::HashOut;
        use crate::plonk::vars::EvaluationVarsBaseBatch;

        const D: usize = 2;
        type F = GoldilocksField;

        fn value(i: usize) -> F {
            let small = ((i as u64).wrapping_mul(0x9e37_79b9) ^ 0x5a5a_a5a5) & 0xffff;
            if i % 3 == 0 {
                GoldilocksField(GoldilocksField::ORDER + small)
            } else {
                F::from_canonical_u64(small)
            }
        }

        for num_coeffs in [1usize, 4, 8, 22] {
            let gate = ReducingGate::<D>::new(num_coeffs);
            for n in [1usize, 3, 7, 31, 32, 33, 64, 65] {
                let wires = (0..gate.num_wires() * n)
                    .map(|i| value(i + 1))
                    .collect::<Vec<_>>();
                let constants = (0..gate.num_constants() * n)
                    .map(|i| value(i + 10_001))
                    .collect::<Vec<_>>();
                let filters = (0..n)
                    .map(|i| match i % 7 {
                        0 => F::ZERO,
                        1 => GoldilocksField(GoldilocksField::ORDER),
                        _ => value(i + 20_001),
                    })
                    .collect::<Vec<_>>();
                let hash = HashOut::ZERO;
                let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);

                let materialized = gate.eval_unfiltered_base_batch(vars);
                let initial = (0..gate.num_constraints() * n)
                    .map(|i| match i % 11 {
                        0 => F::ZERO,
                        1 => GoldilocksField(GoldilocksField::ORDER),
                        _ => value(i + 30_001),
                    })
                    .collect::<Vec<_>>();
                let mut expected = initial.clone();
                for (acc, constraints) in expected
                    .chunks_exact_mut(n)
                    .zip(materialized.chunks_exact(n))
                {
                    batch_multiply_add_inplace(acc, constraints, &filters);
                }

                let mut actual = initial;
                gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);

                for (i, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
                    assert_eq!(
                        actual.to_noncanonical_u64(),
                        expected.to_noncanonical_u64(),
                        "num_coeffs={num_coeffs}, n={n}, output={i}"
                    );
                }
            }
        }
    }
}
