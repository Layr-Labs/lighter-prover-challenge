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

/// A gate which can perform a weighted multiply-add, i.e. `result = c0.x.y + c1.z`. If the config
/// has enough routed wires, it can support several such operations in one gate.
#[derive(Debug, Clone)]
pub struct ArithmeticExtensionGate<const D: usize> {
    /// Number of arithmetic operations performed by an arithmetic gate.
    pub num_ops: usize,
}

impl<const D: usize> ArithmeticExtensionGate<D> {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }

    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let wires_per_op = 4 * D;
        config.num_routed_wires / wires_per_op
    }

    pub(crate) const fn wires_ith_multiplicand_0(i: usize) -> Range<usize> {
        4 * D * i..4 * D * i + D
    }
    pub(crate) const fn wires_ith_multiplicand_1(i: usize) -> Range<usize> {
        4 * D * i + D..4 * D * i + 2 * D
    }
    pub(crate) const fn wires_ith_addend(i: usize) -> Range<usize> {
        4 * D * i + 2 * D..4 * D * i + 3 * D
    }
    pub(crate) const fn wires_ith_output(i: usize) -> Range<usize> {
        4 * D * i + 3 * D..4 * D * i + 4 * D
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ArithmeticExtensionGate<D> {
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
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops * D);
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.get_local_ext_algebra(Self::wires_ith_multiplicand_0(i));
            let multiplicand_1 = vars.get_local_ext_algebra(Self::wires_ith_multiplicand_1(i));
            let addend = vars.get_local_ext_algebra(Self::wires_ith_addend(i));
            let output = vars.get_local_ext_algebra(Self::wires_ith_output(i));
            let computed_output =
                (multiplicand_0 * multiplicand_1).scalar_mul(const_0) + addend.scalar_mul(const_1);

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
        let const_1 = vars.local_constants[1];

        for i in 0..self.num_ops {
            let multiplicand_0 = vars.get_local_ext(Self::wires_ith_multiplicand_0(i));
            let multiplicand_1 = vars.get_local_ext(Self::wires_ith_multiplicand_1(i));
            let addend = vars.get_local_ext(Self::wires_ith_addend(i));
            let output = vars.get_local_ext(Self::wires_ith_output(i));
            let computed_output =
                (multiplicand_0 * multiplicand_1).scalar_mul(const_0) + addend.scalar_mul(const_1);

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

        // Goldilocks/D=2 fast path: a fused u160 kernel that folds the whole
        // filtered constraint
        // `combined += filter*out - (filter*const_0)*prod - (filter*const_1)*addend`
        // into delayed-reduction accumulators, reducing once per coefficient
        // per op instead of after every intermediate multiply. Field-exact
        // (not representative-exact) versus the generic path below; see
        // `ext2_scaled_mul_add_constraint_accumulate` for the equivalence and
        // overflow-bound arguments, and the differential test
        // `fused_goldilocks_accumulate_matches_generic` in this file.
        if D == 2
            && core::any::TypeId::of::<F>()
                == core::any::TypeId::of::<crate::field::goldilocks_field::GoldilocksField>()
        {
            use crate::field::goldilocks_extensions::ext2_scaled_mul_add_constraint_accumulate;
            use crate::field::goldilocks_field::GoldilocksField;
            use crate::field::types::Field;
            // SAFETY: the `TypeId` compare proves `F` is exactly
            // `GoldilocksField`, so these are identity casts. Same idiom as
            // the specializations in `field/src/fft.rs`.
            let cast = |s: &[F]| -> &[GoldilocksField] {
                unsafe { &*(s as *const [F] as *const [GoldilocksField]) }
            };
            let wires_gl = cast(vars_base.local_wires);
            let filters_gl = cast(filters);
            let const_0_gl = &cast(vars_base.local_constants)[..n];
            let const_1_gl = &cast(vars_base.local_constants)[n..2 * n];
            // SAFETY: identity cast as above, unique borrow passed through.
            let combined_gl: &mut [GoldilocksField] = unsafe {
                &mut *(combined_gate_constraints as *mut [F] as *mut [GoldilocksField])
            };

            // `-filter*const_0` and `-filter*const_1` per point, shared by
            // every op in this gate. Stack-or-heap idiom matching the generic
            // path's scratch below.
            const NFC_STACK: usize = 128;
            let mut nfc_stack = [GoldilocksField::ZERO; 2 * NFC_STACK];
            let mut nfc_heap;
            let nfc: &mut [GoldilocksField] = if n <= NFC_STACK {
                &mut nfc_stack[..2 * n]
            } else {
                nfc_heap = vec![GoldilocksField::ZERO; 2 * n];
                &mut nfc_heap
            };
            let (nfc0, nfc1) = nfc.split_at_mut(n);
            for p in 0..n {
                nfc0[p] = -(filters_gl[p] * const_0_gl[p]);
                nfc1[p] = -(filters_gl[p] * const_1_gl[p]);
            }

            for i in 0..self.num_ops {
                let m0_start = Self::wires_ith_multiplicand_0(i).start;
                let m1_start = Self::wires_ith_multiplicand_1(i).start;
                let addend_start = Self::wires_ith_addend(i).start;
                let output_start = Self::wires_ith_output(i).start;
                let col = |w: usize| &wires_gl[w * n..][..n];
                let (c0, c1) = combined_gl[(i * D) * n..][..2 * n].split_at_mut(n);
                ext2_scaled_mul_add_constraint_accumulate(
                    (col(m0_start), col(m0_start + 1)),
                    (col(m1_start), col(m1_start + 1)),
                    (col(addend_start), col(addend_start + 1)),
                    (col(output_start), col(output_start + 1)),
                    filters_gl,
                    nfc0,
                    nfc1,
                    (c0, c1),
                );
            }
            return;
        }

        self.eval_accumulate_generic(vars_base, filters, combined_gate_constraints);
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops * D);
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.get_local_ext_algebra(Self::wires_ith_multiplicand_0(i));
            let multiplicand_1 = vars.get_local_ext_algebra(Self::wires_ith_multiplicand_1(i));
            let addend = vars.get_local_ext_algebra(Self::wires_ith_addend(i));
            let output = vars.get_local_ext_algebra(Self::wires_ith_output(i));
            let computed_output = {
                let mul = builder.mul_ext_algebra(multiplicand_0, multiplicand_1);
                let scaled_mul = builder.scalar_mul_ext_algebra(const_0, mul);
                builder.scalar_mul_add_ext_algebra(const_1, addend, scaled_mul)
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
                    ArithmeticExtensionGenerator {
                        row,
                        const_0: local_constants[0],
                        const_1: local_constants[1],
                        i,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * 4 * D
    }

    fn num_constants(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * D
    }
}

impl<const D: usize> ArithmeticExtensionGate<D> {
    /// Field-generic body of [`Gate::eval_unfiltered_base_batch_accumulate`].
    /// Kept as a separate method so the Goldilocks fused fast path above can
    /// be differentially tested against it.
    pub(crate) fn eval_accumulate_generic<F: RichField + Extendable<D>>(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        let wires = vars_base.local_wires;
        let constants = vars_base.local_constants;
        let const_0 = &constants[..n];
        let const_1 = &constants[n..2 * n];
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
            let addend_start = Self::wires_ith_addend(i).start;
            let output_start = Self::wires_ith_output(i).start;
            for p in 0..n {
                let multiplicand_0 = ext(m0_start, p);
                let multiplicand_1 = ext(m1_start, p);
                let addend = ext(addend_start, p);
                let output = ext(output_start, p);
                let computed_output = (multiplicand_0 * multiplicand_1).scalar_mul(const_0[p])
                    + addend.scalar_mul(const_1[p]);
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

#[derive(Clone, Debug, Default)]
pub struct ArithmeticExtensionGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    const_0: F,
    const_1: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for ArithmeticExtensionGenerator<F, D>
{
    fn id(&self) -> String {
        "ArithmeticExtensionGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        ArithmeticExtensionGate::<D>::wires_ith_multiplicand_0(self.i)
            .chain(ArithmeticExtensionGate::<D>::wires_ith_multiplicand_1(
                self.i,
            ))
            .chain(ArithmeticExtensionGate::<D>::wires_ith_addend(self.i))
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

        let multiplicand_0 = extract_extension(
            ArithmeticExtensionGate::<D>::wires_ith_multiplicand_0(self.i),
        );
        let multiplicand_1 = extract_extension(
            ArithmeticExtensionGate::<D>::wires_ith_multiplicand_1(self.i),
        );
        let addend = extract_extension(ArithmeticExtensionGate::<D>::wires_ith_addend(self.i));

        let output_target = ExtensionTarget::from_range(
            self.row,
            ArithmeticExtensionGate::<D>::wires_ith_output(self.i),
        );

        let computed_output = (multiplicand_0 * multiplicand_1).scalar_mul(self.const_0)
            + addend.scalar_mul(self.const_1);

        out_buffer.set_extension_target(output_target, computed_output)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_field(self.const_0)?;
        dst.write_field(self.const_1)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let const_0 = src.read_field()?;
        let const_1 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            row,
            const_0,
            const_1,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate =
            ArithmeticExtensionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate =
            ArithmeticExtensionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    /// Differential test: the fused Goldilocks u160 accumulate path must be
    /// field-equal (canonical-value equal) to the generic scratch+
    /// `batch_multiply_add_inplace` path per constraint per row, for random
    /// witnesses, filters, constants, initial buffer contents, and a sprinkle
    /// of noncanonical (`>= p`) raw wire representatives.
    #[test]
    fn fused_goldilocks_accumulate_matches_generic() {
        use plonky2_field::types::{Field64, PrimeField64, Sample};

        use crate::gates::gate::Gate;
        use crate::hash::hash_types::HashOut;
        use crate::plonk::vars::EvaluationVarsBaseBatch;

        const D: usize = 2;
        type F = GoldilocksField;

        // 10 ops is the shape used by the recursion circuits.
        for num_ops in [1usize, 2, 10] {
            let gate = ArithmeticExtensionGate::<D> { num_ops };
            let num_wires = <ArithmeticExtensionGate<D> as Gate<F, D>>::num_wires(&gate);
            let num_constraints =
                <ArithmeticExtensionGate<D> as Gate<F, D>>::num_constraints(&gate);

            for &n in &[1usize, 3, 4, 5, 7, 31, 32, 33] {
                let mut wires: Vec<F> = (0..num_wires * n).map(|_| F::rand()).collect();
                // Noncanonical representatives are legal inputs on this path;
                // exercise them explicitly (p <= raw < 2^64).
                for (k, w) in wires.iter_mut().enumerate().step_by(5) {
                    *w = GoldilocksField(F::ORDER.wrapping_add(k as u64));
                }
                let constants: Vec<F> = (0..2 * n).map(|_| F::rand()).collect();
                let filters: Vec<F> = (0..n).map(|_| F::rand()).collect();
                let hash = HashOut::<F>::ZERO;
                let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);

                let init: Vec<F> = (0..num_constraints * n).map(|_| F::rand()).collect();
                let mut fused = init.clone();
                let mut generic = init;
                <ArithmeticExtensionGate<D> as Gate<F, D>>::eval_unfiltered_base_batch_accumulate(
                    &gate,
                    vars,
                    &filters,
                    &mut fused,
                );
                gate.eval_accumulate_generic(vars, &filters, &mut generic);

                for k in 0..num_constraints * n {
                    assert_eq!(
                        fused[k].to_canonical_u64(),
                        generic[k].to_canonical_u64(),
                        "mismatch at num_ops={num_ops}, n={n}, flat index {k} \
                         (constraint {}, point {})",
                        k / n,
                        k % n,
                    );
                }
            }
        }
    }

    /// Microbenchmark: fused u160 path vs generic path, one 32-point batch of
    /// the production 10-op shape. Run with:
    /// `cargo test --release --lib arithmetic_extension_fused_accumulate_microbenchmark -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn arithmetic_extension_fused_accumulate_microbenchmark() {
        use core::hint::black_box;
        use std::time::Instant;

        use plonky2_field::types::{Field, Sample};

        use crate::gates::gate::Gate;
        use crate::hash::hash_types::HashOut;
        use crate::plonk::vars::EvaluationVarsBaseBatch;

        const D: usize = 2;
        type F = GoldilocksField;

        let gate = ArithmeticExtensionGate::<D> { num_ops: 10 };
        let num_wires = <ArithmeticExtensionGate<D> as Gate<F, D>>::num_wires(&gate);
        let num_constraints = <ArithmeticExtensionGate<D> as Gate<F, D>>::num_constraints(&gate);
        let n = 32usize;

        let wires: Vec<F> = (0..num_wires * n).map(|_| F::rand()).collect();
        let constants: Vec<F> = (0..2 * n).map(|_| F::rand()).collect();
        let filters: Vec<F> = (0..n).map(|_| F::rand()).collect();
        let hash = HashOut::<F>::ZERO;
        let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);
        let mut buf = vec![F::ZERO; num_constraints * n];

        const WARMUP: usize = 2_000;
        const ITERS: usize = 200_000;

        for _ in 0..WARMUP {
            <ArithmeticExtensionGate<D> as Gate<F, D>>::eval_unfiltered_base_batch_accumulate(
                &gate,
                vars,
                &filters,
                black_box(&mut buf),
            );
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            <ArithmeticExtensionGate<D> as Gate<F, D>>::eval_unfiltered_base_batch_accumulate(
                &gate,
                vars,
                &filters,
                black_box(&mut buf),
            );
        }
        let fused = start.elapsed();

        for _ in 0..WARMUP {
            gate.eval_accumulate_generic(vars, &filters, black_box(&mut buf));
        }
        let start = Instant::now();
        for _ in 0..ITERS {
            gate.eval_accumulate_generic(vars, &filters, black_box(&mut buf));
        }
        let generic = start.elapsed();

        println!(
            "ArithmeticExtensionGate accumulate, {} ops x {n} points, {ITERS} iters:",
            gate.num_ops
        );
        println!(
            "  fused u160 path:  {:?} total, {:.3} us/batch",
            fused,
            fused.as_secs_f64() * 1e6 / ITERS as f64
        );
        println!(
            "  generic path:     {:?} total, {:.3} us/batch",
            generic,
            generic.as_secs_f64() * 1e6 / ITERS as f64
        );
    }
}
