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

#[inline(always)]
fn uses_quadratic_packed_direct_n32<F: RichField + Extendable<D>, const D: usize>(
    num_ops: usize,
    n: usize,
) -> bool {
    D == 2 && num_ops == 10 && n == 32 && <<F as Packable>::Packing as PackedField>::WIDTH == 4
}

/// Exact production D=2/num_ops=10/n=32 packed coefficient accumulator. All other dimensions,
/// operation counts, batch sizes, packing widths, and scalar tails retain the established
/// extension-field path below.
fn eval_quadratic_packed_direct_n32<F: RichField + Extendable<D>, const D: usize>(
    vars: EvaluationVarsBaseBatch<F>,
    filters: &[F],
    combined_gate_constraints: &mut [F],
) {
    type Packing<T> = <T as Packable>::Packing;
    const N: usize = 32;
    const WIDTH: usize = 4;
    const PACKS_PER_COLUMN: usize = N / WIDTH;
    const NUM_OPS: usize = 10;
    const WIRES_PER_OP: usize = 8;
    const CONSTRAINTS_PER_OP: usize = 2;

    debug_assert_eq!(D, 2);
    debug_assert_eq!(vars.len(), N);
    debug_assert_eq!(Packing::<F>::WIDTH, WIDTH);
    debug_assert_eq!(filters.len(), N);
    debug_assert!(combined_gate_constraints.len() >= NUM_OPS * CONSTRAINTS_PER_OP * N);

    // Reinterpret each complete four-point lane group once. Exact prefixes give LLVM fixed
    // bounds for every indexed load/store below and `PackedField` guarantees the scalar/packed
    // layout; this is the AArch64 `WideGoldilocksField` path in production.
    let wires = Packing::<F>::pack_slice(&vars.local_wires[..NUM_OPS * WIRES_PER_OP * N]);
    let const_0 = Packing::<F>::pack_slice(&vars.local_constants[..N]);
    let const_1 = Packing::<F>::pack_slice(&vars.local_constants[N..2 * N]);
    let filters = Packing::<F>::pack_slice(filters);
    let combined = Packing::<F>::pack_slice_mut(
        &mut combined_gate_constraints[..NUM_OPS * CONSTRAINTS_PER_OP * N],
    );
    let w = Packing::<F>::from(<F as Extendable<D>>::W);

    for op in 0..NUM_OPS {
        // With D=2, each operation occupies [a0,a1,b0,b1,add0,add1,out0,out1].
        let m0 = WIRES_PER_OP * PACKS_PER_COLUMN * op;
        for group in 0..PACKS_PER_COLUMN {
            let a0 = wires[m0 + group];
            let a1 = wires[m0 + PACKS_PER_COLUMN + group];
            let b0 = wires[m0 + 2 * PACKS_PER_COLUMN + group];
            let b1 = wires[m0 + 3 * PACKS_PER_COLUMN + group];
            let add0 = wires[m0 + 4 * PACKS_PER_COLUMN + group];
            let add1 = wires[m0 + 5 * PACKS_PER_COLUMN + group];
            let out0 = wires[m0 + 6 * PACKS_PER_COLUMN + group];
            let out1 = wires[m0 + 7 * PACKS_PER_COLUMN + group];

            // Match the raw-safe packed coefficient sequence already used by
            // `MulExtensionGate`: the first product remains the accumulator and the second
            // uses the packed multiply-accumulate primitive.
            let prod0 = (a0 * b0).multiply_accumulate(w * a1, b1);
            let prod1 = (a0 * b1).multiply_accumulate(a1, b0);

            // Preserve the established operation order exactly. Fusing either sum can change a
            // valid noncanonical Goldilocks representative even though the field value agrees.
            let computed0 = prod0 * const_0[group] + add0 * const_1[group];
            let computed1 = prod1 * const_0[group] + add1 * const_1[group];
            let constraint0 = out0 - computed0;
            let constraint1 = out1 - computed1;

            let row0 = (CONSTRAINTS_PER_OP * op) * PACKS_PER_COLUMN + group;
            combined[row0] = combined[row0].multiply_accumulate(constraint0, filters[group]);
            let row1 = row0 + PACKS_PER_COLUMN;
            combined[row1] = combined[row1].multiply_accumulate(constraint1, filters[group]);
        }
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

        if uses_quadratic_packed_direct_n32::<F, D>(self.num_ops, n) {
            eval_quadratic_packed_direct_n32::<F, D>(vars_base, filters, combined_gate_constraints);
            return;
        }

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

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, Field64, PrimeField64};
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::HashOut;
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

    #[test]
    fn low_degree_d2() {
        let gate = ArithmeticExtensionGate::<2>::new_from_config(
            &CircuitConfig::standard_recursion_config(),
        );
        test_low_degree::<GoldilocksField, _, 2>(gate);
    }

    fn scalar_extension_accumulate_reference<const D: usize>(
        gate: &ArithmeticExtensionGate<D>,
        vars: EvaluationVarsBaseBatch<GoldilocksField>,
        filters: &[GoldilocksField],
        combined: &mut [GoldilocksField],
    ) where
        GoldilocksField: Extendable<D>,
    {
        type F = GoldilocksField;
        let n = vars.len();
        let wires = vars.local_wires;
        let const_0 = &vars.local_constants[..n];
        let const_1 = &vars.local_constants[n..2 * n];
        let ext = |start: usize, point: usize| {
            let mut coefficients = [F::ZERO; D];
            for (d, coefficient) in coefficients.iter_mut().enumerate() {
                *coefficient = wires[(start + d) * n + point];
            }
            <F as Extendable<D>>::Extension::from_basefield_array(coefficients)
        };
        let mut scratch = vec![F::ZERO; D * n];
        for op in 0..gate.num_ops {
            let m0 = ArithmeticExtensionGate::<D>::wires_ith_multiplicand_0(op).start;
            let m1 = ArithmeticExtensionGate::<D>::wires_ith_multiplicand_1(op).start;
            let addend = ArithmeticExtensionGate::<D>::wires_ith_addend(op).start;
            let output = ArithmeticExtensionGate::<D>::wires_ith_output(op).start;
            for point in 0..n {
                let product = ext(m0, point) * ext(m1, point);
                let computed = product.scalar_mul(const_0[point])
                    + ext(addend, point).scalar_mul(const_1[point]);
                let coefficients = (ext(output, point) - computed).to_basefield_array();
                for (d, coefficient) in coefficients.iter().enumerate() {
                    scratch[d * n + point] = *coefficient;
                }
            }
            for d in 0..D {
                batch_multiply_add_inplace(
                    &mut combined[(D * op + d) * n..][..n],
                    &scratch[d * n..][..n],
                    filters,
                );
            }
        }
    }

    fn arithmetic_raw_value(i: usize) -> GoldilocksField {
        type F = GoldilocksField;
        const EDGES: [u64; 11] = [
            0,
            1,
            2,
            3,
            (1u64 << 32) - 1,
            1u64 << 32,
            F::ORDER - 1,
            F::ORDER,
            F::ORDER + 1,
            u64::MAX - 1,
            u64::MAX,
        ];
        let edge = i % 29;
        let raw = if edge < EDGES.len() {
            EDGES[edge]
        } else {
            let mut x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            x ^= x >> 29;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x ^ (x >> 31)
        };
        F::from_noncanonical_u64(raw)
    }

    fn nonzero_arithmetic_raw_value(i: usize) -> GoldilocksField {
        type F = GoldilocksField;
        let value = arithmetic_raw_value(i);
        if value == F::ZERO {
            F::from_noncanonical_u64(F::ORDER + 1)
        } else {
            value
        }
    }

    fn raw_words(values: &[GoldilocksField]) -> Vec<u64> {
        values.iter().map(|x| x.to_noncanonical_u64()).collect()
    }

    fn check_arithmetic_accumulate_shapes<const D: usize>()
    where
        GoldilocksField: Extendable<D>,
    {
        type F = GoldilocksField;
        let width = <<F as Packable>::Packing as PackedField>::WIDTH;

        for num_ops in [9usize, 10, 11] {
            let gate = ArithmeticExtensionGate::<D> { num_ops };
            for n in [31usize, 32, 33] {
                let production_shape = D == 2 && num_ops == 10 && n == 32 && width == 4;
                assert_eq!(
                    uses_quadratic_packed_direct_n32::<F, D>(num_ops, n),
                    production_shape,
                    "incorrect dispatch at D={D}, num_ops={num_ops}, n={n}"
                );

                let wires = (0..4 * D * num_ops * n)
                    .map(|i| arithmetic_raw_value(i + 0x1000 + 31 * num_ops + 3 * n + D))
                    .collect::<Vec<_>>();
                let constants = (0..2 * n)
                    .map(|i| arithmetic_raw_value(5 * i + 0x2000 + 37 * num_ops + 7 * n + D))
                    .collect::<Vec<_>>();
                let filters = (0..n)
                    .map(|i| arithmetic_raw_value(7 * i + 0x3000 + 41 * num_ops + 11 * n + D))
                    .collect::<Vec<_>>();
                // Exercise a true accumulate rather than the easier zero-destination case, and
                // retain a guard suffix to detect writes past the advertised constraint rows.
                let initial = (0..D * num_ops * n + 8)
                    .map(|i| {
                        nonzero_arithmetic_raw_value(13 * i + 0x4000 + 43 * num_ops + 17 * n + D)
                    })
                    .collect::<Vec<_>>();
                assert!(initial.iter().all(|x| *x != F::ZERO));
                let hash = HashOut {
                    elements: core::array::from_fn(|i| {
                        arithmetic_raw_value(0x5000 + 19 * i + 47 * num_ops + 23 * n + D)
                    }),
                };
                let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);
                let mut reference = initial.clone();
                scalar_extension_accumulate_reference(&gate, vars, &filters, &mut reference);
                let mut candidate = initial;
                gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut candidate);
                assert_eq!(
                    candidate, reference,
                    "canonical mismatch at D={D}, num_ops={num_ops}, n={n}"
                );
                assert_eq!(
                    raw_words(&candidate),
                    raw_words(&reference),
                    "raw mismatch at D={D}, num_ops={num_ops}, n={n}"
                );
            }
        }
    }

    /// The exact production shape takes the packed direct accumulator. Every adjacent operation
    /// count and batch size, plus D=4, exercises the byte-for-byte generic fallback. Both field
    /// values and raw Goldilocks representatives must match the independent extension oracle.
    #[test]
    fn quadratic_packed_accumulate_matches_extension_oracle_raw_and_canonical() {
        check_arithmetic_accumulate_shapes::<2>();
        check_arithmetic_accumulate_shapes::<4>();
    }

    /// Sabotage control for the independent oracle: reversing `const_0` and `const_1` on the exact
    /// production shape must be detected with the same adversarial, nonzero-accumulator data.
    #[test]
    fn quadratic_packed_accumulate_oracle_detects_reversed_constants() {
        const D: usize = 2;
        const NUM_OPS: usize = 10;
        const N: usize = 32;
        let gate = ArithmeticExtensionGate::<D> { num_ops: NUM_OPS };
        let wires = (0..4 * D * NUM_OPS * N)
            .map(|i| arithmetic_raw_value(i + 0x6100))
            .collect::<Vec<_>>();
        let constants = (0..2 * N)
            .map(|i| arithmetic_raw_value(5 * i + 0x6200))
            .collect::<Vec<_>>();
        let mut reversed_constants = constants[N..].to_vec();
        reversed_constants.extend_from_slice(&constants[..N]);
        let filters = (0..N)
            .map(|i| arithmetic_raw_value(7 * i + 0x6300))
            .collect::<Vec<_>>();
        let initial = (0..D * NUM_OPS * N)
            .map(|i| nonzero_arithmetic_raw_value(11 * i + 0x6400))
            .collect::<Vec<_>>();
        assert!(initial.iter().all(|x| *x != GoldilocksField::ZERO));
        let hash = HashOut {
            elements: core::array::from_fn(|i| arithmetic_raw_value(13 * i + 0x6500)),
        };
        let honest_vars = EvaluationVarsBaseBatch::new(N, &constants, &wires, &hash);
        let reversed_vars = EvaluationVarsBaseBatch::new(N, &reversed_constants, &wires, &hash);
        let mut honest = initial.clone();
        scalar_extension_accumulate_reference(&gate, honest_vars, &filters, &mut honest);
        let mut sabotaged = initial;
        scalar_extension_accumulate_reference(&gate, reversed_vars, &filters, &mut sabotaged);
        assert_ne!(
            sabotaged, honest,
            "canonical oracle accepted reversed constants"
        );
        assert_ne!(
            raw_words(&sabotaged),
            raw_words(&honest),
            "raw oracle accepted reversed constants"
        );
    }
}
