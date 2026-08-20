#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::packable::Packable;
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

/// A gate which can perform a weighted multiply-add, i.e. `result = c0.x.y + c1.z`. If the config
/// has enough routed wires, it can support several such operations in one gate.
#[derive(Debug, Clone)]
pub struct ArithmeticGate {
    /// Number of arithmetic operations performed by an arithmetic gate.
    pub num_ops: usize,
}

impl ArithmeticGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }

    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let wires_per_op = 4;
        config.num_routed_wires / wires_per_op
    }

    pub(crate) const fn wire_ith_multiplicand_0(i: usize) -> usize {
        4 * i
    }
    pub(crate) const fn wire_ith_multiplicand_1(i: usize) -> usize {
        4 * i + 1
    }
    pub(crate) const fn wire_ith_addend(i: usize) -> usize {
        4 * i + 2
    }
    pub(crate) const fn wire_ith_output(i: usize) -> usize {
        4 * i + 3
    }
}

/// Exact production-shape packed evaluator fused with the quotient accumulator. Generic batch
/// sizes, operation counts, and packing widths retain [`PackedEvaluableBase`]'s scratch path.
///
/// The constraint and outer accumulation expressions deliberately match the generic evaluator
/// operation-for-operation. In particular, neither product is reassociated: packed Goldilocks
/// values can have non-canonical raw words, and the final proof serialization must remain raw-word
/// identical to the established path.
#[inline(always)]
fn eval_arithmetic_packed_direct_n32<F: RichField + Extendable<D>, const D: usize>(
    vars: EvaluationVarsBaseBatch<F>,
    filters: &[F],
    combined_gate_constraints: &mut [F],
) {
    type Packing<T> = <T as Packable>::Packing;
    const N: usize = 32;
    const WIDTH: usize = 4;
    const PACKS_PER_COLUMN: usize = N / WIDTH;
    const NUM_OPS: usize = 20;
    const WIRES_PER_OP: usize = 4;

    debug_assert_eq!(vars.len(), N);
    debug_assert_eq!(Packing::<F>::WIDTH, WIDTH);
    debug_assert_eq!(filters.len(), N);
    debug_assert!(combined_gate_constraints.len() >= NUM_OPS * N);

    // Production's complete four-point groups can be reinterpreted once up front. PackedField's
    // slice contract guarantees the scalar/packed layout; no temporary constraint block is needed.
    let wires = Packing::<F>::pack_slice(&vars.local_wires[..NUM_OPS * WIRES_PER_OP * N]);
    let constants = Packing::<F>::pack_slice(&vars.local_constants[..2 * N]);
    let filters = Packing::<F>::pack_slice(filters);
    let combined = Packing::<F>::pack_slice_mut(&mut combined_gate_constraints[..NUM_OPS * N]);

    for op in 0..NUM_OPS {
        let wire_base = op * WIRES_PER_OP * PACKS_PER_COLUMN;
        let constraint_base = op * PACKS_PER_COLUMN;
        for group in 0..PACKS_PER_COLUMN {
            let multiplicand_0 = wires[wire_base + group];
            let multiplicand_1 = wires[wire_base + PACKS_PER_COLUMN + group];
            let addend = wires[wire_base + 2 * PACKS_PER_COLUMN + group];
            let output = wires[wire_base + 3 * PACKS_PER_COLUMN + group];
            let const_0 = constants[group];
            let const_1 = constants[PACKS_PER_COLUMN + group];

            // Keep exactly the packed evaluator's arithmetic order, followed by exactly the
            // generic fused path's multiply-accumulate order.
            let computed_output = multiplicand_0 * multiplicand_1 * const_0 + addend * const_1;
            let constraint = output - computed_output;
            let index = constraint_base + group;
            combined[index] = combined[index].multiply_accumulate(constraint, filters[group]);
        }
    }
}

#[inline(always)]
fn is_arithmetic_packed_direct_n32_shape<F: Packable, const D: usize>(
    num_ops: usize,
    n: usize,
) -> bool {
    D == 2 && num_ops == 20 && n == 32 && <F::Packing as PackedField>::WIDTH == 4
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ArithmeticGate {
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

        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.local_wires[Self::wire_ith_multiplicand_0(i)];
            let multiplicand_1 = vars.local_wires[Self::wire_ith_multiplicand_1(i)];
            let addend = vars.local_wires[Self::wire_ith_addend(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = multiplicand_0 * multiplicand_1 * const_0 + addend * const_1;

            constraints.push(output - computed_output);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= self.num_ops * n);
        if is_arithmetic_packed_direct_n32_shape::<F, D>(self.num_ops, n) {
            eval_arithmetic_packed_direct_n32::<F, D>(
                vars_base,
                filters,
                combined_gate_constraints,
            );
            return;
        }
        self.eval_unfiltered_base_batch_accumulate_packed(
            vars_base,
            filters,
            combined_gate_constraints,
        );
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.local_wires[Self::wire_ith_multiplicand_0(i)];
            let multiplicand_1 = vars.local_wires[Self::wire_ith_multiplicand_1(i)];
            let addend = vars.local_wires[Self::wire_ith_addend(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = {
                let scaled_mul =
                    builder.mul_many_extension([const_0, multiplicand_0, multiplicand_1]);
                builder.mul_add_extension(const_1, addend, scaled_mul)
            };

            let diff = builder.sub_extension(output, computed_output);
            constraints.push(diff);
        }

        constraints
    }

    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    ArithmeticBaseGenerator {
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
        self.num_ops * 4
    }

    fn num_constants(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        self.num_ops
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for ArithmeticGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        for i in 0..self.num_ops {
            let multiplicand_0 = vars.local_wires[Self::wire_ith_multiplicand_0(i)];
            let multiplicand_1 = vars.local_wires[Self::wire_ith_multiplicand_1(i)];
            let addend = vars.local_wires[Self::wire_ith_addend(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = multiplicand_0 * multiplicand_1 * const_0 + addend * const_1;

            yield_constr.one(output - computed_output);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArithmeticBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    const_0: F,
    const_1: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for ArithmeticBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "ArithmeticBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        [
            ArithmeticGate::wire_ith_multiplicand_0(self.i),
            ArithmeticGate::wire_ith_multiplicand_1(self.i),
            ArithmeticGate::wire_ith_addend(self.i),
        ]
        .iter()
        .map(|&i| Target::wire(self.row, i))
        .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let get_wire = |wire: usize| -> F { witness.get_target(Target::wire(self.row, wire)) };

        let multiplicand_0 = get_wire(ArithmeticGate::wire_ith_multiplicand_0(self.i));
        let multiplicand_1 = get_wire(ArithmeticGate::wire_ith_multiplicand_1(self.i));
        let addend = get_wire(ArithmeticGate::wire_ith_addend(self.i));

        let output_target = Target::wire(self.row, ArithmeticGate::wire_ith_output(self.i));

        let computed_output =
            multiplicand_0 * multiplicand_1 * self.const_0 + addend * self.const_1;

        out_buffer.set_target(output_target, computed_output)
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
        let gate = ArithmeticGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = ArithmeticGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    fn generic_packed_accumulate_reference(
        gate: &ArithmeticGate,
        vars: EvaluationVarsBaseBatch<GoldilocksField>,
        filters: &[GoldilocksField],
        combined: &mut [GoldilocksField],
    ) {
        <ArithmeticGate as PackedEvaluableBase<GoldilocksField, 2>>::
            eval_unfiltered_base_batch_accumulate_packed(gate, vars, filters, combined);
    }

    fn arithmetic_raw_value(i: usize) -> GoldilocksField {
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
        let edge = i % 23;
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

    fn raw_words(values: &[GoldilocksField]) -> Vec<u64> {
        values.iter().map(|x| x.to_noncanonical_u64()).collect()
    }

    /// The exact production shape takes the direct path on four-lane targets. Every adjacent
    /// operation count and batch size remains on the generic packed fallback.
    #[test]
    fn arithmetic_direct_shape_is_exact() {
        type F = GoldilocksField;
        let width_is_four = <<F as Packable>::Packing as PackedField>::WIDTH == 4;
        assert_eq!(
            is_arithmetic_packed_direct_n32_shape::<F, 2>(20, 32),
            width_is_four
        );
        for (num_ops, n) in [(19, 32), (21, 32), (20, 31), (20, 33)] {
            assert!(!is_arithmetic_packed_direct_n32_shape::<F, 2>(num_ops, n));
        }
        assert!(!is_arithmetic_packed_direct_n32_shape::<F, 4>(20, 32));
    }

    /// Compare against the pre-specialization packed scratch implementation, including raw field
    /// words. The adjacent shapes exercise fallback dispatch; n=32/20 ops exercises the direct
    /// production path on the ranked four-lane target.
    #[test]
    fn arithmetic_direct_and_fallback_match_generic_packed_raw() {
        type F = GoldilocksField;
        for num_ops in [19usize, 20, 21] {
            let gate = ArithmeticGate { num_ops };
            for n in [31usize, 32, 33] {
                let wires = (0..4 * num_ops * n)
                    .map(|i| arithmetic_raw_value(i + 0x1000 + 29 * num_ops + n))
                    .collect::<Vec<_>>();
                let constants = (0..2 * n)
                    .map(|i| arithmetic_raw_value(5 * i + 0x2000 + 31 * num_ops + n))
                    .collect::<Vec<_>>();
                let filters = (0..n)
                    .map(|i| arithmetic_raw_value(7 * i + 0x3000 + 37 * num_ops + n))
                    .collect::<Vec<_>>();
                let initial = (0..num_ops * n)
                    .map(|i| arithmetic_raw_value(13 * i + 0x4000 + 41 * num_ops + n))
                    .collect::<Vec<_>>();
                let hash = HashOut {
                    elements: core::array::from_fn(|i| {
                        arithmetic_raw_value(0x5000 + 17 * i + 43 * num_ops + n)
                    }),
                };
                let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);
                let mut reference = initial.clone();
                generic_packed_accumulate_reference(&gate, vars, &filters, &mut reference);
                let mut candidate = initial;
                <ArithmeticGate as Gate<F, 2>>::eval_unfiltered_base_batch_accumulate(
                    &gate,
                    vars,
                    &filters,
                    &mut candidate,
                );
                assert_eq!(
                    candidate, reference,
                    "canonical mismatch at num_ops={num_ops}, n={n}"
                );
                assert_eq!(
                    raw_words(&candidate),
                    raw_words(&reference),
                    "raw mismatch at num_ops={num_ops}, n={n}"
                );
            }
        }
    }

    /// Focused manual timing for the exact quotient batch used by production. Run with:
    /// `cargo test --release -p plonky2 arithmetic_accumulate_micro -- --ignored --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn arithmetic_accumulate_microbenchmark() {
        use core::hint::black_box;
        use std::time::Instant;

        use crate::field::types::Sample;

        type F = GoldilocksField;
        const N: usize = 32;
        const ITERS: usize = 100_000;
        let gate = ArithmeticGate { num_ops: 20 };
        let wires = F::rand_vec(<ArithmeticGate as Gate<F, 2>>::num_wires(&gate) * N);
        let constants = F::rand_vec(<ArithmeticGate as Gate<F, 2>>::num_constants(&gate) * N);
        let filters = F::rand_vec(N);
        let hash = HashOut::ZERO;
        let vars = EvaluationVarsBaseBatch::new(N, &constants, &wires, &hash);
        let initial = F::rand_vec(<ArithmeticGate as Gate<F, 2>>::num_constraints(&gate) * N);

        let time_generic = |combined: &mut [F]| {
            let start = Instant::now();
            for _ in 0..ITERS {
                generic_packed_accumulate_reference(
                    &gate,
                    vars,
                    &filters,
                    black_box(&mut *combined),
                );
            }
            start.elapsed()
        };
        let time_direct = |combined: &mut [F]| {
            let start = Instant::now();
            for _ in 0..ITERS {
                <ArithmeticGate as Gate<F, 2>>::eval_unfiltered_base_batch_accumulate(
                    &gate,
                    vars,
                    &filters,
                    black_box(&mut *combined),
                );
            }
            start.elapsed()
        };

        // Alternate order to make host drift visible rather than consistently favoring one path.
        let mut generic_seconds = 0.0;
        let mut direct_seconds = 0.0;
        for round in 0..6 {
            let mut generic = initial.clone();
            let mut direct = initial.clone();
            let (tg, td) = if round % 2 == 0 {
                (time_generic(&mut generic), time_direct(&mut direct))
            } else {
                let td = time_direct(&mut direct);
                let tg = time_generic(&mut generic);
                (tg, td)
            };
            assert_eq!(raw_words(&direct), raw_words(&generic));
            generic_seconds += tg.as_secs_f64();
            direct_seconds += td.as_secs_f64();
            println!(
                "round {round}: generic {:.3} us, direct {:.3} us, {:.3}x",
                tg.as_secs_f64() * 1e6 / ITERS as f64,
                td.as_secs_f64() * 1e6 / ITERS as f64,
                tg.as_secs_f64() / td.as_secs_f64(),
            );
        }
        println!(
            "arithmetic n=32/ops=20 accumulate: generic {:.3} us, direct {:.3} us, {:.3}x",
            generic_seconds * 1e6 / (6 * ITERS) as f64,
            direct_seconds * 1e6 / (6 * ITERS) as f64,
            generic_seconds / direct_seconds,
        );
    }
}
