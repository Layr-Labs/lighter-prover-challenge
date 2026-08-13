#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::ops::Square;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate for raising a value to a power.
#[derive(Clone, Debug, Default)]
pub struct ExponentiationGate<F: RichField + Extendable<D>, const D: usize> {
    pub num_power_bits: usize,
    pub _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> ExponentiationGate<F, D> {
    pub const fn new(num_power_bits: usize) -> Self {
        Self {
            num_power_bits,
            _phantom: PhantomData,
        }
    }

    pub fn new_from_config(config: &CircuitConfig) -> Self {
        let num_power_bits = Self::max_power_bits(config.num_wires, config.num_routed_wires);
        Self::new(num_power_bits)
    }

    fn max_power_bits(num_wires: usize, num_routed_wires: usize) -> usize {
        // 2 wires are reserved for the base and output.
        let max_for_routed_wires = num_routed_wires - 2;
        let max_for_wires = (num_wires - 2) / 2;
        max_for_routed_wires.min(max_for_wires)
    }

    pub(crate) const fn wire_base(&self) -> usize {
        0
    }

    /// The `i`th bit of the exponent, in little-endian order.
    pub(crate) const fn wire_power_bit(&self, i: usize) -> usize {
        debug_assert!(i < self.num_power_bits);
        1 + i
    }

    pub const fn wire_output(&self) -> usize {
        1 + self.num_power_bits
    }

    pub(crate) const fn wire_intermediate_value(&self, i: usize) -> usize {
        debug_assert!(i < self.num_power_bits);
        2 + self.num_power_bits + i
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ExponentiationGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<D={D}>")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_power_bits)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_power_bits = src.read_usize()?;
        Ok(Self::new(num_power_bits))
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let base = vars.local_wires[self.wire_base()];

        let power_bits: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_power_bit(i)])
            .collect();
        let intermediate_values: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_intermediate_value(i)])
            .collect();

        let output = vars.local_wires[self.wire_output()];

        let mut constraints = Vec::with_capacity(self.num_constraints());

        for i in 0..self.num_power_bits {
            let prev_intermediate_value = if i == 0 {
                F::Extension::ONE
            } else {
                intermediate_values[i - 1].square()
            };

            // power_bits is in LE order, but we accumulate in BE order.
            let cur_bit = power_bits[self.num_power_bits - i - 1];

            let not_cur_bit = F::Extension::ONE - cur_bit;
            let computed_intermediate_value =
                prev_intermediate_value * (cur_bit * base + not_cur_bit);
            constraints.push(computed_intermediate_value - intermediate_values[i]);
        }

        constraints.push(output - intermediate_values[self.num_power_bits - 1]);

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
        assert!(combined_gate_constraints.len() >= self.num_constraints() * n);

        // This gate has 68 constraints in production and is deliberately kept
        // on the CPU. The generic packed accumulator materializes every packed
        // constraint into a temporary 68 x WIDTH matrix, then immediately
        // reads that matrix back to multiply by the selector filter. Accumulate
        // each constraint directly instead: the packed arithmetic and emission
        // order are identical, but the scratch write/read pass disappears.
        let width = <<F as Packable>::Packing as PackedField>::WIDTH;
        let (vars_packed_iter, vars_leftovers_iter) =
            vars_base.pack::<<F as Packable>::Packing>();
        let leftovers_start = n - vars_leftovers_iter.len();

        for (group, vars) in vars_packed_iter.enumerate() {
            type P<F> = <F as Packable>::Packing;
            let offset = group * width;
            let mut filter = P::<F>::ZEROS;
            filter
                .as_slice_mut()
                .copy_from_slice(&filters[offset..offset + width]);

            let base = vars.local_wires[self.wire_base()];
            let power_bits = vars
                .local_wires
                .view(self.wire_power_bit(0)..self.wire_power_bit(0) + self.num_power_bits);
            let intermediate_values = vars.local_wires.view(
                self.wire_intermediate_value(0)
                    ..self.wire_intermediate_value(0) + self.num_power_bits,
            );
            let base_minus_one = base - P::<F>::ONES;

            for i in 0..self.num_power_bits {
                let prev = if i == 0 {
                    P::<F>::ONES
                } else {
                    intermediate_values[i - 1].square()
                };
                let bit = power_bits[self.num_power_bits - i - 1];
                let mul_by = P::<F>::ONES.multiply_accumulate(bit, base_minus_one);
                let constraint = prev * mul_by - intermediate_values[i];
                let combined = &mut combined_gate_constraints[i * n + offset..][..width];
                let mut acc = P::<F>::ZEROS;
                acc.as_slice_mut().copy_from_slice(combined);
                combined.copy_from_slice(acc.multiply_accumulate(constraint, filter).as_slice());
            }

            let constraint =
                vars.local_wires[self.wire_output()] - intermediate_values[self.num_power_bits - 1];
            let combined =
                &mut combined_gate_constraints[self.num_power_bits * n + offset..][..width];
            let mut acc = P::<F>::ZEROS;
            acc.as_slice_mut().copy_from_slice(combined);
            combined.copy_from_slice(acc.multiply_accumulate(constraint, filter).as_slice());
        }

        for (lane, vars) in vars_leftovers_iter.enumerate() {
            let point = leftovers_start + lane;
            let filter = filters[point];
            let base = vars.local_wires[self.wire_base()];
            let power_bits = vars
                .local_wires
                .view(self.wire_power_bit(0)..self.wire_power_bit(0) + self.num_power_bits);
            let intermediate_values = vars.local_wires.view(
                self.wire_intermediate_value(0)
                    ..self.wire_intermediate_value(0) + self.num_power_bits,
            );
            let base_minus_one = base - F::ONE;
            for i in 0..self.num_power_bits {
                let prev = if i == 0 {
                    F::ONE
                } else {
                    intermediate_values[i - 1].square()
                };
                let bit = power_bits[self.num_power_bits - i - 1];
                let mul_by = F::ONE.multiply_accumulate(bit, base_minus_one);
                let constraint = prev * mul_by - intermediate_values[i];
                combined_gate_constraints[i * n + point] += constraint * filter;
            }
            let constraint =
                vars.local_wires[self.wire_output()] - intermediate_values[self.num_power_bits - 1];
            combined_gate_constraints[self.num_power_bits * n + point] += constraint * filter;
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let base = vars.local_wires[self.wire_base()];

        let power_bits: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_power_bit(i)])
            .collect();
        let intermediate_values: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_intermediate_value(i)])
            .collect();

        let output = vars.local_wires[self.wire_output()];

        let mut constraints = Vec::with_capacity(self.num_constraints());

        let one = builder.one_extension();
        for i in 0..self.num_power_bits {
            let prev_intermediate_value = if i == 0 {
                one
            } else {
                builder.square_extension(intermediate_values[i - 1])
            };

            // power_bits is in LE order, but we accumulate in BE order.
            let cur_bit = power_bits[self.num_power_bits - i - 1];
            let mul_by = builder.select_ext_generalized(cur_bit, base, one);
            let intermediate_value_diff =
                builder.mul_sub_extension(prev_intermediate_value, mul_by, intermediate_values[i]);
            constraints.push(intermediate_value_diff);
        }

        let output_diff =
            builder.sub_extension(output, intermediate_values[self.num_power_bits - 1]);
        constraints.push(output_diff);

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = ExponentiationGenerator::<F, D> {
            row,
            gate: self.clone(),
        };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        self.wire_intermediate_value(self.num_power_bits - 1) + 1
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        4
    }

    fn num_constraints(&self) -> usize {
        self.num_power_bits + 1
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for ExponentiationGate<F, D>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let base = vars.local_wires[self.wire_base()];

        // Both wire blocks are contiguous (bits at `1..1 + n`, intermediates at
        // `2 + n..2 + 2n`), so borrow them as strided views instead of collecting
        // copies. This runs once per packed lane group, so the two `Vec`s were
        // allocated and filled several times per batch.
        let power_bits = vars
            .local_wires
            .view(self.wire_power_bit(0)..self.wire_power_bit(0) + self.num_power_bits);
        let intermediate_values = vars.local_wires.view(
            self.wire_intermediate_value(0)
                ..self.wire_intermediate_value(0) + self.num_power_bits,
        );

        let output = vars.local_wires[self.wire_output()];

        // Rewrite `bit * base + (1 - bit)` as `1 + bit * (base - 1)`.
        // Besides deleting one packed subtraction per bit, this exposes the
        // existing AArch64 multiply-accumulate specialization. Evaluate two
        // independent transitions at a time to increase instruction-level
        // parallelism without changing constraint emission order.
        let base_minus_one = base - P::ONES;
        let mut i = 0;
        while i + 1 < self.num_power_bits {
            let prev_0 = if i == 0 {
                P::ONES
            } else {
                intermediate_values[i - 1].square()
            };
            let prev_1 = intermediate_values[i].square();

            // power_bits is in LE order, but we accumulate in BE order.
            let bit_0 = power_bits[self.num_power_bits - i - 1];
            let bit_1 = power_bits[self.num_power_bits - i - 2];
            let mul_by_0 = P::ONES.multiply_accumulate(bit_0, base_minus_one);
            let mul_by_1 = P::ONES.multiply_accumulate(bit_1, base_minus_one);

            yield_constr.one(prev_0 * mul_by_0 - intermediate_values[i]);
            yield_constr.one(prev_1 * mul_by_1 - intermediate_values[i + 1]);
            i += 2;
        }
        if i < self.num_power_bits {
            let prev = if i == 0 {
                P::ONES
            } else {
                intermediate_values[i - 1].square()
            };
            let bit = power_bits[self.num_power_bits - i - 1];
            let mul_by = P::ONES.multiply_accumulate(bit, base_minus_one);
            yield_constr.one(prev * mul_by - intermediate_values[i]);
        }

        yield_constr.one(output - intermediate_values[self.num_power_bits - 1]);
    }
}

#[derive(Debug, Default)]
pub struct ExponentiationGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    gate: ExponentiationGate<F, D>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for ExponentiationGenerator<F, D>
{
    fn id(&self) -> String {
        "ExponentiationGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| Target::wire(self.row, column);

        let mut deps = Vec::with_capacity(self.gate.num_power_bits + 1);
        deps.push(local_target(self.gate.wire_base()));
        for i in 0..self.gate.num_power_bits {
            deps.push(local_target(self.gate.wire_power_bit(i)));
        }
        deps
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let get_local_wire = |column| witness.get_wire(local_wire(column));

        let num_power_bits = self.gate.num_power_bits;
        let base = get_local_wire(self.gate.wire_base());

        let power_bits = (0..num_power_bits)
            .map(|i| get_local_wire(self.gate.wire_power_bit(i)))
            .collect::<Vec<_>>();
        let mut intermediate_values = Vec::with_capacity(num_power_bits);

        let mut current_intermediate_value = F::ONE;
        for i in 0..num_power_bits {
            if power_bits[num_power_bits - i - 1] == F::ONE {
                current_intermediate_value *= base;
            }
            intermediate_values.push(current_intermediate_value);
            current_intermediate_value *= current_intermediate_value;
        }

        for i in 0..num_power_bits {
            let intermediate_value_wire = local_wire(self.gate.wire_intermediate_value(i));
            out_buffer.set_wire(intermediate_value_wire, intermediate_values[i])?;
        }

        let output_wire = local_wire(self.gate.wire_output());
        out_buffer.set_wire(output_wire, intermediate_values[num_power_bits - 1])
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        self.gate.serialize(dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let gate = ExponentiationGate::deserialize(src, _common_data)?;
        Ok(Self { row, gate })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rand::rngs::OsRng;
    use rand::Rng;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Sample;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::HashOut;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::util::log2_ceil;

    const MAX_POWER_BITS: usize = 17;

    #[test]
    fn wire_indices() {
        let gate = ExponentiationGate::<GoldilocksField, 4> {
            num_power_bits: 5,
            _phantom: PhantomData,
        };

        assert_eq!(gate.wire_base(), 0);
        assert_eq!(gate.wire_power_bit(0), 1);
        assert_eq!(gate.wire_power_bit(4), 5);
        assert_eq!(gate.wire_output(), 6);
        assert_eq!(gate.wire_intermediate_value(0), 7);
        assert_eq!(gate.wire_intermediate_value(4), 11);
    }

    #[test]
    fn low_degree() {
        let config = CircuitConfig {
            num_wires: 120,
            num_routed_wires: 30,
            ..CircuitConfig::standard_recursion_config()
        };

        test_low_degree::<GoldilocksField, _, 4>(ExponentiationGate::new_from_config(&config));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(ExponentiationGate::new_from_config(
            &CircuitConfig::standard_recursion_config(),
        ))
    }

    #[test]
    fn eval_fns_production_67_bits() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = CircuitConfig {
            num_wires: 136,
            num_routed_wires: 80,
            ..CircuitConfig::standard_recursion_config()
        };
        let gate = ExponentiationGate::new_from_config(&config);
        assert_eq!(gate.num_power_bits, 67);
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    fn direct_filtered_accumulation_matches_materialized_production_gate() {
        const D: usize = 2;
        type F = GoldilocksField;
        use crate::field::batch_util::batch_multiply_add_inplace;
        use crate::field::types::{Field64, PrimeField64};

        fn value(i: usize) -> F {
            let small = ((i as u64).wrapping_mul(0x9e37_79b9) ^ 0x5a5a_a5a5) & 0xffff;
            if i % 3 == 0 {
                GoldilocksField(F::ORDER + small)
            } else {
                F::from_canonical_u64(small)
            }
        }

        let config = CircuitConfig {
            num_wires: 136,
            num_routed_wires: 80,
            ..CircuitConfig::standard_recursion_config()
        };
        let gate = ExponentiationGate::<F, D>::new_from_config(&config);
        assert_eq!(gate.num_power_bits, 67);
        for n in [1, 3, 4, 5, 7, 11, 31, 32, 33] {
            let wires = (0..gate.num_wires() * n)
                .map(|i| value(i + 1))
                .collect::<Vec<_>>();
            let filters = (0..n)
                .map(|i| if i % 5 == 0 { F::ZERO } else { value(i + 10_001) })
                .collect::<Vec<_>>();
            let hash = HashOut::ZERO;
            let vars = EvaluationVarsBaseBatch::new(n, &[], &wires, &hash);
            let materialized = gate.eval_unfiltered_base_batch(vars);
            let initial = (0..gate.num_constraints() * n)
                .map(|i| value(i + 20_001))
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
                    "n={n}, output={i}"
                );
            }
        }
    }

    #[test]
    fn test_gate_constraint() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type FF = <C as GenericConfig<D>>::FE;

        /// Returns the local wires for an exponentiation gate given the base, power, and power bit
        /// values.
        fn get_wires(base: F, power: u64) -> Vec<FF> {
            let mut power_bits = Vec::new();
            let mut cur_power = power;
            while cur_power > 0 {
                power_bits.push(cur_power % 2);
                cur_power /= 2;
            }

            let num_power_bits = power_bits.len();

            let power_bits_f: Vec<_> = power_bits
                .iter()
                .map(|b| F::from_canonical_u64(*b))
                .collect();

            let mut v = vec![base];
            v.extend(power_bits_f);

            let mut intermediate_values = Vec::new();
            let mut current_intermediate_value = F::ONE;
            for i in 0..num_power_bits {
                if power_bits[num_power_bits - i - 1] == 1 {
                    current_intermediate_value *= base;
                }
                intermediate_values.push(current_intermediate_value);
                current_intermediate_value *= current_intermediate_value;
            }
            let output_value = intermediate_values[num_power_bits - 1];
            v.push(output_value);
            v.extend(intermediate_values);

            v.iter().map(|&x| x.into()).collect::<Vec<_>>()
        }

        let mut rng = OsRng;

        let base = F::TWO;
        let power = rng.gen::<usize>() % (1 << MAX_POWER_BITS);
        let num_power_bits = log2_ceil(power + 1);
        let gate = ExponentiationGate::<F, D> {
            num_power_bits,
            _phantom: PhantomData,
        };

        let vars = EvaluationVars {
            local_constants: &[],
            local_wires: &get_wires(base, power as u64),
            public_inputs_hash: &HashOut::rand(),
        };
        assert!(
            gate.eval_unfiltered(vars).iter().all(|x| x.is_zero()),
            "Gate constraints are not satisfied."
        );
    }

    /// Manual timing harness comparing the packed-fused accumulate against the
    /// materialize-then-add default it replaced. Run with:
    /// `cargo test --release -p plonky2 exp_accumulate_micro -- --ignored --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn exp_accumulate_microbenchmark() {
        use core::hint::black_box;
        use std::time::Instant;

        use plonky2_field::types::Sample;

        use crate::field::batch_util::batch_multiply_add_inplace;
        use crate::gates::gate::Gate;
        use crate::plonk::vars::EvaluationVarsBaseBatch;

        const D: usize = 2;
        type F = GoldilocksField;
        let gate = ExponentiationGate::<F, D>::new_from_config(&CircuitConfig::standard_recursion_config());
        let n = 32;
        let wires = F::rand_vec(gate.num_wires() * n);
        let constants: Vec<F> = Vec::new();
        let hash = crate::hash::hash_types::HashOut::ZERO;
        let filters = F::rand_vec(n);
        let nc = gate.num_constraints();
        let mut combined = vec![F::ZERO; nc * n];
        let iters = 50_000u32;
        let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);

        let mut t_fused = 0.0f64;
        let mut t_mat = 0.0f64;
        for _ in 0..4 {
            let s = Instant::now();
            for _ in 0..iters {
                gate.eval_unfiltered_base_batch_accumulate(vars, &filters, black_box(&mut combined));
            }
            t_fused += s.elapsed().as_secs_f64();

            let s = Instant::now();
            for _ in 0..iters {
                let res = gate.eval_unfiltered_base_batch(vars);
                for (acc, row) in combined.chunks_exact_mut(n).zip(res.chunks_exact(n)) {
                    batch_multiply_add_inplace(acc, row, &filters);
                }
            }
            t_mat += s.elapsed().as_secs_f64();
        }
        let per = |t: f64| t / (4.0 * iters as f64) * 1e6;
        println!(
            "exponentiation accumulate per batch (n=32): packed-fused {:.2} us, materialized-default {:.2} us",
            per(t_fused), per(t_mat)
        );
    }

}
