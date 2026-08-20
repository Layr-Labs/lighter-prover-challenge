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

    /// Direct quotient accumulator for the one exponentiation shape used by
    /// production recursion. Keeping this separate from the generic packed
    /// evaluator avoids its 68-by-4 constraint scratch, zero-fill, iterator,
    /// and `StridedConstraintConsumer` while retaining exactly the same packed
    /// operations and constraint order.
    #[inline(never)]
    fn eval_production_batch_accumulate<P: PackedField<Scalar = F>>(
        &self,
        vars_base: EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        const BITS: usize = 67;
        const N: usize = 32;
        const WIDTH: usize = 4;
        const PACKS_PER_COLUMN: usize = N / WIDTH;
        const NUM_WIRES: usize = 2 + 2 * BITS;
        const OUTPUT: usize = 1 + BITS;
        const INTERMEDIATE_START: usize = 2 + BITS;
        debug_assert_eq!(D, 2);
        debug_assert_eq!(self.num_power_bits, BITS);
        debug_assert_eq!(vars_base.len(), N);
        debug_assert_eq!(P::WIDTH, WIDTH);
        assert_eq!(filters.len(), N);
        assert!(combined_gate_constraints.len() >= (BITS + 1) * N);

        // Reinterpret each complete four-lane group once up front. This is the
        // same layout used by `EvaluationVarsBaseBatch::pack`, but without its
        // packed/leftover iterators and per-group strided views.
        let wires = P::pack_slice(&vars_base.local_wires[..NUM_WIRES * N]);
        let filters = P::pack_slice(filters);
        let combined = P::pack_slice_mut(
            &mut combined_gate_constraints[..(BITS + 1) * N],
        );

        macro_rules! accumulate {
            ($row:expr, $constraint:expr, $group:expr, $filter:expr) => {{
                let index = $row * PACKS_PER_COLUMN + $group;
                combined[index] =
                    combined[index].multiply_accumulate($constraint, $filter);
            }};
        }

        for group in 0..PACKS_PER_COLUMN {
            let wire = |column: usize| wires[column * PACKS_PER_COLUMN + group];
            let filter = filters[group];
            let base = wire(0);
            let base_minus_one = base - P::ONES;

            // Match `eval_unfiltered_base_packed`'s four-way transition
            // schedule exactly. In particular, keep the multiply-accumulate
            // operand order so noncanonical Goldilocks representatives agree.
            let mut i = 0;
            while i + 3 < BITS {
                let prev_0 = if i == 0 {
                    P::ONES
                } else {
                    wire(INTERMEDIATE_START + i - 1).square()
                };
                let prev_1 = wire(INTERMEDIATE_START + i).square();
                let prev_2 = wire(INTERMEDIATE_START + i + 1).square();
                let prev_3 = wire(INTERMEDIATE_START + i + 2).square();

                let bit_0 = wire(BITS - i);
                let bit_1 = wire(BITS - i - 1);
                let bit_2 = wire(BITS - i - 2);
                let bit_3 = wire(BITS - i - 3);
                let mul_by_0 = P::ONES.multiply_accumulate(bit_0, base_minus_one);
                let mul_by_1 = P::ONES.multiply_accumulate(bit_1, base_minus_one);
                let mul_by_2 = P::ONES.multiply_accumulate(bit_2, base_minus_one);
                let mul_by_3 = P::ONES.multiply_accumulate(bit_3, base_minus_one);

                let constraint_0 = prev_0 * mul_by_0 - wire(INTERMEDIATE_START + i);
                let constraint_1 = prev_1 * mul_by_1 - wire(INTERMEDIATE_START + i + 1);
                let constraint_2 = prev_2 * mul_by_2 - wire(INTERMEDIATE_START + i + 2);
                let constraint_3 = prev_3 * mul_by_3 - wire(INTERMEDIATE_START + i + 3);
                accumulate!(i, constraint_0, group, filter);
                accumulate!(i + 1, constraint_1, group, filter);
                accumulate!(i + 2, constraint_2, group, filter);
                accumulate!(i + 3, constraint_3, group, filter);
                i += 4;
            }

            // The production width is 67, so the exact four-way schedule ends
            // with the same pair and scalar tails as the generic evaluator.
            while i + 1 < BITS {
                let prev_0 = if i == 0 {
                    P::ONES
                } else {
                    wire(INTERMEDIATE_START + i - 1).square()
                };
                let prev_1 = wire(INTERMEDIATE_START + i).square();
                let bit_0 = wire(BITS - i);
                let bit_1 = wire(BITS - i - 1);
                let mul_by_0 = P::ONES.multiply_accumulate(bit_0, base_minus_one);
                let mul_by_1 = P::ONES.multiply_accumulate(bit_1, base_minus_one);
                let constraint_0 = prev_0 * mul_by_0 - wire(INTERMEDIATE_START + i);
                let constraint_1 = prev_1 * mul_by_1 - wire(INTERMEDIATE_START + i + 1);
                accumulate!(i, constraint_0, group, filter);
                accumulate!(i + 1, constraint_1, group, filter);
                i += 2;
            }
            if i < BITS {
                let prev = if i == 0 {
                    P::ONES
                } else {
                    wire(INTERMEDIATE_START + i - 1).square()
                };
                let bit = wire(BITS - i);
                let mul_by = P::ONES.multiply_accumulate(bit, base_minus_one);
                let constraint = prev * mul_by - wire(INTERMEDIATE_START + i);
                accumulate!(i, constraint, group, filter);
            }

            let constraint = wire(OUTPUT) - wire(INTERMEDIATE_START + BITS - 1);
            accumulate!(BITS, constraint, group, filter);
        }
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
        // This is deliberately an exact-shape gate. All other extension
        // degrees, exponent widths, batch lengths, and packing widths retain
        // the generic implementation verbatim.
        if D == 2
            && self.num_power_bits == 67
            && vars_base.len() == 32
            && <<F as Packable>::Packing as PackedField>::WIDTH == 4
        {
            self.eval_production_batch_accumulate::<<F as Packable>::Packing>(
                vars_base,
                filters,
                combined_gate_constraints,
            );
        } else {
            self.eval_unfiltered_base_batch_accumulate_packed(
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
        // existing AArch64 multiply-accumulate specialization. Evaluate four
        // independent transitions at a time to give the backend more reduction
        // chains to overlap without changing constraint emission order.
        let base_minus_one = base - P::ONES;
        let mut i = 0;
        while i + 3 < self.num_power_bits {
            let prev_0 = if i == 0 {
                P::ONES
            } else {
                intermediate_values[i - 1].square()
            };
            let prev_1 = intermediate_values[i].square();
            let prev_2 = intermediate_values[i + 1].square();
            let prev_3 = intermediate_values[i + 2].square();

            // power_bits is in LE order, but we accumulate in BE order.
            let bit_0 = power_bits[self.num_power_bits - i - 1];
            let bit_1 = power_bits[self.num_power_bits - i - 2];
            let bit_2 = power_bits[self.num_power_bits - i - 3];
            let bit_3 = power_bits[self.num_power_bits - i - 4];
            let mul_by_0 = P::ONES.multiply_accumulate(bit_0, base_minus_one);
            let mul_by_1 = P::ONES.multiply_accumulate(bit_1, base_minus_one);
            let mul_by_2 = P::ONES.multiply_accumulate(bit_2, base_minus_one);
            let mul_by_3 = P::ONES.multiply_accumulate(bit_3, base_minus_one);

            yield_constr.one(prev_0 * mul_by_0 - intermediate_values[i]);
            yield_constr.one(prev_1 * mul_by_1 - intermediate_values[i + 1]);
            yield_constr.one(prev_2 * mul_by_2 - intermediate_values[i + 2]);
            yield_constr.one(prev_3 * mul_by_3 - intermediate_values[i + 3]);
            i += 4;
        }
        // Preserve the pair and scalar fallback shapes for arbitrary gate widths.
        while i + 1 < self.num_power_bits {
            let prev_0 = if i == 0 {
                P::ONES
            } else {
                intermediate_values[i - 1].square()
            };
            let prev_1 = intermediate_values[i].square();
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
    use crate::field::types::{Field64, PrimeField64, Sample};
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

    /// The four-way schedule must only change instruction overlap: for every
    /// lane it must retain the exact representative produced by the scalar
    /// transition sequence, not merely a congruent field value.
    #[test]
    fn packed_eval_matches_scalar_raw_and_canonical_noncanonical_lanes() {
        type F = GoldilocksField;
        const D: usize = 2;
        const EDGES: &[u64] = &[
            0,
            1,
            2,
            u32::MAX as u64,
            1 << 32,
            0xffff_fffe_ffff_ffff,
            F::ORDER - 1,
            F::ORDER,
            F::ORDER + 1,
            0xffff_ffff_ffff_fffe,
            u64::MAX,
        ];

        fn scalar_reference(
            gate: &ExponentiationGate<F, D>,
            n: usize,
            wires: &[F],
        ) -> Vec<F> {
            let mut constraints = vec![F::ZERO; gate.num_constraints() * n];
            for point in 0..n {
                let wire = |column: usize| wires[column * n + point];
                let base_minus_one = wire(gate.wire_base()) - F::ONE;
                for i in 0..gate.num_power_bits {
                    let prev = if i == 0 {
                        F::ONE
                    } else {
                        wire(gate.wire_intermediate_value(i - 1)).square()
                    };
                    let bit = wire(gate.wire_power_bit(gate.num_power_bits - i - 1));
                    let mul_by = Field::multiply_accumulate(&F::ONE, bit, base_minus_one);
                    constraints[i * n + point] =
                        prev * mul_by - wire(gate.wire_intermediate_value(i));
                }
                constraints[gate.num_power_bits * n + point] =
                    wire(gate.wire_output())
                        - wire(gate.wire_intermediate_value(gate.num_power_bits - 1));
            }
            constraints
        }

        // Widths 1..7 exercise the scalar, pair, and four-way transition
        // fallbacks. Width 67 is the production gate shape; batches include
        // both packed/ragged layouts and the exact quotient batch size of 32.
        for bits in (1..=7).chain(core::iter::once(67)) {
            let gate = ExponentiationGate::<F, D>::new(bits);
            let batch_sizes: &[usize] = if bits == 67 {
                &[1, 3, 4, 5, 32]
            } else {
                &[7]
            };
            for &n in batch_sizes {
                let wires = (0..gate.num_wires() * n)
                    .map(|index| {
                        // Column and lane use different odd strides, so each
                        // packed vector contains a different borrow/carry mix.
                        let column = index / n;
                        let lane = index % n;
                        GoldilocksField(EDGES[(5 * column + 7 * lane + bits) % EDGES.len()])
                    })
                    .collect::<Vec<_>>();
                assert!(wires.iter().any(|x| x.0 >= F::ORDER));
                let canonical_wires = wires
                    .iter()
                    .map(PrimeField64::to_canonical)
                    .collect::<Vec<_>>();
                let hash = HashOut::ZERO;

                let got = gate.eval_unfiltered_base_batch(EvaluationVarsBaseBatch::new(
                    n,
                    &[],
                    &wires,
                    &hash,
                ));
                let want = scalar_reference(&gate, n, &wires);
                let canonical_got = gate.eval_unfiltered_base_batch(EvaluationVarsBaseBatch::new(
                    n,
                    &[],
                    &canonical_wires,
                    &hash,
                ));

                for constraint in 0..gate.num_constraints() {
                    for lane in 0..n {
                        let index = constraint * n + lane;
                        assert_eq!(
                            got[index].0,
                            want[index].0,
                            "raw mismatch: bits={bits} n={n} constraint={constraint} lane={lane}",
                        );
                        assert_eq!(
                            got[index].to_canonical_u64(),
                            canonical_got[index].to_canonical_u64(),
                            "canonical mismatch: bits={bits} n={n} constraint={constraint} lane={lane}",
                        );
                    }
                }
            }
        }
    }

    /// The direct production accumulator is a pure dispatch specialization:
    /// compare its raw representatives with the generic packed oracle across
    /// every adjacent shape dimension, starting from nonzero accumulators.
    #[test]
    fn production_accumulate_dispatch_matches_generic_raw_matrix() {
        type F = GoldilocksField;
        const WIRE_EDGES: &[u64] = &[
            0,
            1,
            2,
            u32::MAX as u64,
            1 << 32,
            0xffff_fffe_ffff_ffff,
            F::ORDER - 1,
            F::ORDER,
            F::ORDER + 1,
            0xffff_ffff_ffff_fffe,
            u64::MAX,
        ];
        // Every one of these is nonzero both as a raw representative and as a
        // field value; the noncanonical entries stress the final fused add.
        const ACC_EDGES: &[u64] = &[
            1,
            2,
            u32::MAX as u64,
            1 << 32,
            0xffff_fffe_ffff_ffff,
            F::ORDER - 1,
            F::ORDER + 1,
            0xffff_ffff_ffff_fffe,
            u64::MAX,
        ];

        fn check<const D: usize>()
        where
            F: Extendable<D>,
        {
            for bits in [66, 67, 68] {
                for n in [31, 32, 33] {
                    let gate = ExponentiationGate::<F, D>::new(bits);
                    let wires = (0..gate.num_wires() * n)
                        .map(|index| {
                            let column = index / n;
                            let lane = index % n;
                            GoldilocksField(
                                WIRE_EDGES
                                    [(11 * column + 7 * lane + bits + n) % WIRE_EDGES.len()],
                            )
                        })
                        .collect::<Vec<_>>();
                    let filters = (0..n)
                        .map(|lane| {
                            GoldilocksField(
                                WIRE_EDGES[(5 * lane + bits + 3 * n) % WIRE_EDGES.len()],
                            )
                        })
                        .collect::<Vec<_>>();
                    // Include a non-row-aligned suffix to check that neither
                    // implementation touches beyond the advertised constraints.
                    let initial = (0..gate.num_constraints() * n + 5)
                        .map(|index| {
                            GoldilocksField(
                                ACC_EDGES[(13 * index + bits + n) % ACC_EDGES.len()],
                            )
                        })
                        .collect::<Vec<_>>();
                    assert!(initial.iter().all(|x| !x.is_zero()));

                    let hash = HashOut::ZERO;
                    let vars = EvaluationVarsBaseBatch::new(n, &[], &wires, &hash);
                    let mut got = initial.clone();
                    let mut generic = initial;
                    gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut got);
                    <ExponentiationGate<F, D> as PackedEvaluableBase<F, D>>::
                        eval_unfiltered_base_batch_accumulate_packed(
                            &gate,
                            vars,
                            &filters,
                            &mut generic,
                        );

                    let direct_shape = D == 2
                        && bits == 67
                        && n == 32
                        && <<F as Packable>::Packing as PackedField>::WIDTH == 4;
                    for index in 0..got.len() {
                        assert_eq!(
                            got[index].0,
                            generic[index].0,
                            "raw mismatch: D={D} bits={bits} n={n} direct={direct_shape} index={index}",
                        );
                        assert_eq!(
                            got[index].to_canonical_u64(),
                            generic[index].to_canonical_u64(),
                            "canonical mismatch: D={D} bits={bits} n={n} direct={direct_shape} index={index}",
                        );
                    }
                }
            }
        }

        check::<2>();
        check::<4>();
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

    /// Manual timing harness comparing the exact direct accumulator with its
    /// generic packed fallback. Run with:
    /// `cargo test --manifest-path vendor/plonky2/Cargo.toml --release -p plonky2 exp_accumulate_micro -- --ignored --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn exp_accumulate_microbenchmark() {
        use core::hint::black_box;
        use std::time::Instant;

        use plonky2_field::types::Sample;

        use crate::gates::gate::Gate;
        use crate::plonk::vars::EvaluationVarsBaseBatch;

        const D: usize = 2;
        type F = GoldilocksField;
        let config = CircuitConfig {
            num_wires: 136,
            num_routed_wires: 80,
            ..CircuitConfig::standard_recursion_config()
        };
        let gate = ExponentiationGate::<F, D>::new_from_config(&config);
        assert_eq!(gate.num_power_bits, 67);
        // Quotient evaluation dispatches batches of exactly 32 points.
        let n = 32;
        let wires = F::rand_vec(gate.num_wires() * n);
        let constants: Vec<F> = Vec::new();
        let hash = crate::hash::hash_types::HashOut::ZERO;
        let filters = F::rand_vec(n);
        let nc = gate.num_constraints();
        let initial = F::rand_vec(nc * n);
        const ITERS: usize = 50_000;
        const ROUNDS: usize = 6;
        let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);

        let time_generic = |combined: &mut [F]| {
            let start = Instant::now();
            for _ in 0..ITERS {
                <ExponentiationGate<F, D> as PackedEvaluableBase<F, D>>::
                    eval_unfiltered_base_batch_accumulate_packed(
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
                gate.eval_unfiltered_base_batch_accumulate(
                    vars,
                    &filters,
                    black_box(&mut *combined),
                );
            }
            start.elapsed()
        };

        // Alternate order so host drift is visible rather than consistently
        // favoring one implementation.
        let mut generic_seconds = 0.0;
        let mut direct_seconds = 0.0;
        for round in 0..ROUNDS {
            let mut generic = initial.clone();
            let mut direct = initial.clone();
            let (tg, td) = if round % 2 == 0 {
                (time_generic(&mut generic), time_direct(&mut direct))
            } else {
                let td = time_direct(&mut direct);
                let tg = time_generic(&mut generic);
                (tg, td)
            };
            assert!(generic
                .iter()
                .zip(&direct)
                .all(|(a, b)| a.0 == b.0));
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
            "exponentiation n=32/bits=67 accumulate: generic {:.3} us, direct {:.3} us, {:.3}x",
            generic_seconds * 1e6 / (ROUNDS * ITERS) as f64,
            direct_seconds * 1e6 / (ROUNDS * ITERS) as f64,
            generic_seconds / direct_seconds,
        );
    }

}
