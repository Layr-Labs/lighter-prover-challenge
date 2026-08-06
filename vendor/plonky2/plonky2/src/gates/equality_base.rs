#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};
extern crate alloc;
use alloc::string::ToString;

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::{BoolTarget, Target};
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate specialized for Equality Checks
#[derive(Debug, Clone, Default)]
pub struct EqualityGate {
    /// Number of additions operations performed by an Equality gate.
    pub num_ops: usize,
}

impl EqualityGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }
    //Number of routed wires necessary for an operation
    const ROUTED_PER_OP: usize = 3;
    const NOT_ROUTED_PER_OP: usize = 3;
    const TOTAL_PER_OP: usize = Self::ROUTED_PER_OP + Self::NOT_ROUTED_PER_OP;
    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let routed_packed_count = config.num_routed_wires / Self::ROUTED_PER_OP;
        let unrouted_packed_count = config.num_wires / Self::TOTAL_PER_OP;
        if routed_packed_count < unrouted_packed_count {
            routed_packed_count
        } else {
            unrouted_packed_count
        }
    }

    pub(crate) const fn wire_ith_element_0(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i
    }
    pub(crate) const fn wire_ith_element_1(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i + 1
    }
    pub(crate) const fn wire_ith_output(&self, i: usize) -> usize {
        assert!(i < self.num_ops);
        Self::ROUTED_PER_OP * i + 2
    }

    pub(crate) const fn wire_ith_temporary(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < Self::NOT_ROUTED_PER_OP);
        Self::ROUTED_PER_OP * self.num_ops + i * Self::NOT_ROUTED_PER_OP + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for EqualityGate {
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
        let const_0 = vars.local_constants[0]; //"one" value
        let mut constraints = Vec::with_capacity(self.num_ops * 4);

        for i in 0..self.num_ops {
            let x = vars.local_wires[self.wire_ith_element_0(i)];
            let y = vars.local_wires[self.wire_ith_element_1(i)];
            let equal = vars.local_wires[self.wire_ith_output(i)];
            let diff = vars.local_wires[self.wire_ith_temporary(i, 0)];
            let invdiff = vars.local_wires[self.wire_ith_temporary(i, 1)];
            let prod = vars.local_wires[self.wire_ith_temporary(i, 2)];
            constraints.push((x - y) - diff);
            constraints.push((diff * invdiff) - prod);
            constraints.push((prod * diff) - diff);
            constraints.push((const_0 - prod) - equal);
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

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = vars.local_constants[0];
        let mut constraints = Vec::with_capacity(self.num_ops * 4);

        for i in 0..self.num_ops {
            let x = vars.local_wires[self.wire_ith_element_0(i)];
            let y = vars.local_wires[self.wire_ith_element_1(i)];
            let equal = vars.local_wires[self.wire_ith_output(i)];
            let diff = vars.local_wires[self.wire_ith_temporary(i, 0)];
            let invdiff = vars.local_wires[self.wire_ith_temporary(i, 1)];
            let prod = vars.local_wires[self.wire_ith_temporary(i, 2)];

            constraints.push({
                let inner = builder.sub_extension(x, y);
                builder.sub_extension(inner, diff)
            });
            constraints.push(builder.mul_sub_extension(diff, invdiff, prod));
            constraints.push(builder.mul_sub_extension(prod, diff, diff));
            let inner = builder.sub_extension(const_0, prod);
            constraints.push(builder.sub_extension(inner, equal))
        }

        constraints
    }

    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        // One cheap generator per op (diff/equal/prod, no inversion) plus a
        // single per-row generator that batch-inverts every nonzero diff with
        // one field inversion. `invdiff` sits on a non-routed wire, so nothing
        // outside this row's constraints can depend on it and deferring it to
        // row completion cannot create a dependency cycle.
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    EqualityBaseGenerator {
                        gate: self.clone(),
                        row,
                        const_0: local_constants[0],
                        i,
                    }
                    .adapter(),
                )
            })
            .chain(core::iter::once(WitnessGeneratorRef::new(
                EqualityRowInverseGenerator {
                    gate: self.clone(),
                    row,
                    const_0: local_constants[0],
                }
                .adapter(),
            )))
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * Self::TOTAL_PER_OP
    }

    // The default implementation counts generators, which no longer matches
    // the op count now that one fused generator covers the whole row.
    fn num_ops(&self) -> usize {
        self.num_ops
    }

    fn num_constants(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * 4
    }

    fn input_wires_defaults(&self, index: usize) -> Vec<(usize, F)> {
        Vec::from([
            (self.wire_ith_element_0(index), F::ZERO),
            (self.wire_ith_element_1(index), F::ZERO),
        ])
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for EqualityGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let const_0 = vars.local_constants[0];
        for i in 0..self.num_ops {
            let x = vars.local_wires[self.wire_ith_element_0(i)];
            let y = vars.local_wires[self.wire_ith_element_1(i)];
            let equal = vars.local_wires[self.wire_ith_output(i)];
            let diff = vars.local_wires[self.wire_ith_temporary(i, 0)];
            let invdiff = vars.local_wires[self.wire_ith_temporary(i, 1)];
            let prod = vars.local_wires[self.wire_ith_temporary(i, 2)];

            yield_constr.one((x - y) - diff);
            yield_constr.one((diff * invdiff) - prod);
            yield_constr.one((prod * diff) - diff);
            yield_constr.one((const_0 - prod) - equal);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EqualityBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    pub gate: EqualityGate,
    pub row: usize,
    pub const_0: F,
    pub i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for EqualityBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "EqualityBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        [
            self.gate.wire_ith_element_0(self.i),
            self.gate.wire_ith_element_1(self.i),
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

        let x = get_wire(self.gate.wire_ith_element_0(self.i));
        let y = get_wire(self.gate.wire_ith_element_1(self.i));
        let equal = Target::wire(self.row, self.gate.wire_ith_output(self.i));
        let diff = Target::wire(self.row, self.gate.wire_ith_temporary(self.i, 0));
        let prod = Target::wire(self.row, self.gate.wire_ith_temporary(self.i, 2));

        let prod_value = if x != y { F::ONE } else { F::ZERO };

        // `invdiff` is produced by EqualityRowInverseGenerator, which shares
        // one field inversion across the whole row.
        out_buffer.set_target(diff, x - y)?;
        out_buffer.set_bool_target(BoolTarget::new_unsafe(equal), x == y)?;
        out_buffer.set_target(prod, prod_value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_field(self.const_0)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = EqualityGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let const_0 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            const_0,
            i,
        })
    }
}

/// Fills the `invdiff` wire of every equality slot in one gate row, sharing a
/// single field inversion across the row via Montgomery's batch-inverse trick
/// instead of paying one full inversion per unequal slot.
///
/// It depends only on the row's `diff` temporaries, which are non-routed wires
/// written exclusively by the per-slot `EqualityBaseGenerator`s, so it becomes
/// ready exactly when the whole row is decided and cannot participate in a
/// copy-constraint dependency cycle.
#[derive(Clone, Debug, Default)]
pub struct EqualityRowInverseGenerator<F: RichField + Extendable<D>, const D: usize> {
    pub gate: EqualityGate,
    pub row: usize,
    pub const_0: F,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for EqualityRowInverseGenerator<F, D>
{
    fn id(&self) -> String {
        "EqualityRowInverseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..self.gate.num_ops)
            .map(|i| Target::wire(self.row, self.gate.wire_ith_temporary(i, 0)))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let num_ops = self.gate.num_ops;

        // prefix[i] = product of the nonzero diffs before slot i.
        let mut diffs = Vec::with_capacity(num_ops);
        let mut prefix = Vec::with_capacity(num_ops);
        let mut acc = F::ONE;
        for i in 0..num_ops {
            let diff =
                witness.get_target(Target::wire(self.row, self.gate.wire_ith_temporary(i, 0)));
            prefix.push(acc);
            if diff != F::ZERO {
                acc *= diff;
            }
            diffs.push(diff);
        }

        // One inversion for the whole row; unwind to per-slot inverses. Each
        // result equals `diff.inverse()` exactly (field inverses are unique).
        let mut inv = acc.inverse();
        for i in (0..num_ops).rev() {
            let diff = diffs[i];
            let inv_value = if diff != F::ZERO {
                let inv_i = inv * prefix[i];
                inv *= diff;
                inv_i
            } else {
                F::ZERO
            };
            out_buffer.set_target(
                Target::wire(self.row, self.gate.wire_ith_temporary(i, 1)),
                inv_value,
            )?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_field(self.const_0)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = EqualityGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let const_0 = src.read_field()?;
        Ok(Self { gate, row, const_0 })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;
    #[allow(unused_imports)]
    use crate::field::types::Field64;
    use crate::gates::equality_base::EqualityGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::iop::target::{BoolTarget, Target};
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate = EqualityGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = EqualityGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    fn test_succes() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());

        // Create targets for x and y
        let x = builder.add_virtual_target();
        let y = builder.add_virtual_target();

        // Instantiate your custom EqualityGate and get BoolTarget
        let gate = EqualityGate::new_from_config(&config);
        let ref_gate = gate.clone();
        let constants = [F::ONE];
        let (gate_row, i) = builder.find_slot(gate, &constants, &constants);

        let wire_x = Target::wire(gate_row, ref_gate.wire_ith_element_0(i));
        let wire_y = Target::wire(gate_row, ref_gate.wire_ith_element_1(i));
        let wire_equal = Target::wire(gate_row, ref_gate.wire_ith_output(i));

        builder.connect(x, wire_x);
        builder.connect(y, wire_y);

        let equal = BoolTarget::new_unsafe(wire_equal);

        // Optionally use equal in the circuit logic
        builder.assert_bool(equal);

        let circuit_data = builder.build::<C>();

        // Now set values for x and y such that x == y, so equal = 1
        let mut pw = PartialWitness::new();
        let value1 = F::from_canonical_u64(17);
        let value2 = F::from_canonical_u64(18);
        pw.set_target(x, value1)?;
        pw.set_target(y, value2)?;

        let proof = circuit_data.prove(pw)?;
        circuit_data.verify(proof)?;

        Ok(())
    }

    #[test]
    fn row_inverse_generator_matches_per_slot_values() -> Result<()> {
        use crate::field::types::PrimeField64;
        use crate::iop::generator::generate_partial_witness;
        use crate::iop::witness::Witness;

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let gate = EqualityGate::new_from_config(&config);

        // Span several rows, leaving the last row partially filled so the
        // default-valued (zero) slots are exercised too.
        let n = gate.num_ops * 2 + 3;
        let mut inputs = Vec::new();
        let mut slots = Vec::new();
        for _ in 0..n {
            let x = builder.add_virtual_target();
            let y = builder.add_virtual_target();
            let (row, i) = builder.find_slot(gate.clone(), &[F::ONE], &[F::ONE]);
            builder.connect(x, Target::wire(row, gate.wire_ith_element_0(i)));
            builder.connect(y, Target::wire(row, gate.wire_ith_element_1(i)));
            inputs.push((x, y));
            slots.push((row, i));
        }
        let circuit = builder.build_prover::<C>();

        let mut pw = PartialWitness::new();
        let mut values = Vec::new();
        for (k, &(x, y)) in inputs.iter().enumerate() {
            let (xv, yv) = match k % 4 {
                // equal pair (diff = 0)
                0 => (
                    F::from_canonical_u64(k as u64 + 7),
                    F::from_canonical_u64(k as u64 + 7),
                ),
                // generic unequal pair
                1 => (
                    F::from_canonical_u64(0x1234_5678_9abc_def0 ^ k as u64),
                    F::from_canonical_u64(99 + k as u64),
                ),
                // zero minus nonzero
                2 => (F::ZERO, F::from_canonical_u64(k as u64 + 1)),
                // p - 1 vs arbitrary
                _ => (F::NEG_ONE, F::from_canonical_u64(k as u64 * 31 + 1)),
            };
            pw.set_target(x, xv)?;
            pw.set_target(y, yv)?;
            values.push((xv, yv));
        }

        let witness = generate_partial_witness(pw, &circuit.prover_only, &circuit.common)?;

        for (k, &(row, i)) in slots.iter().enumerate() {
            let (xv, yv) = values[k];
            let diff = xv - yv;
            // Reference values exactly as the original per-slot generator
            // computed them.
            let expected_inv = if diff == F::ZERO {
                F::ZERO
            } else {
                diff.inverse()
            };
            let expected_prod = if diff == F::ZERO { F::ZERO } else { F::ONE };
            let expected_equal = if diff == F::ZERO { F::ONE } else { F::ZERO };

            let got_diff = witness.get_target(Target::wire(row, gate.wire_ith_temporary(i, 0)));
            let got_inv = witness.get_target(Target::wire(row, gate.wire_ith_temporary(i, 1)));
            let got_prod = witness.get_target(Target::wire(row, gate.wire_ith_temporary(i, 2)));
            let got_equal = witness.get_target(Target::wire(row, gate.wire_ith_output(i)));

            assert_eq!(got_diff, diff, "slot {k}: diff mismatch");
            assert_eq!(got_inv, expected_inv, "slot {k}: invdiff mismatch");
            assert_eq!(
                got_inv.to_canonical_u64(),
                expected_inv.to_canonical_u64(),
                "slot {k}: invdiff bit pattern mismatch"
            );
            assert_eq!(got_prod, expected_prod, "slot {k}: prod mismatch");
            assert_eq!(got_equal, expected_equal, "slot {k}: equal mismatch");
            if diff != F::ZERO {
                assert_eq!(got_inv * diff, F::ONE, "slot {k}: not a true inverse");
            }
        }

        Ok(())
    }

    #[test]
    fn row_inverse_generator_proves_mixed_rows() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let gate = EqualityGate::new_from_config(&config);

        let n = gate.num_ops + 2;
        let mut inputs = Vec::new();
        for _ in 0..n {
            let x = builder.add_virtual_target();
            let y = builder.add_virtual_target();
            let (row, i) = builder.find_slot(gate.clone(), &[F::ONE], &[F::ONE]);
            builder.connect(x, Target::wire(row, gate.wire_ith_element_0(i)));
            builder.connect(y, Target::wire(row, gate.wire_ith_element_1(i)));
            builder.assert_bool(BoolTarget::new_unsafe(Target::wire(
                row,
                gate.wire_ith_output(i),
            )));
            inputs.push((x, y));
        }
        let circuit_data = builder.build::<C>();

        let mut pw = PartialWitness::new();
        for (k, &(x, y)) in inputs.iter().enumerate() {
            let xv = F::from_canonical_u64(k as u64);
            let yv = if k % 3 == 0 {
                xv
            } else {
                F::from_canonical_u64(1000 + k as u64)
            };
            pw.set_target(x, xv)?;
            pw.set_target(y, yv)?;
        }
        let proof = circuit_data.prove(pw)?;
        circuit_data.verify(proof)
    }
}
