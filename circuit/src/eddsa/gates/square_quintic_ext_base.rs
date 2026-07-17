// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use anyhow::Result;
use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::gates::gate::Gate;
use plonky2::gates::packed_util::PackedEvaluableBase;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};

use crate::plonky2::util::serialization::{Buffer, IoResult, Read, Write};

#[derive(Debug, Clone, Default)]
pub struct QuinticSquaringGate {
    /// Number of Quintic Squarings performed by a Gate
    pub num_ops: usize,
}

impl QuinticSquaringGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }
    //Number of routed wires necessary for an operation
    const ROUTED_PER_OP: usize = 10;
    const NOT_ROUTED_PER_OP: usize = 10;
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

    pub(crate) const fn wire_ith_multiplicand_jth_limb(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < 5);
        Self::ROUTED_PER_OP * i + j
    }
    pub(crate) const fn wire_ith_output_jth_limb(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < 5);
        Self::ROUTED_PER_OP * i + 5 + j
    }
    pub(crate) const fn temporary_wire(&self, i: usize, j: usize) -> usize {
        assert!(i < self.num_ops);
        assert!(j < 10);
        Self::ROUTED_PER_OP * self.num_ops + i * Self::NOT_ROUTED_PER_OP + j
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for QuinticSquaringGate {
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
        let const_2 = F::Extension::from_basefield(F::from_canonical_u64(2));
        let const_3 = F::Extension::from_basefield(F::from_canonical_u64(3));
        let const_6 = F::Extension::from_basefield(F::from_canonical_u64(6));
        let mut constraints = Vec::with_capacity(self.num_ops * 15);

        for i in 0..self.num_ops {
            let a = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_multiplicand_jth_limb(i, j)])
                .collect::<Vec<_>>();
            let c = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_output_jth_limb(i, j)])
                .collect::<Vec<_>>();

            // Compute each output limb (copied from mul_quintic_ext structure)
            let extra = (0..10)
                .map(|j| vars.local_wires[self.temporary_wire(i, j)])
                .collect::<Vec<_>>();

            //c[0]
            constraints.push(a[0] * a[0] - extra[0]);
            constraints.push((const_6 * a[1] * a[4] + extra[0]) - extra[1]);
            constraints.push((const_6 * a[2] * a[3] + extra[1]) - c[0]);

            //c[1]
            constraints.push(const_3 * a[3] * a[3] - extra[2]);
            constraints.push((const_2 * a[0] * a[1] + extra[2]) - extra[3]);
            constraints.push((const_6 * a[2] * a[4] + extra[3]) - c[1]);

            //c[2]
            constraints.push(a[1] * a[1] - extra[4]);
            constraints.push((const_2 * a[0] * a[2] + extra[4]) - extra[5]);
            constraints.push((const_6 * a[3] * a[4] + extra[5]) - c[2]);

            //c[3]
            constraints.push((const_3 * a[4] * a[4]) - extra[6]);
            constraints.push((const_2 * a[0] * a[3] + extra[6]) - extra[7]);
            constraints.push((const_2 * a[1] * a[2] + extra[7]) - c[3]);

            //c[4]
            constraints.push(a[2] * a[2] - extra[8]);
            constraints.push((const_2 * a[0] * a[4] + extra[8]) - extra[9]);
            constraints.push((const_2 * a[1] * a[3] + extra[9]) - c[4]);
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
        let const_0 = F::from_canonical_u64(0);
        let const_1 = F::from_canonical_u64(1);
        let const_2 = F::from_canonical_u64(2);
        let const_3 = F::from_canonical_u64(3);
        let const_6 = F::from_canonical_u64(6);

        let mut constraints = Vec::with_capacity(self.num_ops * 24); // 24 intermediate constraints

        for i in 0..self.num_ops {
            let a = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_multiplicand_jth_limb(i, j)])
                .collect::<Vec<_>>();
            let out = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_output_jth_limb(i, j)])
                .collect::<Vec<_>>();
            let extra = (0..10)
                .map(|j| vars.local_wires[self.temporary_wire(i, j)])
                .collect::<Vec<_>>();

            let [a0, a1, a2, a3, a4] = <[ExtensionTarget<D>; 5]>::try_from(a).unwrap();
            let [c0, c1, c2, c3, c4] = <[ExtensionTarget<D>; 5]>::try_from(out).unwrap();

            // --- c[0] ---
            let t0 = builder.mul_extension(a0, a0);
            constraints.push(builder.sub_extension(t0, extra[0]));
            let t1 = builder.arithmetic_extension(const_6, const_1, a1, a4, extra[0]);
            constraints.push(builder.sub_extension(t1, extra[1]));
            let t2 = builder.arithmetic_extension(const_6, const_1, a2, a3, extra[1]);
            constraints.push(builder.sub_extension(t2, c0));

            // --- c[1] ---
            let t4 = builder.arithmetic_extension(const_3, const_0, a3, a3, a3);
            constraints.push(builder.sub_extension(t4, extra[2]));
            let t6 = builder.arithmetic_extension(const_2, const_1, a0, a1, extra[2]);
            constraints.push(builder.sub_extension(t6, extra[3]));
            let t8 = builder.arithmetic_extension(const_6, const_1, a2, a4, extra[3]);
            constraints.push(builder.sub_extension(t8, c1));

            // --- c[2] ---
            let t9 = builder.mul_extension(a1, a1);
            constraints.push(builder.sub_extension(t9, extra[4]));
            let t11 = builder.arithmetic_extension(const_2, const_1, a0, a2, extra[4]);
            constraints.push(builder.sub_extension(t11, extra[5]));
            let t13 = builder.arithmetic_extension(const_6, const_1, a3, a4, extra[5]);
            constraints.push(builder.sub_extension(t13, c2));

            // --- c[3] ---
            let t15 = builder.arithmetic_extension(const_3, const_0, a4, a4, a4);
            constraints.push(builder.sub_extension(t15, extra[6]));
            let t17 = builder.arithmetic_extension(const_2, const_1, a0, a3, extra[6]);
            constraints.push(builder.sub_extension(t17, extra[7]));
            let t19 = builder.arithmetic_extension(const_2, const_1, a1, a2, extra[7]);
            constraints.push(builder.sub_extension(t19, c3));

            // --- c[4] ---
            let t20 = builder.mul_extension(a2, a2);
            constraints.push(builder.sub_extension(t20, extra[8]));
            let t22 = builder.arithmetic_extension(const_2, const_1, a0, a4, extra[8]);
            constraints.push(builder.sub_extension(t22, extra[9]));
            let t24 = builder.arithmetic_extension(const_2, const_1, a1, a3, extra[9]);
            constraints.push(builder.sub_extension(t24, c4));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    QuinticSquaringBaseGenerator {
                        gate: self.clone(),
                        row,
                        const_2: F::from_canonical_u64(2),
                        const_3: F::from_canonical_u64(3),
                        const_6: F::from_canonical_u64(6),
                        i,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * Self::TOTAL_PER_OP
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * 15
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D>
    for QuinticSquaringGate
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let const_2 = P::from(F::from_canonical_u64(2));
        let const_3 = P::from(F::from_canonical_u64(3));
        let const_6 = P::from(F::from_canonical_u64(6));

        for i in 0..self.num_ops {
            let a = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_multiplicand_jth_limb(i, j)])
                .collect::<Vec<_>>();
            let c = (0..5)
                .map(|j| vars.local_wires[self.wire_ith_output_jth_limb(i, j)])
                .collect::<Vec<_>>();
            let extra = (0..10)
                .map(|j| vars.local_wires[self.temporary_wire(i, j)])
                .collect::<Vec<_>>();

            //c[0]
            yield_constr.one(a[0] * a[0] - extra[0]);
            yield_constr.one((const_6 * a[1] * a[4] + extra[0]) - extra[1]);
            yield_constr.one((const_6 * a[2] * a[3] + extra[1]) - c[0]);

            //c[1]
            yield_constr.one(const_3 * a[3] * a[3] - extra[2]);
            yield_constr.one((const_2 * a[0] * a[1] + extra[2]) - extra[3]);
            yield_constr.one((const_6 * a[2] * a[4] + extra[3]) - c[1]);

            //c[2]
            yield_constr.one(a[1] * a[1] - extra[4]);
            yield_constr.one((const_2 * a[0] * a[2] + extra[4]) - extra[5]);
            yield_constr.one((const_6 * a[3] * a[4] + extra[5]) - c[2]);

            //c[3]
            yield_constr.one((const_3 * a[4] * a[4]) - extra[6]);
            yield_constr.one((const_2 * a[0] * a[3] + extra[6]) - extra[7]);
            yield_constr.one((const_2 * a[1] * a[2] + extra[7]) - c[3]);

            //c[4]
            yield_constr.one(a[2] * a[2] - extra[8]);
            yield_constr.one((const_2 * a[0] * a[4] + extra[8]) - extra[9]);
            yield_constr.one((const_2 * a[1] * a[3] + extra[9]) - c[4]);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct QuinticSquaringBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    gate: QuinticSquaringGate,
    row: usize,
    const_2: F,
    const_3: F,
    const_6: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for QuinticSquaringBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "QuinticSquaringBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        [
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 0),
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 1),
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 2),
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 3),
            self.gate.wire_ith_multiplicand_jth_limb(self.i, 4),
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
        let const_2 = self.const_2;
        let const_3 = self.const_3;
        let const_6 = self.const_6;

        let a = (0..5)
            .map(|j| {
                witness.get_target(Target::wire(
                    self.row,
                    self.gate.wire_ith_multiplicand_jth_limb(self.i, j),
                ))
            })
            .collect::<Vec<_>>();
        let mut extra = [F::ZERO; 10];

        // c[0]
        extra[0] = a[0] * a[0];
        extra[1] = const_6 * a[1] * a[4] + extra[0];
        let c0 = const_6 * a[2] * a[3] + extra[1];

        // c[1]
        extra[2] = const_3 * a[3] * a[3];
        extra[3] = const_2 * a[0] * a[1] + extra[2];
        let c1 = const_6 * a[2] * a[4] + extra[3];

        // c[2]
        extra[4] = a[1] * a[1];
        extra[5] = const_2 * a[0] * a[2] + extra[4];
        let c2 = const_6 * a[3] * a[4] + extra[5];

        // c[3]
        extra[6] = const_3 * a[4] * a[4];
        extra[7] = const_2 * a[0] * a[3] + extra[6];
        let c3 = const_2 * a[1] * a[2] + extra[7];

        // c[4]
        extra[8] = a[2] * a[2];
        extra[9] = const_2 * a[0] * a[4] + extra[8];
        let c4 = const_2 * a[1] * a[3] + extra[9];

        // Set outputs
        for j in 0..5 {
            out_buffer.set_target(
                Target::wire(self.row, self.gate.wire_ith_output_jth_limb(self.i, j)),
                match j {
                    0 => c0,
                    1 => c1,
                    2 => c2,
                    3 => c3,
                    4 => c4,
                    _ => unreachable!(),
                },
            )?;
        }

        // Set extra/intermediate wires
        for j in 0..10 {
            out_buffer.set_target(
                Target::wire(self.row, self.gate.temporary_wire(self.i, j)),
                extra[j],
            )?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.gate.serialize(dst, common_data)?;
        dst.write_usize(self.row)?;
        dst.write_field(self.const_2)?;
        dst.write_field(self.const_3)?;
        dst.write_field(self.const_6)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gate = QuinticSquaringGate::deserialize(src, common_data)?;
        let row = src.read_usize()?;
        let const_2 = src.read_field()?;
        let const_3 = src.read_field()?;
        let const_6 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            gate,
            row,
            const_2,
            const_3,
            const_6,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::eddsa::gates::square_quintic_ext_base::QuinticSquaringGate;
    use crate::plonky2::field::goldilocks_field::GoldilocksField;
    use crate::plonky2::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonky2::plonk::circuit_data::CircuitConfig;
    use crate::plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate =
            QuinticSquaringGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate =
            QuinticSquaringGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }
}
