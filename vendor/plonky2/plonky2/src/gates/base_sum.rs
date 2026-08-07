#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};
use core::ops::Range;

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::field::types::{Field, Field64};
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
use crate::plonk::plonk_common::{reduce_with_powers, reduce_with_powers_ext_circuit};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::log_floor;
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate which can decompose a number into base B little-endian limbs.
#[derive(Copy, Clone, Debug)]
pub struct BaseSumGate<const B: usize> {
    pub num_limbs: usize,
}

impl<const B: usize> BaseSumGate<B> {
    pub const fn new(num_limbs: usize) -> Self {
        Self { num_limbs }
    }

    pub fn new_from_config<F: Field64>(config: &CircuitConfig) -> Self {
        let num_limbs =
            log_floor(F::ORDER - 1, B as u64).min(config.num_routed_wires - Self::START_LIMBS);
        Self::new(num_limbs)
    }

    pub(crate) const WIRE_SUM: usize = 0;
    pub(crate) const START_LIMBS: usize = 1;

    /// Returns the index of the `i`th limb wire.
    pub(crate) const fn limbs(&self) -> Range<usize> {
        Self::START_LIMBS..Self::START_LIMBS + self.num_limbs
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const B: usize> Gate<F, D> for BaseSumGate<B> {
    fn id(&self) -> String {
        format!("{self:?} + Base: {B}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_limbs)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_limbs = src.read_usize()?;
        Ok(Self { num_limbs })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let sum = vars.local_wires[Self::WIRE_SUM];
        let limbs = vars.local_wires[self.limbs()].to_vec();
        let computed_sum = reduce_with_powers(&limbs, F::Extension::from_canonical_usize(B));
        let mut constraints = vec![computed_sum - sum];
        for limb in limbs {
            constraints.push(
                (0..B)
                    .map(|i| limb - F::Extension::from_canonical_usize(i))
                    .product(),
            );
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
        let base = builder.constant(F::from_canonical_usize(B));
        let sum = vars.local_wires[Self::WIRE_SUM];
        let limbs = vars.local_wires[self.limbs()].to_vec();
        let computed_sum = reduce_with_powers_ext_circuit(builder, &limbs, base);
        let mut constraints = vec![builder.sub_extension(computed_sum, sum)];
        for limb in limbs {
            constraints.push({
                let mut acc = builder.one_extension();
                (0..B).for_each(|i| {
                    // We update our accumulator as:
                    // acc' = acc (x - i)
                    //      = acc x + (-i) acc
                    // Since -i is constant, we can do this in one arithmetic_extension call.
                    let neg_i = -F::from_canonical_usize(i);
                    acc = builder.arithmetic_extension(F::ONE, neg_i, acc, limb, acc)
                });
                acc
            });
        }
        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = BaseSplitGenerator::<B> {
            row,
            num_limbs: self.num_limbs,
        };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    // 1 for the sum then `num_limbs` for the limbs.
    fn num_wires(&self) -> usize {
        1 + self.num_limbs
    }

    fn num_constants(&self) -> usize {
        0
    }

    // Bounded by the range-check (x-0)*(x-1)*...*(x-B+1).
    fn degree(&self) -> usize {
        B
    }

    // 1 for checking the sum then `num_limbs` for range-checking the limbs.
    fn num_constraints(&self) -> usize {
        1 + self.num_limbs
    }
}

impl<F: RichField + Extendable<D>, const D: usize, const B: usize> PackedEvaluableBase<F, D>
    for BaseSumGate<B>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let sum = vars.local_wires[Self::WIRE_SUM];
        let limbs = vars.local_wires.view(self.limbs());

        if B == 2 {
            // Production instantiation. Same terms, same order, same values as the
            // generic path below: `reduce_with_powers` is Horner from the most
            // significant limb down, and doubling is `x + x` in any field; the
            // per-limb range product `prod_{i<2}(limb - i)` is `limb * (limb - 1)`.
            // Written out so the generic base-`B` product loop and its
            // `from_canonical_usize` per term disappear for the case that runs.
            let mut computed_sum = P::ZEROS;
            for &limb in limbs.iter().rev() {
                computed_sum = computed_sum + computed_sum + limb;
            }
            yield_constr.one(computed_sum - sum);

            let constraints_iter = limbs.iter().map(|&limb| limb * (limb - F::ONE));
            yield_constr.many(constraints_iter);
        } else {
            let computed_sum = reduce_with_powers(limbs, F::from_canonical_usize(B));

            yield_constr.one(computed_sum - sum);

            let constraints_iter = limbs.iter().map(|&limb| {
                (0..B)
                    .map(|i| limb - F::from_canonical_usize(i))
                    .product::<P>()
            });
            yield_constr.many(constraints_iter);
        }
    }
}

#[derive(Debug, Default)]
pub struct BaseSplitGenerator<const B: usize> {
    row: usize,
    num_limbs: usize,
}

impl<F: RichField + Extendable<D>, const B: usize, const D: usize> SimpleGenerator<F, D>
    for BaseSplitGenerator<B>
{
    fn id(&self) -> String {
        format!("BaseSplitGenerator + Base: {B}")
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![Target::wire(self.row, BaseSumGate::<B>::WIRE_SUM)]
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let sum_value = witness
            .get_target(Target::wire(self.row, BaseSumGate::<B>::WIRE_SUM))
            .to_canonical_u64();
        debug_assert_eq!(
            (0..self.num_limbs).fold(sum_value, |acc, _| acc / (B as u64)),
            0,
            "Integer too large to fit in given number of limbs"
        );

        let limbs = (BaseSumGate::<B>::START_LIMBS..BaseSumGate::<B>::START_LIMBS + self.num_limbs)
            .map(|i| Target::wire(self.row, i));
        let limbs_value = (0..self.num_limbs).scan(sum_value, |acc, _| {
            let tmp = *acc % (B as u64);
            *acc /= B as u64;
            Some(F::from_canonical_u64(tmp))
        });

        for (b, b_value) in limbs.zip(limbs_value) {
            out_buffer.set_target(b, b_value)?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.num_limbs)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let num_limbs = src.read_usize()?;
        Ok(Self { row, num_limbs })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::base_sum::BaseSumGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(BaseSumGate::<6>::new(11))
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(BaseSumGate::<6>::new(11))
    }

    /// Manual timing harness for the packed accumulate path of the production
    /// `B = 2` gate, modeled on `exp_accumulate_micro` in `gates/exponentiation.rs`.
    /// Run with:
    /// `cargo test --release -p plonky2 base_sum_accumulate_micro -- --ignored --nocapture`
    #[test]
    #[ignore = "manual timing harness"]
    fn base_sum_accumulate_microbenchmark() {
        use core::hint::black_box;
        use std::time::Instant;

        use plonky2_field::types::{Field, Sample};

        use crate::gates::gate::Gate;
        use crate::plonk::vars::EvaluationVarsBaseBatch;

        const D: usize = 2;
        type F = GoldilocksField;
        let gate = BaseSumGate::<2>::new(32);
        let n = 32;
        let wires = F::rand_vec(<BaseSumGate<2> as Gate<F, D>>::num_wires(&gate) * n);
        let constants: Vec<F> = Vec::new();
        let hash = crate::hash::hash_types::HashOut::ZERO;
        let filters = F::rand_vec(n);
        let nc = <BaseSumGate<2> as Gate<F, D>>::num_constraints(&gate);
        let mut combined = vec![F::ZERO; nc * n];
        let iters = 50_000u32;
        let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &hash);

        let mut t = 0.0f64;
        for _ in 0..4 {
            let s = Instant::now();
            for _ in 0..iters {
                <BaseSumGate<2> as Gate<F, D>>::eval_unfiltered_base_batch_accumulate(
                    &gate, vars, &filters, black_box(&mut combined),
                );
            }
            t += s.elapsed().as_secs_f64();
        }
        println!(
            "base_sum B=2 accumulate per batch (n=32): {:.3} us",
            t / (4.0 * iters as f64) * 1e6
        );
    }
}
