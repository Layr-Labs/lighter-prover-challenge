//! Implementation of a Plonky2 gate for an entire Poseidon2 permutation over a
//! state of width 12
use core::marker::PhantomData;

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon2::config::*;
use crate::hash::poseidon2::hash::Poseidon2;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Evaluates a full Poseidon2 permutation with 12 state elements.
///
/// This also has some extra features to make it suitable for efficiently
/// verifying Merkle proofs. It has a flag which can be used to swap the first
/// four inputs with the next four, for ordering sibling digests.
#[derive(Debug, Default)]
pub struct Poseidon2Gate<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Poseidon2Gate<F, D> {
    pub fn new() -> Self {
        Poseidon2Gate {
            _phantom: PhantomData,
        }
    }

    /// The wire index for the `i`th input to the permutation.
    pub fn wire_input(i: usize) -> usize {
        i
    }

    /// The wire index for the `i`th output to the permutation.
    pub fn wire_output(i: usize) -> usize {
        WIDTH + i
    }

    /// If this is set to 1, the first four inputs will be swapped with the next
    /// four inputs. This is useful for ordering hashes in Merkle proofs.
    /// Otherwise, this should be set to 0.
    pub const WIRE_SWAP: usize = 2 * WIDTH;

    const START_DELTA: usize = 2 * WIDTH + 1;

    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute
    /// the swapped inputs.
    fn wire_delta(i: usize) -> usize {
        assert!(i < 4);
        Self::START_DELTA + i
    }

    const START_ROUND_F_BEGIN: usize = Self::START_DELTA + 4;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th
    /// round of the first set of full rounds.
    fn wire_full_sbox_0(round: usize, i: usize) -> usize {
        debug_assert!(
            round != 0,
            "First round S-box inputs are not stored as wires"
        );
        debug_assert!(round < ROUNDS_F_HALF);
        debug_assert!(i < WIDTH);
        Self::START_ROUND_F_BEGIN + WIDTH * (round - 1) + i
    }

    const START_PARTIAL: usize = Self::START_ROUND_F_BEGIN + WIDTH * (ROUNDS_F_HALF - 1);

    /// A wire which stores the input of the S-box of the `round`-th round of
    /// the partial rounds.
    const fn wire_partial_sbox(round: usize) -> usize {
        debug_assert!(round < ROUNDS_P);
        Self::START_PARTIAL + round
    }

    const START_ROUND_F_END: usize = Self::START_PARTIAL + ROUNDS_P;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th
    /// round of the second set of full rounds.
    const fn wire_full_sbox_1(round: usize, i: usize) -> usize {
        debug_assert!(round < ROUNDS_F_HALF);
        debug_assert!(i < WIDTH);
        Self::START_ROUND_F_END + WIDTH * round + i
    }

    /// End of wire indices, exclusive.
    const fn end() -> usize {
        Self::START_ROUND_F_END + WIDTH * ROUNDS_F_HALF
    }
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Gate<F, D> for Poseidon2Gate<F, D> {
    fn id(&self) -> String {
        format!("{:?}<WIDTH={}>", self, WIDTH)
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Poseidon2Gate::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(swap * (swap - F::Extension::ONE));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            constraints.push(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::Extension::ZERO; WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // The initial linear layer.
        <F as Poseidon2>::external_linear_layer_extension(&mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            <F as Poseidon2>::add_rc_extension(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon2>::sbox_extension(&mut state);
            <F as Poseidon2>::external_linear_layer_extension(&mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            state[0] += F::Extension::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = <F as Poseidon2>::sbox_p_extension(&state[0]);
            <F as Poseidon2>::internal_linear_layer_extension(&mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            <F as Poseidon2>::add_rc_extension(&mut state, r);
            for i in 0..WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i)];
                constraints.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            <F as Poseidon2>::sbox_extension(&mut state);
            <F as Poseidon2>::external_linear_layer_extension(&mut state);
        }

        for i in 0..WIDTH {
            constraints.push(state[i] - vars.local_wires[Self::wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_batch(
        &self,
        vars_base: crate::plonk::vars::EvaluationVarsBaseBatch<F>,
    ) -> Vec<F> {
        let n = vars_base.len();
        let wires = vars_base.local_wires;
        let col = |w: usize| &wires[w * n..][..n];
        let mut res = vec![F::ZERO; n * self.num_constraints()];
        let mut chunks = res.chunks_exact_mut(n);

        // Per-point state rows, contiguous per point so the existing scalar
        // Poseidon2 round helpers apply unchanged; wire reads and constraint
        // writes are contiguous columns.
        let mut states = vec![[F::ZERO; WIDTH]; n];

        // Assert that `swap` is binary.
        let swap = col(Self::WIRE_SWAP);
        let out = chunks.next().unwrap();
        for p in 0..n {
            out[p] = swap[p] * swap[p].sub_one();
        }

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = col(Self::wire_input(i));
            let input_rhs = col(Self::wire_input(i + 4));
            let delta_i = col(Self::wire_delta(i));
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = swap[p] * (input_rhs[p] - input_lhs[p]) - delta_i[p];
            }
        }

        // Compute the possibly-swapped input layer.
        for i in 0..4 {
            let delta_i = col(Self::wire_delta(i));
            let input_lhs = col(Self::wire_input(i));
            let input_rhs = col(Self::wire_input(i + 4));
            for p in 0..n {
                states[p][i] = input_lhs[p] + delta_i[p];
                states[p][i + 4] = input_rhs[p] - delta_i[p];
            }
        }
        for i in 8..WIDTH {
            let input = col(Self::wire_input(i));
            for p in 0..n {
                states[p][i] = input[p];
            }
        }

        // The initial linear layer.
        for state in states.iter_mut() {
            <F as Poseidon2>::external_linear_layer(state);
        }

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            for state in states.iter_mut() {
                <F as Poseidon2>::add_rc(state, r);
            }
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = col(Self::wire_full_sbox_0(r, i));
                    let out = chunks.next().unwrap();
                    for p in 0..n {
                        out[p] = states[p][i] - sbox_in[p];
                        states[p][i] = sbox_in[p];
                    }
                }
            }
            for state in states.iter_mut() {
                <F as Poseidon2>::sbox(state);
                <F as Poseidon2>::external_linear_layer(state);
            }
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            let rc = F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let sbox_in = col(Self::wire_partial_sbox(r));
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = states[p][0] + rc - sbox_in[p];
                states[p][0] = <F as Poseidon2>::sbox_p(&sbox_in[p]);
            }
            for state in states.iter_mut() {
                <F as Poseidon2>::internal_linear_layer(state);
            }
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            for state in states.iter_mut() {
                <F as Poseidon2>::add_rc(state, r);
            }
            for i in 0..WIDTH {
                let sbox_in = col(Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i));
                let out = chunks.next().unwrap();
                for p in 0..n {
                    out[p] = states[p][i] - sbox_in[p];
                    states[p][i] = sbox_in[p];
                }
            }
            for state in states.iter_mut() {
                <F as Poseidon2>::sbox(state);
                <F as Poseidon2>::external_linear_layer(state);
            }
        }

        for i in 0..WIDTH {
            let output = col(Self::wire_output(i));
            let out = chunks.next().unwrap();
            for p in 0..n {
                out[p] = states[p][i] - output[p];
            }
        }

        res
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * swap.sub_one());

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::ZERO; WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // The initial linear layer.
        <F as Poseidon2>::external_linear_layer(&mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            <F as Poseidon2>::add_rc(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon2>::sbox(&mut state);
            <F as Poseidon2>::external_linear_layer(&mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            state[0] += F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = <F as Poseidon2>::sbox_p(&state[0]);
            <F as Poseidon2>::internal_linear_layer(&mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            <F as Poseidon2>::add_rc(&mut state, r);
            for i in 0..WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            <F as Poseidon2>::sbox(&mut state);
            <F as Poseidon2>::external_linear_layer(&mut state);
        }

        for i in 0..WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(builder.mul_sub_extension(swap, swap, swap));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let diff = builder.sub_extension(input_rhs, input_lhs);
            constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
        }

        // Compute the possibly-swapped input layer.
        let mut state = [builder.zero_extension(); WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = builder.add_extension(input_lhs, delta_i);
            state[i + 4] = builder.sub_extension(input_rhs, delta_i);
        }
        for i in 8..WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // The initial linear layer.
        <F as Poseidon2>::external_linear_layer_circuit(builder, &mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            <F as Poseidon2>::add_rc_circuit(builder, &mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(builder.sub_extension(state[i], sbox_in));
                    state[i] = sbox_in;
                }
            }
            <F as Poseidon2>::sbox_circuit(builder, &mut state);
            <F as Poseidon2>::external_linear_layer_circuit(builder, &mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            let round_constant = F::Extension::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let round_constant = builder.constant_extension(round_constant);
            state[0] = builder.add_extension(state[0], round_constant);

            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(builder.sub_extension(state[0], sbox_in));
            state[0] = sbox_in;
            state[0] = <F as Poseidon2>::sbox_p_circuit(builder, state[0]);
            <F as Poseidon2>::internal_linear_layer_circuit(builder, &mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            <F as Poseidon2>::add_rc_circuit(builder, &mut state, r);

            for i in 0..WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i)];
                constraints.push(builder.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }
            <F as Poseidon2>::sbox_circuit(builder, &mut state);
            <F as Poseidon2>::external_linear_layer_circuit(builder, &mut state);
        }

        for i in 0..WIDTH {
            constraints
                .push(builder.sub_extension(state[i], vars.local_wires[Self::wire_output(i)]));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let g = Poseidon2Generator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![WitnessGeneratorRef::new(g.adapter())]
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        7
    }

    fn num_constraints(&self) -> usize {
        WIDTH * (ROUNDS_F - 1) + ROUNDS_P + WIDTH + 1 + 4
    }
}

#[derive(Debug, Default)]
pub struct Poseidon2Generator<F: RichField + Extendable<D> + Poseidon2, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

#[cfg(test)]
static POSEIDON2_GENERATOR_BATCHES: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Poseidon2Generator<F, D> {
    fn run_rows<const N: usize>(
        rows: [usize; N],
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        const GENERATED_VALUES_COUNT: usize =
            4 + WIDTH * (ROUNDS_F_HALF - 1) + ROUNDS_P + WIDTH * ROUNDS_F_HALF + WIDTH;

        let local_wire = |row, column| Wire { row, column };
        let mut states: [[F; WIDTH]; N] = core::array::from_fn(|j| {
            core::array::from_fn(|i| {
                witness.get_wire(local_wire(rows[j], Poseidon2Gate::<F, D>::wire_input(i)))
            })
        });
        let swap_values: [F; N] = core::array::from_fn(|j| {
            witness.get_wire(local_wire(rows[j], Poseidon2Gate::<F, D>::WIRE_SWAP))
        });
        let mut generated = [[F::ZERO; GENERATED_VALUES_COUNT]; N];
        let mut columns = [0usize; GENERATED_VALUES_COUNT];
        let mut slot = 0;

        for i in 0..4 {
            columns[slot + i] = Poseidon2Gate::<F, D>::wire_delta(i);
            for j in 0..N {
                debug_assert!(swap_values[j] == F::ZERO || swap_values[j] == F::ONE);
                generated[j][slot + i] = swap_values[j] * (states[j][i + 4] - states[j][i]);
            }
        }
        slot += 4;

        for j in 0..N {
            if swap_values[j] == F::ONE {
                for i in 0..4 {
                    states[j].swap(i, 4 + i);
                }
            }
            <F as Poseidon2>::external_linear_layer(&mut states[j]);
        }

        for r in 0..ROUNDS_F_HALF {
            for state in &mut states {
                <F as Poseidon2>::add_rc(state, r);
            }
            if r != 0 {
                for i in 0..WIDTH {
                    columns[slot + i] = Poseidon2Gate::<F, D>::wire_full_sbox_0(r, i);
                    for j in 0..N {
                        generated[j][slot + i] = states[j][i];
                    }
                }
                slot += WIDTH;
            }
            for state in &mut states {
                <F as Poseidon2>::sbox(state);
            }
            for state in &mut states {
                <F as Poseidon2>::external_linear_layer(state);
            }
        }

        for r in 0..ROUNDS_P {
            columns[slot] = Poseidon2Gate::<F, D>::wire_partial_sbox(r);
            for j in 0..N {
                states[j][0] += F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
                generated[j][slot] = states[j][0];
            }
            slot += 1;
            for state in &mut states {
                state[0] = <F as Poseidon2>::sbox_p(&state[0]);
            }
            for state in &mut states {
                <F as Poseidon2>::internal_linear_layer(state);
            }
        }

        for r in ROUNDS_F_HALF..ROUNDS_F {
            for state in &mut states {
                <F as Poseidon2>::add_rc(state, r);
            }
            for i in 0..WIDTH {
                columns[slot + i] = Poseidon2Gate::<F, D>::wire_full_sbox_1(r - ROUNDS_F_HALF, i);
                for j in 0..N {
                    generated[j][slot + i] = states[j][i];
                }
            }
            slot += WIDTH;
            for state in &mut states {
                <F as Poseidon2>::sbox(state);
            }
            for state in &mut states {
                <F as Poseidon2>::external_linear_layer(state);
            }
        }

        for i in 0..WIDTH {
            columns[slot + i] = Poseidon2Gate::<F, D>::wire_output(i);
            for j in 0..N {
                generated[j][slot + i] = states[j][i];
            }
        }
        slot += WIDTH;
        debug_assert_eq!(slot, GENERATED_VALUES_COUNT);

        for j in 0..N {
            for i in 0..GENERATED_VALUES_COUNT {
                out_buffer.set_wire(local_wire(rows[j], columns[i]), generated[j][i])?;
            }
        }
        Ok(())
    }
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> SimpleGenerator<F, D>
    for Poseidon2Generator<F, D>
{
    fn id(&self) -> String {
        "Poseidon2Generator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..WIDTH)
            .map(|i| Poseidon2Gate::<F, D>::wire_input(i))
            .chain(Some(Poseidon2Gate::<F, D>::WIRE_SWAP))
            .map(|column| Target::wire(self.row, column))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        Self::run_rows([self.row], witness, out_buffer)
    }

    fn batch_hint(&self) -> Option<(&'static str, usize)> {
        Some(("Poseidon2Generator", self.row))
    }

    fn run_ready_batch(
        &self,
        items: &[usize],
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Option<Result<()>> {
        #[cfg(test)]
        POSEIDON2_GENERATOR_BATCHES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        match items {
            &[a, b] => Some(Self::run_rows([a, b], witness, out_buffer)),
            &[a, b, c, d] => Some(Self::run_rows([a, b, c, d], witness, out_buffer)),
            _ => None,
        }
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self {
            row,
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::{Poseidon2Gate, *};
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::poseidon::PoseidonGate;
    use crate::iop::generator::generate_partial_witness;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};

    #[test]
    fn wire_indices() {
        type F = GoldilocksField;
        type Gate = Poseidon2Gate<F, 4>;

        assert_eq!(Gate::wire_input(0), 0);
        assert_eq!(Gate::wire_input(11), 11);
        assert_eq!(Gate::wire_output(0), 12);
        assert_eq!(Gate::wire_output(11), 23);
        assert_eq!(Gate::WIRE_SWAP, 24);
        assert_eq!(Gate::wire_delta(0), 25);
        assert_eq!(Gate::wire_delta(3), 28);
        assert_eq!(Gate::wire_full_sbox_0(1, 0), 29);
        assert_eq!(Gate::wire_full_sbox_0(3, 0), 53);
        assert_eq!(Gate::wire_full_sbox_0(3, 11), 64);
        assert_eq!(Gate::wire_partial_sbox(0), 65);
        assert_eq!(Gate::wire_partial_sbox(21), 86);
        assert_eq!(Gate::wire_full_sbox_1(0, 0), 87);
        assert_eq!(Gate::wire_full_sbox_1(3, 0), 123);
        assert_eq!(Gate::wire_full_sbox_1(3, 11), 134);
    }

    #[test]
    fn generated_output() {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig {
            num_wires: 143,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::new(config);
        type Gate = Poseidon2Gate<F, D>;
        let gate = Gate::new();
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build_prover::<C>();

        let permutation_inputs = (0..WIDTH).map(F::from_canonical_usize).collect::<Vec<_>>();

        let mut inputs = PartialWitness::new();
        inputs
            .set_wire(
                Wire {
                    row,
                    column: Gate::WIRE_SWAP,
                },
                F::ZERO,
            )
            .unwrap();
        for i in 0..WIDTH {
            inputs
                .set_wire(
                    Wire {
                        row,
                        column: Gate::wire_input(i),
                    },
                    permutation_inputs[i],
                )
                .unwrap();
        }

        let witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();

        let expected_outputs: [F; WIDTH] = F::poseidon2(permutation_inputs.try_into().unwrap());
        for i in 0..WIDTH {
            let out = witness.get_wire(Wire {
                row: 0,
                column: Gate::wire_output(i),
            });
            assert_eq!(out, expected_outputs[i]);
        }
    }

    #[test]
    fn generated_output_batches_four_adjacent_rows() {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type Gate = Poseidon2Gate<F, D>;

        POSEIDON2_GENERATOR_BATCHES.store(0, core::sync::atomic::Ordering::Relaxed);
        let config = CircuitConfig {
            num_wires: 143,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::new(config);
        let rows: [usize; 4] = core::array::from_fn(|_| builder.add_gate(Gate::new(), vec![]));
        let circuit = builder.build_prover::<C>();
        let mut inputs = PartialWitness::new();
        let mut expected = [[F::ZERO; WIDTH]; 4];

        for j in 0..4 {
            let swap = if j % 2 == 0 { F::ZERO } else { F::ONE };
            inputs
                .set_wire(
                    Wire {
                        row: rows[j],
                        column: Gate::WIRE_SWAP,
                    },
                    swap,
                )
                .unwrap();
            for i in 0..WIDTH {
                let value = F::from_canonical_usize(j * WIDTH + i);
                expected[j][i] = value;
                inputs
                    .set_wire(
                        Wire {
                            row: rows[j],
                            column: Gate::wire_input(i),
                        },
                        value,
                    )
                    .unwrap();
            }
            if swap == F::ONE {
                for i in 0..4 {
                    expected[j].swap(i, i + 4);
                }
            }
            expected[j] = F::poseidon2(expected[j]);
        }

        let witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();
        assert_eq!(
            POSEIDON2_GENERATOR_BATCHES.load(core::sync::atomic::Ordering::Relaxed),
            1
        );
        for j in 0..4 {
            for i in 0..WIDTH {
                assert_eq!(
                    witness.get_wire(Wire {
                        row: rows[j],
                        column: Gate::wire_output(i),
                    }),
                    expected[j][i]
                );
            }
        }
    }

    #[test]
    fn low_degree() {
        type F = GoldilocksField;
        let gate = Poseidon2Gate::<F, 4>::new();
        test_low_degree(gate);

        let gate = PoseidonGate::<F, 4>::new();
        test_low_degree(gate)
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = Poseidon2Gate::<F, D>::new();
        test_eval_fns::<F, C, _, D>(gate)?;

        let gate = PoseidonGate::<F, D>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }
}
