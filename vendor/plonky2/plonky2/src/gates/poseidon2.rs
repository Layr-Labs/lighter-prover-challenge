//! Implementation of a Plonky2 gate for an entire Poseidon2 permutation over a
//! state of width 12
use core::marker::PhantomData;

use anyhow::Result;

use crate::field::batch_util::batch_multiply_add_inplace;
use crate::field::extension::Extendable;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
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
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Packed-lane mirrors of the scalar `Poseidon2` round primitives, evaluating
/// `P::WIDTH` quotient points per call. Each helper computes exactly the same
/// field expressions as its extension-field counterpart in `Poseidon2`
/// (`add_rc_extension`, `sbox_p_extension`, `apply_mat4_mut_extension`,
/// `external_linear_layer_extension`, `internal_linear_layer_extension`),
/// lane by lane, so every produced value is field-equal to the scalar
/// evaluation of that point.
#[inline]
fn poseidon2_add_rc_packed<P: PackedField>(state: &mut [P; WIDTH], external_round: usize) {
    debug_assert!(external_round < EXTERNAL_CONSTANTS.len());
    for i in 0..WIDTH {
        state[i] += P::Scalar::from_canonical_u64(EXTERNAL_CONSTANTS[external_round][i]);
    }
}

#[inline]
fn poseidon2_sbox_p_packed<P: PackedField>(x: P) -> P {
    // x^7 as x^3 * x^4.
    let x2 = x.square();
    let x3 = x2 * x;
    let x4 = x2.square();
    x3 * x4
}

#[inline]
fn poseidon2_sbox_packed<P: PackedField>(state: &mut [P; WIDTH]) {
    for i in 0..WIDTH {
        state[i] = poseidon2_sbox_p_packed(state[i]);
    }
}

#[inline]
fn poseidon2_apply_mat4_packed<P: PackedField>(x: &mut [P; 4]) {
    let t01 = x[0] + x[1];
    let t23 = x[2] + x[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + x[1];
    let t01233 = t0123 + x[3];
    // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].
    x[3] = t01233 + x[0] + x[0]; // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = t01123 + x[2] + x[2]; // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]
}

#[inline]
fn poseidon2_external_linear_layer_packed<P: PackedField>(state: &mut [P; WIDTH]) {
    // First, we apply M_4 to each consecutive four elements of the state.
    for i in (0..WIDTH).step_by(4) {
        let mut block = [state[i], state[i + 1], state[i + 2], state[i + 3]];
        poseidon2_apply_mat4_packed(&mut block);
        state[i..i + 4].copy_from_slice(&block);
    }
    // Now, we apply the outer circulant matrix: precompute the four sums of
    // every four elements, then add the appropriate sum into each element.
    let sums: [P; 4] = core::array::from_fn(|k| {
        (0..WIDTH)
            .step_by(4)
            .map(|j| state[j + k])
            .fold(P::ZEROS, |acc, x| acc + x)
    });
    for i in 0..WIDTH {
        state[i] += sums[i % 4];
    }
}

#[inline]
fn poseidon2_internal_linear_layer_packed<P: PackedField>(state: &mut [P; WIDTH]) {
    let mut sum = P::ZEROS;
    for x in state.iter() {
        sum += *x;
    }
    for i in 0..WIDTH {
        state[i] = state[i] * P::Scalar::from_canonical_u64(MATRIX_DIAG_12_U64[i]) + sum;
    }
}

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
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_base_batch_accumulate(
        &self,
        vars_base: crate::plonk::vars::EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        // Packed direct-fold: evaluate `P::WIDTH` points per lane group with
        // the packed round primitives and fold each constraint row into the
        // shared accumulator immediately (one packed constraint-times-filter
        // multiply per group per constraint). This preserves the base's
        // zero-materialization property while vectorizing the round
        // arithmetic. Leftover points (batch size not a multiple of the
        // packing width) run the identical code through the width-1 scalar
        // packing, so every point's fold is `combined += constraint * filter`
        // in the gate's constraint order either way.
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= self.num_constraints() * n);

        let (vars_packed_iter, vars_leftovers_iter) =
            vars_base.pack::<<F as Packable>::Packing>();
        let leftovers_start = n - vars_leftovers_iter.len();
        for (i, vars_packed) in vars_packed_iter.enumerate() {
            let offset = <F as Packable>::Packing::WIDTH * i;
            self.accumulate_packed_lanes(
                vars_packed,
                filters,
                combined_gate_constraints,
                n,
                offset,
            );
        }
        for (i, vars_leftovers) in vars_leftovers_iter.enumerate() {
            let offset = leftovers_start + i;
            self.accumulate_packed_lanes(
                vars_leftovers,
                filters,
                combined_gate_constraints,
                n,
                offset,
            );
        }
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

/// Retains the pre-packing scalar column-major evaluator as a differential
/// reference for tests. Not called on any production path.
#[cfg(test)]
impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Poseidon2Gate<F, D> {
    pub(crate) fn eval_unfiltered_base_batch_accumulate_scalar(
        &self,
        vars_base: crate::plonk::vars::EvaluationVarsBaseBatch<F>,
        filters: &[F],
        combined_gate_constraints: &mut [F],
    ) {
        let n = vars_base.len();
        assert_eq!(filters.len(), n);
        assert!(combined_gate_constraints.len() >= self.num_constraints() * n);
        let wires = vars_base.local_wires;
        let col = |w: usize| &wires[w * n..][..n];

        // Batches are 32 points in this prover; keep the scratch row on the
        // stack and fall back to the heap only for oversized batches.
        let mut scratch_stack = [F::ZERO; 64];
        let mut scratch_heap;
        let scratch: &mut [F] = if n <= 64 {
            &mut scratch_stack[..n]
        } else {
            scratch_heap = vec![F::ZERO; n];
            &mut scratch_heap
        };
        let mut constraint_index = 0;
        // Mirrors `eval_unfiltered_base_batch` constraint-for-constraint; each
        // row lands in `scratch` and is folded straight into the shared
        // accumulator instead of a materialized matrix.
        macro_rules! emit {
            () => {{
                let combined = &mut combined_gate_constraints
                    [constraint_index * n..(constraint_index + 1) * n];
                batch_multiply_add_inplace(combined, &scratch, filters);
                constraint_index += 1;
            }};
        }

        let mut states = vec![[F::ZERO; WIDTH]; n];

        // Assert that `swap` is binary.
        let swap = col(Self::WIRE_SWAP);
        for p in 0..n {
            scratch[p] = swap[p] * swap[p].sub_one();
        }
        emit!();

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = col(Self::wire_input(i));
            let input_rhs = col(Self::wire_input(i + 4));
            let delta_i = col(Self::wire_delta(i));
            for p in 0..n {
                scratch[p] = swap[p] * (input_rhs[p] - input_lhs[p]) - delta_i[p];
            }
            emit!();
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
                    for p in 0..n {
                        scratch[p] = states[p][i] - sbox_in[p];
                        states[p][i] = sbox_in[p];
                    }
                    emit!();
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
            for p in 0..n {
                scratch[p] = states[p][0] + rc - sbox_in[p];
                states[p][0] = <F as Poseidon2>::sbox_p(&sbox_in[p]);
            }
            emit!();
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
                for p in 0..n {
                    scratch[p] = states[p][i] - sbox_in[p];
                    states[p][i] = sbox_in[p];
                }
                emit!();
            }
            for state in states.iter_mut() {
                <F as Poseidon2>::sbox(state);
                <F as Poseidon2>::external_linear_layer(state);
            }
        }

        for i in 0..WIDTH {
            let output = col(Self::wire_output(i));
            for p in 0..n {
                scratch[p] = states[p][i] - output[p];
            }
            emit!();
        }

        debug_assert_eq!(constraint_index, self.num_constraints());
    }

    pub(crate) fn eval_unfiltered_base_batch_scalar(
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
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Poseidon2Gate<F, D> {
    /// Runs the packed round schedule once for one lane group, passing each
    /// emitted constraint value to `sink` in the gate's constraint order
    /// (swap-binary, four deltas, first-half s-box rounds, partial rounds,
    /// second-half s-box rounds, outputs).
    #[inline]
    fn eval_packed_sink<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        sink: &mut impl FnMut(P),
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        sink(swap * swap - swap);

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            sink(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [P::ZEROS; WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = input_lhs + delta_i;
            state[i + 4] = input_rhs - delta_i;
        }
        for i in 8..WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // The initial linear layer.
        poseidon2_external_linear_layer_packed(&mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            poseidon2_add_rc_packed(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    sink(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            poseidon2_sbox_packed(&mut state);
            poseidon2_external_linear_layer_packed(&mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            state[0] += P::Scalar::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            sink(state[0] - sbox_in);
            state[0] = poseidon2_sbox_p_packed(sbox_in);
            poseidon2_internal_linear_layer_packed(&mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            poseidon2_add_rc_packed(&mut state, r);
            for i in 0..WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF, i)];
                sink(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            poseidon2_sbox_packed(&mut state);
            poseidon2_external_linear_layer_packed(&mut state);
        }

        for i in 0..WIDTH {
            sink(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    /// Folds one lane group's constraints into the combined accumulator:
    /// `combined[k * n + offset + l] += constraint_lane_l * filters[offset + l]`,
    /// the same per-element expression as `batch_multiply_add_inplace` on the
    /// scalar path.
    #[inline]
    fn accumulate_packed_lanes<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        filters: &[F],
        combined: &mut [F],
        n: usize,
        offset: usize,
    ) {
        let mut filt = P::ZEROS;
        filt.as_slice_mut()
            .copy_from_slice(&filters[offset..offset + P::WIDTH]);
        let mut k = 0usize;
        self.eval_packed_sink(vars, &mut |constraint: P| {
            let prod = constraint * filt;
            let row = k * n + offset;
            for (l, &v) in prod.as_slice().iter().enumerate() {
                combined[row + l] += v;
            }
            k += 1;
        });
    }
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> PackedEvaluableBase<F, D>
    for Poseidon2Gate<F, D>
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        self.eval_packed_sink(vars, &mut |constraint| yield_constr.one(constraint));
    }
}

#[derive(Debug, Default)]
pub struct Poseidon2Generator<F: RichField + Extendable<D> + Poseidon2, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
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
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut state: [F; WIDTH] = core::array::from_fn(|i| {
            witness.get_wire(local_wire(Poseidon2Gate::<F, D>::wire_input(i)))
        });

        let swap_value = witness.get_wire(local_wire(Poseidon2Gate::<F, D>::WIRE_SWAP));
        debug_assert!(swap_value == F::ZERO || swap_value == F::ONE);

        for i in 0..4 {
            let delta_i = swap_value * (state[i + 4] - state[i]);
            out_buffer.set_wire(local_wire(Poseidon2Gate::<F, D>::wire_delta(i)), delta_i)?;
        }

        if swap_value == F::ONE {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }

        <F as Poseidon2>::external_linear_layer(&mut state);

        // The first half of the external rounds.
        for r in 0..ROUNDS_F_HALF {
            <F as Poseidon2>::add_rc(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH {
                    out_buffer.set_wire(
                        local_wire(Poseidon2Gate::<F, D>::wire_full_sbox_0(r, i)),
                        state[i],
                    )?;
                }
            }
            <F as Poseidon2>::sbox(&mut state);
            <F as Poseidon2>::external_linear_layer(&mut state);
        }

        // The internal rounds.
        for r in 0..ROUNDS_P {
            state[0] += F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
            out_buffer.set_wire(
                local_wire(Poseidon2Gate::<F, D>::wire_partial_sbox(r)),
                state[0],
            )?;
            state[0] = <F as Poseidon2>::sbox_p(&state[0]);
            <F as Poseidon2>::internal_linear_layer(&mut state);
        }

        // The second half of the external rounds.
        for r in ROUNDS_F_HALF..ROUNDS_F {
            <F as Poseidon2>::add_rc(&mut state, r);
            for i in 0..WIDTH {
                out_buffer.set_wire(
                    local_wire(Poseidon2Gate::<F, D>::wire_full_sbox_1(
                        r - ROUNDS_F_HALF,
                        i,
                    )),
                    state[i],
                )?;
            }
            <F as Poseidon2>::sbox(&mut state);
            <F as Poseidon2>::external_linear_layer(&mut state);
        }

        for i in 0..WIDTH {
            out_buffer.set_wire(local_wire(Poseidon2Gate::<F, D>::wire_output(i)), state[i])?;
        }

        Ok(())
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

    /// Differential: the packed batch evaluator must be field-equal to the
    /// retained pre-packing scalar column-major evaluator, for batch sizes
    /// covering full packed lanes, leftovers, and mixtures, over random wire
    /// values including noncanonical encodings.
    #[test]
    fn packed_batch_matches_scalar_batch() {
        use crate::field::types::{Field, Field64, PrimeField64};
        use crate::hash::hash_types::HashOut;
        use crate::plonk::vars::EvaluationVarsBaseBatch;

        const D: usize = 2;
        type F = GoldilocksField;
        let gate = Poseidon2Gate::<F, D>::new();
        let num_wires = <Poseidon2Gate<F, D> as Gate<F, D>>::num_wires(&gate);

        fn splitmix64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        for n in [1usize, 2, 3, 4, 5, 7, 8, 9, 12, 32] {
            // Point-major wire buffer: wire w for all n points, then wire w+1.
            let wires: Vec<F> = (0..num_wires * n)
                .map(|i| {
                    let raw = splitmix64(&mut seed);
                    if i % 17 == 0 {
                        // Force some noncanonical encodings.
                        F::from_noncanonical_u64(F::ORDER.wrapping_add(raw % 1024))
                    } else {
                        F::from_noncanonical_u64(raw)
                    }
                })
                .collect();
            let constants: Vec<F> = Vec::new();
            let pis = HashOut::ZERO;
            let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &pis);

            let packed = gate.eval_unfiltered_base_batch(vars);
            let scalar = gate.eval_unfiltered_base_batch_scalar(vars);
            assert_eq!(packed.len(), scalar.len(), "length mismatch at n={n}");
            for (k, (a, b)) in packed.iter().zip(scalar.iter()).enumerate() {
                assert_eq!(
                    a.to_canonical_u64(),
                    b.to_canonical_u64(),
                    "canonical mismatch at n={n}, flat index {k}"
                );
            }
        }
    }

    /// Differential: the packed direct-fold `_accumulate` must be field-equal
    /// to the retained scalar direct-fold reference, on random pre-filled
    /// accumulators and random filters, across lane/leftover mixes.
    #[test]
    fn packed_accumulate_matches_scalar_accumulate() {
        use crate::field::types::{Field, Field64, PrimeField64};
        use crate::hash::hash_types::HashOut;
        use crate::plonk::vars::EvaluationVarsBaseBatch;

        const D: usize = 2;
        type F = GoldilocksField;
        let gate = Poseidon2Gate::<F, D>::new();
        let num_wires = <Poseidon2Gate<F, D> as Gate<F, D>>::num_wires(&gate);
        let num_constraints = <Poseidon2Gate<F, D> as Gate<F, D>>::num_constraints(&gate);

        fn splitmix64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        let mut seed = 0xFEED_FACE_CAFE_BEEFu64;
        for n in [1usize, 3, 4, 5, 8, 9, 32] {
            let wires: Vec<F> = (0..num_wires * n)
                .map(|i| {
                    let raw = splitmix64(&mut seed);
                    if i % 13 == 0 {
                        F::from_noncanonical_u64(F::ORDER.wrapping_add(raw % 512))
                    } else {
                        F::from_noncanonical_u64(raw)
                    }
                })
                .collect();
            let filters: Vec<F> = (0..n)
                .map(|_| F::from_noncanonical_u64(splitmix64(&mut seed)))
                .collect();
            let base: Vec<F> = (0..n * num_constraints)
                .map(|_| F::from_noncanonical_u64(splitmix64(&mut seed)))
                .collect();
            let constants: Vec<F> = Vec::new();
            let pis = HashOut::ZERO;
            let vars = EvaluationVarsBaseBatch::new(n, &constants, &wires, &pis);

            let mut packed = base.clone();
            <Poseidon2Gate<F, D> as Gate<F, D>>::eval_unfiltered_base_batch_accumulate(
                &gate,
                vars,
                &filters,
                &mut packed,
            );
            let mut scalar = base.clone();
            gate.eval_unfiltered_base_batch_accumulate_scalar(vars, &filters, &mut scalar);
            for (k, (a, b)) in packed.iter().zip(scalar.iter()).enumerate() {
                assert_eq!(
                    a.to_canonical_u64(),
                    b.to_canonical_u64(),
                    "accumulate mismatch at n={n}, flat index {k}"
                );
            }
        }
    }

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

    #[test]
    fn direct_filtered_accumulation_matches_materialized_batch() {
        const D: usize = 2;
        const N: usize = 11;
        type F = GoldilocksField;

        let gate = Poseidon2Gate::<F, D>::new();
        let wires = (0..gate.num_wires() * N)
            .map(|i| F::from_canonical_usize(3 * i + 5))
            .collect::<Vec<_>>();
        let constants = Vec::new();
        let hash = crate::hash::hash_types::HashOut::ZERO;
        let vars = crate::plonk::vars::EvaluationVarsBaseBatch::new(N, &constants, &wires, &hash);
        let filters = (0..N)
            .map(|i| F::from_canonical_usize(2 * i + 1))
            .collect::<Vec<_>>();
        let mut expected = vec![F::ZERO; gate.num_constraints() * N];
        let materialized = gate.eval_unfiltered_base_batch(vars);
        for (acc, constraints) in expected
            .chunks_exact_mut(N)
            .zip(materialized.chunks_exact(N))
        {
            crate::field::batch_util::batch_multiply_add_inplace(acc, constraints, &filters);
        }
        let mut actual = vec![F::ZERO; expected.len()];
        gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);
        assert_eq!(actual, expected);
    }
}
