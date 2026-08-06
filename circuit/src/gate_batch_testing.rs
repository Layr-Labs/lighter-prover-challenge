// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Test-only helper checking a gate's `eval_unfiltered_base_batch` against
//! per-point `eval_unfiltered` across a multi-point batch. The upstream
//! `test_eval_fns` only exercises a batch of one point, which cannot catch
//! column-indexing mistakes in hand-written batched evaluations.

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::{Field, Field64};
use plonky2::gates::gate::Gate;
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::vars::{EvaluationVars, EvaluationVarsBaseBatch};
use rand::Rng;

#[cfg(test)]
mod vendored_gate_tests {
    use plonky2::gates::poseidon2::Poseidon2Gate;

    use super::*;

    #[test]
    fn poseidon2_base_batch_matches_eval_unfiltered_across_batch() {
        let gate = Poseidon2Gate::<GoldilocksField, 2>::new();
        assert_base_batch_matches_eval_unfiltered(&gate);
    }


    #[test]
    fn range_check_base_batch_matches_eval_unfiltered_across_batch() {
        use crate::uint::range_check::RangeCheckGate;
        use plonky2::plonk::circuit_data::CircuitConfig;

        for bit_size in [15, 16, 32, 48] {
            let gate = RangeCheckGate::<GoldilocksField, 2>::new_from_config(
                &CircuitConfig::standard_recursion_config(),
                bit_size,
            );
            assert_base_batch_matches_eval_unfiltered(&gate);
        }
    }


    #[test]
    fn comparison_base_batch_matches_eval_unfiltered_across_batch() {
        use crate::uint::u32::gates::comparison::ComparisonGate;

        for (num_bits, num_chunks) in [(32, 16), (32, 8), (30, 10)] {
            let gate = ComparisonGate::<GoldilocksField, 2>::new(num_bits, num_chunks);
            assert_base_batch_matches_eval_unfiltered(&gate);
        }
    }


    #[test]
    fn u16_arithmetic_base_batch_matches_eval_unfiltered_across_batch() {
        use crate::uint::u16::gates::arithmetic_u16::U16ArithmeticGate;
        use plonky2::plonk::circuit_data::CircuitConfig;

        let gate = U16ArithmeticGate::<GoldilocksField, 2>::new_from_config(
            &CircuitConfig::standard_recursion_config(),
        );
        assert_base_batch_matches_eval_unfiltered(&gate);
    }

    #[test]
    fn random_access_base_batch_matches_eval_unfiltered_across_batch() {
        use plonky2::gates::random_access::RandomAccessGate;
        use plonky2::plonk::circuit_data::CircuitConfig;

        for bits in [1, 2, 4, 6] {
            let gate = RandomAccessGate::<GoldilocksField, 2>::new_from_config(
                &CircuitConfig::standard_recursion_config(),
                bits,
            );
            assert_base_batch_matches_eval_unfiltered(&gate);
        }
    }
}

pub fn assert_base_batch_matches_eval_unfiltered<G>(gate: &G)
where
    G: Gate<GoldilocksField, 2>,
{
    const D: usize = 2;
    type F = GoldilocksField;

    let mut rng = rand::thread_rng();
    let n = 32;
    let num_wires = gate.num_wires();
    let num_constants = gate.num_constants();
    let num_constraints = gate.num_constraints();

    // Column-major layout: wire w for point p at [w * n + p].
    let wires_batch: Vec<F> = (0..num_wires * n)
        .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
        .collect();
    let constants_batch: Vec<F> = (0..num_constants * n)
        .map(|_| F::from_canonical_u64(rng.gen_range(0..GoldilocksField::ORDER)))
        .collect();
    let public_inputs_hash = HashOut::<F>::ZERO;

    let vars_batch =
        EvaluationVarsBaseBatch::new(n, &constants_batch, &wires_batch, &public_inputs_hash);
    let batch_out = gate.eval_unfiltered_base_batch(vars_batch);
    assert_eq!(batch_out.len(), n * num_constraints);

    for p in 0..n {
        let to_ext = |value: F| {
            <<F as Extendable<D>>::Extension as FieldExtension<D>>::from_basefield(value)
        };
        let wires_one: Vec<_> = (0..num_wires)
            .map(|w| to_ext(wires_batch[w * n + p]))
            .collect();
        let constants_one: Vec<_> = (0..num_constants)
            .map(|c| to_ext(constants_batch[c * n + p]))
            .collect();
        let vars_one = EvaluationVars::<F, D> {
            local_constants: &constants_one,
            local_wires: &wires_one,
            public_inputs_hash: &public_inputs_hash,
        };
        let expected = gate.eval_unfiltered(vars_one);
        for (j, expected_j) in expected.iter().enumerate() {
            assert_eq!(
                to_ext(batch_out[j * n + p]),
                *expected_j,
                "gate {}, point {p}, constraint {j}",
                gate.id()
            );
        }
    }
}
