// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Test-only helper checking a gate's `eval_unfiltered_base_batch` against
//! per-point `eval_unfiltered` across a multi-point batch. The upstream
//! `test_eval_fns` only exercises a batch of one point, which cannot catch
//! column-indexing mistakes in hand-written batched evaluations.

use plonky2::field::batch_util::batch_multiply_add_inplace;
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

/// Checks a gate's `eval_unfiltered_base_batch_accumulate` override against
/// the default path (materialize the full unfiltered constraint matrix, then
/// `batch_multiply_add_inplace` row by row) across a multi-point batch with
/// deterministic pseudo-random wires, constants, and filters. Constraint order
/// and values must be bit-identical.
pub fn assert_direct_accumulation_matches_materialized_batch<G>(gate: &G)
where
    G: Gate<GoldilocksField, 2>,
{
    type F = GoldilocksField;
    const N: usize = 11;

    let wires = (0..gate.num_wires() * N)
        .map(|i| F::from_canonical_usize(3 * i + 5))
        .collect::<Vec<_>>();
    let constants = (0..gate.num_constants() * N)
        .map(|i| F::from_canonical_usize(7 * i + 11))
        .collect::<Vec<_>>();
    let hash = HashOut::ZERO;
    let vars = EvaluationVarsBaseBatch::new(N, &constants, &wires, &hash);
    let filters = (0..N)
        .map(|i| F::from_canonical_usize(2 * i + 1))
        .collect::<Vec<_>>();

    let mut expected = vec![F::ZERO; gate.num_constraints() * N];
    let materialized = gate.eval_unfiltered_base_batch(vars);
    for (acc, constraints) in expected
        .chunks_exact_mut(N)
        .zip(materialized.chunks_exact(N))
    {
        batch_multiply_add_inplace(acc, constraints, &filters);
    }
    let mut actual = vec![F::ZERO; expected.len()];
    gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut actual);
    assert_eq!(actual, expected, "gate {}", gate.id());
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
