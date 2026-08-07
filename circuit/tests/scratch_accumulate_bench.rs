// Measures whether the thread-local scratch override on Poseidon2Gate's
// filtered accumulation is actually worth its complexity, against the trait
// default (materialize a fresh Vec, then fold). Equality of the accumulated
// output is asserted so this doubles as a correctness check.

use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::{Field, Sample};
use plonky2::gates::gate::Gate;
use plonky2::gates::poseidon2::Poseidon2Gate;
use plonky2::hash::hash_types::HashOut;
use plonky2::plonk::vars::EvaluationVarsBaseBatch;

const D: usize = 2;

#[test]
fn time_scratch_accumulate_vs_materialize() {
    let gate = Poseidon2Gate::<F, D>::new();
    let n = 32;
    let num_wires = <Poseidon2Gate<F, D> as Gate<F, D>>::num_wires(&gate);
    let num_constraints = <Poseidon2Gate<F, D> as Gate<F, D>>::num_constraints(&gate);
    let wires: Vec<F> = (0..num_wires * n).map(|_| F::rand()).collect();
    let filters: Vec<F> = (0..n).map(|_| F::rand()).collect();
    let pih = HashOut::<F>::ZERO;
    let iters = 2_000;

    // Old path: the trait default, which allocates and zeroes a fresh
    // `num_constraints * n` buffer on every batch call.
    let mut acc_old = vec![F::ZERO; n * num_constraints];
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let vars = EvaluationVarsBaseBatch::new(n, &[], &wires, &pih);
        gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut acc_old);
    }
    let old_time = t0.elapsed();

    // New path: identical work, but materializing into a caller-owned typed
    // scratch buffer that is reused across calls.
    let mut acc_new = vec![F::ZERO; n * num_constraints];
    let mut scratch: Vec<F> = Vec::new();
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        let vars = EvaluationVarsBaseBatch::new(n, &[], &wires, &pih);
        gate.eval_unfiltered_base_batch_accumulate_scratch(
            vars,
            &filters,
            &mut scratch,
            &mut acc_new,
        );
    }
    let new_time = t1.elapsed();

    assert_eq!(acc_old, acc_new, "scratch accumulate diverged from the allocating default");
    println!(
        "Poseidon2Gate accumulate: allocating {:?}  reused scratch {:?}  speedup {:.3}x  ({num_constraints} constraints, n={n})",
        old_time,
        new_time,
        old_time.as_secs_f64() / new_time.as_secs_f64()
    );
}
