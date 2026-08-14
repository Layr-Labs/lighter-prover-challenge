//! Focused ranking of the gate evaluators that still run on the CPU quotient
//! pass.
//!
//! The production gate census (`PLONKY2_GPU_POSEIDON_DIAGNOSTICS=1`) shows
//! which gates each shape offloads to Metal. What it cannot show is which of
//! the survivors is actually worth rewriting: constraint count is a poor proxy
//! because a 12-constraint barycentric interpolation costs far more per row
//! than a 136-constraint linear range check. This benchmark measures
//! `eval_unfiltered_base_batch_accumulate` directly, at each gate's production
//! parameters, so a strength-reduction candidate can be chosen from measured
//! cost per row rather than from constraint counts.
//!
//! Not part of the proving path. Run with:
//! `cargo test --release --lib gates::cpu_survivor_bench -- --ignored --nocapture`

#[cfg(test)]
mod tests {
    use core::time::Duration;
    use std::time::Instant;

    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, Field64, Sample};
    use crate::gates::arithmetic_base::ArithmeticGate;
    use crate::gates::arithmetic_extension::ArithmeticExtensionGate;
    use crate::gates::coset_interpolation::CosetInterpolationGate;
    use crate::gates::exponentiation::ExponentiationGate;
    use crate::gates::gate::Gate;
    use crate::gates::multiplication_extension::MulExtensionGate;
    use crate::gates::random_access::RandomAccessGate;
    use crate::hash::hash_types::HashOut;
    use crate::plonk::vars::EvaluationVarsBaseBatch;

    type F = GoldilocksField;
    const D: usize = 2;

    /// One 32-point batch is exactly the production `BATCH_SIZE` the quotient
    /// pass feeds these evaluators.
    const BATCH: usize = 32;

    fn measure<G: Gate<F, D>>(gate: &G, label: &str) -> (Duration, usize) {
        let num_constraints = gate.num_constraints();
        let num_wires = gate.num_wires();
        let num_constants = gate.num_constants();

        let mut rng = StdRng::seed_from_u64(0x4350_5553_5552_5630);
        let mut sample = |count: usize| -> Vec<F> {
            (0..count)
                .map(|_| F::from_canonical_u64(rng.next_u64() % F::ORDER))
                .collect()
        };
        let wires = sample(num_wires * BATCH);
        let constants = sample(num_constants * BATCH);
        let filters = sample(BATCH);
        let hash = HashOut::<F>::rand();
        let mut combined = vec![F::default(); num_constraints * BATCH];

        let run = || {
            let vars = EvaluationVarsBaseBatch::new(BATCH, &constants, &wires, &hash);
            let mut out = combined.clone();
            gate.eval_unfiltered_base_batch_accumulate(vars, &filters, &mut out);
            out[0]
        };

        // Warm the caches and the branch predictors before timing.
        let mut sink = F::default();
        for _ in 0..64 {
            sink += run();
        }

        let mut samples = Vec::with_capacity(9);
        for _ in 0..9 {
            let start = Instant::now();
            for _ in 0..256 {
                sink += run();
            }
            samples.push(start.elapsed() / (256 * BATCH) as u32);
        }
        core::hint::black_box(sink);
        samples.sort_unstable();
        let median = samples[samples.len() / 2];
        let _ = label;
        (median, num_constraints)
    }

    /// Ranks the CPU survivors by measured cost per quotient row. Reported as
    /// nanoseconds per row so shapes with different constraint counts are
    /// directly comparable.
    #[test]
    #[ignore = "manual focused CPU gate evaluator ranking"]
    fn benchmark_cpu_survivor_gate_evaluators() {
        let mut rows: Vec<(String, Duration, usize)> = Vec::new();

        let coset = CosetInterpolationGate::<F, D>::with_max_degree(4, 6);
        let (time, constraints) = measure(&coset, "coset");
        rows.push(("CosetInterpolationGate<bits=4,deg=6>".into(), time, constraints));

        let exp = ExponentiationGate::<F, D>::new(67);
        let (time, constraints) = measure(&exp, "exponentiation");
        rows.push(("ExponentiationGate<67 bits>".into(), time, constraints));

        let arithmetic = ArithmeticGate::new_from_config(&Default::default());
        let (time, constraints) = measure(&arithmetic, "arithmetic");
        rows.push((
            format!("ArithmeticGate<{} ops>", arithmetic.num_ops),
            time,
            constraints,
        ));

        let arithmetic_ext = ArithmeticExtensionGate::<D>::new_from_config(&Default::default());
        let (time, constraints) = measure(&arithmetic_ext, "arithmetic_ext");
        rows.push((
            format!("ArithmeticExtensionGate<{} ops>", arithmetic_ext.num_ops),
            time,
            constraints,
        ));

        let mul_ext = MulExtensionGate::<D>::new_from_config(&Default::default());
        let (time, constraints) = measure(&mul_ext, "mul_ext");
        rows.push((
            format!("MulExtensionGate<{} ops>", mul_ext.num_ops),
            time,
            constraints,
        ));

        let random_access = RandomAccessGate::<F, D>::new_from_config(&Default::default(), 6);
        let (time, constraints) = measure(&random_access, "random_access");
        rows.push((
            "RandomAccessGate<bits=6,copies=1>".into(),
            time,
            constraints,
        ));

        // Same-binary A/B of the delayed-reduction specialization, alternating
        // arms so machine drift cannot land on one side. The five gates above
        // are untouched by the change and act as an environment control.
        use core::sync::atomic::Ordering;

        use crate::gates::coset_interpolation::FORCE_GENERIC_INTERPOLATE;

        let mut specialized = Vec::with_capacity(7);
        let mut generic = Vec::with_capacity(7);
        for sample in 0..7 {
            let mut run_specialized = || {
                FORCE_GENERIC_INTERPOLATE.store(false, Ordering::Relaxed);
                measure(&coset, "coset-specialized").0
            };
            let mut run_generic = || {
                FORCE_GENERIC_INTERPOLATE.store(true, Ordering::Relaxed);
                measure(&coset, "coset-generic").0
            };
            if sample & 1 == 0 {
                specialized.push(run_specialized());
                generic.push(run_generic());
            } else {
                generic.push(run_generic());
                specialized.push(run_specialized());
            }
        }
        FORCE_GENERIC_INTERPOLATE.store(false, Ordering::Relaxed);
        specialized.sort_unstable();
        generic.sort_unstable();
        let specialized_median = specialized[specialized.len() / 2];
        let generic_median = generic[generic.len() / 2];

        rows.sort_by_key(|(_, time, _)| core::cmp::Reverse(*time));
        eprintln!("CPU survivor gate evaluators, {BATCH}-point batch, median of 9:");
        for (name, time, constraints) in &rows {
            eprintln!(
                "  {:>10.1} ns/row  {:>3} constraints  {:>7.2} ns/constraint  {name}",
                time.as_secs_f64() * 1e9,
                constraints,
                time.as_secs_f64() * 1e9 / *constraints as f64,
            );
        }
        eprintln!(
            "CosetInterpolation delayed reduction, same binary, median of 7 alternating:\n  \
             specialized={:.1} ns/row  generic={:.1} ns/row  speedup={:.4}x",
            specialized_median.as_secs_f64() * 1e9,
            generic_median.as_secs_f64() * 1e9,
            generic_median.as_secs_f64() / specialized_median.as_secs_f64(),
        );
        eprintln!("  specialized: {specialized:?}");
        eprintln!("  generic:     {generic:?}");
    }
}
