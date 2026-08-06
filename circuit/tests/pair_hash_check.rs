// Temporary validation for the interleaved Poseidon2 pair sponge:
// bit-equality against the individual hash across leaf widths, plus a
// single-threaded timing comparison. Removed before submission if noisy.

use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::Sample;
use plonky2::hash::poseidon2::hash::Poseidon2Hash;
use plonky2::plonk::config::Hasher;

#[test]
fn pair_hash_matches_individual_across_widths() {
    for width in [1usize, 2, 4, 5, 7, 8, 9, 16, 17, 24, 33, 87, 135] {
        let a: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let b: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let (ha, hb) = Poseidon2Hash::hash_or_noop_pair(&a, &b);
        assert_eq!(
            ha,
            <Poseidon2Hash as Hasher<F>>::hash_or_noop(&a),
            "width {width} a"
        );
        assert_eq!(
            hb,
            <Poseidon2Hash as Hasher<F>>::hash_or_noop(&b),
            "width {width} b"
        );
    }
}

#[test]
fn time_sequential_vs_pair_leaf_hash() {
    for width in [8usize, 16, 87, 135] {
        let a: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let b: Vec<F> = (0..width).map(|_| F::rand()).collect();
        let iters = 100_000;

        let t0 = std::time::Instant::now();
        let mut sink_old = F::default();
        for _ in 0..iters {
            let ha = <Poseidon2Hash as Hasher<F>>::hash_or_noop(core::hint::black_box(&a));
            let hb = <Poseidon2Hash as Hasher<F>>::hash_or_noop(core::hint::black_box(&b));
            sink_old += ha.elements[0] + hb.elements[0];
        }
        let old_time = t0.elapsed();

        let t1 = std::time::Instant::now();
        let mut sink_new = F::default();
        for _ in 0..iters {
            let (ha, hb) = Poseidon2Hash::hash_or_noop_pair(
                core::hint::black_box(&a),
                core::hint::black_box(&b),
            );
            sink_new += ha.elements[0] + hb.elements[0];
        }
        let new_time = t1.elapsed();

        assert_eq!(sink_old, sink_new);
        println!(
            "width {width}: sequential {:?}  interleaved {:?}  speedup {:.2}x",
            old_time,
            new_time,
            old_time.as_secs_f64() / new_time.as_secs_f64()
        );
    }
}
