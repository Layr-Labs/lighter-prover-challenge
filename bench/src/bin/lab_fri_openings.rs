//! LOCAL LAB ONLY — never part of a submission.
//!
//! Kernel-rewrite lab for FRI query-round opening extraction and Merkle proof
//! gathers: `fri_prover_query_rounds` (28 query rounds), the initial-tree
//! `leaf_vec` + `prove` gathers over column-store leaves with level-order
//! retained digests, and the commit-phase `get`/`unflatten`/`prove` walks.
//!
//! Production shapes (measured 2026-08-07 by env-gated instrumentation of the
//! full fixture prove, `LAB_FRI_SHAPES=1 target/release/prove bench/bench_test.json`;
//! 106 FRI instances per prove):
//!
//!   A (x51): n=2^19 arity=[4,4,4] cap=4
//!       init  = 2^19 leaves x widths [88,136,20,16], column leaves, level-order digests
//!       commit= [2^15, 2^11, 2^7] x 32, row leaves, interleaved digests
//!   B (x53): n=2^17 arity=[4,4,4] cap=4
//!       init  = 2^17 x [86,136] cols level-order; 2^17 x [20,16] cols interleaved
//!       commit= [2^13, 2^9, 2^5] x 32, rows, interleaved
//!   C (x1):  n=2^21 arity=[4,4,4,4] cap=4
//!       init  = 2^21 x [88,136,20,16] cols level-order
//!       commit= [2^17 rows level-order, 2^13, 2^9, 2^5 rows interleaved] x 32
//!
//! Whole-prove query-phase wall time (sum over 106 instances): ~1.05 s of a
//! ~90 s prove, extremely skewed (median 690 us, p90 6.1 ms, max 277 ms) —
//! consistent with rayon-pool queueing behind pipelined proof work, not with
//! gather cost (the largest instance C took 2.2 ms).
//!
//! RESULTS (2026-08-07, MacBook Air M4 10-core, quiet machine; interleaved
//! ABAB, medians of 12-28 reps; all variants passed full structural identity
//! on 1120 random query indices per shape before timing):
//!
//!   quiet pool, per-instance (us):            A       B       C     mix(ms)
//!     production (par 28 rounds)            46.5    44.9    61.0    ~5.0
//!     V1 serial rounds                      67.4    60.9   109.2    ~7.3
//!     V2 batched tree sweep                 67.2    53.8    90.0    ~6.6
//!     V3 tree-par sweep                     66.7    68.6    94.6    ~7.2
//!   -> on an idle pool the 28-task par_iter is the FASTEST arrangement;
//!      every serial/batched rewrite is a 30-50% kernel regression. NULL for
//!      the pure-gather rewrite hypothesis on quiet hardware.
//!
//!   saturated pool (background rayon load = pipelined-prove regime), mix:
//!     production (par 28 rounds)   med 21.6 ms   min 13.6   max 40.8
//!     V1 serial rounds             med  9.6 ms   min  7.7   max 13.3
//!     V2 batched tree sweep        med  8.3 ms   min  7.1   max 10.9  (+61.5%)
//!     V3 tree-par sweep            med 27.8 ms   (queues like production)
//!   -> FINDING: pool-independent serial extraction dominates under
//!      contention; the two distributions are fully disjoint (baseline min
//!      13.6 ms > V2 max 10.9 ms). V2 wired into `fri_proof`.
//!
//! Method (lab charter): trees are built with the exact production layouts
//! (`MerkleLeaves::Columns`/`Rows`, `LevelOrderDigests` level offsets,
//! interleaved digest arrays) and pseudorandom contents — the extraction path
//! under test performs no hashing, only index math and gathers, so digest
//! values are irrelevant to both identity and timing. Variants are separate
//! functions in vendor/plonky2 `fri::prover`; each is checked for full
//! structural equality (`FriQueryRound: Eq`) against the production function
//! on 1000+ random query indices per production shape BEFORE timing. Timing
//! is interleaved ABAB, median of >= 12 reps, on the production instance mix,
//! both on a quiet pool and under a saturated rayon pool (the pipelined-prove
//! regime).

#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field64;
use plonky2::fri::prover::{
    fri_prover_query_rounds, fri_prover_query_rounds_batched, fri_prover_query_rounds_serial,
    fri_prover_query_rounds_tree_par,
};
use plonky2::fri::reduction_strategies::FriReductionStrategy;
use plonky2::fri::{FriConfig, FriParams};
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::merkle_tree::{
    ColumnStore, LevelOrderDigests, MerkleCap, MerkleLeaves, MerkleTree,
};
use plonky2::iop::challenger::Challenger;
use plonky2::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};

const D: usize = 2;
type C = Poseidon2GoldilocksConfig;
type F = GoldilocksField;
type H = <C as GenericConfig<D>>::Hasher;
type Tree = MerkleTree<F, H>;
type QueryRounds = Vec<plonky2::fri::proof::FriQueryRound<F, H, D>>;

// Keep the bench thread on a P-core.
unsafe extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

// ---------------------------------------------------------------------- RNG --

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_f(state: &mut u64) -> F {
    GoldilocksField(splitmix64(state) % GoldilocksField::ORDER)
}

fn rand_hash(state: &mut u64) -> HashOut<F> {
    HashOut {
        elements: [
            rand_f(state),
            rand_f(state),
            rand_f(state),
            rand_f(state),
        ],
    }
}

// --------------------------------------------------- production tree layouts --

const CAP_HEIGHT: usize = 4;

fn random_cap(state: &mut u64) -> MerkleCap<F, H> {
    MerkleCap((0..1 << CAP_HEIGHT).map(|_| rand_hash(state)).collect())
}

/// Level-order digest storage exactly as the GPU pipeline retains it:
/// level 0 = leaf digests (n nodes), each parent level halves, last level is
/// the cap. `nodes.len() == 2n - 16`, matching the production debug_asserts.
fn random_level_digests(num_leaves: usize, state: &mut u64) -> LevelOrderDigests<HashOut<F>> {
    let num_layers = num_leaves.ilog2() as usize - CAP_HEIGHT;
    let mut level_offsets = Vec::with_capacity(num_layers + 1);
    let mut total = 0usize;
    for l in 0..=num_layers {
        level_offsets.push(total);
        total += num_leaves >> l;
    }
    let nodes = (0..total).map(|_| rand_hash(state)).collect();
    LevelOrderDigests {
        nodes,
        level_offsets,
    }
}

/// Interleaved recursive-subtree digest array (CPU layout): `2 * (n - cap)`.
fn random_interleaved_digests(num_leaves: usize, state: &mut u64) -> Vec<HashOut<F>> {
    (0..2 * (num_leaves - (1 << CAP_HEIGHT)))
        .map(|_| rand_hash(state))
        .collect()
}

/// Initial-oracle tree: natural-order poly-major column leaves.
fn column_tree(num_leaves: usize, width: usize, level_order: bool, state: &mut u64) -> Tree {
    let columns: Vec<Vec<F>> = (0..width)
        .map(|_| (0..num_leaves).map(|_| rand_f(state)).collect())
        .collect();
    Tree {
        leaves: MerkleLeaves::Columns {
            columns: ColumnStore::Owned(columns),
            log_rows: num_leaves.ilog2() as usize,
        },
        num_leaves,
        digests: if level_order {
            Vec::new()
        } else {
            random_interleaved_digests(num_leaves, state)
        },
        level_digests: level_order.then(|| random_level_digests(num_leaves, state)),
        cap: random_cap(state),
    }
}

/// Commit-phase tree: flat row-major leaves (built by `MerkleTree::new_flat`).
fn row_tree(num_leaves: usize, width: usize, level_order: bool, state: &mut u64) -> Tree {
    let data: Vec<F> = (0..num_leaves * width).map(|_| rand_f(state)).collect();
    Tree {
        leaves: MerkleLeaves::Rows { data, width },
        num_leaves,
        digests: if level_order {
            Vec::new()
        } else {
            random_interleaved_digests(num_leaves, state)
        },
        level_digests: level_order.then(|| random_level_digests(num_leaves, state)),
        cap: random_cap(state),
    }
}

struct Shape {
    name: &'static str,
    /// Instances of this shape per full fixture prove.
    count_per_prove: usize,
    n: usize,
    initial: Vec<Tree>,
    commit: Vec<Tree>,
    params: FriParams,
}

fn fri_params(degree_bits: usize, arity: &[usize]) -> FriParams {
    FriParams {
        config: FriConfig {
            rate_bits: 3,
            cap_height: CAP_HEIGHT,
            proof_of_work_bits: 16,
            reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
            num_query_rounds: 28,
        },
        hiding: false,
        degree_bits,
        reduction_arity_bits: arity.to_vec(),
    }
}

fn build_shapes(state: &mut u64) -> Vec<Shape> {
    let shape_a = Shape {
        name: "A n=2^19",
        count_per_prove: 51,
        n: 1 << 19,
        initial: vec![
            column_tree(1 << 19, 88, true, state),
            column_tree(1 << 19, 136, true, state),
            column_tree(1 << 19, 20, true, state),
            column_tree(1 << 19, 16, true, state),
        ],
        commit: vec![
            row_tree(1 << 15, 32, false, state),
            row_tree(1 << 11, 32, false, state),
            row_tree(1 << 7, 32, false, state),
        ],
        params: fri_params(16, &[4, 4, 4]),
    };
    let shape_b = Shape {
        name: "B n=2^17",
        count_per_prove: 53,
        n: 1 << 17,
        initial: vec![
            column_tree(1 << 17, 86, true, state),
            column_tree(1 << 17, 136, true, state),
            column_tree(1 << 17, 20, false, state),
            column_tree(1 << 17, 16, false, state),
        ],
        commit: vec![
            row_tree(1 << 13, 32, false, state),
            row_tree(1 << 9, 32, false, state),
            row_tree(1 << 5, 32, false, state),
        ],
        params: fri_params(14, &[4, 4, 4]),
    };
    let shape_c = Shape {
        name: "C n=2^21",
        count_per_prove: 1,
        n: 1 << 21,
        initial: vec![
            column_tree(1 << 21, 88, true, state),
            column_tree(1 << 21, 136, true, state),
            column_tree(1 << 21, 20, true, state),
            column_tree(1 << 21, 16, true, state),
        ],
        commit: vec![
            row_tree(1 << 17, 32, true, state),
            row_tree(1 << 13, 32, false, state),
            row_tree(1 << 9, 32, false, state),
            row_tree(1 << 5, 32, false, state),
        ],
        params: fri_params(18, &[4, 4, 4, 4]),
    };
    vec![shape_a, shape_b, shape_c]
}

// ------------------------------------------------------------------ harness --

type ExtractFn = fn(&[&Tree], &[Tree], &mut Challenger<F, H>, usize, &FriParams) -> QueryRounds;

const IMPLS: &[(&str, ExtractFn)] = &[
    ("production (par 28 rounds)", fri_prover_query_rounds::<F, C, D>),
    ("V1 serial rounds", fri_prover_query_rounds_serial::<F, C, D>),
    ("V2 batched tree sweep", fri_prover_query_rounds_batched::<F, C, D>),
    ("V3 tree-par sweep", fri_prover_query_rounds_tree_par::<F, C, D>),
];

fn seeded_challenger(seed: u64) -> Challenger<F, H> {
    let mut challenger = Challenger::<F, H>::new();
    let mut s = seed;
    for _ in 0..4 {
        challenger.observe_element(rand_f(&mut s));
    }
    challenger
}

fn run_impl(shape: &Shape, f: ExtractFn, seed: u64) -> QueryRounds {
    let initial: Vec<&Tree> = shape.initial.iter().collect();
    let mut challenger = seeded_challenger(seed);
    f(&initial, &shape.commit, &mut challenger, shape.n, &shape.params)
}

/// Full structural-equality check of every variant against production on
/// `seeds_per_shape` fresh challenger seeds per shape (each seed = 28 random
/// query indices; 40 seeds x 28 = 1120 random inputs per shape per variant).
fn identity_checks(shapes: &[Shape], seeds_per_shape: u64) {
    for shape in shapes {
        for &(name, f) in &IMPLS[1..] {
            for seed in 0..seeds_per_shape {
                let seed = 0xF0F0_0000 + seed * 7919 + shape.n as u64;
                let expected = run_impl(shape, IMPLS[0].1, seed);
                let actual = run_impl(shape, f, seed);
                assert_eq!(
                    expected, actual,
                    "{name} diverges from production on shape {} seed {seed:#x}",
                    shape.name
                );
            }
        }
        println!(
            "identity OK: shape {} ({} seeds x 28 queries per variant, full FriQueryRound equality)",
            shape.name, seeds_per_shape
        );
    }
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    }
}

/// One rep of one impl: the full production mix (51xA + 53xB + 1xC), fresh
/// seeds per instance, returns (elapsed_ms, xor checksum of proof structure).
fn mix_rep(shapes: &[Shape], f: ExtractFn, rep_seed: u64) -> (f64, u64) {
    let mut checksum = 0u64;
    let start = Instant::now();
    for shape in shapes {
        for inst in 0..shape.count_per_prove {
            let seed = rep_seed ^ (shape.n as u64) ^ ((inst as u64) << 32);
            let rounds = run_impl(shape, f, seed);
            // Cheap structural checksum: fold a few sibling limbs + eval limbs.
            for r in rounds.iter() {
                for (leaf, proof) in &r.initial_trees_proof.evals_proofs {
                    checksum ^= leaf[0].0 ^ proof.siblings[0].elements[0].0;
                }
                for s in &r.steps {
                    checksum ^= s.merkle_proof.siblings[0].elements[0].0;
                }
            }
            black_box(&rounds);
        }
    }
    (start.elapsed().as_secs_f64() * 1e3, checksum)
}

fn time_all(shapes: &[Shape], reps: usize, label: &str) {
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); IMPLS.len()];
    let mut checksums = vec![0u64; IMPLS.len()];
    // Warmup.
    for (i, &(_, f)) in IMPLS.iter().enumerate() {
        let (_, c) = mix_rep(shapes, f, 0xAB0);
        checksums[i] = c;
    }
    for c in &checksums[1..] {
        assert_eq!(*c, checksums[0], "checksum mismatch across impls");
    }
    for rep in 0..reps {
        for (i, &(_, f)) in IMPLS.iter().enumerate() {
            let (ms, c) = mix_rep(shapes, f, 0xAB0 + rep as u64 + 1);
            assert_eq!(c, checksums[0] ^ checksums[0] ^ c); // keep c used
            samples[i].push(ms);
        }
    }
    println!("\n== {label}: full prove mix (105 instances), ms/mix, {reps} reps ==");
    let base_med = median(&samples[0]);
    for (i, (name, _)) in IMPLS.iter().enumerate() {
        let mut v = samples[i].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = median(&samples[i]);
        println!(
            "  {:<28} med {:>8.3}  min {:>8.3}  max {:>8.3}  spread {:>5.1}%  vs-base {:>+6.2}%",
            name,
            med,
            v[0],
            v[v.len() - 1],
            (v[v.len() - 1] - v[0]) / med * 100.0,
            (base_med - med) / base_med * 100.0,
        );
    }
}

fn time_per_shape(shapes: &[Shape], reps: usize) {
    for shape in shapes {
        // Enough draws per rep for measurable time.
        let draws = 64usize;
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); IMPLS.len()];
        for (_, f) in IMPLS.iter() {
            black_box(run_impl(shape, *f, 0xCAFE));
        }
        for rep in 0..reps {
            for (i, &(_, f)) in IMPLS.iter().enumerate() {
                let start = Instant::now();
                for d in 0..draws {
                    let rounds = run_impl(shape, f, 0xB00 + rep as u64 * 131 + d as u64);
                    black_box(&rounds);
                }
                samples[i].push(start.elapsed().as_secs_f64() * 1e6 / draws as f64);
            }
        }
        println!("\n== quiet pool: shape {} (us/instance, {reps} reps x {draws} draws) ==", shape.name);
        let base_med = median(&samples[0]);
        for (i, (name, _)) in IMPLS.iter().enumerate() {
            let mut v = samples[i].clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = median(&samples[i]);
            println!(
                "  {:<28} med {:>8.1}  min {:>8.1}  max {:>8.1}  spread {:>5.1}%  vs-base {:>+6.2}%",
                name,
                med,
                v[0],
                v[v.len() - 1],
                (v[v.len() - 1] - v[0]) / med * 100.0,
                (base_med - med) / base_med * 100.0,
            );
        }
    }
}

/// Saturate the global rayon pool from a background thread with fat compute
/// tasks (the pipelined-prove regime: FFT/hash chunks of other proofs), then
/// time the extraction impls interleaved. Models the measured production
/// pathology where 28-task par_iter rounds queue behind fat tasks.
fn time_contended(shapes: &[Shape], reps: usize) {
    let stop = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicU64::new(0));
    let bg_stop = stop.clone();
    let bg_progress = progress.clone();
    let bg = std::thread::spawn(move || {
        use rayon::prelude::*;
        let data: Vec<u64> = (0..1u64 << 20).collect();
        while !bg_stop.load(Ordering::Relaxed) {
            // ~10-40 ms of 8-way fat tasks, the grain of production FFT work.
            let s: u64 = data
                .par_chunks(1 << 17)
                .map(|c| {
                    let mut acc = 0u64;
                    for _ in 0..24 {
                        for &x in c {
                            acc = acc
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                .wrapping_add(x);
                        }
                    }
                    acc
                })
                .reduce(|| 0, u64::wrapping_add);
            bg_progress.fetch_add(s | 1, Ordering::Relaxed);
        }
    });
    // Let the background load ramp up.
    std::thread::sleep(std::time::Duration::from_millis(200));

    time_all(shapes, reps, "CONTENDED pool");

    stop.store(true, Ordering::Relaxed);
    bg.join().unwrap();
    black_box(progress.load(Ordering::Relaxed));
}

fn main() {
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
    let args: Vec<String> = std::env::args().collect();
    let skip_identity = args.iter().any(|a| a == "--skip-identity");
    let reps: usize = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    println!("building production-shaped trees (A: 4x2^19 cols + 3 commit, B: 4x2^17, C: 4x2^21)...");
    let t0 = Instant::now();
    let mut state = 0x5EED_F121_0000_0001u64;
    let shapes = build_shapes(&mut state);
    println!("built in {:.1} s", t0.elapsed().as_secs_f64());

    if !skip_identity {
        println!("\n--- bit-identity checks (before any timing) ---");
        identity_checks(&shapes, 40);
        println!("--- all identity checks passed ---");
    }

    println!("\n--- timing (interleaved ABAB, {reps} reps) ---");
    time_per_shape(&shapes, reps);
    time_all(&shapes, reps, "quiet pool");
    time_contended(&shapes, reps);

    println!("\nlab_fri_openings done.");
}
