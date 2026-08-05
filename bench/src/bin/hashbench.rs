// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

//! Local-only microbenchmark for the Poseidon2 permutation and Merkle leaf
//! hashing. Not built by `setup.sh`, which compiles only `--bin prove`.
//!
//! Question it answers: is the permutation throughput-bound (all execution
//! units busy) or latency-bound (stalled on the serial dependency chain inside
//! the 22 partial rounds)? If interleaving K independent permutations scales
//! better than K, the hot path is latency-bound and batching leaf hashing is
//! worth doing inside plonky2.

use std::hint::black_box;
use std::time::Instant;

use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::{Field, PrimeField64};
use plonky2::hash::hashing::hash_n_to_hash_no_pad;
use plonky2::hash::poseidon2::config::{
    INTERNAL_CONSTANTS, MATRIX_DIAG_12_U64, ROUNDS_F_HALF, ROUNDS_P, WIDTH,
};
use plonky2::hash::poseidon2::hash::{Poseidon2, Poseidon2Permutation};

const LEAF_LEN: usize = 136;

fn rand_state(seed: u64) -> [F; WIDTH] {
    let mut x = seed | 1;
    core::array::from_fn(|_| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_canonical_u64(x >> 1)
    })
}

/// Reference permutation, exactly what the prover calls today.
#[inline]
fn perm1(state: [F; WIDTH]) -> [F; WIDTH] {
    F::poseidon2(state)
}

/// `K` independent permutations, interleaved per operation so that the serial
/// sum chain inside each partial round overlaps across lanes.
#[inline]
fn perm_k<const K: usize>(states: &mut [[F; WIDTH]; K]) {
    for s in states.iter_mut() {
        F::external_linear_layer(s);
    }
    for r in 0..ROUNDS_F_HALF {
        for s in states.iter_mut() {
            F::add_rc(s, r);
        }
        for s in states.iter_mut() {
            F::sbox(s);
        }
        for s in states.iter_mut() {
            F::external_linear_layer(s);
        }
    }
    for r in 0..ROUNDS_P {
        let rc = F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
        for s in states.iter_mut() {
            s[0] += rc;
            s[0] = F::sbox_p(&s[0]);
        }
        for s in states.iter_mut() {
            F::internal_linear_layer(s);
        }
    }
    for r in ROUNDS_F_HALF..(2 * ROUNDS_F_HALF) {
        for s in states.iter_mut() {
            F::add_rc(s, r);
        }
        for s in states.iter_mut() {
            F::sbox(s);
        }
        for s in states.iter_mut() {
            F::external_linear_layer(s);
        }
    }
}

/// `perm_k` with the balanced-tree internal sum.
#[inline]
fn perm_k_tree<const K: usize>(states: &mut [[F; WIDTH]; K]) {
    for s in states.iter_mut() {
        F::external_linear_layer(s);
    }
    for r in 0..ROUNDS_F_HALF {
        for s in states.iter_mut() {
            F::add_rc(s, r);
        }
        for s in states.iter_mut() {
            F::sbox(s);
        }
        for s in states.iter_mut() {
            F::external_linear_layer(s);
        }
    }
    for r in 0..ROUNDS_P {
        let rc = F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
        for s in states.iter_mut() {
            s[0] += rc;
            s[0] = F::sbox_p(&s[0]);
        }
        for s in states.iter_mut() {
            internal_linear_layer_tree(s);
        }
    }
    for r in ROUNDS_F_HALF..(2 * ROUNDS_F_HALF) {
        for s in states.iter_mut() {
            F::add_rc(s, r);
        }
        for s in states.iter_mut() {
            F::sbox(s);
        }
        for s in states.iter_mut() {
            F::external_linear_layer(s);
        }
    }
}

/// Same as `internal_linear_layer` but with the 12-term sum built as a balanced
/// tree instead of a left-to-right chain of `u128` additions.
#[inline]
fn internal_linear_layer_tree(state: &mut [F; WIDTH]) {
    let a = (state[0].to_noncanonical_u64() as u128 + state[1].to_noncanonical_u64() as u128)
        + (state[2].to_noncanonical_u64() as u128 + state[3].to_noncanonical_u64() as u128);
    let b = (state[4].to_noncanonical_u64() as u128 + state[5].to_noncanonical_u64() as u128)
        + (state[6].to_noncanonical_u64() as u128 + state[7].to_noncanonical_u64() as u128);
    let c = (state[8].to_noncanonical_u64() as u128 + state[9].to_noncanonical_u64() as u128)
        + (state[10].to_noncanonical_u64() as u128 + state[11].to_noncanonical_u64() as u128);
    let sum = F::from_noncanonical_u128_with_96_bits((a + b) + c);
    for i in 0..WIDTH {
        state[i] = sum.multiply_accumulate(state[i], F::from_canonical_u64(MATRIX_DIAG_12_U64[i]));
    }
}

#[inline]
fn perm1_tree(mut state: [F; WIDTH]) -> [F; WIDTH] {
    F::external_linear_layer(&mut state);
    for r in 0..ROUNDS_F_HALF {
        F::add_rc(&mut state, r);
        F::sbox(&mut state);
        F::external_linear_layer(&mut state);
    }
    for r in 0..ROUNDS_P {
        state[0] += F::from_canonical_u64(INTERNAL_CONSTANTS[r]);
        state[0] = F::sbox_p(&state[0]);
        internal_linear_layer_tree(&mut state);
    }
    for r in ROUNDS_F_HALF..(2 * ROUNDS_F_HALF) {
        F::add_rc(&mut state, r);
        F::sbox(&mut state);
        F::external_linear_layer(&mut state);
    }
    state
}

fn time<T>(label: &str, iters: usize, unit: f64, mut f: impl FnMut() -> T) {
    // warm up
    for _ in 0..(iters / 8).max(1) {
        black_box(f());
    }
    let start = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let per = elapsed / (iters as f64 * unit);
    println!("{label:44} {elapsed:8.4} s   {:9.2} ns/perm", per * 1e9);
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(2_000_000);

    println!("== permutation throughput (single thread) ==");
    let s0 = rand_state(1);
    time("perm1 (dependent chain)", iters, 1.0, || {
        let mut s = black_box(s0);
        s = perm1(s);
        s
    });

    // Independent lanes: feed a fresh state each iteration so the measurement is
    // throughput, not the latency of a self-feeding chain.
    let mut lanes1 = [rand_state(11)];
    time("perm_k<1> independent", iters, 1.0, || {
        perm_k::<1>(black_box(&mut lanes1));
    });
    let mut lanes2 = [rand_state(21), rand_state(22)];
    time("perm_k<2> independent", iters / 2, 2.0, || {
        perm_k::<2>(black_box(&mut lanes2));
    });
    let mut lanes3 = [rand_state(31), rand_state(32), rand_state(33)];
    time("perm_k<3> independent", iters / 3, 3.0, || {
        perm_k::<3>(black_box(&mut lanes3));
    });
    let mut lanes4 = [
        rand_state(41),
        rand_state(42),
        rand_state(43),
        rand_state(44),
    ];
    time("perm_k<4> independent", iters / 4, 4.0, || {
        perm_k::<4>(black_box(&mut lanes4));
    });
    let mut lanes6: [[F; WIDTH]; 6] = core::array::from_fn(|i| rand_state(60 + i as u64));
    time("perm_k<6> independent", iters / 6, 6.0, || {
        perm_k::<6>(black_box(&mut lanes6));
    });

    println!();
    println!("== balanced-tree sum_12 variant ==");
    time("perm1_tree (dependent chain)", iters, 1.0, || {
        let mut s = black_box(s0);
        s = perm1_tree(s);
        s
    });

    println!();
    println!("== 136-element leaf hashing (17 perms per leaf) ==");
    let leaf: Vec<F> = (0..LEAF_LEN)
        .map(|i| F::from_canonical_u64(i as u64 * 7 + 3))
        .collect();
    let leaf_iters = iters / 17;
    time(
        "hash_n_to_hash_no_pad (Vec alloc per call)",
        leaf_iters,
        17.0,
        || hash_n_to_hash_no_pad::<F, Poseidon2Permutation<F>>(black_box(&leaf)),
    );

    // Allocation-free equivalent: absorb in overwrite mode, squeeze 4 elements.
    let hash_flat = |input: &[F]| -> [F; 4] {
        let mut state = [F::ZERO; WIDTH];
        for chunk in input.chunks(8) {
            state[..chunk.len()].copy_from_slice(chunk);
            state = F::poseidon2(state);
        }
        [state[0], state[1], state[2], state[3]]
    };
    time("flat sponge, no Vec", leaf_iters, 17.0, || {
        hash_flat(black_box(&leaf))
    });

    // Sanity: the allocation-free sponge must match the library output exactly.
    let reference = hash_n_to_hash_no_pad::<F, Poseidon2Permutation<F>>(&leaf);
    assert_eq!(reference.elements, hash_flat(&leaf), "sponge mismatch");

    // 4 leaves at a time with interleaved permutations.
    let hash_flat_x4 = |inputs: [&[F]; 4]| -> [[F; 4]; 4] {
        let mut states = [[F::ZERO; WIDTH]; 4];
        let nchunks = inputs[0].len().div_ceil(8);
        for c in 0..nchunks {
            for (state, input) in states.iter_mut().zip(inputs.iter()) {
                let chunk = &input[c * 8..((c + 1) * 8).min(input.len())];
                state[..chunk.len()].copy_from_slice(chunk);
            }
            perm_k::<4>(&mut states);
        }
        core::array::from_fn(|i| [states[i][0], states[i][1], states[i][2], states[i][3]])
    };
    time(
        "flat sponge x4 interleaved",
        leaf_iters / 4,
        4.0 * 17.0,
        || hash_flat_x4(black_box([&leaf, &leaf, &leaf, &leaf])),
    );
    assert_eq!(
        reference.elements,
        hash_flat_x4([&leaf, &leaf, &leaf, &leaf])[2],
        "x4 sponge mismatch"
    );

    // Same, plus the balanced-tree internal sum.
    fn hash_flat_kt<const K: usize>(inputs: &[&[F]; K], out: &mut [[F; 4]; K]) {
        let mut states = [[F::ZERO; WIDTH]; K];
        let nchunks = inputs[0].len().div_ceil(8);
        for c in 0..nchunks {
            for (state, input) in states.iter_mut().zip(inputs.iter()) {
                let chunk = &input[c * 8..((c + 1) * 8).min(input.len())];
                state[..chunk.len()].copy_from_slice(chunk);
            }
            perm_k_tree::<K>(&mut states);
        }
        for i in 0..K {
            out[i] = [states[i][0], states[i][1], states[i][2], states[i][3]];
        }
    }

    let mut out2 = [[F::ZERO; 4]; 2];
    time(
        "flat sponge x2 + tree sum",
        leaf_iters / 2,
        2.0 * 17.0,
        || {
            hash_flat_kt::<2>(black_box(&[&leaf, &leaf]), &mut out2);
        },
    );
    let mut out4 = [[F::ZERO; 4]; 4];
    time(
        "flat sponge x4 + tree sum",
        leaf_iters / 4,
        4.0 * 17.0,
        || {
            hash_flat_kt::<4>(black_box(&[&leaf; 4]), &mut out4);
        },
    );
    let mut out6 = [[F::ZERO; 4]; 6];
    time(
        "flat sponge x6 + tree sum",
        leaf_iters / 6,
        6.0 * 17.0,
        || {
            hash_flat_kt::<6>(black_box(&[&leaf; 6]), &mut out6);
        },
    );
    let mut out8 = [[F::ZERO; 4]; 8];
    time(
        "flat sponge x8 + tree sum",
        leaf_iters / 8,
        8.0 * 17.0,
        || {
            hash_flat_kt::<8>(black_box(&[&leaf; 8]), &mut out8);
        },
    );
    hash_flat_kt::<4>(&[&leaf; 4], &mut out4);
    assert_eq!(reference.elements, out4[3], "x4 tree sponge mismatch");
    hash_flat_kt::<8>(&[&leaf; 8], &mut out8);
    assert_eq!(reference.elements, out8[7], "x8 tree sponge mismatch");

    println!();
    println!("== absorb copy shape ==");
    // The variable-length `copy_from_slice` in the absorb loop compiles to a
    // `memmove` call because the length is not a compile-time constant. Split
    // the full-rate chunks out so the copy is a fixed 8 elements.
    let hash_flat_x4_fixed = |inputs: [&[F]; 4]| -> [[F; 4]; 4] {
        let mut states = [[F::ZERO; WIDTH]; 4];
        let len = inputs[0].len();
        let full = len / 8;
        for c in 0..full {
            for (state, input) in states.iter_mut().zip(inputs.iter()) {
                let chunk: &[F; 8] = input[c * 8..c * 8 + 8].try_into().unwrap();
                state[..8].copy_from_slice(chunk);
            }
            perm_k::<4>(&mut states);
        }
        if full * 8 < len {
            let rest = len - full * 8;
            for (state, input) in states.iter_mut().zip(inputs.iter()) {
                state[..rest].copy_from_slice(&input[full * 8..]);
            }
            perm_k::<4>(&mut states);
        }
        core::array::from_fn(|i| [states[i][0], states[i][1], states[i][2], states[i][3]])
    };
    time(
        "flat sponge x4, fixed-size absorb copy",
        leaf_iters / 4,
        4.0 * 17.0,
        || hash_flat_x4_fixed(black_box([&leaf, &leaf, &leaf, &leaf])),
    );
    assert_eq!(
        reference.elements,
        hash_flat_x4_fixed([&leaf; 4])[1],
        "fixed-copy sponge mismatch"
    );
}
