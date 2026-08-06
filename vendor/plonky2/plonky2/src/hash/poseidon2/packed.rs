//! Packed (multi-lane) Poseidon2 permutation for the CPU Merkle quad path.
//!
//! `permute_packed` runs `P::WIDTH` independent Poseidon2 states through one
//! instruction stream over `PackedField` lanes. On aarch64/macOS the packing
//! for Goldilocks is `WideGoldilocksField` (4 lanes as two interleaved
//! branch-free `mul`/`umulh` asm pairs), so the four lockstep sponges of
//! `hash_quad_no_pad` collapse into a single 4-lane permutation instead of
//! four interleaved scalar ones.
//!
//! Every helper computes exactly the same field values as the scalar
//! `Poseidon2` trait methods (all operations are exact field arithmetic), so
//! each lane of the packed permutation is canonically equal to the scalar
//! permutation of that lane's inputs. Raw (noncanonical) representations may
//! differ between the reduction paths; `GoldilocksField` equality,
//! serialization, and every downstream permutation are canonical, so digests
//! are interchangeable with the scalar path's.
//!
//! Pair decision: `two_to_one_pair` / `compress_pair` (and
//! `hash_pair_no_pad`) deliberately stay on the scalar-interleaved
//! `poseidon2_x2` path. Routing a pair through the 4-lane permutation would
//! pay four lanes' work for two states, and a 2-lane `NeonGoldilocksField`
//! instantiation has the same per-lane op count as the 4-lane one (the wide
//! type is literally two Neon asm blocks) with half the cross-lane ILP, so it
//! has no op-count edge over the tuned x2 path. Pair compressions are also a
//! small minority of tree permutations (one per four leaves hashed), so the
//! upside would be marginal even if it measured ahead.

use core::any::TypeId;

use crate::field::goldilocks_field::GoldilocksField;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::field::types::{Field, PrimeField64};
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::poseidon2::config::*;

/// The 4-lane Goldilocks packing used by the quad fast path.
type P = <GoldilocksField as Packable>::Packing;

/// The quad fast path transposes exactly four sponges into the lanes.
const _: () = assert!(<P as PackedField>::WIDTH == 4);

#[inline]
fn add_rc_packed<Pk: PackedField>(state: &mut [Pk; WIDTH], external_round: usize) {
    for i in 0..WIDTH {
        state[i] += Pk::Scalar::from_canonical_u64(EXTERNAL_CONSTANTS[external_round][i]);
    }
}

#[inline]
fn sbox_p_packed<Pk: PackedField>(x: Pk) -> Pk {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x3 = x * x2;
    x3 * x4
}

#[inline]
fn mat4_packed<Pk: PackedField>(x: &mut [Pk]) {
    let t01 = x[0] + x[1];
    let t23 = x[2] + x[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + x[1];
    let t01233 = t0123 + x[3];
    // Overwrite x[0] and x[2] after x[1] and x[3], as in the scalar layers.
    x[3] = t01233 + x[0].doubles();
    x[1] = t01123 + x[2].doubles();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

#[inline]
fn external_linear_layer_packed<Pk: PackedField>(state: &mut [Pk; WIDTH]) {
    for chunk in state.chunks_mut(4) {
        mat4_packed(chunk);
    }
    let sums: [Pk; 4] = core::array::from_fn(|k| {
        (0..WIDTH)
            .step_by(4)
            .map(|j| state[j + k])
            .fold(Pk::ZEROS, |acc, x| acc + x)
    });
    for i in 0..WIDTH {
        state[i] += sums[i % 4];
    }
}

#[inline]
fn internal_linear_layer_packed<Pk: PackedField>(state: &mut [Pk; WIDTH]) {
    let sum = state.iter().fold(Pk::ZEROS, |acc, &x| acc + x);
    for i in 0..WIDTH {
        state[i] = state[i] * Pk::Scalar::from_canonical_u64(MATRIX_DIAG_12_U64[i]) + sum;
    }
}

/// Full Poseidon2 permutation over `Pk::WIDTH` independent lanes. Mirrors the
/// scalar `Poseidon2::poseidon2` round structure and constants exactly, so
/// lane `l` of the result equals `poseidon2` of lane `l` of the input
/// (canonically; raw representations may differ between reduction paths).
#[inline]
fn permute_packed<Pk: PackedField>(mut state: [Pk; WIDTH]) -> [Pk; WIDTH] {
    external_linear_layer_packed(&mut state);

    for r in 0..ROUNDS_F_HALF {
        add_rc_packed(&mut state, r);
        for value in state.iter_mut() {
            *value = sbox_p_packed(*value);
        }
        external_linear_layer_packed(&mut state);
    }

    for r in 0..ROUNDS_P {
        state[0] += Pk::Scalar::from_canonical_u64(INTERNAL_CONSTANTS[r]);
        state[0] = sbox_p_packed(state[0]);
        internal_linear_layer_packed(&mut state);
    }

    for r in ROUNDS_F_HALF..ROUNDS_F {
        add_rc_packed(&mut state, r);
        for value in state.iter_mut() {
            *value = sbox_p_packed(*value);
        }
        external_linear_layer_packed(&mut state);
    }

    state
}

/// Four lockstep overwrite-mode sponges over equal-length inputs, one lane
/// each, permuted together via `permute_packed`. Absorbs `RATE`-element
/// chunks in overwrite mode (a final partial chunk overwrites only its `len`
/// leading state elements), then squeezes `state[0..4]` — exactly the
/// `hash_quad_no_pad` sponge schedule, so each output is canonically equal to
/// `hash_n_to_hash_no_pad` on the corresponding input.
///
/// Must only be called with `F = GoldilocksField` (the caller dispatches on
/// `TypeId`); raw noncanonical `u64` representations are moved across
/// unchanged in both directions, as in the established Goldilocks fast paths.
pub(crate) fn hash_quad_no_pad_packed<F: RichField>(
    input_a: &[F],
    input_b: &[F],
    input_c: &[F],
    input_d: &[F],
) -> [HashOut<F>; 4] {
    debug_assert_eq!(TypeId::of::<F>(), TypeId::of::<GoldilocksField>());
    debug_assert_eq!(input_a.len(), input_b.len());
    debug_assert_eq!(input_a.len(), input_c.len());
    debug_assert_eq!(input_a.len(), input_d.len());

    let inputs = [input_a, input_b, input_c, input_d];
    let len = input_a.len();
    let mut state = [P::ZEROS; WIDTH];

    let mut offset = 0;
    while offset < len {
        let chunk_len = RATE.min(len - offset);
        for k in 0..chunk_len {
            let slot = state[k].as_slice_mut();
            for (l, input) in inputs.iter().enumerate() {
                slot[l] = GoldilocksField(input[offset + k].to_noncanonical_u64());
            }
        }
        state = permute_packed(state);
        offset += chunk_len;
    }

    core::array::from_fn(|l| HashOut {
        elements: core::array::from_fn(|d| {
            F::from_noncanonical_u64(state[d].as_slice()[l].to_noncanonical_u64())
        }),
    })
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use super::*;
    use crate::field::types::Field64;
    use crate::hash::poseidon2::hash::Poseidon2;

    type F = GoldilocksField;

    /// Boundary-heavy raw u64 patterns, including noncanonical values, like
    /// the metal differential tests.
    fn raw_value(index: u64, rng: &mut StdRng) -> u64 {
        match index & 7 {
            0 => 0,
            1 => 1,
            2 => F::ORDER - 1,
            3 => F::ORDER,
            4 => F::ORDER + 1,
            5 => u64::MAX,
            _ => rng.next_u64(),
        }
    }

    #[test]
    fn packed_permutation_matches_scalar_lanes() {
        let mut rng = StdRng::seed_from_u64(0x5041_434b_4544);
        for trial in 0..16u64 {
            let lanes: Vec<[F; WIDTH]> = (0..<P as PackedField>::WIDTH)
                .map(|lane| {
                    core::array::from_fn(|i| {
                        GoldilocksField(raw_value(trial + lane as u64 + i as u64, &mut rng))
                    })
                })
                .collect();

            let mut state = [P::ZEROS; WIDTH];
            for (l, lane) in lanes.iter().enumerate() {
                for i in 0..WIDTH {
                    state[i].as_slice_mut()[l] = lane[i];
                }
            }
            let state = permute_packed(state);

            for (l, lane) in lanes.iter().enumerate() {
                let expected = <F as Poseidon2>::poseidon2(*lane);
                for i in 0..WIDTH {
                    assert_eq!(
                        state[i].as_slice()[l].to_canonical_u64(),
                        expected[i].to_canonical_u64(),
                        "trial {trial}, lane {l}, element {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn packed_quad_sponge_matches_scalar_across_widths_and_raw_values() {
        let mut rng = StdRng::seed_from_u64(0x5041_434b_5134);
        for width in [0usize, 1, 4, 5, 7, 8, 9, 16, 17, 24, 33, 87, 135] {
            let inputs: [Vec<F>; 4] = core::array::from_fn(|lane| {
                (0..width)
                    .map(|i| GoldilocksField(raw_value((lane * width + i) as u64, &mut rng)))
                    .collect()
            });
            let packed =
                hash_quad_no_pad_packed::<F>(&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
            for (lane, input) in inputs.iter().enumerate() {
                let expected = crate::hash::hashing::hash_n_to_hash_no_pad::<
                    F,
                    crate::hash::poseidon2::hash::Poseidon2Permutation<F>,
                >(input);
                assert_eq!(packed[lane], expected, "width {width}, lane {lane}");
            }
        }
    }
}
