//! Packed (multi-lane) Poseidon2 layers shared by the gate's batched
//! constraint evaluation and the packed CPU Merkle tree builder.
//!
//! Every helper computes exactly the same field values as the scalar
//! `Poseidon2` trait methods (all operations are exact field arithmetic), so
//! each lane of the packed permutation is canonically equal to the scalar
//! permutation of that lane's inputs.

use crate::field::packed::PackedField;
use crate::field::types::Field;
use crate::hash::poseidon2::config::*;

#[inline]
pub(crate) fn add_rc_packed<P: PackedField>(state: &mut [P; WIDTH], external_round: usize) {
    for i in 0..WIDTH {
        state[i] += P::Scalar::from_canonical_u64(EXTERNAL_CONSTANTS[external_round][i]);
    }
}

#[inline]
pub(crate) fn sbox_p_packed<P: PackedField>(x: P) -> P {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x3 = x * x2;
    x3 * x4
}

#[inline]
pub(crate) fn mat4_packed<P: PackedField>(x: &mut [P]) {
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
pub(crate) fn external_linear_layer_packed<P: PackedField>(state: &mut [P; WIDTH]) {
    for chunk in state.chunks_mut(4) {
        mat4_packed(chunk);
    }
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
pub(crate) fn internal_linear_layer_packed<P: PackedField>(state: &mut [P; WIDTH]) {
    let sum = state.iter().fold(P::ZEROS, |acc, &x| acc + x);
    for i in 0..WIDTH {
        state[i] =
            state[i] * P::Scalar::from_canonical_u64(MATRIX_DIAG_12_U64[i]) + sum;
    }
}

/// Full Poseidon2 permutation over `P::WIDTH` independent lanes. Mirrors the
/// scalar `Poseidon2::poseidon2` round structure and constants exactly, so
/// lane `l` of the result equals `poseidon2` of lane `l` of the input
/// (canonically; raw representations may differ between reduction paths).
#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
#[inline]
pub(crate) fn permute_packed<P: PackedField>(mut state: [P; WIDTH]) -> [P; WIDTH] {
    external_linear_layer_packed(&mut state);

    for r in 0..ROUNDS_F_HALF {
        add_rc_packed(&mut state, r);
        for value in state.iter_mut() {
            *value = sbox_p_packed(*value);
        }
        external_linear_layer_packed(&mut state);
    }

    for r in 0..ROUNDS_P {
        state[0] += P::Scalar::from_canonical_u64(INTERNAL_CONSTANTS[r]);
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

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
mod merkle {
    //! Packed CPU Merkle tree builder for `Poseidon2Hash`. Handles the flat
    //! row-major leaf layout of `fill_digests_buf_flat`, hashing
    //! `P::WIDTH = 4` independent nodes per packed permutation. Used for trees
    //! the Metal GPU path declines (below `MIN_GPU_PERMUTATIONS`).

    use core::mem::{size_of, MaybeUninit};

    use plonky2_maybe_rayon::*;

    use super::permute_packed;
    use crate::field::packable::Packable;
    use crate::field::packed::PackedField;
    use crate::hash::hash_types::{HashOut, RichField, NUM_HASH_OUT_ELTS};
    use crate::hash::merkle_tree::capacity_up_to_mut;
    use crate::hash::poseidon2::config::{RATE, WIDTH};
    use crate::hash::poseidon2::hash::Poseidon2Hash;
    use crate::plonk::config::Hasher;

    /// Packed CPU equivalent of `metal::build_merkle_tree`: same inputs, same
    /// recursive digest layout, same `(digests, cap)` result. Returns `None`
    /// when the eligibility checks fail (non-Goldilocks field or non-8-byte
    /// representation), letting the caller fall back to the scalar CPU path.
    pub(crate) fn build_merkle_tree<F: RichField>(
        leaves: &[F],
        leaf_width: usize,
        num_leaves: usize,
        cap_height: usize,
    ) -> Option<(Vec<HashOut<F>>, Vec<HashOut<F>>)> {
        if F::ORDER != 0xffff_ffff_0000_0001
            || size_of::<F>() != size_of::<u64>()
            || num_leaves == 0
            || !num_leaves.is_power_of_two()
            || leaves.len() != num_leaves * leaf_width
            || cap_height > num_leaves.trailing_zeros() as usize
        {
            return None;
        }

        let len_cap = 1 << cap_height;
        let num_digests = 2 * (num_leaves - len_cap);
        let mut digests = Vec::with_capacity(num_digests);
        let mut cap = Vec::with_capacity(len_cap);
        {
            let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
            let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
            fill_digests_buf_flat_packed::<F>(
                digests_buf,
                cap_buf,
                leaves,
                leaf_width,
                num_leaves,
                cap_height,
            );
        }
        unsafe {
            // SAFETY: `fill_digests_buf_flat_packed` initialized the spare
            // capacity up to `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }
        Some((digests, cap))
    }

    /// Column-major (natural-order poly-major) variant: materializes the
    /// bit-reversed row-major matrix (exactly what the scalar CPU fallback in
    /// `MerkleTree::new_columns` would do) and reuses the packed flat builder.
    pub(crate) fn build_merkle_tree_columns<F: RichField>(
        columns: &[Vec<F>],
        cap_height: usize,
    ) -> Option<(Vec<HashOut<F>>, Vec<HashOut<F>>)> {
        let leaf_width = columns.len();
        let num_leaves = columns.first().map_or(0, Vec::len);
        if F::ORDER != 0xffff_ffff_0000_0001
            || size_of::<F>() != size_of::<u64>()
            || num_leaves == 0
            || !num_leaves.is_power_of_two()
            || columns.iter().any(|column| column.len() != num_leaves)
        {
            return None;
        }
        let flat = crate::util::transpose_to_bitrev_flat(columns);
        build_merkle_tree(&flat, leaf_width, num_leaves, cap_height)
    }

    /// Packed equivalent of `fill_digests_buf_flat`: identical digest layout
    /// and identical hash values (each lane runs the same sponge as the
    /// scalar path).
    fn fill_digests_buf_flat_packed<F: RichField>(
        digests_buf: &mut [MaybeUninit<HashOut<F>>],
        cap_buf: &mut [MaybeUninit<HashOut<F>>],
        leaves: &[F],
        leaf_width: usize,
        num_leaves: usize,
        cap_height: usize,
    ) {
        // Special case of a tree that's all cap.
        if digests_buf.is_empty() {
            debug_assert_eq!(cap_buf.len(), num_leaves);
            let digests = leaf_digests_packed::<F>(leaves, leaf_width, num_leaves);
            for (slot, digest) in cap_buf.iter_mut().zip(digests) {
                slot.write(digest);
            }
            return;
        }

        let subtree_digests_len = digests_buf.len() >> cap_height;
        let subtree_leaves_len = num_leaves >> cap_height;
        let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
        assert_eq!(digests_chunks.len(), cap_buf.len());
        digests_chunks.zip(cap_buf).enumerate().for_each(
            |(subtree_index, (subtree_digests, subtree_cap))| {
                let leaf_start = subtree_index * subtree_leaves_len * leaf_width;
                let leaf_end = (subtree_index + 1) * subtree_leaves_len * leaf_width;
                subtree_cap.write(fill_subtree_flat_packed::<F>(
                    subtree_digests,
                    &leaves[leaf_start..leaf_end],
                    leaf_width,
                    subtree_leaves_len,
                ));
            },
        );
    }

    /// Builds one cap subtree iteratively, layer by layer, writing each stored
    /// layer into the recursive digest layout via the closed-form index (the
    /// sibling pair `p` of layer `i` starts at digest index
    /// `2 * ((p << (i + 1)) + (1 << i) - 1)`; cf. `merkle_tree_prove`) and
    /// returning the subtree root for the cap.
    fn fill_subtree_flat_packed<F: RichField>(
        digests_buf: &mut [MaybeUninit<HashOut<F>>],
        leaves: &[F],
        leaf_width: usize,
        num_leaves: usize,
    ) -> HashOut<F> {
        debug_assert_eq!(num_leaves, digests_buf.len() / 2 + 1);
        if digests_buf.is_empty() {
            return <Poseidon2Hash as Hasher<F>>::hash_or_noop(leaves);
        }

        // Layer 0: the leaf digests.
        let mut layer = leaf_digests_packed::<F>(leaves, leaf_width, num_leaves);
        write_layer(digests_buf, &layer, 0);

        // Layers 1..log2(num_leaves) - 1 are stored; the final compression of
        // two nodes yields the subtree root, which lives in the cap.
        let mut height = 1;
        while layer.len() > 2 {
            layer = compress_layer_packed::<<F as Packable>::Packing>(&layer);
            write_layer(digests_buf, &layer, height);
            height += 1;
        }
        <Poseidon2Hash as Hasher<F>>::two_to_one(layer[0], layer[1])
    }

    /// Scatters one tree layer (`height` levels above the leaves, ordered by
    /// node index) into the recursive digest layout.
    fn write_layer<F: RichField>(
        digests_buf: &mut [MaybeUninit<HashOut<F>>],
        layer: &[HashOut<F>],
        height: usize,
    ) {
        for (j, digest) in layer.iter().enumerate() {
            let pair = j >> 1;
            let index = 2 * ((pair << (height + 1)) + (1 << height) - 1) + (j & 1);
            digests_buf[index].write(*digest);
        }
    }

    /// Hashes every leaf (`hash_or_noop` semantics), `P::WIDTH` leaves per
    /// packed permutation. Leaves of width <= 4 are copied and zero-padded,
    /// exactly like the scalar `hash_or_noop`; no permutation is involved.
    fn leaf_digests_packed<F: RichField>(
        leaves: &[F],
        leaf_width: usize,
        num_leaves: usize,
    ) -> Vec<HashOut<F>> {
        debug_assert_eq!(leaves.len(), num_leaves * leaf_width);
        let mut out = vec![HashOut::<F>::ZERO; num_leaves];

        if leaf_width <= NUM_HASH_OUT_ELTS {
            // The copy-and-pad case canonicalizes through byte round-trips in
            // the scalar path; reuse it verbatim (it performs no hashing).
            out.par_iter_mut().enumerate().for_each(|(i, slot)| {
                *slot = <Poseidon2Hash as Hasher<F>>::hash_or_noop(
                    &leaves[i * leaf_width..(i + 1) * leaf_width],
                );
            });
            return out;
        }

        let lanes = <<F as Packable>::Packing as PackedField>::WIDTH;
        out.par_chunks_mut(lanes)
            .zip(leaves.par_chunks(lanes * leaf_width))
            .for_each(|(digests, rows)| {
                if digests.len() == lanes {
                    hash_leaf_group::<<F as Packable>::Packing>(digests, rows, leaf_width);
                } else {
                    // Group remainder (< lanes leaves): scalar sponge.
                    for (digest, row) in digests.iter_mut().zip(rows.chunks_exact(leaf_width)) {
                        *digest = <Poseidon2Hash as Hasher<F>>::hash_no_pad(row);
                    }
                }
            });
        out
    }

    /// Sponge-hashes `P::WIDTH` equal-width leaves, one per lane, mirroring
    /// `hash_n_to_hash_no_pad`: absorb `RATE`-element chunks in overwrite
    /// mode (a final partial chunk overwrites only its `len` leading state
    /// elements), then squeeze `state[0..4]`.
    fn hash_leaf_group<P: PackedField>(
        digests: &mut [HashOut<P::Scalar>],
        rows: &[P::Scalar],
        leaf_width: usize,
    ) where
        P::Scalar: RichField,
    {
        debug_assert_eq!(digests.len(), P::WIDTH);
        debug_assert_eq!(rows.len(), P::WIDTH * leaf_width);
        let mut state = [P::ZEROS; WIDTH];
        let mut offset = 0;
        while offset < leaf_width {
            let chunk_len = RATE.min(leaf_width - offset);
            for k in 0..chunk_len {
                let slot = state[k].as_slice_mut();
                for (l, value) in slot.iter_mut().enumerate() {
                    *value = rows[l * leaf_width + offset + k];
                }
            }
            state = permute_packed(state);
            offset += chunk_len;
        }
        for (l, digest) in digests.iter_mut().enumerate() {
            digest.elements = core::array::from_fn(|d| state[d].as_slice()[l]);
        }
    }

    /// Compresses one layer into its parents, `P::WIDTH` parents per packed
    /// permutation (mirroring `compress`: state = left || right || 0, one
    /// permutation, squeeze `state[0..4]`). Group remainders use the scalar
    /// `two_to_one`.
    fn compress_layer_packed<P: PackedField>(
        prev: &[HashOut<P::Scalar>],
    ) -> Vec<HashOut<P::Scalar>>
    where
        P::Scalar: RichField,
    {
        debug_assert!(prev.len() >= 2 && prev.len() % 2 == 0);
        let lanes = P::WIDTH;
        let mut next = vec![HashOut::<P::Scalar>::ZERO; prev.len() / 2];
        next.par_chunks_mut(lanes)
            .zip(prev.par_chunks(2 * lanes))
            .for_each(|(parents, children)| {
                if parents.len() == lanes {
                    let mut state = [P::ZEROS; WIDTH];
                    for l in 0..lanes {
                        for d in 0..NUM_HASH_OUT_ELTS {
                            state[d].as_slice_mut()[l] = children[2 * l].elements[d];
                            state[NUM_HASH_OUT_ELTS + d].as_slice_mut()[l] =
                                children[2 * l + 1].elements[d];
                        }
                    }
                    let state = permute_packed(state);
                    for (l, parent) in parents.iter_mut().enumerate() {
                        parent.elements = core::array::from_fn(|d| state[d].as_slice()[l]);
                    }
                } else {
                    for (l, parent) in parents.iter_mut().enumerate() {
                        *parent = <Poseidon2Hash as Hasher<P::Scalar>>::two_to_one(
                            children[2 * l],
                            children[2 * l + 1],
                        );
                    }
                }
            });
        next
    }
}

#[cfg(all(feature = "std", target_arch = "aarch64", target_os = "macos"))]
pub(crate) use merkle::{build_merkle_tree, build_merkle_tree_columns};

#[cfg(all(test, feature = "std", target_arch = "aarch64", target_os = "macos"))]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::packable::Packable;
    use crate::field::types::{Field64, PrimeField64};
    use crate::hash::hash_types::HashOut;
    use crate::hash::merkle_tree::{capacity_up_to_mut, fill_digests_buf_flat};
    use crate::hash::poseidon2::hash::{Poseidon2, Poseidon2Hash};

    type F = GoldilocksField;
    type P = <F as Packable>::Packing;

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

    fn scalar_tree(
        leaves: &[F],
        leaf_width: usize,
        num_leaves: usize,
        cap_height: usize,
    ) -> (Vec<HashOut<F>>, Vec<HashOut<F>>) {
        let cap_len = 1 << cap_height;
        let digest_len = 2 * (num_leaves - cap_len);
        let mut digests = Vec::with_capacity(digest_len);
        let mut cap = Vec::with_capacity(cap_len);
        let digests_buf = capacity_up_to_mut(&mut digests, digest_len);
        let cap_buf = capacity_up_to_mut(&mut cap, cap_len);
        fill_digests_buf_flat::<F, Poseidon2Hash>(
            digests_buf,
            cap_buf,
            leaves,
            leaf_width,
            num_leaves,
            cap_height,
        );
        unsafe {
            digests.set_len(digest_len);
            cap.set_len(cap_len);
        }
        (digests, cap)
    }

    fn assert_tree_eq(
        actual: &(Vec<HashOut<F>>, Vec<HashOut<F>>),
        expected: &(Vec<HashOut<F>>, Vec<HashOut<F>>),
        context: &str,
    ) {
        assert_eq!(actual.0.len(), expected.0.len(), "{context}");
        assert_eq!(actual.1.len(), expected.1.len(), "{context}");
        for (index, (actual, expected)) in actual
            .0
            .iter()
            .chain(&actual.1)
            .zip(expected.0.iter().chain(&expected.1))
            .enumerate()
        {
            let actual = actual.elements.map(|value| value.to_canonical_u64());
            let expected = expected.elements.map(|value| value.to_canonical_u64());
            assert_eq!(actual, expected, "{context}, node {index}");
        }
    }

    #[test]
    fn packed_permutation_matches_scalar_lanes() {
        let mut rng = StdRng::seed_from_u64(0x5041_434b_4544);
        for trial in 0..16u64 {
            let lanes: Vec<[F; WIDTH]> = (0..P::WIDTH)
                .map(|lane| {
                    core::array::from_fn(|i| {
                        GoldilocksField(raw_value(
                            trial + lane as u64 + i as u64,
                            &mut rng,
                        ))
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
    fn packed_merkle_matches_scalar_across_sponge_boundaries() {
        let mut rng = StdRng::seed_from_u64(0x5041_434b_3252);
        for num_leaves in [64usize, 256] {
            for width in [0usize, 1, 4, 5, 8, 9, 16, 17, 31, 64, 137] {
                let flat: Vec<F> = (0..num_leaves * width)
                    .map(|i| GoldilocksField(raw_value(i as u64, &mut rng)))
                    .collect();
                for cap_height in [0usize, 3, 6] {
                    let context = format!(
                        "num_leaves {num_leaves}, width {width}, cap height {cap_height}"
                    );
                    let packed = build_merkle_tree::<F>(&flat, width, num_leaves, cap_height)
                        .expect("packed builder must accept Goldilocks flat trees");
                    let scalar = scalar_tree(&flat, width, num_leaves, cap_height);
                    assert_tree_eq(&packed, &scalar, &context);
                }
            }
        }
    }

    #[test]
    fn packed_merkle_columns_matches_scalar() {
        let mut rng = StdRng::seed_from_u64(0x5041_434b_434f);
        for (num_leaves, width) in [(64usize, 7usize), (256, 33)] {
            let log_rows = num_leaves.trailing_zeros() as usize;
            // Natural-order poly-major columns; tree leaf `i` is
            // `columns[j][reverse_bits(i, log_rows)]`.
            let columns: Vec<Vec<F>> = (0..width)
                .map(|j| {
                    (0..num_leaves)
                        .map(|natural| {
                            GoldilocksField(raw_value((j * num_leaves + natural) as u64, &mut rng))
                        })
                        .collect()
                })
                .collect();
            let flat: Vec<F> = (0..num_leaves)
                .flat_map(|leaf| {
                    let natural = crate::util::reverse_bits(leaf, log_rows);
                    columns.iter().map(move |column| column[natural])
                })
                .collect();
            for cap_height in [0usize, 3] {
                let context = format!(
                    "num_leaves {num_leaves}, width {width}, cap height {cap_height}"
                );
                let packed = build_merkle_tree_columns::<F>(&columns, cap_height)
                    .expect("packed column builder must accept Goldilocks columns");
                let scalar = scalar_tree(&flat, width, num_leaves, cap_height);
                assert_tree_eq(&packed, &scalar, &context);
            }
        }
    }
}
