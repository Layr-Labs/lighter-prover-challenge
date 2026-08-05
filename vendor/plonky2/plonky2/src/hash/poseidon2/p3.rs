use std::sync::OnceLock;

use p3_field::{AbstractField, PrimeField64 as _};
use p3_goldilocks::{DiffusionMatrixGoldilocks, Goldilocks};
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::Permutation;

use super::config::*;
use crate::field::goldilocks_field::GoldilocksField;
use crate::field::types::PrimeField64;

type GoldilocksPoseidon2 =
    Poseidon2<Goldilocks, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, WIDTH, D>;

fn permutation() -> &'static GoldilocksPoseidon2 {
    static PERMUTATION: OnceLock<GoldilocksPoseidon2> = OnceLock::new();

    PERMUTATION.get_or_init(|| {
        let external_constants = EXTERNAL_CONSTANTS
            .map(|row| row.map(Goldilocks::from_canonical_u64))
            .to_vec();
        let internal_constants = INTERNAL_CONSTANTS
            .map(Goldilocks::from_canonical_u64)
            .to_vec();

        GoldilocksPoseidon2::new(
            ROUNDS_F,
            external_constants,
            Poseidon2ExternalMatrixGeneral,
            ROUNDS_P,
            internal_constants,
            DiffusionMatrixGoldilocks,
        )
    })
}

#[inline]
pub(crate) fn p3_poseidon2_permute(input: [GoldilocksField; WIDTH]) -> [GoldilocksField; WIDTH] {
    let mut state = input.map(|value| Goldilocks::from_wrapped_u64(value.to_noncanonical_u64()));
    permutation().permute_mut(&mut state);
    state.map(|value| GoldilocksField(value.as_canonical_u64()))
}

// Poseidon2 from plonky3
#[cfg(test)]
pub fn p3_poseidon2_hash_n_to_m_no_pad(
    inputs: &[Goldilocks],
    num_outputs: usize,
) -> Vec<Goldilocks> {
    let external_linear_layer = Poseidon2ExternalMatrixGeneral;
    let internal_linear_layer = DiffusionMatrixGoldilocks;

    let external_constants = EXTERNAL_CONSTANTS
        .iter()
        .map(|v| {
            v.iter()
                .map(|&x| Goldilocks::from_canonical_u64(x))
                .collect::<Vec<Goldilocks>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<[Goldilocks; WIDTH]>>();

    let internal_constants = INTERNAL_CONSTANTS
        .iter()
        .map(|&x| Goldilocks::from_canonical_u64(x))
        .collect::<Vec<Goldilocks>>();

    let poseidon = Poseidon2::<
        Goldilocks,
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixGoldilocks,
        WIDTH,
        D,
    >::new(
        ROUNDS_F,
        external_constants,
        external_linear_layer,
        ROUNDS_P,
        internal_constants,
        internal_linear_layer,
    );

    let mut perm = [Goldilocks::zero(); WIDTH];

    #[allow(clippy::manual_memcpy)]
    for input_chunk in inputs.chunks(RATE) {
        for i in 0..RATE.min(input_chunk.len()) {
            perm[i] = input_chunk[i];
        }
        poseidon.permute_mut(&mut perm);
    }

    let mut outputs: Vec<Goldilocks> = Vec::new();
    loop {
        for &item in perm[0..RATE].iter() {
            outputs.push(item);
            if outputs.len() == num_outputs {
                return outputs;
            }
        }
        poseidon.permute_mut(&mut perm);
    }
}
