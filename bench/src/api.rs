// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block_constraints::{BlockCircuit, BlockTarget, Circuit as _};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;

pub type Proof = ProofWithPublicInputs<F, C, D>;

pub const CHAIN_ID: u32 = 304;
pub const HEAVY_TX_PER_PROOF: usize = 4;
pub const HEAVY_TX_MODE: u8 = TX_HEAVY;
pub const LIGHT_TX_PER_PROOF: usize = 10;
pub const LIGHT_TX_MODE: u8 = TX_LIGHT;
pub const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
pub const PUBLIC_HEAVY_TX_COUNT: usize = 10;
pub const PUBLIC_LIGHT_TX_COUNT: usize = 490;
pub const PROVER_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

pub struct Circuits {
    pub heavy_tx_target: BlockTxTarget,
    pub heavy_tx_data: CircuitData<F, C, D>,
    pub light_tx_target: BlockTxTarget,
    pub light_tx_data: CircuitData<F, C, D>,
    pub pre_target: BlockPreExecutionTarget,
    pub pre_data: CircuitData<F, C, D>,
    pub heavy_chain_target: BlockTxChainTarget,
    pub heavy_chain_data: CircuitData<F, C, D>,
    pub light_chain_target: BlockTxChainTarget,
    pub light_chain_data: CircuitData<F, C, D>,
    pub block_target: BlockTarget,
    pub block_data: CircuitData<F, C, D>,
    pub dummy_heavy_proof: Proof,
    pub dummy_light_proof: Proof,
}
struct PathCircuits {
    tx_target: BlockTxTarget,
    tx_data: CircuitData<F, C, D>,
    chain_target: BlockTxChainTarget,
    chain_data: CircuitData<F, C, D>,
    dummy_proof: Proof,
}

impl PathCircuits {
    fn new(tx_per_proof: usize, tx_mode: u8) -> Self {
        let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, tx_mode);
        let tx_target = tx.target;
        let tx_data = tx.builder.build::<C>();

        let chain =
            BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
        let chain_target = chain.target;
        let chain_data = chain.builder.build::<C>();

        let proof_bytes: &[u8] = match tx_mode {
            TX_HEAVY => include_bytes!("../dummy-heavy-chain-proof.bin"),
            TX_LIGHT => include_bytes!("../dummy-light-chain-proof.bin"),
            _ => panic!("unsupported block transaction mode {tx_mode}"),
        };
        let dummy_proof =
            bincode::deserialize(proof_bytes).expect("embedded chain dummy proof is invalid");

        Self {
            tx_target,
            tx_data,
            chain_target,
            chain_data,
            dummy_proof,
        }
    }
}

impl Circuits {
    pub fn new() -> Self {
        let ((pre_target, pre_data), (heavy, light)) = rayon::join(
            || {
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                (pre.target, pre.builder.build::<C>())
            },
            || {
                rayon::join(
                    || PathCircuits::new(HEAVY_TX_PER_PROOF, HEAVY_TX_MODE),
                    || PathCircuits::new(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE),
                )
            },
        );

        let block = BlockCircuit::define(
            CIRCUIT_CONFIG,
            &pre_data,
            &light.chain_data,
            &heavy.chain_data,
            ON_CHAIN_OPERATIONS_LIMIT,
        );
        let block_target = block.target;
        let block_data = block.builder.build::<C>();

        Self {
            heavy_tx_target: heavy.tx_target,
            heavy_tx_data: heavy.tx_data,
            light_tx_target: light.tx_target,
            light_tx_data: light.tx_data,
            pre_target,
            pre_data,
            heavy_chain_target: heavy.chain_target,
            heavy_chain_data: heavy.chain_data,
            light_chain_target: light.chain_target,
            light_chain_data: light.chain_data,
            block_target,
            block_data,
            dummy_heavy_proof: heavy.dummy_proof,
            dummy_light_proof: light.dummy_proof,
        }
    }
}

#[cfg(test)]
mod tests {
    use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
    use plonky2::field::types::Field;
    use plonky2::plonk::config::{GenericConfig, Hasher as _};

    use super::*;

    type BlockHasher = <C as GenericConfig<D>>::Hasher;

    /// Guards the vendored hashing change: Merkle leaf hashing now goes through
    /// `Hasher::hash_or_noop_x4`, which advances four Poseidon2 sponges in
    /// lockstep, and the Metal backend uses it for the CPU share of a split leaf
    /// level. Every output must equal the one-at-a-time result, otherwise every
    /// Merkle cap changes, the Fiat-Shamir transcript changes, and the pinned
    /// verifier rejects the proof.
    #[test]
    fn batched_leaf_hashing_matches_scalar() {
        let value = |lane: u64, i: usize| F::from_canonical_u64(lane * 1_000_003 + i as u64 + 1);
        // 136 is `num_wires`, the wires-oracle leaf width; 32 is the FRI
        // commit-phase leaf width (arity 16 times extension degree 2); 20 and 16
        // are the Zs/partial-products and quotient widths.
        for len in [0usize, 1, 4, 5, 8, 16, 20, 32, 135, 136, 137] {
            let lanes: Vec<Vec<F>> = (0..4)
                .map(|lane| (0..len).map(|i| value(lane, i)).collect())
                .collect();
            let inputs: [&[F]; 4] = [&lanes[0], &lanes[1], &lanes[2], &lanes[3]];
            let batched = BlockHasher::hash_or_noop_x4(inputs);
            for lane in 0..4 {
                assert_eq!(
                    batched[lane],
                    BlockHasher::hash_or_noop(inputs[lane]),
                    "lane {lane} mismatch at len {len}"
                );
            }
        }

        // Unequal lengths must fall back rather than absorb the wrong number of
        // chunks in some lane.
        let wide: Vec<F> = (0..136).map(|i| value(9, i)).collect();
        let narrow: Vec<F> = (0..40).map(|i| value(8, i)).collect();
        let inputs: [&[F]; 4] = [&wide, &narrow, &wide, &narrow];
        let batched = BlockHasher::hash_or_noop_x4(inputs);
        for lane in 0..4 {
            assert_eq!(batched[lane], BlockHasher::hash_or_noop(inputs[lane]));
        }
    }

    /// A whole Merkle tree must have the same cap however its leaf level was
    /// computed. This exercises the real `MerkleTree::new` entry point, so on a
    /// Metal-capable host it also covers the GPU path and, for a large enough
    /// tree, the CPU/GPU split.
    #[test]
    fn merkle_tree_cap_matches_scalar_leaf_hashing() {
        use plonky2::hash::merkle_tree::MerkleTree;

        let leaves: Vec<Vec<F>> = (0..64)
            .map(|row: u64| {
                (0..136)
                    .map(|col: u64| F::from_canonical_u64(row * 977 + col * 31 + 1))
                    .collect()
            })
            .collect();
        let tree = MerkleTree::<F, BlockHasher>::new(leaves.clone(), 2);

        let mut expected: Vec<_> = leaves
            .iter()
            .map(|leaf| BlockHasher::hash_or_noop(leaf))
            .collect();
        while expected.len() > 4 {
            expected = expected
                .chunks(2)
                .map(|pair| BlockHasher::two_to_one(pair[0], pair[1]))
                .collect();
        }
        assert_eq!(tree.cap.0, expected);
    }

    #[test]
    fn production_mixed_circuit_parameters_are_fixed() {
        assert_eq!(CHAIN_ID, 304);
        assert_eq!(HEAVY_TX_PER_PROOF, 4);
        assert_eq!(HEAVY_TX_MODE, TX_HEAVY);
        assert_eq!(LIGHT_TX_PER_PROOF, 10);
        assert_eq!(LIGHT_TX_MODE, TX_LIGHT);
        assert_eq!(ON_CHAIN_OPERATIONS_LIMIT, 1);
    }

    #[test]
    fn embedded_chain_dummy_proofs_deserialize() {
        let _: Proof = bincode::deserialize(include_bytes!("../dummy-heavy-chain-proof.bin"))
            .expect("embedded heavy dummy proof is invalid");
        let _: Proof = bincode::deserialize(include_bytes!("../dummy-light-chain-proof.bin"))
            .expect("embedded light dummy proof is invalid");
    }
}
