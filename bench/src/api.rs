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

    use super::*;

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
