// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use circuit::block_constraints::{BlockCircuit, BlockTarget, Circuit as _};
use circuit::block_pre_execution_constraints::{
    BlockPreExecutionCircuit, BlockPreExecutionTarget, Circuit as _,
};
use circuit::block_tx_chain_constraints::{BlockTxChainCircuit, BlockTxChainTarget, Circuit as _};
use circuit::block_tx_constraints::{BlockTxCircuit, BlockTxTarget, Circuit as _};
use circuit::builder::custom::cyclic_base_proof;
use circuit::types::config::{C, CIRCUIT_CONFIG, D, F};
use circuit::types::constants::{TX_HEAVY, TX_LIGHT};
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::plonk::proof::ProofWithPublicInputs;
use plonky2::recursion::dummy_circuit::dummy_circuit;

pub type Proof = ProofWithPublicInputs<F, C, D>;

pub const CHAIN_ID: u32 = 304;
pub const HEAVY_TX_PER_PROOF: usize = 4;
pub const HEAVY_TX_MODE: u8 = TX_HEAVY;
pub const LIGHT_TX_PER_PROOF: usize = 10;
pub const LIGHT_TX_MODE: u8 = TX_LIGHT;
pub const ON_CHAIN_OPERATIONS_LIMIT: usize = 1;
pub const PUBLIC_HEAVY_TX_COUNT: usize = 10;
pub const PUBLIC_LIGHT_TX_COUNT: usize = 490;

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
    pub dummy_heavy_chain_circuit: CircuitData<F, C, D>,
    pub dummy_light_chain_circuit: CircuitData<F, C, D>,
    pub dummy_heavy_proof: Proof,
    pub dummy_light_proof: Proof,
}

/// The transaction circuit, its chain-recursion circuit, and the chain's dummy artifacts for one
/// of the two transaction widths. Nothing here depends on the other width.
struct PathCircuits {
    tx_target: BlockTxTarget,
    tx_data: CircuitData<F, C, D>,
    chain_target: BlockTxChainTarget,
    chain_data: CircuitData<F, C, D>,
    dummy_chain_circuit: CircuitData<F, C, D>,
    dummy_proof: Proof,
}

impl PathCircuits {
    fn build(tx_per_proof: usize, mode: u8, label: &str) -> Self {
        let tx = BlockTxCircuit::define(CIRCUIT_CONFIG, tx_per_proof, CHAIN_ID, mode);
        let tx_target = tx.target;
        let tx_data = tx.builder.build::<C>();

        let chain =
            BlockTxChainCircuit::define(CIRCUIT_CONFIG, &tx_data, ON_CHAIN_OPERATIONS_LIMIT);
        let chain_target = chain.target;
        let chain_data = chain.builder.build::<C>();

        let dummy_chain_circuit = dummy_circuit(&chain_data.common);
        let dummy_proof = cyclic_base_proof(
            &chain_data.common,
            &chain_data.verifier_only,
            &dummy_chain_circuit,
            [].into_iter().collect(),
        )
        .unwrap_or_else(|error| panic!("cannot construct {label} chain dummy proof: {error:?}"));

        Self {
            tx_target,
            tx_data,
            chain_target,
            chain_data,
            dummy_chain_circuit,
            dummy_proof,
        }
    }
}

impl Circuits {
    pub fn new() -> Self {
        // The heavy path, the light path, and the pre-execution circuit are mutually independent,
        // so build all three concurrently. Only the final block circuit needs the others and
        // therefore stays sequential. The circuit definitions themselves are unchanged, so the
        // resulting verifier data is identical to the sequential build.
        let ((heavy, light), (pre_target, pre_data)) = rayon::join(
            || {
                rayon::join(
                    || PathCircuits::build(HEAVY_TX_PER_PROOF, HEAVY_TX_MODE, "heavy"),
                    || PathCircuits::build(LIGHT_TX_PER_PROOF, LIGHT_TX_MODE, "light"),
                )
            },
            || {
                let pre = BlockPreExecutionCircuit::define(CIRCUIT_CONFIG);
                (pre.target, pre.builder.build::<C>())
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
            dummy_heavy_chain_circuit: heavy.dummy_chain_circuit,
            dummy_light_chain_circuit: light.dummy_chain_circuit,
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
}
